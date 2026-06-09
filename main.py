import os
import csv
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import Config
from data.dataset import SatelliteSegmentationDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.lightweight_unet import LightweightUNet
from utils.engine import train_one_epoch, evaluate
from utils.plots import plot_loss_curves, save_predictions


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(mode):
    """Costruisce il modello in base alla modalita' di pretraining."""
    if mode == "satmaepp":
        from models.satmaepp_segmenter import SatMAEppSegmenter
        model = SatMAEppSegmenter(num_classes=Config.NUM_CLASSES,
                                  ckpt_path=Config.SATMAEPP_CKPT_PATH,
                                  img_size=Config.IMAGE_SIZE, freeze_encoder=True)
        encoder_params = None  # encoder gia' congelato by design
    else:
        model = LightweightUNet(num_classes=Config.NUM_CLASSES, encoder_name=Config.ENCODER_NAME,
                                pretraining_mode=mode, rsp_weights_path=Config.RSP_WEIGHTS_PATH)
        encoder_params = model.model.encoder.parameters
    return model, encoder_params


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice, entrambe con ignore_index per scartare il 'no-data'."""
    def __init__(self, ignore_index=-100):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        ign = ignore_index if ignore_index >= 0 else None
        self.dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=ign)

    def forward(self, preds, targets):
        return self.ce(preds, targets) + self.dice(preds, targets)


def make_loader(split, transform, subset_size):
    ds = SatelliteSegmentationDataset(data_dir=Config.DATA_DIR, transform=transform, split=split)
    if subset_size is not None and subset_size < len(ds):
        g = torch.Generator().manual_seed(Config.SEED)
        idx = torch.randperm(len(ds), generator=g)[:subset_size].tolist()
        ds = Subset(ds, idx)
    return ds


def main(args):
    set_seed(Config.SEED)
    mode = args.mode or Config.PRETRAINING_MODE
    epochs = args.epochs or Config.NUM_EPOCHS
    train_subset = Config.TRAIN_SUBSET_SIZE if args.train_subset is None else args.train_subset
    val_subset = Config.VAL_SUBSET_SIZE if args.val_subset is None else args.val_subset
    tag = args.tag or f"{mode}_n{train_subset or 'full'}"

    print("=" * 70)
    print(f"RUN: {tag} | pretraining={mode} | train_subset={train_subset} | epochs={epochs}")
    print(f"Device: {Config.DEVICE} | Wavelet: {Config.USE_WAVELET_AUGMENTATION} | "
          f"ignore_index={Config.IGNORE_INDEX} | ImageNetNorm={Config.USE_IMAGENET_NORM}")
    print("=" * 70)

    # ---------- Data ----------
    train_tf = get_train_transforms(Config.IMAGE_SIZE, use_wavelet=Config.USE_WAVELET_AUGMENTATION)
    val_tf = get_val_transforms(Config.IMAGE_SIZE, use_wavelet=Config.USE_WAVELET_AUGMENTATION)
    train_ds = make_loader("train", train_tf, train_subset)
    val_ds = make_loader("val", val_tf, val_subset)

    nw = Config.NUM_WORKERS
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=nw,
                              pin_memory=True, persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=nw,
                            pin_memory=True, persistent_workers=(nw > 0))
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ---------- Model / loss / optim ----------
    model, encoder_params = build_model(mode)
    model = model.to(Config.DEVICE)
    criterion = CombinedLoss(ignore_index=Config.IGNORE_INDEX)

    # Warmup con encoder congelato ha senso solo se l'encoder e' PRETRAINED ed esiste
    do_warmup = (mode in ("imagenet", "rsp")) and Config.FREEZE_EPOCHS > 0 and encoder_params is not None
    if do_warmup:
        print(f"Backbone congelato per i primi {Config.FREEZE_EPOCHS} epoch (warmup)...")
        for p in encoder_params():
            p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(Config.DEVICE == 'cuda'))

    if args.dry_run:
        print("[Dry Run] Setup ok, esco senza allenare.")
        return

    # ---------- Training loop ----------
    train_losses, val_losses = [], []
    best_miou = -1.0
    hist_path = f"history_{tag}.csv"
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "pixel_acc", "mIoU", "Dice"])

    for epoch in range(epochs):
        if do_warmup and epoch == Config.FREEZE_EPOCHS:
            print("\n🔥 SCONGELAMENTO BACKBONE: fine-tuning profondo con LR ridotto.")
            for p in encoder_params():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE / 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch)

        print(f"\n--- Epoch {epoch + 1}/{epochs} [{tag}] ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE, scaler)
        scheduler.step()
        train_losses.append(train_loss)

        do_eval = ((epoch + 1) % Config.EVAL_EVERY == 0) or (epoch == epochs - 1)
        if not do_eval:
            print(f"Train Loss: {train_loss:.4f} (eval saltata)")
            continue

        val_loss, val_acc, val_miou, val_dice, iou_pc = evaluate(
            model, val_loader, criterion, Config.DEVICE,
            num_classes=Config.NUM_CLASSES, ignore_index=Config.IGNORE_INDEX)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Acc: {val_acc*100:.2f}% | mIoU: {val_miou:.4f} | Dice: {val_dice:.4f}")
        # IoU per classe (salta la classe ignorata)
        per_cls = "  ".join(
            f"{Config.CLASS_NAMES[c] if c < len(Config.CLASS_NAMES) else c}:{iou_pc[c]:.2f}"
            for c in range(Config.NUM_CLASSES) if c != Config.IGNORE_INDEX)
        print(f"   IoU/classe -> {per_cls}")

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.4f}", f"{val_loss:.4f}",
                                    f"{val_acc:.4f}", f"{val_miou:.4f}", f"{val_dice:.4f}"])

        plot_loss_curves(train_losses, val_losses, save_path=f"loss_{tag}.png")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), f"best_{tag}.pth")
            print(f"   ✅ Nuovo best mIoU={best_miou:.4f} -> salvato best_{tag}.pth")
            try:
                model.eval()
                with torch.no_grad():
                    vi, vm = next(iter(val_loader))
                    vi, vm = vi.to(Config.DEVICE), vm.to(Config.DEVICE)
                    save_predictions(vi, vm, model(vi), "output_samples", epoch, 0,
                                     mIoU=val_miou, mDice=val_dice)
            except Exception as e:
                print(f"   (viz saltata: {e})")

    # ---------- Riga di riepilogo (per la tabella di ablation) ----------
    summary = "results_summary.csv"
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "mode", "train_subset", "epochs", "best_mIoU"])
        w.writerow([tag, mode, train_subset or "full", epochs, f"{best_miou:.4f}"])
    print(f"\n✔ Fine '{tag}'. Best mIoU = {best_miou:.4f} (riga aggiunta a {summary}).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Satellite Segmentation - training + ablation")
    p.add_argument("--mode", type=str, default=None,
                   choices=["scratch", "imagenet", "rsp", "satmaepp"],
                   help="override di Config.PRETRAINING_MODE")
    p.add_argument("--train-subset", dest="train_subset", type=int, default=None,
                   help="numero immagini di train (data-scarce). Default: Config")
    p.add_argument("--val-subset", dest="val_subset", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", type=str, default=None, help="nome del run nei file di output")
    p.add_argument("--dry-run", action="store_true")
    main(p.parse_args())
