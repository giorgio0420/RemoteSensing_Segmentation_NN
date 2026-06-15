# -*- coding: utf-8 -*-
"""
TASK 2 (secondaria) - Multi-modal / multi-spectral inputs su DFC2020 (Sentinel-1 + Sentinel-2).

Risponde a DUE domande della consegna, in modo robusto e runnabile:
  (Q1) le bande extra aiutano?   -> RGB (3) vs multispettrale S2 (10) vs +radar SAR (12)
  (Q2) il pretraining aiuta?     -> encoder ImageNet vs scratch (stessa rete)

Dataset: GFM-Bench/DFC2020 (HuggingFace), patch 96x96, 8 classi land-cover.
Modello: U-Net (ResNet-34) via segmentation-models-pytorch, in_channels variabile.

NB: i pesi SatMAE++ fMoW-Sentinel (ViT group-channel) sono uno step separato e piu' fragile;
qui usiamo ImageNet come pretraining robusto per il confronto pretrained-vs-scratch.
"""
import argparse, csv, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---- Bande Sentinel-2 in DFC2020 (ordine): 0 B01,1 B02,2 B03,3 B04,4 B05,5 B06,
#      6 B07,7 B08,8 B8A,9 B09,10 B10,11 B11,12 B12
RGB_IDX = [3, 2, 1]                              # R,G,B = B04,B03,B02
MSI_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]       # 10 bande (scarta B01,B09,B10) = set SatMAE-Sentinel
N_CLASSES = 8
SEED = 42


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _to_chw(arr, nbands):
    """Porta un array immagine a [C,H,W] gestendo [C,H,W] o [H,W,C]."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, ...]
    elif a.ndim == 3 and a.shape[-1] == nbands and a.shape[0] != nbands:
        a = np.transpose(a, (2, 0, 1))
    return a


class DFC2020(Dataset):
    def __init__(self, hf_split, mode="msi", subset=None, seed=SEED):
        """mode: 'rgb' | 'msi' | 'msi_sar'."""
        self.ds = hf_split
        self.mode = mode
        self.bands = RGB_IDX if mode == "rgb" else MSI_IDX
        idx = list(range(len(hf_split)))
        if subset and subset < len(idx):
            random.Random(seed).shuffle(idx); idx = idx[:subset]
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        s = self.ds[self.idx[i]]
        opt = _to_chw(s["optical"], 13)                 # [13,H,W]
        x = opt[self.bands] / 10000.0                   # riflettanza S2 ~0..10000 -> ~0..1
        if self.mode == "msi_sar":
            sar = _to_chw(s["radar"], 2)                # [2,H,W] (VV,VH in dB, ~-25..0)
            sar = np.clip((sar + 25.0) / 25.0, 0, 2)    # normalizzazione grezza
            x = np.concatenate([x, sar], axis=0)
        m = np.asarray(s["label"]).squeeze().astype(np.int64)
        # alcune versioni indicizzano le classi da 1; portiamo a 0-based se necessario
        if m.min() >= 1 and m.max() <= N_CLASSES:
            m = m - 1
        return torch.from_numpy(x).float(), torch.from_numpy(m).long()


def in_channels_for(mode):
    return {"rgb": 3, "msi": 10, "msi_sar": 12}[mode]


def build_model(in_ch, pretrained):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34",
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=in_ch, classes=N_CLASSES)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = torch.zeros(N_CLASSES, device=device)
    union = torch.zeros(N_CLASSES, device=device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
        for c in range(N_CLASSES):
            pc, yc = pred == c, y == c
            inter[c] += (pc & yc).sum(); union[c] += (pc | yc).sum()
    iou = inter / (union + 1e-6)
    valid = union > 0
    miou = iou[valid].mean().item() if valid.any() else 0.0
    return correct / total, miou, iou.cpu().numpy()


def main(a):
    set_seed()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from datasets import load_dataset
    print(f"Carico GFM-Bench/DFC2020 ... (mode={a.bands}, pretrained={not a.scratch})")
    ds = load_dataset("GFM-Bench/DFC2020")
    tr = DFC2020(ds["train"], a.bands, subset=a.subset)
    va = DFC2020(ds["validation"], a.bands, subset=a.val_subset)
    print(f"Train {len(tr)} | Val {len(va)} | in_channels={in_channels_for(a.bands)}")
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(va, batch_size=a.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(in_channels_for(a.bands), pretrained=not a.scratch).to(dev)
    import segmentation_models_pytorch as smp
    ce = nn.CrossEntropyLoss(); dice = smp.losses.DiceLoss(mode="multiclass")
    crit = lambda p, t: ce(p, t) + dice(p, t)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    best = -1.0
    for ep in range(a.epochs):
        model.train(); run = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        acc, miou, iou = evaluate(model, vl, dev)
        best = max(best, miou)
        print(f"ep {ep+1}/{a.epochs} | loss {run/len(tl):.3f} | acc {acc*100:.1f}% | mIoU {miou:.4f}"
              + ("  <- best" if miou == best else ""))
    print(f"   IoU/classe: {np.round(iou, 3).tolist()}")

    new = not os.path.exists("results_task2.csv")
    with open("results_task2.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "bands", "in_ch", "pretrained", "epochs", "best_mIoU"])
        w.writerow([a.tag, a.bands, in_channels_for(a.bands), not a.scratch, a.epochs, f"{best:.4f}"])
    print(f"OK '{a.tag}' best mIoU={best:.4f} -> results_task2.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bands", choices=["rgb", "msi", "msi_sar"], default="msi")
    p.add_argument("--scratch", action="store_true", help="encoder random (default: ImageNet)")
    p.add_argument("--subset", type=int, default=1500)
    p.add_argument("--val-subset", dest="val_subset", type=int, default=600)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", default="t2")
    main(p.parse_args())
