# -*- coding: utf-8 -*-
"""
TASK 2 (secondaria) - Multi-modal / multi-spectral inputs su DFC2020 (Sentinel-1 + Sentinel-2).

Legge i .tif DIRETTAMENTE dallo zip HF (niente libreria `datasets`, niente estrazione su disco
-> nessun problema di estrazioni parziali). Rimappa le etichette a 8 classi (+255 ignore).

  (Q1) bande extra aiutano? -> RGB(3) vs S2 multispettrale(10) vs +radar(12)
  (Q2) il pretraining aiuta? -> encoder ImageNet vs scratch (stessa rete)
Modello: U-Net (ResNet-34) via segmentation-models-pytorch, in_channels variabile.
"""
import argparse, csv, io, os, random, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

S2_MEAN = np.array([1370.19,1184.38,1120.77,1136.26,1263.74,1645.40,1846.87,1762.60,1972.62,582.73,14.77,1732.16,1247.92], dtype=np.float32)
S2_STD  = np.array([633.15,650.28,712.13,965.23,948.98,1108.07,1258.36,1233.15,1364.39,472.38,14.31,1310.37,1087.60], dtype=np.float32)
S1_MEAN = np.array([-12.55,-20.19], dtype=np.float32)
S1_STD  = np.array([5.26,5.91], dtype=np.float32)

# 13 bande S2: 0 B1,1 B2,2 B3,3 B4,4 B5,5 B6,6 B7,7 B8,8 B8A,9 B9,10 B10,11 B11,12 B12
RGB_IDX = [3, 2, 1]                              # R,G,B = B4,B3,B2
MSI_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]       # 10 bande (scarta B1,B9,B10) = set SatMAE-Sentinel

# rimappatura etichette grezze (0..17) -> 8 classi; 255 = ignore (dal loading script ufficiale)
DFC_MAP = np.array([255, 0,0,0,0,0, 1,1, 255,255, 2, 3, 4, 5, 4, 255, 6, 7], dtype=np.int64)
N_CLASSES = 8
IGNORE = 255
SEED = 42


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def prepare_data():
    """Scarica DFC2020.zip (cache HF) e legge metadata DIRETTAMENTE dallo zip. Niente estrazione."""
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download("GFM-Bench/DFC2020", "data/DFC2020.zip", repo_type="dataset")
    with zipfile.ZipFile(zp) as z:
        meta = [n for n in z.namelist() if n.endswith("metadata.csv")][0]
        prefix = meta[: -len("metadata.csv")]        # es. "DFC2020/"
        df = pd.read_csv(io.BytesIO(z.read(meta)))
    print(f"zip={os.path.basename(zp)} | prefix='{prefix}' | righe metadata={len(df)}")
    return zp, prefix, df


class DFC2020(Dataset):
    def __init__(self, zip_path, prefix, df_split, mode="msi", subset=None, seed=SEED):
        self.zip_path = zip_path
        self.prefix = prefix
        self._z = None                              # ZipFile aperto pigramente per-worker (post-fork)
        self.df = df_split.reset_index(drop=True)
        self.mode = mode
        self.bands = RGB_IDX if mode == "rgb" else MSI_IDX
        idx = list(range(len(self.df)))
        if subset and subset < len(idx):
            random.Random(seed).shuffle(idx); idx = idx[:subset]
        self.idx = idx

    def _read(self, rel):
        import tifffile
        if self._z is None:
            self._z = zipfile.ZipFile(self.zip_path)
        im = tifffile.imread(io.BytesIO(self._z.read(self.prefix + rel)))
        if im.ndim == 2:
            im = im[..., None]
        return np.transpose(im, (2, 0, 1))          # H,W,C -> C,H,W

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        row = self.df.iloc[self.idx[i]]
        opt = self._read(row.optical_path).astype(np.float32)           # [13,96,96]
        opt = (opt - S2_MEAN[:, None, None]) / S2_STD[:, None, None]
        x = opt[self.bands]
        if self.mode == "msi_sar":
            rad = self._read(row.radar_path).astype(np.float32)         # [2,96,96]
            rad = (rad - S1_MEAN[:, None, None]) / S1_STD[:, None, None]
            x = np.concatenate([x, rad], axis=0)
        lab = self._read(row.label_path)[0].astype(np.int64)            # [96,96] valori 0..17
        lab = DFC_MAP[lab]                                              # -> 0..7 / 255
        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(lab).long()


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
    inter = torch.zeros(N_CLASSES, device=device); union = torch.zeros(N_CLASSES, device=device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            pred = model(x).argmax(1)
        valid = y != IGNORE
        correct += ((pred == y) & valid).sum().item(); total += valid.sum().item()
        for c in range(N_CLASSES):
            pc = (pred == c) & valid; yc = (y == c) & valid
            inter[c] += (pc & yc).sum(); union[c] += (pc | yc).sum()
    iou = inter / (union + 1e-6); v = union > 0
    return correct / max(total, 1), (iou[v].mean().item() if v.any() else 0.0), iou.cpu().numpy()


def main(a):
    set_seed()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    zp, prefix, df = prepare_data()
    tr = DFC2020(zp, prefix, df[df.split == "train"], a.bands, subset=a.subset)
    va = DFC2020(zp, prefix, df[df.split == "val"], a.bands, subset=a.val_subset)
    print(f"mode={a.bands} | pretrained={not a.scratch} | in_ch={in_channels_for(a.bands)} | "
          f"train {len(tr)} | val {len(va)} | dev={dev}")
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(va, batch_size=a.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    import segmentation_models_pytorch as smp
    model = build_model(in_channels_for(a.bands), pretrained=not a.scratch).to(dev)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)
    crit = lambda p, t: ce(p, t) + dice(p, t)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    best = -1.0; best_iou = None
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
        if miou > best:
            best = miou; best_iou = iou
        print(f"ep {ep+1}/{a.epochs} | loss {run/len(tl):.3f} | acc {acc*100:.1f}% | mIoU {miou:.4f}"
              + ("  <- best" if miou == best else ""))
    print(f"   best mIoU={best:.4f} | IoU/classe={np.round(best_iou,3).tolist()}")

    new = not os.path.exists("results_task2.csv")
    with open("results_task2.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "bands", "in_ch", "pretrained", "epochs", "best_mIoU"])
        w.writerow([a.tag, a.bands, in_channels_for(a.bands), not a.scratch, a.epochs, f"{best:.4f}"])
    print(f"OK '{a.tag}' -> results_task2.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bands", choices=["rgb", "msi", "msi_sar"], default="msi")
    p.add_argument("--scratch", action="store_true", help="encoder random (default: ImageNet)")
    p.add_argument("--subset", type=int, default=2000)
    p.add_argument("--val-subset", dest="val_subset", type=int, default=800)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", default="t2")
    main(p.parse_args())
