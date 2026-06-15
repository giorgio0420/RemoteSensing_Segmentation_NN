# -*- coding: utf-8 -*-
"""
SatMAE++ fMoW-Sentinel (ViT-L group-channel) come encoder CONGELATO + decoder, per segmentazione.
Riusa il modello UFFICIALE da techmn/satmae_pp (clonato in repo_dir) per garantire il match dei pesi.

Input: 10 bande ottiche S2 (ordine [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]); gruppi [[0,1,2,6],[3,4,5,7],[8,9]].
Se use_sar: input 12 canali (10 ottici + 2 SAR); il SAR entra come ramo laterale nel decoder.
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]


def _up(ic, oc):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
    )


class SatMAESentinelSeg(nn.Module):
    def __init__(self, num_classes, ckpt_path, repo_dir="/content/satmae_pp",
                 img_size=96, patch_size=8, ft_blocks=0, use_sar=False, sar_ch=16):
        super().__init__()
        if repo_dir not in sys.path:
            sys.path.append(repo_dir)
        import models_vit_group_channels as mvg

        self.embed_dim = 1024
        self.grid = img_size // patch_size            # 12
        self.encoder = mvg.vit_large_patch16(
            img_size=img_size, patch_size=patch_size, in_chans=10,
            channel_groups=GROUPS, num_classes=num_classes,
            global_pool=False, drop_path_rate=0.0)
        self.loaded = self._load(ckpt_path)

        # hook: cattura i token dell'ULTIMO blocco -> [B, 1+3*144, 1024] (robusto al pooling di forward)
        self._feat = None
        self.encoder.blocks[-1].register_forward_hook(lambda m, i, o: setattr(self, "_feat", o))

        for p in self.encoder.parameters():
            p.requires_grad = False
        self._frozen = (ft_blocks == 0)
        if ft_blocks > 0:
            for blk in list(self.encoder.blocks)[-ft_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            try:
                self.encoder.set_grad_checkpointing(True)
            except Exception:
                pass
        if self._frozen:
            self.encoder.eval()

        self.dec = nn.Sequential(_up(self.embed_dim, 256), _up(256, 128), _up(128, 64))  # 12->24->48->96
        self.use_sar = use_sar
        if use_sar:
            self.sar = nn.Sequential(
                nn.Conv2d(2, sar_ch, 3, padding=1), nn.BatchNorm2d(sar_ch), nn.ReLU(inplace=True),
                nn.Conv2d(sar_ch, sar_ch, 3, padding=1), nn.BatchNorm2d(sar_ch), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64 + (sar_ch if use_sar else 0), num_classes, 1)

    def _load(self, ckpt_path):
        if not ckpt_path or not os.path.exists(ckpt_path):
            print("[SatMAE-S] checkpoint non trovato -> encoder RANDOM (controllo)")
            return False
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ck = ck.get("model", ck)
        ck = {k: v for k, v in ck.items()
              if not (k.startswith("decoder") or k.startswith("proj_up")
                      or k in ("mask_token", "head.weight", "head.bias"))}
        missing, unexp = self.encoder.load_state_dict(ck, strict=False)
        tot = len(self.encoder.state_dict()); matched = tot - len(missing)
        print(f"[SatMAE-S] pesi {matched}/{tot} (missing={len(missing)}, unexpected={len(unexp)})")
        if matched < 0.7 * tot:
            print("[SatMAE-S] ⚠️ match basso: controlla i nomi delle chiavi.")
        return matched >= 0.7 * tot

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_frozen", True):
            self.encoder.eval()
        return self

    def _encode(self, x10):
        if self._frozen:
            with torch.no_grad():
                self.encoder.forward_features(x10)
        else:
            self.encoder.forward_features(x10)
        t = self._feat                                   # [B, 1+3*144, 1024]
        t = t[:, 1:, :]                                  # via cls
        g = self.grid * self.grid                        # 144
        ng = t.shape[1] // g                             # 3
        t = t.view(t.shape[0], ng, g, self.embed_dim).mean(1)              # media gruppi -> [B,144,1024]
        return t.transpose(1, 2).reshape(t.shape[0], self.embed_dim, self.grid, self.grid)  # [B,1024,12,12]

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        opt = x[:, :10]
        f = self._encode(opt)
        d = self.dec(f)
        if d.shape[-2:] != (H, W):
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        if self.use_sar:
            s = self.sar(x[:, 10:12])
            if s.shape[-2:] != d.shape[-2:]:
                s = F.interpolate(s, size=d.shape[-2:], mode="bilinear", align_corners=False)
            d = torch.cat([d, s], dim=1)
        return self.head(d)
