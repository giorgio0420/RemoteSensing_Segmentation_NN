import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SatMAEppSegmenter(nn.Module):
    """
    PATH B (opzionale, coerente col titolo della consegna).

    SatMAE++ (Noman et al., CVPR 2024) usato come encoder ViT-L *CONGELATO* + decoder leggero.
    Idea: SatMAE++ a pieno regime (fine-tuning di ViT-L) e' troppo pesante per Colab.
    Congelando l'encoder NON si memorizzano le attivazioni per il backward: memoria e tempo
    crollano e su una T4 diventa fattibile. Si allena solo il decoder.

    Requisiti:
      - `timm` (per lo scheletro ViT-L/16 @224)
      - un checkpoint SatMAE++ ViT-L (es. pretrain fMoW-RGB) salvato in `ckpt_path`.
        Repo ufficiale: https://github.com/techmn/satmae_pp

    NOTA ONESTA: la rimappatura delle chiavi del checkpoint puo' richiedere un piccolo
    aggiustamento a seconda di come e' salvato il file. Il modulo stampa quante chiavi
    combaciano: se il match e' basso, lancia `python inspect_pth.py` sul checkpoint e
    si sistema la mappatura.
    """

    def __init__(self, num_classes=8, ckpt_path="satmaepp_vitl_fmow.pth",
                 img_size=224, freeze_encoder=True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("Serve `timm` per SatMAE++ (pip install timm).") from e

        self.img_size = img_size
        self.patch = 16
        self.grid = img_size // self.patch      # 14
        self.embed_dim = 1024                   # ViT-L

        # Scheletro ViT-L/16 (i pesi SatMAE++ ci vengono caricati sopra)
        self.encoder = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
        self.encoder_loaded = self._load_satmae(ckpt_path)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # Decoder leggero: [B,1024,14,14] -> feature, poi upsample a piena risoluzione
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def _load_satmae(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"[SatMAE++] ⚠️ checkpoint '{ckpt_path}' non trovato: encoder RANDOM.")
            return False

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict"):
                if key in ckpt:
                    ckpt = ckpt[key]
                    break

        # I checkpoint MAE/SatMAE usano nomi compatibili con timm (patch_embed.proj, blocks.*,
        # norm, pos_embed, cls_token). Rimuoviamo solo prefissi comuni e le chiavi del
        # decoder MAE (che a noi non servono).
        cleaned = {}
        for k, v in ckpt.items():
            nk = k.replace("module.", "").replace("encoder.", "")
            if nk.startswith("decoder") or "mask_token" in nk:
                continue
            cleaned[nk] = v

        missing, unexpected = self.encoder.load_state_dict(cleaned, strict=False)
        total = len(self.encoder.state_dict())
        matched = total - len(missing)
        print(f"[SatMAE++] Pesi caricati: {matched}/{total} tensori "
              f"(missing={len(missing)}, unexpected={len(unexpected)}).")
        if matched < 0.5 * total:
            print("[SatMAE++] ⚠️ Match basso: i nomi delle chiavi non combaciano. "
                  "Esegui inspect_pth.py sul checkpoint e adatta la rimappatura.")
            return False
        print("[SatMAE++] 🚀 Encoder ViT-L inizializzato con pesi SatMAE++ (congelato).")
        return True

    def _forward_tokens(self, x):
        feats = self.encoder.forward_features(x)          # timm: [B, N(+1), C]
        if feats.dim() == 3 and feats.shape[1] == self.grid * self.grid + 1:
            feats = feats[:, 1:, :]                        # rimuovi cls token
        return feats

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = self._forward_tokens(x)               # [B, 196, 1024]
        B = x.shape[0]
        feats = feats.transpose(1, 2).reshape(B, self.embed_dim, self.grid, self.grid)  # [B,1024,14,14]
        d = self.decoder(feats)
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return self.classifier(d)
