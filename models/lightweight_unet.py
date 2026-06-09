import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class LightweightUNet(nn.Module):
    def __init__(self, num_classes=2, encoder_name="tu-swin_tiny_patch4_window7_224",
                 pretraining_mode="rsp", rsp_weights_path="rsp-swin-t-ckpt.pth"):
        """
        U-Net leggera con backbone Swin-Tiny (segmentation-models-pytorch + timm).

        pretraining_mode:
          "scratch"  -> encoder inizializzato a caso (baseline SENZA pretraining)
          "imagenet" -> encoder con pesi ImageNet
          "rsp"      -> encoder con pesi SATELLITARI RSP (caricati da rsp_weights_path)
        """
        super().__init__()
        self.pretraining_mode = pretraining_mode
        print(f"Initializing U-Net | backbone: {encoder_name} | pretraining: {pretraining_mode}")

        encoder_weights = "imagenet" if pretraining_mode == "imagenet" else None

        try:
            self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                  in_channels=3, classes=num_classes)
            print("Backbone smp caricato.")
        except Exception as e:
            print(f"Impossibile caricare {encoder_name}: {e}\nFallback su ResNet-34 (ImageNet).")
            self.model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                                  in_channels=3, classes=num_classes)
            return

        if pretraining_mode == "scratch":
            print("⚪ Backbone RANDOM (baseline: nessun pretraining).")
        elif pretraining_mode == "imagenet":
            print("🟢 Backbone ImageNet caricato.")
        elif pretraining_mode == "rsp":
            self._load_rsp_weights(rsp_weights_path)

    def _load_rsp_weights(self, rsp_weights_path):
        if not os.path.exists(rsp_weights_path):
            print(f"⚠️ Pesi RSP '{rsp_weights_path}' non trovati. Scaricali (gdown) o l'encoder resta random.")
            return

        state_dict = torch.load(rsp_weights_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        # === MAPPATURA CHIAVI Swin "vecchio" (RSP) -> Swin "moderno" (smp/timm) ===
        # Nel vecchio Swin il downsample sta alla FINE dello stage i (layers.i.downsample);
        # nello Swin moderno sta all'INIZIO dello stage i+1 (layers_{i+1}.downsample).
        mapped = {}
        for k, v in state_dict.items():
            new_k = "model." + k
            if "downsample" in k:
                parts = new_k.split('.')
                if parts[1] == "layers":
                    parts[2] = str(int(parts[2]) + 1)
                new_k = ".".join(parts).replace("layers.", "layers_")
            else:
                new_k = new_k.replace("layers.", "layers_")
            mapped[new_k] = v

        missing, unexpected = self.model.encoder.load_state_dict(mapped, strict=False)
        if len(missing) > 100:
            print("⚠️ ATTENZIONE: la maggior parte dei pesi RSP non ha combaciato. Encoder rimasto random.")
        else:
            print("🚀 Pesi satellitari Swin-T (RSP) iniettati con successo!")

    def forward(self, x):
        return self.model(x)
