import torch
import os

class Config:
    # ===================== PATHS =====================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "dataset")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")

    # ===================== BACKBONE / PRETRAINING =====================
    # Questa e' la leva centrale del progetto (la description chiede di confrontare
    # "with and without pretraining"). Valori possibili:
    #   "scratch"  -> Swin-T inizializzato a CASO (baseline: NESSUN pretraining)
    #   "imagenet" -> Swin-T con pesi ImageNet (timm)            -> pretraining generico
    #   "rsp"      -> Swin-T con pesi SATELLITARI RSP            -> pretraining nel dominio  [DEFAULT]
    #   "satmaepp" -> SatMAE++ ViT-L come encoder CONGELATO + decoder leggero (Path B, opzionale)
    PRETRAINING_MODE = "rsp"

    ENCODER_NAME = "tu-swin_tiny_patch4_window7_224"  # backbone Swin-T (per le 3 modalita' sopra tranne satmaepp)
    RSP_WEIGHTS_PATH = "rsp-swin-t-ckpt.pth"          # checkpoint RSP (gdown nel notebook Colab)
    SATMAEPP_CKPT_PATH = "satmaepp_vitl_fmow.pth"     # checkpoint SatMAE++ ViT-L (repo techmn/satmae_pp)

    # ===================== DATASET =====================
    DATASET_NAME = "loveda"  # "loveda", "landcoverai", "deepglobe"

    CLASS_WEIGHTS = None       # pesi-classe per la loss (usati solo con --class-weights). None = nessuno
    if DATASET_NAME == "loveda":
        NUM_CLASSES = 8       # le maschere hanno valori 0..7
        IGNORE_INDEX = 0      # 0 = "no-data": va ESCLUSO da loss e mIoU (convenzione ufficiale LoveDA)
        # Median-frequency balancing (indice 0 = no-data -> peso 0). road/water/building pesano di piu'.
        CLASS_WEIGHTS = [0.0, 0.21, 1.04, 2.29, 1.22, 1.00, 0.57, 0.30]
    elif DATASET_NAME == "landcoverai":
        NUM_CLASSES = 5
        IGNORE_INDEX = -100   # nessuna classe ignorata
    elif DATASET_NAME == "deepglobe":
        NUM_CLASSES = 7
        IGNORE_INDEX = -100
    else:
        NUM_CLASSES = 2
        IGNORE_INDEX = -100

    # Nomi classi (solo per stampe/plot leggibili). LoveDA: indice 0 = no-data.
    CLASS_NAMES = ["no-data", "background", "building", "road",
                   "water", "barren", "forest", "agriculture"]

    # ===================== DATA-SCARCE / VELOCITA' =====================
    # Cuore della narrativa "data-scarce" della consegna E leva principale sui tempi.
    # Allenando su pochi campioni l'epoca dura pochi secondi -> ablation fattibili.
    TRAIN_SUBSET_SIZE = None   # es. 100, 300, 600 ... None = tutto il train (~2520 img)
    VAL_SUBSET_SIZE = None      # None = tutto il val (numero "ufficiale"). run_ablation lo cappa per velocita'
    EVAL_EVERY = 1             # valuta ogni N epoche (alza per risparmiare tempo di eval)
    SEED = 42                  # riproducibilita' (stesso subset, stessi risultati)

    # ===================== TRAINING =====================
    BATCH_SIZE = 32            # Swin-T @224 su T4. Per satmaepp (ViT-L) abbassa a 8-16.
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20            # con backbone pretrained converge prima di 30
    IMAGE_SIZE = 224           # richiesto da Swin-T (patch embedding 224)
    INPUT_MODE = "resize"      # "resize" (immagine intera ridotta) o "crop" (patch nativa 224 random)
    CROP_RESIZE = 0            # se >0 e INPUT_MODE='crop': resize a N poi crop (es. 512 -> ~20% area, mezza risoluzione)
    FREEZE_EPOCHS = 3          # epoche con encoder congelato (warmup). 0 = mai. Ignorato in "scratch"/"satmaepp".

    # ===================== WAVELET (ISPAMM) =====================
    # Ora la wavelet e' una AUGMENTATION non distruttiva (unsharp-masking) applicata SOLO in
    # training con probabilita' WAVELET_P e intensita' WAVELET_ALPHA. Val sempre pulito.
    USE_WAVELET_AUGMENTATION = False
    WAVELET_TYPE = 'haar'
    WAVELET_ALPHA = 0.3        # intensita' dello sharpening (0.1=delicato, 0.5=marcato)
    WAVELET_P = 0.5            # probabilita' di applicarla per immagine (augmentation)

    # ===================== NORMALIZZAZIONE =====================
    # I backbone pretrained si aspettano normalizzazione ImageNet (mean/std), non solo /255.
    USE_IMAGENET_NORM = True

    # ===================== CLASS WEIGHTING =====================
    # Pesa le classi rare nella loss (best practice su dataset sbilanciati come LoveDA).
    # ON di default; disattivabile per-run con --no-class-weights.
    USE_CLASS_WEIGHTS = True

    # ===================== HARDWARE =====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
