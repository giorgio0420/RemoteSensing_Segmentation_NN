import albumentations as A
import numpy as np
import pywt
import cv2
from functools import partial


def apply_wavelet_sharpen(image, wavelet='haar', alpha=0.3):
    """
    Wavelet "unsharp masking" NON distruttivo (versione migliorata della vecchia fusion).

    La vecchia apply_wavelet_fusion SOSTITUIVA la luminanza con un grigio a bordi amplificati:
    input fuori distribuzione rispetto al pretraining RSP -> peggiorava le performance.

    Qui invece:
      - lavoriamo sul solo canale L (LAB); il colore (a,b) resta intatto;
      - ricostruiamo la SOLA componente ad alta frequenza (azzerando l'approssimazione LL);
      - la RIAGGIUNGIAMO all'originale pesata da alpha:  L_out = L + alpha * dettaglio.
    Risultato: bordi piu' nitidi ma immagine vicina all'originale -> adatta come AUGMENTATION.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)

    LL, (LH, HL, HH) = pywt.dwt2(L, wavelet)
    detail = pywt.idwt2((np.zeros_like(LL), (LH, HL, HH)), wavelet)   # solo alte frequenze
    detail = cv2.resize(detail, (L.shape[1], L.shape[0]))             # riallinea (padding DWT)

    L_out = np.clip(L + alpha * detail, 0, 255).astype(np.uint8)
    lab[:, :, 0] = L_out
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# Funzione a livello di modulo (picklable -> compatibile con DataLoader multiprocess).
# alpha viene legato con functools.partial al momento della creazione del transform.
def _wavelet_sharpen_fn(image, alpha=0.3, **kwargs):
    return apply_wavelet_sharpen(image, 'haar', alpha)


def _intake(image_size, input_mode, train):
    """resize = riduce l'immagine intera; crop = patch NATIVA image_size (Random in train, Center in val)."""
    if input_mode == "crop":
        return [
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(image_size, image_size) if train else A.CenterCrop(image_size, image_size),
        ]
    return [A.Resize(image_size, image_size)]


def get_train_transforms(image_size, use_wavelet=False, wavelet_alpha=0.3, wavelet_p=0.5, input_mode="resize"):
    """
    (resize | crop) -> (opz.) wavelet sharpening RANDOM -> flip/rotate.
    crop = patch nativa 224x224 random dentro l'immagine: piu' dettaglio, meno contesto.
    """
    tfs = _intake(image_size, input_mode, train=True)
    if use_wavelet:
        tfs.append(A.Lambda(
            name="wavelet_sharpen",
            image=partial(_wavelet_sharpen_fn, alpha=wavelet_alpha),
            p=wavelet_p))
    tfs += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
    return A.Compose(tfs)


def get_val_transforms(image_size, use_wavelet=False, input_mode="resize"):
    """
    Validation: nessuna augmentation. resize -> immagine intera ridotta; crop -> CenterCrop nativo.
    NB: in crop mode la mIoU e' misurata sul ritaglio centrale, non sull'intero tile (vedi nota).
    """
    return A.Compose(_intake(image_size, input_mode, train=False))
