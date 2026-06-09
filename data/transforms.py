import albumentations as A
import numpy as np
import pywt
import cv2


def apply_wavelet_fusion(image, wavelet='haar'):
    """
    Strategia ispirata alla ricerca wavelet del laboratorio ISPAMM.
    Applica una DWT 2D, amplifica le alte frequenze (bordi/texture) e le rifonde
    sul canale di luminanza (L in LAB). Utile per accentuare i confini di edifici/campi.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    coeffs2 = pywt.dwt2(gray, wavelet)
    LL, (LH, HL, HH) = coeffs2

    # Amplifica le alte frequenze
    LH, HL, HH = LH * 1.5, HL * 1.5, HH * 1.5

    enhanced_gray = pywt.idwt2((LL, (LH, HL, HH)), wavelet)
    enhanced_gray = np.clip(enhanced_gray, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    enhanced_gray = cv2.resize(enhanced_gray, (lab.shape[1], lab.shape[0]))
    lab[:, :, 0] = enhanced_gray
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result


# Funzione a livello di modulo (NON una lambda): cosi' e' "picklable" e compatibile
# con il multiprocessing del DataLoader (num_workers>0) -> niente piu' UserWarning.
def _wavelet_image_fn(image, **kwargs):
    return apply_wavelet_fusion(image, 'haar')


def get_train_transforms(image_size, use_wavelet=False):
    transforms = []
    if use_wavelet:
        transforms.append(A.Lambda(name="wavelet_fusion", image=_wavelet_image_fn, p=1.0))
    transforms.extend([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
    return A.Compose(transforms)


def get_val_transforms(image_size, use_wavelet=False):
    transforms = []
    # IMPORTANTE: se il wavelet e' attivo va applicato anche in validazione,
    # altrimenti train e val avrebbero statistiche diverse (mismatch di dominio).
    if use_wavelet:
        transforms.append(A.Lambda(name="wavelet_fusion", image=_wavelet_image_fn, p=1.0))
    transforms.extend([
        A.Resize(image_size, image_size),
    ])
    return A.Compose(transforms)
