# Satellite Image Segmentation with Pretrained Models

Segmentazione di immagini satellitari/aeree (LoveDA) con backbone **pre-addestrati su dati
satellitari**, per quantificare quanto il pretraining aiuti in scenari con **poche etichette**
(*data-scarce*) — la domanda centrale della consegna.

## Idea del progetto (la "direzione")
> **Il pretraining su immagini satellitari migliora la segmentazione quando le etichette sono poche?**

Si confronta lo stesso modello (Swin-T U-Net) con encoder:
- **scratch** — pesi casuali (nessun pretraining, baseline);
- **imagenet** — pretraining generico;
- **rsp** — pretraining *satellitare* (Remote Sensing Pretraining, ViTAE/RSP);
- **satmaepp** *(opzionale)* — encoder **SatMAE++ ViT-L congelato** + decoder leggero.

…a parità di tutto, su sotto-insiemi crescenti di dati (100, 300, full). L'output è una
curva **mIoU vs numero di etichette** per ogni tipo di pretraining.

## Struttura
```
main.py                 # training di un singolo run (con override da CLI)
run_ablation.py         # lancia l'intera griglia di esperimenti -> results_summary.csv
config.py               # tutti i flag (pretraining, subset, ignore_index, wavelet, ...)
data/
  dataset.py            # wrapper torchgeo (LoveDA/LandCoverAI/DeepGlobe) + normalizzazione
  transforms.py         # augmentations + wavelet ISPAMM (toggle)
models/
  lightweight_unet.py   # Swin-T U-Net (scratch / imagenet / rsp)
  satmaepp_segmenter.py # SatMAE++ ViT-L congelato + decoder (Path B opzionale)
  sam_wrapper.py        # encoder SAM (per il pseudo-labeling)
utils/
  engine.py             # train/eval con mIoU, Dice e IoU per-classe (ignore_index)
  plots.py              # curve di loss e predizioni qualitative
pseudo_labeling.py      # SAM + U-Net -> pseudo-label (estensione "enlarge dataset")
```

## Setup su Google Colab
```python
# 1. Codice
!git clone https://github.com/giorgio0420/NN_segmentation.git
%cd NN_segmentation

# 2. Pesi satellitari RSP (Swin-T)  -> file rsp-swin-t-ckpt.pth
!pip install -q gdown
!gdown 1G5wjbjIHepmT6VVOuW03bWmyvrhcfe1F -O rsp-swin-t-ckpt.pth

# 3. Dipendenze
!pip install -q -r requirements.txt

# 4a. UN singolo run (default = rsp su tutto il dataset)
!python main.py

# 4b. ...oppure L'ESPERIMENTO completo (pretraining vs scratch, data-scarce)
!python run_ablation.py            # aggiungi --full per includere il run su tutti i dati
```
Risultati: `results_summary.csv` (tabella finale), `history_<tag>.csv` (per-epoca),
`loss_<tag>.png`, `best_<tag>.pth`, immagini in `output_samples/`.

## Flag principali (`config.py`)
| Flag | Cosa fa |
|---|---|
| `PRETRAINING_MODE` | `scratch` / `imagenet` / `rsp` / `satmaepp` |
| `TRAIN_SUBSET_SIZE` | n. immagini di train (scenario data-scarce). `None` = tutto |
| `VAL_SUBSET_SIZE` | cap sul val per eval veloci |
| `EVAL_EVERY` | valuta ogni N epoche (riduce i tempi) |
| `IGNORE_INDEX` | LoveDA: `0` = no-data, escluso da loss e mIoU |
| `USE_WAVELET_AUGMENTATION` | wavelet ISPAMM (applicato a train **e** val) |
| `USE_IMAGENET_NORM` | normalizzazione richiesta dai backbone pretrained |

## SatMAE++ (Path B, opzionale)
SatMAE++ a pieno regime (fine-tuning di ViT-L) è troppo pesante per Colab. Qui è usato come
**encoder congelato** (nessun backward sui ~300M parametri) + decoder leggero allenabile:
fattibile su T4. Serve un checkpoint SatMAE++ ViT-L (repo
[techmn/satmae_pp](https://github.com/techmn/satmae_pp)) salvato come `satmaepp_vitl_fmow.pth`,
poi `PRETRAINING_MODE = "satmaepp"` (consigliato `BATCH_SIZE` 8–16).

## Riferimenti
- Noman et al., *Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery* (SatMAE++), CVPR 2024.
- Wang et al., *RSP: An Empirical Study of Remote Sensing Pretraining*.
- Kirillov et al., *Segment Anything* (SAM) — usato nel pseudo-labeling.
