# Satellite Image Segmentation with Pretrained Models

Neural Networks project — **does pretraining on satellite imagery improve land-cover
segmentation when labels are scarce?** Two complementary studies on remote-sensing data.

| | Task 1 — Pretraining (main) | Task 2 — Multi-modal inputs (secondary) |
|---|---|---|
| **Dataset** | LoveDA (RGB aerial, 7 classes) | DFC2020 (Sentinel-1 + Sentinel-2, 8 classes) |
| **Question** | satellite pretraining vs scratch, in data-scarce | multispectral + radar; SatMAE++ pretraining vs random |
| **Backbones** | Swin-T U-Net (RSP), ViT-L (SatMAE++ fMoW-RGB) | ViT-L group-channel (SatMAE++ fMoW-Sentinel) |
| **Entry point** | `main.py` | `task2_multispectral.py` |

A full technical report (models, weights, comparisons, dataset-switching guide) is in
**`Report_Tecnico_NN.docx`**.

## Repository structure
```
main.py                      # Task 1: training/eval/ablation (modes, input scale, wavelet)
config.py                    # hyperparams, dataset, paths
run_ablation.py              # Task 1: grid runner -> results_summary.csv
data/dataset.py              # LoveDA via torchgeo (+ landcoverai/deepglobe)
data/transforms.py           # resize|crop preprocessing + wavelet augmentation (ISPAMM)
models/lightweight_unet.py   # Swin-T U-Net (scratch/imagenet/RSP)
models/satmaepp_segmenter.py # SatMAE++ ViT-L fMoW-RGB (frozen) + decoder
models/rsp_wavelet_unet.py   # Swin + wavelet-detail decoder (wavelet ablation)
utils/engine.py, plots.py    # train/eval loops, metrics (mIoU/Dice), figures
task2_multispectral.py       # Task 2: DFC2020 loader + SatMAE++-Sentinel / ResNet U-Net
satmae_sentinel.py           # SatMAE++ ViT-L group-channel (frozen) + decoder
```

## Pretrained weights
| Backbone | Source |
|---|---|
| Swin-T **RSP** (MillionAID) | Google Drive `1G5wjbjIHepmT6VVOuW03bWmyvrhcfe1F` -> `rsp-swin-t-ckpt.pth` |
| **SatMAE++** ViT-L fMoW-RGB | HF `mubashir04/checkpoint_ViT-L_pretrain_fmow_rgb` |
| **SatMAE++** ViT-L fMoW-Sentinel | HF `mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel` |

## Quick start (Colab, GPU T4)
**Task 1** — pretraining ablation on LoveDA:
```bash
python main.py --mode rsp      --train-subset 300 --epochs 20 --tag rsp
python main.py --mode scratch  --train-subset 300 --epochs 20 --tag scratch
python main.py --mode satmaepp --train-subset 300 --epochs 20 --tag satmae   # needs satmaepp_vitl_fmow.pth
```
**Task 2** — multispectral on DFC2020 (needs `git clone techmn/satmae_pp` + Sentinel ckpt):
```bash
python task2_multispectral.py --model satmae --ckpt <sentinel.pth> --bands msi     --class-weights --ft-blocks 4 --lr 1e-4 --tag satmae_pre
python task2_multispectral.py --model satmae                       --bands msi     --class-weights --ft-blocks 4 --lr 1e-4 --tag satmae_rand
python task2_multispectral.py --model satmae --ckpt <sentinel.pth> --bands msi_sar --class-weights --ft-blocks 4 --lr 1e-4 --tag satmae_pre_sar
```

## Key findings
- **Pretraining helps in data-scarce** (LoveDA, n=300): SatMAE++ frozen ~= **0.31 mIoU**, RSP ~= 0.25, scratch ~= 0.09.
- **Class-weighting** recovers rare classes (road/water) — large mIoU gain.
- **Wavelet** strategies: rigorously evaluated (input + decoder, Swin + ViT) -> **neutral** on semantic
  segmentation (bottleneck is semantics, not frequency) — an explained negative result.
- **Multispectral + radar** (Task 2, DFC2020): SatMAE++-Sentinel pretrained > random; radar (S1) aids water.

## References
SatMAE++ (Noman et al., CVPR 2024) - RSP (ViTAE-Transformer) - SAMRS - DFC2020 (GFM-Bench).
