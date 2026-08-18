# Beyond Classification

Code for the BMVC 2026 paper:

> Beyond Classification: Structured Supervision Aligns Visual Evidence with Medical Semantics

## Requirements

Python 3.10+ and a GPU. Install deps with:

```
pip install -r requirements.txt
```

If the default PyTorch wheel does not match your CUDA version, install torch first, then the rest of the file.

## Data

Download BUSI (`Dataset_BUSI_with_GT`, Al-Dhabyani et al., Data in Brief 2020). The directory should contain `benign/`, `malignant/`, and `normal/`. Point the code at it:

```
export BUSI_DATA_ROOT=/path/to/Dataset_BUSI_with_GT
```

The 70/15/15 splits we used are in `configs/busi_splits/`. Paths in those json files are relative to `BUSI_DATA_ROOT`.

## Weights

ViT and ResNet pull ImageNet weights from torchvision.

For BiomedCLIP, download

https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

and put `open_clip_config.json` and `open_clip_pytorch_model.bin` in `checkpoints/BiomedCLIP/`, or set `BIOMEDCLIP_CKPT_DIR`.

For LVM-Med, download `lvmmed_vit.pth` from https://github.com/duyhominhnguyen/LVM-Med and put it at `checkpoints/lvmmed_vit.pth`, or set `LVM_MED_WEIGHTS`. You do not need the rest of that repo.

## How to run

All of this is from the repository root.

```
python -m co_pretraining.run list
python -m co_pretraining.run show vit_standard
```

Standard classifiers (AdamW, 5e-5, 50 epochs, batch 8). ResNet-50 is trained at 448 so layer-4 is 14x14, same grid as ViT patches:

```
python -m co_pretraining.run run vit_standard train --gpu 0 --epochs 50
python -m co_pretraining.run run resnet50_standard train --gpu 0 --epochs 50
```

Co-training (256 input, segmentation loss weight 2.0):

```
python -m co_pretraining.run run vit_cotraining train --gpu 0 --epochs 50
```

BiomedCLIP and LVM-Med:

```
python -m co_pretraining.run run biomedclip_zeroshot eval --gpu 0
python -m co_pretraining.run run biomedclip_linear_probe train --gpu 0
python -m co_pretraining.run run biomedclip_finetune train --gpu 0
python -m co_pretraining.run run lvm_med_linear_probe train --gpu 0
python -m co_pretraining.run run lvm_med_finetune train --gpu 0
```

BiomedCLIP fine-tuning unfreezes the last 12 visual blocks (visual lr 2e-5, head lr 1e-3, 30 epochs). LVM-Med linear probe is 20 epochs at 5e-4 on a frozen encoder; fine-tuning is 30 epochs at 1e-4 with grad accum 2.

Outputs go under `outputs/<name>/`. After a run, `eval` loads that checkpoint. Compare local numbers with:

```
python -m co_pretraining.run collect
```

## BUSI results

Accuracy is test-only. PIB@1: top-1 patch on the 14x14 grid inside the lesion box, pooled over train/val/test. Normal images are left out of PIB.

## Citation

```
@inproceedings{bai2026beyond,
  title={Beyond Classification: Structured Supervision Aligns Visual Evidence with Medical Semantics},
  author={Hexiang Bai and Hanyang Xu and Xiaoxue Li and Xiaoliang Wu and Shangde Gao and Hongxia Xu and Ke Liu},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2026}
}
```
