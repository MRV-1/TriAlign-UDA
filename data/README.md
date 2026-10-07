# Data and feature cache layout

The released notebooks operate on pre-extracted frozen UNI2-H feature representations and do not redistribute the underlying datasets.

Expected local cache files:

```text
cache/uni2_h/
├── nct_rgb.npy
├── nct_labels_trafiq_order.npy
├── crc7k_rgb.npy
├── crc7k_labels_trafiq_order.npy
├── tcga_rgb.npy
└── class_names_trafiq_order.npy
```

## Dataset roles

- **NCT-CRC-HE-100K** — labeled source domain. A stratified 90/10 split (seed 42) is used for source training and source-validation checkpoint selection.
- **TCGA-COAD/READ** — completely unlabeled target domain. The final experiments use one fixed 10,000-patch subset selected with seed 42.
- **CRC-VAL-HE-7K** — independent external evaluation only. It is not used for training, adaptation, hyperparameter selection, or checkpoint selection.

## TCGA patch protocol reported in the manuscript

The manuscript reports 459 TCGA-COAD/READ whole-slide images. Non-overlapping 512 × 512 Level-0 patches were extracted with stride 512 and resized to 256 × 256 pixels. Patches with at least 80% background were excluded, and at most 500 eligible patches per WSI were retained. The fixed 10,000-patch target subset was then sampled from the eligible pooled patch set using seed 42.

The notebooks in this release start from the resulting frozen UNI2-H feature banks. Dataset download, WSI tiling, and UNI2-H feature extraction must therefore be performed before running the adaptation notebooks.
