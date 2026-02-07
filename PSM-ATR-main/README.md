# PSM-ATR: Learning Robust Disease-Treatment Semantics with Associative Treatment Retrieval for Medication Recommendation

This repository contains the official PyTorch implementation of the paper:

**PSM-ATR: Learning Robust Disease-Treatment Semantics with Associative Treatment Retrieval for Medication Recommendation**



## Repository Structure

```
PSM-ATR-main/
├── code/
│   ├── model.py              # PSM-ATR model architecture
│   ├── train.py              # Training and evaluation script
│   ├── utils_.py             # Utility functions and metrics
│   └── hflayers/             # Modern Hopfield Network layers
├── data/
│   ├── mimic-iii/            # MIMIC-III processed data
│   └── mimic-iv/             # MIMIC-IV processed data
├── weights/                  # Saved model weights
├── processing.py             # Data pre-processing script
├── run.sh                    # Quick-start training commands
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- tqdm

## Data Preparation

1. Download the raw EHR datasets from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) (credentials required).

2. We follow the pre-processing pipeline from [Carmen](https://github.com/bit1029public/Carmen) for a fair comparison. The processed data (in JSON format) should be placed under `data/mimic-iii/` and `data/mimic-iv/` with the following files:
   - `records_final.json` — patient visit records
   - `voc_final.json` — vocabulary mappings
   - `ddi_A_final.json` — drug-drug interaction adjacency matrix
   - `ehr.json` — EHR co-occurrence adjacency matrix

## Training & Evaluation

### Quick Start

```bash
# Train on MIMIC-III with pre-training
python code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203 \
    --pretrain_mask --pretrain_epochs 15

# Train on MIMIC-IV with pre-training
python code/train.py --dataset 4 --dim 256 --batch 32 --visit 3 --seed 1203 \
    --pretrain_mask --pretrain_epochs 15
```


```

### Train without Pre-training (Ablation)

```bash
python code/train.py --dataset 3 --dim 256 --batch 32 --visit 3 --seed 1203
```

[//]: # (## Citation)

[//]: # ()
[//]: # (If you find this repository useful for your research, please cite our paper:)

[//]: # (```bibtex)

[//]: # (@article{psm-atr2026,)

[//]: # (  title={PSM-ATR: Learning Robust Disease-Treatment Semantics with Associative Treatment Retrieval for Medication Recommendation},)

[//]: # (  author={},)

[//]: # (  year={2026})

[//]: # (})

[//]: # (```)

## Acknowledgements

This codebase builds upon and references the following excellent works:

- [Carmen](https://github.com/bit1029public/Carmen)
- [COGNet](https://github.com/BarryRun/COGNet)
- [SafeDrug](https://github.com/ycq091044/SafeDrug)
- [GAMENet](https://github.com/sjy1203/GAMENet)
- [Hopfield Networks is All You Need](https://github.com/ml-jku/hopfield-layers)
