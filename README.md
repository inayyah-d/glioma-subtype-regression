# Glioma Subtype Proportion Regression from Multi-Modal MRI

Predicting how much of a glioma is **necrotic core**, **edema**, and **enhancing tumor** directly from a brain MRI slice. The model is a VGG-16 adapted for 4-channel MRI regression.

Course project for CSC 509 (Data Science for Medical Image Analysis) at San Francisco State University, Spring 2025.
Team: Iris Cruz, Paige Camaya, Dona Inayyah. Full write-up: [`report/glioma_subtype_regression_report.pdf`](report/glioma_subtype_regression_report.pdf).

## Problem

Tumor composition matters for diagnosis and treatment planning. Enhancing tumor marks active, growing tissue. Edema is fluid build-up that can raise pressure on the brain. Necrotic core is dead tissue at the center of the tumor.

This project does not segment each pixel. It treats the task as **whole-image regression**: given the four MRI modalities of a patient's middle axial slice, predict the proportions `[p1, p2, p4]` of the tumor made up of necrotic core (label 1), edema (label 2) and enhancing tumor (label 4).

## Data

**BraTS 2020**, the Multimodal Brain Tumor Segmentation Challenge from CBICA, University of Pennsylvania. Download it from: https://www.med.upenn.edu/cbica/brats2020/data.html

- Multi-modal 3D MRI in NIfTI (`.nii.gz`) format: T1, T1CE, T2 and FLAIR, plus expert segmentation masks.
- Original split: 369 training patients and 125 validation patients.
- Preprocessed by the organizers: skull-stripped, co-registered, and resampled to 240 × 240 × 155 voxels at 1 mm³.

During the course, the data was streamed from a course Google Cloud Storage bucket, which requires course credentials. Anyone else should download BraTS 2020 from the link above and arrange it like this:

```
DATA_DIR/
├── MICCAI_BraTS2020_TrainingData/
│   ├── BraTS20_Training_001/
│   │   ├── BraTS20_Training_001_t1.nii.gz
│   │   ├── BraTS20_Training_001_t1ce.nii.gz
│   │   ├── BraTS20_Training_001_t2.nii.gz
│   │   ├── BraTS20_Training_001_flair.nii.gz
│   │   └── BraTS20_Training_001_seg.nii.gz
│   └── ...
└── MICCAI_BraTS2020_ValidationData/
    └── ...
```

If you use BraTS data, please cite Menze et al. (IEEE TMI, 2015), Bakas et al. (Nature Scientific Data, 2017) and Bakas et al. (arXiv:1811.02629, 2018). The report's reference list has full citations.

## Approach

1. **Split.** All patient folders are pooled, shuffled with seed 42, and split 80/10/10. The split is saved to `split_log.csv`. Patients without a segmentation mask are dropped, leaving **294 train / 37 validation / 38 test** samples.
2. **Labels.** For each patient, the proportion of each tumor label is computed from the pixel counts in the middle axial slice.
3. **Preprocessing.**
   - Each modality slice gets percentile-based min-max normalization to [0, 1].
   - The four modalities are stacked as channels and resized to 224 × 224.
   - Preprocessed tensors are cached to disk. Before caching, 10 training epochs took about 2 hours.
   - Batch size is 8.
4. **Model.** VGG-16 pretrained on ImageNet, with these changes:
   - The first conv layer accepts **4 channels**. The pretrained RGB weights are copied, and the red-channel weights initialize the 4th channel.
   - The classifier is replaced with a **3-layer dense regression head** ending in a **sigmoid**, which outputs three values in [0, 1].
   - Training uses MSE loss and the Adam optimizer.
5. **Tuning** (per the report):
   - First, train only the regression head, with a grid search over 72 combinations of learning rate, dropout, weight decay, optimizer and epochs.
   - Then fine-tune manually, unfreezing more of the network and adjusting hyperparameters near the best grid-search configuration.

## Key results

These figures come from the final project report (Tables 2 and 3). Errors are in proportion units, where proportions run from 0 to 1.

| Model | Split | MSE | RMSE | MAE |
|---|---|---|---|---|
| Grid-search base (Exp 0: head only, lr 1e-4, 15 epochs, dropout 0.5) | Validation | – | 0.2157 | 0.1738 |
| Best fine-tuned (Exp 5: 8 unfrozen, lr 1e-4, 15 epochs, dropout 0.5, weight decay 1e-4) | Validation | – | 0.2045 | 0.1574 |
| **Best fine-tuned (Exp 5)** | **Held-out test** | **0.0876** | **0.2960** | **0.1976** |

The team set targets of MAE < 0.15 and RMSE < 0.20, and **the test results did not meet them**. Test error was also higher than validation error. The report puts this mainly down to the small training set: one 2D slice per patient, 294 training samples. It also notes that the ImageNet (natural-image) pretraining may not suit MRI.

The exploratory analysis in the report found that edema dominates tumor composition, with a mean proportion of 0.638. Necrotic core and enhancing tumor are right-skewed, with means of 0.179 and 0.183. Edema correlates negatively with necrotic core (−0.77) and with enhancing tumor (−0.62).

## About the notebook in this repo

[`notebooks/brats_vgg16_tumor_proportion_regression.ipynb`](notebooks/brats_vgg16_tumor_proportion_regression.ipynb) contains the whole pipeline: data split, normalization, labels, Dataset class and caching, the VGG-16 regression model, the training loop and learning-curve plots.

It is **an earlier experiment run than the one behind the report's numbers**. It has no grid search, no weight decay and no test-set evaluation, and its experiment numbering differs from the report's. Its saved outputs record the following validation metrics, measured after each run's final epoch:

| Run | Trainable params | `dropout` | lr | Epochs | RMSE | MAE | R² |
|---|---|---|---|---|---|---|---|
| Base | regression head only | 0.5 | 1e-4 | 10 | 0.2250 | 0.1698 | 0.0910 |
| 1 | last 4 tensors | 0.5 | 1e-4 | 15 | 0.2311 | 0.1803 | 0.0464 |
| 2 | last 6 tensors | 0.5 | 1e-4 | 9 | 0.2393 | 0.1923 | −0.0207 |
| 3 | last 6 tensors | 0.5 | 1e-4 | 20 | 0.2271 | 0.1756 | 0.0708 |
| 4 | last 6 tensors | 0.5 | 1e-4 | 9 | 0.2226 | 0.1777 | 0.1023 |
| 5 | last 3 tensors | 0.5 | 1e-4 | 9 | 0.2214 | 0.1758 | 0.1145 |
| 6 | last 6 tensors | 0.1 | 1e-4 | 20 | 0.2301 | 0.1765 | 0.0592 |
| 7 | last 6 tensors | 0.1 | 1e-5 | 20 | 0.2265 | 0.1798 | 0.0815 |
| 8 | last 6 tensors | 0.2 | 1e-5 | 20 | 0.2280 | 0.1852 | 0.0645 |
| 9 | last 6 tensors | 0.6 | 1e-5 | 20 | 0.2144 | 0.1760 | 0.1740 |
| 10 | last 4 tensors | 0.6 | 1e-5 | 20 | 0.2118 | 0.1738 | 0.1916 |
| 11 | last 6 tensors | 0.7 | 1e-5 | 20 | 0.2099 | 0.1752 | 0.2003 |
| 12 | last 6 tensors | 0.7 | 1e-4 | 20 | 0.2237 | 0.1825 | 0.0950 |
| 13 | last 6 tensors | 0.7 | 1e-5 | 25 | 0.2113 | 0.1733 | 0.1940 |

Notes for anyone re-running it:

- **The runs are chained.** Each fine-tuning run loads the previous run's checkpoint.
- **`unfreeze=N` unfreezes the last N parameter tensors, not the last N layers.** The regression head has 6 tensors (3 linear layers, each with a weight and a bias). So N ≤ 6 trains only parts of the head.
- **Non-default dropout resets the head.** When `dropout` ≠ 0.5, the training function rebuilds the regression head with new, randomly initialized weights after loading the checkpoint. This affects runs 6–13.

## How to run

A GPU is strongly recommended; the original run used a Colab T4.

**Locally**

```bash
pip install -r requirements.txt
# download BraTS 2020 (see Data) and arrange it as shown above
export DATA_DIR=/path/to/BraTS2020      # default: data/BraTS2020
export OUTPUT_DIR=/path/to/outputs      # split log, tensor cache, checkpoints, metrics; default: outputs
jupyter notebook notebooks/brats_vgg16_tumor_proportion_regression.ipynb
```

Run the **0. Configuration** cell. Skip the cells marked *Colab only*, which mount GCS and Google Drive, then run the rest in order.

**On Google Colab**

Run the Colab-only setup cells. The GCS bucket needs course access, so you can instead upload the data to Drive. In the configuration cell, set `DATA_DIR` and `OUTPUT_DIR` to wherever your data and Drive folder are mounted. The original run used `/content/images/Module1_BraTS` and `/content/drive/MyDrive`.

## Limitations and future work

From the report:

- **Small data.** The dataset is small for deep learning. The next steps would be data augmentation and pretraining on medical images (e.g. RadImageNet, MedicalNet) instead of ImageNet.
- **Tuning.** Early stopping and Bayesian hyperparameter search are left for future work.

One more limitation follows from the method rather than from the report: labels come from the middle axial slice only, so they describe that slice, not the whole tumor volume.

## Repository layout

```
├── notebooks/brats_vgg16_tumor_proportion_regression.ipynb   # full pipeline (earlier experiment run)
├── report/glioma_subtype_regression_report.pdf               # final project report
├── requirements.txt
└── README.md
```
