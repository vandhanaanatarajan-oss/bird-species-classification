# 🐦 Bird Species Classification + Sound Event Detection
### ECS7013P Deep Learning — Vandhana Natarajan Chitra

A full deep learning pipeline that combines a **custom CNN classifier** with **BirdNET** (a pretrained bird sound detector) to identify bird species from audio recordings, using data sourced from Xeno-canto.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vandhanaanatarajan-oss/bird-species-classification/blob/main/bird_coursework.ipynb)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Outputs](#outputs)
- [Results](#results)
- [Key Parameters](#key-parameters)
- [References](#references)

---

## Overview

This project tackles bird species identification from `.mp3` audio recordings using two complementary approaches, then combines them into a hybrid system:

| Task | Approach | Description |
|------|----------|-------------|
| **Task A** | Custom CNN | Classifies full audio clips converted to Mel-spectrograms |
| **Task B** | BirdNET SED | Pretrained Sound Event Detection to find bird call segments |
| **Combined A** | SED-guided majority vote | CNN runs on BirdNET-detected segments; majority label wins |
| **Combined B** | Confidence-weighted vote | BirdNET confidence scores weight CNN predictions per segment |

---

## Project Structure

```
bird-species-classification/
│
├── bird_coursework.ipynb            # Full pipeline notebook
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore
│
├── DL Report.pdf                    # Written report
│
└── outputs/                         # Generated after running notebook
    ├── spectrograms.png             # Example Mel-spectrograms per species
    ├── training_curves.png          # Loss & accuracy over training
    ├── confusion_matrix_A.png       # CNN confusion matrix
    ├── comparison_per_species.png   # Baseline vs Combined per species
    ├── all_systems_comparison.png   # Bar chart — all 5 systems
    └── case_study.png               # Whinchat qualitative case study
```

---

## Pipeline

```
Xeno-canto Audio Dataset (.mp3)
        │
        ▼
  Metadata Parsing
  (Family_Genus_species_Country_Date_XCID_type.mp3)
        │
        ▼
  Mel-Spectrogram Extraction
  (128 mel bins, 5s clips, 500–12000 Hz)
        │
        ├─────────────────────┐
        ▼                     ▼
  Task A: CNN           Task B: BirdNET
  (train from scratch)  (pretrained SED)
        │                     │
        └──────────┬──────────┘
                   ▼
        Combined System
        (SED segments → CNN → vote)
                   │
                   ▼
        Evaluation (Accuracy, PSNR, SSIM,
        Confusion Matrix, Case Study)
```

---

## Model Architecture

### BirdCNN (Task A)
A 4-block convolutional network for Mel-spectrogram classification:

```
Input: (batch, 1, 128, T)  — single-channel Mel-spectrogram

Block 1: Conv2d(1→32,  3×3) → BatchNorm → ReLU → MaxPool(2)
Block 2: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2)
Block 3: Conv2d(64→128,3×3) → BatchNorm → ReLU → MaxPool(2)
Block 4: Conv2d(128→128,3×3)→ BatchNorm → ReLU → AdaptiveAvgPool(4×4)

Classifier:
  Flatten → Linear(2048, 256) → ReLU → Dropout(0.5) → Linear(256, N_CLASSES)

Output: (batch, N_CLASSES)
```

### BirdNET (Task B)
Pretrained model via `birdnetlib` — detects bird call segments with confidence scores and species labels. Operates on raw audio without training.

### Combined System
```
Audio file
    │
    ├─ BirdNET → segments [start, end, confidence]
    │
    └─ For each segment → CNN prediction
              │
              ├─ Approach A: majority vote across segments
              └─ Approach B: confidence-weighted class sums
```

---

## Installation

> ☁️ Designed to run on **Google Colab** with a GPU runtime.

### Requirements

```bash
pip install librosa birdnetlib matplotlib scikit-learn tqdm pandas seaborn torch torchvision
```

All dependencies are installed automatically in Cell 0 of the notebook.

### Google Colab (Recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload Notebook** → select `bird_coursework.ipynb`
3. **Runtime → Change runtime type → T4 GPU**
4. Follow **Dataset Setup** below, then **Runtime → Run all**

---

## Dataset Setup

The notebook expects `.mp3` audio files from **Xeno-canto** in a zipped folder.

### File Naming Convention
```
Family_Genus_species_Country_Date_XCID_type.mp3
# e.g. Turdidae_Turdus_merula_GB_20220501_XC123456_song.mp3
```

### Steps
1. Download bird recordings from [xeno-canto.org](https://xeno-canto.org)
2. Zip all `.mp3` files into `Audio.zip`
3. Upload `Audio.zip` to **Google Drive**
4. Update the path in the notebook:

```python
ZIP_PATH = '/content/drive/MyDrive/Audio.zip'   # ← update this
```

The notebook automatically:
- Filters species with ≥ 10 recordings
- Caps at 50 recordings per species
- Keeps the top 10 species by recording count
- Splits 70% train / 15% val / 15% test (stratified)

---

## Usage

Run cells in order:

| Step | Section | Description |
|------|---------|-------------|
| 0 | Install Dependencies | Installs all packages |
| 1 | Load Dataset | Mounts Drive, extracts zip, builds metadata CSV |
| 2 | Audio Preprocessing | Generates Mel-spectrograms, visualises one per species |
| 3 | Dataset & Splits | PyTorch Dataset with time-stretch augmentation; 70/15/15 split |
| 4a | Hyperparameter Search | Compares LR candidates (1e-4, 1e-3, 3e-3) over 10 probe epochs |
| 4b | CNN Training | 30-epoch full training with Adam + Cosine LR decay |
| 4c | Task A Evaluation | Accuracy, classification report, confusion matrix |
| 5 | BirdNET SED | Runs BirdNET on test files; strict and matched-only accuracy |
| 6 | Combined System | SED-guided majority vote (A) and confidence-weighted (B) |
| 7 | Quantitative Comparison | Summary table + bar chart across all 5 systems |
| 8 | Qualitative Case Study | Whinchat waveform + BirdNET detections + CNN per-segment |
| 9 | Final Summary | Prints all metrics |

---

## Outputs

| File | Description |
|------|-------------|
| `spectrograms.png` | Example Mel-spectrogram per species |
| `training_curves.png` | Train/val loss and validation accuracy curves |
| `confusion_matrix_A.png` | CNN confusion matrix on test set |
| `comparison_per_species.png` | Per-species accuracy: CNN vs Combined system |
| `all_systems_comparison.png` | Bar chart comparing all 5 systems |
| `case_study.png` | Whinchat waveform with BirdNET overlays + spectrogram |

---

## Results

| System | Accuracy |
|--------|----------|
| Task A — CNN (full clips) | — |
| Task B — BirdNET (strict) | — |
| Task B — BirdNET (matched only) | — |
| Combined A — SED-guided majority vote | — |
| Combined B — Confidence-weighted vote | — |

> Run the notebook with your dataset to populate these values.

---

## Key Parameters

```python
# Audio
SR              = 22050    # Sample rate (Hz)
DURATION        = 5.0      # Clip length (seconds)

# Mel-spectrogram
N_MELS          = 128      # Mel filterbank bins
N_FFT           = 1024     # FFT window size
HOP_LEN         = 512      # Hop length
F_MIN           = 500      # Min frequency (Hz)
F_MAX           = 12000    # Max frequency (Hz)

# Dataset
MAX_PER_SPECIES = 50       # Cap per species
SEED            = 42       # Reproducibility seed

# Training
EPOCHS          = 30
LR              = 1e-3     # Selected via hyperparameter search
WEIGHT_DECAY    = 1e-4
BATCH_SIZE      = 16
```

---

## References

- Kahl et al. (2021). *BirdNET: A deep learning solution for avian diversity monitoring.* Ecological Informatics, 61, 101236.
- McFee et al. (2015). *librosa: Audio and music signal analysis in Python.* Proc. 14th Python in Science Conference.
- Simonyan & Zisserman (2015). *Very deep convolutional networks for large-scale image recognition.* ICLR 2015.
- Stowell & Plumbley (2014). *Automatic large-scale classification of bird sounds is strongly improved by unsupervised feature learning.* PeerJ, 2.

---

## Author

**Vandhana Natarajan Chitra**  
ECS7013P — Deep Learning  
Queen Mary University of London
