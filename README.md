# SSEMG-Net

Official implementation of **SSEMG-Net: A Spectrogram-Based Mamba Network for Surface Electromyography Denoising**.

SSEMG-Net removes ECG artifacts from surface electromyography signals using:

- compressed magnitude and wrapped-phase spectrograms
- time-frequency bidirectional Mamba blocks
- a physiologically guided sub-band magnitude mask
- explicit phase estimation
- time-domain, complex-domain, consistency, and multi-resolution STFT losses

## Main files

- `prepare_data.py`: NinaPro DB2 and ECG preprocessing, segmentation, and noisy-mixture generation
- `convert.py`: waveform-to-spectrogram conversion
- `make_dataset_spectrogram.py`: noisy-clean spectrogram pairing and tensor dataset construction
- `MECG-E/models/SSEMGNet.py`: SSEMG-Net model architecture
- `pipeline_spectrogram.py`: training pipeline
- `inference_demo.py`: inference, evaluation, and visualization
- `spectrogram_utils.py`: STFT and inverse STFT utilities
- `utils.py`: evaluation metrics
- `config/config_spectrogram_v19_tt_mask.yaml`: SSEMG-Net paper configuration
- `config/local_cfg.example.yaml`: example dataset-path configuration

## Data

The raw datasets are not redistributed in this repository.

Required datasets:

- NinaPro DB2 for clean sEMG signals
- MIT-BIH Normal Sinus Rhythm Database for ECG interference

Create the local path configuration:

    cp config/local_cfg.example.yaml config/local_cfg.yaml

Edit `config/local_cfg.yaml`:

    ECG_corpus_dir: /path/to/mit-bih-normal-sinus-rhythm-database
    ECG_storage_dir: /path/to/processed/ecg

    EMG_corpus_dir: /path/to/ninapro-db2
    sEMG_dataset_dir: semg_data/processed

The default pipeline assumes that `sEMG_dataset_dir` contains:

    semg_data/processed/
    ├── train/
    ├── valid/
    └── test/

## 1. Prepare ECG-contaminated sEMG waveforms

Run:

    python prepare_data.py

This script performs the following operations:

1. Reads Channel 1 ECG recordings from the MIT-BIH Normal Sinus Rhythm Database.
2. Resamples ECG from 128 Hz to 1000 Hz.
3. Applies 10 Hz high-pass and 200 Hz low-pass filtering to ECG.
4. Reads NinaPro DB2 sEMG recordings.
5. Applies a 20–500 Hz Butterworth band-pass filter.
6. Downsamples sEMG from 2000 Hz to 1000 Hz.
7. Normalizes and divides sEMG into 10-second segments.
8. Generates ECG-contaminated sEMG signals at multiple SNR levels.

The default data split is:

- Training:
  - NinaPro DB2 subjects 11–40
  - Exercise 1
  - Channel 2
  - SNR: −15, −13, −11, −9, −7, −5 dB

- Validation:
  - NinaPro DB2 subjects 1–10
  - Exercise 3
  - Channel 2
  - SNR: −15, −13, −11, −9, −7, −5 dB

- Testing:
  - NinaPro DB2 subjects 1–10
  - Exercise 2
  - Channels 9–12
  - SNR: −14, −12, −10, −8, −6, −4, −2, 0 dB

The generated waveform structure is:

    semg_data/processed/
    ├── train/
    │   ├── clean/
    │   └── noisy/
    ├── valid/
    │   ├── clean/
    │   └── noisy/
    └── test/
        ├── clean/
        └── noisy/

## 2. Convert waveforms to spectrograms

Run the conversion for each split:

    python convert.py --split train
    python convert.py --split valid
    python convert.py --split test

The script converts waveform `.npy` files into compressed magnitude and wrapped-phase spectrogram `.pt` files using:

- sampling rate: 1000 Hz
- FFT size: 512
- window size: 512
- hop size: 128
- magnitude compression exponent: 0.5

The generated directories are:

    spectrogram_cf05/
    spectrogram_cf05_clean/

## 3. Build tensor datasets

Run:

    python make_dataset_spectrogram.py \
        --data_root semg_data/processed \
        --noisy_dir spectrogram_cf05 \
        --clean_dir spectrogram_cf05_clean \
        --output_dir dataset

This produces:

    dataset/
    ├── train_spectrogram.pt
    ├── valid_spectrogram.pt
    └── test_spectrogram.pt

Each dataset contains paired tensors:

    noisy spectrogram: [N, 2, F, T]
    clean spectrogram: [N, 2, F, T]

The two channels represent:

    channel 0: compressed magnitude
    channel 1: wrapped phase

## 4. Train SSEMG-Net

Run:

    python pipeline_spectrogram.py \
        --config config/config_spectrogram_v19_tt_mask.yaml \
        --data_root dataset

The paper configuration uses:

- epochs: 30
- batch size: 4
- gradient accumulation: 4
- effective batch size: 16
- optimizer: AdamW
- learning rate: 3e-4
- dense channels: 64
- TF-Bi-Mamba blocks: 4
- sub-band mask cutoff: 150 Hz

The training objective contains:

- time-domain loss
- complex-domain loss
- STFT consistency loss
- multi-resolution STFT loss

## 5. Evaluate SSEMG-Net

Run:

    python inference_demo.py \
        --config config/config_spectrogram_v19_tt_mask.yaml \
        --weights /path/to/checkpoint.pth \
        --dataset dataset/test_spectrogram.pt \
        --batch 64 \
        --index 0

The evaluation reports:

- SNR improvement
- RMSE
- RMSE of average rectified value
- RMSE of mean frequency

In this repository, `RMSE_MF(Hz)` denotes the error of **mean frequency**, defined as the spectral centroid.

## Repository workflow

The complete preprocessing and training pipeline is:

    Raw NinaPro DB2 and ECG data
        ↓
    prepare_data.py
        ↓
    Clean and ECG-contaminated waveform files
        ↓
    convert.py
        ↓
    Magnitude-phase spectrogram files
        ↓
    make_dataset_spectrogram.py
        ↓
    Train, validation, and test tensor datasets
        ↓
    pipeline_spectrogram.py
        ↓
    SSEMG-Net checkpoint
        ↓
    inference_demo.py
        ↓
    Evaluation metrics and visualizations

## Installation

SSEMG-Net uses the official `mamba-ssm` package and does not vendor a separate
copy of the Mamba source code.

Install PyTorch first, followed by the remaining dependencies:

    pip install torch==2.3.1
    pip install -r requirements.txt --no-build-isolation

The released environment uses:

- `mamba-ssm==2.2.2`
- `causal-conv1d>=1.4.0`
- `triton==2.3.1`

Linux, an NVIDIA GPU, and a compatible CUDA toolkit are required for the
CUDA-accelerated Mamba implementation.
