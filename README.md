# SSEMG-Net

Official implementation of **SSEMG-Net: A Spectrogram-Based Mamba Network for Surface Electromyography Denoising**, together with a cross-architecture knowledge distillation (KD) pipeline that compresses SSEMG-Net into a lightweight, CPU-deployable student model.

SSEMG-Net (Teacher) removes ECG artifacts from surface electromyography signals using:

- compressed magnitude and wrapped-phase spectrograms
- time-frequency bidirectional Mamba blocks
- a physiologically guided sub-band magnitude mask
- explicit phase estimation
- time-domain, complex-domain, consistency, and multi-resolution STFT losses

The Student model (`StudentSSEMGNet`) replaces the TF-Bi-Mamba backbone with depthwise separable, dilated convolutions, removing any dependency on `mamba_ssm` / CUDA / Triton so it can run on CPU-only devices, and is trained via knowledge distillation from the Teacher.

## Main files

- `prepare_data.py`: NinaPro DB2 and ECG preprocessing, segmentation, and noisy-mixture generation
- `convert.py`: waveform-to-spectrogram conversion
- `make_dataset_spectrogram.py`: noisy-clean spectrogram pairing and tensor dataset construction
- `MECG-E/models/SSEMGNet.py`: SSEMG-Net (Teacher) model architecture
- `MECG-E/models/StudentNet.py`: StudentSSEMGNet (Student) model architecture
- `pipeline_spectrogram.py`: Teacher training pipeline
- `pipeline_distill_crossarch.py`: cross-architecture knowledge distillation pipeline (trains the Student)
- `distill_loss.py`: KD loss functions, annealing and curriculum schedulers
- `check_loss_balance.py`: sanity-checks whether KD loss terms are balanced
- `inference_demo.py`: Teacher inference, evaluation, and visualization
- `inference_student.py`: Student inference and evaluation
- `inference_student_patched.py`: Student inference with optional per-SNR grouped evaluation (`--snr_labels`); use this instead of `inference_student.py` when reproducing the SNR-conditioned results reported in the paper
- `spectrogram_utils.py`: STFT and inverse STFT utilities
- `utils.py`: evaluation metrics
- `config/local_cfg.example.yaml`: example dataset-path configuration
- `config/config_spectrogram_v19_tt_mask.yaml`: SSEMG-Net (Teacher) paper configuration
- `config/config_student_crossarch.yaml`: Student model configuration (32ch, 2 TSConvBlocks)
- `config/student_16ch_1blk.yaml`: edge-deployment Student configuration (16ch, 1 TSConvBlock — used for the reported result)
- `config/local_cfg.example.yaml`: example dataset-path configuration
- `baseline_model/`: baseline model training and inference scripts (FCN, MSEMG, SDEMG) used for comparison against SSEMG-Net

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

## 6. Knowledge distillation (train the Student)

The Student (`StudentSSEMGNet`) is trained with a Teacher SSEMG-Net checkpoint frozen (`eval()`, no gradient updates), using the same train/valid/test tensor datasets built in step 3.

The command below reproduces the reported edge-deployment result (16ch/1blk, Response KD only, SNRimp ≈ 20.05 dB):

    python pipeline_distill_crossarch.py \
        --teacher_config config/config_spectrogram_v19_tt_mask.yaml \
        --teacher_weights <path/to/teacher_checkpoint.pth> \
        --student_config config/student_16ch_1blk.yaml \
        --data_root dataset \
        --epochs 150 \
        --kd_weight_mode annealed --anneal_schedule linear \
        --w_resp_mask 3.5 --w_resp_mag 5.0 --w_resp_pha 0.3 --w_resp_com 2.3 \
        --w_feature 0 \
        --patience 10 \
        --log_csv log_student_16ch_1blk_respKD.csv \
        --model_save model_weight/student_16ch_1blk_respKD.pth

Training stops via early stopping well before reaching `--epochs`; `--epochs` only sets the upper bound and the length of the α annealing schedule (see `--schedule_epochs` below to decouple the two).

Key options:

- `--kd_weight_mode {fixed,annealed}` and `--anneal_schedule {linear,cosine,exponential}`: control how the GT/KD weight α is scheduled over training
- `--schedule_epochs`: decouple the α / noise / layer-curriculum schedule length from the actual training epoch budget (`--epochs`). Defaults to `--epochs` if unset (fully backward compatible) — set this when you want to extend training beyond a schedule that was already validated, without changing its pace
- `--w_feature`, `--w_response`: weight of Feature-based and Response-based KD losses
- `--w_resp_mask`, `--w_resp_mag`, `--w_resp_pha`, `--w_resp_com`: per-component weights of the Response KD loss (Mask, Magnitude, Phase, Complex)
- `--include_relation` / `--w_relation`: enable Relation-based KD loss (self-similarity matrix across time/frequency)
- `--similarity_preserving` / `--w_similarity` / `--similarity_student_idx` / `--similarity_teacher_idx`: enable Similarity-Preserving KD (batch-wise pairwise-similarity matching); unlike Feature/Relation KD it is not limited by student/teacher block-count alignment, so it also works with 1-block students
- `--feature_noise` / `--noise_std_start` / `--noise_std_end` / `--noise_schedule`: Feature Noise Annealing on the Teacher's target features
- `--layer_curriculum`: progressively expand the set of Teacher layers used for Feature KD
- `--two_stage` / `--two_stage_ratio`: hard-switch training into a Teacher-only representation stage followed by a Ground-Truth-only fine-tuning stage
- `--seed`: fix random/numpy/torch seeds for closer run-to-run reproducibility
- `--resume` / `--resume_path`: resume training from a full training-state checkpoint (model + optimizer + scheduler + epoch + best_val + no_improve + RNG state), kept separate from the `--model_save` inference checkpoint
- `--use_amp` / `--num_workers` / `--no_cudnn_benchmark`: pure speed options that do not change training results
- `--swa` / `--swa_dir`: collect per-epoch checkpoints after the schedule ends, for Stochastic Weight Averaging

(Optional) check that the logged loss terms are on a comparable scale:

    python check_loss_balance.py <path/to/log.csv> \
        --w_mask 3.5 --w_mag 5.0 --w_pha 0.3 --w_com 2.3 --w_feature 0.3
        
## 7. Evaluate the Student

Run:

    python inference_student.py \
        --config config/config_student_crossarch.yaml \
        --weights <path/to/student_checkpoint.pth> \
        --dataset dataset/test_spectrogram.pt \
        --csv-out <path/to/results.csv> \
        --exp-alias "Student_KD"
        
Run:

python inference_student_patched.py \
    --config config/student_16ch_1blk.yaml \
    --weights <path/to/student_checkpoint.pth> \
    --dataset dataset/test_spectrogram.pt \
    --csv-out <path/to/results.csv> \
    --exp-alias "Student_KD"

To reproduce the per-SNR breakdown reported in the paper, additionally pass --snr_labels <path/to/test_snr_labels.json> (built with build_snr_labels.py, not yet committed to this repo — regenerate it from the test split definition in step 3, or add the script here for full reproducibility). This writes an extra *_by_snr.csv file alongside the main results CSV.

`inference_demo.py` also works with a Student checkpoint and config for single-sample visualization, the same as step 5.

## Repository workflow

The complete preprocessing, training, and distillation pipeline is:

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
    SSEMG-Net (Teacher) checkpoint
        ↓
    pipeline_distill_crossarch.py  (Teacher frozen, KD training)
        ↓
    StudentSSEMGNet (Student) checkpoint
        ↓
    inference_demo.py / inference_student.py
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
CUDA-accelerated Mamba implementation used by the **Teacher**.

The **Student** (`StudentSSEMGNet`) uses only standard PyTorch convolutions and does
not import `mamba_ssm` / `causal-conv1d` / `triton`, so once a Student checkpoint is
trained, `inference_student.py` can run on a CPU-only machine.
