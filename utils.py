# /data/member1/user_howardshih/shihsemg/utils.py
import os
import math
import torch
import numpy as np
import librosa
from scipy import signal   # 用 signal.windows.hamming 與對照程式一致

def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)

def get_filepaths(directory, ftype='.npy'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
    return sorted(file_paths)

def creat_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# helpers
def exists(x): return x is not None
def default(val, d): return val if exists(val) else (d() if callable(d) else d)
def identity(t, *args, **kwargs): return t
def cycle(dl):
    while True:
        for data in dl:
            yield data
def has_int_squareroot(num): return (math.sqrt(num) ** 2) == num
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def resample(x, fs, fs_2):
    return signal.resample(x, int(x.shape[0] / fs * fs_2))

def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)

# ---------------- Features ----------------
def extract_arv_feature(x, segment_size=1000):
    L = len(x)
    if L == 0: return np.array([0.0])
    if L < segment_size:
        return np.array([np.mean(np.abs(x))])
    n = L // segment_size
    return np.array([ np.mean(np.abs(x[i*segment_size:(i+1)*segment_size])) for i in range(n) ])

def _stft_params(cfg):
    m = cfg['metrics']
    fs      = int(m['sampling_rate'])
    n_fft   = int(m['n_fft'])
    hop_len = int(m['hop_size'])
    win_len = int(m['win_size'])
    return fs, n_fft, hop_len, win_len

def _freq_axis(cfg):
    fs, n_fft, _, _ = _stft_params(cfg)
    return librosa.fft_frequencies(sr=fs, n_fft=n_fft)

def _stft_mag(x, cfg):
    """與對照程式一致：線性幅度、SciPy Hamming 窗。"""
    fs, n_fft, hop_len, win_len = _stft_params(cfg)
    D = librosa.stft(x, center=True,
                     n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                     window=signal.windows.hamming(win_len))
    return np.abs(D)

def extract_meanF_feature(x, segment_size, cfg, stimulus=None, return_series=False):
    """
    Mean Frequency（頻譜質心，|X| 權重），與你貼的 cal_MF 一致：
      - sr=1000, n_fft=256, hop=32, win=128（參考 cfg['metrics']）
      - window = scipy.signal.windows.hamming
      - 只取 >=10 Hz
      - 若傳入 stimulus，會用 stimulus[::hop] > 0 篩 frame（與 cal_MF 相同）
    回傳：
      - return_series=False：每段的 frame MF 先平均，再回傳「每段平均 MF」的向量
      - return_series=True ：回傳每段的 frame MF（拼接）
    """
    freqs = _freq_axis(cfg)
    start = np.searchsorted(freqs, 10.0)  # 10 Hz 起
    L = len(x)
    if L == 0:
        return np.array([0.0])

    def frame_mf(seg):
        S = _stft_mag(seg, cfg)[start:, :]   # [F', T]
        f = freqs[start:, None]              # [F', 1]
        num = (f * S).sum(axis=0)            # [T]
        den = S.sum(axis=0) + 1e-12
        mf = num / den                       # Hz, [T]

        # stimulus gating（可選）
        if stimulus is not None:
            fs, _, hop_len, _ = _stft_params(cfg)
            gate = (np.asarray(stimulus)[::hop_len] > 0)
            gate = gate[:mf.shape[0]]
            if gate.any():
                mf = mf[gate]
            # 若全為 False → 保留原 mf（避免空）
        return mf

    if L < segment_size:
        mf_frames = frame_mf(x)
        return mf_frames if return_series else np.array([float(mf_frames.mean())])

    n = L // segment_size
    seg_vals = []
    for i in range(n):
        seg = x[i*segment_size:(i+1)*segment_size]
        mf_frames = frame_mf(seg)
        seg_vals.append(mf_frames if return_series else float(mf_frames.mean()))
    return np.concatenate(seg_vals) if return_series else np.array(seg_vals)

def extract_medianF_feature(x, segment_size, cfg):
    """
    Median Frequency（50% 累積功率頻率，|X|^2 權重），同窗函數/參數。
    """
    freqs = _freq_axis(cfg)
    start = np.searchsorted(freqs, 10.0)
    f_sel = freqs[start:]
    L = len(x)
    if L == 0:
        return np.array([0.0])

    def seg_medianf(seg):
        S = _stft_mag(seg, cfg)
        P = (S ** 2)[start:, :]              # [F', T]
        csum  = np.cumsum(P, axis=0)
        total = csum[-1, :] + 1e-12
        idx   = (csum >= (0.5 * total)).argmax(axis=0)  # 第一次 >= 50%
        return f_sel[idx].mean()

    if L < segment_size:
        return np.array([seg_medianf(x)])
    n = L // segment_size
    return np.array([ seg_medianf(x[i*segment_size:(i+1)*segment_size]) for i in range(n) ])

# ---------------- Metrics（batch/tensor 友好） ----------------
def compute_metrics_tensor(clean, denoised, noisy=None, cfg=None, segment_size=None):
    import torch
    def to_np(z): return z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)

    clean    = to_np(clean)
    denoised = to_np(denoised)
    noisy    = to_np(noisy) if noisy is not None else None

    if clean.ndim == 1:
        clean    = clean[None, :]
        denoised = denoised[None, :]
        if noisy is not None: noisy = noisy[None, :]

    m        = cfg['metrics']
    fs       = int(m['sampling_rate'])
    seg_sec  = float(m['segment_sec'])
    seg_samp = int(fs * seg_sec)
    segment_size = min(seg_samp, clean.shape[-1]) if segment_size is None else int(segment_size)

    total_signal_pow = 0.0
    total_noise_pow  = 0.0
    total_num_pts    = 0
    snrimp_list, rmsem_list = [], []
    rmse_arv_list, rmse_meanf_list, rmse_medianf_list = [], [], []
    eps = 1e-8

    for i in range(len(clean)):
        x, x_hat = clean[i], denoised[i]
        noise = x - x_hat

        local_snr = 10 * np.log10((np.sum(x**2) + eps) / (np.sum(noise**2) + eps))
        total_signal_pow += np.sum(x**2)
        total_noise_pow  += np.sum(noise**2)
        total_num_pts    += x.size

        rmsem_list.append(np.sqrt(np.mean((noise - noise.mean())**2)))

        if noisy is not None:
            nn = x - noisy[i]
            local_nn_snr = 10 * np.log10((np.sum(x**2) + eps) / (np.sum(nn**2) + eps))
            snrimp_list.append(local_snr - local_nn_snr)

        # ARV
        arv_c = extract_arv_feature(x,     segment_size)
        arv_h = extract_arv_feature(x_hat, segment_size)
        rmse_arv_list.append(np.sqrt(np.mean((arv_c - arv_h)**2)))

        # MeanF（= 舊 MF；centroid, |X| 權重）
        meanf_c = extract_meanF_feature(x,     segment_size, cfg)      # stimulus=None
        meanf_h = extract_meanF_feature(x_hat, segment_size, cfg)
        rmse_meanf_list.append(np.sqrt(np.mean((meanf_c - meanf_h)**2)))

        # MedianF（50% 功率頻率）
        medianf_c = extract_medianF_feature(x,     segment_size, cfg)
        medianf_h = extract_medianF_feature(x_hat, segment_size, cfg)
        rmse_medianf_list.append(np.sqrt(np.mean((medianf_c - medianf_h)**2)))

    snr_global = 10 * np.log10((total_signal_pow + eps) / (total_noise_pow + eps))
    rmse_global = np.sqrt(total_noise_pow / max(1, total_num_pts))

    out = {
        "SNR":   float(snr_global),
        "RMSE":  float(rmse_global),
        "RMSEM": float(np.mean(rmsem_list)),
    }
    if rmse_arv_list:
        out["RMSE_ARV"] = float(np.mean(rmse_arv_list))

    # ★ 維持相容：RMSE_MF(Hz) = MeanF
    if rmse_meanf_list:
        out["RMSE_MeanF(Hz)"] = float(np.mean(rmse_meanf_list))
        out["RMSE_MF(Hz)"]    = out["RMSE_MeanF(Hz)"]
    if rmse_medianf_list:
        out["RMSE_MedianF(Hz)"] = float(np.mean(rmse_medianf_list))

    if snrimp_list:
        sim = float(np.mean(snrimp_list))
        out["SNR_IMP(dB)"] = sim
        out["SNRimp"]      = sim

    return out