"""
inference_hp_ts_baseline_rawwav.py  (v3 — 用真正的 restimulus 標籤)
HP / TS baseline，直接吃 prepare_data.py 產生的原始波形 .npy，完全跳過
spectrogram -> ISTFT 重建這條路。

v3 相較 v2 的改動：
  RMSE_MF 的「官方風格」版本，原本用 energy_threshold_mask() 拿 clean 訊號的
  RMS 包絡去近似官方的 stimulus 遮罩 —— 這只是近似。這一版改成直接讀
  prepare_data.py 產生 test set 時就已經存好的 "<basename>_sti.npy" 檔案，
  這是原始 NinaPro DB2 的 restimulus 通道，跟 emg 逐 sample 對齊、經過完全
  相同的 2000Hz->1000Hz 降採樣（都是 [::2]），是官方 cal_MF() 真正用的那個
  stimulus，不是近似值。

  對應規則（跟 prepare_data.py 的存檔慣例一致）：
    clean:  clean/<basename>.npy
    sti:    clean/<basename>_sti.npy      （只有 test set 才有存這個）
    noisy:  noisy/<snr>/<noise_name>/<basename>.npy
  三者 <basename> 完全相同，只有資料夾/檔名後綴不同。
"""
import os, sys, yaml, numpy as np
import csv, math, re
from glob import glob
from tqdm import tqdm
import argparse
from collections import defaultdict
from scipy import signal
import scipy.optimize as spo
import librosa

# ========= HP / TS algorithms（逐字照抄官方 repo）=========
def apply_hp(noisy, fs=1000, fc=40):
    b, a = signal.butter(4, fc, 'highpass', fs=fs)
    return signal.filtfilt(b, a, noisy).astype('float64')

def filtered_template_subtraction(n_emg, fc=50):
    """對應官方 evaluate_FTSHP 實際呼叫的版本：樣板直接用個別 peak-aligned
    波形本身，不做鄰近波峰平均、不做振幅 scalar 擬合。"""
    error = 0
    fs = 1000
    pad = 1500
    b, a = signal.butter(4, [2.5, fc], 'bp', fs=fs)
    clean_signal = np.pad(signal.filtfilt(b, a, n_emg), (pad, pad))

    signal_rec = abs(clean_signal)
    movingavg_1 = np.ones(fs * 1)
    movingavg_2 = np.ones(int(fs * 0.1))
    signal_1 = np.convolve(signal_rec, movingavg_1, 'same') / 1000
    signal_2 = np.convolve(signal_rec, movingavg_2, 'same') / 100

    r_peaks = []
    j, mark = 0, 0
    for i in range(clean_signal.shape[0]):
        if i < mark:
            continue
        if signal_1[i] < signal_2[i]:
            for j in range(i, clean_signal.shape[0]):
                if signal_1[j] > signal_2[j]:
                    mark = j
                    if j - i < 140:
                        break
                    peak_idx = i + np.where(clean_signal[i:j] == np.amax(clean_signal[i:j]))[0][0]
                    r_peaks.append(peak_idx)
                    break

    if not r_peaks:
        return n_emg, 1
    if r_peaks[0] < pad:
        r_peaks.pop(0)
    waveform = []
    number = len(r_peaks)
    if number < 11:
        return n_emg, 1

    trr = min([j - i for i, j in zip(r_peaks[:-1], r_peaks[1:])])
    left = math.floor(0.25 * trr)
    right = math.floor(0.45 * trr)

    for i in range(number - 1):
        waveform.append(clean_signal[r_peaks[i] - left:r_peaks[i] + right + 1])
    waveform.append(clean_signal[r_peaks[-1] - left:r_peaks[-1] + right + 1])

    try:
        _ = sum(waveform[:11]) / 11
        _ = sum(waveform[-11:]) / 11
    except Exception:
        return n_emg, 2

    template = waveform  # 官方原文註解：No average

    clean_signal = clean_signal[pad:-pad]
    r_peaks = [p - pad for p in r_peaks]

    all_template = template[0]
    for i in range(1, number):
        all_template = np.concatenate((all_template, np.zeros(r_peaks[i] - r_peaks[i - 1] - left - right - 1), template[i]))

    l_pad = 0 if r_peaks[0] - left < 0 else r_peaks[0] - left
    if clean_signal.shape[0] - all_template.shape[0] - r_peaks[0] + left + 1 < 0:
        all_template = np.pad(all_template, (l_pad, 0))
    else:
        all_template = np.pad(all_template, (l_pad, clean_signal.shape[0] - all_template.shape[0] - r_peaks[0] + left + 1))
    if l_pad == 0:
        all_template = all_template[-(r_peaks[0] - left):]

    enh_EMG = n_emg - all_template[:n_emg.shape[0]]
    return enh_EMG, error

def apply_ts(noisy, fs=1000):
    enhanced, error = filtered_template_subtraction(noisy, 50)
    highpass = signal.butter(4, 40, 'highpass', fs=fs)
    enhanced = signal.filtfilt(highpass[0], highpass[1], enhanced)
    return enhanced.astype('float64'), error

def denoise(noisy_np, method):
    if method == 'HP':
        return apply_hp(noisy_np), 0
    else:
        return apply_ts(noisy_np)

# ========= 官方 cal_MF：逐字對照官方 util.py =========
def official_style_mf(emg, cfg, active_mask_full=None):
    """逐字對照 FCN 官方 util.py::cal_MF。
    active_mask_full: 完整長度（跟 emg 同長）的陣列，代表每個 sample 是否屬於
    restimulus>0 的區間；函式內部用 [::hop_size] 降採樣到跟 STFT frame 對齊，
    完全比照官方 `stimulus[::32]` 的寫法。"""
    m = cfg['metrics']; fs = m['sampling_rate']; n_fft = m['n_fft']
    hop_len = m['hop_size']; win_len = m['win_size']
    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    start = np.searchsorted(freqs, 10.0)
    freqs = freqs[start:]
    D = librosa.stft(emg, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                      window='hamming', center=True)
    spec = np.abs(D)[start:, :]
    weighted_f = (freqs[:, None] * spec).sum(axis=0)
    spec_pow = spec.sum(axis=0) + 1e-12
    mf = weighted_f / spec_pow
    if active_mask_full is not None:
        frame_mask = active_mask_full[::hop_len] > 0
        n = min(len(mf), len(frame_mask))
        mf = mf[:n][frame_mask[:n]]
    return mf


def rmse_mf_official_style(clean_emg, denoised_emg, cfg, restimulus_full):
    """RMSE_MF = RMSE(cal_MF(clean, restimulus), cal_MF(enhanced, restimulus))，
    restimulus_full 是真正從 prepare_data.py 存下來的 _sti.npy 讀出來的標籤，
    不是近似值。"""
    mf_clean = official_style_mf(clean_emg, cfg, active_mask_full=restimulus_full)
    mf_denoised = official_style_mf(denoised_emg, cfg, active_mask_full=restimulus_full)
    n = min(len(mf_clean), len(mf_denoised))
    if n == 0:
        return float('nan'), 0
    return float(np.sqrt(np.mean((mf_clean[:n] - mf_denoised[:n]) ** 2))), n

# ========= inline metrics（SNRimp/RMSE/PRD/RMSE_ARV/RMSEM）=========
def _inline_extract_arv_feature(signal_, segment_size=1000):
    L = len(signal_)
    if L == 0: return np.array([0.0])
    if L < segment_size: return np.array([np.mean(np.abs(signal_))])
    n = L // segment_size
    return np.array([np.mean(np.abs(signal_[i*segment_size:(i+1)*segment_size])) for i in range(n)])

def _inline_extract_medianF_feature(signal_, segment_size, cfg):
    m = cfg['metrics']; fs=m['sampling_rate']; n_fft=m['n_fft']; hop_len=m['hop_size']; win_len=m['win_size']
    L = len(signal_)
    if L == 0: return np.array([0.0])
    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    start = np.searchsorted(freqs, 10.0); f_sel = freqs[start:]
    def median_f_of(seg):
        D = librosa.stft(seg, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window='hamming', center=True)
        P = (np.abs(D)**2)[start:,:]
        csum = np.cumsum(P, axis=0); total = csum[-1,:]+1e-12; half = 0.5*total
        idx = (csum >= half).argmax(axis=0)
        return f_sel[idx].mean()
    if L < segment_size: return np.array([median_f_of(signal_)])
    n = L // segment_size
    return np.array([median_f_of(signal_[i*segment_size:(i+1)*segment_size]) for i in range(n)])

def _inline_compute_metrics_tensor(clean, denoised_, noisy=None, cfg=None, segment_size=None):
    clean=np.asarray(clean); denoised_=np.asarray(denoised_); noisy=np.asarray(noisy) if noisy is not None else None
    if clean.ndim==1:
        clean=clean[None,:]; denoised_=denoised_[None,:]
        if noisy is not None: noisy=noisy[None,:]
    m=cfg['metrics']; fs=m['sampling_rate']; seg_samp=int(fs*m['segment_sec'])
    segment_size = min(seg_samp, clean.shape[-1]) if segment_size is None else segment_size
    total_signal_pow=0.0; total_noise_pow=0.0; total_num_pts=0
    snrimp_list=[]; rmsem_list=[]; rmse_arv_list=[]; rmse_medianf_list=[]; prd_list=[]; eps=1e-12
    for i in range(len(clean)):
        x,x_hat=clean[i],denoised_[i]; noise=x-x_hat
        local_snr=10*np.log10((np.sum(x**2)+eps)/(np.sum(noise**2)+eps))
        total_signal_pow+=np.sum(x**2); total_noise_pow+=np.sum(noise**2); total_num_pts+=x.size
        rmsem_list.append(np.sqrt(np.mean((noise-noise.mean())**2)))
        if noisy is not None:
            nn=x-noisy[i]; local_nn_snr=10*np.log10((np.sum(x**2)+eps)/(np.sum(nn**2)+eps))
            snrimp_list.append(local_snr-local_nn_snr)
        prd=np.sqrt((np.sum((x_hat-x)**2)+eps)/(np.sum(x**2)+eps))*100.0; prd_list.append(prd)
        arv_c=_inline_extract_arv_feature(x,segment_size); arv_h=_inline_extract_arv_feature(x_hat,segment_size)
        rmse_arv_list.append(np.sqrt(np.mean((arv_c-arv_h)**2)))
        medf_c=_inline_extract_medianF_feature(x,segment_size,cfg); medf_h=_inline_extract_medianF_feature(x_hat,segment_size,cfg)
        rmse_medianf_list.append(np.sqrt(np.mean((medf_c-medf_h)**2)))
    snr_global=10*np.log10((total_signal_pow+eps)/(total_noise_pow+eps))
    rmse_global=np.sqrt(total_noise_pow/max(1,total_num_pts))
    result={"SNR":float(snr_global),"RMSE":float(rmse_global),"RMSEM":float(np.mean(rmsem_list))}
    if prd_list: result["PRD(%)"]=float(np.mean(prd_list))
    if rmse_arv_list: result["RMSE_ARV"]=float(np.mean(rmse_arv_list))
    if rmse_medianf_list: result["RMSE_MedianF(Hz)"]=float(np.mean(rmse_medianf_list))
    if snrimp_list:
        sim=float(np.mean(snrimp_list)); result["SNR_IMP(dB)"]=sim; result["SNRimp"]=sim
    return result

# ============================================================
def find_paired_paths(noisy_path, noisy_root, clean_root):
    """noisy 檔案路徑 -> (clean 路徑, sti 路徑)。basename 完全相同，只有
    資料夾/後綴不同（比照 prepare_data.py 的存檔慣例）。"""
    rel = os.path.relpath(noisy_path, noisy_root)
    fname = os.path.basename(rel)
    stem = fname[:-4] if fname.endswith('.npy') else fname
    clean_path = os.path.join(clean_root, fname)
    sti_path = os.path.join(clean_root, stem + '_sti.npy')
    if not os.path.exists(clean_path):
        return None, None
    if not os.path.exists(sti_path):
        sti_path = None
    return clean_path, sti_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--noisy_dir', default='noisy')
    p.add_argument('--clean_dir', default='clean')
    p.add_argument('--method', required=True, choices=['HP', 'TS'])
    p.add_argument('--csv-out', default='')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if 'metrics' not in cfg:
        cfg['metrics'] = {'sampling_rate': 1000, 'n_fft': 256, 'hop_size': 32, 'win_size': 128, 'segment_sec': 1}

    split_dir = os.path.join(args.data_root, args.split)
    noisy_root = os.path.join(split_dir, args.noisy_dir)
    clean_root = os.path.join(split_dir, args.clean_dir)

    noisy_files = sorted(glob(os.path.join(noisy_root, "**", "*.npy"), recursive=True))
    if not noisy_files:
        raise FileNotFoundError(f"在 {noisy_root} 底下找不到任何 .npy 檔案，路徑對嗎？")
    print(f"✔ 找到 {len(noisy_files)} 個 noisy 檔案（原始波形，未經 spectrogram/ISTFT）")

    csv_path = args.csv_out or f"{args.method.lower()}_metrics_rawwav_v3.csv"
    if os.path.dirname(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_header = ["path", "SNR", "SNR_in", "SNR_out", "SNRimp", "RMSE", "PRD(%)",
                  "RMSE_ARV", "RMSE_MF_official(Hz)", "n_active_frames",
                  "RMSE_MedianF(Hz)", "RMSEM", "error_flag", "has_sti"]
    f_csv = open(csv_path, 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(csv_header)

    sum_metrics = {"SNRimp":0.,"RMSE":0.,"PRD(%)":0.,"RMSE_ARV":0.,
                   "RMSE_MF_official(Hz)":0.,"RMSE_MedianF(Hz)":0.,"RMSEM":0.}
    count_samples = 0
    n_error = 0
    n_skipped = 0
    n_no_sti = 0
    n_zero_active = 0
    n_mf_valid = 0

    snr_sum_metrics = defaultdict(lambda: {"SNRimp":0.,"RMSE":0.,"PRD(%)":0.,"RMSE_ARV":0.,
                                            "RMSE_MF_official(Hz)":0.,"RMSE_MedianF(Hz)":0.,"RMSEM":0.})
    snr_count = defaultdict(int)
    snr_mf_count = defaultdict(int)

    try:
        for noisy_path in tqdm(noisy_files, desc=f"[{args.method}] Inference (raw waveform, real restimulus)"):
            clean_path, sti_path = find_paired_paths(noisy_path, noisy_root, clean_root)
            if clean_path is None:
                n_skipped += 1
                continue

            rel = os.path.relpath(noisy_path, noisy_root)
            snr_str = rel.split(os.sep)[0]
            if not re.match(r"^-?\d+$", snr_str):
                n_skipped += 1
                continue
            snr = int(snr_str)

            noisy_i = np.load(noisy_path).astype('float64').squeeze()
            clean_i = np.load(clean_path).astype('float64').squeeze()
            has_sti = sti_path is not None
            sti_i = np.load(sti_path).astype('float64').squeeze() if has_sti else None
            if not has_sti:
                n_no_sti += 1

            enhanced, error = denoise(noisy_i, args.method)
            if error != 0:
                n_error += 1

            minL = min(len(clean_i), len(enhanced), len(noisy_i))
            if sti_i is not None:
                minL = min(minL, len(sti_i))
            clean_i2, enh_i2, noisy_i2 = clean_i[:minL], enhanced[:minL], noisy_i[:minL]
            sti_i2 = sti_i[:minL] if sti_i is not None else None

            m_out = _inline_compute_metrics_tensor(clean_i2, enh_i2, noisy_i2, cfg=cfg)
            m_in = _inline_compute_metrics_tensor(clean_i2, noisy_i2, None, cfg=cfg)

            rmse_mf_valid = False
            if sti_i2 is not None:
                rmse_mf_official, n_active = rmse_mf_official_style(clean_i2, enh_i2, cfg, sti_i2)
                if n_active == 0:
                    n_zero_active += 1
                elif not np.isnan(rmse_mf_official):
                    rmse_mf_valid = True
            else:
                rmse_mf_official, n_active = float('nan'), 0

            for k in sum_metrics:
                if k == "RMSE_MF_official(Hz)":
                    if rmse_mf_valid:
                        sum_metrics[k] += rmse_mf_official
                else:
                    sum_metrics[k] += float(m_out.get(k, 0.))
            count_samples += 1
            if rmse_mf_valid:
                n_mf_valid += 1

            sm = snr_sum_metrics[snr]
            for k in sum_metrics:
                if k == "RMSE_MF_official(Hz)":
                    if rmse_mf_valid:
                        sm[k] += rmse_mf_official
                else:
                    sm[k] += float(m_out.get(k, 0.))
            snr_count[snr] += 1
            if rmse_mf_valid:
                snr_mf_count[snr] += 1

            writer.writerow([
                noisy_path, snr,
                float(m_in["SNR"]), float(m_out["SNR"]),
                float(m_out.get("SNRimp", 0.)), float(m_out["RMSE"]),
                float(m_out.get("PRD(%)", float('nan'))),
                float(m_out.get("RMSE_ARV", 0.)),
                rmse_mf_official, n_active,
                float(m_out.get("RMSE_MedianF(Hz)", float('nan'))),
                float(m_out["RMSEM"]), error, has_sti,
            ])
    finally:
        f_csv.close()

    if n_no_sti:
        print(f"⚠️ {n_no_sti}/{len(noisy_files)} 筆找不到對應的 _sti.npy，"
              f"這些筆的 RMSE_MF_official 被排除在平均之外（不是用近似值頂替）")
    if n_zero_active:
        print(f"⚠️ {n_zero_active} 筆的 restimulus 整段都是 0（沒有偵測到動作），"
              f"這些筆的 RMSE_MF 也被排除在平均之外")
    if n_skipped:
        print(f"⚠️ 跳過 {n_skipped} 筆（找不到對應 clean 檔案或路徑無法解析 SNR）")

    avg = {}
    for k, v in sum_metrics.items():
        denom = n_mf_valid if k == "RMSE_MF_official(Hz)" else count_samples
        avg[k] = v / max(1, denom)
    print(f"\n===== Test-set Average ({args.method}, raw waveform, real restimulus) =====")
    for k, v in avg.items():
        print(f"  {k:25s}: {v:.4f}")
    print(f"  (RMSE_MF_official 是用 {n_mf_valid}/{count_samples} 筆有效樣本算出來的)")
    if args.method == 'TS' and n_error > 0:
        print(f"  ⚠ {n_error}/{count_samples} 筆因為 R-peak 偵測數量不足（<11 個），退回原始 noisy 波形")

    print(f"\n===== Per-SNR Average ({args.method}, raw waveform, real restimulus) =====")
    snr_avg_rows = []
    for snr in sorted(snr_count.keys(), reverse=True):
        n_s = snr_count[snr]
        n_mf_s = snr_mf_count[snr]
        row = {"SNR": snr, "n": n_s}
        for k, v in snr_sum_metrics[snr].items():
            denom = n_mf_s if k == "RMSE_MF_official(Hz)" else n_s
            row[k] = v / max(1, denom)
        snr_avg_rows.append(row)
        print(f"  SNR={snr:>4} dB (n={n_s:4d}) | "
              f"SNRimp={row['SNRimp']:.4f}  RMSE={row['RMSE']:.4f}  "
              f"RMSE_MF(官方,真標籤,n={n_mf_s})={row['RMSE_MF_official(Hz)']:.4f}")

    snr_csv_path = os.path.splitext(csv_path)[0] + '_by_snr.csv'
    with open(snr_csv_path, 'w', newline='') as f:
        fieldnames = ["SNR", "n", "SNRimp", "RMSE", "PRD(%)", "RMSE_ARV",
                      "RMSE_MF_official(Hz)", "RMSE_MedianF(Hz)", "RMSEM"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in snr_avg_rows:
            w.writerow(row)
    print(f"\n[info] 每個 SNR 檔位的分數已存到 {snr_csv_path}")


if __name__ == '__main__':
    main()
