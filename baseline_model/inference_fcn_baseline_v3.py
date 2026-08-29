# inference_fcn_baseline.py
import os, sys, yaml, torch, argparse, json, csv
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import librosa

from fcn_baseline_model import FCN_01
from pipeline_spectrogram import auto_select_gpu
from models.StudentNet import mag_pha_istft

# =====================================================================
# Inline metrics 計算函式
# =====================================================================
def _inline_extract_arv_feature(signal, segment_size=1000):
    L = len(signal)
    if L == 0: return np.array([0.0])
    if L < segment_size: return np.array([np.mean(np.abs(signal))])
    n = L // segment_size
    return np.array([np.mean(np.abs(signal[i*segment_size:(i+1)*segment_size])) for i in range(n)])

def _inline_extract_mf_feature(signal, segment_size, cfg):
    m = cfg['metrics']; fs=m['sampling_rate']; n_fft=m['n_fft']; hop_len=m['hop_size']; win_len=m['win_size']
    L = len(signal)
    if L == 0: return np.array([0.0])
    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    start = np.searchsorted(freqs, 10.0)
    def mf_of(seg):
        D = librosa.stft(seg, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window='hamming', center=True)
        mag = np.abs(D)[start:,:]; f = freqs[start:,None]
        mf_frames = (f*mag).sum(axis=0) / (mag.sum(axis=0)+1e-12)
        return mf_frames.mean()
    if L < segment_size: return np.array([mf_of(signal)])
    n = L // segment_size
    return np.array([mf_of(signal[i*segment_size:(i+1)*segment_size]) for i in range(n)])

def _inline_extract_medianF_feature(signal, segment_size, cfg):
    m = cfg['metrics']; fs=m['sampling_rate']; n_fft=m['n_fft']; hop_len=m['hop_size']; win_len=m['win_size']
    L = len(signal)
    if L == 0: return np.array([0.0])
    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    start = np.searchsorted(freqs, 10.0); f_sel = freqs[start:]
    def median_f_of(seg):
        D = librosa.stft(seg, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window='hamming', center=True)
        P = (np.abs(D)**2)[start:,:]
        csum = np.cumsum(P, axis=0); total = csum[-1,:]+1e-12; half = 0.5*total
        idx = (csum >= half).argmax(axis=0)
        return f_sel[idx].mean()
    if L < segment_size: return np.array([median_f_of(signal)])
    n = L // segment_size
    return np.array([median_f_of(signal[i*segment_size:(i+1)*segment_size]) for i in range(n)])

def _inline_compute_metrics_tensor(clean, denoised, noisy=None, cfg=None, segment_size=None):
    def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    clean=to_np(clean); denoised=to_np(denoised); noisy=to_np(noisy) if noisy is not None else None
    if clean.ndim==1:
        clean=clean[None,:]; denoised=denoised[None,:]
        if noisy is not None: noisy=noisy[None,:]
    
    if 'metrics' not in cfg:
        cfg['metrics'] = {'sampling_rate': 1000, 'n_fft': 256, 'hop_size': 32, 'win_size': 128, 'segment_sec': 1}
        
    m=cfg['metrics']; fs=m['sampling_rate']; seg_samp=int(fs*m['segment_sec'])
    segment_size = min(seg_samp, clean.shape[-1]) if segment_size is None else segment_size
    
    total_signal_pow=0.0; total_noise_pow=0.0; total_num_pts=0
    snrimp_list=[]; rmsem_list=[]; rmse_arv_list=[]; rmse_mf_list=[]; rmse_medianf_list=[]; prd_list=[]; eps=1e-12
    
    for i in range(len(clean)):
        x, x_hat = clean[i], denoised[i]
        noise = x - x_hat
        local_snr = 10*np.log10((np.sum(x**2)+eps)/(np.sum(noise**2)+eps))
        total_signal_pow += np.sum(x**2); total_noise_pow += np.sum(noise**2); total_num_pts += x.size
        rmsem_list.append(np.sqrt(np.mean((noise-noise.mean())**2)))
        
        if noisy is not None:
            nn = x - noisy[i]
            local_nn_snr = 10*np.log10((np.sum(x**2)+eps)/(np.sum(nn**2)+eps))
            snrimp_list.append(local_snr - local_nn_snr)
            
        prd = np.sqrt((np.sum((x_hat-x)**2)+eps)/(np.sum(x**2)+eps))*100.0
        prd_list.append(prd)
        
        arv_c = _inline_extract_arv_feature(x, segment_size)
        arv_h = _inline_extract_arv_feature(x_hat, segment_size)
        rmse_arv_list.append(np.sqrt(np.mean((arv_c-arv_h)**2)))
        
        meanf_c = _inline_extract_mf_feature(x, segment_size, cfg)
        meanf_h = _inline_extract_mf_feature(x_hat, segment_size, cfg)
        rmse_mf_list.append(np.sqrt(np.mean((meanf_c-meanf_h)**2)))
        
        medf_c = _inline_extract_medianF_feature(x, segment_size, cfg)
        medf_h = _inline_extract_medianF_feature(x_hat, segment_size, cfg)
        rmse_medianf_list.append(np.sqrt(np.mean((medf_c-medf_h)**2)))
        
    snr_global = 10*np.log10((total_signal_pow+eps)/(total_noise_pow+eps))
    rmse_global = np.sqrt(total_noise_pow/max(1,total_num_pts))
    result = {"SNR": float(snr_global), "RMSE": float(rmse_global), "RMSEM": float(np.mean(rmsem_list))}
    
    if prd_list: result["PRD(%)"] = float(np.mean(prd_list))
    if rmse_arv_list: result["RMSE_ARV"] = float(np.mean(rmse_arv_list))
    if rmse_mf_list:
        result["RMSE_MeanF(Hz)"] = float(np.mean(rmse_mf_list))
        result["RMSE_MF(Hz)"] = result["RMSE_MeanF(Hz)"]
    if rmse_medianf_list:
        result["RMSE_MedianF(Hz)"] = float(np.mean(rmse_medianf_list))
    if snrimp_list:
        sim = float(np.mean(snrimp_list))
        result["SNR_IMP(dB)"] = sim
        result["SNRimp"] = sim
    return result

def official_style_mf(emg, cfg, active_mask_full=None):
    """逐字對照 FCN 官方 util.py::cal_MF。active_mask_full 是完整長度（跟 emg
    同長）的 restimulus 陣列，函式內部用 [::hop_size] 降採樣到跟 STFT frame
    對齊，完全比照官方 `stimulus[::32]` 的寫法。"""
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
    mf_clean = official_style_mf(clean_emg, cfg, active_mask_full=restimulus_full)
    mf_denoised = official_style_mf(denoised_emg, cfg, active_mask_full=restimulus_full)
    n = min(len(mf_clean), len(mf_denoised))
    if n == 0:
        return float('nan'), 0
    return float(np.sqrt(np.mean((mf_clean[:n] - mf_denoised[:n]) ** 2))), n


def load_sti(basename, sti_root):
    """讀 <basename>_sti.npy，找不到回傳 None（呼叫端要處理成「這筆不算進
    RMSE_MF_official 的平均」，不要用近似值頂替）。"""
    path = os.path.join(sti_root, basename + '_sti.npy')
    if not os.path.exists(path):
        return None
    return np.load(path).astype('float64').squeeze()


# =====================================================================
# 主推論流程 (極致節省 RAM 避開 OOM)
# =====================================================================
def spec_to_wav(spec, n_fft, hop_size, win_size, compress_factor):
    mag = spec[:, 0]
    pha = spec[:, 1]
    return mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weight_path', default='model_weight/fcn_baseline.pth')
    p.add_argument('--student_config', default='config/config_student_crossarch.yaml')
    p.add_argument('--dataset_path', default='dataset/test_spectrogram.pt')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--index_manifest', default='',
                    help='build_index_manifest.py 產生的 {split}_index_manifest.json，'
                         '取代舊的 --snr_labels：除了 SNR 分組，還多存了 basename，'
                         '用來去 raw waveform 資料夾底下找對應的 _sti.npy（真實 restimulus），'
                         '藉此算出官方風格、遮罩過的 RMSE_MF_official。')
    p.add_argument('--sti_root', default='',
                    help='raw waveform 的 clean 資料夾路徑（<basename>_sti.npy 存放處），'
                         '例如 processed_fcn_paper_aligned/semg/test/clean。'
                         '不指定的話就不算 RMSE_MF_official，只印舊版 segment-averaged 的 RMSE_MF。')
    p.add_argument('--csv_out', default='analysis/fcn_baseline_metrics_by_snr.csv')
    args = p.parse_args()

    device = auto_select_gpu()
    model = FCN_01().to(device)
    
    weight_path = args.weight_path
    print(f"→ loading weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    with open(args.student_config) as f:
        cfg = yaml.safe_load(f)
    n_fft = cfg['model']['n_fft']
    hop_size = cfg['model']['hop_size']
    win_size = cfg['model']['win_size']
    compress_factor = cfg['model']['compress_factor']
    
    if 'metrics' not in cfg:
        cfg['metrics'] = {'sampling_rate': 1000, 'n_fft': 256, 'hop_size': 32, 'win_size': 128, 'segment_sec': 1}

    print("→ loading dataset (Test Set)...")
    dataset_path = args.dataset_path
    X_all, y_all = torch.load(dataset_path, map_location='cpu')
    N = X_all.size(0)
    print(f"✔ test dataset size = {N}")

    # ---- index manifest（snr + basename，用來做 SNR 分組 + 真實 restimulus MF）----
    snr_labels = None
    basenames = None
    if args.index_manifest:
        with open(args.index_manifest) as f:
            manifest = json.load(f)
        if len(manifest) != N:
            print(f"⚠️ 警告：--index_manifest 檔案有 {len(manifest)} 筆，跟資料集的 {N} 筆對不上，"
                  f"忽略這個檔案，不做 SNR 分組統計、也不算 RMSE_MF_official（只印總平均）。")
        else:
            snr_labels = [m['snr'] for m in manifest]
            basenames = [m['basename'] for m in manifest]
            print(f"✔ 已載入 index manifest，共 {len(set(snr_labels))} 個 SNR 檔位："
                  f"{sorted(set(snr_labels), reverse=True)}")
            if not args.sti_root:
                print("⚠️ 沒有指定 --sti_root，不會算 RMSE_MF_official（只有 SNR 分組，"
                      "MF 還是用舊版 segment-averaged 算法）")

    sum_metrics = {"SNRimp":0., "RMSE":0., "PRD(%)":0., "RMSE_ARV":0.,
                   "RMSE_MF(Hz)":0., "RMSE_MeanF(Hz)":0., "RMSE_MedianF(Hz)":0., "RMSEM":0.}
    count_samples = 0
    batch_size = args.batch_size

    sum_mf_official = 0.0
    n_mf_valid = 0
    n_no_sti = 0
    n_zero_active = 0

    # 每個 SNR 檔位各自的累加器
    snr_sum_metrics = defaultdict(lambda: {"SNRimp":0., "RMSE":0., "PRD(%)":0., "RMSE_ARV":0.,
                                            "RMSE_MF(Hz)":0., "RMSE_MeanF(Hz)":0., "RMSE_MedianF(Hz)":0., "RMSEM":0.,
                                            "RMSE_MF_official(Hz)":0.})
    snr_count = defaultdict(int)
    snr_mf_count = defaultdict(int)

    with torch.no_grad():
        for st in tqdm(range(0, N, batch_size), desc="[FCN] Inference"):
            ed = min(st + batch_size, N)
            noisy_spec = X_all[st:ed].to(device)
            clean_spec = y_all[st:ed].to(device)

            clean_wav = spec_to_wav(clean_spec, n_fft, hop_size, win_size, compress_factor)
            noisy_wav = spec_to_wav(noisy_spec, n_fft, hop_size, win_size, compress_factor)

            pred_wav = model(noisy_wav)
            
            minL = int(min(clean_wav.size(-1), pred_wav.size(-1), noisy_wav.size(-1)))
            clean_wav = clean_wav[:, :minL]
            pred_wav = pred_wav[:, :minL]
            noisy_wav = noisy_wav[:, :minL]

            m_batch = _inline_compute_metrics_tensor(clean_wav.cpu().numpy(), pred_wav.cpu().numpy(), noisy_wav.cpu().numpy(), cfg=cfg)
            B = clean_wav.size(0)
            
            sum_metrics["SNRimp"]           += float(m_batch.get("SNRimp", 0.)) * B
            sum_metrics["RMSE"]             += float(m_batch.get("RMSE", 0.)) * B
            sum_metrics["PRD(%)"]           += float(m_batch.get("PRD(%)", 0.)) * B
            sum_metrics["RMSE_ARV"]         += float(m_batch.get("RMSE_ARV", 0.)) * B
            sum_metrics["RMSE_MF(Hz)"]      += float(m_batch.get("RMSE_MF(Hz)", 0.)) * B
            sum_metrics["RMSE_MeanF(Hz)"]   += float(m_batch.get("RMSE_MeanF(Hz)", 0.)) * B
            sum_metrics["RMSE_MedianF(Hz)"] += float(m_batch.get("RMSE_MedianF(Hz)", 0.)) * B
            sum_metrics["RMSEM"]            += float(m_batch.get("RMSEM", 0.)) * B
            count_samples += B

            # ── SNR 分組統計：逐 sample 算，不用整個 batch 的聚合值去分組，
            # 避免 batch 剛好橫跨兩個 SNR 檔位邊界時把不同檔位的分數混在一起。
            if snr_labels is not None:
                Wc = clean_wav.cpu().numpy(); Wd = pred_wav.cpu().numpy(); Wn = noisy_wav.cpu().numpy()
                for j in range(B):
                    idx = st + j
                    snr = snr_labels[idx]
                    m_s = _inline_compute_metrics_tensor(Wc[j], Wd[j], Wn[j], cfg=cfg)
                    sm = snr_sum_metrics[snr]
                    sm["SNRimp"]           += float(m_s.get("SNRimp", 0.))
                    sm["RMSE"]             += float(m_s.get("RMSE", 0.))
                    sm["PRD(%)"]           += float(m_s.get("PRD(%)", 0.))
                    sm["RMSE_ARV"]         += float(m_s.get("RMSE_ARV", 0.))
                    sm["RMSE_MF(Hz)"]      += float(m_s.get("RMSE_MF(Hz)", 0.))
                    sm["RMSE_MeanF(Hz)"]   += float(m_s.get("RMSE_MeanF(Hz)", 0.))
                    sm["RMSE_MedianF(Hz)"] += float(m_s.get("RMSE_MedianF(Hz)", 0.))
                    sm["RMSEM"]            += float(m_s.get("RMSEM", 0.))
                    snr_count[snr] += 1

                    # ── RMSE_MF_official：用真實 restimulus 遮罩，不是近似值 ──
                    if args.sti_root and basenames is not None:
                        sti_full = load_sti(basenames[idx], args.sti_root)
                        if sti_full is None:
                            n_no_sti += 1
                        else:
                            minL_sti = min(len(sti_full), Wc[j].shape[-1])
                            rmse_mf_off, n_active = rmse_mf_official_style(
                                Wc[j][:minL_sti], Wd[j][:minL_sti], cfg, sti_full[:minL_sti])
                            if n_active == 0 or np.isnan(rmse_mf_off):
                                n_zero_active += 1
                            else:
                                sum_mf_official += rmse_mf_off
                                n_mf_valid += 1
                                sm["RMSE_MF_official(Hz)"] += rmse_mf_off
                                snr_mf_count[snr] += 1

    avg = {k: (v / max(1, count_samples)) for k, v in sum_metrics.items()}
    
    print("\n===== Test-set Average (FCN Baseline) =====")
    for k, v in avg.items():
        print(f"  {k:25s}: {v:.4f}")
    if args.sti_root:
        avg_mf_official = sum_mf_official / max(1, n_mf_valid)
        print(f"  {'RMSE_MF_official(Hz)':25s}: {avg_mf_official:.4f}  "
              f"(用 {n_mf_valid}/{count_samples} 筆有效樣本；"
              f"{n_no_sti} 筆找不到 _sti.npy，{n_zero_active} 筆 restimulus 全零，皆已排除不計入平均)")

    if snr_labels is not None and snr_count:
        print("\n===== Per-SNR Average (FCN Baseline) =====")
        snr_avg_rows = []
        for snr in sorted(snr_count.keys(), reverse=True):
            n_s = snr_count[snr]
            n_mf_s = snr_mf_count[snr]
            row = {"SNR": snr, "n": n_s}
            row.update({k: v / max(1, n_s) for k, v in snr_sum_metrics[snr].items() if k != "RMSE_MF_official(Hz)"})
            row["RMSE_MF_official(Hz)"] = (snr_sum_metrics[snr]["RMSE_MF_official(Hz)"] / n_mf_s) if n_mf_s > 0 else float('nan')
            snr_avg_rows.append(row)
            mf_off_str = f"{row['RMSE_MF_official(Hz)']:.4f}" if n_mf_s > 0 else "N/A"
            print(f"  SNR={snr:>4} dB (n={n_s:4d}) | "
                  f"SNRimp={row['SNRimp']:.4f}  RMSE={row['RMSE']:.4f}  "
                  f"RMSE_MF(舊)={row['RMSE_MF(Hz)']:.4f}  RMSE_MF(官方,n={n_mf_s})={mf_off_str}")

        csv_path = args.csv_out
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ["SNR", "n", "SNRimp", "RMSE", "PRD(%)", "RMSE_ARV",
                          "RMSE_MF(Hz)", "RMSE_MF_official(Hz)", "RMSE_MeanF(Hz)", "RMSE_MedianF(Hz)", "RMSEM"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in snr_avg_rows:
                writer.writerow(row)
        print(f"\n[info] 每個 SNR 檔位的分數已存到 {csv_path}")
        print("       這份 CSV 可以直接餵進 plot_snrimp_curve.py 畫折線圖。")

if __name__ == '__main__':
    main()