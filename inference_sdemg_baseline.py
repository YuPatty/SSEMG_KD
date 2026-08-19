# inference_sdemg_baseline.py
# 跟 inference_fcn_baseline.py 同一套 metrics 計算方式（逐字沿用）。
#
# ⚠️ SDEMG 是 diffusion model，推論不是單次 forward，而是要跑完整的 reverse
# diffusion sampling（denoise()，預設 50 步，直接呼叫 SDEMG 原始碼裡
# GaussianDiffusion1D.denoise()，沒有自己重寫 sampling 邏輯）。這比 FCN/MSEMG
# 的單次 forward 慢很多，batch_size 預設故意設比較小（2），跑完整個 test set
# 會花不少時間，是正常的，不是卡住。
import os, sys, argparse, yaml, torch, importlib.util, json, csv
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import librosa

from pipeline_spectrogram import auto_select_gpu
from models.StudentNet import mag_pha_istft

# =====================================================================
# Inline metrics 計算函式（跟 inference_fcn_baseline.py 完全一致）
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

# =====================================================================
def load_module_from_path(module_name, file_path, extra_sys_path=None):
    added = False
    if extra_sys_path and extra_sys_path not in sys.path:
        sys.path.insert(0, extra_sys_path)
        added = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if added:
            sys.path.remove(extra_sys_path)

def spec_to_wav(spec, n_fft, hop_size, win_size, compress_factor):
    mag = spec[:, 0]
    pha = spec[:, 1]
    return mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor)

def _fit_length(wav, target_len):
    """裁切或右側 zero-pad 到 target_len，符合 GaussianDiffusion1D 對固定
    seq_length 的要求。"""
    L = wav.size(-1)
    if L == target_len:
        return wav
    if L > target_len:
        return wav[:, :target_len]
    pad = target_len - L
    return torch.nn.functional.pad(wav, (0, pad))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sdemg_repo', required=True)
    p.add_argument('--student_config', default='config/config_student_crossarch.yaml')
    p.add_argument('--weight_path', default='model_weight/sdemg_baseline.pth')
    p.add_argument('--dataset_path', default='dataset/test_spectrogram.pt')

    # 跟訓練時同一套 SDEMG 官方預設值，務必跟 train_sdemg_baseline.py 用的參數一致，
    # 否則載入的 checkpoint 跟 GaussianDiffusion1D 的結構會對不上
    p.add_argument('--feats', type=int, default=128)
    p.add_argument('--seq_length', type=int, default=10000)
    p.add_argument('--sampling_steps', type=int, default=50, help='對應訓練時的 timesteps')
    p.add_argument('--denoise_timesteps', type=int, default=None,
                    help='推論時實際跑幾步 reverse diffusion，預設跟 sampling_steps 一樣；'
                         '可以設更小的值換取推論速度（犧牲品質）')
    p.add_argument('--objective', choices=['pred_noise', 'pred_x0', 'pred_v'], default='pred_noise')
    p.add_argument('--loss_function', choices=['l1', 'l2'], default='l2')
    p.add_argument('--beta_schedule', choices=['linear', 'cosine', 'quad'], default='cosine')
    p.add_argument('--condition', action='store_true', default=True)

    p.add_argument('--batch_size', type=int, default=2,
                    help='diffusion sampling 很吃記憶體/時間，預設故意比 FCN/MSEMG 小')
    p.add_argument('--snr_labels', default='',
                    help='build_snr_labels.py 產生的 {split}_snr_labels.json 路徑；'
                         '有提供的話，除了總平均，也會印出各 SNR 檔位分開統計的結果')
    p.add_argument('--csv_out', default='analysis/sdemg_baseline_metrics_by_snr.csv')
    args = p.parse_args()

    device = auto_select_gpu()

    dfm = load_module_from_path('sdemg_deep_filter_model',
                                 os.path.join(args.sdemg_repo, 'deep_filter_model.py'))
    ddpm = load_module_from_path('sdemg_ddpm_1d',
                                  os.path.join(args.sdemg_repo, 'ddpm_1d.py'),
                                  extra_sys_path=args.sdemg_repo)
    ConditionalModel = dfm.ConditionalModel
    GaussianDiffusion1D = ddpm.GaussianDiffusion1D

    denoiser = ConditionalModel(feats=args.feats)
    print(f"→ loading weights from: {args.weight_path}")
    denoiser.load_state_dict(torch.load(args.weight_path, map_location=device))

    diffusion = GaussianDiffusion1D(
        denoiser,
        seq_length=args.seq_length,
        timesteps=args.sampling_steps,
        objective=args.objective,
        loss_function=args.loss_function,
        beta_schedule=args.beta_schedule,
        condition=args.condition,
    ).to(device)
    diffusion.eval()

    with open(args.student_config) as f:
        cfg = yaml.safe_load(f)
    n_fft = cfg['model']['n_fft']
    hop_size = cfg['model']['hop_size']
    win_size = cfg['model']['win_size']
    compress_factor = cfg['model']['compress_factor']

    if 'metrics' not in cfg:
        cfg['metrics'] = {'sampling_rate': 1000, 'n_fft': 256, 'hop_size': 32, 'win_size': 128, 'segment_sec': 1}

    print("→ loading dataset (Test Set)...")
    X_all, y_all = torch.load(args.dataset_path, map_location='cpu')
    N = X_all.size(0)
    print(f"✔ test dataset size = {N}")
    print(f"⏳ 每個 batch 要跑 {args.denoise_timesteps or args.sampling_steps} 步 reverse diffusion，"
          f"整個 test set（{N} 筆，batch_size={args.batch_size}）預期會花不少時間。")

    # ---- SNR labels（用來做「照 SNR 分組統計」）----
    snr_labels = None
    if args.snr_labels:
        with open(args.snr_labels) as f:
            snr_labels = json.load(f)
        if len(snr_labels) != N:
            print(f"⚠️ 警告：--snr_labels 檔案有 {len(snr_labels)} 筆，跟資料集的 {N} 筆對不上，"
                  f"忽略這個標籤檔，不做 SNR 分組統計（只印總平均）。")
            snr_labels = None
        else:
            print(f"✔ 已載入 SNR 標籤，共 {len(set(snr_labels))} 個檔位：{sorted(set(snr_labels), reverse=True)}")

    sum_metrics = {"SNRimp":0., "RMSE":0., "PRD(%)":0., "RMSE_ARV":0.,
                   "RMSE_MF(Hz)":0., "RMSE_MeanF(Hz)":0., "RMSE_MedianF(Hz)":0., "RMSEM":0.}
    count_samples = 0

    # 每個 SNR 檔位各自的累加器
    snr_sum_metrics = defaultdict(lambda: {"SNRimp":0., "RMSE":0., "PRD(%)":0., "RMSE_ARV":0.,
                                            "RMSE_MF(Hz)":0., "RMSE_MeanF(Hz)":0., "RMSE_MedianF(Hz)":0., "RMSEM":0.})
    snr_count = defaultdict(int)

    with torch.no_grad():
        for st in tqdm(range(0, N, args.batch_size), desc="[SDEMG baseline] Inference"):
            ed = min(st + args.batch_size, N)
            noisy_spec = X_all[st:ed].to(device)
            clean_spec = y_all[st:ed].to(device)

            clean_wav = spec_to_wav(clean_spec, n_fft, hop_size, win_size, compress_factor)
            noisy_wav = spec_to_wav(noisy_spec, n_fft, hop_size, win_size, compress_factor)
            orig_len = int(min(clean_wav.size(-1), noisy_wav.size(-1)))

            noisy_wav_fit = _fit_length(noisy_wav, args.seq_length)   
            pred_wav = diffusion.denoise(noisy_wav_fit.unsqueeze(1),
                                          denoise_timesteps=args.denoise_timesteps).squeeze(1)         
            minL = int(min(orig_len, pred_wav.size(-1)))

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

    avg = {k: (v / max(1, count_samples)) for k, v in sum_metrics.items()}

    print("\n===== Test-set Average (SDEMG no-KD Baseline) =====")
    for k, v in avg.items():
        print(f"  {k:25s}: {v:.4f}")

    if snr_labels is not None and snr_count:
        print("\n===== Per-SNR Average (SDEMG no-KD Baseline) =====")
        snr_avg_rows = []
        for snr in sorted(snr_count.keys(), reverse=True):
            n_s = snr_count[snr]
            row = {"SNR": snr, "n": n_s}
            row.update({k: v / max(1, n_s) for k, v in snr_sum_metrics[snr].items()})
            snr_avg_rows.append(row)
            print(f"  SNR={snr:>4} dB (n={n_s:4d}) | "
                  f"SNRimp={row['SNRimp']:.4f}  RMSE={row['RMSE']:.4f}  "
                  f"RMSE_MF={row['RMSE_MF(Hz)']:.4f}")

        csv_path = args.csv_out
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ["SNR", "n", "SNRimp", "RMSE", "PRD(%)", "RMSE_ARV",
                          "RMSE_MF(Hz)", "RMSE_MeanF(Hz)", "RMSE_MedianF(Hz)", "RMSEM"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in snr_avg_rows:
                writer.writerow(row)
        print(f"\n[info] 每個 SNR 檔位的分數已存到 {csv_path}")
        print("       這份 CSV 可以直接餵進 plot_snrimp_curve.py 畫折線圖。")

if __name__ == '__main__':
    main()
