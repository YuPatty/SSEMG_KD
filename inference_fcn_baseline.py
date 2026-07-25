# inference_fcn_baseline.py
import os, sys, yaml, torch
import numpy as np
from tqdm import tqdm
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

# =====================================================================
# 主推論流程 (極致節省 RAM 避開 OOM)
# =====================================================================
def spec_to_wav(spec, n_fft, hop_size, win_size, compress_factor):
    mag = spec[:, 0]
    pha = spec[:, 1]
    return mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor)

def main():
    device = auto_select_gpu()
    model = FCN_01().to(device)
    
    weight_path = 'model_weight/fcn_baseline.pth'
    print(f"→ loading weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    with open('config/config_student_crossarch.yaml') as f:
        cfg = yaml.safe_load(f)
    n_fft = cfg['model']['n_fft']
    hop_size = cfg['model']['hop_size']
    win_size = cfg['model']['win_size']
    compress_factor = cfg['model']['compress_factor']
    
    if 'metrics' not in cfg:
        cfg['metrics'] = {'sampling_rate': 1000, 'n_fft': 256, 'hop_size': 32, 'win_size': 128, 'segment_sec': 1}

    print("→ loading dataset (Test Set)...")
    dataset_path = 'dataset/test_spectrogram.pt'
    X_all, y_all = torch.load(dataset_path, map_location='cpu')
    N = X_all.size(0)
    print(f"✔ test dataset size = {N}")

    sum_metrics = {"SNRimp":0., "RMSE":0., "PRD(%)":0., "RMSE_ARV":0.,
                   "RMSE_MF(Hz)":0., "RMSE_MeanF(Hz)":0., "RMSE_MedianF(Hz)":0., "RMSEM":0.}
    count_samples = 0
    batch_size = 4

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

    avg = {k: (v / max(1, count_samples)) for k, v in sum_metrics.items()}
    
    print("\n===== Test-set Average (FCN Baseline) =====")
    for k, v in avg.items():
        print(f"  {k:25s}: {v:.4f}")

if __name__ == '__main__':
    main()