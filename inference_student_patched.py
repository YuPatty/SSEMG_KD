"""
inference_student_patched.py
Batch inference and paper-metric evaluation for SSEMG-Net.

Usage:
    python3 inference_student_patched.py \
        --config config/student_16ch_1blk.yaml \
        --weights model_weight/student_student_16ch_1blk_noKD.pth \
        --dataset dataset/test_spectrogram.pt \
        --snr_labels test_snr_labels.json \
        --csv-out results/student_16ch_1blk_noKD_metrics.csv \
        --exp-alias "16ch_1blk_noKD"

Notes:
  - Loads SSEMGNet checkpoints strictly by default.
  - RMSE_MF denotes Mean Frequency (spectral centroid), matching the paper.
  - Median Frequency is reported only as an optional additional metric.
"""
import os, sys, yaml, torch, numpy as np
import csv, json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
import argparse
import inspect

# ========= inline metrics (same as inference_demo.py v19) =========
def _inline_extract_arv_feature(signal, segment_size=1000):
    import numpy as _np
    L = len(signal)
    if L == 0: return _np.array([0.0])
    if L < segment_size: return _np.array([_np.mean(_np.abs(signal))])
    n = L // segment_size
    return _np.array([_np.mean(_np.abs(signal[i*segment_size:(i+1)*segment_size])) for i in range(n)])

def _inline_extract_mf_feature(signal, segment_size, cfg):
    import numpy as _np, librosa as _librosa
    m = cfg['metrics']; fs=m['sampling_rate']; n_fft=m['n_fft']; hop_len=m['hop_size']; win_len=m['win_size']
    L = len(signal)
    if L == 0: return _np.array([0.0])
    freqs = _librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    start = _np.searchsorted(freqs, 10.0)
    def mf_of(seg):
        D = _librosa.stft(seg, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window='hamming', center=True)
        mag = _np.abs(D)[start:,:]; f = freqs[start:,None]
        mf_frames = (f*mag).sum(axis=0) / (mag.sum(axis=0)+1e-12)
        return mf_frames.mean()
    if L < segment_size: return _np.array([mf_of(signal)])
    n = L // segment_size
    return _np.array([mf_of(signal[i*segment_size:(i+1)*segment_size]) for i in range(n)])

def _inline_extract_medianF_feature(signal, segment_size, cfg):
    import numpy as _np, librosa as _librosa
    m = cfg['metrics']; fs=m['sampling_rate']; n_fft=m['n_fft']; hop_len=m['hop_size']; win_len=m['win_size']
    L = len(signal)
    if L == 0: return _np.array([0.0])
    freqs = _librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    start = _np.searchsorted(freqs, 10.0); f_sel = freqs[start:]
    def median_f_of(seg):
        D = _librosa.stft(seg, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window='hamming', center=True)
        P = (_np.abs(D)**2)[start:,:]
        csum = _np.cumsum(P, axis=0); total = csum[-1,:]+1e-12; half = 0.5*total
        idx = (csum >= half).argmax(axis=0)
        return f_sel[idx].mean()
    if L < segment_size: return _np.array([median_f_of(signal)])
    n = L // segment_size
    return _np.array([median_f_of(signal[i*segment_size:(i+1)*segment_size]) for i in range(n)])

def _inline_compute_metrics_tensor(clean, denoised, noisy=None, cfg=None, segment_size=None):
    import numpy as _np, torch as _torch
    def to_np(x): return x.detach().cpu().numpy() if isinstance(x, _torch.Tensor) else _np.asarray(x)
    clean=to_np(clean); denoised=to_np(denoised); noisy=to_np(noisy) if noisy is not None else None
    if clean.ndim==1:
        clean=clean[None,:]; denoised=denoised[None,:]
        if noisy is not None: noisy=noisy[None,:]
    m=cfg['metrics']; fs=m['sampling_rate']; seg_samp=int(fs*m['segment_sec'])
    segment_size = min(seg_samp, clean.shape[-1]) if segment_size is None else segment_size
    total_signal_pow=0.0; total_noise_pow=0.0; total_num_pts=0
    snrimp_list=[]; rmsem_list=[]; rmse_arv_list=[]; rmse_mf_list=[]; rmse_medianf_list=[]; prd_list=[]; eps=1e-12
    for i in range(len(clean)):
        x,x_hat=clean[i],denoised[i]; noise=x-x_hat
        local_snr=10*_np.log10((_np.sum(x**2)+eps)/(_np.sum(noise**2)+eps))
        total_signal_pow+=_np.sum(x**2); total_noise_pow+=_np.sum(noise**2); total_num_pts+=x.size
        rmsem_list.append(_np.sqrt(_np.mean((noise-noise.mean())**2)))
        if noisy is not None:
            nn=x-noisy[i]; local_nn_snr=10*_np.log10((_np.sum(x**2)+eps)/(_np.sum(nn**2)+eps))
            snrimp_list.append(local_snr-local_nn_snr)
        prd=_np.sqrt((_np.sum((x_hat-x)**2)+eps)/(_np.sum(x**2)+eps))*100.0; prd_list.append(prd)
        arv_c=_inline_extract_arv_feature(x,segment_size); arv_h=_inline_extract_arv_feature(x_hat,segment_size)
        rmse_arv_list.append(_np.sqrt(_np.mean((arv_c-arv_h)**2)))
        meanf_c=_inline_extract_mf_feature(x,segment_size,cfg); meanf_h=_inline_extract_mf_feature(x_hat,segment_size,cfg)
        rmse_mf_list.append(_np.sqrt(_np.mean((meanf_c-meanf_h)**2)))
        medf_c=_inline_extract_medianF_feature(x,segment_size,cfg); medf_h=_inline_extract_medianF_feature(x_hat,segment_size,cfg)
        rmse_medianf_list.append(_np.sqrt(_np.mean((medf_c-medf_h)**2)))
    snr_global=10*_np.log10((total_signal_pow+eps)/(total_noise_pow+eps))
    rmse_global=_np.sqrt(total_noise_pow/max(1,total_num_pts))
    result={"SNR":float(snr_global),"RMSE":float(rmse_global),"RMSEM":float(_np.mean(rmsem_list))}
    if prd_list: result["PRD(%)"]=float(_np.mean(prd_list))
    if rmse_arv_list: result["RMSE_ARV"]=float(_np.mean(rmse_arv_list))
    if rmse_mf_list:
        result["RMSE_MeanF(Hz)"]=float(_np.mean(rmse_mf_list))
        result["RMSE_MF(Hz)"]=result["RMSE_MeanF(Hz)"]
    if rmse_medianf_list:
        result["RMSE_MedianF(Hz)"]=float(_np.mean(rmse_medianf_list))
    if snrimp_list:
        sim=float(_np.mean(snrimp_list)); result["SNR_IMP(dB)"]=sim; result["SNRimp"]=sim
    return result

def _mask_stats(mask_tf, cfg, bands):
    m=mask_tf; eps=1e-8
    mean=float(m.mean()); std=float(m.std())
    entropy=float((-m*np.log(m+eps)-(1-m)*np.log(1-m+eps)).mean())
    pass80=float((m>0.8).mean()); pass95=float((m>0.95).mean())
    sup20=float((m<0.2).mean()); sup05=float((m<0.05).mean())
    mid=float(((m>=0.3)&(m<=0.7)).mean())
    fs=cfg['model']['sampling_rate']; nfft=cfg['model']['n_fft']; F=m.shape[1]
    freqs=np.arange(F)*(fs/nfft); nyq=fs/2.0
    vals=[float(x) for x in str(bands).split(',') if str(x).strip()!='']
    assert len(vals)>=3
    low_lo,low_hi,mid_hi=vals[:3]
    low_lo=max(0.,min(low_lo,nyq)); low_hi=max(low_lo,min(low_hi,nyq)); mid_hi=max(low_hi,min(mid_hi,nyq))
    low=(freqs>=low_lo)&(freqs<low_hi); midb=(freqs>=low_hi)&(freqs<mid_hi); hib=(freqs>=mid_hi)
    b_low=float(m[:,low].mean()) if low.any() else np.nan
    b_mid=float(m[:,midb].mean()) if midb.any() else np.nan
    b_high=float(m[:,hib].mean()) if hib.any() else np.nan
    return {"mask_mean":mean,"mask_std":std,"mask_entropy":entropy,"pass@0.8":pass80,"pass@0.95":pass95,
            "suppr@0.2":sup20,"suppr@0.05":sup05,"mid@0.3-0.7":mid,
            "band_low_mean":b_low,"band_mid_mean":b_mid,"band_high_mean":b_high}

# ============================================================
ROOT = os.path.dirname(__file__)

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--weights', required=True, help='Path to the SSEMG-Net checkpoint')
p.add_argument('--dataset', default='dataset/test_spectrogram.pt', help='Path to test_spectrogram.pt')
p.add_argument('--batch', type=int, default=64)
p.add_argument('--index', type=int, default=1786)
p.add_argument('--metrics-source', choices=['utils','inline'], default='utils')
p.add_argument('--csv-out', default='')
p.add_argument('--dump-dir', default='')
p.add_argument('--bands', default='20,40,150')
p.add_argument('--only-idxs', default='')
p.add_argument('--no-csv', action='store_true')
p.add_argument('--exp-alias', default='ssemgnet')
p.add_argument('--snr_labels', default='',
                help='build_snr_labels.py 產生的 {split}_snr_labels.json 路徑；'
                     '有提供的話，除了總平均，也會印出各 SNR 檔位分開統計的結果')
args = p.parse_args()

CFG_PATH = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

only_set = None
if args.only_idxs.strip():
    only_set = set(int(x) for x in args.only_idxs.split(',') if x.strip()!='')

exp_name = args.exp_alias or (cfg.get('exp') or {}).get('name', 'ssemgnet')
n_type = ((cfg.get('data') or {}).get('noise_type') or cfg.get('n_type') or 'mix')
nv = ((cfg.get('data') or {}).get('nv') or cfg.get('nv') or 0)

MODEL_PATH = args.weights if os.path.isabs(args.weights) else os.path.join(ROOT, args.weights)

BATCH = args.batch
INDEX = args.index
DATASET_PATH = args.dataset if os.path.isabs(args.dataset) else os.path.join(ROOT, args.dataset)
SAVE_FIG = os.path.join(ROOT, f'inference_{exp_name}_idx{INDEX}.png')

print(f"[cfg]  {CFG_PATH}")
print(f"[ckpt] {MODEL_PATH}")
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
if not os.path.isfile(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

# GPU pick
def find_least_used_gpu(threshold_mb=500):
    try:
        result = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,nounits,noheader'])
        memory_info = result.decode().strip().split('\n')
        memory_info = [(i,int(used),int(total)) for i,(used,total) in enumerate([x.split(',') for x in memory_info])]
        candidates = [(i,used) for i,used,total in memory_info if total-used>threshold_mb]
        if not candidates: return None
        return sorted(candidates, key=lambda x: x[1])[0][0]
    except Exception: return None

gpu_id = find_least_used_gpu(threshold_mb=8000)
if torch.cuda.is_available() and gpu_id is not None:
    device = torch.device('cuda')
    print(f'✔ Auto-selected GPU {gpu_id}')
else:
    device = torch.device('cpu')
    print('⚠ No GPU with enough memory → CPU')

# ---- Import SSEMG-Net ----
sys.path.insert(0, os.path.join(ROOT, 'MECG-E'))
from models.StudentNet import StudentSSEMGNet

from spectrogram_utils import mag_pha_istft

# metrics source
if args.metrics_source == 'utils':
    import utils as metrics_utils
    compute_metrics_tensor = metrics_utils.compute_metrics_tensor
    print("[metrics] source = utils module")
else:
    compute_metrics_tensor = _inline_compute_metrics_tensor
    print("[metrics] source = inline")

try:
    import librosa
    print(f"[versions] torch={torch.__version__} | numpy={np.__version__} | librosa={librosa.__version__}")
except Exception as e:
    print("[versions] librosa issue:", e)
print("[metrics cfg] =", cfg.get("metrics", {}))

# ---- Model ----
model = StudentSSEMGNet(cfg).to(device)

def load_checkpoint_strict(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    # Accept checkpoints saved from DataParallel/DDP.
    if checkpoint and all(key.startswith('module.') for key in checkpoint):
        checkpoint = {key[len('module.'):]: value for key, value in checkpoint.items()}

    incompatible = model.load_state_dict(checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

print(f"→ loading weights from: {MODEL_PATH}")
load_checkpoint_strict(model, MODEL_PATH, device)
print("✔ strict load_state_dict OK")
model.eval()
print(f"✔ model on {device} | fea={cfg['model'].get('fea','pha')}")

# ---- Data ----
X_all, y_all = torch.load(DATASET_PATH, map_location='cpu', mmap=True)
N = X_all.size(0)
print(f"✔ dataset size = {N}")
if not (0 <= INDEX < N):
    raise IndexError(f"--index must be in [0, {N - 1}], got {INDEX}")

# ---- SNR labels（用 build_snr_labels.py 重建出來的，用來做「照 SNR 分組統計」）----
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

def istft_from_FT(mag_FT, pha_FT, cfg):
    return mag_pha_istft(
        mag_FT.float(), pha_FT.float(),
        n_fft=cfg['model']['n_fft'], hop_size=cfg['model']['hop_size'],
        win_size=cfg['model']['win_size'], compress_factor=cfg['model']['compress_factor'],
    )

_printed_input_debug = False

@torch.no_grad()
def denoise_batch(noisy_spec):
    global _printed_input_debug
    x_in = noisy_spec.permute(0, 1, 3, 2).contiguous()  # [B,2,T,F]
    if cfg['model']['fea'] == 'cpx':
        real_TF, imag_TF = x_in[:,0], x_in[:,1]
        mag_TF = torch.sqrt(real_TF**2 + imag_TF**2 + 1e-12)
        pha_TF = torch.atan2(imag_TF, real_TF)
        xin_2TF = x_in
    else:
        ch0, ch1 = x_in[:,0], x_in[:,1]
        need_convert = ((ch0 < 0).float().mean() > 0.05) or (ch1.abs().max() > 3.6)
        if not _printed_input_debug:
            _printed_input_debug = True
            print(f"[input] ch0_neg={float((ch0<0).float().mean()):.3f} ch1_max={float(ch1.abs().max()):.3f} convert={need_convert}")
        if need_convert:
            real_TF, imag_TF = ch0, ch1
            mag_TF = torch.sqrt(real_TF**2 + imag_TF**2 + 1e-12)
            pha_TF = torch.atan2(imag_TF, real_TF)
        else:
            mag_TF, pha_TF = ch0, ch1
        xin_2TF = torch.stack([mag_TF, pha_TF], dim=1)

    feat = model.dense_encoder(xin_2TF)
    for blk in model.TSConv:
        feat = blk(feat)
    mask = model.mask_decoder(feat)
    if noisy_spec.size(0) == 1:
        print(f"[mask] mean={mask.mean().item():.4f} min={mask.min().item():.4f} max={mask.max().item():.4f}")

    mag_g_TF = (mag_TF.unsqueeze(1) * mask).squeeze(1)
    pha_g_TF = model.phase_decoder(feat).squeeze(1) if (model.phase_decoder is not None) else pha_TF

    mag_g_FT = mag_g_TF.permute(0,2,1).contiguous()
    pha_g_FT = pha_g_TF.permute(0,2,1).contiguous()
    wav_deno  = istft_from_FT(mag_g_FT, pha_g_FT, cfg)

    mag_noisy_FT = mag_TF.permute(0,2,1).contiguous()
    pha_noisy_FT = pha_TF.permute(0,2,1).contiguous()
    wav_noisy = istft_from_FT(mag_noisy_FT, pha_noisy_FT, cfg)

    return wav_noisy, wav_deno, mask.detach().cpu()

# ---- CSV setup ----
csv_path = args.csv_out or os.path.join(ROOT, 'analysis', f'{exp_name}_metrics.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
csv_header = [
    "idx","exp","use_mrstft","use_entropy",
    "SNR_in","SNR_out","SNRimp","RMSE","PRD(%)",
    "RMSE_ARV","RMSE_MF(Hz)","RMSE_MeanF(Hz)","RMSE_MedianF(Hz)","RMSEM",
    "mask_mean","mask_std","mask_entropy","pass@0.8","pass@0.95",
    "suppr@0.2","suppr@0.05","mid@0.3-0.7",
    "band_low_mean","band_mid_mean","band_high_mean","lsigmoid_slope"
]
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

dump_dir = args.dump_dir
if dump_dir:
    for sub in ['wavs_clean','wavs_noisy','wavs_deno','masks']:
        os.makedirs(os.path.join(dump_dir, sub), exist_ok=True)

use_mrstft = bool((cfg.get('ablation') or {}).get('use_mrstft', True))
use_entropy = bool((cfg.get('ablation') or {}).get('use_entropy', False))

try:
    lslope = float(model.mask_decoder.lsigmoid._positive_slope().mean().item())
except Exception:
    lslope = float('nan')

sum_metrics = {"SNRimp":0.,"RMSE":0.,"PRD(%)":0.,"RMSE_ARV":0.,
               "RMSE_MF(Hz)":0.,"RMSE_MeanF(Hz)":0.,"RMSE_MedianF(Hz)":0.,"RMSEM":0.}
count_samples = 0

# 每個 SNR 檔位各自的累加器，key 是 SNR (int)，value 是跟 sum_metrics 同樣結構的 dict
from collections import defaultdict
snr_sum_metrics = defaultdict(lambda: {"SNRimp":0.,"RMSE":0.,"PRD(%)":0.,"RMSE_ARV":0.,
                                        "RMSE_MF(Hz)":0.,"RMSE_MeanF(Hz)":0.,"RMSE_MedianF(Hz)":0.,"RMSEM":0.})
snr_count = defaultdict(int)

def _process_block(noisy_spec, clean_spec, st_idx, only_set_effective):
    global sum_metrics, count_samples
    with torch.no_grad():
        wav_noisy, wav_deno, mask_b = denoise_batch(noisy_spec)
        x_clean = clean_spec.permute(0,1,3,2).contiguous()
        cmag_TF, cpha_TF = x_clean[:,0], x_clean[:,1]
        cmag_FT = cmag_TF.permute(0,2,1).contiguous()
        cpha_FT = cpha_TF.permute(0,2,1).contiguous()
        wav_clean = istft_from_FT(cmag_FT, cpha_FT, cfg)
    minL = int(min(wav_clean.size(-1), wav_deno.size(-1), wav_noisy.size(-1)))
    wav_clean=wav_clean[:,:minL]; wav_deno=wav_deno[:,:minL]; wav_noisy=wav_noisy[:,:minL]
    if not only_set_effective:
        m_batch = compute_metrics_tensor(wav_clean.cpu().numpy(), wav_deno.cpu().numpy(), wav_noisy.cpu().numpy(), cfg=cfg)
        B = wav_clean.size(0)
        sum_metrics["SNRimp"]          += float(m_batch.get("SNRimp", m_batch.get("SNR_IMP(dB)", 0.))) * B
        sum_metrics["RMSE"]            += float(m_batch["RMSE"]) * B
        sum_metrics["PRD(%)"]          += float(m_batch.get("PRD(%)", 0.)) * B
        sum_metrics["RMSE_ARV"]        += float(m_batch.get("RMSE_ARV", 0.)) * B
        sum_metrics["RMSE_MF(Hz)"]     += float(m_batch["RMSE_MF(Hz)"]) * B
        sum_metrics["RMSE_MeanF(Hz)"]  += float(m_batch.get("RMSE_MeanF(Hz)", 0.)) * B
        sum_metrics["RMSE_MedianF(Hz)"]+=float(m_batch.get("RMSE_MedianF(Hz)", 0.)) * B
        sum_metrics["RMSEM"]           += float(m_batch["RMSEM"]) * B
        count_samples += B
    Wc=wav_clean.cpu().numpy(); Wd=wav_deno.cpu().numpy(); Wn=wav_noisy.cpu().numpy()
    Mk=mask_b.squeeze(1).cpu().numpy()
    do_csv = (not args.no_csv)
    f_csv = open(csv_path,'a',newline='') if do_csv else None
    writer = csv.writer(f_csv) if f_csv else None
    try:
        B=Wc.shape[0]
        for j in range(B):
            idx=st_idx+j
            if only_set_effective and (idx not in only_set): continue
            m_out=compute_metrics_tensor(Wc[j],Wd[j],Wn[j],cfg=cfg)
            m_in=compute_metrics_tensor(Wc[j],Wn[j],None,cfg=cfg)

            # ── SNR 分組統計：用這個 sample 真正屬於的 SNR 檔位累加，
            # 而不是用整個 batch 的聚合值，避免 batch 剛好橫跨兩個 SNR 檔位
            # 邊界時把不同檔位的分數混在一起。
            if snr_labels is not None:
                snr = snr_labels[idx]
                sm = snr_sum_metrics[snr]
                sm["SNRimp"]           += float(m_out.get("SNRimp", m_out.get("SNR_IMP(dB)", 0.)))
                sm["RMSE"]             += float(m_out["RMSE"])
                sm["PRD(%)"]           += float(m_out.get("PRD(%)", 0.))
                sm["RMSE_ARV"]         += float(m_out.get("RMSE_ARV", 0.))
                sm["RMSE_MF(Hz)"]      += float(m_out["RMSE_MF(Hz)"])
                sm["RMSE_MeanF(Hz)"]   += float(m_out.get("RMSE_MeanF(Hz)", 0.))
                sm["RMSE_MedianF(Hz)"] += float(m_out.get("RMSE_MedianF(Hz)", 0.))
                sm["RMSEM"]            += float(m_out["RMSEM"])
                snr_count[snr] += 1

            ms=_mask_stats(Mk[j],cfg,args.bands)
            if writer:
                writer.writerow([
                    idx,exp_name,int(use_mrstft),int(use_entropy),
                    float(m_in["SNR"]),float(m_out["SNR"]),
                    float(m_out.get("SNRimp",m_out.get("SNR_IMP(dB)",0.))),
                    float(m_out["RMSE"]),float(m_out.get("PRD(%)",float("nan"))),
                    float(m_out.get("RMSE_ARV",0.)),float(m_out["RMSE_MF(Hz)"]),
                    float(m_out.get("RMSE_MeanF(Hz)",float("nan"))),
                    float(m_out.get("RMSE_MedianF(Hz)",float("nan"))),
                    float(m_out["RMSEM"]),
                    ms["mask_mean"],ms["mask_std"],ms["mask_entropy"],
                    ms["pass@0.8"],ms["pass@0.95"],ms["suppr@0.2"],ms["suppr@0.05"],
                    ms["mid@0.3-0.7"],ms["band_low_mean"],ms["band_mid_mean"],ms["band_high_mean"],
                    lslope,
                ])
            if dump_dir:
                stem=f"{idx:06d}.npy"
                np.save(os.path.join(dump_dir,'wavs_clean',stem),Wc[j])
                np.save(os.path.join(dump_dir,'wavs_noisy',stem),Wn[j])
                np.save(os.path.join(dump_dir,'wavs_deno', stem),Wd[j])
                np.save(os.path.join(dump_dir,'masks',     stem),Mk[j].astype(np.float16))
    finally:
        if f_csv: f_csv.close()

# ---- Main ----
if args.only_idxs.strip():
    for idx in tqdm(sorted(only_set), desc="Inference (only_idxs)"):
        _process_block(X_all[idx:idx+1].to(device), y_all[idx:idx+1].to(device), idx, True)
else:
    for st in tqdm(range(0, N, BATCH), desc="Inference"):
        ed=min(st+BATCH,N)
        _process_block(X_all[st:ed].to(device), y_all[st:ed].to(device), st, False)

if not args.only_idxs.strip():
    avg={k:(v/max(1,count_samples)) for k,v in sum_metrics.items()}
    print("\n===== Test-set Average (SSEMG-Net) =====")
    for k,v in avg.items():
        print(f"  {k:25s}: {v:.4f}")

    if snr_labels is not None and snr_count:
        print("\n===== Per-SNR Average (SSEMG-Net) =====")
        snr_avg_rows = []
        for snr in sorted(snr_count.keys(), reverse=True):
            n_s = snr_count[snr]
            row = {"SNR": snr, "n": n_s}
            row.update({k: v / max(1, n_s) for k, v in snr_sum_metrics[snr].items()})
            snr_avg_rows.append(row)
            print(f"  SNR={snr:>4} dB (n={n_s:4d}) | "
                  f"SNRimp={row['SNRimp']:.4f}  RMSE={row['RMSE']:.4f}  "
                  f"RMSE_MF={row['RMSE_MF(Hz)']:.4f}")

        snr_csv_path = os.path.splitext(csv_path)[0] + '_by_snr.csv'
        with open(snr_csv_path, 'w', newline='') as f:
            fieldnames = ["SNR", "n", "SNRimp", "RMSE", "PRD(%)", "RMSE_ARV",
                          "RMSE_MF(Hz)", "RMSE_MeanF(Hz)", "RMSE_MedianF(Hz)", "RMSEM"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in snr_avg_rows:
                writer.writerow(row)
        print(f"\n[info] 每個 SNR 檔位的分數已存到 {snr_csv_path}")
        print("       這份 CSV 可以直接餵進 plot_snrimp_curve.py 畫折線圖。")

# ---- Plot ----
from matplotlib import mlab
idx=INDEX
noisy_spec=X_all[idx:idx+1].to(device); clean_spec=y_all[idx:idx+1].to(device)
with torch.no_grad():
    wav_noisy, wav_deno, mask = denoise_batch(noisy_spec)
    x_clean=clean_spec.permute(0,1,3,2).contiguous()
    cmag_TF,cpha_TF=x_clean[:,0],x_clean[:,1]
    cmag_FT=cmag_TF.permute(0,2,1).contiguous(); cpha_FT=cpha_TF.permute(0,2,1).contiguous()
    wav_clean=istft_from_FT(cmag_FT,cpha_FT,cfg)
minL=int(min(wav_clean.size(-1),wav_deno.size(-1),wav_noisy.size(-1)))
clean_wav=wav_clean[0,:minL].cpu().numpy(); denoised=wav_deno[0,:minL].cpu().numpy(); noisy_wav=wav_noisy[0,:minL].cpu().numpy()

# 建立 3 個上下排列的子圖
fig, axs = plt.subplots(3, 1, figsize=(15, 7), sharex=True, sharey=True)

axs[0].plot(clean_wav, label='Clean', color='tab:blue')
axs[0].legend(loc='upper right')
axs[1].plot(noisy_wav, label='Noisy (Input)', color='tab:orange', alpha=0.8)
axs[1].legend(loc='upper right')
axs[2].plot(denoised, label='Denoised (Output)', color='tab:green')
axs[2].legend(loc='upper right')

plt.suptitle(f"[SSEMG-Net] Index {idx} waveform Comparison", fontsize=14)
plt.tight_layout()
plt.savefig(SAVE_FIG, dpi=300) # 加上 dpi=300 讓圖片輸出更清晰
plt.close()
print(f"✓ plot saved => {SAVE_FIG}")