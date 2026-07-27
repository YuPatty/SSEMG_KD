# ────────────────────────────────────────────────────
# eval_fcn_native.py
# 載入用原始 repo（main.py/Trainer.py，完全未修改）訓練出來的 FCN_01
# checkpoint，接上你們專案 inference_student.py 裡實際使用的指標計算邏輯，
# 逐字移植（不是重新推導），確保跟 Teacher/KD-Student/No-KD 的數字完全可比。
#
# ── 這一版取代了先前的近似版本，原因 ──
# 先前那版我自己土法煉鋼寫了 SNRimp/RMSE/PRD 三個函式，跟 inference_student.py
# 核對後發現：
#   1) SNRimp 公式數學上剛好等價（純屬巧合，不代表可以放心自己重新推導其他指標）
#   2) RMSE 你們用的是「batch-size 加權平均」，不是我原本寫的逐樣本平均——
#      雖然通常很接近，但不是同一個公式，嚴謹上不能算「一致」
#   3) 完全沒做 RMSEM、RMSE_ARV、RMSE_MF、RMSE_MedianF 這幾項
#   4) RMSE_MF / RMSE_MedianF 用的是 cfg['metrics'] 這組獨立的 STFT 參數
#      （n_fft=256, hop=32，跟主模型的 n_fft=512, hop=128 不同），
#      而且只考慮 >=10Hz 的頻率範圍——這些細節如果自己重新猜，幾乎不可能猜對
# 所以這一版把 inference_student.py 裡的計算函式整段逐字複製過來，
# 不再自己重新推導公式，確保數字口徑完全一致。
#
# 用法：
#   python eval_fcn_native.py \
#       --fcn_checkpoint /path/to/save_model/denoise_FCN_01_epochs100_adam_l1_batch16_lr0.0001.pth.tar \
#       --student_config config/config_student_crossarch.yaml \
#       --data_root dataset
# ────────────────────────────────────────────────────
import os, sys, argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))

from fcn_baseline_model import FCN_01
from pipeline_spectrogram import load_dataset, auto_select_gpu
from models.StudentNet import mag_pha_istft


def spec_to_wav_batch(spec, n_fft, hop_size, win_size, compress_factor):
    mag = spec[:, 0]
    pha = spec[:, 1]
    return mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor)


# ══════════════════════════════════════════════════════════
# 以下逐字複製自 inference_student.py，不做任何修改
# （唯一調整：把 `_np`/`_torch`/`_librosa` 的區域別名保留一致，
#  避免跟本檔案其他地方的 import 混淆）
# ══════════════════════════════════════════════════════════
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
# ══════════════════════════════════════════════════════════ 逐字複製區段結束


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fcn_checkpoint', required=True)
    p.add_argument('--student_config', default='config/config_student_crossarch.yaml',
                    help="需要包含 'metrics' 段落（RMSE_MF/RMSE_MedianF 用的獨立 STFT 參數），"
                         "確認你的 config 檔案裡有跟 inference_student.py 一致的 metrics 設定")
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--batch_size', type=int, default=8)
    args = p.parse_args()

    with open(args.student_config) as f:
        cfg = yaml.safe_load(f)

    if 'metrics' not in cfg:
        raise ValueError(
            "config 裡沒有 'metrics' 段落，RMSE_MF/RMSE_MedianF 無法計算。"
            "請確認你用的 config 檔案跟 inference_student.py 用的是同一份，"
            "裡面要有 metrics: {sampling_rate, n_fft, hop_size, win_size, segment_sec}"
        )

    n_fft = cfg['model']['n_fft']
    hop_size = cfg['model']['hop_size']
    win_size = cfg['model']['win_size']
    compress_factor = cfg['model']['compress_factor']

    device = auto_select_gpu()

    checkpoint = torch.load(args.fcn_checkpoint, map_location=device)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError(
            f"checkpoint 格式跟預期不符（應該要有 'model' 這個 key），"
            f"實際 keys: {checkpoint.keys() if isinstance(checkpoint, dict) else type(checkpoint)}"
        )
    print(f"✅ checkpoint 讀取成功，訓練到 epoch={checkpoint.get('epoch')}, "
          f"best_loss={checkpoint.get('best_loss')}")

    model = FCN_01().to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"✅ FCN_01 權重載入成功（strict=True），參數量: {n_params:,}")

    X_te, y_te = load_dataset('test', args.data_root)
    test_loader = DataLoader(TensorDataset(y_te, X_te), batch_size=args.batch_size, shuffle=False)

    # ── 跟 inference_student.py 完全相同的聚合方式：batch-size 加權平均 ──
    sum_metrics = {"SNRimp": 0., "RMSE": 0., "PRD(%)": 0., "RMSE_ARV": 0.,
                   "RMSE_MF(Hz)": 0., "RMSE_MeanF(Hz)": 0., "RMSE_MedianF(Hz)": 0., "RMSEM": 0.}
    count_samples = 0

    with torch.no_grad():
        for clean_spec, noisy_spec in tqdm(test_loader, desc="Evaluating FCN (native-trained)"):
            clean_spec = clean_spec.to(device)
            noisy_spec = noisy_spec.to(device)

            clean_wav = spec_to_wav_batch(clean_spec, n_fft, hop_size, win_size, compress_factor)
            noisy_wav = spec_to_wav_batch(noisy_spec, n_fft, hop_size, win_size, compress_factor)
            pred_wav = model(noisy_wav)

            minL = int(min(clean_wav.size(-1), pred_wav.size(-1), noisy_wav.size(-1)))
            clean_wav = clean_wav[:, :minL]
            pred_wav = pred_wav[:, :minL]
            noisy_wav = noisy_wav[:, :minL]

            m_batch = _inline_compute_metrics_tensor(
                clean_wav.cpu().numpy(), pred_wav.cpu().numpy(), noisy_wav.cpu().numpy(), cfg=cfg
            )
            B = clean_wav.size(0)
            sum_metrics["SNRimp"] += float(m_batch.get("SNRimp", m_batch.get("SNR_IMP(dB)", 0.))) * B
            sum_metrics["RMSE"] += float(m_batch["RMSE"]) * B
            sum_metrics["PRD(%)"] += float(m_batch.get("PRD(%)", 0.)) * B
            sum_metrics["RMSE_ARV"] += float(m_batch.get("RMSE_ARV", 0.)) * B
            sum_metrics["RMSE_MF(Hz)"] += float(m_batch["RMSE_MF(Hz)"]) * B
            sum_metrics["RMSE_MeanF(Hz)"] += float(m_batch.get("RMSE_MeanF(Hz)", 0.)) * B
            sum_metrics["RMSE_MedianF(Hz)"] += float(m_batch.get("RMSE_MedianF(Hz)", 0.)) * B
            sum_metrics["RMSEM"] += float(m_batch["RMSEM"]) * B
            count_samples += B

    avg = {k: (v / max(1, count_samples)) for k, v in sum_metrics.items()}

    print("\n===== Test-set Average (FCN, 原生 repo 訓練) =====")
    print(f"  SNRimp                   : {avg['SNRimp']:.4f}")
    print(f"  RMSE                     : {avg['RMSE']:.4f}")
    print(f"  PRD(%)                   : {avg['PRD(%)']:.4f}")
    print(f"  RMSE_ARV                 : {avg['RMSE_ARV']:.4f}")
    print(f"  RMSE_MF(Hz)              : {avg['RMSE_MF(Hz)']:.4f}")
    print(f"  RMSE_MeanF(Hz)           : {avg['RMSE_MeanF(Hz)']:.4f}")
    print(f"  RMSE_MedianF(Hz)         : {avg['RMSE_MedianF(Hz)']:.4f}")
    print(f"  RMSEM                    : {avg['RMSEM']:.4f}")
    print("\n這幾項指標的計算公式跟聚合方式，跟你們 Teacher/Student/No-KD 的")
    print("inference_student.py 完全一致，可以直接放進同一張比較表。")


if __name__ == '__main__':
    main()
