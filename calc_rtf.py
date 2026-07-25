import yaml
import os

try:
    # 讀取你的 Teacher 設定檔來獲取訊號處理參數
    with open('config/config_spectrogram_v19_tt_mask.yaml') as f:
        cfg = yaml.safe_load(f)
        
    sr = float(cfg['model'].get('sampling_rate', 1000))  # 預設 1000 Hz (生理訊號常見採樣率)
    hop_size = float(cfg['model']['hop_size'])
    n_frames = 79

    # 計算 79 幀對應的實際物理時間
    # 時間 (秒) = (幀數 * hop_size) / 取樣率
    signal_duration_sec = (n_frames * hop_size) / sr
    signal_duration_ms = signal_duration_sec * 1000

    print("==============================================")
    print("      邊緣裝置 (CPU) 即時運算率 (RTF) 分析")
    print("==============================================")
    print(f"📍 訊號參數:")
    print(f"   - 採樣率 (Sampling Rate): {sr} Hz")
    print(f"   - 步進長度 (Hop Size): {hop_size}")
    print(f"   - {n_frames} 幀頻譜對應的實際時間: {signal_duration_ms:.2f} ms ({signal_duration_sec:.4f} 秒)\n")

    # 剛剛測量出的 CPU 延遲數據
    t_latency = 3674.10
    s_latency = 51.28
    
    rtf_t = t_latency / signal_duration_ms
    rtf_s = s_latency / signal_duration_ms

    print(f"📍 RTF 結算 (公式: 推論耗時 / 實際時間，< 1 代表可即時處理):")
    print(f"   - Teacher (Mamba): {t_latency:>7.2f} ms / {signal_duration_ms:.2f} ms = {rtf_t:.4f}")
    print(f"   - Student (Conv):  {s_latency:>7.2f} ms / {signal_duration_ms:.2f} ms = {rtf_s:.4f}")
    
    print("==============================================")
    if rtf_s < 1.0:
        print("✅ 結論: Student 模型 RTF 小於 1，完美達成邊緣裝置的即時處理標準！")
    else:
        print("❌ 結論: Student 模型 RTF 大於 1，仍有延遲風險。")
    print("==============================================")

except Exception as e:
    print(f"讀取設定檔失敗，請確認檔案路徑或內容: {e}")
