
from scipy.ndimage import gaussian_filter1d
import numpy as np
import skrf as rf
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.special import erfc

# ================= 1. 系統參數設定區 =================
GBPS = 112 # 總資料率 224 Gbps
BAUD = GBPS / 2  # 符號率 112 GBaud
T_RISE_FALL = 1.5e-12  # TX 上升/下降時間 1.5 ps
SPS = 256  # 每 UI 取樣點數
N_SYMBOLS = 15000  # 擷取的 QPRBS23 符號數量
V_SWING = 0.8  # 差分峰對峰電壓 (0.8 Vpp)
S4P_FILE = 'your_file.s4p'  # 請替換為您的 Keysight VNA 檔案

# --- 16-Taps DFE 權重設定 ---
# 在實際硬體中，這些權重由 LMS (Least Mean Squares) 演算法自動收斂。
# 這裡我們模擬一組典型的衰減權重，您可以根據 S 參數的反射情況微調。
DFE_TAPS = np.array([
    0.150, 0.080, 0.040, 0.020,  # Tap 1-4 (處理主游標後的強烈 ISI)
    0.010, 0.008, 0.005, 0.002,  # Tap 5-8 (處理中度反射)
    0.001, -0.002, -0.005, -0.001,  # Tap 9-12 (處理連接器/封裝反射，可能有負值)
    0.000, 0.000, 0.000, 0.000  # Tap 13-16 (深層微調)
])
# ==================================================

ui = 1 / (BAUD * 1e9)  # 1 UI = 8.928 ps
fs = SPS / ui  # 採樣頻率
unit_voltage = V_SWING / 6  # PAM4 每個 Level 之間的基礎電壓差



# ================= 2. QPRBS23 生成與 TX 塑形 =================
def generate_qprbs23(num_symbols):
    """產生 PRBS23 並透過 Gray Code 映射為 PAM4 符號 (-3, -1, 1, 3)"""
    state = 0x7FFFFF
    bits = np.zeros(num_symbols * 2, dtype=np.int8)
    for i in range(num_symbols * 2):
        bits[i] = state & 1
        feedback = ((state >> 22) ^ (state >> 17)) & 1
        state = ((state << 1) | feedback) & 0x7FFFFF

    symbols = np.zeros(num_symbols)
    for i in range(num_symbols):
        b1, b0 = bits[2 * i], bits[2 * i + 1]
        if b1 == 0 and b0 == 0:
            symbols[i] = -3
        elif b1 == 0 and b0 == 1:
            symbols[i] = -1
        elif b1 == 1 and b0 == 1:
            symbols[i] = 1
        elif b1 == 1 and b0 == 0:
            symbols[i] = 3
    return symbols


print("Generating 224Gbps PAM4 QPRBS23 Signal...")
pam4_symbols = generate_qprbs23(N_SYMBOLS)
tx_ideal = np.repeat(pam4_symbols * unit_voltage, SPS)
sigma = (T_RISE_FALL / 1.683) / (ui / SPS)
tx_signal = gaussian_filter1d(tx_ideal, sigma=sigma)


# ================= 3. S 參數讀取與 FFT 精確對齊 =================
def derive_minimum_phase(magnitude):
    """
    利用倒頻譜 (Cepstrum) 方法從振幅響應推導 Minimum Phase
    """
    # 1. 避免 log(0) 的數學錯誤，設定一個極小的地板值
    log_mag = np.log(np.maximum(magnitude, 1e-12))

    # 2. IFFT 得到實數倒頻譜 (Real Cepstrum)
    cepstrum = ifft(log_mag).real

    # 3. 建立因果窗 (Causal Window)
    n = len(cepstrum)
    window = np.zeros(n)
    window[0] = 1.0  # DC 分量保持不變

    if n % 2 == 0:
        window[1:n // 2] = 2.0  # 正時間軸能量翻倍 (補償負時間軸被消除的能量)
        window[n // 2] = 1.0  # 奈奎斯特點
    else:
        window[1:(n + 1) // 2] = 2.0

    # 4. 套用因果窗並透過 FFT 轉回頻域
    min_phase_complex = fft(cepstrum * window)

    # 5. 取自然指數重建包含 Minimum Phase 的系統響應
    h_min_phase = np.exp(min_phase_complex)

    return h_min_phase


def apply_s4p_channel(sig_in, file_path, fs, use_min_phase=True):
    try:
        print(f"Loading S-parameter: {file_path}")
        nw = rf.Network(file_path)

        if nw.nports == 4:
            nw.renumber([0, 1, 2, 3], [0, 2, 1, 3])
            nw.se2gmm(p=2)
            sdd21 = nw.s[:, 1, 0]
        else:
            sdd21 = nw.s[:, 1, 0]

        freqs_vna = nw.f
        sig_freqs = fftfreq(len(sig_in), 1 / fs)

        # --- 插值處理 (對齊頻率網格) ---
        dc_value = np.abs(sdd21[0]) + 0j
        hf_value = sdd21[-1]
        interp_func = interp1d(freqs_vna, sdd21, kind='linear',
                               bounds_error=False, fill_value=(dc_value, hf_value))
        h_freq = interp_func(np.abs(sig_freqs))

        # --- 最小相位推導 (Minimum Phase Derivation) ---
        if use_min_phase:
            print("Deriving Minimum Phase to enforce causality...")
            # 提取插值後的純振幅
            magnitude = np.abs(h_freq)
            # 重新推導最小相位響應
            h_freq = derive_minimum_phase(magnitude)
        else:
            print("Using original VNA phase...")
            h_freq[0] = np.abs(h_freq[0])  # 確保 DC 為實數即可

        print("Executing FFT Convolution...")
        rx_fft = fft(sig_in) * h_freq
        return np.real(ifft(rx_fft))

    except Exception as e:
        print(f"S-Parameter Warning: {e}. Outputting raw TX signal.")
        return sig_in



# ================= 4. 接收端等化器 (RX CTLE & 16-Taps DFE) =================
def apply_ctle(sig_in, fs, f_peak_ghz, peaking_db):
    print(f"Applying CTLE (Peaking: {peaking_db}dB @ {f_peak_ghz}GHz)...")
    freqs = fftfreq(len(sig_in), 1 / fs)
    s = 2j * np.pi * freqs
    w_z = 2 * np.pi * (f_peak_ghz * 1e9) / (10 ** (peaking_db / 20))
    w_p1 = 2 * np.pi * (f_peak_ghz * 1e9)
    w_p2 = 2 * np.pi * (f_peak_ghz * 1e9) * 2

    h_ctle = (s + w_z) / ((s + w_p1) * (s + w_p2))
    h_ctle /= np.abs(h_ctle[np.argmin(np.abs(freqs - 1e6))])
    h_ctle *= (10 ** (-peaking_db / 20))
    return np.real(ifft(fft(sig_in) * h_ctle))


def apply_dfe_pam4_16taps(sig_in, sps, tap_weights):
    """PAM4 專用 16-Taps 判決回饋等化器"""
    print(f"Applying PAM4 {len(tap_weights)}-Taps DFE...")
    sample_indices = np.arange(sps // 2, len(sig_in), sps).astype(int)
    sig_dfe = np.copy(sig_in)

    thresholds = [unit_voltage * -2, 0, unit_voltage * 2]
    history = np.zeros(len(tap_weights))  # 儲存過去 16 個 UI 的判決結果

    for i in range(len(sample_indices)):
        idx = sample_indices[i]
        curr_val = sig_dfe[idx]

        # 1. 計算這 16 個 Tap 造成的總 ISI 懲罰
        isi_penalty = np.sum(tap_weights * history) * unit_voltage

        # 2. 從當前取樣點扣除 ISI (等化)
        corrected_val = curr_val - isi_penalty

        # 3. 進行判決 (Slicing)
        if corrected_val > thresholds[2]:
            d = 3
        elif corrected_val > thresholds[1]:
            d = 1
        elif corrected_val > thresholds[0]:
            d = -1
        else:
            d = -3

        # 4. 更新歷史紀錄 (將陣列向右平移，新判決放入 index 0)
        history = np.roll(history, 1)
        history[0] = d

        # 5. 為了繪圖，將這個 ISI 修正量應用到整個 UI 週期上
        # (真實硬體只在取樣點做減法，但視覺化時我們會看到階梯狀的波形修正)
        start, end = i * sps, (i + 1) * sps
        sig_dfe[start:end] -= isi_penalty

    return sig_dfe

# ================= Insertion LOSS =================
def plot_insertion_loss(file_path, nyquist_freq_ghz):
    nw = rf.Network(file_path)
    if nw.nports == 4:
        nw.renumber([0, 1, 2, 3], [0, 2, 1, 3])
        nw.se2gmm(p=2)
        sdd21 = nw.s_db[:, 1, 0]
        sdd11 = nw.s_db[:, 1, 1]
    else:
        sdd21 = nw.s_db[:, 1, 0]

    freqs_ghz = nw.f / 1e9

    plt.figure(figsize=(10, 5))
    plt.plot(freqs_ghz, sdd21, label='Sdd21 (Insertion Loss)', color='cyan', linewidth=2)

    # 標註 Nyquist 點
    idx_nyquist = np.argmin(np.abs(freqs_ghz - nyquist_freq_ghz))
    loss_at_nyquist = sdd21[idx_nyquist]

    plt.axvline(x=nyquist_freq_ghz, color='red', linestyle='--', alpha=0.7)
    plt.plot(nyquist_freq_ghz, loss_at_nyquist, 'ro')
    plt.annotate(f'Nyquist @ {nyquist_freq_ghz}GHz\nLoss: {loss_at_nyquist:.2f} dB',
                 xy=(nyquist_freq_ghz, loss_at_nyquist), xytext=(nyquist_freq_ghz + 5, loss_at_nyquist + 5),
                 arrowprops=dict(facecolor='white', shrink=0.05), color='white')

    plt.title(f"Channel Frequency Response: {file_path.split('/')[-1]}", color='white')
    plt.xlabel("Frequency (GHz)", color='white')
    plt.ylabel("Magnitude (dB)", color='white')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.gca().set_facecolor('#1e1e1e')
    plt.gcf().set_facecolor('#1e1e1e')
    plt.tick_params(colors='white')
    plt.ylim(bottom=-40)  # 聚焦在關鍵損耗區間
    plt.show()

# ================= 繪製 CTLE 頻響 =================
def plot_equalized_response(file_path, f_peak_ghz, peaking_db):
    print(f"Plotting Frequency Response for {file_path}")

    # 1. 讀取原始通道 S 參數 (Sdd21)
    NR=28
    nw = rf.Network(file_path)
    if nw.nports == 4:
        nw.renumber([0, 1, 2, 3], [0, 2, 1, 3])
        nw.se2gmm(p=2)
        sdd21 = nw.s[:, 1, 0]
    else:
        sdd21 = nw.s[:, 1, 0]

    freqs_vna = nw.f
    freqs_ghz = freqs_vna / 1e9

    # 計算原始通道的 dB 值 (避免 log(0))
    channel_mag_db = 20 * np.log10(np.abs(sdd21) + 1e-12)

    # 2. 建立 CTLE 頻域轉移函數 (與時域模擬相同的公式)
    s = 2j * np.pi * freqs_vna
    w_z = 2 * np.pi * (f_peak_ghz * 1e9) / (10 ** (peaking_db / 20))
    w_p1 = 2 * np.pi * (f_peak_ghz * 1e9)
    w_p2 = 2 * np.pi * (f_peak_ghz * 1e9) * 2

    h_ctle = (s + w_z) / ((s + w_p1) * (s + w_p2))

    # 正規化：讓 DC (極低頻) 的增益對應到我們設定的衰減量
    idx_1mhz = np.argmin(np.abs(freqs_vna - 1e6))
    h_ctle /= np.abs(h_ctle[idx_1mhz])
    h_ctle *= (10 ** (-peaking_db / 20))

    ctle_mag_db = 20 * np.log10(np.abs(h_ctle) + 1e-12)

    # 3. 計算等化後的總頻率響應 (相乘在 dB 域就是相加)
    combined_mag_db = channel_mag_db + ctle_mag_db

    # 4. 繪圖可視化
    plt.figure(figsize=(12, 7), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')

    # 繪製三條曲線
    plt.plot(freqs_ghz, channel_mag_db, label='Original Channel (Sdd21)', color='cyan', linewidth=2, alpha=0.7)
    plt.plot(freqs_ghz, ctle_mag_db, label=f'CTLE (Peaking: {peaking_db}dB)', color='magenta', linewidth=2,
             linestyle='--')
    plt.plot(freqs_ghz, combined_mag_db, label='Combined Equalized Response', color='yellow', linewidth=2.5)

    # 標註 Nyquist 頻率點 (以 224G PAM4 的 56GHz 為例)
    plt.axvline(x=NR, color='red', linestyle=':', alpha=0.8)

    idx_nyquist = np.argmin(np.abs(freqs_ghz - NR))
    loss_before = channel_mag_db[idx_nyquist]
    loss_after = combined_mag_db[idx_nyquist]

    plt.plot(NR, loss_before, 'ro')
    plt.plot(NR, loss_after, 'ro')

    plt.annotate(f'Nyquist (28GHz)\nBefore: {loss_before:.1f} dB\nAfter: {loss_after:.1f} dB',
                 xy=(NR, loss_after), xytext=(NR + 3, loss_after + 10),
                 arrowprops=dict(arrowstyle='->', color='white'), color='white', fontsize=11)

    # 圖表美化
    plt.title(f"Frequency Response: Channel vs CTLE vs Combined\n{file_path.split('/')[-1]}", color='white',
              fontsize=14)
    plt.xlabel("Frequency (GHz)", color='white', fontsize=12)
    plt.ylabel("Magnitude (dB)", color='white', fontsize=12)
    plt.legend(facecolor='#1e1e1e', edgecolor='gray', labelcolor='white')
    plt.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.tick_params(colors='white')

    # 設定合理的觀察範圍 (限制在 100GHz 以內)
    plt.xlim(0, 100)
    plt.ylim(bottom=-50, top=10)
    plt.show()




# ================= 5. 繪製 224G PAM4 眼圖 =================
def plot_pam4_eye(signal, sps, title):
    num_eyes = 2
    samples_per_eye = sps * num_eyes
    start_offset = sps * 200  # 避開初始狀態與 FFT 邊界效應
    segments = (len(signal) - start_offset) // samples_per_eye

    plt.figure(figsize=(12, 7), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')

    time_axis = np.linspace(0, num_eyes, samples_per_eye)
    max_traces = min(segments, 4000)

    for i in range(max_traces):
        idx = start_offset + i * samples_per_eye
        plt.plot(time_axis, signal[idx: idx + samples_per_eye],
                 color='#00FFFF', alpha=0.03, linewidth=0.5)

    plt.title(title, color='white', fontsize=14)
    plt.xlabel("Time (UI)", color='white')
    plt.ylabel("Voltage (V)", color='white')

    # 畫出 3 個判決門檻的參考線
    plt.axhline(y=unit_voltage * 2, color='red', linestyle=':', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle=':', alpha=0.5)
    plt.axhline(y=unit_voltage * -2, color='red', linestyle=':', alpha=0.5)

    plt.grid(True, color='gray', linestyle=':', alpha=0.3)
    plt.tick_params(colors='white')
    plt.xlim(0, num_eyes)
    plt.ylim(-V_SWING / 1.3, V_SWING / 1.3)
    plt.show()


def plot_pam4_bathtub(rx_signal, sps, v_swing):
    print("Calculating PAM4 Statistical Bathtub Curve...")

    # 1. 避開初始的暫態效應與 FFT 邊界，並將訊號折疊成 UI 矩陣
    start_idx = sps * 200
    valid_signal = rx_signal[start_idx:]
    num_full_uis = len(valid_signal) // sps
    folded_eye = valid_signal[:num_full_uis * sps].reshape((num_full_uis, sps))

    # 建立時間軸 (0 到 1 UI)
    time_ui = np.linspace(0, 1, sps)

    # PAM4 理論理想準位與判決門檻
    unit_v = v_swing / 6
    thresholds = [-2 * unit_v, 0, 2 * unit_v]  # Bottom, Middle, Top 門檻

    # 儲存每個相位的總 BER
    ber_curve = np.zeros(sps)

    # 2. 掃描眼圖中的每一個相位 (0 到 1 UI)
    for phase_idx in range(sps):
        # 取得該相位下所有的電壓取樣點
        voltages = folded_eye[:, phase_idx]

        # 3. 根據門檻將電壓點分群為四個 PAM4 準位 (-3, -1, 1, 3)
        lvl_m3 = voltages[voltages < thresholds[0]]
        lvl_m1 = voltages[(voltages >= thresholds[0]) & (voltages < thresholds[1])]
        lvl_p1 = voltages[(voltages >= thresholds[1]) & (voltages < thresholds[2])]
        lvl_p3 = voltages[voltages >= thresholds[2]]

        # 若某個群組沒有資料 (眼圖完全閉合/錯位)，給予一個極差的預設值
        if len(lvl_m3) < 2 or len(lvl_m1) < 2 or len(lvl_p1) < 2 or len(lvl_p3) < 2:
            ber_curve[phase_idx] = 0.5
            continue

        # 4. 計算各群組的平均值 (mu) 與標準差 (sigma)
        mu = [np.mean(lvl_m3), np.mean(lvl_m1), np.mean(lvl_p1), np.mean(lvl_p3)]
        sig = [np.std(lvl_m3), np.std(lvl_m1), np.std(lvl_p1), np.std(lvl_p3)]

        # 為了避免除以零，設定最小 sigma
        sig = np.maximum(sig, 1e-6)

        # 5. 計算三個眼孔的 Q-factor (訊雜比的一種表現)
        q_bot = (mu[1] - mu[0]) / (sig[1] + sig[0])
        q_mid = (mu[2] - mu[1]) / (sig[2] + sig[1])
        q_top = (mu[3] - mu[2]) / (sig[3] + sig[2])

        # 6. 使用互補誤差函數 (erfc) 將 Q-factor 轉換為 BER
        # 公式: BER = 0.5 * erfc(Q / sqrt(2))
        ber_bot = 0.5 * erfc(q_bot / np.sqrt(2))
        ber_mid = 0.5 * erfc(q_mid / np.sqrt(2))
        ber_top = 0.5 * erfc(q_top / np.sqrt(2))

        # PAM4 的總 BER 通常受限於最差的那一個眼孔
        ber_curve[phase_idx] = max(ber_bot, ber_mid, ber_top)

    # 7. 限制 BER 的最小值以利繪圖 (例如限制在 1e-18)
    ber_curve = np.maximum(ber_curve, 1e-18)

    # ================= 繪圖 =================
    plt.figure(figsize=(10, 6), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')

    # 繪製 Bathtub 曲線 (使用對數 Y 軸)
    plt.plot(time_ui, np.log10(ber_curve), color='#00FFCC', linewidth=3)

    # 標示業界標準 BER 參考線 (例如 1e-6 若有 FEC，或 1e-12 若無 FEC)
    plt.axhline(y=-6, color='yellow', linestyle='--', alpha=0.7, label='BER 1e-6 (FEC Target)')
    plt.axhline(y=-12, color='red', linestyle=':', alpha=0.7, label='BER 1e-12 (Legacy Target)')

    # 尋找最佳採樣點 (BER 最低的地方)
    best_phase = np.argmin(ber_curve)
    best_ber = ber_curve[best_phase]
    best_time = time_ui[best_phase]

    plt.plot(best_time, np.log10(best_ber), 'ro', markersize=8)
    plt.annotate(f'Best Phase: {best_time:.2f} UI\nMin BER: {best_ber:.1e}',
                 xy=(best_time, np.log10(best_ber)), xytext=(best_time + 0.05, np.log10(best_ber) + 2),
                 arrowprops=dict(arrowstyle='->', color='white'), color='white', fontsize=12)

    plt.title("PAM4 Statistical Bathtub Curve (Horizontal Jitter)", color='white', fontsize=14)
    plt.xlabel("Time (UI)", color='white', fontsize=12)
    plt.ylabel("Log10(BER)", color='white', fontsize=12)
    plt.ylim(-18, 0)  # 從 10^0 (BER=1) 到 10^-18
    plt.xlim(0, 1)
    plt.grid(True, which='both', color='gray', linestyle=':', alpha=0.3)
    plt.legend(facecolor='#1e1e1e', edgecolor='gray', labelcolor='white')
    plt.tick_params(colors='white')

    plt.show()





# 執行等化
rx_channel = apply_s4p_channel(tx_signal, S4P_FILE, fs, use_min_phase=True)
rx_ctle = apply_ctle(rx_channel, fs, f_peak_ghz=32, peaking_db=0)
rx_dfe = apply_dfe_pam4_16taps(rx_ctle, SPS, tap_weights=DFE_TAPS)
#plot_insertion_loss(S4P_FILE, nyquist_freq_ghz=56)
plot_equalized_response(S4P_FILE, f_peak_ghz=32, peaking_db=0)
plot_pam4_eye(rx_channel, SPS, title="224 Gbps PAM4 QPRBS23 ")
plot_pam4_eye(rx_ctle, SPS, title="224 Gbps PAM4 QPRBS23 Post CTLE ")
plot_pam4_eye(rx_dfe, SPS, title="224 Gbps PAM4 QPRBS23 Post CTLE & 16-Tap DFE")
plot_pam4_bathtub(rx_channel, SPS, V_SWING) #plot bathtub