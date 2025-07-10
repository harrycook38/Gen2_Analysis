import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend for interactive plotting
import matplotlib.pyplot as plt
import mne

# %% --- Constants ---
file_name = 'mne_raw_filtered_3-45Hz.fif'
file_location = r'W:\Data\2025_07_09_ania_brain\ania_mag_1_1.6k\concat\mne_raw'
fif_fname = os.path.join(file_location, file_name)

sens_type = 1               # 0 = NMOR grad, 1 = NMOR mag, 2 = Fieldline, 3 = Fieldline + DiN
perm_test = False          # Set to True to run the cluster-based permutation test
add_10Hz_filter = False    # Set to True to apply 10 Hz low-pass filter to epochs
baseline_interval = (-0.4, -0.1)  # Baseline interval for TFR and epochs

# %% --- Helper Function: Plot ASD Before/After Epoch Rejection ---
from mne.time_frequency import psd_array_welch

def plot_asd_comparison_epochs(raw_data, epochs, picks, title='ASD Before and After Epoch Rejection', fmax=100):
    sfreq = raw_data.info['sfreq']
    data_before, _ = raw_data[picks]

    # --- Before rejection ---
    n_fft = min(round(10 * sfreq), data_before.shape[1])
    psds_before, freqs = psd_array_welch(data_before, sfreq=sfreq, fmin=0, fmax=fmax, n_fft=n_fft)
    asd_before = np.sqrt(psds_before)

    # --- After rejection ---
    epochs_data = epochs.get_data()
    n_epochs, n_channels, n_times = epochs_data.shape
    data_after = epochs_data.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)
    n_fft = min(round(10 * sfreq), data_after.shape[1])
    psds_after, _ = psd_array_welch(data_after, sfreq=sfreq, fmin=0, fmax=fmax, n_fft=n_fft)
    asd_after = np.sqrt(psds_after)

    # --- Noise floor ---
    freq_mask = (freqs >= 2) & (freqs <= 40)
    noise_floor_before = np.median(asd_before[:, freq_mask])
    noise_floor_after = np.median(asd_after[:, freq_mask])

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, asd_before.T, alpha=0.4, label='Before Rejection')
    plt.plot(freqs, asd_after.T, alpha=0.6, label='After Rejection')
    plt.axhline(noise_floor_before, color='blue', linestyle='--', linewidth=2,
                label=f'Noise Floor Before (2–40 Hz): {noise_floor_before:.1e}')
    plt.axhline(noise_floor_after, color='orange', linestyle='--', linewidth=2,
                label=f'Noise Floor After (2–40 Hz): {noise_floor_after:.1e}')
    plt.yscale('log')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (T/√Hz)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% --- Load Data and Extract Events ---
raw_filtered = mne.io.read_raw_fif(fif_fname, preload=True)

if sens_type == 0:
    events = mne.find_events(raw_filtered, stim_channel='trigin1', verbose=True)
    picks = mne.pick_channels(raw_filtered.info['ch_names'], include=['B_field'])
    reject = dict(mag=5e-12)

if sens_type == 1:
    events = mne.find_events(raw_filtered, stim_channel='trigin1', verbose=True)
    picks = mne.pick_channels(raw_filtered.info['ch_names'], include=['B_field'])
    reject = dict(mag=7e-12)

elif sens_type == 2:
    events = mne.find_events(raw_filtered, stim_channel='ai113', verbose=True, min_duration=0.0005, output='onset', consecutive=True)
    if len(events) > 500:
        from scipy.ndimage import uniform_filter1d
        data, times = raw_filtered.copy().pick('ai113').get_data(return_times=True)
        signal = data[0]
        smoothed = uniform_filter1d(signal, size=100)
        threshold = 0.5
        binary = smoothed > threshold
        rising = np.where(np.diff(binary.astype(int)) == 1)[0]
        fs = int(raw_filtered.info['sfreq'])
        min_interval = fs
        clean_rising = [rising[0]]
        for r in rising[1:]:
            if r - clean_rising[-1] > min_interval:
                clean_rising.append(r)
        clean_rising = np.array(clean_rising)
        events = np.column_stack([clean_rising, np.zeros_like(clean_rising), np.ones_like(clean_rising, dtype=int)])
        print(f"Detected {len(clean_rising)} events after cleaning.")
    picks = mne.pick_channels(raw_filtered.info['ch_names'], include=['s69_bz'])
    reject = dict(mag=3.8e-12)

elif sens_type == 3:
    events = mne.find_events(raw_filtered, stim_channel='di32', verbose=True, min_duration=0.0005, output='onset', consecutive=True)
    picks = mne.pick_channels(raw_filtered.info['ch_names'], include=['s69_bz'])
    reject = dict(mag=3.8e-12)

# %% --- Create Epochs ---
epochs = mne.Epochs(
    raw_filtered, events, event_id=None,
    tmin=-0.5, tmax=3,
    baseline=baseline_interval,
    detrend=1,
    picks=picks,
    preload=True,
    verbose=True,
    reject=reject
)

if add_10Hz_filter:
    epochs.filter(l_freq=10, h_freq=45, method='iir', iir_params=dict(order=4, ftype='butter'))

# %% --- Evoked Response ---
evoked = epochs.average()
evoked.plot(titles='Evoked Response', time_unit='s', spatial_colors=True)

plot_asd_comparison_epochs(raw_filtered, epochs, picks=picks, title='ASD Before and After Epoch Rejection', fmax=100)

# %% --- TFR (Average) ---
def plot_tfr(tfr, channel_idx=0, vmin=None, vmax=None, cmap='RdBu_r'):
    power = tfr.data[channel_idx]
    if vmin is None or vmax is None:
        vmin = np.percentile(power, 1)
        vmax = np.percentile(power, 99)
    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(tfr.times, tfr.freqs, power, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(mesh, label='Power')
    plt.title(f'TFR (Multitaper) - {tfr.ch_names[channel_idx]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

frequencies = np.linspace(10, 30, 100)
n_cycles = frequencies / 3.0
time_bandwidth = 2.0

tfr = mne.time_frequency.tfr_multitaper(
    epochs, freqs=frequencies, n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, return_itc=False, average=True, n_jobs=-1
)
tfr = tfr.apply_baseline(baseline=baseline_interval, mode='mean')

plot_tfr(tfr, channel_idx=0, cmap='RdBu_r')

# %% --- TFR (Per Epoch) and Envelope Analysis ---
channel_idx = 0

tfr = mne.time_frequency.tfr_multitaper(
    epochs, freqs=frequencies, n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, return_itc=False, average=False, n_jobs=-1
)
tfr.apply_baseline(baseline=baseline_interval, mode='mean')

power_data = tfr.data[:, channel_idx]
mean_power = power_data.mean(axis=0)
peak_idx = np.unravel_index(np.argmax(mean_power), mean_power.shape)
peak_freq_idx, peak_time_idx = peak_idx
peak_freq = tfr.freqs[peak_freq_idx]
peak_time = tfr.times[peak_time_idx]
print(f"Peak frequency = {peak_freq:.2f} Hz at {peak_time:.2f} s")

freq_window = 2.5  # Hz
freq_mask = (tfr.freqs >= peak_freq - freq_window) & (tfr.freqs <= peak_freq + freq_window)

signed_amp = np.sign(power_data[:, freq_mask]) * np.sqrt(np.abs(power_data[:, freq_mask]))
signed_amp_band = signed_amp.mean(axis=1)

mean_amp = signed_amp_band.mean(axis=0)
sem_amp = signed_amp_band.std(axis=0, ddof=1) / np.sqrt(signed_amp_band.shape[0])

safe_tmin = -0.3
safe_tmax = 2.8

plot_times = tfr.times[(tfr.times >= safe_tmin) & (tfr.times <= safe_tmax)]
time_mask = (tfr.times >= safe_tmin) & (tfr.times <= safe_tmax)
mean_amp_safe = mean_amp[time_mask]
sem_amp_safe = sem_amp[time_mask]

if not (safe_tmin <= peak_time <= safe_tmax):
    peak_time = plot_times[np.argmax(mean_amp_safe)]

plt.figure(figsize=(10, 4))
plt.plot(plot_times, mean_amp_safe, label=f'{peak_freq:.2f} ± {freq_window} Hz')
plt.fill_between(plot_times, mean_amp_safe - sem_amp_safe, mean_amp_safe + sem_amp_safe, alpha=0.3)
plt.axvline(peak_time, color='r', ls='--', label='Peak time')
plt.axhline(0, color='k', ls=':')
plt.xlabel('Time (s)')
plt.ylabel('Signed amplitude (T)')
plt.title(f"Amplitude envelope (edge-trimmed) — {tfr.ch_names[channel_idx]}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% --- Cluster-based Permutation Test (Optional) ---
from mne.stats import permutation_cluster_1samp_test

if perm_test:
    alpha = 0.05
    tfr_epochs = mne.time_frequency.tfr_multitaper(
        epochs, freqs=frequencies, n_cycles=n_cycles,
        time_bandwidth=time_bandwidth, return_itc=False, average=False, n_jobs=-1
    )
    tfr_epochs.apply_baseline(baseline=baseline_interval, mode='mean')
    data = tfr_epochs.data[:, channel_idx, :, :]

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        data, threshold=None, tail=0, n_permutations=1000, seed=42, n_jobs=-1
    )

    significant_mask = np.zeros(data.shape[1:], dtype=bool)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val < alpha:
            significant_mask[c] = True

    tfr_avg = tfr_epochs.average()
    power = tfr_avg.data[channel_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        tfr_avg.times, tfr_avg.freqs, power,
        shading='auto', cmap='RdBu_r', vmin=-1.1e-24, vmax=1.1e-24
    )
    plt.colorbar(mesh, ax=ax, label='Power')
    ax.contour(tfr_avg.times, tfr_avg.freqs, significant_mask, levels=[0.5], colors='black', linewidths=1.5)
    ax.set_title(f'TFR with Significant Clusters - {tfr_avg.ch_names[channel_idx]}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()