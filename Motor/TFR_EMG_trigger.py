import os
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend for interactive plotting
import matplotlib.pyplot as plt
import mne

#%% --- Constants ---
file_name = 'mne_raw_filtered_3-45Hz.fif'

file_location = r'W:\Data\2025_7_7_empty_room\brain_FL-on_000\concat\mne_raw'

fif_fname = os.path.join(file_location, file_name)

sens_type = 0 # 0 for NMOR, 1 for Fieldline, 2 for Fieldline with DiN

#%%f
raw_filtered = mne.io.read_raw_fif(fif_fname, preload=True)  # Load the filtered raw data
# Find the events from the stimulus channel ('trigin1'), which is the audio onset
events = mne.find_events(raw_filtered, stim_channel='di32', verbose=True)

#%% EMG Triggered TFR
#emg_ch = 'ai117'
emg_ch = 'auxin0'  # Change to the actual EMG channel name in your data

print('All channels:', raw_filtered.ch_names)
print(f"Original type of {emg_ch}:", raw_filtered.get_channel_types(picks=[raw_filtered.ch_names.index(emg_ch)]))

# Change channel type to EMG for filtering and processing
raw_filtered.set_channel_types({emg_ch: 'emg'})

# Now pick and filter the EMG channel
raw_emg = raw_filtered.copy().pick_channels([emg_ch])

raw_emg.filter(20, 250, picks=[0], fir_design='firwin')

emg_filt = raw_emg.get_data()[0]

# Rectify (absolute value)
emg_rect = np.abs(emg_filt)

# Smooth with moving average window (e.g., 50 ms)
sfreq = raw_filtered.info['sfreq']
window_size = int(0.05 * sfreq)
emg_smooth = np.convolve(emg_rect, np.ones(window_size)/window_size, mode='same')

from scipy.signal import find_peaks

# Set global threshold (optional: tune or compute adaptively)
threshold = np.percentile(emg_smooth, 93)

# Search for EMG peaks only near digital triggers
sfreq = raw_filtered.info['sfreq']
search_window = 0.5  # seconds
event_id_emg = 999
emg_onsets = []

for evt in events:
    evt_sample = evt[0]
    start = int(evt_sample - search_window * sfreq)
    end = int(evt_sample + search_window * sfreq)

    # Guard against boundaries
search_window = 0.5  # seconds
event_id_emg = 999
emg_onsets = []

for evt in events:
    evt_sample = evt[0]
    start = int(evt_sample - search_window * sfreq)
    end = int(evt_sample + search_window * sfreq)

    # Guard against boundaries
    if start < 0 or end >= len(emg_smooth):
        continue

    segment = emg_smooth[start:end]

    # Find peaks with minimum height (threshold) and optional minimum distance
    peaks, properties = find_peaks(segment, height=threshold, distance=0.1*sfreq)

    if len(peaks) > 0:
        peak_sample = start + peaks[0]  # take first peak in window
        emg_onsets.append([peak_sample, 0, event_id_emg])

# Convert to numpy array
events_emg = np.array(emg_onsets, dtype=int)


import matplotlib.pyplot as plt

times = np.arange(len(emg_smooth)) / sfreq

plt.figure(figsize=(15, 4))
plt.plot(times, emg_smooth, label='EMG Envelope (smoothed)')
plt.axhline(threshold, color='r', linestyle='--', label='Threshold')

for onset in events_emg[:, 0]:
    plt.axvline(onset / sfreq, color='g', linestyle=':', alpha=0.7)

for evt in events[:, 0]:
    plt.axvline(evt / sfreq, color='b', linestyle='--', alpha=0.4)  # Digital triggers

plt.xlabel('Time (s)')
plt.ylabel('EMG amplitude (a.u.)')
plt.title('EMG + Detected Peaks (green) and Digital Triggers (blue)')
plt.legend()
plt.tight_layout()
plt.show()


#%% Pick the channels to process
picks = mne.pick_types(raw_filtered.info, meg='mag')

epochs_emg = mne.Epochs(
    raw_filtered, 
    events_emg, 
    event_id={ 'EMG_contraction': event_id_emg },
    tmin=-0.5, 
    tmax=2.0,
    baseline=(-0.5, -0.1),
    detrend=1,
    picks=picks,  # magnetometers, same as before
    preload=True,
    verbose=True
)

evoked_emg = epochs_emg.average()
evoked_emg.plot(titles='Evoked Response (EMG Triggered)', time_unit='s', spatial_colors=True)

#%% --- Time-Frequency Representation (TFR) ---
def plot_tfr(tfr, channel_idx=0, vmin=None, vmax=None, cmap='RdBu_r'):
    power = tfr.data[channel_idx]
    if vmin is None or vmax is None:
        vmin = np.percentile(power, 1)
        vmax = np.percentile(power, 99)

    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(
        tfr.times,
        tfr.freqs,
        power,
        shading='auto',
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )
    plt.colorbar(mesh, label='Power')
    plt.title(f'TFR (Multitaper) - {tfr.ch_names[channel_idx]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
# --- Time-frequency analysis parameters ---
frequencies = np.linspace(10, 30, 100)  # Define the frequency range for TFR
n_cycles = frequencies / 3.0  # Set the number of cycles for each frequency
time_bandwidth = 2.0  # Time-bandwidth product for multitaper method

# Compute TFR
tfr = mne.time_frequency.tfr_multitaper(
    epochs_emg,
    freqs=frequencies,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
    return_itc=False,
    average=True,  # average over epochs
    n_jobs=-1
)

# Baseline correct
tfr.apply_baseline(baseline=(-0.5, -0.1), mode='mean')

# Plot channel 0 (or any other)
plot_tfr(tfr, channel_idx=0)
#%% --- Calculate timing differences between EMG peaks and digital TTL triggers ---

# Convert sample indices to seconds
emg_times_sec = events_emg[:, 0] / sfreq
ttl_times_sec = events[:, 0] / sfreq

# Find the closest TTL to each EMG onset
latencies = []
matched_ttls = []

for emg_time in emg_times_sec:
    idx_closest_ttl = np.argmin(np.abs(ttl_times_sec - emg_time))
    closest_ttl_time = ttl_times_sec[idx_closest_ttl]
    latency = emg_time - closest_ttl_time
    latencies.append(latency)
    matched_ttls.append(closest_ttl_time)

latencies = np.array(latencies)

plt.figure(figsize=(8, 4))
plt.hist(latencies * 1000, bins=20, color='purple', alpha=0.7)
plt.axvline(0, color='k', linestyle='--', label='TTL onset')
plt.title('Latency of EMG Peak Relative to TTL Trigger')
plt.xlabel('Latency (ms)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Mean latency: {np.mean(latencies)*1000:.1f} ms")
print(f"Std latency: {np.std(latencies)*1000:.1f} ms")
print(f"Min latency: {np.min(latencies)*1000:.1f} ms")
print(f"Max latency: {np.max(latencies)*1000:.1f} ms")

#%% --- Epoch EMG signal around TTL triggers to extract average EMG burst shape ---

# Epoch EMG envelope around TTL triggers
window_start = -0.75  # seconds before TTL
window_end = 2.0     # seconds after TTL

emg_epochs_data = []

for ev in events:
    onset = ev[0]
    start = int(onset + window_start * sfreq)
    end = int(onset + window_end * sfreq)
    if start < 0 or end > len(emg_smooth):
        continue
    segment = emg_smooth[start:end]
    emg_epochs_data.append(segment)

emg_epochs_data = np.array(emg_epochs_data)

# Calculate the mean envelope across epochs
mean_env = np.mean(emg_epochs_data, axis=0)

# Create a common time vector for plotting
times = np.linspace(window_start, window_end, mean_env.size)

# Now extract TTL channel and epoch it the same way:
ttl_ch = 'di32'
ttl_data = raw_filtered.copy().pick_channels([ttl_ch]).get_data()[0]

ttl_epochs = []
for ev in events:
    onset = ev[0]
    start = int(onset + window_start * sfreq)
    end = int(onset + window_end * sfreq)
    if start < 0 or end > len(ttl_data):
        continue
    ttl_segment = ttl_data[start:end]
    ttl_epochs.append(ttl_segment)

ttl_epochs = np.array(ttl_epochs)
ttl_avg = np.mean(ttl_epochs, axis=0)

# Plotting both
plt.figure(figsize=(10, 4))
plt.plot(times, mean_env, label='Avg EMG Envelope')
plt.plot(times, ttl_avg / np.max(np.abs(ttl_avg)) * np.max(mean_env),  # scale TTL to EMG range
         label='Avg TTL (scaled)', color='purple', linestyle='--')
plt.axvline(0, color='r', linestyle=':', label='TTL nominal time 0')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Average EMG Envelope + TTL Signal')
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6))

# Plot all EMG envelope trials
for trial in emg_epochs_data:
    plt.plot(times, trial, color='gray', alpha=0.3, label='_nolegend_')  # no legend for individual trials

# Plot average EMG envelope
plt.plot(times, mean_env, color='blue', linewidth=2, label='Avg EMG Envelope')

# Plot all TTL trials, scaled to EMG range
for ttl_trial in ttl_epochs:
    ttl_scaled = ttl_trial / np.max(np.abs(ttl_trial)) * np.max(mean_env)
    plt.plot(times, ttl_scaled, color='purple', alpha=0.3, linestyle='--', label='_nolegend_')

# Plot average TTL signal (scaled)
ttl_avg_scaled = ttl_avg / np.max(np.abs(ttl_avg)) * np.max(mean_env)
plt.plot(times, ttl_avg_scaled, color='purple', linewidth=2, linestyle='--', label='Avg TTL (scaled)')

plt.axvline(0, color='red', linestyle=':', label='TTL nominal time 0')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.title('EMG Envelope and TTL Signal - Individual Trials + Average')
plt.legend()
plt.tight_layout()
plt.show()
# %%
threshold_frac = 0.3  # threshold fraction of max amplitude per trial, adjust as needed
burst_durations = []

for trial in emg_epochs_data:
    max_amp = np.max(trial)
    threshold = threshold_frac * max_amp
    
    above_thresh = np.where(trial >= threshold)[0]
    if len(above_thresh) == 0:
        # No burst detected for this trial, skip or assign zero
        continue
    
    burst_start_idx = above_thresh[0]
    burst_end_idx = above_thresh[-1]
    
    # Convert indices to time (seconds)
    burst_start_time = times[burst_start_idx]
    burst_end_time = times[burst_end_idx]
    
    duration = burst_end_time - burst_start_time
    burst_durations.append(duration)

# Convert to numpy array
burst_durations = np.array(burst_durations)

print(f"Average EMG burst duration across trials: {np.mean(burst_durations):.3f} s")
print(f"STD of EMG burst duration: {np.std(burst_durations):.3f} s")

# Optional: Plot histogram of burst durations
plt.figure(figsize=(6,4))
plt.hist(burst_durations, bins=20, color='skyblue', edgecolor='k')
plt.xlabel('EMG burst duration (s)')
plt.ylabel('Count')
plt.title('Histogram of EMG Burst Durations Across Trials')
plt.tight_layout()
plt.show()


# %%
