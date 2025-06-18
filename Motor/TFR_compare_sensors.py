import os
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend for interactive plotting
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import psd_array_welch
import matplotlib.gridspec as gridspec

#%% Set MNE configuration
files = [
    {
        "label": "NMOR",
        "path": r"W:\Data\2025_05_29_Motor_and_FL\Us\Motor_w_FL_1_000\concat\mne_raw\mne_raw_filtered_3-45Hz.fif",
        "sens_type": 0
    },
    {
        "label": "FieldLine",
        "path": r"W:\Data\2025_05_29_Motor_and_FL\FL\processed\3-45Hz_20250529_142407_sub-Ania_file-BRAINandUs1_raw.fif",
        "sens_type": 1
    }
]

#%% Define functions and process dataset

# Fieldline analog input requires custom edge detection
def detect_ttl_rising_edges(raw, channel_name, threshold=2.5, min_interval=0.2, event_id=1, verbose=True):
    # Extract signal
    signal = raw.get_data(picks=channel_name)[0]
    sfreq = raw.info['sfreq']

    # Find rising edges
    above = signal > threshold
    rising_edges = np.where(np.diff(above.astype(int)) == 1)[0] + 1  # +1 for shift

    # Debounce (enforce minimum spacing between triggers)
    if rising_edges.size == 0:
        if verbose:
            print(f"No rising edges found in channel '{channel_name}'.")
        return np.empty((0, 3), dtype=int)

    min_samples = int(sfreq * min_interval)
    filtered = [rising_edges[0]]
    for idx in rising_edges[1:]:
        if idx - filtered[-1] > min_samples:
            filtered.append(idx)

    # Format events
    events = np.column_stack((filtered, np.zeros(len(filtered), dtype=int), np.full(len(filtered), event_id)))

    if verbose:
        print(f"Detected {len(events)} rising edge events in '{channel_name}'")

    return events

# Process dataset, extracting epochs, evoked responses, and time-frequency representations
def process_dataset(file_path, sens_type):
    raw = mne.io.read_raw_fif(file_path, preload=True)

    if sens_type == 0:
        events = mne.find_events(raw, stim_channel='trigin1', verbose=True)
        picks = mne.pick_channels(raw.info['ch_names'], include=['B_field'])
        reject = dict(mag=4.5e-12)
    else:
        events = detect_ttl_rising_edges(raw, channel_name='ai120', threshold=2.5)
        picks = mne.pick_channels(raw.info['ch_names'], include=['s69_bz'])
        reject = dict(mag=3e-12)

    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=-0.5, tmax=2.0,
        baseline=(-0.5, -0.1), detrend=1, picks=picks, preload=True, reject=reject
    )

    evoked = epochs.average()

    frequencies = np.linspace(10, 30, 100)
    n_cycles = frequencies / 3.0
    tfr = mne.time_frequency.tfr_multitaper(
        evoked, freqs=frequencies, n_cycles=n_cycles,
        time_bandwidth=2.0, return_itc=False, n_jobs=1, average=False
    )
    tfr.apply_baseline(baseline=(-0.5, -0.1), mode='mean')

    return raw, epochs, evoked, tfr, frequencies, picks

# Format results into a dictionary
results = {}
for f in files:
    raw, epochs, evoked, tfr, freqs, picks = process_dataset(f["path"], f["sens_type"])
    results[f["label"]] = {
        "raw": raw, "epochs": epochs, "evoked": evoked, "tfr": tfr,
        "frequencies": freqs, "picks": picks
    }

#%% plot spectra to compare
plt.figure(figsize=(10, 6))
for label, res in results.items():
    raw_data, epochs, picks = res["raw"], res["epochs"], res["picks"]
    sfreq = raw_data.info['sfreq']
    data_before, _ = raw_data[picks]
    n_fft = min(round(10 * sfreq), data_before.shape[1])
    psds_before, freqs = psd_array_welch(data_before, sfreq=sfreq, fmin=0, fmax=100, n_fft=n_fft)
    asd_before = np.sqrt(psds_before)
    plt.plot(freqs, asd_before.T, alpha=0.5, label=f'{label} - Before')

    data_after = epochs.get_data().transpose(1, 0, 2).reshape(len(picks), -1)
    n_fft = min(round(10 * sfreq), data_after.shape[1])
    psds_after, _ = psd_array_welch(data_after, sfreq=sfreq, fmin=0, fmax=100, n_fft=n_fft)
    asd_after = np.sqrt(psds_after)
    plt.plot(freqs, asd_after.T, alpha=0.8, label=f'{label} - After')

plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (T/√Hz)')
plt.title('ASD Comparison Across Sensors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot evoked responses
plt.figure(figsize=(10, 5))
for label, res in results.items():
    evoked = res["evoked"]
    plt.plot(evoked.times, evoked.data[0], label=label)
plt.title('Evoked Response Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (T)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% TFRs, requires custom plotting due to multiple subplots
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)

axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]
cax = plt.subplot(gs[2])  # colourbar axis

for ax, (label, res) in zip(axs, results.items()):
    tfr = res["tfr"]
    evoked = res["evoked"]
    freqs = res["frequencies"]
    power = tfr.data[0].squeeze()
    mesh = ax.pcolormesh(
        evoked.times,
        freqs,
        power,
        shading='auto',
        cmap='RdBu_r',
        vmin=-0.5e-25,
        vmax=2.0e-25
    )
    ax.set_title(f"TFR - {label}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

# Properly attach the colorbar to its own axis
fig.colorbar(mesh, cax=cax, label='Power')

plt.tight_layout()
plt.show()