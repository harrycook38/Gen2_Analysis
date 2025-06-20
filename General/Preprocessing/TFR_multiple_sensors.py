import os
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend for interactive plotting
import matplotlib.pyplot as plt
import mne

mne.set_log_level("WARNING")

from mne.time_frequency import psd_array_welch
import matplotlib.gridspec as gridspec

#%% Set MNE configuration
NMOR_filename = 'mne_raw_filtered_2-45Hz.fif'
NMOR_path = r"W:\Data\2025_05_29_Motor_and_FL\Us\Tom_motor_2_000\mne_raw"

FL_filename = '3-45Hz_20250529_171911_sub-Tom_file-BrainvsUs1wholehand_raw.fif'
FL_path = r"W:\Data\2025_05_29_Motor_and_FL\FL\processed"

files = [
    {
        "label": "NMOR",
        "path": os.path.join(NMOR_path, NMOR_filename),
        "sens_type": 0
    },
    {
        "label": "FieldLine",
        "path": os.path.join(FL_path, FL_filename),
        "sens_type": 1
    }
]

#%% Define functions and process dataset
# Process dataset, extracting epochs, evoked responses, and time-frequency representations
def process_dataset(file_path, sens_type):
    raw = mne.io.read_raw_fif(file_path, preload=True)

    if sens_type == 0:  # NMOR
        events = mne.find_events(raw, stim_channel='trigin1', verbose=False, output='onset', consecutive=True)
        picks = mne.pick_channels(raw.info['ch_names'], include=['B_field'])
        reject = dict(mag=5e-12)
    else:  # FieldLine
        events = mne.find_events(raw, stim_channel='ai120', verbose=False, min_duration=0.0005, output='onset', consecutive=True)
        picks = mne.pick_types(raw.info, meg=True)
        reject = dict(mag=4e-12)

    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=-0.5, tmax=2.0,
        baseline=(-0.5, -0.1), detrend=1, picks=picks, preload=True, reject=reject, verbose=False
    )

    evoked = epochs.average()

    frequencies = np.linspace(10, 30, 100)
    n_cycles = frequencies / 3.0
    tfr = mne.time_frequency.tfr_multitaper(
        evoked, freqs=frequencies, n_cycles=n_cycles,
        time_bandwidth=2.0, return_itc=False, n_jobs=1, average=False, verbose=False
    )
    tfr.apply_baseline(baseline=(-0.5, -0.1), mode='mean')

    # --- Drop stats summary ---
    label = 'Fieldline' if sens_type == 1 else 'NMOR'
    n_total = len(epochs.drop_log)
    n_dropped = sum(len(x) > 0 for x in epochs.drop_log)
    percent_rejected = 100 * n_dropped / n_total if n_total else 0

    print(f"\n{label}: {n_dropped} of {n_total} trials rejected ({percent_rejected:.1f}% rejected overall)")

    # Count how many times each channel caused a drop (may sum to > n_dropped)
    ch_names = [raw.ch_names[i] for i in picks]
    drop_counts = {ch: 0 for ch in ch_names}

    for log_entry in epochs.drop_log:
        for reason in log_entry:
            for ch in ch_names:
                if ch in reason:
                    drop_counts[ch] += 1

    # Only print channels that actually caused a rejection
    contributing_sensors = {ch: count for ch, count in drop_counts.items() if count > 0}
    if contributing_sensors:
        print(f"\n{label} - Drop counts per sensor (note: one epoch may be dropped due to multiple sensors):")
        for ch, count in sorted(contributing_sensors.items()):
            print(f"  {ch}: {count} dropped epochs")

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
plt.figure(figsize=(12, 7))

for label, res in results.items():
    raw_data, epochs, picks = res["raw"], res["epochs"], res["picks"]
    sfreq = raw_data.info['sfreq']
    ch_names = [raw_data.ch_names[p] for p in picks]

    n_sensors = len(picks)
    colors = plt.cm.get_cmap('tab10', n_sensors)  # distinct colors

    # --- Before Rejection ---
    data_before, _ = raw_data[picks]
    n_fft = min(round(10 * sfreq), data_before.shape[1])
    psds_before, freqs = psd_array_welch(data_before, sfreq=sfreq, fmin=0, fmax=100, n_fft=n_fft, verbose=None)
    asd_before = np.sqrt(psds_before)

    for i in range(n_sensors):
        plt.plot(freqs, asd_before[i], alpha=0.5, color=colors(i), linestyle='-', 
                 label=f'{label} {ch_names[i]} Before')

    # --- After Rejection ---
    data_after = epochs.get_data().transpose(1, 0, 2).reshape(n_sensors, -1)
    n_fft = min(round(10 * sfreq), data_after.shape[1])
    psds_after, _ = psd_array_welch(data_after, sfreq=sfreq, fmin=0, fmax=100, n_fft=n_fft)
    asd_after = np.sqrt(psds_after)

    for i in range(n_sensors):
        plt.plot(freqs, asd_after[i], alpha=0.8, color=colors(i), linestyle='--', 
                 label=f'{label} {ch_names[i]} After')

        # Per-sensor mean line (10–48 Hz)
        freq_mask = (freqs >= 10) & (freqs <= 48)
        mean_asd_band = asd_after[i, freq_mask].mean()
        mean_ft = mean_asd_band * 1e15
        plt.axhline(mean_asd_band, linestyle=':', color=colors(i), alpha=0.7,
                    label=f'{label} {ch_names[i]} Mean 10–48 Hz ({mean_ft:.1f} fT/√Hz)')

plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (T/√Hz)')
plt.title('ASD Comparison Across Sensors (Individual with Per-Sensor Mean Lines)')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()

#%% Plot evoked responses (all sensor traces, all labels shown)
plt.figure(figsize=(14, 7))

for label, res in results.items():
    evoked = res["evoked"]
    ch_names = evoked.ch_names

    for i, ch in enumerate(ch_names):
        trace_label = f'{label} - {ch}'
        plt.plot(evoked.times, evoked.data[i] * 1e15, alpha=0.7, label=trace_label)

plt.title('Evoked Responses: All Sensor Traces')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (fT)')
plt.legend(fontsize='small', ncol=2)  # Adjust ncol or fontsize as needed
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
    picks = res["picks"]
    # Average across sensors (axis=0)
    power = tfr.data[picks].mean(axis=0).squeeze()

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

fig.colorbar(mesh, cax=cax, label='Power')
plt.tight_layout()
plt.show()