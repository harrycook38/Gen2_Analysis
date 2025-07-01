import os
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend for interactive plotting
import matplotlib.pyplot as plt
import mne

#%% --- Constants ---
file_name = 'mne_raw_filtered_3-45Hz.fif'

file_location = r'W:\Data\2025_07_01_Sensor_retest\reup_empty_room_000\concat\mne_raw'

fif_fname = os.path.join(file_location, file_name)

sens_type = 0 # 0 for NMOR, 1 for Fieldline, 2 for Fieldline with DiN

#%%

from mne.time_frequency import psd_array_welch

def plot_asd_comparison_epochs(raw_data, epochs, picks, title='ASD Before and After Epoch Rejection', fmax=100):
    sfreq = raw_data.info['sfreq']
    data_before, _ = raw_data[picks]

    # --- Before rejection ---
    n_fft = min(round(10 * sfreq), data_before.shape[1])
    psds_before, freqs = psd_array_welch(
        data_before, sfreq=sfreq, fmin=0, fmax=fmax, n_fft=n_fft
    )
    asd_before = np.sqrt(psds_before)

    # --- After rejection ---
    epochs_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = epochs_data.shape

    # Reshape to (n_channels, n_epochs * n_times) to mimic continuous signal
    data_after = epochs_data.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)
    n_fft = min(round(10 * sfreq), data_after.shape[1])  # update n_fft again
    psds_after, _ = psd_array_welch(
        data_after, sfreq=sfreq, fmin=0, fmax=fmax, n_fft=n_fft
    )
    asd_after = np.sqrt(psds_after)

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, asd_before.T, alpha=0.4, label='Before Rejection')
    plt.plot(freqs, asd_after.T, alpha=0.6, label='After Rejection')
    plt.yscale('log')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (T/√Hz)')
    plt.legend()
    plt.tight_layout()
    plt.show()

raw_filtered = mne.io.read_raw_fif(fif_fname, preload=True)  # Load the filtered raw data

if sens_type == 0:
    events = mne.find_events(raw_filtered, stim_channel='trigin1', verbose=True)
    picks = mne.pick_channels(raw_filtered.info['ch_names'], include=['B_field'])
    reject = dict(mag=4.5e-12)

elif sens_type == 1:
    events = mne.find_events(raw_filtered, stim_channel='ai113', verbose=True, min_duration=0.0005, output='onset', consecutive=True)
    if len(events) > 500:
        # Get the raw signal from the stim channel
        data, times = raw_filtered.copy().pick('ai113').get_data(return_times=True)
        signal = data[0]

        # Smooth and threshold
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(signal, size=100)
        threshold = 0.5
        binary = smoothed > threshold

        # Detect rising edges
        rising = np.where(np.diff(binary.astype(int)) == 1)[0]

        # Debounce
        fs = int(raw_filtered.info['sfreq'])
        min_interval = fs
        clean_rising = [rising[0]]
        for r in rising[1:]:
            if r - clean_rising[-1] > min_interval:
                clean_rising.append(r)
        clean_rising = np.array(clean_rising)

        events = np.column_stack([
            clean_rising,
            np.zeros_like(clean_rising),
            np.ones_like(clean_rising, dtype=int)
        ])
        print(f"Detected {len(clean_rising)} events after cleaning.")

    picks = mne.pick_channels(raw_filtered.info['ch_names'], include=['s69_bz'])
    reject = dict(mag=3.8e-12)

elif sens_type == 2:
    events = mne.find_events(raw_filtered, stim_channel='di32', verbose=True, min_duration=0.0005, output='onset', consecutive=True)
    picks = mne.pick_channels(raw_filtered.info['ch_names'], include=['s69_bz'])
    reject = dict(mag=3.8e-12)


epochs = mne.Epochs(
    raw_filtered, events, event_id=None,  # Use filtered data and event informatio
    tmin=-0.5, tmax=2.,
    baseline=(-0.5, -0.1),
    detrend=1, 
    picks=picks,  # Select the channels for analysis (e.g., 'B_field')
    preload=True,
    verbose=True,
    reject = reject
)
# --- Evoked Response ---
evoked = epochs.average()  # Compute the evoked response
evoked.plot(titles='Evoked Response', time_unit='s', spatial_colors=True)  # Plot the evoked response

plot_asd_comparison_epochs(raw_filtered, epochs, picks=picks, title='ASD Before and After Epoch Rejection', fmax=100)


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
    epochs,
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
plot_tfr(tfr, channel_idx=0,vmin=-2e-24, vmax=2e-24, cmap='RdBu_r')

# %%
