import os
import numpy as np 
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import mne

#%% --- Constants ---
file_name = 'mne_raw_filtered_3-45Hz.fif'

file_location = r'W:\Data\2025_05_29_Motor_and_FL\Us\Tom_motor_1_000\mne_raw'

fif_fname = os.path.join(file_location, file_name)

sens_type = 0 # 0 for NMOR, 1 for Fieldline

#%%
raw_filtered = mne.io.read_raw_fif(fif_fname, preload=True)  # Load the filtered raw data
# Find the events from the stimulus channel ('trigin1'), which is the audio onset
events = mne.find_events(raw_filtered, stim_channel='trigin1', verbose=True)

#%% EMG Triggered TFR
emg_ch = 'auxin0'

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

# Set threshold as 95th percentile or adjust accordingly
threshold = np.percentile(emg_smooth, 95)

# Boolean mask of above-threshold samples
above_thresh = emg_smooth > threshold

# Find rising edges (onsets)
diff = np.diff(above_thresh.astype(int))
onsets = np.where(diff == 1)[0] + 1  # +1 because diff shifts by 1 sample

# Create MNE events array: [sample, 0, event_id]
event_id_emg = 999
events_emg = np.array([[onset, 0, event_id_emg] for onset in onsets])

import matplotlib.pyplot as plt

times = np.arange(len(emg_smooth)) / sfreq

plt.figure(figsize=(15, 4))
plt.plot(times, emg_smooth, label='EMG Envelope (smoothed)')
plt.axhline(threshold, color='r', linestyle='--', label='Threshold')

# Plot vertical lines for detected triggers
for onset in onsets:
    plt.axvline(onset / sfreq, color='g', linestyle=':', alpha=0.7)

plt.xlabel('Time (s)')
plt.ylabel('EMG amplitude (a.u.)')
plt.title('EMG Channel with Detected Muscle Contraction Triggers')
plt.legend()
plt.tight_layout()
plt.show()
# Pick the channels to process
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

# --- Evoked Response ---
evoked = epochs_emg.average()  # Compute the evoked response
evoked.plot(titles='Evoked Response', time_unit='s', spatial_colors=True)  # Plot the evoked response

#%% --- Time-Frequency Representation (TFR) ---
def plot_tfr(tfr, evoked, vmin=-0.5e-25, vmax=2.0e-25, cmap='RdBu_r'):
    # Extract power data for the first condition (tfr.data[0])
    power = tfr.data[0]
    power = power.squeeze()  # Collapse singleton dimensions for correct shape (100, 587)

    # Create a plot for the time-frequency representation
    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(
        evoked.times,
        frequencies,
        power,
        shading='auto',
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )
    plt.colorbar(mesh, label='Power')
    plt.title('TFR (Multitaper)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

# --- Time-frequency analysis parameters ---
frequencies = np.linspace(10, 30, 100)  # Define the frequency range for TFR
n_cycles = frequencies / 3.0  # Set the number of cycles for each frequency
time_bandwidth = 2.0  # Time-bandwidth product for multitaper method

tfr = mne.time_frequency.tfr_multitaper(
    evoked,
    freqs=frequencies,
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth,
    return_itc=False,
    n_jobs=1, 
    average=False,  # Return individual TFR values (no averaging across trials)
)

# Apply baseline correction: using the mean of the pre-stimulus period
tfr.apply_baseline(baseline=(-0.5,-0.1), mode='mean')

# Plot the time-frequency representation
plot_tfr(tfr, evoked)

# %%
# Create epochs for EMG channel using the same events
raw_emg_for_epochs = raw_filtered.copy().pick_channels([emg_ch])

# Optional but recommended: filter again just in case
raw_emg_for_epochs.filter(20, 250, picks=[0], fir_design='firwin')

# Create epochs for EMG using the same event timing
epochs_emg_signal = mne.Epochs(
    raw_emg_for_epochs,
    events_emg,
    event_id={'EMG_contraction': event_id_emg},
    tmin=-0.5,
    tmax=2.0,
    baseline=(-0.5, -0.1),
    detrend=1,
    preload=True,
    verbose=True
)

# Average across trials
evoked_emg_signal = epochs_emg_signal.average()

# Plot trial-averaged EMG waveform
evoked_emg_signal.plot(
    titles='Trial-Averaged EMG Signal',
    time_unit='s',
    spatial_colors=False,  # single channel, so use simple line plot
    show=True
)
# %%
