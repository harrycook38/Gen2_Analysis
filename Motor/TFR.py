import os
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend for interactive plotting
import matplotlib.pyplot as plt
import mne

#%% --- Constants ---
file_name = '3-45Hz_20250529_165519_sub-Tom_file-BrainvsUs1_raw.fif'

file_location = r'W:\Data\2025_05_29_Motor_and_FL\FL\processed'

fif_fname = os.path.join(file_location, file_name)

sens_type = 1 # 0 for NMOR, 1 for Fieldline



#%%

#MNE doesn't like aux triggers sometimes, so we need to help it out with an edge-detection function

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


raw_filtered = mne.io.read_raw_fif(fif_fname, preload=True)  # Load the filtered raw data


if sens_type == 0:
    events = mne.find_events(raw_filtered, stim_channel='trigin1', verbose=True)
    picks = mne.pick_channels(raw_filtered.info['ch_names'], include=['B_field'])
if sens_type == 1:

    events = detect_ttl_rising_edges(raw_filtered, channel_name='ai120', threshold=2.5)
    picks = mne.pick_channels(raw_filtered.info['ch_names'], include=['s69_bz'])

# Pick the channels to process

reject = dict(mag=5e-12)  # Define rejection criteria for the magnetometer channel
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
