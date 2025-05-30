import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mne

#%% --- Constants ---
file_name = 'mne_raw_filtered_3-45Hz.fif'

file_location = r'W:\Data\2025_05_29_Motor_and_FL\Us\Tom_motor_1_000\mne_raw'

fif_fname = os.path.join(file_location, file_name)
#%%
raw_filtered = mne.io.read_raw_fif(fif_fname, preload=True)  # Load the filtered raw data
# Find the events from the stimulus channel ('trigin1'), which is the audio onset
events = mne.find_events(raw_filtered, stim_channel='trigin1', verbose=True)

# Pick the channels to process
picks = mne.pick_types(raw_filtered.info, meg='mag')

reject = dict(mag=4e-12)  # Define rejection criteria for the magnetometer channel
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
#.filter(5., 35., method='iir', iir_params=dict(order=4, ftype='butter'))  # Apply narrower band filter

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
