import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')  # Use TkAgg backend for interactive plotting
import mne
from matplotlib.colors import LogNorm


#%%%

file_name = 'mne_raw.fif'

file_location = r'W:\Data\2025_07_01_Sensor_retest\reup_empty_room_000\concat\mne_raw'

fif_fname = os.path.join(file_location, file_name)

l_freq = 3.0  # Low frequency for bandpass filter
h_freq = 45.0  # High frequency for bandpass filter

generate_filtered_fif = True  # Set to True to generate the filtered .fif file in same directory as the raw .fif file
sens_type = 0  # 0 for NMOR, 1 for Fieldline, 2 for Fieldline with DiN

#%% --- Load + Process Raw in MNE ---
# Load the raw object from the saved .fif file
raw = mne.io.read_raw_fif(fif_fname, preload=True)

sfreq = raw.info['sfreq'] 

# Find the events from the stimulus channel ('trigin1'), which is the audio onset
if sens_type == 0:
    events = mne.find_events(raw, stim_channel='trigin1', verbose=True)
    picks = mne.pick_channels(raw.info['ch_names'], include=['B_field'])
if sens_type == 1:
    events = mne.find_events(raw, stim_channel='ai113', verbose=True, min_duration=0.0005,   output='onset', consecutive=True)
    picks = mne.pick_channels(raw.info['ch_names'], include=['s69_bz'])
if sens_type == 2:
    events = mne.find_events(raw, stim_channel='di32', verbose=True, min_duration=0.0005, output='onset', consecutive=True)
    picks = mne.pick_channels(raw.info['ch_names'], include=['s69_bz'])
# Pick the channels to process]

# Copy raw data to avoid modifying it
raw_filtered = raw.copy()
# Apply Notch Filter to raw_filtered (but not raw)
raw_filtered.notch_filter(freqs=[50, 100], picks=picks, filter_length='auto', trans_bandwidth=8, verbose=True)
# Apply Butterworth Bandpass Filter to the same raw_filtered data
raw_filtered.filter(
    l_freq=l_freq, h_freq=h_freq, method='iir',
    iir_params=dict(order=4, ftype='butter')
)
#%% --- Spectrogram ---
#I want to create a time-dependent spectrogram of the data, to see if we get any 'walking' noise
def plot_spectrogram(raw_data, sfreq):
    # Import the scipy function for computing the spectrogram
    from scipy.signal import spectrogram
    
    # Compute the spectrogram: f - frequencies, t - time, Sxx - power spectral density
    f, t, Sxx = spectrogram(raw_data, fs=sfreq, nperseg=2048, noverlap=1024)

    # Convert to Amplitude Spectral Density (sqrt of PSD)
    ASD = np.sqrt(Sxx)  # Convert to picotesla/√Hz
    
    # Plot ASD directly (linear scale, not dB)
    plt.figure(figsize=(12, 5))
    mesh = plt.pcolormesh(t, f, ASD, shading='gouraud', norm=LogNorm(vmin=ASD[ASD > 0].min(), vmax=ASD.max()))
    plt.ylim(0, 100)
    plt.title('Spectrogram (Log-scaled ASD)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(mesh, label='Amplitude (T/√Hz)')
    plt.tight_layout()
    plt.show()

# Call the spectrogram function for the 'B_field' channel of the raw data
if sens_type == 0:
    plot_spectrogram(raw.get_data(picks='B_field')[0], sfreq)
if sens_type == 1:
    plot_spectrogram(raw.get_data(picks='s69_bz')[0], sfreq)
#%% --- PSD plots ---
from mne.time_frequency import psd_array_welch

def plot_asd(raw_data, title, fmax=100):
    # Get data and sampling frequency
    data, _ = raw_data[picks]
    sfreq = raw_data.info['sfreq']

    # Compute PSD using multitaper
    psds, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=0,
        fmax=100,
        n_fft=round(10 * sfreq),    
        )

    # Convert PSD to ASD
    asd = np.sqrt(psds)

    # Plot ASD
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, asd.T, alpha=0.5)
    plt.yscale('log')
    plt.grid(True)
    plt.title(f'{title} B_field - Amplitude Spectral Density (Welch)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (T/√Hz)')
    plt.tight_layout()
    plt.show()

# Generate PSD plots for both the unfiltered and filtered data``
for title, dat in zip(["Unfiltered", f"Filtered"], [raw, raw_filtered]):
    plot_asd(dat, title)

# --- Timecourse Comparison ---
def plot_timecourse(unf, filt, times):
    plt.figure(figsize=(15, 5))

    # Plot unfiltered traces for all channels
    for i in range(unf.shape[0]):
        plt.plot(times, unf[i] / 1e-12, label=f'Unfiltered Ch{i+1}', alpha=0.5)

    # Plot filtered traces for all channels
    for i in range(filt.shape[0]):
        plt.plot(times, filt[i] / 1e-12, label=f'Filtered Ch{i+1}', alpha=0.8)

    plt.title('Timecourses for All Channels')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (pT)')
    plt.grid(True)
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.show()

# Extract the data for the unfiltered and filtered timecourses and plot them
unf, times = raw[picks]
filt, _ = raw_filtered[picks]
plot_timecourse(unf, filt, times)

#%% --- Save the filtered raw data ---
if generate_filtered_fif is True:
    # Build a filename with filtering info
    if sens_type == 0:
        filtered_file_name = f'mne_raw_filtered_{int(l_freq)}-{int(h_freq)}Hz.fif'
        filtered_fif_path = os.path.join(file_location, filtered_file_name)
    if sens_type in (1,2):
        filtered_file_name = f'{int(l_freq)}-{int(h_freq)}Hz_' + file_name
        filtered_fif_path = os.path.join(file_location+'\processed', filtered_file_name)

    # Save the filtered Raw object
    raw_filtered.save(filtered_fif_path, overwrite=True)
    print(f"Filtered data saved to: {filtered_fif_path}")
# %%


