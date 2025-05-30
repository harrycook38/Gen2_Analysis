import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mne


#%%%

file_name = 'mne_raw.fif'

file_location = r'W:\Data\2025_05_29_Motor_and_FL\Us\Tom_motor_1_000\mne_raw'

fif_fname = os.path.join(file_location, file_name)

l_freq = 3.0  # Low frequency for bandpass filter
h_freq = 45.0  # High frequency for bandpass filter

generate_filtered_fif = True  # Set to True to generate the filtered .fif file in same directory as the raw .fif file


#%% --- Load + Process Raw in MNE ---
# Load the raw object from the saved .fif file
raw = mne.io.read_raw_fif(fif_fname, preload=True)

sfreq = raw.info['sfreq'] 

# Find the events from the stimulus channel ('trigin1'), which is the audio onset
events = mne.find_events(raw, stim_channel='trigin1', verbose=True)

# Pick the channels to process
picks = mne.pick_channels(raw.info['ch_names'], include=['B_field'])

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
    
    # Plot the spectrogram using pcolormesh to create a 2D heatmap of the power spectrum
    plt.figure(figsize=(12, 5)) 
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')  # Convert power to dB scale and plot
    plt.ylim(0, 100)
    plt.title('Spectrogram')  
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')  
    plt.colorbar(label='Power (dB)') 
    plt.tight_layout()  
    plt.show()

# Call the spectrogram function for the 'B_field' channel of the raw data
plot_spectrogram(raw.get_data(picks='B_field')[0], sfreq)

#%% --- PSD plots ---
def plot_psd(raw_data, title):
    # Create a figure for the Power Spectral Density (PSD) plot
    plt.figure(figsize=(10, 5))
    raw_data.plot_psd(picks=picks, fmax=100, average=True, spatial_colors=False, dB=False)
    plt.yscale('log') 
    plt.grid(True) 
    plt.title(f'{title} B_field - Amplitude Spectral Density') 

# Generate PSD plots for both the unfiltered and filtered data
for title, dat in zip(["Unfiltered", f"Filtered"], [raw, raw_filtered]):
    plot_psd(dat, title)

# --- Timecourse Comparison ---
def plot_timecourse(unf, filt, times):
    # Create a figure for plotting the timecourse comparison
    plt.figure(figsize=(15, 4))
    plt.plot(times, unf.flatten(), label='Unfiltered', alpha=0.7)  # Plot the unfiltered data
    plt.plot(times, filt.flatten(), label='Filtered', alpha=0.7)  # Plot the filtered data
    plt.title('Timecourse of B_field') 
    plt.xlabel('Time (s)') 
    plt.ylabel('Amplitude')
    plt.legend() 
    plt.grid(True)  
    plt.tight_layout()
    plt.show() 

# Extract the data for the unfiltered and filtered timecourses and plot them
unf, times = raw[picks]  # Unfiltered data for the 'B_field' channel
filt, _ = raw_filtered[picks]  # Filtered data for the 'B_field' channel
plot_timecourse(unf, filt, times)  # Plot the comparison of the unfiltered and filtered timecourses

#%% --- Save the filtered raw data ---
if generate_filtered_fif is True:
    # Build a filename with filtering info
    filtered_file_name = f'mne_raw_filtered_{int(l_freq)}-{int(h_freq)}Hz.fif'
    filtered_fif_path = os.path.join(file_location, filtered_file_name)

    # Save the filtered Raw object
    raw_filtered.save(filtered_fif_path, overwrite=True)
    print(f"Filtered data saved to: {filtered_fif_path}")
# %%
