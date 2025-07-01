import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mne

#%% --- Constants ---
# Set the base directory where the data files are stored
base_directory = r'W:\Data\2025_07_01_Sensor_retest\reup_empty_room_000'

# Set the output directory where the processed MNE Raw file will be saved
output_directory = r'W:\Data\2025_07_01_Sensor_retest\reup_empty_room_000\concat\mne_raw'

# Extract the folder name from the path for use in filenames later
folder_name = os.path.basename(base_directory)
output_name = os.path.basename(output_directory)

fif_fname = os.path.join(output_directory, f'{output_name}.fif')

# Scaling factor input, P value of PID on PLL
P1 = 50

# Compute the scaling factor
scal_fac = round(-299 * P1 ** (-0.779), 2)

# Define the digital sampling frequency of the Zbox
sfreq = 837.1

# Define the list of target strings used to import the corresponding csv file
target_strings = ['stream_shift_avg', 'trigin1_avg', 'auxin0_avg']

#%% --- File Loading functions ---
# Function to load data from CSV files in the specified directory
def load_data(base_dir, target_strings):
    # Initialize a dictionary to store 'data' and 'header' for each target file
    extracted_files = {key: {'data': None, 'header': None} for key in target_strings}
    
    # Loop through all files in the base directory
    for f in os.listdir(base_dir):
        # Only process CSV files
        if not f.endswith('.csv'):
            continue
        
        # Generate the full file path for the current file
        file_path = os.path.join(base_dir, f)
        
        # Check if the file matches any of the target strings
        for key in target_strings:
            if key in f:
                # Read the CSV file content using pandas with a semicolon delimiter
                content = pd.read_csv(file_path, sep=';')
                
                # Check if the file is a header or data file and assign accordingly
                if 'header' in f:
                    extracted_files[key]['header'] = content  # Store header info
                else:
                    extracted_files[key]['data'] = content  # Store data info
    
    # Return the dictionary containing the extracted data and header information
    return extracted_files

# --- Scaling and Conversion ---
# Function to scale and convert data from a pandas DataFrame to a numpy array
# Only scale data if a scale factor is provided (to convert shift data to field data)
def scale_and_convert(entry, scale_factor=None):
    data_np = entry['data'].to_numpy(dtype='float64')  # Convert data to NumPy array
    if scale_factor is not None:  # Only scale if a scale factor is provided
        data_np[:, 2] *= scale_factor  # Apply the scale factor to the data in the third column (value data)
    return {'data': data_np, 'header': entry['header']}

# --- Clean Duplicate Timestamps ---
# Function to remove duplicate timestamps from the data array.
# It cleans the timestamp column and adjusts the timestamps by dividing by a divisor and normalises them.
def remove_repd_timestamps(data_array, column_index=1, keep='last', timestamp_divisor=60000000):
    # If the input data is a pandas DataFrame, convert it to a numpy array
    if isinstance(data_array, pd.DataFrame):
        data_array = data_array.to_numpy()
    
    # Extract the column that contains the timestamps (given by 'column_index'), it should be column 1
    col_values = data_array[:, column_index]
    
    # Identify duplicates based on 'last' method
    if keep == 'last':
        # Reverse the column, find unique values and get their indices
        _, idx = np.unique(col_values[::-1], return_index=True)
        # Adjust indices back to original order after reversing
        idx = len(col_values) - 1 - idx
    else:
        # If 'keep' is not 'last', retain the first occurrence of each unique timestamp
        _, idx = np.unique(col_values, return_index=True)
        
    # Sort the indices to restore the data order after removing duplicates
    cleaned = data_array[np.sort(idx)]
    
    # Normalize the timestamps by dividing by the timestamp_divisor
    # This converts the timestamp into seconds, based on the clockrate of the Zbox
    cleaned[:, column_index] /= timestamp_divisor
    
    # Normalize the timestamp to start from 0 by subtracting the first timestamp
    cleaned[:, column_index] -= cleaned[0, column_index]
    
    # Return the cleaned data with unique timestamps
    return cleaned

#%% --- Data Preparation ---

# Load the data from the base directory, extracting relevant files based on the target strings.
# This will return a dictionary with the 'data' and 'header' for each target string.
extracted_files = load_data(base_directory, target_strings)

# Structure the data into a dictionary by folder name, containing 'shift_raw', 'B_field', and 'trigin1', and other targeted files.
# The 'shift_raw' data is extracted as is, and the 'B_field' data is scaled using the scaling factor.
folder_data = {
    folder_name: {
        # The raw shift data from 'stream_shift_avg' is stored here
        'shift_raw': extracted_files['stream_shift_avg'],
        
        # The 'B_field' data is scaled using a specific factor, converting the raw signal to the correct units (e.g., nanoTesla)
        'B_field': scale_and_convert(extracted_files['stream_shift_avg'], scal_fac * 0.071e-9),
        
        # The 'trigin1' data is processed without scaling in this case
        'trigin1': scale_and_convert(extracted_files['trigin1_avg']),

        'auxin0': scale_and_convert(extracted_files['auxin0_avg'])
    }
}

# Iterate through the data entries for this folder (shift_raw, B_field, trigin1)
for entry in folder_data[folder_name].values():
    if 'data' in entry:
        # Clean duplicate timestamps in each data entry by calling the 'remove_repd_timestamps' function
        # This ensures that we have unique and normalized timestamps for each data channel
        entry['data'] = remove_repd_timestamps(entry['data'], column_index=1)


#We have now created a data structure, which can be used as is, or we can convert it to the fif format.

#%% --- Build Raw object ---
def build_raw(data_dict, ch_names, ch_types, sfreq):
    # Stack the data from each channel (B_field, trigin1) into a 2D numpy array.
    # The slicing[:, 2] selects the third column (signal) from each channel's data.
    data = np.vstack([data_dict[ch]['data'][:, 2] for ch in ch_names])
    
    # Create info structure for MNE Raw object. It contains channel names, types, and sampling frequency.
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Return the Raw object with the data and info.
    return mne.io.RawArray(data, info)

# Channel names and types for MNE Raw object creation, we cast our gradiometer to the 'mag' channel for 'b' units
ch_names = ['B_field', 'trigin1','auxin0']
ch_types = ['mag', 'stim','stim'] 

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Build the Raw object using the structured data (folder_data[folder_name])
raw = build_raw(folder_data[folder_name], ch_names, ch_types, sfreq)

# Cast the raw object to disk as a .fif file
raw.save(fif_fname, overwrite=True)
print(f"Saved: {fif_fname}")
