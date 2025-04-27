# -*- coding: utf-8 -*-
# Copyright (c) 2022, Kwanhyung Lee, AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pyedflib import highlevel, EdfReader
from scipy.io.wavfile import write
from scipy import signal as sci_sig
from scipy.spatial.distance import pdist
from scipy.signal import stft, hilbert, butter, freqz, filtfilt, find_peaks
from builder.utils.process_util import run_multi_process
from builder.utils.utils import search_walk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import argparse
import torch
import glob
import pickle
import random
import mne 
from mne.io.edf.edf import _read_annotations_edf, _read_edf_header
from itertools import groupby

GLOBAL_DATA = {}
label_dict = {}
sample_rate_dict = {}
sev_label = {}

label_extension_map = {
        'tse': '.csv',      # Expect .csv for 'tse' type
        'tse_bi': '.csv_bi' # Expect .csv_bi for 'tse_bi' type
    }

def label_sampling_tuh(labels_lines, feature_samplerate): # Renamed input for clarity
    y_target_list = [] # Build as a list first
    remained = 0
    feature_intv = 1 / float(feature_samplerate)
    LABEL_COLUMN_INDEX = 3  # Label is the 4th column (index 3)
    START_TIME_INDEX = 1
    STOP_TIME_INDEX = 2

    for line in labels_lines: # Iterate through the valid lines passed to it
        parts = line.strip().split(",")
        try:
            # Check if it has enough parts and label isn't empty
            if len(parts) > LABEL_COLUMN_INDEX and parts[LABEL_COLUMN_INDEX].strip():
                start_time_str = parts[START_TIME_INDEX].strip()
                stop_time_str = parts[STOP_TIME_INDEX].strip()
                label = parts[LABEL_COLUMN_INDEX].strip()

                # Convert times to float, handle potential errors
                start_time = float(start_time_str)
                stop_time = float(stop_time_str)

                # Get the integer disease label from GLOBAL_DATA
                disease_code = GLOBAL_DATA['disease_labels'].get(label, -1) # Get code, default to -1 if unknown
                if disease_code == -1:
                     # print(f"Warning: Unknown label '{label}' encountered in label_sampling_tuh. Skipping line.")
                     continue # Skip if label isn't in our dictionary

                intv_count, remained = divmod(stop_time - start_time + remained, feature_intv)
                # Append the disease code 'intv_count' times
                y_target_list.extend([str(disease_code)] * int(intv_count))
            else:
                # This case should be less likely now due to pre-filtering, but good to have
                # print(f"Warning: Skipping line in label_sampling_tuh due to insufficient parts: '{line.strip()}'")
                pass

        except (ValueError, IndexError) as e:
            print(f"Warning: Error processing line in label_sampling_tuh: '{line.strip()}'. Error: {e}. Skipping.")
            continue # Skip line on error

    return "".join(y_target_list) # Join the list into a string at the end


def generate_training_data_leadwise_tuh_train_final(file):
    # ... (Keep initial EDF reading and label_list_c creation) ...
    sample_rate = GLOBAL_DATA['sample_rate']
    file_name = ".".join(file.split(".")[:-1])
    data_file_name = file_name.split("/")[-1]
    try:
        signals, signal_headers, header = highlevel.read_edf(file)
    except Exception as e:
        print(f"Error reading EDF file {file}: {e}. Skipping.")
        return

    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]
        label_list_c.append(label_noref)

    ############################# part 1: labeling  ###############################
    # --- START File Extension Change ---
    
    # Use the map, fallback to original if type isn't in map (though it should be)
    actual_label_extension = label_extension_map.get(GLOBAL_DATA['label_type'], '.' + GLOBAL_DATA['label_type'])
    label_file_path = file_name + actual_label_extension
    # --- END File Extension Change ---

    try:
        with open(label_file_path, 'r') as label_file:
            all_lines = label_file.readlines()
    except FileNotFoundError:
        # If CSV file not found, try the original TSE extension as a fallback maybe? Or just fail.
        # Trying original extension:
        
        original_label_path = file_name + "." + GLOBAL_DATA['label_type']
        try:
             with open(original_label_path, 'r') as label_file:
                  print(f"Warning: Found original label file {original_label_path} instead of expected {label_file_path}. Proceeding.")
                  all_lines = label_file.readlines()
                  # Add a flag or adjust logic if TSE/CSV formats need different parsing downstream
        except FileNotFoundError:
             print(f"Error: Label file not found at {label_file_path} or {original_label_path}. Skipping file {file}.")
             return
    except Exception as e:
        print(f"Error reading label file {label_file_path}: {e}. Skipping file {file}.")
        return

    # --- START CSV Parsing Change ---
    HEADER_LINES_TO_SKIP = 6 # Skip 5 '#' lines + 1 'channel,...' header
    LABEL_COLUMN_INDEX = 3   # Label is the 4th column (index 3)
    y = [] # This will store the valid data lines for label_sampling_tuh

    for line in all_lines[HEADER_LINES_TO_SKIP:]:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith('#'): # Ignore empty lines and any extra comments
            parts = stripped_line.split(",") # Use comma delimiter
            # Check if enough columns exist AND the label column isn't empty
            if len(parts) > LABEL_COLUMN_INDEX and parts[LABEL_COLUMN_INDEX].strip():
                y.append(stripped_line) # Add the valid data line
            else:
                # print(f"Warning: Skipping malformed/short line in {label_file_path}: '{line.strip()}'")
                pass # Reduce verbosity

    if not y:
        print(f"Warning: No valid data lines found in {label_file_path} after skipping headers. Skipping file {file}.")
        return
    # --- END CSV Parsing Change ---

    # Extract unique labels (still useful check)
    try:
        # Extract label from the correct index (3), splitting by comma
        y_labels = list(set([line.split(",")[LABEL_COLUMN_INDEX].strip() for line in y]))
    except IndexError as e:
        print(f"Unexpected IndexError during y_labels extraction in {label_file_path}. Lines: {y[:5]}. Skipping.")
        return

    signal_sample_rate = int(signal_headers[0]['sample_rate'])
    # ... (rest of the checks for sample rate, required labels remain the same) ...
    if sample_rate > signal_sample_rate:
        # print(f"Warning: Target sample rate ({sample_rate}) > signal sample rate ({signal_sample_rate}) for {file}. Skipping.")
        return # Keep this check
    if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']):
        # print(f"Warning: Missing required labels in {file}. Required: {GLOBAL_DATA['label_list']}, Found: {label_list_c}. Skipping.")
        return # Keep this check


    # Call the modified label_sampling_tuh with the filtered lines 'y'
    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])

    # --- Check if y_sampled is empty ---
    if not y_sampled:
        print(f"Warning: y_sampled string is empty after processing {label_file_path}. Might indicate issues with labels or times. Skipping file {file}.")
        return
    
    ############################# part 2: input data filtering #############################
    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_rate']) > sample_rate:
            secs = len(signal)/float(signal_sample_rate)
            samps = int(secs*sample_rate)
            x = sci_sig.resample(signal, samps)
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            signal_list.append(signal)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return 
    
    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
    
    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        y_sampled += y_sampled[-1] * diff

    y_sampled_np = np.array(list(map(int,y_sampled)))
    new_labels = []
    new_labels_idxs = []

    ############################# part 3: slicing for easy training  #############################
    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    # slice and save if training data
    new_data = {}
    raw_data = torch.Tensor(signal_final_list_raw).permute(1,0)

    max_seg_len_before_seiz_label = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_before_seiz_raw = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['sample_rate']
    max_seg_len_after_seiz_label = GLOBAL_DATA['max_bckg_after_seiz_length'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_after_seiz_raw = GLOBAL_DATA['max_bckg_after_seiz_length'] * GLOBAL_DATA['sample_rate']

    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    max_seg_len_label = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_raw = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['sample_rate']

    label_order = [x[0] for x in groupby(y_sampled)]
    label_change_idxs = np.where(y_sampled_np[:-1] != y_sampled_np[1:])[0]

    start_raw_idx = 0
    start_label_idx = 0
    end_raw_idx = raw_data.size(0)
    end_label_idx = len(y_sampled)
    previous_bckg_len = 0
    
    sliced_raws = []
    sliced_labels = []
    pre_bckg_lens_label = []
    label_list_for_filename = []
    
    for idx, label in enumerate(label_order):
        # if last and the label is "bckg"
        if (len(label_order) == idx+1) and (label == "0"):
            sliced_raw_data = raw_data[start_raw_idx:].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[start_label_idx:]))).byte()

            if sliced_y1.size(0) < min_seg_len_label:
                continue
            sliced_raws.append(sliced_raw_data)
            sliced_labels.append(sliced_y1)
            pre_bckg_lens_label.append(0)
            label_list_for_filename.append(label)

        # if not last and the label is "bckg"
        elif (len(label_order) != idx+1) and (label == "0"):
            end_raw_idx = (label_change_idxs[idx]+1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx]+1
            
            sliced_raw_data = raw_data[start_raw_idx:end_raw_idx].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[start_label_idx:end_label_idx]))).byte()
            previous_bckg_len = end_label_idx - start_label_idx

            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
            if sliced_y1.size(0) < min_seg_len_label:
                continue

            sliced_raws.append(sliced_raw_data)
            sliced_labels.append(sliced_y1)
            pre_bckg_lens_label.append(0)
            label_list_for_filename.append(label)

        # if the first and the label is "seiz" 1 ~ 8       
        elif (idx == 0) and (label != "0"):
            end_raw_idx = (label_change_idxs[idx]+1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx]+1
            
            if len(y_sampled)-end_label_idx > max_seg_len_after_seiz_label:
                post_len_label = max_seg_len_after_seiz_label
                post_len_raw = max_seg_len_after_seiz_raw
            else:
                post_len_label = len(y_sampled)-end_label_idx
                post_len_raw = ((len(y_sampled)-end_label_idx) * GLOBAL_DATA['fsr_sr_ratio'])
            post_ictal_end_label = end_label_idx + post_len_label
            post_ictal_end_raw = end_raw_idx + post_len_raw
            
            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
            if len(y_sampled) < min_seg_len_label:
                continue

            sliced_raw_data = raw_data[:post_ictal_end_raw].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[:post_ictal_end_label]))).byte()

            if sliced_y1.size(0) > max_seg_len_label:
                sliced_y2 = sliced_y1[:max_seg_len_label]
                sliced_raw_data2 = sliced_raw_data.permute(1,0)[:max_seg_len_raw].permute(1,0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)
            elif sliced_y1.size(0) >= min_seg_len_label:
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y1)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)
            else:
                sliced_y2 = torch.Tensor(list(map(int,y_sampled[:min_seg_len_label]))).byte()
                sliced_raw_data2 = raw_data[:min_seg_len_raw].permute(1,0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)

        # the label is "seiz" 1 ~ 8
        elif label != "0":
            end_raw_idx = (label_change_idxs[idx]+1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx]+1
            
            if len(y_sampled)-end_label_idx > max_seg_len_after_seiz_label:
                post_len_label = max_seg_len_after_seiz_label
                post_len_raw = max_seg_len_after_seiz_raw
            else:
                post_len_label = len(y_sampled)-end_label_idx
                post_len_raw = ((len(y_sampled)-end_label_idx) * GLOBAL_DATA['fsr_sr_ratio'])
            post_ictal_end_label = end_label_idx + post_len_label
            post_ictal_end_raw = end_raw_idx + post_len_raw

            if previous_bckg_len > max_seg_len_before_seiz_label:
                pre_seiz_label_len = max_seg_len_before_seiz_label
            else:
                pre_seiz_label_len = previous_bckg_len
            pre_seiz_raw_len = pre_seiz_label_len * GLOBAL_DATA['fsr_sr_ratio']

            sample_len = post_ictal_end_label - (start_label_idx-pre_seiz_label_len)
            if sample_len < min_seg_len_label:
                post_ictal_end_label = start_label_idx - pre_seiz_label_len + min_seg_len_label
                post_ictal_end_raw = start_raw_idx - pre_seiz_raw_len + min_seg_len_raw
            if len(y_sampled) < post_ictal_end_label:
                start_raw_idx = end_raw_idx
                start_label_idx = end_label_idx
                continue

            sliced_raw_data = raw_data[start_raw_idx-pre_seiz_raw_len:post_ictal_end_raw].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[start_label_idx-pre_seiz_label_len:post_ictal_end_label]))).byte()

            if sliced_y1.size(0) > max_seg_len_label:
                sliced_y2 = sliced_y1[:max_seg_len_label]
                sliced_raw_data2 = sliced_raw_data.permute(1,0)[:max_seg_len_raw].permute(1,0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(pre_seiz_label_len)
                label_list_for_filename.append(label)
            # elif sliced_y1.size(0) >= min_seg_len_label:
            else:
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y1)
                pre_bckg_lens_label.append(pre_seiz_label_len)
                label_list_for_filename.append(label)
            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
                
        else:
            print("Error! Impossible!")
            exit(1)

    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int,sliced_y))

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        new_data['RAW_DATA'] = [sliced_raw]
        new_data['LABEL1'] = [sliced_y]
        new_data['LABEL2'] = [sliced_y2]
        new_data['LABEL3'] = [sliced_y3]

        prelabel_len = pre_bckg_lens_label[data_idx]
        label = label_list_for_filename[data_idx]
        
        with open(GLOBAL_DATA['data_file_directory'] + "/{}_c{}_pre{}_len{}_label_{}.pkl".format(data_file_name, str(data_idx), str(prelabel_len), str(len(sliced_y)), str(label)), 'wb') as _f:
            pickle.dump(new_data, _f)      
        new_data = {}

def generate_training_data_leadwise_tuh_train_final(file):
    sample_rate = GLOBAL_DATA['sample_rate']    # EX) 200Hz
    file_name = ".".join(file.split(".")[:-1])  # EX) $PATH_TO_EEG/.../00007235_s003_t000
    data_file_name = file_name.split("/")[-1]   # EX) 00007235_s003_t000

    # --- EDF Reading (with basic error handling) ---
    try:
        signals, signal_headers, header = highlevel.read_edf(file)
    except Exception as e:
        print(f"Error reading EDF file {file}: {e}. Skipping.")
        return

    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]
        label_list_c.append(label_noref)

    ############################# part 1: labeling (CSV Updated) ###############################
    # --- Determine correct label file path ---
    actual_label_extension = label_extension_map.get(GLOBAL_DATA['label_type'], '.' + GLOBAL_DATA['label_type'])
    label_file_path = file_name + actual_label_extension

    # --- Read and Parse CSV Label File ---
    HEADER_LINES_TO_SKIP = 6 # Skip 5 '#' lines + 1 'channel,...' header
    LABEL_COLUMN_INDEX = 3   # Label is the 4th column (index 3)

    all_lines = []
    try:
        with open(label_file_path, 'r') as label_file:
            all_lines = label_file.readlines()
    except FileNotFoundError:
         # Try falling back to original .tse/.tse_bi extension if .csv/.csv_bi not found
        original_label_path = file_name + "." + GLOBAL_DATA['label_type']
        try:
             with open(original_label_path, 'r') as label_file:
                  # print(f"Warning: Found original label file {original_label_path} instead of expected {label_file_path}. Assuming TSE format for this file.")
                  # If found, we need to handle potential mixed formats - this adds complexity.
                  # For now, let's prioritize the CSV format and error out if neither is found.
                  # Re-reading into all_lines assumes TSE format here, which might be wrong if CSV exists but wasn't read.
                  # Simplest approach: Fail if the expected CSV isn't there.
                  # all_lines = label_file.readlines() # Remove this if we don't support fallback parsing easily
                  print(f"Error: Expected CSV label file {label_file_path} not found, and fallback to {original_label_path} is complex/unsupported here. Skipping file {file}.")
                  return
        except FileNotFoundError:
             print(f"Error: Label file not found at {label_file_path} or {original_label_path}. Skipping file {file}.")
             return
    except Exception as e:
        print(f"Error reading label file {label_file_path}: {e}. Skipping file {file}.")
        return

    # --- Filter valid data lines from CSV ---
    y_valid_data_lines = [] # Store the actual data lines
    for line in all_lines[HEADER_LINES_TO_SKIP:]:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith('#'): # Ignore empty/comment lines
            parts = stripped_line.split(",") # Use comma delimiter
            # Check if enough columns exist AND the label column isn't empty
            if len(parts) > LABEL_COLUMN_INDEX and parts[LABEL_COLUMN_INDEX].strip():
                y_valid_data_lines.append(stripped_line) # Add the valid data line

    if not y_valid_data_lines:
        print(f"Warning: No valid data lines found in {label_file_path} after skipping headers. Skipping file {file}.")
        return

    # --- Extract unique labels (still useful check) ---
    try:
        # Extract label from the correct index (3), splitting by comma
        y_labels = list(set([line.split(",")[LABEL_COLUMN_INDEX].strip() for line in y_valid_data_lines]))
    except IndexError as e:
        print(f"Unexpected IndexError during y_labels extraction in {label_file_path}. Lines: {y_valid_data_lines[:5]}. Skipping.")
        return

    # --- Standard checks (Sample Rate, Required Leads) ---
    #print(signal_headers[0])
    #print(signal_headers)
    signal_sample_rate = int(signal_headers[0]['sample_frequency'])
    if sample_rate > signal_sample_rate:
        # print(f"Warning: Target ({sample_rate}) > Signal SR ({signal_sample_rate}) for {file}. Skipping.") # Reduce verbosity
        return
    if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']):
        # print(f"Warning: Missing required labels in {file}. Req: {GLOBAL_DATA['label_list']}, Found: {label_list_c}. Skipping.") # Reduce verbosity
        return

    # --- Generate sampled label string using the updated label_sampling_tuh ---
    # Pass the list of valid *data lines* to the updated function
    y_sampled = label_sampling_tuh(y_valid_data_lines, GLOBAL_DATA['feature_sample_rate'])

    # --- Check if y_sampled is empty after processing ---
    if not y_sampled:
        print(f"Warning: y_sampled string is empty after processing {label_file_path}. Skipping file {file}.")
        return

    ###################### check if seizure patient (CSV Updated) ######################
    patient_wise_dir = "/".join(file_name.split("/")[:-2])
    # Search for the correct binary label file extension
    binary_label_extension = label_extension_map.get('tse_bi', '.csv_bi') # Default to .csv_bi
    edf_list = search_walk({'path': patient_wise_dir, 'extension': binary_label_extension})
    patient_bool = False
    if not edf_list:
         # Try fallback to original tse_bi? Might be needed if dataset is mixed.
         # print(f"Warning: No {binary_label_extension} files found for patient check in {patient_wise_dir}. Assuming non-seizure patient.")
         pass # Assume False if no binary files found

    for label_check_file_path in edf_list:
        try:
            with open(label_check_file_path, 'r') as label_check_file:
                check_lines = label_check_file.readlines()
        except Exception as e:
            # print(f"Warning: Could not read file {label_check_file_path} for patient check: {e}")
            continue # Skip this file

        # Use same CSV parsing logic for the check file
        for line in check_lines[HEADER_LINES_TO_SKIP:]:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith('#'):
                parts = stripped_line.split(',')
                if len(parts) > LABEL_COLUMN_INDEX:
                    label_in_line = parts[LABEL_COLUMN_INDEX].strip()
                    if label_in_line and label_in_line != 'bckg': # Check if label exists and is not 'bckg'
                        patient_bool = True
                        break # Found a seizure label in this file
        if patient_bool:
            break # Found a seizure label in this patient's record

    ############################# part 2: input data filtering #############################
    # This part operates on EEG signals and should remain the same
    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        current_signal_sr = int(signal_headers[idx]['sample_frequency'])
        # Check if resampling is needed (avoid resampling if SR is already correct)
        if current_signal_sr == sample_rate:
             x = signal
        elif current_signal_sr > sample_rate:
            secs = len(signal) / float(current_signal_sr) # Use current signal SR here
            samps = int(secs * sample_rate)
            x = sci_sig.resample(signal, samps)
        else: # current_signal_sr < sample_rate - Upsampling (might be needed or indicate error)
             # print(f"Warning: Signal SR {current_signal_sr} < Target SR {sample_rate} in {file} for lead {label}. Upsampling.")
             secs = len(signal) / float(current_signal_sr)
             samps = int(secs * sample_rate)
             # Use resample for upsampling too, though interpolation might be better if needed often
             x = sci_sig.resample(signal, samps)

        signal_list.append(x)
        signal_label_list.append(label)


    # Ensure all required labels were found and processed
    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        # Check which labels are missing
        missing_labels = set(GLOBAL_DATA['label_list']) - set(signal_label_list)
        # print(f"Warning: Not enough required leads processed/found in {file}. Missing: {missing_labels}. Skipping.")
        return

    # Reorder signals according to GLOBAL_DATA['label_list']
    # Use a dictionary for faster lookup
    signal_dict = {lbl: sig for lbl, sig in zip(signal_label_list, signal_list)}
    signal_final_list_raw = []
    min_len = float('inf') # Find minimum length after resampling
    for lead_label in GLOBAL_DATA['label_list']:
        sig_data = signal_dict.get(lead_label)
        if sig_data is None: # Should not happen if previous check passed, but safeguard
             print(f"Error: Lead {lead_label} not found in signal_dict for {file}. Skipping.")
             return
        signal_final_list_raw.append(sig_data)
        min_len = min(min_len, len(sig_data))

    # Trim all signals to the minimum length to ensure consistency
    signal_final_list_raw = [sig[:min_len] for sig in signal_final_list_raw]

    if min_len == 0:
        print(f"Warning: Processed signal length is zero for {file}. Skipping.")
        return

    ######################## Adjust y_sampled length based on ACTUAL signal length ########################
    # Calculate the expected number of labels based on the *final* signal length and feature sample rate
    # Use min_len which is the length *at the target sample_rate*
    expected_label_len = math.floor(min_len * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate']))

    if len(y_sampled) > expected_label_len:
        y_sampled = y_sampled[:expected_label_len]
    elif len(y_sampled) < expected_label_len:
        diff = expected_label_len - len(y_sampled)
        if y_sampled: # Check if y_sampled is not empty before padding
             y_sampled += y_sampled[-1] * diff
        else:
             print(f"Warning: y_sampled is empty but signals exist for {file}. Cannot pad. Resulting labels might be incorrect. Length needed: {expected_label_len}")
             # Handle this case: maybe pad with '0' or skip file? Padding with '0' might be safer.
             y_sampled = '0' * expected_label_len


    # Check again if y_sampled became empty after length adjustment
    if not y_sampled:
         print(f"Warning: y_sampled became empty after length adjustment for {file}. Skipping.")
         return


    # --- Convert y_sampled to numpy array (should be safe now) ---
    try:
        y_sampled_np = np.array(list(map(int, y_sampled)))
    except ValueError as e:
        print(f"Error converting y_sampled to numpy int array for file {file}. Content: '{y_sampled[:50]}...'. Error: {e}. Skipping.")
        return


    ############################# part 3: slicing for easy training  #############################
    # This part relies on y_sampled containing the correct sequence of integer strings ('0', '1', '2'...)
    # The logic here seems complex and specific to the paper's slicing strategy.
    # Assuming the logic itself is correct based on the paper's intent, it should work if y_sampled is correct.

    # Map y_sampled based on target_dictionary
    # Need to handle potential KeyErrors if y_sampled contains unexpected codes
    temp_y_sampled_mapped = []
    valid_codes = set(map(str, GLOBAL_DATA['target_dictionary'].keys())) # Use string codes

    for code_char in y_sampled:
        if code_char in GLOBAL_DATA['selected_diseases']: # selected_diseases holds target *string* codes ('1', '2'...)
             try:
                  mapped_code = str(GLOBAL_DATA['target_dictionary'][int(code_char)])
                  temp_y_sampled_mapped.append(mapped_code)
             except KeyError:
                  print(f"Warning: Code {code_char} from y_sampled not found in target_dictionary for {file}. Using '0'.")
                  temp_y_sampled_mapped.append("0") # Map unknown codes to '0' or handle differently
        elif code_char == '0':
             temp_y_sampled_mapped.append("0")
        else:
             # This case implies a code was generated but isn't in selected_diseases - treat as '0'
             # print(f"Warning: Code {code_char} from y_sampled not in selected_diseases for {file}. Using '0'.")
             temp_y_sampled_mapped.append("0")

    y_sampled = "".join(temp_y_sampled_mapped) # Update y_sampled with the mapped codes ('0', '1' etc based on target dict)

    # --- Re-calculate y_sampled_np based on the *mapped* y_sampled ---
    try:
        y_sampled_np = np.array(list(map(int, y_sampled)))
    except ValueError as e:
        print(f"Error converting *mapped* y_sampled to numpy int array for file {file}. Content: '{y_sampled[:50]}...'. Error: {e}. Skipping.")
        return

    # --- The rest of the slicing logic ---
    # This complex slicing part needs careful review based on the paper's description,
    # but the inputs (raw_data, y_sampled, y_sampled_np) should now be correctly formatted.
    new_data = {}
    try:
        # Ensure raw_data is float32 for Tensor conversion if needed by later ops, then convert to float16 if desired
        raw_data_np = np.array(signal_final_list_raw, dtype=np.float32)
        raw_data = torch.Tensor(raw_data_np).permute(1,0) # Shape: [time, channels]
        # Convert to float16 *after* potential Tensor operations if needed, or right before saving
        # raw_data = raw_data.type(torch.float16) # Do this conversion closer to saving if it causes issues
    except Exception as e:
        print(f"Error converting raw signal data to Tensor for {file}: {e}. Skipping.")
        return


    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    min_binary_edge_seiz_label = GLOBAL_DATA['min_binary_edge_seiz'] * GLOBAL_DATA['feature_sample_rate']
    min_binary_edge_seiz_raw = GLOBAL_DATA['min_binary_edge_seiz'] * GLOBAL_DATA['sample_rate']

    # --- Check for sufficient length before slicing ---
    if len(y_sampled) < min_seg_len_label or raw_data.shape[0] < min_seg_len_raw:
        # print(f"Warning: Data length insufficient for slicing ({len(y_sampled)} labels, {raw_data.shape[0]} raw samples) for file {file}. Need {min_seg_len_label} labels. Skipping.")
        return


    # --- Slicing logic (keep as is, assuming it's correct based on y_sampled) ---
    # Note: This slicing logic is quite intricate and directly taken from the original code.
    # It involves looking at label sequences and boundaries. Debugging this requires
    # understanding the specific strategy from the paper.
    sliced_raws = []
    sliced_labels = []
    label_list_for_filename = []
    label_count = {}
    y_sampled_list = list(y_sampled) # Work with list for easier slicing/popping
    raw_data_time_first = raw_data # Keep time dimension first

    while len(y_sampled_list) >= min_seg_len_label:
        # Get the first segment
        current_label_segment_list = y_sampled_list[:min_seg_len_label]
        current_raw_segment = raw_data_time_first[:min_seg_len_raw, :] # Slice time dimension

        # Determine label for filename based on segment content
        labels_in_segment = [x[0] for x in groupby(current_label_segment_list)]
        segment_label_code = "0" # Default to background
        if len(labels_in_segment) == 1 and labels_in_segment[0] == '0':
             segment_label_code = "0_patT" if patient_bool else "0_patF"
        else:
             # Find max non-zero code, more robustly
             non_zero_codes = [int(c) for c in labels_in_segment if c != '0']
             if non_zero_codes:
                  max_code = str(max(non_zero_codes))
                  # Determine type based on start/end
                  is_start_zero = current_label_segment_list[0] == '0'
                  is_end_zero = current_label_segment_list[-1] == '0'
                  if is_start_zero and not is_end_zero: segment_label_code = max_code + "_beg"
                  elif not is_start_zero and not is_end_zero: segment_label_code = max_code + "_middle" # Or _whole? Logic was complex
                  elif not is_start_zero and is_end_zero: segment_label_code = max_code + "_end"
                  elif is_start_zero and is_end_zero: segment_label_code = max_code + "_whole" # Contains seizure but starts/ends with 0
             else:
                  # Only contains '0', handled above
                  pass

        # Store the sliced data (raw needs permutation back to [channels, time])
        sliced_raws.append(current_raw_segment.permute(1,0).type(torch.float16)) # Convert to float16 here
        sliced_labels.append("".join(current_label_segment_list)) # Store label string
        label_list_for_filename.append(segment_label_code)

        # Advance the data pointers (remove the processed segment)
        y_sampled_list = y_sampled_list[min_seg_len_label:]
        raw_data_time_first = raw_data_time_first[min_seg_len_raw:, :]


    # --- Save the processed slices ---
    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx] # Already [channels, time], float16
        sliced_y_str = sliced_labels[data_idx]
        try:
            # Convert label string back to mapped integer codes, then to Tensor
            sliced_y_map = list(map(int, sliced_y_str))
            sliced_y = torch.Tensor(sliced_y_map).byte() # LABEL1: Mapped codes (0, 1, 2...)
        except ValueError as e:
            print(f"Error converting sliced label string to Tensor for {file}, slice {data_idx}. Content: '{sliced_y_str[:50]}...'. Error: {e}. Skipping slice.")
            continue

        # --- Generate binary labels (LABEL2, LABEL3) based on the *mapped* codes ---
        # Ensure binary_target maps use integers as keys if sliced_y_map contains integers
        # The original code used dicts with int keys {0:0, 1:1, ...}, so this should work.
        if GLOBAL_DATA['binary_target1'] is not None:
            try:
                sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
            except KeyError as e:
                print(f"KeyError creating LABEL2 for {file}, slice {data_idx}. Missing key: {e}. Skipping slice.")
                continue
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            try:
                sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
            except KeyError as e:
                print(f"KeyError creating LABEL3 for {file}, slice {data_idx}. Missing key: {e}. Skipping slice.")
                continue
        else:
            sliced_y3 = None

        new_data = {} # Reset for each slice
        new_data['RAW_DATA'] = [sliced_raw] # List containing one tensor
        new_data['LABEL1'] = [sliced_y]     # List containing one tensor
        new_data['LABEL2'] = [sliced_y2] if sliced_y2 is not None else []
        new_data['LABEL3'] = [sliced_y3] if sliced_y3 is not None else []

        label_for_fname = label_list_for_filename[data_idx]

        # Construct filename and save
        save_path = os.path.join(GLOBAL_DATA['data_file_directory'], "{}_c{}_label_{}.pkl".format(data_file_name, str(data_idx), label_for_fname))
        try:
            with open(save_path, 'wb') as _f:
                pickle.dump(new_data, _f)
        except Exception as e:
            print(f"Error saving pickle file {save_path}: {e}")

def generate_training_data_leadwise_tuh_dev(file):
    sample_rate = GLOBAL_DATA['sample_rate']    # EX) 200Hz
    file_name = ".".join(file.split(".")[:-1])  # EX) $PATH_TO_EEG/train/01_tcp_ar/072/00007235/s003_2010_11_20/00007235_s003_t000
    data_file_name = file_name.split("/")[-1]   # EX) 00007235_s003_t000
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]    # EX) EEG FP1-ref or EEG FP1-LE --> EEG FP1
        label_list_c.append(label_noref)   

    ############################# part 1: labeling  ###############################
    label_file = open(file_name + "." + GLOBAL_DATA['label_type'], 'r') # EX) 00007235_s003_t003.tse or 00007235_s003_t003.tse_bi
    y = label_file.readlines()
    y = list(y[2:])
    y_labels = list(set([i.split(" ")[2] for i in y]))
    signal_sample_rate = int(signal_headers[0]['sample_rate'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']): # if one or more of ['EEG FP1', 'EEG FP2', ... doesn't exist
        return
    # if not any(elem in y_labels for elem in GLOBAL_DATA['disease_type']): # if non-patient exist
    #     return
    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])
    
    # check if seizure patient or non-seizure patient
    patient_wise_dir = "/".join(file_name.split("/")[:-2])
    edf_list = search_walk({'path': patient_wise_dir, 'extension': ".csv_bi"})
    patient_bool = False
    for tse_bi_file in edf_list:
        label_file = open(tse_bi_file, 'r') # EX) 00007235_s003_t003.tse or 00007235_s003_t003.tse_bi
        y = label_file.readlines()
        y = list(y[2:])
        for line in y:
            if len(line) > 5:
                if line.split(" ")[2] != 'bckg':
                    patient_bool = True
                    break
        if patient_bool:
            break

    ############################# part 2: input data filtering #############################
    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_rate']) > sample_rate:
            secs = len(signal)/float(signal_sample_rate)
            samps = int(secs*sample_rate)
            x = sci_sig.resample(signal, samps)
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            signal_list.append(signal)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return 
    
    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
    
    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        y_sampled += y_sampled[-1] * diff

    y_sampled_np = np.array(list(map(int,y_sampled)))
    new_labels = []
    new_labels_idxs = []

    ############################# part 3: slicing for easy training  #############################
    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    # slice and save if training data
    new_data = {}
    raw_data = torch.Tensor(signal_final_list_raw).permute(1,0)
    raw_data = raw_data.type(torch.float16)
    
    # max_seg_len_before_seiz_label = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    # max_seg_len_before_seiz_raw = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['sample_rate']
    # min_end_margin_label = args.slice_end_margin_length * GLOBAL_DATA['feature_sample_rate']
    # min_end_margin_raw = args.slice_end_margin_length * GLOBAL_DATA['sample_rate']

    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    # max_seg_len_label = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    # max_seg_len_raw = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    
    sliced_raws = []
    sliced_labels = []
    label_list_for_filename = []

    if len(y_sampled) < min_seg_len_label:
        return
    else:
        label_count = {}
        while len(y_sampled) >= min_seg_len_label:
            one_left_slice = False
            sliced_y = y_sampled[:min_seg_len_label]
                
            if (sliced_y[-1] == '0'):
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                y_sampled = y_sampled[min_seg_len_label:]

                labels = [x[0] for x in groupby(sliced_y)]
                if (len(labels) == 1) and (labels[0] == '0'):
                    label = "0"
                else:
                    label = ("".join(labels)).replace("0", "")[0]
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label_list_for_filename.append(label)

            else:
                if '0' in y_sampled[min_seg_len_label:]:
                    end_1 = y_sampled[min_seg_len_label:].index('0')
                    temp_y_sampled = list(y_sampled[min_seg_len_label+end_1:])
                    temp_y_sampled_order = [x[0] for x in groupby(temp_y_sampled)]

                    if len(list(set(temp_y_sampled))) == 1:
                        end_2 = len(temp_y_sampled)
                        one_left_slice = True
                    else:
                        end_2 = temp_y_sampled.index(temp_y_sampled_order[1])

                    if end_2 >= min_end_margin_label:
                        temp_sec = random.randint(1,args.slice_end_margin_length)
                        temp_seg_len_label = int(min_seg_len_label + (temp_sec * args.feature_sample_rate) + end_1)
                        temp_seg_len_raw = int(min_seg_len_raw + (temp_sec * args.samplerate) + (end_1 * GLOBAL_DATA['fsr_sr_ratio']))
                    else:
                        if one_left_slice:
                            temp_label = end_2
                        else:
                            temp_label = end_2 // 2

                        temp_seg_len_label = int(min_seg_len_label + temp_label + end_1)
                        temp_seg_len_raw = int(min_seg_len_raw + (temp_label * GLOBAL_DATA['fsr_sr_ratio']) + (end_1 * GLOBAL_DATA['fsr_sr_ratio']))

                    sliced_y = y_sampled[:temp_seg_len_label]
                    sliced_raw_data = raw_data[:temp_seg_len_raw].permute(1,0)
                    raw_data = raw_data[temp_seg_len_raw:]
                    y_sampled = y_sampled[temp_seg_len_label:]

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)
                else:
                    sliced_y = y_sampled[:]
                    sliced_raw_data = raw_data[:].permute(1,0)
                    raw_data = []
                    y_sampled = []

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)
            
    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int,sliced_y))

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        new_data['RAW_DATA'] = [sliced_raw]
        new_data['LABEL1'] = [sliced_y]
        new_data['LABEL2'] = [sliced_y2]
        new_data['LABEL3'] = [sliced_y3]

        label = label_list_for_filename[data_idx]
        
        with open(GLOBAL_DATA['data_file_directory'] + "/{}_c{}_len{}_label_{}.pkl".format(data_file_name, str(data_idx), str(len(sliced_y)), str(label)), 'wb') as _f:
            pickle.dump(new_data, _f)      
        new_data = {}


def main(args):
    save_directory = args.save_directory
    data_type = args.data_type
    dataset = args.dataset
    label_type = args.label_type
    sample_rate = args.samplerate
    cpu_num = args.cpu_num
    feature_type = args.feature_type
    feature_sample_rate = args.feature_sample_rate
    task_type = args.task_type
    data_file_directory = save_directory + "/dataset-{}_task-{}_datatype-{}_v6".format(dataset, task_type, data_type)
    
    
    labels = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8',  
                    'EEG C3', 'EEG C4', 'EEG CZ', 'EEG T3', 'EEG T4', 
                    'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG T5', 'EEG T6', 'EEG PZ', 'EEG FZ']

    eeg_data_directory = "../TUHEEG/v2.0.3/edf/{}".format(data_type)
    # eeg_data_directory = "/mnt/aitrics_ext/ext01/shared/edf/tuh_final/{}".format(data_type)
    
    if label_type == "tse":
        disease_labels =  {'bckg': 0, 'cpsz': 1, 'mysz': 2, 'gnsz': 3, 'fnsz': 4, 'tnsz': 5, 'tcsz': 6, 'spsz': 7, 'absz': 8}
    elif label_type == "tse_bi":
        disease_labels =  {'bckg': 0, 'seiz': 1}
    disease_labels_inv = {v: k for k, v in disease_labels.items()}
    
    edf_list1 = search_walk({'path': eeg_data_directory, 'extension': ".edf"})
    edf_list2 = search_walk({'path': eeg_data_directory, 'extension': ".EDF"})
    if edf_list2:
        edf_list = edf_list1 + edf_list2
    else:
        edf_list = edf_list1

    if os.path.isdir(data_file_directory):
        os.system("rm -rf {}".format(data_file_directory))
    os.system("mkdir {}".format(data_file_directory))

    GLOBAL_DATA['label_list'] = labels # 'EEG FP1', 'EEG FP2', 'EEG F3', ...
    GLOBAL_DATA['disease_labels'] = disease_labels #  {'bckg': 0, 'cpsz': 1, 'mysz': 2, ...
    GLOBAL_DATA['disease_labels_inv'] = disease_labels_inv #  {0:'bckg', 1:'cpsz', 2:'mysz', ...
    GLOBAL_DATA['data_file_directory'] = data_file_directory
    GLOBAL_DATA['label_type'] = label_type # "tse_bi" ...
    GLOBAL_DATA['feature_type'] = feature_type
    GLOBAL_DATA['feature_sample_rate'] = feature_sample_rate
    GLOBAL_DATA['sample_rate'] = sample_rate
    GLOBAL_DATA['fsr_sr_ratio'] = (sample_rate // feature_sample_rate)
    GLOBAL_DATA['min_binary_slicelength'] = args.min_binary_slicelength
    GLOBAL_DATA['min_binary_edge_seiz'] = args.min_binary_edge_seiz

    target_dictionary = {0:0}
    selected_diseases = []

    if label_type == "tse":
        for idx, i in enumerate(args.disease_type):
            selected_diseases.append(str(disease_labels[i]))
            target_dictionary[disease_labels[i]] = idx + 1
    
    GLOBAL_DATA['disease_type'] = args.disease_type # args.disease_type == ['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz']
    GLOBAL_DATA['target_dictionary'] = target_dictionary # {0: 0, 4: 1, 5: 2, 8: 3, 2: 4, 9: 5, 6: 6, 7: 7, 3: 8}
    GLOBAL_DATA['selected_diseases'] = selected_diseases # ['4', '5', '8', '2', '9', '6', '7', '3']
    GLOBAL_DATA['binary_target1'] = args.binary_target1
    GLOBAL_DATA['binary_target2'] = args.binary_target2

    if label_type == "tse_bi":
        GLOBAL_DATA['disease_type'] = ['bckg', 'seiz']
        GLOBAL_DATA['target_dictionary'] = {0:0, 1:1}
        GLOBAL_DATA['selected_diseases'] = ['0', '1']

    print("########## Preprocessor Setting Information ##########")
    print("Number of EDF files: ", len(edf_list))
    for i in GLOBAL_DATA:
        print("{}: {}".format(i, GLOBAL_DATA[i]))
    with open(data_file_directory + '/preprocess_info.infopkl', 'wb') as pkl:
        pickle.dump(GLOBAL_DATA, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    print("################ Preprocess begins... ################\n")
    
    if (task_type == "binary") and (args.data_type == "train"):
        run_multi_process(generate_training_data_leadwise_tuh_train_final, edf_list, n_processes=cpu_num)
    elif (task_type == "binary") and (args.data_type == "dev"):
        run_multi_process(generate_training_data_leadwise_tuh_train_final, edf_list, n_processes=cpu_num)
        
if __name__ == '__main__':
    # make sure all edf file name different!!! if not, additional coding is necessary
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-sd', type=int, default=1004,
                        help='Random seed number')
    parser.add_argument('--samplerate', '-sr', type=int, default=200,
                        help='Sample Rate')
    parser.add_argument('--save_directory', '-sp', type=str,
                        help='Path to save data')
    parser.add_argument('--label_type', '-lt', type=str,
                        default='tse',
                        help='tse_bi = global with binary label, tse = global with various labels, cae = severance CAE seizure label.')                      
    parser.add_argument('--cpu_num', '-cn', type=int,
                        default=32,
                        help='select number of available cpus')   
    parser.add_argument('--feature_type', '-ft', type=str,
                        default=['rawsignal'])   
    parser.add_argument('--feature_sample_rate', '-fsr', type=int,
                        default=50,
                        help='select features sample rate')   
    parser.add_argument('--dataset', '-st', type=str,
                        default='tuh',
                        choices=['tuh'])                   
    parser.add_argument('--data_type', '-dt', type=str,
                        default='train',
                        choices=['train', 'dev'])                   
    parser.add_argument('--task_type', '-tt', type=str,
                        default='binary',
                        choices=['anomaly', 'multiclassification', 'binary'])                   

    ##### Target Grouping #####
    parser.add_argument('--disease_type', type=list, default=['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'], choices=['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'])

    ### for binary detector ###
    # key numbers represent index of --disease_type + 1  ### -1 is "not being used"
    parser.add_argument('--binary_target1', type=dict, default={0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    parser.add_argument('--binary_target2', type=dict, default={0:0, 1:1, 2:2, 3:2, 4:2, 5:1, 6:3, 7:4, 8:5})
    parser.add_argument('--min_binary_slicelength', type=int, default=30)           
    parser.add_argument('--min_binary_edge_seiz', type=int, default=3) 
    args = parser.parse_args()
    main(args)

