import numpy as np
import os
import pandas as pd
import open_ephys_IO
import samplerate
import OpenEphys
import PostSorting.parameters
import math
import gc
import sys
from scipy import stats
import PostSorting.vr_stop_analysis
import PostSorting.post_process_sorted_data_vr
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import settings
import os
import traceback
import control_sorting_analysis
import warnings
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel

def correct_for_restart(location):
    cummulative_minimums = np.minimum.accumulate(location)
    cummulative_maximums = np.maximum.accumulate(location)

    local_min_median = np.median(cummulative_minimums)
    local_max_median = np.median(cummulative_maximums)

    #location [location >local_max_median] = local_max_median
    location [location <local_min_median] = local_min_median # deals with if the VR is switched off during recording - location value drops to 0 - min is usually 0.56 approx
    return location


def get_raw_location(recording_folder):
    """
    Loads raw location continuous channel from ACD1.continuous
    # input: spatial dataframe, path to local recording folder, initialised parameters
    # output: raw location as numpy array
    """
    print('Extracting raw location...')
    file_path = recording_folder + '/' + settings.movement_channel
    if os.path.exists(file_path):
        location = open_ephys_IO.get_data_continuous(file_path)
    else:
        raise FileNotFoundError('Movement data was not found.')
    location=correct_for_restart(location)
    return np.asarray(location, dtype=np.float16)


def calculate_track_location(position_data, recording_folder, track_length):
    recorded_location = get_raw_location(recording_folder) # get raw location from DAQ pin
    print('Converting raw location input to cm...')

    recorded_startpoint = min(recorded_location)
    recorded_endpoint = max(recorded_location)
    recorded_track_length = recorded_endpoint - recorded_startpoint
    distance_unit = recorded_track_length/track_length  # Obtain distance unit (cm) by dividing recorded track length to actual track length
    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit
    position_data['x_position_cm'] = np.asarray(location_in_cm, dtype=np.float16) # fill in dataframe
    return position_data

def recalculate_track_location(position_data, recording_folder, track_length):
    recorded_location = get_raw_location(recording_folder) # get raw location from DAQ pin
    trial_numbers = np.asarray(position_data["trial_number"])
    print('Converting raw location input to cm...')
    location_in_cm = np.array([])

    global_recorded_startpoint = min(recorded_location)
    global_recorded_endpoint = max(recorded_location)
    unique_trial_numbers =  np.unique(position_data["trial_number"])
    for tn in unique_trial_numbers:
        trial_locations = recorded_location[trial_numbers == tn]

        if tn == unique_trial_numbers[0]:
            recorded_startpoint = global_recorded_startpoint
            recorded_endpoint = max(trial_locations)
        elif tn == unique_trial_numbers[-1]:
            recorded_startpoint = min(trial_locations)
            recorded_endpoint = global_recorded_endpoint
        else:
            recorded_startpoint = min(trial_locations)
            recorded_endpoint = max(trial_locations)
        recorded_track_length = recorded_endpoint - recorded_startpoint
        distance_unit = recorded_track_length/track_length  # Obtain distance unit (cm) by dividing recorded track length to actual track length
        trial_location_in_cm = (trial_locations - recorded_startpoint) / distance_unit

        location_in_cm = np.concatenate((location_in_cm, trial_location_in_cm))
    position_data['x_position_cm'] = np.asarray(location_in_cm, dtype=np.float16) # fill in dataframe
    return position_data


# calculate time from start of recording in seconds for each sampling point
def calculate_time(position_data):
    print('Calculating time...')
    position_data['time_seconds'] = position_data['trial_number'].index/settings.sampling_rate # convert sampling rate to time (seconds) by dividing by 30
    return position_data


# for each sampling point, calculates time from last sample point
def calculate_instant_dwell_time(position_data, pos_sampling_rate):
    print('Calculating dwell time...')
    position_data['dwell_time_ms'] = 1/pos_sampling_rate
    return position_data


def check_for_trial_restarts(trial_indices):
    new_trial_indices=[]
    for icount,i in enumerate(range(len(trial_indices)-1)):
        index_difference = trial_indices[icount] - trial_indices[icount+1]
        if index_difference > - settings.sampling_rate/2:
            continue
        else:
            index = trial_indices[icount]
            new_trial_indices = np.append(new_trial_indices,index)
    return new_trial_indices


def get_new_trial_indices(position_data):
    location_diff = position_data['x_position_cm'].diff()  # Get the raw location from the movement channel
    trial_indices = np.where(location_diff < -20)[0]# return indices where is new trial
    trial_indices = check_for_trial_restarts(trial_indices)# check if trial_indices values are within 1500 of eachother, if so, delete
    new_trial_indices=np.hstack((0,trial_indices,len(location_diff))) # add start and end indicies so fills in whole arrays
    return new_trial_indices


def fill_in_trial_array(new_trial_indices,trials):
    trial_count = 1
    for icount,i in enumerate(range(len(new_trial_indices)-1)):
        new_trial_index = int(new_trial_indices[icount])
        next_trial_index = int(new_trial_indices[icount+1])
        trials[new_trial_index:next_trial_index] = trial_count
        trial_count += 1
    return trials


# calculates trial number from location
def calculate_trial_numbers(position_data):
    print('Calculating trial numbers...')
    trials = np.zeros((position_data.shape[0]))
    new_trial_indices = get_new_trial_indices(position_data)
    trials = fill_in_trial_array(new_trial_indices,trials)

    position_data['trial_number'] = np.asarray(trials, dtype=np.uint16)
    position_data['new_trial_indices'] = pd.Series(new_trial_indices)
    print('This mouse did ', int(max(trials)), ' trials')
    gc.collect()
    return position_data


# two continuous channels represent trial type
def load_first_trial_channel(recording_folder):
    first = []
    file_path = recording_folder + '/' + settings.first_trial_channel
    trial_first = open_ephys_IO.get_data_continuous(file_path)
    first.append(trial_first)
    return np.asarray(first, dtype=np.uint8)


# two continuous channels represent trial type
def load_second_trial_channel(recording_folder):
    second = []
    file_path = recording_folder + '/' + settings.second_trial_channel
    trial_second = open_ephys_IO.get_data_continuous(file_path)
    second.append(trial_second)
    return np.asarray(second, dtype=np.uint8)


def calculate_trial_types(position_data, recording_folder):
    print('Loading trial types from continuous...')
    first_ch = load_first_trial_channel(recording_folder)
    second_ch = load_second_trial_channel(recording_folder)
    trial_type = np.zeros((second_ch.shape[1]));trial_type[:]=np.nan
    new_trial_indices = position_data['new_trial_indices'].values
    new_trial_indices = new_trial_indices[~np.isnan(new_trial_indices)]

    print('Calculating trial type...')
    for icount,i in enumerate(range(len(new_trial_indices)-1)):
        new_trial_index = int(new_trial_indices[icount])
        next_trial_index = int(new_trial_indices[icount+1])
        second = stats.mode(second_ch[0,new_trial_index:next_trial_index])[0]
        first = stats.mode(first_ch[0,new_trial_index:next_trial_index])[0]
        if second < 2 and first < 2: # if beaconed
            trial_type[new_trial_index:next_trial_index] = int(0)
        if second > 2 and first < 2: # if non beaconed
            trial_type[new_trial_index:next_trial_index] = int(1)
        if second > 2 and first > 2: # if probe
            trial_type[new_trial_index:next_trial_index] = int(2)
    position_data['trial_type'] = np.asarray(trial_type, dtype=np.uint8)
    return position_data

def calculate_instant_velocity(position_data):
    print('Calculating velocity...')
    location = np.array(position_data['x_position_cm'], dtype=np.float32)

    sampling_points_per200ms = int(settings.sampling_rate/5)
    end_of_loc_to_subtr = location[:-sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other
    beginning_of_loc_to_subtr = location[:sampling_points_per200ms]# Rearrange arrays in a way that they just need to be subtracted from each other
    location_to_subtract_from = np.append(beginning_of_loc_to_subtr, end_of_loc_to_subtr)
    velocity = location - location_to_subtract_from

    # use new trial indices to fix velocity around teleports
    new_trial_indices = np.unique(position_data["new_trial_indices"][~np.isnan(position_data["new_trial_indices"])])
    for new_trial_indice in new_trial_indices:
        if new_trial_indice > sampling_points_per200ms: # ignores first trial index
            velocity[int(new_trial_indice-sampling_points_per200ms)-100:int(new_trial_indice+sampling_points_per200ms)+100] = np.nan

    #now interpolate where these nan values are
    ok = ~np.isnan(velocity)
    xp = ok.ravel().nonzero()[0]
    fp = velocity[~np.isnan(velocity)]
    x  = np.isnan(velocity).ravel().nonzero()[0]
    velocity[np.isnan(velocity)] = np.interp(x, xp, fp)
    velocity = velocity*5

    position_data['velocity'] = velocity

    return position_data

def running_mean(a, n):
    '''
    Calculates moving average

    input
        a : array,  to calculate averages on
        n : integer, number of points that is used for one average calculation

    output
        array, contains rolling average values (each value is the average of the previous n values)
    '''
    cumsum = np.cumsum(np.insert(a,0,0), dtype=float)
    return np.append(a[0:n-1], ((cumsum[n:] - cumsum[:-n]) / n))

def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window is too big, plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out

def get_avg_speed_200ms(position_data):
    print('Calculating average speed...')
    velocity = np.array(position_data['velocity'])  # Get the raw location from the movement channel
    sampling_points_per200ms = int(settings.sampling_rate/5)
    position_data['speed_per200ms'] = running_mean(velocity, sampling_points_per200ms)  # Calculate average speed at each point by averaging instant velocities
    return position_data

def downsampled_position_data(raw_position_data, sampling_rate = settings.sampling_rate, down_sampled_rate = settings.location_ds_rate):
    position_data = pd.DataFrame()
    downsample_factor = int(sampling_rate/ down_sampled_rate)
    position_data["x_position_cm"] = raw_position_data["x_position_cm"][::downsample_factor]
    position_data["time_seconds"] =  raw_position_data["time_seconds"][::downsample_factor]
    position_data["speed_per200ms"] = raw_position_data["speed_per200ms"][::downsample_factor]
    position_data["trial_number"] = raw_position_data["trial_number"][::downsample_factor]
    position_data["trial_type"] = raw_position_data["trial_type"][::downsample_factor]

    return position_data

def syncronise_position_data(recording_folder, track_length):
    raw_position_data = pd.DataFrame()
    raw_position_data = calculate_track_location(raw_position_data, recording_folder, track_length)
    raw_position_data = calculate_trial_numbers(raw_position_data)
    raw_position_data = recalculate_track_location(raw_position_data, recording_folder, track_length)

    raw_position_data = calculate_trial_types(raw_position_data, recording_folder)
    raw_position_data = calculate_time(raw_position_data)
    raw_position_data = calculate_instant_dwell_time(raw_position_data, pos_sampling_rate=settings.sampling_rate)
    raw_position_data = calculate_instant_velocity(raw_position_data)
    raw_position_data = get_avg_speed_200ms(raw_position_data)
    position_data = downsampled_position_data(raw_position_data)

    rpd = np.asarray(raw_position_data["x_position_cm"])
    trial_numbers_raw = np.array(raw_position_data['trial_number'], dtype=np.int64)
    rpd_elapsed = (track_length*(trial_numbers_raw-1))+rpd
    gauss_kernel = Gaussian1DKernel(stddev=200)
    rpd = convolve(rpd, gauss_kernel)
    rpd = moving_sum(rpd, window=100)/100
    rpd = np.append(rpd, np.zeros(len(raw_position_data["x_position_cm"])-len(rpd)))
    raw_position_data["x_position_cm"] = rpd

    # potential bug fix
    rpd_elapsed = convolve(rpd_elapsed, gauss_kernel)
    rpd_elapsed = moving_sum(rpd_elapsed, window=100)/100
    rpd_elapsed = np.append(rpd_elapsed, np.zeros(len(raw_position_data["x_position_cm"])-len(rpd_elapsed)))
    rpd = rpd_elapsed%track_length

    raw_position_data["x_position_cm"] = rpd
    gc.collect()
    return raw_position_data, position_data

def get_track_length(recording_path):
    parameter_file_path = control_sorting_analysis.get_tags_parameter_file(recording_path)
    stop_threshold, track_length, cue_conditioned_goal = PostSorting.post_process_sorted_data_vr.process_running_parameter_tag(parameter_file_path)
    return track_length

def hotfix_recording(broken_recording, log_file):

    '''
    if os.path.exists(broken_recording):
        file_path = broken_recording + '/' + settings.first_trial_channel
        ch = OpenEphys.loadContinuous(file_path)
        ch['data'][:] = 0
        OpenEphys.writeContinuousFile(file_path, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])
    '''
    return


def pad_shorter_array_with_0s(array1, array2):
    if len(array1) < len(array2):
        array1 = np.pad(array1, (0, len(array2)-len(array1)), 'constant')
    if len(array2) < len(array1):
        array2 = np.pad(array2, (0, len(array1)-len(array2)), 'constant')
    return array1, array2

def hotfix2(recording, log):

    log_numpy = np.loadtxt(log,comments='#',delimiter=';',skiprows=4)
    log_numpy[:, 1] = log_numpy[:, 1]*10
    # trial y position is collumn 8
    # position is collum 1

    mean_blender_sampling_rate = 1/np.mean(np.diff(log_numpy[:,0]))
    ephys_sampling_rate = 30000
    raw_position_data, position_data = syncronise_position_data(recording, track_length=200)

    position_data = pd.DataFrame()
    downsample_factor = int(ephys_sampling_rate/ mean_blender_sampling_rate)
    position_data["x_position_cm"] = raw_position_data["x_position_cm"][::downsample_factor]
    position_data["time_seconds"] =  raw_position_data["time_seconds"][::downsample_factor]
    position_data["speed_per200ms"] = raw_position_data["speed_per200ms"][::downsample_factor]
    position_data["trial_number"] = raw_position_data["trial_number"][::downsample_factor]
    position_data["trial_type"] = raw_position_data["trial_type"][::downsample_factor]

    log_locations = log_numpy[:, 1]
    ephys_locations = np.asarray(position_data["x_position_cm"])
    log_locations, ephys_locations = pad_shorter_array_with_0s(log_locations, ephys_locations)
    corr = np.correlate(log_locations, ephys_locations, "full")  # this is the correlation array between the sync pulse series
    lag = (np.argmax(corr) - (corr.size + 1)/2)/mean_blender_sampling_rate  # lag between sync pulses is based on max correlation
    lag_in_sampling_frequency = int(lag*mean_blender_sampling_rate)

    #log_tn_transitions = np.diff(log_numpy[:, 9])
    #ephys_tn_transitions = np.diff(np.asarray(position_data["trial_number"]))
    #log_tn_transitions, ephys_tn_transitions = pad_shorter_array_with_0s(log_tn_transitions, ephys_tn_transitions)
    #corr = np.correlate(log_tn_transitions, ephys_tn_transitions, "full")  # this is the correlation array between the sync pulse series
    #lag = (np.argmax(corr) - (corr.size + 1)/2)/mean_blender_sampling_rate  # lag between sync pulses is based on max correlation
    #lag_in_sampling_frequency = int(lag*mean_blender_sampling_rate)

    log_numpy_pseudo_tt = log_numpy[:, -6]
    log_numpy_tt = (log_numpy_pseudo_tt/10).astype(np.int64)

    trial_types_for_position_data = log_numpy_tt[lag_in_sampling_frequency:]
    trial_types_for_position_data = trial_types_for_position_data[:len(position_data)]

    a = log_locations[lag_in_sampling_frequency:]
    b = a[:len(position_data)]
    b = np.repeat(b, len(raw_position_data)/len(b))
    b = np.concatenate([b, np.zeros(len(raw_position_data)-len(b))])
    b_ephys = np.asarray(raw_position_data["x_position_cm"])

    pearson = stats.pearsonr(b, b_ephys)[0]

    tts = np.repeat(trial_types_for_position_data, len(raw_position_data)/len(trial_types_for_position_data))
    tts_same_length_as_raw = np.concatenate([tts, tts[-1]*np.ones(len(raw_position_data)-len(tts))])

    to_voltages = tts_same_length_as_raw*3

    # now we need to rewrite channel 2
    if os.path.exists(recording):
        file_path = recording + '/' + settings.second_trial_channel
        ch = OpenEphys.loadContinuous(file_path)
        ch['data'] = to_voltages
        OpenEphys.writeContinuousFile(file_path, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])

    print("I hOPE THAT WORKED LOLLLLLLLLLL")

def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array


def hotfix3(recording, log):
    recorded_location = get_raw_location(recording) # get raw location from DAQ pin
    print('Converting raw location input to cm...')

    recorded_startpoint = min(recorded_location)
    recorded_endpoint = max(recorded_location)
    recorded_track_length = recorded_endpoint - recorded_startpoint
    distance_unit = recorded_track_length/200  # Obtain distance unit (cm) by dividing recorded track length to actual track length
    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit

    log_numpy = np.loadtxt(log,comments='#',delimiter=';',skiprows=4)
    log_numpy[:, 1] = log_numpy[:, 1]*10
    # trial y position is collumn 8
    # position is collum 1

    mean_blender_sampling_rate = 1/np.mean(np.diff(log_numpy[:,0]))
    ephys_sampling_rate = 30000
    raw_position_data, position_data = syncronise_position_data(recording, track_length=200)

    position_data = pd.DataFrame()
    downsample_factor = int(ephys_sampling_rate/ mean_blender_sampling_rate)
    position_data["x_position_cm"] = raw_position_data["x_position_cm"][::downsample_factor]
    position_data["time_seconds"] =  raw_position_data["time_seconds"][::downsample_factor]
    position_data["speed_per200ms"] = raw_position_data["speed_per200ms"][::downsample_factor]
    position_data["trial_number"] = raw_position_data["trial_number"][::downsample_factor]
    position_data["trial_type"] = raw_position_data["trial_type"][::downsample_factor]

    log_locations = log_numpy[:, 1]
    ephys_locations = np.asarray(position_data["x_position_cm"])
    log_locations, ephys_locations = pad_shorter_array_with_0s(log_locations, ephys_locations)
    corr = np.correlate(log_locations, ephys_locations, "full")  # this is the correlation array between the sync pulse series
    lag = (np.argmax(corr) - (corr.size + 1)/2)/mean_blender_sampling_rate  # lag between sync pulses is based on max correlation
    lag_in_sampling_frequency = int(lag*mean_blender_sampling_rate)

    #log_tn_transitions = np.diff(log_numpy[:, 9])
    #ephys_tn_transitions = np.diff(np.asarray(position_data["trial_number"]))
    #log_tn_transitions, ephys_tn_transitions = pad_shorter_array_with_0s(log_tn_transitions, ephys_tn_transitions)
    #corr = np.correlate(log_tn_transitions, ephys_tn_transitions, "full")  # this is the correlation array between the sync pulse series
    #lag = (np.argmax(corr) - (corr.size + 1)/2)/mean_blender_sampling_rate  # lag between sync pulses is based on max correlation
    #lag_in_sampling_frequency = int(lag*mean_blender_sampling_rate)

    log_locations = min_max_normlise(log_locations, min_val=recorded_startpoint, max_val=recorded_endpoint)
    log_locations_for_position_data = log_locations[lag_in_sampling_frequency:]
    log_locations_for_position_data = log_locations_for_position_data[:len(position_data)]

    a = log_locations[lag_in_sampling_frequency:]
    b = a[:len(position_data)]
    b = np.repeat(b, len(raw_position_data)/len(b))
    b = np.concatenate([b, np.zeros(len(raw_position_data)-len(b))])
    b_ephys = np.asarray(raw_position_data["x_position_cm"])

    pearson = stats.pearsonr(b, b_ephys)[0]

    tts = np.repeat(log_locations_for_position_data, len(raw_position_data)/len(log_locations_for_position_data))
    tts_same_length_as_raw = np.concatenate([tts, tts[-1]*np.ones(len(raw_position_data)-len(tts))])

    to_voltages = tts_same_length_as_raw

    # now we need to rewrite channel 2
    if os.path.exists(recording):
        file_path = recording + '/' + settings.movement_channel
        ch = OpenEphys.loadContinuous(file_path)
        ch['data'] = to_voltages
        OpenEphys.writeContinuousFile(file_path, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])

    print("I hOPE THAT WORKED LOLLLLLLLLLL")


def hotfix_junji(recording):
    # all trials have been mislabeled as probe trials so the first channel pin will be set to 0,
    # this should make all probe trials now non beaconed trials
    if os.path.exists(recording):
        file_path = recording + '/' + settings.first_trial_channel
        ch = OpenEphys.loadContinuous(file_path)
        ch['data'] = np.repeat(0, len(ch['data']))
        OpenEphys.writeContinuousFile(file_path, ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])
    print("I hOPE THAT WORKED LOLLLLLLLLLL")


def process_recordings(vr_recording_path_list):

    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            output_path = recording+'/'+settings.sorterName
            track_length = get_track_length(recording)
            #processed_position_data= pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")

            hotfix_junji(recording)
            #hotfix2(recording, log)
            #hotfix3(recording, log)
            #raw_position_data, position_data = syncronise_position_data(recording, track_length)
            #raw_position_data, processed_position_data, position_data = PostSorting.post_process_sorted_data_vr.process_position_data(recording, output_path, track_length, stop_threshold=4.7)
            #processed_position_data.to_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            #position_data.to_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")

            print("successfully processed and saved vr_grid analysis on "+recording)
        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)


def main():
    print('-------------------------------------------------------------')

    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_Junji/vr") if f.is_dir()]
    #process_recordings(J1_list)
    #process_recordings(J2_list)


    print("look now`")


if __name__ == '__main__':
    main()