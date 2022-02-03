import numpy as np
import pandas as pd
import PostSorting.parameters
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from Edmond.VR_grid_analysis.remake_position_data import syncronise_position_data
from PostSorting.vr_spatial_firing import bin_fr_in_time, add_position_x
from scipy import stats
from scipy import signal
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
import os
import traceback
import warnings
import matplotlib.ticker as ticker
import sys
import Edmond.plot_utility2
import Edmond.VR_grid_analysis.hit_miss_try_firing_analysis
import settings
import matplotlib.pylab as plt
import matplotlib as mpl
import control_sorting_analysis
import PostSorting.post_process_sorted_data_vr
from astropy.timeseries import LombScargle
from Edmond.utility_functions.array_manipulations import *
from joblib import Parallel, delayed
import multiprocessing
import open_ephys_IO
warnings.filterwarnings('ignore')
from scipy.stats.stats import pearsonr
from scipy.stats import shapiro
plt.rc('axes', linewidth=3)

def add_avg_trial_speed(processed_position_data):
    avg_trial_speeds = []
    for trial_number in np.unique(processed_position_data["trial_number"]):
        trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number]
        speeds = np.asarray(trial_processed_position_data['speeds_binned_in_time'])[0]
        avg_speed = np.nanmean(speeds)
        avg_trial_speeds.append(avg_speed)
    processed_position_data["avg_trial_speed"] = avg_trial_speeds
    return processed_position_data


def add_avg_RZ_speed(processed_position_data, track_length):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30

    avg_speed_in_RZs = []
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number]
        speeds_in_time = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_processed_position_data['speeds_binned_in_time'])
        pos_in_time = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_processed_position_data['pos_binned_in_time'])
        in_rz_mask = (pos_in_time > reward_zone_start) & (pos_in_time <= reward_zone_end)
        speeds_in_time_in_RZ = speeds_in_time[in_rz_mask]
        #speeds_in_time_in_RZ = speeds_in_time_in_RZ[speeds_in_time_in_RZ > 0] # remove dead stops or backwards movements
        if len(speeds_in_time_in_RZ)==0:
            avg_speed_in_RZ = np.nan
        else:
            avg_speed_in_RZ = np.nanmean(speeds_in_time_in_RZ)
        avg_speed_in_RZs.append(avg_speed_in_RZ)

    processed_position_data["avg_speed_in_RZ"] = avg_speed_in_RZs
    return processed_position_data

def add_avg_track_speed(processed_position_data, track_length):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30
    track_start = 30
    track_end = track_length-30

    avg_speed_on_tracks = []
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number]
        speeds_in_time = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_processed_position_data['speeds_binned_in_time'])
        pos_in_time = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_processed_position_data['pos_binned_in_time'])
        in_rz_mask = (pos_in_time > reward_zone_start) & (pos_in_time <= reward_zone_end)
        speeds_in_time_outside_RZ = speeds_in_time[~in_rz_mask]
        pos_in_time = pos_in_time[~in_rz_mask]
        track_mask = (pos_in_time > track_start) & (pos_in_time <= track_end)
        speeds_in_time_outside_RZ = speeds_in_time_outside_RZ[track_mask]

        #speeds_in_time_in_RZ = speeds_in_time_in_RZ[speeds_in_time_in_RZ > 0] # remove dead stops or backwards movements
        if len(speeds_in_time_outside_RZ)==0:
            avg_speed_on_track = np.nan
        else:
            avg_speed_on_track = np.nanmean(speeds_in_time_outside_RZ)
        avg_speed_on_tracks.append(avg_speed_on_track)

    processed_position_data["avg_speed_on_track"] = avg_speed_on_tracks
    return processed_position_data

def add_RZ_bias(processed_position_data):
    avg_RZ_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_in_RZ"])
    avg_track_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_on_track"])
    RZ_stop_bias = avg_RZ_speed/avg_track_speed
    processed_position_data["RZ_stop_bias"] =RZ_stop_bias
    return processed_position_data

def add_hit_miss_try2(processed_position_data, track_length):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30

    hmts=[]
    for index, row in processed_position_data.iterrows():
        rewarded = row["rewarded"]
        RZ_speed = row["avg_speed_in_RZ"]
        track_speed = row["avg_speed_on_track"]
        TI = row["RZ_stop_bias"]

        if rewarded and (track_speed<20):
            hmt="slow_hit"
        elif rewarded and (track_speed>20):
            hmt="hit"
        elif (track_speed>20) and (TI<1):
            hmt="try"
        elif (track_speed>20) and (TI>1):
            hmt="miss"
        else:
            hmt="slow miss"
        hmts.append(hmt)

    processed_position_data["hit_miss_try"] = hmts
    return processed_position_data

def add_hit_miss_try3(processed_position_data, track_length):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30

    rewarded_processed_position_data = processed_position_data[(processed_position_data["rewarded"] == True)]
    minimum_speeds_in_rz = []
    speeds_in_rz = []
    for trial_number in np.unique(rewarded_processed_position_data["trial_number"]):
        trial_rewarded_processed_position_data = rewarded_processed_position_data[rewarded_processed_position_data["trial_number"] == trial_number]
        rewarded_speeds_in_space = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_rewarded_processed_position_data['speeds_binned_in_space'])
        rewarded_bin_centres = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_rewarded_processed_position_data['position_bin_centres'])
        in_rz_mask = (rewarded_bin_centres > reward_zone_start) & (rewarded_bin_centres <= reward_zone_end)
        rewarded_speeds_in_space_in_reward_zone = rewarded_speeds_in_space[in_rz_mask]
        rewarded_speeds_in_space_in_reward_zone = rewarded_speeds_in_space_in_reward_zone[~np.isnan(rewarded_speeds_in_space_in_reward_zone)]
        minimum_speed = min(rewarded_speeds_in_space_in_reward_zone)
        minimum_speeds_in_rz.append(minimum_speed)
        speeds_in_rz.extend(rewarded_speeds_in_space_in_reward_zone.tolist())

    speeds_in_rz = np.array(speeds_in_rz)
    minimum_speeds_in_rz = np.array(minimum_speeds_in_rz)
    mean, sigma = np.nanmean(speeds_in_rz), np.nanstd(speeds_in_rz)
    #mean, sigma = np.nanmean(minimum_speeds_in_rz), np.nanstd(minimum_speeds_in_rz)
    interval = stats.norm.interval(0.95, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]

    hit_miss_try =[]
    avg_speed_in_rz =[]
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_process_position_data = processed_position_data[(processed_position_data.trial_number == trial_number)]
        track_speed = trial_process_position_data["avg_speed_on_track"].iloc[0]
        trial_speeds_in_space = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_process_position_data['speeds_binned_in_space'])
        trial_bin_centres = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_process_position_data['position_bin_centres'])
        in_rz_mask = (trial_bin_centres > reward_zone_start) & (trial_bin_centres <= reward_zone_end)
        trial_speeds_in_reward_zone = trial_speeds_in_space[in_rz_mask]
        trial_speeds_in_reward_zone = trial_speeds_in_reward_zone[~np.isnan(trial_speeds_in_reward_zone)]
        avg_trial_speed_in_reward_zone = np.mean(trial_speeds_in_reward_zone)

        if (trial_process_position_data["rewarded"].iloc[0] == True) and (track_speed>20):
            hit_miss_try.append("hit")
        elif (avg_trial_speed_in_reward_zone >= lower) and (avg_trial_speed_in_reward_zone <= upper) and (track_speed>20):
            hit_miss_try.append("try")
        elif (avg_trial_speed_in_reward_zone < lower) or (avg_trial_speed_in_reward_zone > upper) and (track_speed>20):
            hit_miss_try.append("miss")
        else:
            hit_miss_try.append("rejected")

        avg_speed_in_rz.append(avg_trial_speed_in_reward_zone)

    processed_position_data["hit_miss_try"] = hit_miss_try
    processed_position_data["avg_speed_in_rz"] = avg_speed_in_rz
    return processed_position_data, upper

def add_hit_miss_try(processed_position_data, track_length):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30

    rewarded_processed_position_data = processed_position_data[(processed_position_data["rewarded"] == True)]
    minimum_speeds_in_rz = []
    speeds_in_rz = []
    for trial_number in np.unique(rewarded_processed_position_data["trial_number"]):
        trial_rewarded_processed_position_data = rewarded_processed_position_data[rewarded_processed_position_data["trial_number"] == trial_number]

        rewarded_speeds_in_space = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_rewarded_processed_position_data['speeds_binned_in_space'])
        rewarded_bin_centres = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_rewarded_processed_position_data['position_bin_centres'])
        in_rz_mask = (rewarded_bin_centres > reward_zone_start) & (rewarded_bin_centres <= reward_zone_end)
        rewarded_speeds_in_space_in_reward_zone = rewarded_speeds_in_space[in_rz_mask]
        rewarded_speeds_in_space_in_reward_zone = rewarded_speeds_in_space_in_reward_zone[~np.isnan(rewarded_speeds_in_space_in_reward_zone)]
        minimum_speed = min(rewarded_speeds_in_space_in_reward_zone)
        minimum_speeds_in_rz.append(minimum_speed)
        speeds_in_rz.extend(rewarded_speeds_in_space_in_reward_zone.tolist())

    speeds_in_rz = np.array(speeds_in_rz)
    minimum_speeds_in_rz = np.array(minimum_speeds_in_rz)
    mean, sigma = np.nanmean(speeds_in_rz), np.nanstd(speeds_in_rz)
    #mean, sigma = np.nanmean(minimum_speeds_in_rz), np.nanstd(minimum_speeds_in_rz)
    interval = stats.norm.interval(0.95, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]

    hit_miss_try =[]
    avg_speed_in_rz =[]
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_process_position_data = processed_position_data[(processed_position_data.trial_number == trial_number)]
        trial_speeds_in_space = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_process_position_data['speeds_binned_in_space'])
        trial_bin_centres = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_process_position_data['position_bin_centres'])
        in_rz_mask = (trial_bin_centres > reward_zone_start) & (trial_bin_centres <= reward_zone_end)
        trial_speeds_in_reward_zone = trial_speeds_in_space[in_rz_mask]
        trial_speeds_in_reward_zone = trial_speeds_in_reward_zone[~np.isnan(trial_speeds_in_reward_zone)]
        avg_trial_speed_in_reward_zone = np.mean(trial_speeds_in_reward_zone)

        if trial_process_position_data["rewarded"].iloc[0] == True:
            hit_miss_try.append("hit")
        elif (avg_trial_speed_in_reward_zone >= lower) and (avg_trial_speed_in_reward_zone <= upper):
            hit_miss_try.append("try")
        elif (avg_trial_speed_in_reward_zone < lower) or (avg_trial_speed_in_reward_zone > upper):
            hit_miss_try.append("miss")
        else:
            hit_miss_try.append("miss")

        avg_speed_in_rz.append(avg_trial_speed_in_reward_zone)

    processed_position_data["hit_miss_try"] = hit_miss_try
    processed_position_data["avg_speed_in_rz"] = avg_speed_in_rz
    return processed_position_data, upper

def calculate_putative_fields(cluster_spike_data, position_data, track_length):
    firing_rate_maps_per_trial = cluster_spike_data["fr_binned_in_space"].iloc[0]
    firing_rate_map_bin_centres_per_trial = cluster_spike_data["fr_binned_in_space_bin_centres"].iloc[0]

    firing_rate_across_trials = np.array(firing_rate_maps_per_trial).flatten()
    firing_rate_map_bin_centres_across_trials = np.array(firing_rate_map_bin_centres_per_trial).flatten()

    # define the global maximum
    global_maxima_bin_idx = np.nanargmax(firing_rate_across_trials)
    global_maxima = firing_rate_across_trials[global_maxima_bin_idx]
    field_threshold = 0.2*global_maxima

    # detect local maxima
    local_maxima_idx, _ = signal.find_peaks(firing_rate_across_trials, height=0)

    # detect fields
    firing_fields = []
    firing_field_sizes = []
    for i in local_maxima_idx:
        neighbouring_local_mins = find_neighbouring_minima(firing_rate_across_trials, i)
        closest_minimum_bin_idx = neighbouring_local_mins[np.argmin(np.abs(neighbouring_local_mins-i))]
        field_size_in_bins = neighbouring_local_mins[1]-neighbouring_local_mins[0]
        field_size = field_size_in_bins*settings.vr_grid_analysis_bin_size

        if firing_rate_across_trials[i] - firing_rate_across_trials[closest_minimum_bin_idx] > field_threshold:
            # calculate the fields centre of mass
            field = firing_rate_across_trials[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
            field_bins = firing_rate_map_bin_centres_across_trials[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
            field_weights = field/np.sum(field)
            field_com = np.sum(field_weights*field_bins)
            field_com = firing_rate_map_bin_centres_across_trials[i]
            firing_fields.append(field_com)
            firing_field_sizes.append(field_size)

    firing_field_sizes = np.array(firing_field_sizes)
    firing_field_locations_elapsed_distance = np.array(firing_fields)
    firing_fields_trial_numbers = ((firing_field_locations_elapsed_distance//track_length)+1).astype(np.int64)
    firing_field_locations = firing_field_locations_elapsed_distance%track_length

    # assign fields to a trial number
    firing_fields_per_trial = []
    firing_field_sizes_per_trial = []
    for tn in np.arange(1, max(position_data["trial_number"])+1):
        firing_field_locations_per_trial = (firing_field_locations[firing_fields_trial_numbers == tn]).tolist()
        firing_field_sizes_per_trial = (firing_field_sizes[firing_fields_trial_numbers == tn]).tolist()

        firing_fields_per_trial.append(firing_field_locations_per_trial)
        firing_field_sizes_per_trial.append(firing_field_sizes_per_trial)

    # centre of mass of field used
    return firing_fields_per_trial, firing_field_sizes_per_trial

def calculate_putative_fields2(cluster_spike_data, position_data, track_length):
    firing_times=cluster_spike_data.firing_times/(settings.sampling_rate/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    if len(firing_times)==0:
        firing_rate_maps = np.zeros(int(track_length))
        return [], []

    trial_numbers = np.array(position_data['trial_number'].to_numpy())
    trial_types = np.array(position_data['trial_type'].to_numpy())
    time_seconds = np.array(position_data['time_seconds'].to_numpy())
    x_position_cm = np.array(position_data['x_position_cm'].to_numpy())
    x_position_cm_elapsed = x_position_cm+((trial_numbers-1)*track_length)

    instantaneous_firing_rate_per_ms = extract_instantaneous_firing_rate_for_spike2(cluster_spike_data) # returns firing rate per millisecond time bin
    instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[0:len(x_position_cm)]

    if not (len(instantaneous_firing_rate_per_ms) == len(trial_numbers)):
        # 0 pad until it is the same size (padding with 0 hz firing rate
        instantaneous_firing_rate_per_ms = np.append(instantaneous_firing_rate_per_ms, np.zeros(len(trial_numbers)-len(instantaneous_firing_rate_per_ms)))

    max_distance_elapsed = track_length*max(trial_numbers)
    numerator, bin_edges = np.histogram(x_position_cm_elapsed, bins=int(max_distance_elapsed/settings.vr_grid_analysis_bin_size), range=(0, max_distance_elapsed), weights=instantaneous_firing_rate_per_ms)
    denominator, bin_edges = np.histogram(x_position_cm_elapsed, bins=int(max_distance_elapsed/settings.vr_grid_analysis_bin_size), range=(0, max_distance_elapsed))
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    firing_rate_map = numerator/denominator
    firing_rate_map = np.nan_to_num(firing_rate_map)

    local_maxima_bin_idx = signal.argrelextrema(firing_rate_map, np.greater)[0]
    global_maxima_bin_idx = np.nanargmax(firing_rate_map)
    global_maxima = firing_rate_map[global_maxima_bin_idx]
    field_threshold = 0.2*global_maxima

    # detect fields
    firing_fields = []
    firing_field_sizes = []
    for local_maximum_idx in local_maxima_bin_idx:
        neighbouring_local_mins = find_neighbouring_minima(firing_rate_map, local_maximum_idx)
        closest_minimum_bin_idx = neighbouring_local_mins[np.argmin(np.abs(neighbouring_local_mins-local_maximum_idx))]
        field_size_in_bins = neighbouring_local_mins[1]-neighbouring_local_mins[0]
        field_size = field_size_in_bins*settings.vr_grid_analysis_bin_size

        if firing_rate_map[local_maximum_idx] - firing_rate_map[closest_minimum_bin_idx] > field_threshold:
            #firing_field.append(neighbouring_local_mins)

            field =  firing_rate_map[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
            field_bins = bin_centres[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
            field_weights = field/np.sum(field)
            field_com = np.sum(field_weights*field_bins)
            firing_fields.append(field_com)
            firing_field_sizes.append(field_size)

    firing_field_sizes = np.array(firing_field_sizes)
    firing_field_locations_elapsed_distance = np.array(firing_fields)
    firing_fields_trial_numbers = ((firing_field_locations_elapsed_distance//track_length)+1).astype(np.int64)
    firing_field_locations = firing_field_locations_elapsed_distance%track_length

    # assign fields to a trial number
    firing_fields_per_trial = []
    firing_field_sizes_per_trial = []
    for tn in np.arange(1, max(position_data["trial_number"])+1):
        firing_field_locations_per_trial = (firing_field_locations[firing_fields_trial_numbers == tn]).tolist()
        firing_field_sizes_per_trial = (firing_field_sizes[firing_fields_trial_numbers == tn]).tolist()

        firing_fields_per_trial.append(firing_field_locations_per_trial)
        firing_field_sizes_per_trial.append(firing_field_sizes_per_trial)

    # centre of mass of field used
    return firing_fields_per_trial, firing_field_sizes_per_trial


def find_neighbouring_minima(firing_rate_map, local_maximum_idx):
    # walk right
    local_min_right = local_maximum_idx
    local_min_right_found = False
    for i in np.arange(local_maximum_idx, len(firing_rate_map)): #local max to end
        if local_min_right_found == False:
            if np.isnan(firing_rate_map[i]):
                continue
            elif firing_rate_map[i] < firing_rate_map[local_min_right]:
                local_min_right = i
            elif firing_rate_map[i] > firing_rate_map[local_min_right]:
                local_min_right_found = True

    # walk left
    local_min_left = local_maximum_idx
    local_min_left_found = False
    for i in np.arange(0, local_maximum_idx)[::-1]: # local max to start
        if local_min_left_found == False:
            if np.isnan(firing_rate_map[i]):
                continue
            elif firing_rate_map[i] < firing_rate_map[local_min_left]:
                local_min_left = i
            elif firing_rate_map[i] > firing_rate_map[local_min_left]:
                local_min_left_found = True

    return (local_min_left, local_min_right)


def extract_instantaneous_firing_rate_for_spike(cluster_data, prm):
    firing_times=cluster_data.firing_times/(prm.get_sampling_rate()/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    bins = np.arange(0,np.max(firing_times)+500, 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    smoothened_instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)

    inds = np.digitize(firing_times, bins)

    ifr = []
    for i in inds:
        ifr.append(smoothened_instantaneous_firing_rate[i-1])

    smoothened_instantaneous_firing_rate_per_spike = np.array(ifr)
    return smoothened_instantaneous_firing_rate_per_spike

def extract_instantaneous_firing_rate_for_spike2(cluster_data):
    firing_times=cluster_data.firing_times/(settings.sampling_rate/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    bins = np.arange(0,np.max(firing_times)+2000, 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)

    return instantaneous_firing_rate

def extract_instantaneous_firing_rate_for_spike3(cluster_data):
    firing_times=cluster_data.firing_times/(settings.sampling_rate/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    bins = np.arange(0,np.max(firing_times)+2000, 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)

    return instantaneous_firing_rate


def bin_fr_in_space_for_field_analysis(spike_data, raw_position_data, track_length):

    # make an empty list of list for all firing rates binned in time for each cluster
    fr_binned_in_space = []
    fr_binned_in_space_bin_centres = []

    elapsed_distance_bins = np.arange(0, (track_length*max(raw_position_data["trial_number"]))+1,  settings.vr_grid_analysis_bin_size) # might be buggy with anything but 1cm space bins
    trial_numbers_raw = np.array(raw_position_data['trial_number'], dtype=np.int64)
    x_position_elapsed_cm = (track_length*(trial_numbers_raw-1))+np.array(raw_position_data['x_position_cm'], dtype="float64")
    x_dwell_time = np.array(raw_position_data['dwell_time_ms'], dtype="float64")

    for i, cluster_id in enumerate(spike_data.cluster_id):
        if len(elapsed_distance_bins)>1:
            spikes_x_position_cm = np.array(spike_data[spike_data["cluster_id"] == cluster_id]["x_position_cm"].iloc[0])
            trial_numbers = np.array(spike_data[spike_data["cluster_id"] == cluster_id]["trial_number"].iloc[0])

            # convert spike locations into elapsed distance
            spikes_x_position_elapsed_cm = (track_length*(trial_numbers-1))+spikes_x_position_cm

            # count the spikes in each space bin and normalise by the total time spent in that bin for the trial
            fr_hist, bin_edges = np.histogram(spikes_x_position_elapsed_cm, elapsed_distance_bins)
            fr_hist = fr_hist/(np.histogram(x_position_elapsed_cm, elapsed_distance_bins, weights=x_dwell_time)[0])

            # get location bin centres and ascribe them to their trial numbers
            bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

            # nans to zero and smooth
            fr_hist[np.isnan(fr_hist)] = 0

            fr_binned_in_space.append(fr_hist)
            fr_binned_in_space_bin_centres.append(bin_centres)
        else:
            fr_binned_in_space.append([])
            fr_binned_in_space_bin_centres.append([])

    spike_data["fr_binned_in_space"] = fr_binned_in_space
    spike_data["fr_binned_in_space_bin_centres"] = fr_binned_in_space_bin_centres

    return spike_data

def process_vr_grid(spike_data, raw_position_data, position_data, track_length):

    # temporary steps to improve position resolution of spikes
    #spike_data = add_position_x(spike_data, raw_position_data)
    spike_data = bin_fr_in_space_for_field_analysis(spike_data, raw_position_data, track_length)

    firing_fields_com = []
    firing_field_sizes = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        if len(cluster_df.firing_times.iloc[0])>1:
            fields_com, field_sizes = calculate_putative_fields(cluster_df, position_data, track_length)
            firing_fields_com.append(fields_com)
            firing_field_sizes.append(field_sizes)
        else:
            firing_fields_com.append([])
            firing_field_sizes.append([])

    spike_data["firing_fields_com"] = firing_fields_com
    spike_data["firing_field_sizes"] = firing_field_sizes

    return spike_data

def analyse_fields(spike_data, processed_position_data, track_length, pre_post_rz=""):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30

    fields_per_trial_hmt_by_trial_type = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_field_coms = np.array(cluster_df["firing_fields_com"].iloc[0])

        fields_per_trial_cluster = np.zeros((3,3))
        for i, tt in enumerate([0,1,2]):
            for j, hmt in enumerate(["hit", "miss", "try"]):
                subset_processed_position_data = processed_position_data[(processed_position_data["trial_type"] == tt)]
                subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["hit_miss_try"] == hmt)]
                subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])

                n_fields = 0
                if len(subset_trial_numbers)>0:
                    for tn in subset_trial_numbers:
                        firing_field_locations = np.array(firing_field_coms[tn-1])

                        # count only fields before or after reward zone if specified
                        if pre_post_rz == "_pre_rz":
                            firing_field_locations = firing_field_locations[firing_field_locations < reward_zone_start]
                        elif pre_post_rz == "_post_rz":
                            firing_field_locations = firing_field_locations[firing_field_locations > reward_zone_end]

                        n_fields += len(firing_field_locations)
                    fields_per_trial = n_fields/len(subset_trial_numbers)
                else:
                    fields_per_trial = np.nan

                fields_per_trial_cluster[i,j] = fields_per_trial

        fields_per_trial_hmt_by_trial_type.append(fields_per_trial_cluster.tolist())

    spike_data["fields_per_trial_hmt_by_trial_type"+pre_post_rz] = fields_per_trial_hmt_by_trial_type
    return spike_data

def calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type):

    cluster_firing_com = np.array(cluster_df["fields_com"].iloc[0])
    cluster_firing_com_trial_types = np.array(cluster_df["fields_com_trial_type"].iloc[0])

    if trial_type == "beaconed":
        n_trials = processed_position_data.beaconed_total_trial_number.iloc[0]
        firing_com = cluster_firing_com[cluster_firing_com_trial_types == 0]
    elif trial_type == "non-beaconed":
        n_trials = processed_position_data.nonbeaconed_total_trial_number.iloc[0]
        firing_com = cluster_firing_com[cluster_firing_com_trial_types == 1]
    elif trial_type == "probe":
        n_trials = processed_position_data.probe_total_trial_number.iloc[0]
        firing_com = cluster_firing_com[cluster_firing_com_trial_types == 2]
    else:
        print("no valid trial type was given")

    if n_trials==0:
        return np.nan
    else:
        return len(firing_com)/n_trials

def process_vr_field_stats(spike_data, processed_position_data):
    n_beaconed_fields_per_trial = []
    n_nonbeaconed_fields_per_trial = []
    n_probe_fields_per_trial = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        n_beaconed_fields_per_trial.append(calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type="beaconed"))
        n_nonbeaconed_fields_per_trial.append(calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type="non-beaconed"))
        n_probe_fields_per_trial.append(calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type="probe"))

    spike_data["n_beaconed_fields_per_trial"] = n_beaconed_fields_per_trial
    spike_data["n_nonbeaconed_fields_per_trial"] = n_nonbeaconed_fields_per_trial
    spike_data["n_probe_fields_per_trial"] = n_probe_fields_per_trial

    return spike_data

def process_vr_field_distances(spike_data, track_length):
    distance_between_fields = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        cluster_firing_com = np.array(cluster_df["fields_com"].iloc[0])
        cluster_firing_com_trial_types = np.array(cluster_df["fields_com_trial_type"].iloc[0])
        cluster_firing_com_trial_numbers = np.array(cluster_df["fields_com_trial_number"].iloc[0])

        distance_covered = (cluster_firing_com_trial_numbers*track_length)-track_length #total elapsed distance
        cluster_firing_com = cluster_firing_com+distance_covered

        cluster_firing_com_distance_between = np.diff(cluster_firing_com)
        distance_between_fields.append(cluster_firing_com_distance_between)

    spike_data["distance_between_fields"] = distance_between_fields

    return spike_data

def find_paired_recording(recording_path, of_recording_path_list):
    mouse=recording_path.split("/")[-1].split("_")[0]
    training_day=recording_path.split("/")[-1].split("_")[1]

    for paired_recording in of_recording_path_list:
        paired_mouse=paired_recording.split("/")[-1].split("_")[0]
        paired_training_day=paired_recording.split("/")[-1].split("_")[1]

        if (mouse == paired_mouse) and (training_day == paired_training_day):
            return paired_recording, True
    return None, False

def find_set(a,b):
    return set(a) & set(b)

def plot_spatial_autocorrelogram_fr(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length, suffix=""):
    print('plotting spike spatial autocorrelogram fr...')
    save_path = output_path + '/Figures/spatial_autocorrelograms_fr'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    n_trials = len(processed_position_data)
    # get distances from raw position data which has been smoothened
    rpd = np.asarray(raw_position_data["x_position_cm"])
    tn = np.asarray(raw_position_data["trial_number"])
    elapsed_distance30 = rpd+(tn*track_length)-track_length

    # get denominator and handle nans
    denominator, _ = np.histogram(elapsed_distance30, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        recording_length_sampling_points = int(cluster_spike_data['recording_length_sampling_points'].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        firing_locations_cluster = np.array(cluster_spike_data["x_position_cm"].iloc[0])
        firing_trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])
        firing_locations_cluster_elapsed = firing_locations_cluster+(firing_trial_numbers*track_length)-track_length

        if len(firing_times_cluster)>1:
            numerator, bin_edges = np.histogram(firing_locations_cluster_elapsed, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))
            fr = numerator/denominator
            elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
            trial_numbers_by_bin=((0.5*(bin_edges[1:]+bin_edges[:-1])//track_length)+1).astype(np.int32)
            gauss_kernel = Gaussian1DKernel(stddev=1)

            # remove nan values that coincide with start and end of the track before convolution
            fr[fr==np.inf] = np.nan
            nan_mask = ~np.isnan(fr)
            fr = fr[nan_mask]
            trial_numbers_by_bin = trial_numbers_by_bin[nan_mask]
            elapsed_distance = elapsed_distance[nan_mask]

            fr = convolve(fr, gauss_kernel)
            fr = moving_sum(fr, window=2)/2
            fr = np.append(fr, np.zeros(len(elapsed_distance)-len(fr)))
            normalised_elapsed_distance = elapsed_distance/track_length

            autocorr_window_size = track_length*4
            lags = np.arange(0, autocorr_window_size, 1) # were looking at 10 timesteps back and 10 forward
            autocorrelogram = []
            for i in range(len(lags)):
                fr_lagged = fr[i:]
                corr = stats.pearsonr(fr_lagged, fr[:len(fr_lagged)])[0]
                autocorrelogram.append(corr)
            autocorrelogram= np.array(autocorrelogram)
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            for f in range(1,6):
                ax.axvline(x=track_length*f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
            ax.axhline(y=0, color="black", linewidth=2,linestyle="dashed")
            ax.plot(lags, autocorrelogram, color="black", linewidth=3)
            plt.ylabel('Spatial Autocorrelation', fontsize=25, labelpad = 10)
            plt.xlabel('Lag (cm)', fontsize=25, labelpad = 10)
            plt.xlim(0,(track_length*2)+3)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_ylim([np.floor(min(autocorrelogram[5:])*10)/10,np.ceil(max(autocorrelogram[5:])*10)/10])
            if np.floor(min(autocorrelogram[5:])*10)/10 < 0:
                ax.set_yticks([np.floor(min(autocorrelogram[5:])*10)/10, 0, np.ceil(max(autocorrelogram[5:])*10)/10])
            else:
                ax.set_yticks([-0.1, 0, np.ceil(max(autocorrelogram[5:])*10)/10])
            tick_spacing = track_length
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_autocorrelogram_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
            plt.close()

    return spike_data


def plot_spatial_autocorrelogram(spike_data, processed_position_data, output_path, track_length, suffix=""):
    print('plotting spike spatial autocorrelogram...')
    save_path = output_path + '/Figures/spatial_autocorrelograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # get trial numbers to use from processed_position_data
    trial_number_to_use = np.unique(processed_position_data["trial_number"])

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])
        x_position_cluster = np.array(cluster_spike_data["x_position_cm"].iloc[0])

        # get and apply set_mask (we are only concerned with the trial numbers in processed_position_data)
        set_mask = np.isin(trial_numbers, trial_number_to_use)
        firing_times_cluster = firing_times_cluster[set_mask]
        trial_numbers = trial_numbers[set_mask]
        x_position_cluster = x_position_cluster[set_mask]

        if len(firing_times_cluster)>1:
            lap_distance_covered = (trial_numbers*track_length)-track_length #total elapsed distance
            x_position_cluster = x_position_cluster+lap_distance_covered
            x_position_cluster = x_position_cluster[~np.isnan(x_position_cluster)]
            x_position_cluster_bins = np.floor(x_position_cluster).astype(int)

            autocorr_window_size = 400
            lags = np.arange(0, autocorr_window_size, 1).astype(int) # were looking at 10 timesteps back and 10 forward

            autocorrelogram = np.array([])
            for lag in lags:
                correlated = len(find_set(x_position_cluster_bins+lag, x_position_cluster_bins))
                autocorrelogram = np.append(autocorrelogram, correlated)

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.bar(lags[1:], autocorrelogram[1:], edgecolor="black", align="edge")
            plt.ylabel('Counts', fontsize=20, labelpad = 10)
            plt.xlabel('Lag (cm)', fontsize=20, labelpad = 10)
            plt.xlim(0,400)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            Edmond.plot_utility2.style_vr_plot(ax, x_max=max(autocorrelogram[1:]))
            plt.locator_params(axis = 'x', nbins  = 8)
            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_autocorrelogram_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
            plt.close()

    return spike_data


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

def reduce_digits(numeric_float, n_digits=6):
    #if len(list(str(numeric_float))) > n_digits:
    scientific_notation = "{:.1e}".format(numeric_float)
    return scientific_notation


def plot_moving_lomb_scargle_periodogram(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length):
    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    shuffle_data = pd.DataFrame()

    # get trial numbers to use from processed_position_data
    trial_number_to_use = np.unique(processed_position_data["trial_number"])
    n_trials = len(processed_position_data)

    # get distances from raw position data which has been smoothened
    rpd = np.asarray(raw_position_data["x_position_cm"])
    tn = np.asarray(raw_position_data["trial_number"])
    elapsed_distance30 = rpd+(tn*track_length)-track_length

    # get denominator and handle nans
    denominator, _ = np.histogram(elapsed_distance30, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))

    freqs = []
    SNRs = []
    avg_powers = []
    all_powers = []
    all_centre_trials=[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        recording_length_sampling_points = int(cluster_spike_data['recording_length_sampling_points'].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        firing_locations_cluster = np.array(cluster_spike_data["x_position_cm"].iloc[0])
        firing_trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])
        firing_locations_cluster_elapsed = firing_locations_cluster+(firing_trial_numbers*track_length)-track_length

        if len(firing_times_cluster)>1:
            numerator, bin_edges = np.histogram(firing_locations_cluster_elapsed, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))
            fr = numerator/denominator
            elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
            trial_numbers_by_bin=((0.5*(bin_edges[1:]+bin_edges[:-1])//track_length)+1).astype(np.int32)
            gauss_kernel = Gaussian1DKernel(stddev=1)

            # remove nan values that coincide with start and end of the track before convolution
            fr[fr==np.inf] = np.nan
            nan_mask = ~np.isnan(fr)
            fr = fr[nan_mask]
            trial_numbers_by_bin = trial_numbers_by_bin[nan_mask]
            elapsed_distance = elapsed_distance[nan_mask]

            fr = convolve(fr, gauss_kernel)
            fr = moving_sum(fr, window=2)/2
            fr = np.append(fr, np.zeros(len(elapsed_distance)-len(fr)))

            # make and apply the set mask
            set_mask = np.isin(trial_numbers_by_bin, trial_number_to_use)
            fr = fr[set_mask]
            elapsed_distance = elapsed_distance[set_mask]

            # construct the lomb-scargle periodogram
            step = 0.02
            frequency = np.arange(0.1, 5+step, step)
            sliding_window_size=track_length*3

            powers = []
            centre_distances = []
            indices_to_test = np.arange(0, len(fr)-sliding_window_size, 1, dtype=np.int64)[::2]
            for m in indices_to_test:
                ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr[m:m+sliding_window_size])
                power = ls.power(frequency)
                powers.append(power.tolist())
                centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
            powers = np.array(powers)
            centre_trials = np.round(np.array(centre_distances)).astype(np.int64)
            #centre_trials = np.array(centre_distances)

            avg_power = np.nanmean(powers, axis=0)
            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
            max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
            max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            for f in range(1,6):
                ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
            subset_trial_numbers = processed_position_data["trial_number"]
            subset_trial_numbers = np.asarray(subset_trial_numbers)
            subset_mask = np.isin(centre_trials, subset_trial_numbers)
            subset_mask = np.vstack([subset_mask]*len(powers[0])).T
            subset_powers = powers.copy()
            subset_powers[subset_mask == False] = np.nan
            avg_subset_powers = np.nanmean(subset_powers, axis=0)
            sem_subset_powers = stats.sem(subset_powers, axis=0, nan_policy="omit")
            ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color="black", alpha=0.3)
            ax.plot(frequency, avg_subset_powers, color="black", linestyle="solid", linewidth=1)
            plt.ylabel('Periodic Power', fontsize=20, labelpad = 10)
            plt.xlabel("Track Frequency", fontsize=20, labelpad = 10)
            plt.xlim(0,5.05)
            ax.set_xticks([0,5])
            ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
            ax.set_ylim(bottom=0)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_scargle_avg_periodogram_Cluster_' + str(cluster_id) + '.png', dpi=300)
            plt.close()


            n_x_ticks = int(max(centre_trials)//50)+1
            x_tick_locs= np.linspace(np.ceil(min(centre_trials)), max(centre_trials), n_x_ticks, dtype=np.int64)
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            powers[np.isnan(powers)] = 0
            X, Y = np.meshgrid(centre_trials, frequency)
            #powers = np.flip(powers, axis=0)
            cmap = plt.cm.get_cmap("inferno")
            ax.pcolormesh(X, Y, powers.T, cmap=cmap, shading="flat")
            for f in range(1,6):
                ax.axhline(y=f, color="white", linewidth=2,linestyle="dotted")
            #ax.imshow(powers.T, origin='lower', aspect="auto", cmap="jet")
            plt.ylabel('Track Frequency', fontsize=20, labelpad = 10)
            plt.xlabel('Centre Trial', fontsize=20, labelpad = 10)
            ax.set_yticks([0, 1, 2, 3, 4, 5])
            ax.set_xticks(x_tick_locs.tolist())
            #ax.set_yticklabels(["0", "", "", "", "", "5"])
            ax.set_ylim([0.1,5])
            ax.set_xlim([min(centre_trials), max(centre_trials)])
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_scargle_periodogram_Cluster_' + str(cluster_id) +'.png', dpi=300)
            plt.close()



            #====================================================# Attempt bootstrapped approach using standard spike time shuffling procedure
            np.random.seed(0)
            for i in range(1):
                random_firing_additions = np.random.randint(low=int(20*settings.sampling_rate), high=int(580*settings.sampling_rate), size=len(firing_times_cluster))
                shuffled_firing_times = firing_times_cluster + random_firing_additions
                shuffled_firing_times[shuffled_firing_times >= recording_length_sampling_points] = shuffled_firing_times[shuffled_firing_times >= recording_length_sampling_points] - recording_length_sampling_points # wrap around
                shuffled_firing_locations_elapsed = elapsed_distance30[shuffled_firing_times.astype(np.int64)]
                numerator, bin_edges = np.histogram(shuffled_firing_locations_elapsed, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))
                fr = numerator/denominator
                elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length

                # remove nan values that coincide with start and end of the track before convolution and then convolve
                fr[fr==np.inf] = np.nan
                nan_mask = ~np.isnan(fr)
                fr = fr[nan_mask]
                elapsed_distance = elapsed_distance[nan_mask]
                fr = convolve(fr, gauss_kernel)
                fr = moving_sum(fr, window=2)/2
                fr = np.append(fr, np.zeros(len(elapsed_distance)-len(fr)))

                # make and apply the set mask
                fr = fr[set_mask]
                elapsed_distance = elapsed_distance[set_mask]

                # run moving window lomb on the shuffled firing
                shuffle_powers = []
                shuffle_centre_distances = []
                indices_to_test = np.arange(0, len(fr)-sliding_window_size, 1, dtype=np.int64)[::2]
                for m in indices_to_test:
                    ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr[m:m+sliding_window_size])
                    shuffle_power = ls.power(frequency)
                    shuffle_powers.append(shuffle_power.tolist())
                    shuffle_centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
                shuffle_powers = np.array(shuffle_powers)
                #shuffle_centre_distances = np.round(shuffle_centre_distances).astype(np.int64)
                shuffle_centre_distances = np.array(shuffle_centre_distances)
                avg_shuffle_powers = np.nanmean(shuffle_powers, axis=0)
                max_shuffle_freq = frequency[np.argmax(avg_shuffle_powers)]
                max_shuffle_power = avg_shuffle_powers[np.argmax(avg_shuffle_powers)]
                single_shuffle=pd.DataFrame()
                single_shuffle["cluster_id"] = [cluster_id]
                single_shuffle["shuffle_id"] = [i]
                single_shuffle["avg_shuffle_powers"] = [avg_shuffle_powers]
                single_shuffle["max_shuffle_freq"] = [max_shuffle_freq]
                single_shuffle["max_shuffle_power"] = [max_shuffle_power]
                single_shuffle["MOVING_LOMB_all_powers"] = [shuffle_powers]
                single_shuffle["MOVING_LOMB_all_centre_trials"] = [shuffle_centre_distances]
                single_shuffle["firing_times"] = [shuffled_firing_times.tolist()]
                shuffle_data = pd.concat([shuffle_data, single_shuffle], ignore_index=True)

            '''
            #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            #########################################################TRIAL NUMBER ASSAY######################################################
        
            n_sample_assay = []
            n_sample_assay_freq = []
            samples_to_assay = [1, 4, 8, 16, 32, 64, 100, 200, 500, 750, 1000, int(len(shuffle_powers)/2), len(shuffle_powers)-1]
            for n_samples in samples_to_assay:
                permuation_assay_power = []
                permuation_assay_freq = []
                for permuation in range(50):
                    np.random.seed(permuation)
                    TNA_shuffle_powers_idx = np.random.choice(np.arange(0, len(shuffle_powers)), size=n_samples, replace=False).astype(np.int64)
                    TNA_shuffle_powers = shuffle_powers[TNA_shuffle_powers_idx]
                    TNA_avg_shuffle_powers = np.nanmean(TNA_shuffle_powers, axis=0)
                    TNA_max_shuffle_freq = frequency[np.argmax(TNA_avg_shuffle_powers)]
                    TNA_max_shuffle_power = TNA_avg_shuffle_powers[np.argmax(TNA_avg_shuffle_powers)]
                    permuation_assay_power.append(TNA_max_shuffle_power)
                    permuation_assay_freq.append(TNA_max_shuffle_freq)
                n_sample_assay.append(permuation_assay_power)
                n_sample_assay_freq.append(permuation_assay_freq)
            n_sample_assay = np.array(n_sample_assay)
            n_sample_assay_freq = np.array(n_sample_assay_freq)

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.fill_between(samples_to_assay, np.nanmean(n_sample_assay, axis=1)-np.nanstd(n_sample_assay, axis=1), np.nanmean(n_sample_assay, axis=1)+np.nanstd(n_sample_assay, axis=1), alpha=0.3, color="black")
            ax.plot(samples_to_assay, np.nanmean(n_sample_assay, axis=1), marker="o", color="black")
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.locator_params(axis='y', nbins=6)
            plt.locator_params(axis='x', nbins=3)
            ax.set_xlim(left=0, right=max(samples_to_assay))
            ax.set_ylim(bottom=0)
            plt.ylabel('Max Power', fontsize=20, labelpad = 10)
            plt.xlabel('N Samples', fontsize=20, labelpad = 10)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_trial_number_assay_shuffle_Cluster_' + str(cluster_id) +'.png', dpi=300)
            plt.close()

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.fill_between(samples_to_assay, np.nanmean(n_sample_assay_freq, axis=1)-np.nanstd(n_sample_assay_freq, axis=1), np.nanmean(n_sample_assay_freq, axis=1)+np.nanstd(n_sample_assay_freq, axis=1), alpha=0.3, color="black")
            ax.plot(samples_to_assay, np.nanmean(n_sample_assay_freq, axis=1), marker="o", color="black")
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.locator_params(axis='y', nbins=6)
            plt.locator_params(axis='x', nbins=3)
            ax.set_xlim(left=0, right=max(samples_to_assay))
            ax.set_ylim(bottom=0, top=10)
            plt.ylabel('Hz @ Max Power', fontsize=20, labelpad = 10)
            plt.xlabel('N Samples', fontsize=20, labelpad = 10)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_trial_number_freq_assay_shuffle_Cluster_' + str(cluster_id) +'.png', dpi=300)
            plt.close()
            #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            '''


            '''
            #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            #########################################################TRIAL NUMBER ASSAY######################################################

            n_sample_assay = []
            n_sample_assay_freq = []
            samples_to_assay = [1, 4, 8, 16, 32, 64, 100, 200, 500, 750, 1000, int(len(powers)/2), len(powers)-1]
            for n_samples in samples_to_assay:
                permuation_assay_power = []
                permuation_assay_freq = []
                for permuation in range(50):
                    np.random.seed(permuation)
                    TNA_powers_idx = np.random.choice(np.arange(0, len(powers)), size=n_samples, replace=False).astype(np.int64)
                    TNA_powers = powers[TNA_powers_idx]
                    TNA_avg_powers = np.nanmean(TNA_powers, axis=0)
                    TNA_max_freq = frequency[np.argmax(TNA_avg_powers)]
                    TNA_max_power = TNA_avg_powers[np.argmax(TNA_avg_powers)]
                    permuation_assay_power.append(TNA_max_power)
                    permuation_assay_freq.append(TNA_max_freq)
                n_sample_assay.append(permuation_assay_power)
                n_sample_assay_freq.append(permuation_assay_freq)
            n_sample_assay = np.array(n_sample_assay)
            n_sample_assay_freq = np.array(n_sample_assay_freq)

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.fill_between(samples_to_assay, np.nanmean(n_sample_assay, axis=1)-np.nanstd(n_sample_assay, axis=1), np.nanmean(n_sample_assay, axis=1)+np.nanstd(n_sample_assay, axis=1), alpha=0.3, color="blue")
            ax.plot(samples_to_assay, np.nanmean(n_sample_assay, axis=1), marker="o", color="blue")
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.locator_params(axis='y', nbins=6)
            plt.locator_params(axis='x', nbins=3)
            ax.set_xlim(left=0, right=max(samples_to_assay))
            ax.set_ylim(bottom=0)
            plt.ylabel('Max Power', fontsize=20, labelpad = 10)
            plt.xlabel('N Samples', fontsize=20, labelpad = 10)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_trial_number_assay_Cluster_' + str(cluster_id) +'.png', dpi=300)
            plt.close()

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.fill_between(samples_to_assay, np.nanmean(n_sample_assay_freq, axis=1)-np.nanstd(n_sample_assay_freq, axis=1), np.nanmean(n_sample_assay_freq, axis=1)+np.nanstd(n_sample_assay_freq, axis=1), alpha=0.3, color="blue")
            ax.plot(samples_to_assay, np.nanmean(n_sample_assay_freq, axis=1), marker="o", color="blue")
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.locator_params(axis='y', nbins=6)
            plt.locator_params(axis='x', nbins=3)
            ax.set_xlim(left=0, right=max(samples_to_assay))
            ax.set_ylim(bottom=0, top=10)
            plt.ylabel('Hz @ Max Power', fontsize=20, labelpad = 10)
            plt.xlabel('N Samples', fontsize=20, labelpad = 10)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_trial_number_freq_assay_Cluster_' + str(cluster_id) +'.png', dpi=300)
            plt.close()
            #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            '''

            freqs.append(max_SNR_freq)
            SNRs.append(max_SNR)
            avg_powers.append(avg_power)
            all_powers.append(powers)
            all_centre_trials.append(centre_trials)
        else:
            freqs.append(np.nan)
            SNRs.append(np.nan)
            avg_powers.append(np.nan)
            all_powers.append(np.nan)
            all_centre_trials.append(np.nan)

    spike_data["MOVING_LOMB_freqs"] = freqs
    spike_data["MOVING_LOMB_avg_power"] = avg_powers
    spike_data["MOVING_LOMB_SNR"] = SNRs
    spike_data["MOVING_LOMB_all_powers"] = all_powers
    spike_data["MOVING_LOMB_all_centre_trials"] = all_centre_trials
    return spike_data, shuffle_data


def plot_moving_lomb_scargle_periodogram_by_trial_condition(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length):
    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms_trial_condition'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    shuffle_data = pd.DataFrame()

    # get trial numbers to use from processed_position_data
    n_trials = len(processed_position_data)

    # get distances from raw position data which has been smoothened
    rpd = np.asarray(raw_position_data["x_position_cm"])
    tn = np.asarray(raw_position_data["trial_number"])
    elapsed_distance30 = rpd+(tn*track_length)-track_length

    # get denominator and handle nans
    denominator, _ = np.histogram(elapsed_distance30, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        recording_length_sampling_points = int(cluster_spike_data['recording_length_sampling_points'].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        firing_locations_cluster = np.array(cluster_spike_data["x_position_cm"].iloc[0])
        firing_trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])
        firing_locations_cluster_elapsed = firing_locations_cluster+(firing_trial_numbers*track_length)-track_length

        for tt in [0,1]:
            for hmt in ["hit", "try", "miss"]:
                subset_processed_position_data = processed_position_data[(processed_position_data["trial_type"] == tt)]
                subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["hit_miss_try"] == hmt)]

                if len(firing_times_cluster)>1:
                    numerator, bin_edges = np.histogram(firing_locations_cluster_elapsed, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))
                    fr = numerator/denominator
                    elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
                    trial_numbers_by_bin=((0.5*(bin_edges[1:]+bin_edges[:-1])//track_length)+1).astype(np.int32)
                    gauss_kernel = Gaussian1DKernel(stddev=1)

                    # remove nan values that coincide with start and end of the track before convolution
                    fr[fr==np.inf] = np.nan
                    nan_mask = ~np.isnan(fr)

                    fr = fr[nan_mask]
                    trial_numbers_by_bin = trial_numbers_by_bin[nan_mask]
                    elapsed_distance = elapsed_distance[nan_mask]

                    fr = convolve(fr, gauss_kernel)
                    fr = moving_sum(fr, window=2)/2
                    fr = np.append(fr, np.zeros(len(elapsed_distance)-len(fr)))

                    # make and apply the set mask
                    trial_number_to_use = np.asarray(subset_processed_position_data["trial_number"])
                    set_mask = np.isin(trial_numbers_by_bin, trial_number_to_use)
                    fr = fr[set_mask]
                    elapsed_distance = elapsed_distance[set_mask]

                    # construct the lomb-scargle periodogram
                    step = 0.02
                    frequency = np.arange(0.1, 5+step, step)
                    sliding_window_size=track_length*3

                    powers = []
                    centre_distances = []
                    indices_to_test = np.arange(0, len(fr)-sliding_window_size, 1, dtype=np.int64)[::10]
                    for m in indices_to_test:
                        ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr[m:m+sliding_window_size])
                        power = ls.power(frequency)
                        powers.append(power.tolist())
                        centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
                    powers = np.array(powers)
                    centre_trials = np.round(np.array(centre_distances)).astype(np.int64)
                    #centre_trials = np.array(centre_distances)

                    avg_power = np.nanmean(powers, axis=0)
                    max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
                    max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
                    max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))

                    fig = plt.figure(figsize=(6,6))
                    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
                    for f in range(1,6):
                        ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
                    subset_trial_numbers = np.array(subset_processed_position_data["trial_number"])
                    subset_mask = np.isin(centre_trials, subset_trial_numbers)
                    subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                    subset_powers = powers.copy()
                    subset_powers[subset_mask == False] = np.nan
                    avg_subset_powers = np.nanmean(subset_powers, axis=0)
                    sem_subset_powers = stats.sem(subset_powers, axis=0, nan_policy="omit")
                    ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color="black", alpha=0.3)
                    ax.plot(frequency, avg_subset_powers, color="black", linestyle="solid", linewidth=1)
                    plt.ylabel('Periodic Power', fontsize=20, labelpad = 10)
                    plt.xlabel("Track Frequency", fontsize=20, labelpad = 10)
                    plt.xlim(0,5.05)
                    ax.set_xticks([0,5])
                    ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
                    ax.set_ylim(bottom=0)
                    ax.yaxis.set_ticks_position('left')
                    ax.xaxis.set_ticks_position('bottom')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
                    plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_scargle_avg_periodogram_Cluster_' + str(cluster_id) + '_hmt_'+hmt+'_tt_'+str(tt)+'.png', dpi=300)
                    plt.close()


                    n_x_ticks = int(max(centre_trials)//50)+1
                    x_tick_locs= np.linspace(np.ceil(min(centre_trials)), max(centre_trials), n_x_ticks, dtype=np.int64)
                    fig = plt.figure(figsize=(6,6))
                    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
                    powers[np.isnan(powers)] = 0
                    X, Y = np.meshgrid(centre_trials, frequency)
                    #powers = np.flip(powers, axis=0)
                    cmap = plt.cm.get_cmap("inferno")
                    ax.pcolormesh(X, Y, powers.T, cmap=cmap, shading="flat")
                    for f in range(1,6):
                        ax.axhline(y=f, color="white", linewidth=2,linestyle="dotted")
                    #ax.imshow(powers.T, origin='lower', aspect="auto", cmap="jet")
                    plt.ylabel('Track Frequency', fontsize=20, labelpad = 10)
                    plt.xlabel('Centre Trial', fontsize=20, labelpad = 10)
                    ax.set_yticks([0, 1, 2, 3, 4, 5])
                    ax.set_xticks(x_tick_locs.tolist())
                    #ax.set_yticklabels(["0", "", "", "", "", "5"])
                    ax.set_ylim([0.1,5])
                    ax.set_xlim([min(centre_trials), max(centre_trials)])
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                    plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_scargle_periodogram_Cluster_' + str(cluster_id) + '_hmt_'+hmt+'_tt_'+str(tt)+'.png', dpi=300)
                    plt.close()
    return

def get_allocentric_peak(frequency, avg_subset_powers, tolerance=0.05):
    local_maxima_idx, _ = signal.find_peaks(avg_subset_powers, height=0, distance=5)
    power = 0; freq=0; i_max=0
    for i in local_maxima_idx:
        if (distance_from_integer(frequency[i])<=tolerance) and (avg_subset_powers[i]>power):
            freq = frequency[i]
            power = avg_subset_powers[i]
            i_max = i
    return freq, power, i_max

def get_egocentric_peak(frequency, avg_subset_powers, tolerance=0.05):
    local_maxima_idx, _ = signal.find_peaks(avg_subset_powers, height=0, distance=5)
    power = 0; freq=0; i_max=0
    for i in local_maxima_idx:
        if (distance_from_integer(frequency[i])>=tolerance) and (avg_subset_powers[i]>power):
            freq = frequency[i]
            power = avg_subset_powers[i]
            i_max = i
    return freq, power, i_max


def analyse_lomb_powers_ego_vs_allocentric(spike_data, processed_position_data):
    '''
    Requires the collumns of MOVING LOMB PERIODOGRAM
    This function takes the moving window periodogram and computes the average SNR, and for all combinations of trial types and hit miss try behaviours
    :param spike_data:
    :param processed_position_data:
    :param track_length:
    :return:
    '''

    for code in ["_allo", "_ego"]:
        step = 0.02
        frequency = np.arange(0.1, 5+step, step)

        SNRs = [];                     Freqs = []
        SNRs_all_beaconed = [];        Freqs_all_beaconed = []
        SNRs_all_nonbeaconed = [];     Freqs_all_nonbeaconed = []
        SNRs_all_probe = [];           Freqs_all_probe = []

        SNRs_beaconed_hits = [];       Freqs_beaconed_hits = []
        SNRs_nonbeaconed_hits = [];    Freqs_nonbeaconed_hits = []
        SNRs_probe_hits = [];          Freqs_probe_hits = []
        SNRs_all_hits = [];            Freqs_all_hits = []

        SNRs_beaconed_tries = [];      Freqs_beaconed_tries = []
        SNRs_nonbeaconed_tries = [];   Freqs_nonbeaconed_tries = []
        SNRs_probe_tries = [];         Freqs_probe_tries = []
        SNRs_all_tries = [];           Freqs_all_tries = []

        SNRs_beaconed_misses = [];     Freqs_beaconed_misses = []
        SNRs_nonbeaconed_misses = [];  Freqs_nonbeaconed_misses = []
        SNRs_probe_misses = [];        Freqs_probe_misses = []
        SNRs_all_misses =[];           Freqs_all_misses = []

        for index, spike_row in spike_data.iterrows():
            cluster_spike_data = spike_row.to_frame().T.reset_index(drop=True)
            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            avg_powers = np.nanmean(powers, axis=0)
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)

            if code == "_allo":
                _, _, code_i = get_allocentric_peak(frequency, avg_powers, tolerance=0.05)
            elif code == "_ego":
                _, _, code_i = get_egocentric_peak(frequency, avg_powers, tolerance=0.05)

            firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
            for tt in ["all", 0, 1, 2]:
                for hmt in ["all", "hit", "miss", "try"]:
                    subset_processed_position_data = processed_position_data.copy()
                    if tt != "all":
                        subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["trial_type"] == tt)]
                    if hmt != "all":
                        subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["hit_miss_try"] == hmt)]
                    subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])

                    if len(firing_times_cluster)>1:
                        if len(subset_trial_numbers)>0:
                            subset_mask = np.isin(centre_trials, subset_trial_numbers)
                            subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                            subset_powers = powers.copy()
                            subset_powers[subset_mask == False] = np.nan
                            avg_subset_powers = np.nanmean(subset_powers, axis=0)

                            max_SNR, max_SNR_freq = avg_subset_powers[code_i], frequency[code_i]
                        else:
                            max_SNR, max_SNR_freq = (np.nan, np.nan)
                    else:
                        max_SNR, max_SNR_freq = (np.nan, np.nan)

                    if (tt=="all") and (hmt=="hit"):
                        SNRs_all_hits.append(max_SNR)
                        Freqs_all_hits.append(max_SNR_freq)
                    elif (tt=="all") and (hmt=="try"):
                        SNRs_all_tries.append(max_SNR)
                        Freqs_all_tries.append(max_SNR_freq)
                    elif (tt=="all") and (hmt=="miss"):
                        SNRs_all_misses.append(max_SNR)
                        Freqs_all_misses.append(max_SNR_freq)
                    elif (tt=="all") and (hmt=="all"):
                        SNRs.append(max_SNR)
                        Freqs.append(max_SNR_freq)
                    elif (tt==0) and (hmt=="hit"):
                        SNRs_beaconed_hits.append(max_SNR)
                        Freqs_beaconed_hits.append(max_SNR_freq)
                    elif (tt==0) and (hmt=="try"):
                        SNRs_beaconed_tries.append(max_SNR)
                        Freqs_beaconed_tries.append(max_SNR_freq)
                    elif (tt==0) and (hmt=="miss"):
                        SNRs_beaconed_misses.append(max_SNR)
                        Freqs_beaconed_misses.append(max_SNR_freq)
                    elif (tt==0) and (hmt=="all"):
                        SNRs_all_beaconed.append(max_SNR)
                        Freqs_all_beaconed.append(max_SNR_freq)
                    elif (tt==1) and (hmt=="hit"):
                        SNRs_nonbeaconed_hits.append(max_SNR)
                        Freqs_nonbeaconed_hits.append(max_SNR_freq)
                    elif (tt==1) and (hmt=="try"):
                        SNRs_nonbeaconed_tries.append(max_SNR)
                        Freqs_nonbeaconed_tries.append(max_SNR_freq)
                    elif (tt==1) and (hmt=="miss"):
                        SNRs_nonbeaconed_misses.append(max_SNR)
                        Freqs_nonbeaconed_misses.append(max_SNR_freq)
                    elif (tt==1) and (hmt=="all"):
                        SNRs_all_nonbeaconed.append(max_SNR)
                        Freqs_all_nonbeaconed.append(max_SNR_freq)
                    elif (tt==2) and (hmt=="hit"):
                        SNRs_probe_hits.append(max_SNR)
                        Freqs_probe_hits.append(max_SNR_freq)
                    elif (tt==2) and (hmt=="try"):
                        SNRs_probe_tries.append(max_SNR)
                        Freqs_probe_tries.append(max_SNR_freq)
                    elif (tt==2) and (hmt=="miss"):
                        SNRs_probe_misses.append(max_SNR)
                        Freqs_probe_misses.append(max_SNR_freq)
                    elif (tt==2) and (hmt=="all"):
                        SNRs_all_probe.append(max_SNR)
                        Freqs_all_probe.append(max_SNR_freq)


        spike_data["ML_SNRs"+code] = SNRs;                                         spike_data["ML_Freqs"+code] = Freqs
        spike_data["ML_SNRs_all_beaconed"+code] = SNRs_all_beaconed;               spike_data["ML_Freqs_all_beaconed"+code] = Freqs_all_beaconed
        spike_data["ML_SNRs_all_nonbeaconed"+code] =SNRs_all_nonbeaconed;          spike_data["ML_Freqs_all_nonbeaconed"+code] = Freqs_all_nonbeaconed
        spike_data["ML_SNRs_all_probe"+code] =SNRs_all_probe;                      spike_data["ML_Freqs_all_probe"+code] = Freqs_all_probe

        spike_data["ML_SNRs_beaconed_hits"+code] =SNRs_beaconed_hits;              spike_data["ML_Freqs_beaconed_hits"+code] = Freqs_beaconed_hits
        spike_data["ML_SNRs_nonbeaconed_hits"+code] =SNRs_nonbeaconed_hits;        spike_data["ML_Freqs_nonbeaconed_hits"+code] = Freqs_nonbeaconed_hits
        spike_data["ML_SNRs_probe_hits"+code] =SNRs_probe_hits;                    spike_data["ML_Freqs_probe_hits"+code] = Freqs_probe_hits
        spike_data["ML_SNRs_all_hits"+code] =SNRs_all_hits;                        spike_data["ML_Freqs_all_hits"+code] = Freqs_all_hits

        spike_data["ML_SNRs_beaconed_tries"+code] =SNRs_beaconed_tries;            spike_data["ML_Freqs_beaconed_tries"+code] = Freqs_beaconed_tries
        spike_data["ML_SNRs_nonbeaconed_tries"+code] =SNRs_nonbeaconed_tries;      spike_data["ML_Freqs_nonbeaconed_tries"+code] = Freqs_nonbeaconed_tries
        spike_data["ML_SNRs_probe_tries"+code] =SNRs_probe_tries;                  spike_data["ML_Freqs_probe_tries"+code] = Freqs_probe_tries
        spike_data["ML_SNRs_all_tries"+code] =SNRs_all_tries;                      spike_data["ML_Freqs_all_tries"+code] = Freqs_all_tries

        spike_data["ML_SNRs_beaconed_misses"+code] =SNRs_beaconed_misses;          spike_data["ML_Freqs_beaconed_misses"+code] = Freqs_beaconed_misses
        spike_data["ML_SNRs_nonbeaconed_misses"+code] =SNRs_nonbeaconed_misses;    spike_data["ML_Freqs_nonbeaconed_misses"+code] = Freqs_nonbeaconed_misses
        spike_data["ML_SNRs_probe_misses"+code] =SNRs_probe_misses;                spike_data["ML_Freqs_probe_misses"+code] = Freqs_probe_misses
        spike_data["ML_SNRs_all_misses"+code] =SNRs_all_misses;                    spike_data["ML_Freqs_all_misses"+code] = Freqs_all_misses

    return spike_data

def analyse_lomb_powers(spike_data, processed_position_data):
    '''
    Requires the collumns of MOVING LOMB PERIODOGRAM
    This function takes the moving window periodogram and computes the average SNR, and for all combinations of trial types and hit miss try behaviours
    :param spike_data:
    :param processed_position_data:
    :param track_length:
    :return:
    '''

    step = 0.02
    frequency = np.arange(0.1, 5+step, step)

    SNRs_by_trial_number = [];     Freqs_by_trial_number = []

    SNRs = [];                     Freqs = []
    SNRs_all_beaconed = [];        Freqs_all_beaconed = []
    SNRs_all_nonbeaconed = [];     Freqs_all_nonbeaconed = []
    SNRs_all_probe = [];           Freqs_all_probe = []

    SNRs_beaconed_hits = [];       Freqs_beaconed_hits = []
    SNRs_nonbeaconed_hits = [];    Freqs_nonbeaconed_hits = []
    SNRs_probe_hits = [];          Freqs_probe_hits = []
    SNRs_all_hits = [];            Freqs_all_hits = []

    SNRs_beaconed_tries = [];      Freqs_beaconed_tries = []
    SNRs_nonbeaconed_tries = [];   Freqs_nonbeaconed_tries = []
    SNRs_probe_tries = [];         Freqs_probe_tries = []
    SNRs_all_tries = [];           Freqs_all_tries = []

    SNRs_beaconed_misses = [];     Freqs_beaconed_misses = []
    SNRs_nonbeaconed_misses = [];  Freqs_nonbeaconed_misses = []
    SNRs_probe_misses = [];        Freqs_probe_misses = []
    SNRs_all_misses =[];           Freqs_all_misses = []

    for index, spike_row in spike_data.iterrows():
        cluster_spike_data = spike_row.to_frame().T.reset_index(drop=True)
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        centre_trials = np.round(centre_trials).astype(np.int64)

        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        for tt in ["all", 0, 1, 2]:
            for hmt in ["all", "hit", "miss", "try"]:
                subset_processed_position_data = processed_position_data.copy()
                if tt != "all":
                    subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["trial_type"] == tt)]
                if hmt != "all":
                    subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["hit_miss_try"] == hmt)]
                subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])

                if len(firing_times_cluster)>1:
                    if len(subset_trial_numbers)>0:
                        subset_mask = np.isin(centre_trials, subset_trial_numbers)
                        subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                        subset_powers = powers.copy()
                        subset_powers[subset_mask == False] = np.nan
                        avg_subset_powers = np.nanmean(subset_powers, axis=0)

                        if ((tt== "all") and (hmt== "all")):
                            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_subset_powers)
                            overall_max_SNR_freq = max_SNR_freq
                        else:
                            max_SNR_freq = overall_max_SNR_freq
                            max_SNR = avg_subset_powers[frequency == overall_max_SNR_freq][0]

                    else:
                        max_SNR, max_SNR_freq = (np.nan, np.nan)
                else:
                    max_SNR, max_SNR_freq = (np.nan, np.nan)

                if (tt=="all") and (hmt=="hit"):
                    SNRs_all_hits.append(max_SNR)
                    Freqs_all_hits.append(max_SNR_freq)
                elif (tt=="all") and (hmt=="try"):
                    SNRs_all_tries.append(max_SNR)
                    Freqs_all_tries.append(max_SNR_freq)
                elif (tt=="all") and (hmt=="miss"):
                    SNRs_all_misses.append(max_SNR)
                    Freqs_all_misses.append(max_SNR_freq)
                elif (tt=="all") and (hmt=="all"):
                    SNRs.append(max_SNR)
                    Freqs.append(max_SNR_freq)
                elif (tt==0) and (hmt=="hit"):
                    SNRs_beaconed_hits.append(max_SNR)
                    Freqs_beaconed_hits.append(max_SNR_freq)
                elif (tt==0) and (hmt=="try"):
                    SNRs_beaconed_tries.append(max_SNR)
                    Freqs_beaconed_tries.append(max_SNR_freq)
                elif (tt==0) and (hmt=="miss"):
                    SNRs_beaconed_misses.append(max_SNR)
                    Freqs_beaconed_misses.append(max_SNR_freq)
                elif (tt==0) and (hmt=="all"):
                    SNRs_all_beaconed.append(max_SNR)
                    Freqs_all_beaconed.append(max_SNR_freq)
                elif (tt==1) and (hmt=="hit"):
                    SNRs_nonbeaconed_hits.append(max_SNR)
                    Freqs_nonbeaconed_hits.append(max_SNR_freq)
                elif (tt==1) and (hmt=="try"):
                    SNRs_nonbeaconed_tries.append(max_SNR)
                    Freqs_nonbeaconed_tries.append(max_SNR_freq)
                elif (tt==1) and (hmt=="miss"):
                    SNRs_nonbeaconed_misses.append(max_SNR)
                    Freqs_nonbeaconed_misses.append(max_SNR_freq)
                elif (tt==1) and (hmt=="all"):
                    SNRs_all_nonbeaconed.append(max_SNR)
                    Freqs_all_nonbeaconed.append(max_SNR_freq)
                elif (tt==2) and (hmt=="hit"):
                    SNRs_probe_hits.append(max_SNR)
                    Freqs_probe_hits.append(max_SNR_freq)
                elif (tt==2) and (hmt=="try"):
                    SNRs_probe_tries.append(max_SNR)
                    Freqs_probe_tries.append(max_SNR_freq)
                elif (tt==2) and (hmt=="miss"):
                    SNRs_probe_misses.append(max_SNR)
                    Freqs_probe_misses.append(max_SNR_freq)
                elif (tt==2) and (hmt=="all"):
                    SNRs_all_probe.append(max_SNR)
                    Freqs_all_probe.append(max_SNR_freq)

        SNRs_by_trial_number_cluster = []
        freqs_by_trial_number_cluster =[]
        for trial_number in processed_position_data["trial_number"]:
            subset_processed_position_data = processed_position_data.copy()
            subset_processed_position_data = subset_processed_position_data[subset_processed_position_data["trial_number"] == trial_number]
            subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])

            if len(firing_times_cluster)>1:
                if len(subset_trial_numbers)>0:
                    subset_mask = np.isin(centre_trials, subset_trial_numbers)
                    subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                    subset_powers = powers.copy()
                    subset_powers[subset_mask == False] = np.nan
                    avg_subset_powers = np.nanmean(subset_powers, axis=0)
                    max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_subset_powers)
                else:
                    max_SNR, max_SNR_freq = (np.nan, np.nan)
            else:
                max_SNR, max_SNR_freq = (np.nan, np.nan)
            SNRs_by_trial_number_cluster.append(max_SNR)
            freqs_by_trial_number_cluster.append(max_SNR_freq)
        SNRs_by_trial_number.append(SNRs_by_trial_number_cluster)
        Freqs_by_trial_number.append(freqs_by_trial_number_cluster)

    spike_data["ML_SNRs_by_trial_number"] = SNRs_by_trial_number;         spike_data["ML_Freqs_by_trial_number"] = Freqs_by_trial_number

    spike_data["ML_SNRs"] = SNRs;                                         spike_data["ML_Freqs"] = Freqs
    spike_data["ML_SNRs_all_beaconed"] = SNRs_all_beaconed;               spike_data["ML_Freqs_all_beaconed"] = Freqs_all_beaconed
    spike_data["ML_SNRs_all_nonbeaconed"] =SNRs_all_nonbeaconed;          spike_data["ML_Freqs_all_nonbeaconed"] = Freqs_all_nonbeaconed
    spike_data["ML_SNRs_all_probe"] =SNRs_all_probe;                      spike_data["ML_Freqs_all_probe"] = Freqs_all_probe

    spike_data["ML_SNRs_beaconed_hits"] =SNRs_beaconed_hits;              spike_data["ML_Freqs_beaconed_hits"] = Freqs_beaconed_hits
    spike_data["ML_SNRs_nonbeaconed_hits"] =SNRs_nonbeaconed_hits;        spike_data["ML_Freqs_nonbeaconed_hits"] = Freqs_nonbeaconed_hits
    spike_data["ML_SNRs_probe_hits"] =SNRs_probe_hits;                    spike_data["ML_Freqs_probe_hits"] = Freqs_probe_hits
    spike_data["ML_SNRs_all_hits"] =SNRs_all_hits;                        spike_data["ML_Freqs_all_hits"] = Freqs_all_hits

    spike_data["ML_SNRs_beaconed_tries"] =SNRs_beaconed_tries;            spike_data["ML_Freqs_beaconed_tries"] = Freqs_beaconed_tries
    spike_data["ML_SNRs_nonbeaconed_tries"] =SNRs_nonbeaconed_tries;      spike_data["ML_Freqs_nonbeaconed_tries"] = Freqs_nonbeaconed_tries
    spike_data["ML_SNRs_probe_tries"] =SNRs_probe_tries;                  spike_data["ML_Freqs_probe_tries"] = Freqs_probe_tries
    spike_data["ML_SNRs_all_tries"] =SNRs_all_tries;                      spike_data["ML_Freqs_all_tries"] = Freqs_all_tries

    spike_data["ML_SNRs_beaconed_misses"] =SNRs_beaconed_misses;          spike_data["ML_Freqs_beaconed_misses"] = Freqs_beaconed_misses
    spike_data["ML_SNRs_nonbeaconed_misses"] =SNRs_nonbeaconed_misses;    spike_data["ML_Freqs_nonbeaconed_misses"] = Freqs_nonbeaconed_misses
    spike_data["ML_SNRs_probe_misses"] =SNRs_probe_misses;                spike_data["ML_Freqs_probe_misses"] = Freqs_probe_misses
    spike_data["ML_SNRs_all_misses"] =SNRs_all_misses;                    spike_data["ML_Freqs_all_misses"] = Freqs_all_misses

    #del spike_data["MOVING_LOMB_all_powers"]
    #del spike_data["MOVING_LOMB_all_centre_trials"]
    return spike_data

def get_tt_color(tt):
    if tt == 0:
        return "black"
    elif tt==1:
        return "red"
    elif tt ==2:
        return "blue"

def get_hmt_linestyle(hmt):
    if hmt == "hit":
        return "solid"
    elif hmt=="miss":
        return "dashed"
    elif hmt =="try":
        return "dotted"


def get_numeric_lomb_classifer(lomb_classifier_str):
    if lomb_classifier_str == "Position":
        return 0
    elif lomb_classifier_str == "Distance":
        return 1
    elif lomb_classifier_str == "Null":
        return 2
    else:
        return 3

def get_lomb_classifier(lomb_SNR, lomb_freq, lomb_SNR_thres, lomb_freq_thres, numeric=False):
    lomb_distance_from_int = distance_from_integer(lomb_freq)[0]

    if lomb_SNR>lomb_SNR_thres:
        if lomb_distance_from_int<lomb_freq_thres:
            lomb_classifier = "Position"
        else:
            lomb_classifier = "Distance"
    else:
        if np.isnan(lomb_distance_from_int):
            lomb_classifier = "Unclassifed"
        else:
            lomb_classifier = "Null"

    if numeric:
        return get_numeric_lomb_classifer(lomb_classifier)
    else:
        return lomb_classifier

def add_lomb_classifier(spatial_firing, suffix=""):
    """
    :param spatial_firing:
    :param suffix: specific set string for subsets of results
    :return: spatial_firing with classifier collumn of type ["Lomb_classifer_"+suffix] with either "Distance", "Position" or "Null"
    """
    lomb_classifiers = []
    for index, row in spatial_firing.iterrows():
        if "ML_SNRs"+suffix in list(spatial_firing):
            lomb_SNR = row["ML_SNRs"+suffix]
            lomb_freq = row["ML_Freqs"+suffix]

            lomb_classifier = get_lomb_classifier(lomb_SNR, lomb_freq, 0.023, 0.05, numeric=False)
        else:
            lomb_classifier = "Unclassifed"

        lomb_classifiers.append(lomb_classifier)

    spatial_firing["Lomb_classifier_"+suffix] = lomb_classifiers
    return spatial_firing

def add_pairwise_classifier(spatial_firing, suffix=""):
    """
    :param spatial_firing:
    :param suffix: specific set string for subsets of results
    :return: spatial_firing with classifier collumn of type ["Pairwise_classifer_"+suffix] with either "Distance", "Position" or "Null"
    """
    pairwise_classifiers = []
    for index, row in spatial_firing.iterrows():
        if "SNR"+suffix in list(spatial_firing):
            pairwise_corr = row["pairwise_corr"+suffix]
            pairwise_freq = row["pairwise_freq"+suffix]
            lomb_distance_from_int = distance_from_integer(pairwise_freq)[0]

            if pairwise_corr>0.05:
                if lomb_distance_from_int<0.05:
                    pairwise_classifier = "Position"
                else:
                    pairwise_classifier = "Distance"
            elif lomb_distance_from_int<0.05:
                pairwise_classifier = "Position"
            else:
                pairwise_classifier = "Null"
        else:
            pairwise_classifier = "Unclassifed"

        pairwise_classifiers.append(pairwise_classifier)

    spatial_firing["Pairwise_classifier_"+suffix] = pairwise_classifiers
    return spatial_firing



def get_first_peak(distances, autocorrelogram):
    peaks_i = signal.argrelextrema(autocorrelogram, np.greater, order=20)[0]
    if len(peaks_i)>0:
        return distances[peaks_i][0]
    else:
        return np.nan

def distance_from_integer(frequencies):
    distance_from_zero = np.asarray(frequencies)%1
    distance_from_one = 1-(np.asarray(frequencies)%1)
    tmp = np.vstack((distance_from_zero, distance_from_one))
    return np.min(tmp, axis=0)

def calculate_allocentric_correlation(spike_data, position_data, output_path,track_length):
    save_path = output_path + '/Figures/trial_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    allocentric_avg_correlation = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times = cluster_spike_data["firing_times"].iloc[0]/(settings.sampling_rate/1000) # convert from samples to ms

        if len(firing_times)>1:
            trial_numbers = np.array(position_data['trial_number'].to_numpy())
            trial_types = np.array(position_data['trial_type'].to_numpy())
            time_seconds = np.array(position_data['time_seconds'].to_numpy())
            x_position_cm = np.array(position_data['x_position_cm'].to_numpy())
            x_position_cm_elapsed = x_position_cm+((trial_numbers-1)*track_length)

            instantaneous_firing_rate_per_ms = extract_instantaneous_firing_rate_for_spike2(cluster_spike_data) # returns firing rate per millisecond time bin
            instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[0:len(x_position_cm)]

            if not (len(instantaneous_firing_rate_per_ms) == len(trial_numbers)):
                # 0 pad until it is the same size (padding with 0 hz firing rate
                instantaneous_firing_rate_per_ms = np.append(instantaneous_firing_rate_per_ms, np.zeros(len(trial_numbers)-len(instantaneous_firing_rate_per_ms)))

            # get firing field distance
            firing_field_distance_cluster = cluster_spike_data["spatial_autocorr_peak_cm"].iloc[0]

            rate_maps = []
            for trial_number in np.unique(trial_numbers):
                instantaneous_firing_rate_per_ms_trial = instantaneous_firing_rate_per_ms[trial_numbers == trial_number]
                x_position_cm_trial = x_position_cm[trial_numbers == trial_number]

                numerator, bin_edges = np.histogram(x_position_cm_trial, bins=int(track_length/1), range=(0, track_length), weights=instantaneous_firing_rate_per_ms_trial)
                denominator, _ = np.histogram(x_position_cm_trial, bins=int(track_length/1), range=(0, track_length))

                trial_rate_map = numerator/denominator
                bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

                rate_maps.append(trial_rate_map)

                #peaks = signal.argrelextrema(trial_rate_map, np.greater, order=20)[0]
                #ax.scatter(bin_centres[peaks], np.ones(len(bin_centres[peaks]))*trial_number, marker="x", color="r")
                #if len(peaks)>0:
                #    ax.scatter(bin_centres[peaks][-1], trial_number, marker="x", color="r")

            trial_pair_correlations = []
            for i in range(len(rate_maps)-1):
                rate_map_i = rate_maps[i]
                rate_map_ii = rate_maps[i+1]
                nan_mask = np.logical_or(np.isnan(rate_map_i), np.isnan(rate_map_ii))
                rate_map_i = rate_map_i[~nan_mask]
                rate_map_ii = rate_map_ii[~nan_mask]
                if len(rate_map_i)>1:
                    corr = pearsonr(rate_map_i, rate_map_ii)[0]
                else:
                    corr = 0

                trial_pair_correlations.append(corr)

            avg_pair_correlation = np.nanmean(np.array(trial_pair_correlations))

            allocentric_avg_correlation.append(avg_pair_correlation)
        else:
            allocentric_avg_correlation.append(np.nan)

    spike_data["allocentric_avg_correlation"] = allocentric_avg_correlation
    return spike_data

def calculate_egocentric_correlation(spike_data, position_data, output_path,track_length):
    save_path = output_path + '/Figures/trial_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    egocentric_avg_correlation = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times = cluster_spike_data["firing_times"].iloc[0]/(settings.sampling_rate/1000) # convert from samples to ms

        if len(firing_times)>1:
            trial_numbers = np.array(position_data['trial_number'].to_numpy())
            trial_types = np.array(position_data['trial_type'].to_numpy())
            time_seconds = np.array(position_data['time_seconds'].to_numpy())
            x_position_cm = np.array(position_data['x_position_cm'].to_numpy())
            x_position_cm_elapsed = x_position_cm+((trial_numbers-1)*track_length)

            instantaneous_firing_rate_per_ms = extract_instantaneous_firing_rate_for_spike2(cluster_spike_data) # returns firing rate per millisecond time bin
            instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[0:len(x_position_cm)]

            if not (len(instantaneous_firing_rate_per_ms) == len(trial_numbers)):
                # 0 pad until it is the same size (padding with 0 hz firing rate
                instantaneous_firing_rate_per_ms = np.append(instantaneous_firing_rate_per_ms, np.zeros(len(trial_numbers)-len(instantaneous_firing_rate_per_ms)))

            # get firing field distance
            firing_field_distance_cluster = cluster_spike_data["spatial_autocorr_peak_cm"].iloc[0]

            rate_maps = []
            residuals = []
            for trial_number in np.unique(trial_numbers):
                instantaneous_firing_rate_per_ms_trial = instantaneous_firing_rate_per_ms[trial_numbers == trial_number]
                x_position_cm_trial = x_position_cm[trial_numbers == trial_number]

                numerator, bin_edges = np.histogram(x_position_cm_trial, bins=int(track_length/1), range=(0, track_length), weights=instantaneous_firing_rate_per_ms_trial)
                denominator, _ = np.histogram(x_position_cm_trial, bins=int(track_length/1), range=(0, track_length))

                trial_rate_map = numerator/denominator
                bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
                peaks = signal.argrelextrema(trial_rate_map, np.greater, order=20)[0]

                if len(peaks)>0:
                    last_peak = bin_centres[peaks[-1]]
                    residual = track_length-last_peak
                else:
                    last_peak = np.nan
                    residual = 0

                residuals.append(residual)
                rate_maps.append(trial_rate_map)

                #ax.scatter(bin_centres[peaks], np.ones(len(bin_centres[peaks]))*trial_number, marker="x", color="r")
                #if len(peaks)>0:
                #    ax.scatter(bin_centres[peaks][-1], trial_number, marker="x", color="r")

            trial_pair_correlations = []
            for i in range(len(rate_maps)-1):
                rate_map_i = rate_maps[i]
                rate_map_ii = rate_maps[i+1]
                if int(residuals[i]) > 0:
                    rate_map_ii = rate_map_ii[:-int(residuals[i])]
                    rate_map_i = rate_map_i[int(residuals[i]):]
                nan_mask = np.logical_or(np.isnan(rate_map_i), np.isnan(rate_map_ii))
                rate_map_ii = rate_map_ii[~nan_mask]
                rate_map_i = rate_map_i[~nan_mask]
                if len(rate_map_i)>1:
                    corr = pearsonr(rate_map_i, rate_map_ii)[0]
                else:
                    corr = 0
                trial_pair_correlations.append(corr)

            avg_pair_correlation = np.nanmean(np.array(trial_pair_correlations))

            egocentric_avg_correlation.append(avg_pair_correlation)
        else:
            egocentric_avg_correlation.append(np.nan)

    spike_data["egocentric_avg_correlation"] = egocentric_avg_correlation
    return spike_data


def plot_inter_field_distance_histogram(spike_data, output_path,track_length):
    print('plotting field com histogram...')
    tick_spacing = 100
    save_path = output_path + '/Figures/field_distances'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        if len(firing_times_cluster)>1:

            norm = mpl.colors.Normalize(vmin=0, vmax=track_length)
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
            cmap.set_array([])

            fig, ax = plt.subplots(dpi=200)
            loop_factor=3
            hist_cmap = plt.cm.get_cmap('viridis')
            cluster_firing_com_distances = np.array(spike_data["distance_between_fields"].iloc[cluster_index])
            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            cluster_firing_com_distances = np.append(cluster_firing_com_distances, np.nan)

            for i in range(int(track_length/settings.vr_grid_analysis_bin_size)*loop_factor):
                mask = (cluster_firing_com > i*settings.vr_grid_analysis_bin_size) & \
                       (cluster_firing_com < (i+1)*settings.vr_grid_analysis_bin_size)
                cluster_firing_com_distances_bin_i = cluster_firing_com_distances[mask]
                cluster_firing_com_distances_bin_i = cluster_firing_com_distances_bin_i[~np.isnan(cluster_firing_com_distances_bin_i)]

                field_hist, bin_edges = np.histogram(cluster_firing_com_distances_bin_i,
                                                     bins=int(track_length/settings.vr_grid_analysis_bin_size)*loop_factor,
                                                     range=[0, track_length*loop_factor])

                if i == 0:
                    bottom = np.zeros(len(field_hist))

                ax.bar(bin_edges[:-1], field_hist, width=np.diff(bin_edges), bottom=bottom, edgecolor="black",
                       align="edge", color=hist_cmap(i/int(track_length/settings.vr_grid_analysis_bin_size)*loop_factor))
                bottom += field_hist

            cbar = fig.colorbar(cmap)
            cbar.set_label('Location (cm)', rotation=90, labelpad=20)
            plt.ylabel('Field Counts', fontsize=12, labelpad = 10)
            plt.xlabel('Field to Field Distance (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,400)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            x_max = max(bottom)
            Edmond.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_distance_hist_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def plot_field_com_histogram(spike_data, output_path, track_length):
    tick_spacing = 50

    print('plotting field com histogram...')
    save_path = output_path + '/Figures/field_distributions'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        if len(firing_times_cluster)>1:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            field_hist, bin_edges = np.histogram(cluster_firing_com, bins=int(track_length/settings.vr_grid_analysis_bin_size), range=[0, track_length])
            ax.bar(bin_edges[:-1], field_hist/np.sum(field_hist), width=np.diff(bin_edges), edgecolor="black", align="edge")
            plt.ylabel('Field Density', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            field_hist = np.nan_to_num(field_hist)

            x_max = max(field_hist/np.sum(field_hist))
            Edmond.plot_utility2.style_track_plot(ax, track_length)
            Edmond.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_hist_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def min_max_normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def style_track_plot_no_RZ(ax, track_length):
    ax.axvline(x=track_length-60-30-20, color="black", linestyle="dotted", linewidth=1)
    ax.axvline(x=track_length-60-30, color="black", linestyle="dotted", linewidth=1)
    ax.axvspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axvspan(track_length-30, track_length, facecolor='k', linewidth =0, alpha=.25)# black box

def style_track_plot(ax, track_length):
    ax.axvspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axvspan(track_length-110, track_length-90, facecolor='DarkGreen', alpha=.25, linewidth =0)
    ax.axvspan(track_length-30, track_length, facecolor='k', linewidth =0, alpha=.25)# black box

def plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=200,
                         plot_trials=["beaconed", "non_beaconed", "probe"]):

    print('plotting spike rastas...')
    save_path = output_path + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data.firing_times.iloc[0]
        if len(firing_times_cluster)>1:

            x_max = len(processed_position_data)
            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)

            if "beaconed" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].beaconed_position_cm, cluster_spike_data.iloc[0].beaconed_trial_number, '|', color='Black', markersize=4)
            if "non_beaconed" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].nonbeaconed_position_cm, cluster_spike_data.iloc[0].nonbeaconed_trial_number, '|', color='Black', markersize=4)
            if "probe" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].probe_position_cm, cluster_spike_data.iloc[0].probe_trial_number, '|', color='Black', markersize=4)

            plt.ylabel('Spikes on trials', fontsize=20, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            style_track_plot(ax, track_length)
            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            Edmond.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def plot_firing_rate_maps_per_trial(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:

            x_max = len(processed_position_data)
            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)

            cluster_firing_maps = np.array(spike_data["firing_rate_maps"].iloc[cluster_index])
            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0

            if len(cluster_firing_maps) == 0:
                print("stop here")

            cluster_firing_maps = min_max_normalize(cluster_firing_maps)

            cmap = plt.cm.get_cmap("jet")
            cmap.set_bad(color='white')
            bin_size = settings.vr_grid_analysis_bin_size

            tmp = []
            for i in range(len(cluster_firing_maps[0])):
                for j in range(int(settings.vr_grid_analysis_bin_size)):
                    tmp.append(cluster_firing_maps[:, i].tolist())
            cluster_firing_maps = np.array(tmp).T
            c = ax.imshow(cluster_firing_maps, interpolation='none', cmap=cmap, vmin=0, vmax=np.max(cluster_firing_maps), origin='lower', aspect="auto")

            plt.ylabel('Trial Number', fontsize=20, labelpad = 20)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
            plt.xlim(0, track_length)
            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            Edmond.plot_utility2.style_track_plot(ax, track_length)
            Edmond.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '.png', dpi=300)
            plt.close()

def plot_field_com_histogram_radial(spike_data, output_path, track_length):
    print('plotting field com histogram...')
    save_path = output_path + '/Figures/field_distributions'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        if len(firing_times_cluster)>1:
            ax = plt.subplot(111, polar=True)
            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            field_hist, bin_edges = np.histogram(cluster_firing_com, bins=int(track_length/settings.vr_grid_analysis_bin_size), range=[0, track_length])

            width = (2*np.pi) / len(bin_edges[:-1])
            field_hist = field_hist/np.sum(field_hist)
            bottom = 0.4
            field_hist = min_max_normlise(field_hist, 0, 1)
            y_max = max(field_hist)

            bin_edges = np.linspace(0.0, 2 * np.pi, len(bin_edges[:-1]), endpoint=False)

            ax.bar(np.pi, y_max, width=np.pi*2*(20/track_length), color="DarkGreen", edgecolor=None, alpha=0.25, bottom=bottom)
            ax.bar(0, y_max, width=np.pi*2*(60/track_length), color="black", edgecolor=None, alpha=0.25, bottom=bottom)

            ax.bar(bin_edges, field_hist, width=width, edgecolor="black", align="edge", bottom=bottom)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.grid(alpha=0)
            ax.set_yticklabels([])
            ax.set_ylim([0,y_max])
            ax.set_xticklabels(['0cm', '', '50cm', '', '100cm', '', '150cm', ''], fontsize=15)
            ax.xaxis.set_tick_params(pad=20)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_hist_radial_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def plot_field_centre_of_mass_on_track(spike_data, processed_position_data, output_path, track_length):

    print('plotting field rastas...')
    save_path = output_path + '/Figures/field_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]
        firing_times_cluster = cluster_spike_data.firing_times.iloc[0]
        if len(firing_times_cluster)>1:
            cluster_firing_com = np.array(cluster_spike_data["firing_fields_com"].iloc[0])

            x_max = len(processed_position_data)
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(1, 1, 1)
            # plot spikes first
            ax.scatter(cluster_spike_data.iloc[0].x_position_cm, cluster_spike_data.iloc[0].trial_number, marker='|', color='black', zorder=-1)

            for i, tn in enumerate(processed_position_data[(processed_position_data["hit_miss_try"]=="hit") & (processed_position_data["trial_type"]==1)]["trial_number"]):
                for j in range(len(cluster_firing_com[i])):
                    ax.scatter(cluster_firing_com[i][j], tn, color="green", marker="s")
            for i, tn in enumerate(processed_position_data[(processed_position_data["hit_miss_try"]=="try") & (processed_position_data["trial_type"]==1)]["trial_number"]):
                for j in range(len(cluster_firing_com[i])):
                    ax.scatter(cluster_firing_com[i][j], tn, color="orange", marker="s")
            for i, tn in enumerate(processed_position_data[(processed_position_data["hit_miss_try"]=="miss") & (processed_position_data["trial_type"]==1)]["trial_number"]):
                for j in range(len(cluster_firing_com[i])):
                    ax.scatter(cluster_firing_com[i][j], tn, color="red", marker="s")

            plt.ylabel('Field COM on trials', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            Edmond.plot_utility2.style_track_plot(ax, track_length)
            Edmond.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()
    return

def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array

def plot_field_com_ring_attractor_radial(spike_data, of_spike_data, output_path, track_length):
    print('plotting field com histogram...')
    save_path = output_path + '/Figures/radial_field_distances'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        estimated_grid_spacing = of_spike_data.grid_spacing.iloc[cluster_index]

        if len(firing_times_cluster)>1:
            ax = plt.subplot(111, polar=True)

            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            cluster_firing_com_distances = np.array(spike_data["distance_between_fields"].iloc[cluster_index])
            loop_factor = (max(cluster_firing_com_distances)//track_length)+1

            field_hist, bin_edges = np.histogram(cluster_firing_com_distances,
                                                 bins=int((track_length/settings.vr_grid_analysis_bin_size)*loop_factor),
                                                 range=[0, int(track_length*loop_factor)])

            width = (2*np.pi) / (len(bin_edges[:-1])/loop_factor)
            field_hist = field_hist/np.sum(field_hist)
            bottom = 0.4
            field_hist = min_max_normlise(field_hist, 0, 1)
            y_max = max(field_hist)

            bin_edges = np.linspace(0.0, loop_factor*2*np.pi, int(len(bin_edges[:-1])), endpoint=False)

            cmap = plt.cm.get_cmap('viridis')

            #ax.bar(np.pi, y_max, width=np.pi*2*(20/track_length), color="DarkGreen", edgecolor=None, alpha=0.25, bottom=bottom)
            #ax.bar(0, y_max, width=np.pi*2*(60/track_length), color="black", edgecolor=None, alpha=0.25, bottom=bottom)

            for i in range(int(loop_factor)):
                ax.bar(bin_edges[int(i*(len(bin_edges)/loop_factor)): int((i+1)*(len(bin_edges)/loop_factor))],
                       field_hist[int(i*(len(bin_edges)/loop_factor)): int((i+1)*(len(bin_edges)/loop_factor))],
                       width=width, edgecolor="black", align="edge", bottom=bottom, color=cmap(i/loop_factor), alpha=0.6)

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.grid(alpha=0)
            ax.set_yticklabels([])
            ax.set_ylim([0,y_max])
            estimated_grid_spacing = np.round(estimated_grid_spacing/2, decimals=1)
            ax.set_xticklabels([str(np.round(estimated_grid_spacing, decimals=1))+"cm", "", "", "", "", "", "", ""], fontsize=15)
            ax.xaxis.set_tick_params(pad=20)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_distance_hist_radial_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def get_track_length(recording_path):
    parameter_file_path = control_sorting_analysis.get_tags_parameter_file(recording_path)
    stop_threshold, track_length, cue_conditioned_goal = PostSorting.post_process_sorted_data_vr.process_running_parameter_tag(parameter_file_path)
    return track_length

def plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=200):
    gauss_kernel = Gaussian1DKernel(2)
    print('I am plotting firing rate maps...')
    save_path = output_path + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]

        avg_beaconed_spike_rate = np.array(cluster_spike_data["beaconed_firing_rate_map"].to_list()[0])
        avg_nonbeaconed_spike_rate = np.array(cluster_spike_data["non_beaconed_firing_rate_map"].to_list()[0])
        avg_probe_spike_rate = np.array(cluster_spike_data["probe_firing_rate_map"].to_list()[0])

        beaconed_firing_rate_map_sem = np.array(cluster_spike_data["beaconed_firing_rate_map_sem"].to_list()[0])
        non_beaconed_firing_rate_map_sem = np.array(cluster_spike_data["non_beaconed_firing_rate_map_sem"].to_list()[0])
        probe_firing_rate_map_sem = np.array(cluster_spike_data["probe_firing_rate_map_sem"].to_list()[0])

        avg_beaconed_spike_rate = convolve(avg_beaconed_spike_rate, gauss_kernel) # convolve and smooth beaconed
        beaconed_firing_rate_map_sem = convolve(beaconed_firing_rate_map_sem, gauss_kernel)

        if len(avg_nonbeaconed_spike_rate)>0:
            avg_nonbeaconed_spike_rate = convolve(avg_nonbeaconed_spike_rate, gauss_kernel) # convolve and smooth non beaconed
            non_beaconed_firing_rate_map_sem = convolve(non_beaconed_firing_rate_map_sem, gauss_kernel)

        if len(avg_probe_spike_rate)>0:
            avg_probe_spike_rate = convolve(avg_probe_spike_rate, gauss_kernel) # convolve and smooth probe
            probe_firing_rate_map_sem = convolve(probe_firing_rate_map_sem, gauss_kernel)

        avg_spikes_on_track = plt.figure()
        avg_spikes_on_track.set_size_inches(5, 5, forward=True)
        ax = avg_spikes_on_track.add_subplot(1, 1, 1)
        bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])

        #plotting the rates are filling with the standard error around the mean
        ax.plot(bin_centres, avg_beaconed_spike_rate, '-', color='Black')
        ax.fill_between(bin_centres, avg_beaconed_spike_rate-beaconed_firing_rate_map_sem,
                        avg_beaconed_spike_rate+beaconed_firing_rate_map_sem, color="Black", alpha=0.5)

        if len(avg_nonbeaconed_spike_rate)>0:
            ax.plot(bin_centres, avg_nonbeaconed_spike_rate, '-', color='Red')
            ax.fill_between(bin_centres, avg_nonbeaconed_spike_rate-non_beaconed_firing_rate_map_sem,
                            avg_nonbeaconed_spike_rate+non_beaconed_firing_rate_map_sem, color="Red", alpha=0.5)

        if len(avg_probe_spike_rate)>0:
            ax.plot(bin_centres, avg_probe_spike_rate, '-', color='Blue')
            ax.fill_between(bin_centres, avg_probe_spike_rate-probe_firing_rate_map_sem,
                            avg_probe_spike_rate+probe_firing_rate_map_sem, color="Blue", alpha=0.5)

        tick_spacing = 50
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.ylabel('Spike rate (hz)', fontsize=20, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
        plt.xlim(0,track_length)
        x_max = np.nanmax(avg_beaconed_spike_rate)
        if len(avg_nonbeaconed_spike_rate)>0:
            nb_x_max = np.nanmax(avg_nonbeaconed_spike_rate)
            if nb_x_max > x_max:
                x_max = nb_x_max
        Edmond.plot_utility2.style_vr_plot(ax, x_max)
        Edmond.plot_utility2.style_track_plot(ax, track_length)
        plt.tight_layout()
        plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_rate_map_Cluster_' + str(cluster_id) + '.png', dpi=200)
        plt.close()

def get_trial_color(trial_type):
    if trial_type == 0:
        return "black"
    elif trial_type == 1:
        return "red"
    elif trial_type == 2:
        return "blue"
    else:
        print("invalid trial-type passed to get_trial_color()")

def plot_stops_on_track(processed_position_data, output_path, track_length=200):
    print('I am plotting stop rasta...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_number = trial_row["trial_number"].iloc[0]
        trial_stop_color = get_trial_color(trial_type)

        if trial_stop_color == "blue":
            alpha=0
        else:
            alpha=1

        ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]), trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])), 'o', color=trial_stop_color, markersize=4, alpha=alpha)

    plt.ylabel('Stops on trials', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    Edmond.plot_utility2.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()


def plot_stop_histogram(processed_position_data, output_path, track_length=200):
    print('plotting stop histogram...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stop_histogram = plt.figure(figsize=(6,2))
    ax = stop_histogram.add_subplot(1, 1, 1)
    bin_size = 5

    beaconed_trials = processed_position_data[processed_position_data["trial_type"] == 0]
    non_beaconed_trials = processed_position_data[processed_position_data["trial_type"] == 1]
    probe_trials = processed_position_data[processed_position_data["trial_type"] == 2]

    beaconed_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(beaconed_trials["stop_location_cm"])
    non_beaconed_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(non_beaconed_trials["stop_location_cm"])
    #probe_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(probe_trials["stop_location_cm"])

    beaconed_stop_hist, bin_edges = np.histogram(beaconed_stops, bins=int(track_length/bin_size), range=(0, track_length))
    non_beaconed_stop_hist, bin_edges = np.histogram(non_beaconed_stops, bins=int(track_length/bin_size), range=(0, track_length))
    #probe_stop_hist, bin_edges = np.histogram(probe_stops, bins=int(track_length/bin_size), range=(0, track_length))
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

    ax.plot(bin_centres, beaconed_stop_hist/len(beaconed_trials), '-', color='Black')
    if len(non_beaconed_trials)>0:
        ax.plot(bin_centres, non_beaconed_stop_hist/len(non_beaconed_trials), '-', color='Red')
    #if len(probe_trials)>0:
    #    ax.plot(bin_centres, probe_stop_hist/len(probe_trials), '-', color='Blue')

    plt.ylabel('Per trial', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"])
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_track_plot(ax, track_length)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/stop_histogram' + '.png', dpi=200)
    plt.close()

def plot_speed_per_trial(processed_position_data, output_path, track_length=200):
    print('plotting speed heatmap...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    x_max = len(processed_position_data)
    if x_max>100:
        fig = plt.figure(figsize=(4,(x_max/32)))
    else:
        fig = plt.figure(figsize=(4,(x_max/20)))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    trial_speeds = Edmond.plot_utility2.pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned"])
    cmap = plt.cm.get_cmap("jet")
    cmap.set_bad(color='white')
    trial_speeds = np.clip(trial_speeds, a_min=0, a_max=60)
    c = ax.imshow(trial_speeds, interpolation='none', cmap=cmap, vmin=0, vmax=60)
    clb = fig.colorbar(c, ax=ax, shrink=0.5)
    clb.mappable.set_clim(0, 60)
    plt.ylabel('Trial Number', fontsize=20, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
    plt.xlim(0,track_length)
    tick_spacing = 50
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/speed_heat_map' + '.png', dpi=200)
    plt.close()

def plot_speed_histogram(processed_position_data, output_path, track_length=200, suffix=""):
    if len(processed_position_data)>0:
        trial_averaged_beaconed_speeds, trial_averaged_non_beaconed_speeds, trial_averaged_probe_speeds = \
            PostSorting.vr_spatial_data.trial_average_speed(processed_position_data)

        print('plotting speed histogram...')
        save_path = output_path + '/Figures/behaviour'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        speed_histogram = plt.figure(figsize=(6,4))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])

        if len(trial_averaged_beaconed_speeds)>0:
            ax.plot(bin_centres, trial_averaged_beaconed_speeds, '-', color='Black')

        if len(trial_averaged_non_beaconed_speeds)>0:
            ax.plot(bin_centres, trial_averaged_non_beaconed_speeds, '-', color='Red')

        if len(trial_averaged_probe_speeds)>0:
            ax.plot(bin_centres, trial_averaged_probe_speeds, '-', color='Blue')

        plt.ylabel('Speed (cm/s)', fontsize=20, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
        plt.xlim(0,track_length)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Edmond.plot_utility2.style_track_plot(ax, track_length)
        tick_spacing = 50
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        x_max = 50

        if len(trial_averaged_beaconed_speeds)>0:
            x_max = max(trial_averaged_beaconed_speeds)

        if len(trial_averaged_non_beaconed_speeds)>0:
            max_nb = max(trial_averaged_non_beaconed_speeds)
            x_max = max([x_max, max_nb])

        if len(trial_averaged_probe_speeds)>0:
            max_p = max(trial_averaged_probe_speeds)
            x_max = max([x_max, max_p])

        Edmond.plot_utility2.style_vr_plot(ax, x_max)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
        plt.savefig(output_path + '/Figures/behaviour/speed_histogram_' +suffix+ '.png', dpi=200)
        plt.close()

def plot_allo_vs_ego_firing(vr_recording_path_list, of_recording_path_list, save_path):

    ego_scores = []
    allo_scores = []
    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        if os.path.isfile(recording+"/MountainSort/DataFrames/spatial_firing.pkl"):
            spike_data_vr = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            if found_paired_recording:
                if os.path.isfile(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl"):
                    spike_data_of =  pd.read_pickle(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl")
                    for cluster_index, cluster_id in enumerate(spike_data_vr.cluster_id):
                        cluster_spike_data_vr = spike_data_vr[spike_data_vr["cluster_id"] == cluster_id]
                        cluster_spike_data_of = spike_data_of[spike_data_of["cluster_id"] == cluster_id]

                        if len(cluster_spike_data_of) == 1:
                            grid_score = cluster_spike_data_of["grid_score"].iloc[0]
                            rate_map_corr = cluster_spike_data_of["rate_map_correlation_first_vs_second_half"].iloc[0]
                            if (grid_score > 0.5) and (rate_map_corr > 0):
                                ego_score = cluster_spike_data_vr["egocentric_avg_correlation"].iloc[0]
                                allo_score = cluster_spike_data_vr["allocentric_avg_correlation"].iloc[0]
                                ego_scores.append(ego_score)
                                allo_scores.append(allo_score)
                        else:
                            print("there is no matching of cell")

    allo_scores = np.array(allo_scores)
    ego_scores = np.array(ego_scores)
    nan_mask = np.logical_or(np.isnan(allo_scores), np.isnan(ego_scores))
    allo_scores = allo_scores[~nan_mask]
    ego_scores = ego_scores[~nan_mask]

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.bar([0.6, 1.4], [np.mean(allo_scores), np.mean(ego_scores)], edgecolor="black", align="edge", color="white", alpha=0)
    ax.errorbar([0.6, 1.4], [np.mean(allo_scores), np.mean(ego_scores)], yerr=[stats.sem(allo_scores), stats.sem(ego_scores)], fmt='o')
    for i in range(len(allo_scores)):
        ax.plot([0.7, 1.3], [allo_scores[i], ego_scores[i]], marker="o", alpha=0.3, color="black")
    plt.ylabel('Lap to lap pearson', fontsize=12, labelpad = 10)
    plt.xticks(ticks=[0.6 ,1.4], labels=["allocentric", "egocentric"], fontsize=12)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlim([0, 2])
    plt.ylim([-0.5, -0.5])
    Edmond.plot_utility2.style_vr_plot(ax, x_max=1)
    plt.savefig(save_path + '/allo_vs_ego_grid_cells.png', dpi=200)
    plt.close()


def add_time_elapsed_collumn(position_data):
    time_seconds = np.array(position_data['time_seconds'].to_numpy())
    time_elapsed = np.diff(time_seconds)
    time_elapsed = np.append(time_elapsed[0], time_elapsed)
    position_data["time_in_bin_seconds"] = time_elapsed.tolist()
    return position_data

def extract_PI_trials(processed_position_data, hmt):
    PI_trial_numbers = processed_position_data[(processed_position_data["hit_miss_try"] == hmt) &
                                               ((processed_position_data["trial_type"] == 1) | (processed_position_data["trial_type"] == 2))]["trial_number"]
    trial_numbers_to_keep = []
    for t in PI_trial_numbers:
        if t not in trial_numbers_to_keep:
            trial_numbers_to_keep.append(t)
        if (t-1 not in trial_numbers_to_keep) and (t-1 != 0):
            trial_numbers_to_keep.append(t-1)

    PI_processed_position_data = processed_position_data.loc[processed_position_data["trial_number"].isin(trial_numbers_to_keep)]
    return PI_processed_position_data

def extract_beaconed_hit_trials(processed_position_data):
    b_trial_numbers = processed_position_data[(processed_position_data["hit_miss_try"] == "hit") &
                                               (processed_position_data["trial_type"] == 0)]["trial_number"]
    trial_numbers_to_keep = []
    for t in b_trial_numbers:
        if t not in trial_numbers_to_keep:
            trial_numbers_to_keep.append(t)
        if (t-1 not in trial_numbers_to_keep) and (t-1 != 0):
            trial_numbers_to_keep.append(t-1)

    b_processed_position_data = processed_position_data.loc[processed_position_data["trial_number"].isin(trial_numbers_to_keep)]
    return b_processed_position_data

def get_max_int_SNR(spatial_frequency, powers):
    return 0

def get_max_SNR(spatial_frequency, powers):
    max_SNR = np.abs(np.max(powers)/np.min(powers))
    max_SNR = powers[np.argmax(powers)]
    max_SNR_freq = spatial_frequency[np.argmax(powers)]
    return max_SNR, max_SNR_freq

def add_percentage_hits(spike_data, processed_position_data):
    b_processed_position_data = processed_position_data[processed_position_data["trial_type"] == 0]
    nb_processed_position_data = processed_position_data[processed_position_data["trial_type"] == 1]
    p_processed_position_data = processed_position_data[processed_position_data["trial_type"] == 2]

    if len(b_processed_position_data)>0:
        percentage_beaconed_hits = (len(b_processed_position_data[b_processed_position_data["hit_miss_try"]=="hit"])/len(b_processed_position_data))*100
    else:
        percentage_beaconed_hits = np.nan

    if len(nb_processed_position_data)>0:
        percentage_nonbeaconed_hits = (len(nb_processed_position_data[nb_processed_position_data["hit_miss_try"]=="hit"])/len(nb_processed_position_data))*100
    else:
        percentage_nonbeaconed_hits = np.nan

    if len(p_processed_position_data)>0:
        percentage_probe_hits = (len(p_processed_position_data[p_processed_position_data["hit_miss_try"]=="hit"])/len(p_processed_position_data))*100
    else:
        percentage_probe_hits = np.nan

    percentage_hits = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        percentage_hits.append([percentage_beaconed_hits, percentage_nonbeaconed_hits, percentage_probe_hits])
    spike_data["percentage_hits"] = percentage_hits
    return spike_data

def process_recordings(vr_recording_path_list, of_recording_path_list):

    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            raw_position_data, position_data = syncronise_position_data(recording, get_track_length(recording))

            position_data = add_time_elapsed_collumn(position_data)
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            shuffle_data = pd.read_pickle(recording+"/MountainSort/DataFrames/lomb_shuffle_powers.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")

            # BEHAVIOURAL
            processed_position_data = add_avg_track_speed(processed_position_data, track_length=get_track_length(recording))
            processed_position_data, _ = add_hit_miss_try3(processed_position_data, track_length=get_track_length(recording))
            spike_data = add_percentage_hits(spike_data, processed_position_data)

            # MOVING LOMB PERIODOGRAMS
            #spike_data, shuffle_data = plot_moving_lomb_scargle_periodogram(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length=get_track_length(recording))
            #spike_data = analyse_lomb_powers(spike_data, processed_position_data)
            #spike_data = analyse_lomb_powers_ego_vs_allocentric(spike_data, processed_position_data)
            #shuffle_data = analyse_lomb_powers(shuffle_data, processed_position_data)

            # SPATIAL AUTO CORRELOGRAMS
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data, output_path, track_length=get_track_length(recording), suffix="")
            spike_data = plot_spatial_autocorrelogram_fr(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length=get_track_length(recording), suffix="")

            # FIRING AND BEHAVIOURAL PLOTTING
            #plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))
            #plot_firing_rate_maps_per_trial(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=get_track_length(recording))
            #plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=get_track_length(recording),plot_trials=["beaconed", "non_beaconed", "probe"])
            #plot_stops_on_track(processed_position_data, output_path, track_length=get_track_length(recording))
            #plot_stop_histogram(processed_position_data, output_path, track_length=get_track_length(recording))
            #plot_speed_histogram(processed_position_data, output_path, track_length=get_track_length(recording), suffix="")
            #plot_speed_per_trial(processed_position_data, output_path, track_length=get_track_length(recording))

            #spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            #shuffle_data.to_pickle(recording+"/MountainSort/DataFrames/lomb_shuffle_powers.pkl")

            print("successfully processed and saved vr_grid analysis on "+recording)
        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)


def main():
    print('-------------------------------------------------------------')

    # give a path for a directory of recordings or path of a single recording
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()]
    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M14_D31_2021-06-21_12-07-01', '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D24_2021-06-10_12-01-54', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D18_2021-06-02_12-27-22',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D12_2021-05-25_09-49-23', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D13_2021-05-26_09-46-36', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D14_2021-05-27_10-34-15',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D15_2021-05-28_10-42-15', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D16_2021-05-31_10-21-05', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D17_2021-06-01_10-36-53',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D18_2021-06-02_10-36-39', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D19_2021-06-03_10-50-41', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D1_2021-05-10_10-34-08',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D20_2021-06-04_10-38-58', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D22_2021-06-08_10-55-28', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D23_2021-06-09_10-44-25',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D24_2021-06-10_10-45-20', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D25_2021-06-11_10-55-17', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D26_2021-06-14_10-34-14',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D27_2021-06-15_10-33-47', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D28_2021-06-16_10-34-52', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D29_2021-06-17_10-35-48',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D30_2021-06-18_10-46-48', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D32_2021-06-22_11-08-56', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D33_2021-06-23_11-08-03',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D34_2021-06-24_11-52-48', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D35_2021-06-25_12-02-52', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D37_2021-06-29_11-50-02', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D38_2021-06-30_11-54-56', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D39_2021-07-01_11-47-10',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D3_2021-05-12_09-37-41', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D40_2021-07-02_12-58-24', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D41_2021-07-05_12-05-02',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D43_2021-07-07_11-51-08', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D44_2021-07-08_12-03-21', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D45_2021-07-09_11-39-02',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D5_2021-05-14_09-38-08', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D7_2021-05-18_09-51-25', '/mnt/datastore/Harry/cohort8_may2021/vr/M12_D11_2021-05-24_10-35-27',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M12_D14_2021-05-27_09-55-54', '/mnt/datastore/Harry/cohort8_may2021/vr/M12_D4_2021-05-13_10-30-16', '/mnt/datastore/Harry/cohort8_may2021/vr/M12_D6_2021-05-17_10-26-15',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D17_2021-06-01_11-45-20', '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D25_2021-06-11_12-03-07',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D27_2021-06-15_11-43-42', '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D28_2021-06-16_11-45-54', '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D29_2021-06-17_11-50-37',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D5_2021-05-14_10-53-55', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D11_2021-05-24_11-44-50', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D12_2021-05-25_11-03-39',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D14_2021-05-27_11-46-30', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D15_2021-05-28_12-29-15', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D16_2021-05-31_12-01-35',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D17_2021-06-01_12-47-02', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D19_2021-06-03_12-45-13', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D20_2021-06-04_12-20-57',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D25_2021-06-11_12-36-04', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D26_2021-06-14_12-22-50', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D27_2021-06-15_12-21-58',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D28_2021-06-16_12-26-51', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D29_2021-06-17_12-30-32',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M10_D4_2021-05-13_09-20-38', '/mnt/datastore/Harry/cohort8_may2021/vr/M10_D5_2021-05-14_08-59-54', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D11_2021-05-24_10-00-53',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D33_2021-06-23_12-22-49', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D34_2021-06-24_12-48-57', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D35_2021-06-25_12-41-16',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D37_2021-06-29_12-33-24', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D39_2021-07-01_12-28-46', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D42_2021-07-06_12-38-31',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D5_2021-05-14_11-31-59', '/mnt/datastore/Harry/cohort8_may2021/vr/M15_D6_2021-05-17_12-47-59']
    #vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36']
    #vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D29_2021-06-17_10-35-48']
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()]
    process_recordings(vr_path_list, of_path_list)

    print("look now")

if __name__ == '__main__':
    main()
