import numpy as np
import pandas as pd
import PostSorting.parameters
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from Edmond.VR_grid_analysis.field_analysis import get_field_jitters, get_field_sizes, get_n_fields
from Edmond.VR_grid_analysis.vr_grid_stability_plots import bin_fr_in_space
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
import samplerate
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


def calculate_jitter2(field_coms, track_length, peak_locations, putative_field_frequency):
    jitter_score = np.nan
    if peak_locations == "undefined":
        return jitter_score

    jitter_by_field = []
    for com in field_coms:
        min_distance_to_peak = min(np.abs(peak_locations-com))

        # check if wrapping around trial boundaries minimising the value further
        min_distance_to_peak_wrapped = min(np.abs(peak_locations-track_length-com))
        min_distance_to_peak = min([min_distance_to_peak, min_distance_to_peak_wrapped])

        jitter_by_field.append(min_distance_to_peak)

    jitter_by_field = np.array(jitter_by_field)
    jitter_score = np.nanmean(jitter_by_field)

    # normalise by the track length and the field frequency
    putative_spacing = track_length/putative_field_frequency
    #jitter_score = jitter_score/putative_spacing

    return jitter_score

def calculate_jitter(field_coms, field_sizes, track_length):
    jitter_score = np.nan
    if len(field_coms)>1:
        jitter_by_field = []
        for com, size in zip(field_coms, field_sizes):
            fields_in_size_window = field_coms[(field_coms > (com-(size/2))) & (field_coms < (com+(size/2)))]
            fields_in_size_window = fields_in_size_window[fields_in_size_window != com]

            if com-(size/2) < 0:
                underlapped_fields_in_size_window = field_coms[(field_coms > (track_length+(com-(size/2))))] - track_length
                fields_in_size_window = np.append(fields_in_size_window, underlapped_fields_in_size_window)
            elif com+(size/2) > track_length:
                overlapped_fields_in_size_window = field_coms[(field_coms < ((com+(size/2))-track_length))] + track_length
                fields_in_size_window = np.append(fields_in_size_window, overlapped_fields_in_size_window)

            avg_distances_from_com = np.nanmean(np.abs(com-fields_in_size_window))
            jitter_by_field.append(avg_distances_from_com)
        jitter_by_field = np.array(jitter_by_field)
        jitter_by_field = np.unique(jitter_by_field) # remove duplicate field - field distances
        jitter_score = np.nanstd(jitter_by_field)
        if jitter_score ==0:
            jitter_score = np.nan # not enough fields to calculate metric
    return jitter_score

def get_peak_locations(mean_rate_map, fr_binned_in_space_bin_centres, putative_field_frequency, track_length):
    if putative_field_frequency == 0:
        return "undefined"
    peak_indices = signal.find_peaks(mean_rate_map, distance=(track_length/putative_field_frequency)/2)[0]
    if len(peak_indices) >= putative_field_frequency:
        peak_firing_rates = mean_rate_map[peak_indices]
        peak_locations = fr_binned_in_space_bin_centres[peak_indices]

        arr1inds = peak_firing_rates.argsort()
        peak_location_ordered = np.sort(peak_locations[arr1inds[::-1]][:putative_field_frequency])
        return peak_location_ordered
    else:
        return "undefined"

def add_jitter_by_trial(spike_data, processed_position_data, track_length, pre_post_rz=""):
    jitter_per_trial_cluster = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_field_coms = np.array(cluster_df["fields_com"].iloc[0])
        fields_com_trial_numbers = np.array(cluster_df["fields_com_trial_number"].iloc[0])
        putative_field_frequency = int(np.round(cluster_df["ML_Freqs"].iloc[0]))
        fr_binned_in_space = np.array(cluster_df["fr_binned_in_space"].iloc[0])
        mean_rate_map = np.nanmean(fr_binned_in_space, axis=0)
        fr_binned_in_space_bin_centres = np.array(cluster_df['fr_binned_in_space_bin_centres'].iloc[0])[0]
        peak_locations = get_peak_locations(mean_rate_map, fr_binned_in_space_bin_centres, putative_field_frequency, track_length)

        jitter_per_trial = []
        for i, tn in enumerate(processed_position_data["trial_number"]):
            subset_processed_position_data = processed_position_data[(processed_position_data["trial_number"] == tn)]
            subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])
            trial_fr_binned_in_space = (fr_binned_in_space[subset_trial_numbers-1]).flatten()

            '''
            lags = np.arange(-100, 100, 1) # were looking at 10 timesteps back and 10 forward
            correlogram = []
            for lag in lags:
                corr = stats.pearsonr(np.roll(trial_fr_binned_in_space, lag), mean_rate_map)[0]
                correlogram.append(corr)
            correlogram= np.array(correlogram)
            map_shift = lags[np.argmax(correlogram)]
            '''

            if len(subset_trial_numbers)>0:
                set_mask = np.isin(fields_com_trial_numbers, subset_trial_numbers)
                subset_field_coms = firing_field_coms[set_mask]
                subset_field_coms = subset_field_coms

                field_jitter = calculate_jitter2(subset_field_coms, track_length, peak_locations, putative_field_frequency)
            else:
                field_jitter = np.nan

            #field_jitter=map_shift
            jitter_per_trial.append(field_jitter)

        jitter_per_trial_cluster.append(jitter_per_trial)
    spike_data["jitter_per_trial"] = jitter_per_trial_cluster
    return spike_data


def analyse_fields(spike_data, processed_position_data, track_length, pre_post_rz=""):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30

    fields_per_trial_hmt_by_trial_type = []
    fields_sizes_hmt_by_trial_type = []
    fileds_jitter_hmt_by_trial_type = []
    peak_locations_per_cluster = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_field_coms = np.array(cluster_df["fields_com"].iloc[0])
        fields_com_trial_numbers = np.array(cluster_df["fields_com_trial_number"].iloc[0])
        fields_com_sizes = np.array(cluster_df["fields_com_size"].iloc[0])
        putative_field_frequency = int(np.round(cluster_df["ML_Freqs"].iloc[0]))
        fr_binned_in_space = np.array(cluster_df["fr_binned_in_space"].iloc[0])
        mean_rate_map = np.nanmean(fr_binned_in_space, axis=0)
        fr_binned_in_space_bin_centres = np.array(cluster_df['fr_binned_in_space_bin_centres'].iloc[0])[0]
        peak_locations = get_peak_locations(mean_rate_map, fr_binned_in_space_bin_centres, putative_field_frequency, track_length)

        fields_per_trial_cluster = np.zeros((3,3))
        field_sizes_cluster = np.zeros((3,3))
        field_jitter_cluster = np.zeros((3,3))
        for i, tt in enumerate([0,1,2]):
            for j, hmt in enumerate(["hit", "miss", "try"]):
                subset_processed_position_data = processed_position_data[(processed_position_data["trial_type"] == tt)]
                subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["hit_miss_try"] == hmt)]
                subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])

                if len(subset_trial_numbers)>0:
                    set_mask = np.isin(fields_com_trial_numbers, subset_trial_numbers)

                    subset_field_coms = firing_field_coms[set_mask]
                    subset_field_sizes = fields_com_sizes[set_mask]

                    # count only fields before or after reward zone if specified
                    if pre_post_rz == "_pre_rz":
                        subset_field_sizes = subset_field_sizes[subset_field_coms < reward_zone_start]
                        subset_field_coms = subset_field_coms[subset_field_coms < reward_zone_start]

                    elif pre_post_rz == "_post_rz":
                        subset_field_sizes = subset_field_sizes[subset_field_coms > reward_zone_end]
                        subset_field_coms = subset_field_coms[subset_field_coms > reward_zone_end]
                    else:
                        subset_field_coms = subset_field_coms
                        subset_field_sizes = subset_field_sizes

                    fields_per_trial = len(subset_field_coms)/len(subset_trial_numbers)
                    avg_field_size = np.nanmean(subset_field_sizes)
                    #field_jitter = calculate_jitter(subset_field_coms, subset_field_sizes, track_length)
                    field_jitter = calculate_jitter2(subset_field_coms, track_length, peak_locations, putative_field_frequency)

                    '''
                    map_shifts = []
                    for tn in subset_trial_numbers:
                        subset_processed_position_data = processed_position_data[(processed_position_data["trial_number"] == tn)]
                        subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])
                        trial_fr_binned_in_space = (fr_binned_in_space[subset_trial_numbers-1]).flatten()

                        lags = np.arange(-100, 100, 1) # were looking at 10 timesteps back and 10 forward
                        correlogram = []
                        for lag in lags:
                            corr = stats.pearsonr(np.roll(trial_fr_binned_in_space, lag), mean_rate_map)[0]
                            correlogram.append(corr)
                        correlogram= np.array(correlogram)
                        map_shift = lags[np.argmax(correlogram)]
                        map_shifts.append(map_shift)
                    map_shifts=np.array(map_shifts)
                    field_jitter = np.nanmean(np.abs(map_shifts))
                    '''

                else:
                    fields_per_trial = np.nan
                    avg_field_size = np.nan
                    field_jitter = np.nan

                fields_per_trial_cluster[i,j] = fields_per_trial
                field_sizes_cluster[i,j] = avg_field_size
                field_jitter_cluster[i,j] = field_jitter

        peak_locations_per_cluster.append(peak_locations)
        fields_per_trial_hmt_by_trial_type.append(fields_per_trial_cluster.tolist())
        fields_sizes_hmt_by_trial_type.append(field_sizes_cluster.tolist())
        fileds_jitter_hmt_by_trial_type.append(field_jitter_cluster.tolist())

    spike_data["fields_per_trial_hmt_by_trial_type"+pre_post_rz] = fields_per_trial_hmt_by_trial_type
    spike_data["fields_sizes_hmt_by_trial_type"+pre_post_rz] = fields_sizes_hmt_by_trial_type
    spike_data["fields_jitter_hmt_by_trial_type"+pre_post_rz] = fileds_jitter_hmt_by_trial_type
    spike_data["peak_locations"] = peak_locations_per_cluster
    return spike_data


def calculate_grid_field_com(cluster_spike_data, position_data, track_length):
    '''
    :param spike_data:
    :param prm:
    :return:

    for each trial of each trial type we want to
    calculate the centre of mass of all detected field
    centre of mass is defined as

    '''

    firing_field_com = []
    firing_field_com_trial_numbers = []
    firing_fields_com_size_cluster = []

    firing_times=cluster_spike_data.firing_times/(settings.sampling_rate/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    if len(firing_times)==0:
        return firing_field_com, firing_field_com_trial_numbers, firing_fields_com_size_cluster

    trial_numbers = np.array(position_data['trial_number'].to_numpy())
    trial_types = np.array(position_data['trial_type'].to_numpy())
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

    for local_maximum_idx in local_maxima_bin_idx:
        neighbouring_local_mins = find_neighbouring_minima(firing_rate_map, local_maximum_idx)
        closest_minimum_bin_idx = neighbouring_local_mins[np.argmin(np.abs(neighbouring_local_mins-local_maximum_idx))]
        field_size_in_bins = neighbouring_local_mins[1]-neighbouring_local_mins[0]

        if firing_rate_map[local_maximum_idx] - firing_rate_map[closest_minimum_bin_idx] > field_threshold:
            #firing_field.append(neighbouring_local_mins)

            field =  firing_rate_map[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
            field_bins = bin_centres[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
            field_weights = field/np.sum(field)
            field_com = np.sum(field_weights*field_bins)
            field_size = field_size_in_bins*settings.vr_grid_analysis_bin_size

            # reverse calculate the field_com in cm from track start
            trial_number = (field_com//track_length)+1
            trial_type = stats.mode(trial_types[trial_numbers==trial_number])[0][0]
            field_com = field_com%track_length

            firing_field_com.append(field_com)
            firing_field_com_trial_numbers.append(trial_number)
            firing_fields_com_size_cluster.append(field_size)

    return firing_field_com, firing_field_com_trial_numbers, firing_fields_com_size_cluster

def process_vr_grid(spike_data, position_data, track_length):

    fields_com_cluster = []
    fields_com_trial_numbers_cluster = []
    fields_com_size_cluster = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        fields_com, field_com_trial_numbers, fields_com_size = calculate_grid_field_com(cluster_df, position_data, track_length)

        fields_com_cluster.append(fields_com)
        fields_com_trial_numbers_cluster.append(field_com_trial_numbers)
        fields_com_size_cluster.append(fields_com_size)

    spike_data["fields_com"] = fields_com_cluster
    spike_data["fields_com_trial_number"] = fields_com_trial_numbers_cluster
    spike_data["fields_com_size"] = fields_com_size_cluster
    return spike_data

def order_and_start_from_trial_number_1(trial_numbers, processed_position_data):
    new_trial_numbers=[]
    for i, tn in enumerate(processed_position_data["trial_number"]):
        if len(trial_numbers[trial_numbers == tn])>0:
            trials_to_add = np.ones(len(trial_numbers[trial_numbers == tn]))*(i+1)
            for j in trials_to_add:
                new_trial_numbers.append(j)
    new_trial_numbers = np.array(new_trial_numbers)
    return new_trial_numbers

    #new_trial_numbers=[]
    #diff = np.diff(trial_numbers)
    #new_tn = 1
    #for i, tn in enumerate(trial_numbers):
    #    if (i!=0) and (diff[i-1]!=0):
    #        new_tn+=1
    #    new_trial_numbers.append(new_tn)
    #new_trial_numbers = np.array(new_trial_numbers)
    #return new_trial_numbers

def plot_field_centre_of_mass_on_track_by_tt(spike_data, processed_position_data, output_path, track_length, plot_trials=[2, 0, 1]):
    hmt="hit"
    print('plotting field rastas...')
    save_path = output_path + '/Figures/field_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data = processed_position_data[processed_position_data["hit_miss_try"]=="hit"]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data.firing_times.iloc[0]
        if len(firing_times_cluster)>1:
            cluster_firing_com = np.array(cluster_spike_data["fields_com"].iloc[0])
            cluster_firing_com_trial_numbers = np.array(cluster_spike_data["fields_com_trial_number"].iloc[0])
            peak_locations = np.array(cluster_spike_data["peak_locations"].iloc[0])

            fig, axes = plt.subplots(len(plot_trials), 1, figsize=(6,6), sharex=True)
            for ax, tt, color in zip(axes, plot_trials, ["red", "black", "blue"]):
                tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
                if peak_locations != "undefined":
                    for j in range(len(peak_locations)):
                        ax.axvline(x=peak_locations[j], color="black", linewidth=2,linestyle="solid", alpha=1)

                jitter = get_field_jitters(cluster_spike_data, hmt=hmt, tt=tt, pre_post_rz="")[0]
                field_size = get_field_sizes(cluster_spike_data, hmt=hmt, tt=tt, pre_post_rz="")[0]
                n_fields = get_n_fields(cluster_spike_data, hmt=hmt, tt=tt, pre_post_rz="")[0]
                if tt == 0:
                    style_track_plot(ax, track_length)
                else:
                    style_track_plot_no_RZ(ax, track_length)
                set_mask = np.isin(cluster_firing_com_trial_numbers, np.array(processed_position_data[processed_position_data["trial_type"] == tt]["trial_number"]))
                trial_numbers = cluster_firing_com_trial_numbers[set_mask]
                trial_numbers = order_and_start_from_trial_number_1(trial_numbers, tt_processed_position_data)
                ax.plot(cluster_firing_com[set_mask], trial_numbers, "s", color=color, markersize=4)
                ax.set_title(("Jitter = "+str(np.round(jitter, decimals=2))+ "cm"), fontsize=20)
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.set_yticks([len(tt_processed_position_data)])
                Edmond.plot_utility2.style_vr_plot(ax, x_max=len(tt_processed_position_data))
                ax.set_ylim([0, len(tt_processed_position_data)])
                ax.set_xlim([0,track_length])
                #print(ax.get_ylim())

            fig.tight_layout(pad=2.0)
            #axes[1].set_ylabel('Firing Field on Trials', fontsize=25, labelpad = 15)
            ax.set_xlabel('Location (cm)', fontsize=25, labelpad = 10)
            ax.set_xticks([0, 100, 200])
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.xticks(fontsize=20)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_new_by_tt_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()
    return

def plot_field_centre_of_mass_on_track_by_hmt(spike_data, processed_position_data, output_path, track_length, plot_trials=["hit", "try", "miss"]):
    tt=1
    print('plotting field rastas...')
    save_path = output_path + '/Figures/field_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data = processed_position_data[processed_position_data["trial_type"]==tt]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data.firing_times.iloc[0]
        if len(firing_times_cluster)>1:
            cluster_firing_com = np.array(cluster_spike_data["fields_com"].iloc[0])
            cluster_firing_com_trial_numbers = np.array(cluster_spike_data["fields_com_trial_number"].iloc[0])
            peak_locations = np.array(cluster_spike_data["peak_locations"].iloc[0])

            fig, axes = plt.subplots(len(plot_trials), 1, figsize=(6,6), sharex=True)
            for ax, hmt, color in zip(axes, plot_trials, ["green", "orange", "red"]):
                hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
                if peak_locations != "undefined":
                    for j in range(len(peak_locations)):
                        ax.axvline(x=peak_locations[j], color="black", linewidth=2,linestyle="solid", alpha=1)

                jitter = get_field_jitters(cluster_spike_data, hmt=hmt, tt=tt, pre_post_rz="")[0]
                field_size = get_field_sizes(cluster_spike_data, hmt=hmt, tt=tt, pre_post_rz="")[0]
                n_fields = get_n_fields(cluster_spike_data, hmt=hmt, tt=tt, pre_post_rz="")[0]

                if tt == 0:
                    style_track_plot(ax, track_length)
                else:
                    style_track_plot_no_RZ(ax, track_length)
                set_mask = np.isin(cluster_firing_com_trial_numbers, np.array(processed_position_data[processed_position_data["hit_miss_try"] == hmt]["trial_number"]))
                trial_numbers = cluster_firing_com_trial_numbers[set_mask]
                trial_numbers = order_and_start_from_trial_number_1(trial_numbers, hmt_processed_position_data)
                ax.plot(cluster_firing_com[set_mask], trial_numbers, "s", color=color, markersize=4)
                ax.set_title(("Jitter = "+str(np.round(jitter, decimals=2))+ "cm"), fontsize=20)
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.set_yticks([len(hmt_processed_position_data)])
                Edmond.plot_utility2.style_vr_plot(ax,  x_max=len(hmt_processed_position_data))
                ax.set_ylim([0, len(hmt_processed_position_data)])
                ax.set_xlim([0,track_length])
                #print(ax.get_ylim())

            fig.tight_layout(pad=2.0)
            axes[1].set_ylabel('Firing Field on Trials', fontsize=25, labelpad = 15)
            ax.set_xlabel('Location (cm)', fontsize=25, labelpad = 10)
            ax.set_xticks([0, 100, 200])
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.xticks(fontsize=20)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_new_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()
    return


def plot_field_centre_of_mass_on_track(spike_data, processed_position_data, output_path, track_length, plot_trials=["hit", "try", "miss"]):

    print('plotting field rastas...')
    save_path = output_path + '/Figures/field_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data = processed_position_data[processed_position_data["trial_type"]==1]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:

            x_max = max(processed_position_data["trial_number"])
            spikes_on_track = plt.figure(figsize=(6,6))
            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            cluster_firing_com_trial_numbers = np.array(spike_data["fields_com_trial_number"].iloc[cluster_index])

            hit_mask = np.isin(cluster_firing_com_trial_numbers, np.array(processed_position_data[processed_position_data["hit_miss_try"] == "hit"]["trial_number"]))
            try_mask = np.isin(cluster_firing_com_trial_numbers, np.array(processed_position_data[processed_position_data["hit_miss_try"] == "try"]["trial_number"]))
            miss_mask = np.isin(cluster_firing_com_trial_numbers, np.array(processed_position_data[processed_position_data["hit_miss_try"] == "miss"]["trial_number"]))
            if "hit" in plot_trials:
                ax.plot(cluster_firing_com[hit_mask], cluster_firing_com_trial_numbers[hit_mask], "s", color='green', markersize=4)
            if "try" in plot_trials:
                ax.plot(cluster_firing_com[try_mask], cluster_firing_com_trial_numbers[try_mask], "s", color='orange', markersize=4)
            if "miss" in plot_trials:
                ax.plot(cluster_firing_com[miss_mask], cluster_firing_com_trial_numbers[miss_mask], "s", color='red', markersize=4)

            plt.ylabel('Field COM on trials', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            Edmond.plot_utility2.style_track_plot(ax, track_length)
            Edmond.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()
    return

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

def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array


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

def process_recordings(vr_recording_path_list, of_recording_path_list):

    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            #raw_position_data, position_data = syncronise_position_data(recording, get_track_length(recording))

            position_data = add_time_elapsed_collumn(position_data)
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")

            processed_position_data = add_avg_RZ_speed(processed_position_data, track_length=get_track_length(recording))
            processed_position_data = add_avg_track_speed(processed_position_data, track_length=get_track_length(recording))
            processed_position_data = add_RZ_bias(processed_position_data)
            processed_position_data, _ = add_hit_miss_try(processed_position_data, track_length=get_track_length(recording))
            processed_position_data, _ = add_hit_miss_try3(processed_position_data, track_length=get_track_length(recording))

            #spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length=get_track_length(recording))

            # FIELD ANALYSIS
            #spike_data = process_vr_grid(spike_data, position_data, track_length=get_track_length(recording))
            spike_data = analyse_fields(spike_data, processed_position_data, track_length=get_track_length(recording))
            spike_data = analyse_fields(spike_data, processed_position_data, track_length=get_track_length(recording), pre_post_rz="_pre_rz")
            spike_data = analyse_fields(spike_data, processed_position_data, track_length=get_track_length(recording), pre_post_rz="_post_rz")
            spike_data = add_jitter_by_trial(spike_data, processed_position_data, track_length=get_track_length(recording))

            plot_field_centre_of_mass_on_track(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))
            plot_field_centre_of_mass_on_track_by_hmt(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))
            plot_field_centre_of_mass_on_track_by_tt(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))

            spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
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

    #test_jitter_score(save_path="/mnt/datastore/Harry/Vr_grid_cells/test/jitter_score_test")

    # give a path for a directory of recordings or path of a single recording
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()]
    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M13_D24_2021-06-10_12-01-54', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D18_2021-06-02_12-27-22', '/mnt/datastore/Harry/cohort8_may2021/vr/M10_D4_2021-05-13_09-20-38', '/mnt/datastore/Harry/cohort8_may2021/vr/M10_D5_2021-05-14_08-59-54', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D11_2021-05-24_10-00-53',
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
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D28_2021-06-16_12-26-51', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D29_2021-06-17_12-30-32', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D31_2021-06-21_12-07-01',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D33_2021-06-23_12-22-49', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D34_2021-06-24_12-48-57', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D35_2021-06-25_12-41-16',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D37_2021-06-29_12-33-24', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D39_2021-07-01_12-28-46', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D42_2021-07-06_12-38-31',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D5_2021-05-14_11-31-59', '/mnt/datastore/Harry/cohort8_may2021/vr/M15_D6_2021-05-17_12-47-59']
    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36']
    #vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D30_2021-06-18_10-46-48']
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()]
    process_recordings(vr_path_list, of_path_list)

    print("look now")

if __name__ == '__main__':
    main()
