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
    for trial_number in np.unique(processed_position_data["trial_number"]):
        trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number]
        rewarded = trial_processed_position_data["rewarded"].iloc[0]
        RZ_speed = trial_processed_position_data["avg_speed_in_RZ"].iloc[0]
        track_speed = trial_processed_position_data["avg_speed_on_track"].iloc[0]
        TI = trial_processed_position_data["RZ_stop_bias"].iloc[0]

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
    firing_field_com_trial_types = []
    firing_rate_maps = []
    firing_rates = []

    firing_times=cluster_spike_data.firing_times/(settings.sampling_rate/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    if len(firing_times)==0:
        firing_rate_maps = np.zeros(int(track_length))
        return firing_field_com, firing_field_com_trial_numbers, firing_field_com_trial_types, firing_rate_maps

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

            # reverse calculate the field_com in cm from track start
            trial_number = (field_com//track_length)+1
            trial_type = stats.mode(trial_types[trial_numbers==trial_number])[0][0]
            field_com = field_com%track_length

            firing_field_com.append(field_com)
            firing_field_com_trial_numbers.append(trial_number)
            firing_field_com_trial_types.append(trial_type)

    for trial_number in np.unique(trial_numbers):
        trial_x_position_cm = x_position_cm[trial_numbers==trial_number]
        trial_time_seconds = time_seconds[trial_numbers==trial_number]
        time_elapsed = trial_time_seconds[-1] - trial_time_seconds[0]
        number_of_spikes = len(trial_x_position_cm)

        trial_instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[trial_numbers==trial_number]

        numerator, bin_edges = np.histogram(trial_x_position_cm, bins=int(track_length/settings.vr_grid_analysis_bin_size), range=(0, track_length), weights=trial_instantaneous_firing_rate_per_ms)
        denominator, bin_edges = np.histogram(trial_x_position_cm, bins=int(track_length/settings.vr_grid_analysis_bin_size), range=(0, track_length))
        mean_firing_rate = number_of_spikes/time_elapsed

        firing_rate_map = numerator/denominator
        firing_rate_maps.append(firing_rate_map)
        firing_rates.append(mean_firing_rate)

    return firing_field_com, firing_field_com_trial_numbers, firing_field_com_trial_types, firing_rate_maps, firing_rates


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

def process_vr_grid(spike_data, position_data, track_length):

    fields_com_cluster = []
    fields_com_trial_numbers_cluster = []
    fields_com_trial_types_cluster = []
    firing_rate_maps_cluster = []

    minimum_distance_to_field_in_next_trial =[]
    fields_com_next_trial_type = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        fields_com, field_com_trial_numbers, field_com_trial_types, firing_rate_maps, firing_rates = calculate_grid_field_com(cluster_df, position_data, track_length)

        next_trial_type_cluster = []
        minimum_distance_to_field_in_next_trial_cluster=[]

        for i in range(len(fields_com)):
            field = fields_com[i]
            trial_number=field_com_trial_numbers[i]
            trial_type = int(field_com_trial_types[i])

            trial_type_tmp = position_data["trial_type"].to_numpy()
            trial_number_tmp = position_data["trial_number"].to_numpy()

            fields_in_next_trial = np.array(fields_com)[np.array(field_com_trial_numbers) == int(trial_number+1)]
            fields_in_next_trial = fields_in_next_trial[(fields_in_next_trial>50) & (fields_in_next_trial<150)]

            if len(fields_in_next_trial)>0:
                next_trial_type = int(np.unique(trial_type_tmp[trial_number_tmp == int(trial_number+1)])[0])
                minimum_field_difference = min(np.abs(fields_in_next_trial-field))

                minimum_distance_to_field_in_next_trial_cluster.append(minimum_field_difference)
                next_trial_type_cluster.append(next_trial_type)
            else:
                minimum_distance_to_field_in_next_trial_cluster.append(np.nan)
                next_trial_type_cluster.append(np.nan)

        fields_com_cluster.append(fields_com)
        fields_com_trial_numbers_cluster.append(field_com_trial_numbers)
        fields_com_trial_types_cluster.append(field_com_trial_types)
        firing_rate_maps_cluster.append(firing_rate_maps)

        minimum_distance_to_field_in_next_trial.append(minimum_distance_to_field_in_next_trial_cluster)
        fields_com_next_trial_type.append(next_trial_type_cluster)

    spike_data["fields_com"] = fields_com_cluster
    spike_data["fields_com_trial_number"] = fields_com_trial_numbers_cluster
    spike_data["fields_com_trial_type"] = fields_com_trial_types_cluster
    spike_data["firing_rate_maps"] = firing_rate_maps_cluster
    spike_data["minimum_distance_to_field_in_next_trial"] = minimum_distance_to_field_in_next_trial
    spike_data["fields_com_next_trial_type"] = fields_com_next_trial_type

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

def plot_spatial_autocorrelogram_fr(spike_data, processed_position_data, output_path, track_length, suffix=""):
    print('plotting spike spatial autocorrelogram fr...')
    save_path = output_path + '/Figures/spatial_autocorrelograms_fr'
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


def calculate_moving_window_lomb_for_cluster_parallel(cluster_id, processed_position_data, position_data, elapsed_distance30, spike_data, track_length, save_path):
    shuffle_data = pd.DataFrame()
    new_spike_data = pd.DataFrame()

    # get trial numbers to use from processed_position_data
    trial_number_to_use = np.unique(processed_position_data["trial_number"])
    trial_numbers = np.array(position_data["trial_number"])
    x_positions = np.array(position_data["x_position_cm"])
    x_positions_elapsed = x_positions+(trial_numbers*track_length)-track_length
    n_trials = max(trial_numbers)

    # get denominator and handle nans
    denominator, _ = np.histogram(elapsed_distance30, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))

    # only access the cluster given
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
        step = 0.01
        frequency = np.arange(0.1, 10+step, step)
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

        avg_power = np.nanmean(powers, axis=0)
        max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
        max_power = avg_power[np.argmax(avg_power)]
        max_SNR_text = "Power: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
        max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))

        #====================================================# Attempt bootstrapped approach using standard spike time shuffling procedure
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
            indices_to_test = np.arange(0, len(fr)-sliding_window_size, 1, dtype=np.int64)[::10]
            for m in indices_to_test:
                ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr[m:m+sliding_window_size])
                shuffle_power = ls.power(frequency)
                shuffle_powers.append(shuffle_power.tolist())
                shuffle_centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
            shuffle_powers = np.array(shuffle_powers)
            shuffle_centre_distances = np.array(shuffle_centre_distances)
            avg_shuffle_powers = np.nanmean(shuffle_powers, axis=0)
            max_shuffle_freq = frequency[np.argmax(avg_shuffle_powers)]
            max_shuffle_power = avg_shuffle_powers[np.argmax(avg_shuffle_powers)]
            single_shuffle=pd.DataFrame()
            single_shuffle["cluster_id"] = [cluster_id]
            single_shuffle["shuffle_id"] = [i]
            single_shuffle["shuffle_powers"] = [shuffle_powers]
            single_shuffle["avg_shuffle_powers"] = [avg_shuffle_powers]
            single_shuffle["max_shuffle_freq"] = [max_shuffle_freq]
            single_shuffle["max_shuffle_power"] = [max_shuffle_power]
            single_shuffle["shuffle_centre_distances"] = [shuffle_centre_distances]
            shuffle_data = pd.concat([shuffle_data, single_shuffle], ignore_index=True)
        #====================================================# Attempt bootstrapped approach using standard spike time shuffling procedure

        n_x_ticks = int(max(centre_trials)//50)+1
        x_tick_locs= np.linspace(0, len(centre_distances)-1, n_x_ticks, dtype=np.int64)
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.imshow(powers.T, origin='lower', aspect="auto", cmap="jet")
        plt.ylabel('Spatial Frequency', fontsize=20, labelpad = 10)
        plt.xlabel('Centre Trial', fontsize=20, labelpad = 10) #TODO Change this to centre trial
        ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
        ax.set_yticklabels([0, 2, 4, 6, 8, 10])
        ax.set_xticks(x_tick_locs.tolist())
        ax.set_xticklabels(np.take(centre_distances, x_tick_locs).astype(np.int64).tolist())
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_spatial_moving_lomb_scargle_periodogram_Cluster_' + str(cluster_id) +'.png', dpi=300)
        plt.close()

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(frequency, avg_power, color="black")
        '''
        for tt in [0, 1, 2]:
            for hmt in ["hit", "miss", "try"]:
                subset_trial_numbers = processed_position_data[(processed_position_data["hit_miss_try"] == hmt) & (processed_position_data["trial_type"] == tt)]["trial_number"]
                if len(subset_trial_numbers)>0:
                    subset_trial_numbers = np.asarray(subset_trial_numbers)
                    subset_mask = np.isin(centre_trials, subset_trial_numbers)
                    subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                    subset_powers = powers.copy()
                    subset_powers[subset_mask == False] = np.nan
                    avg_subset_powers = np.nanmean(subset_powers, axis=0)
                    ax.plot(frequency, avg_subset_powers, color=get_tt_color(tt), linestyle=get_hmt_linestyle(hmt), linewidth=0.5)
        '''
        ax.text(0.9, 0.9, max_SNR_text, ha='right', va='center', transform=ax.transAxes, fontsize=10)
        ax.text(0.9, 0.8, max_SNR_freq_test, ha='right', va='center', transform=ax.transAxes, fontsize=10)
        plt.xlabel('Spatial Frequency', fontsize=20, labelpad = 10)
        plt.ylabel('Power', fontsize=20, labelpad = 10)
        plt.xlim(0,max(frequency))
        plt.ylim(bottom=0)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_spatial_moving_lomb_scargle_avg_periodogram_Cluster_' + str(cluster_id) + '.png', dpi=300)
        plt.close()

        new_spike_data["MOVING_LOMB_freqs"] = [max_SNR_freq]
        new_spike_data["MOVING_LOMB_SNR"] = [max_SNR]
        new_spike_data["MOVING_LOMB_power"] = [max_power]
        new_spike_data["MOVING_LOMB_all_powers"] = [powers]
        new_spike_data["MOVING_LOMB_all_centre_trials"] = [centre_trials]

    else:
        new_spike_data["MOVING_LOMB_freqs"] = [np.nan]
        new_spike_data["MOVING_LOMB_SNR"] = [np.nan]
        new_spike_data["MOVING_LOMB_power"] = [np.nan]
        new_spike_data["MOVING_LOMB_all_powers"] = [np.nan]
        new_spike_data["MOVING_LOMB_all_centre_trials"] = [np.nan]

    return (new_spike_data, shuffle_data)


def plot_moving_lomb_scargle_periodogram_parallel(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length):
    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    clusters = spike_data.cluster_id
    num_cores = int(os.environ['HEATMAP_CONCURRENCY']) if os.environ.get('HEATMAP_CONCURRENCY') else multiprocessing.cpu_count()
    num_cores = 4
    print("I have detected", str(num_cores), " cores")

    # get distances from raw position data which has been smoothened
    rpd = np.asarray(raw_position_data["x_position_cm"])
    tn = np.asarray(raw_position_data["trial_number"])
    elapsed_distance30 = rpd+(tn*track_length)-track_length

    packed_results = Parallel(n_jobs=num_cores)(delayed(calculate_moving_window_lomb_for_cluster_parallel)(cluster, processed_position_data, position_data, elapsed_distance30,
                                                                                                           spike_data, track_length, save_path) for cluster in clusters)

    lomb_spike_data = unpack_parallel_lomb_data(packed_results, column=0)
    lomb_shuffle_spike_data = unpack_parallel_lomb_data(packed_results, column=1)

    spike_data["MOVING_LOMB_freqs"] = lomb_spike_data["MOVING_LOMB_freqs"]
    spike_data["MOVING_LOMB_SNR"] = lomb_spike_data["MOVING_LOMB_SNR"]
    spike_data["MOVING_LOMB_all_powers"] = lomb_spike_data["MOVING_LOMB_all_powers"]
    spike_data["MOVING_LOMB_all_centre_trials"] = lomb_spike_data["MOVING_LOMB_all_centre_trials"]

    return spike_data, lomb_shuffle_spike_data

def unpack_parallel_lomb_data(packed_results, column):
    unpacked_lomb_data = pd.DataFrame()
    for i in range(len(packed_results)):
        lomb_data = packed_results[i][column]
        unpacked_lomb_data = pd.concat([unpacked_lomb_data, lomb_data], ignore_index=True)
    return unpacked_lomb_data


def plot_moving_lomb_scargle_periodogram(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length):
    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    clusters = spike_data.cluster_id

    shuffle_data = pd.DataFrame()

    # get trial numbers to use from processed_position_data
    trial_number_to_use = np.unique(processed_position_data["trial_number"])
    trial_numbers = np.array(position_data["trial_number"])
    x_positions = np.array(position_data["x_position_cm"])
    x_positions_elapsed = x_positions+(trial_numbers*track_length)-track_length
    n_trials = max(trial_numbers)

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
            step = 0.01
            frequency = np.arange(0.1, 10+step, step)
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

            avg_power = np.nanmean(powers, axis=0)
            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
            max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
            max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))

            #====================================================# Attempt bootstrapped approach using standard spike time shuffling procedure
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
                indices_to_test = np.arange(0, len(fr)-sliding_window_size, 1, dtype=np.int64)[::10]
                for m in indices_to_test:
                    ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr[m:m+sliding_window_size])
                    shuffle_power = ls.power(frequency)
                    shuffle_powers.append(shuffle_power.tolist())
                    shuffle_centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
                shuffle_powers = np.array(shuffle_powers)
                shuffle_centre_distances = np.array(shuffle_centre_distances)
                shuffle_centre_distances = np.round(shuffle_centre_distances).astype(np.int64)
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


            #====================================================# Attempt bootstrapped approach using standard spike time shuffling procedure


            n_x_ticks = int(max(centre_trials)//50)+1
            x_tick_locs= np.linspace(0, len(centre_distances)-1, n_x_ticks, dtype=np.int64)
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            powers[np.isnan(powers)] = 0
            ax.imshow(powers.T, origin='lower', aspect="auto", cmap="jet")
            plt.ylabel('Spatial Frequency', fontsize=20, labelpad = 10)
            plt.xlabel('Centre Trial', fontsize=20, labelpad = 10) #TODO Change this to centre trial
            ax.set_yticks([0, 200, 400, 600, 800, 1000])
            ax.set_yticklabels([0, 2, 4, 6, 8, 10])
            ax.set_xticks(x_tick_locs.tolist())
            ax.set_xticklabels(np.take(centre_distances, x_tick_locs).astype(np.int64).tolist())
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_scargle_periodogram_Cluster_' + str(cluster_id) +'.png', dpi=300)
            plt.close()

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(frequency, avg_power, color="yellow")

            for tt in [0, 1, 2]:
                for hmt in ["hit", "miss", "try"]:
                    subset_trial_numbers = processed_position_data[(processed_position_data["hit_miss_try"] == hmt) & (processed_position_data["trial_type"] == tt)]["trial_number"]
                    if len(subset_trial_numbers)>0:
                        subset_trial_numbers = np.asarray(subset_trial_numbers)
                        subset_mask = np.isin(centre_trials, subset_trial_numbers)
                        subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                        subset_powers = powers.copy()
                        subset_powers[subset_mask == False] = np.nan
                        avg_subset_powers = np.nanmean(subset_powers, axis=0)
                        ax.plot(frequency, avg_subset_powers, color=get_tt_color(tt), linestyle=get_hmt_linestyle(hmt), linewidth=0.5)

            ax.text(0.9, 0.9, max_SNR_text, ha='right', va='center', transform=ax.transAxes, fontsize=10)
            ax.text(0.9, 0.8, max_SNR_freq_test, ha='right', va='center', transform=ax.transAxes, fontsize=10)
            plt.xlabel('Spatial Frequency', fontsize=20, labelpad = 10)
            plt.ylabel('Power', fontsize=20, labelpad = 10)
            plt.xlim(0,max(frequency))
            plt.ylim(bottom=0)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_scargle_avg_periodogram_Cluster_' + str(cluster_id) + '.png', dpi=300)
            plt.close()

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

def analyse_lomb_powers(spike_data, processed_position_data):
    '''
    Requires the collumns of MOVING LOMB PERIODOGRAM
    This function takes the moving window periodogram and computes the average SNR, and for all combinations of trial types and hit miss try behaviours
    :param spike_data:
    :param processed_position_data:
    :param track_length:
    :return:
    '''

    step = 0.01
    frequency = np.arange(0.1, 10+step, step)

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

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])

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
                        max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_subset_powers)
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

def plot_lomb_scargle_periodogram(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length, suffix="", GaussianKernelSTD_ms=5, fr_integration_window=2):
    print('plotting lomb_scargle periodogram...')
    save_path = output_path + '/Figures/lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # get trial numbers to use from processed_position_data
    trial_number_to_use = np.unique(processed_position_data["trial_number"])
    trial_numbers = np.array(position_data["trial_number"])
    x_positions = np.array(position_data["x_position_cm"])
    x_positions_elapsed = x_positions+(trial_numbers*track_length)-track_length
    n_trials = max(trial_numbers)

    # get distances from raw position data which has been smoothened
    rpd = np.asarray(raw_position_data["x_position_cm"])
    tn = np.asarray(raw_position_data["trial_number"])
    elapsed_distance30 = rpd+(tn*track_length)-track_length

    # get denominator and handle nans
    denominator, _ = np.histogram(elapsed_distance30, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))

    freqs = []
    SNRs = []
    SNR_thresholds = []
    freq_thresholds = []
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
            step = 0.01
            frequency = np.arange(0.1, 10+step, step)
            ls = LombScargle(elapsed_distance, fr)
            power = ls.power(frequency)
            max_SNR, max_SNR_freq = get_max_SNR(frequency, power)
            max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
            max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))

            #====================================================# Attempt bootstrapped approach using standard spike time shuffling procedure
            SNR_shuffles = []
            freqs_shuffles = []
            for i in range(100):
                random_firing_additions = np.random.randint(low=int(20*settings.sampling_rate), high=int(580*settings.sampling_rate), size=len(firing_times_cluster))
                shuffled_firing_times = firing_times_cluster + random_firing_additions
                shuffled_firing_times[shuffled_firing_times >= recording_length_sampling_points] = shuffled_firing_times[shuffled_firing_times >= recording_length_sampling_points] - recording_length_sampling_points # wrap around the firing times that exceed the length of the recording

                #shuffled_firing_times = np.random.randint(low=0, high=recording_length_sampling_points, size=len(firing_times_cluster))

                shuffled_firing_locations_elapsed = elapsed_distance30[shuffled_firing_times.astype(np.int64)]
                numerator, bin_edges = np.histogram(shuffled_firing_locations_elapsed, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))
                fr = numerator/denominator
                elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length

                # remove nan values that coincide with start and end of the track before convolution
                fr[fr==np.inf] = np.nan
                nan_mask = ~np.isnan(fr)
                fr = fr[nan_mask]
                elapsed_distance = elapsed_distance[nan_mask]

                fr = convolve(fr, gauss_kernel)
                fr = moving_sum(fr, window=2)/2
                fr = np.append(fr, np.zeros(len(elapsed_distance)-len(fr)))

                fig = plt.figure(figsize=(12,4))
                ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
                #ax.plot(recorded_location)
                ax.plot(elapsed_distance[:2000], fr[:2000])
                #plt.savefig('/mnt/datastore/Harry/cohort8_may2021/vr/M14_D45_2021-07-09_12-15-03/MountainSort/tmp.png', dpi=200)
                plt.close()

                # make and apply the set mask
                fr = fr[set_mask]
                elapsed_distance = elapsed_distance[set_mask]

                ls_boot = LombScargle(elapsed_distance, fr)
                shuffle_power = ls_boot.power(frequency)
                max_SNR_shuffle, max_freq_shuffle = get_max_SNR(frequency, shuffle_power)
                SNR_shuffles.append(max_SNR_shuffle)
                freqs_shuffles.append(max_freq_shuffle)
            SNR_shuffles = np.array(SNR_shuffles)
            freqs_shuffles = np.array(freqs_shuffles)
            freq_threshold = np.nanpercentile(distance_from_integer(freqs_shuffles), 1)
            SNR_threshold = np.nanpercentile(SNR_shuffles, 99)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
            ax1.set_ylabel("SNR",color="black",fontsize=15, labelpad=10)
            ax1.set_xlabel("Spatial Frequency", color="black", fontsize=15, labelpad=10)
            ax1.set_xticks(np.arange(0, 11, 1.0))
            ax2.set_xticks([0,0.25, 0.5])
            ax1.axhline(y=SNR_threshold, xmin=0, xmax=max(frequency), color="black", linestyle="dashed")
            plt.setp(ax1.get_xticklabels(), fontsize=15)
            plt.setp(ax2.get_xticklabels(), fontsize=10)
            ax1.yaxis.set_ticks_position('left')
            ax1.xaxis.set_ticks_position('bottom')
            ax1.xaxis.grid() # vertical lines
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            ax1.scatter(x=freqs_shuffles, y=SNR_shuffles, color="k", marker="o", alpha=0.3)
            ax1.scatter(x=max_SNR_freq, y=max_SNR, color="r", marker="x")
            ax1.set_xlim([0,10])
            ax1.set_ylim([1,3000])
            ax2.set_ylim([1,3000])
            ax2.set_xlim([-0.1,0.6])
            ax2.set_xlabel(r'$\Delta$ from Integer', color="black", fontsize=15, labelpad=10)
            ax2.axvline(x=freq_threshold, color="black", linestyle="dashed")
            ax2.scatter(x=distance_from_integer(freqs_shuffles), y=SNR_shuffles, color="k", marker="o", alpha=0.3)
            ax2.scatter(x=distance_from_integer(max_SNR_freq), y=max_SNR, color="r", marker="x")
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_lomb_scargle_periodogram_shuffdist_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
            plt.close()
            #====================================================# Attempt bootstrapped approach using standard spike time shuffling procedure

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(frequency, power, color="blue")
            ax.text(0.9, 0.9, max_SNR_text, ha='right', va='center', transform=ax.transAxes, fontsize=10)
            ax.text(0.9, 0.8, max_SNR_freq_test, ha='right', va='center', transform=ax.transAxes, fontsize=10)
            far = ls.false_alarm_level(1-(1.e-10))
            ax.axhline(y=far, xmin=0, xmax=max(frequency), color="black", linestyle="dashed")
            plt.ylabel('Power', fontsize=20, labelpad = 10)
            plt.xlabel('Spatial Frequency', fontsize=20, labelpad = 10)
            plt.xlim(0,max(frequency))
            plt.ylim(bottom=0)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_lomb_scargle_periodogram_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
            plt.close()

            freqs.append(max_SNR_freq)
            SNRs.append(max_SNR)
            SNR_thresholds.append(SNR_threshold)
            freq_thresholds.append(freq_threshold)
        else:
            freqs.append(np.nan)
            SNRs.append(np.nan)
            SNR_thresholds.append(np.nan)
            freq_thresholds.append(np.nan)

    spike_data["freqs"+suffix] = freqs
    spike_data["SNR"+suffix] = SNRs
    spike_data["shuffleSNR"+suffix] = SNR_thresholds
    spike_data["shufflefreqs"+suffix] = freq_thresholds #TODO change these collumn names, they're misleading
    return spike_data

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

            lomb_classifier = get_lomb_classifier(lomb_SNR, lomb_freq, 0.03, 0.05, numeric=False)
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

def plot_field_centre_of_mass_on_track(spike_data, output_path, track_length, plot_trials=["beaconed", "non_beaconed", "probe"]):

    print('plotting field rastas...')
    save_path = output_path + '/Figures/field_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:

            x_max = max(np.array(spike_data.beaconed_trial_number.iloc[cluster_index]))
            if x_max>100:
                spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
            else:
                spikes_on_track = plt.figure(figsize=(4,(x_max/20)))

            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            cluster_firing_com_trial_numbers = np.array(spike_data["fields_com_trial_number"].iloc[cluster_index])
            cluster_firing_com_trial_types = np.array(spike_data["fields_com_trial_type"].iloc[cluster_index])

            if "beaconed" in plot_trials:
                ax.plot(cluster_firing_com[cluster_firing_com_trial_types == 0], cluster_firing_com_trial_numbers[cluster_firing_com_trial_types == 0], "s", color='Black', markersize=4)
            if "non_beaconed" in plot_trials:
                ax.plot(cluster_firing_com[cluster_firing_com_trial_types == 1], cluster_firing_com_trial_numbers[cluster_firing_com_trial_types == 1], "s", color='Red', markersize=4)
            if "probe" in plot_trials:
                ax.plot(cluster_firing_com[cluster_firing_com_trial_types == 2], cluster_firing_com_trial_numbers[cluster_firing_com_trial_types == 2], "s", color='Blue', markersize=4)

            #ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=3)
            plt.ylabel('Field COM on trials', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            Edmond.plot_utility2.style_track_plot(ax, track_length)
            Edmond.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

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

        ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]), trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])), 'o', color=trial_stop_color, markersize=4)

    plt.ylabel('Stops on trials', fontsize=20, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
    plt.xlim(0,track_length)
    tick_spacing = 50
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()


def plot_stop_histogram(processed_position_data, output_path, track_length=200):
    print('plotting stop histogram...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    bin_size = 5

    beaconed_trials = processed_position_data[processed_position_data["trial_type"] == 0]
    non_beaconed_trials = processed_position_data[processed_position_data["trial_type"] == 1]
    probe_trials = processed_position_data[processed_position_data["trial_type"] == 2]

    beaconed_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(beaconed_trials["stop_location_cm"])
    non_beaconed_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(non_beaconed_trials["stop_location_cm"])
    probe_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(probe_trials["stop_location_cm"])

    beaconed_stop_hist, bin_edges = np.histogram(beaconed_stops, bins=int(track_length/bin_size), range=(0, track_length))
    non_beaconed_stop_hist, bin_edges = np.histogram(non_beaconed_stops, bins=int(track_length/bin_size), range=(0, track_length))
    probe_stop_hist, bin_edges = np.histogram(probe_stops, bins=int(track_length/bin_size), range=(0, track_length))
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

    ax.plot(bin_centres, beaconed_stop_hist/len(beaconed_trials), '-', color='Black')
    if len(non_beaconed_trials)>0:
        ax.plot(bin_centres, non_beaconed_stop_hist/len(non_beaconed_trials), '-', color='Red')
    if len(probe_trials)>0:
        ax.plot(bin_centres, probe_stop_hist/len(probe_trials), '-', color='Blue')

    plt.ylabel('Stops/Trial', fontsize=20, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_track_plot(ax, track_length)
    tick_spacing = 50
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    x_max = max(beaconed_stop_hist/len(beaconed_trials))+0.1
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
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


def process_recordings(vr_recording_path_list, of_recording_path_list):

    for recording in vr_recording_path_list:
        print("processing ", recording)
        #recording = "/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36"
        #recording = "/mnt/datastore/Harry/cohort8_may2021/vr/M14_D31_2021-06-21_12-07-01"
        #recording = "/mnt/datastore/Harry/cohort8_may2021/vr/M14_D45_2021-07-09_12-15-03"
        #recording = "/mnt/datastore/Harry/cohort8_may2021/vr/M14_D26_2021-06-14_12-22-50"
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            raw_position_data, position_data = syncronise_position_data(recording, get_track_length(recording))

            position_data = add_time_elapsed_collumn(position_data)
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")

            processed_position_data = add_avg_RZ_speed(processed_position_data, track_length=get_track_length(recording))
            processed_position_data = add_avg_track_speed(processed_position_data, track_length=get_track_length(recording))
            processed_position_data = add_RZ_bias(processed_position_data)
            processed_position_data, _ = add_hit_miss_try(processed_position_data, track_length=get_track_length(recording))
            processed_position_data = add_hit_miss_try2(processed_position_data, track_length=get_track_length(recording))
            #PI_hits_processed_position_data = extract_PI_trials(processed_position_data, hmt="hit")
            #PI_misses_processed_position_data = extract_PI_trials(processed_position_data, hmt="miss")
            #PI_tries_position_data = extract_PI_trials(processed_position_data, hmt="try")
            #b_processed_position_data = extract_beaconed_hit_trials(processed_position_data)

            #spike_data = process_vr_grid(spike_data, position_data, track_length=get_track_length(recording))
            #plot_speed_histogram(processed_position_data[processed_position_data["hit_miss_try"] == "hit"], output_path, track_length=get_track_length(recording), suffix="hit")
            #plot_speed_histogram(processed_position_data[processed_position_data["hit_miss_try"] == "miss"], output_path, track_length=get_track_length(recording), suffix="miss")
            #plot_speed_histogram(processed_position_data[processed_position_data["hit_miss_try"] == "try"], output_path, track_length=get_track_length(recording), suffix="try")

            # MOVING LOMB PERIODOGRAMS
            #spike_data, shuffle_data = plot_moving_lomb_scargle_periodogram_parallel(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length=get_track_length(recording))
            #spike_data, shuffle_data = plot_moving_lomb_scargle_periodogram(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length=get_track_length(recording))
            spike_data = analyse_lomb_powers(spike_data, processed_position_data)
            shuffle_data = analyse_lomb_powers(shuffle_data, processed_position_data)

            # SPATIAL AUTO CORRELOGRAMS
            #TODO make the spatial autocorrelograms a function of the firing rate and not the spike count
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data, output_path, track_length=get_track_length(recording), suffix="")
            #spike_data = plot_spatial_autocorrelogram_fr(spike_data, processed_position_data, output_path, track_length=get_track_length(recording), suffix="")

            #spike_data = calculate_allocentric_correlation(spike_data, position_data, output_path, track_length=get_track_length(recording))
            #spike_data = calculate_egocentric_correlation(spike_data, position_data, output_path, track_length=get_track_length(recording))
            #plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))
            #plot_firing_rate_maps_per_trial(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=get_track_length(recording))
            #plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=get_track_length(recording),
            #                     plot_trials=["beaconed", "non_beaconed", "probe"])

            #plot_stops_on_track(processed_position_data, output_path, track_length=get_track_length(recording))
            #plot_stop_histogram(processed_position_data, output_path, track_length=get_track_length(recording))
            #plot_speed_histogram(processed_position_data, output_path, track_length=get_track_length(recording), suffix="")
            #plot_speed_per_trial(processed_position_data, output_path, track_length=get_track_length(recording))

            #plot_field_com_histogram_radial(spike_data=spike_data, output_path=output_path)
            #plot_field_centre_of_mass_on_track(spike_data=spike_data, output_path=output_path, track_length=get_track_length(recording), plot_trials=["beaconed", "non_beaconed", "probe"])
            #plot_field_centre_of_mass_on_track(spike_data=spike_data, output_path=output_path, track_length=get_track_length(recording), plot_trials=["beaconed"])
            #plot_field_centre_of_mass_on_track(spike_data=spike_data, output_path=output_path, track_length=get_track_length(recording), plot_trials=["non_beaconed"])
            #plot_field_centre_of_mass_on_track(spike_data=spike_data, output_path=output_path, track_length=get_track_length(recording), plot_trials=["probe"])

            #if found_paired_recording:
            #    of_spatial_firing = pd.read_pickle(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            #    plot_field_com_ring_attractor_radial(spike_data=spike_data, of_spike_data=of_spatial_firing, output_path=output_path, track_length=get_track_length(recording))
            spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            shuffle_data.to_pickle(recording+"/MountainSort/DataFrames/lomb_shuffle_powers.pkl")

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
    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M14_D26_2021-06-14_12-22-50', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D27_2021-06-15_12-21-58', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D28_2021-06-16_12-26-51', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D29_2021-06-17_12-30-32', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D31_2021-06-21_12-07-01', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D33_2021-06-23_12-22-49', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D34_2021-06-24_12-48-57', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D35_2021-06-25_12-41-16', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D37_2021-06-29_12-33-24', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D39_2021-07-01_12-28-46', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D42_2021-07-06_12-38-31', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D5_2021-05-14_11-31-59', '/mnt/datastore/Harry/cohort8_may2021/vr/M15_D6_2021-05-17_12-47-59']
    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36']
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()]
    process_recordings(vr_path_list, of_path_list)

    print("look now")

if __name__ == '__main__':
    main()
