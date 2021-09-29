import numpy as np
import pandas as pd
import PostSorting.parameters
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from scipy import stats
from scipy import signal
from astropy.convolution import convolve, Gaussian1DKernel
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
from astroML.time_series import lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap
warnings.filterwarnings('ignore')
from scipy.stats.stats import pearsonr

def add_avg_trial_speed(processed_position_data):
    avg_trial_speeds = []
    for trial_number in np.unique(processed_position_data["trial_number"]):
        trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number]
        speeds = np.asarray(trial_processed_position_data['speeds_binned_in_time'])[0]
        avg_speed = np.nanmean(speeds)
        avg_trial_speeds.append(avg_speed)
    processed_position_data["avg_trial_speed"] = avg_trial_speeds
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

    processed_position_data["hit_miss_try"] = hit_miss_try
    return processed_position_data

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
            EdmondHC.plot_utility2.style_vr_plot(ax, x_max=max(autocorrelogram[1:]))
            plt.locator_params(axis = 'x', nbins  = 8)
            tick_spacing = 50
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_autocorrelogram_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
            plt.close()

    return spike_data


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]


def plot_lomb_scargle_periodogram(spike_data, processed_position_data, position_data, output_path, track_length, suffix="", GaussianKernelSTD_ms=5, fr_integration_window=2):
    print('plotting lomb_scargle periodogram...')
    save_path = output_path + '/Figures/lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # get trial numbers to use from processed_position_data
    trial_number_to_use = np.unique(processed_position_data["trial_number"])
    trial_numbers = np.array(position_data["trial_number"])

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        recording_length_sampling_points = np.array(cluster_spike_data['recording_length_sampling_points'])[0]

        if len(firing_times_cluster)>1:
            # reconstruct instantaneous firing rates
            firing_times_cluster_ms = firing_times_cluster//(settings.sampling_rate/1000) # convert from samples to ms
            milliseconds_in_recording = recording_length_sampling_points//(settings.sampling_rate/1000)
            time_bins = np.arange(0,milliseconds_in_recording+1, 1)
            instantaneous_firing_rate, time_bin_edges = np.histogram(firing_times_cluster_ms, bins=time_bins)
            gauss_kernel = Gaussian1DKernel(stddev=GaussianKernelSTD_ms) # sigma = 5ms
            instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)
            instantaneous_firing_rate = moving_sum(instantaneous_firing_rate, window=fr_integration_window)/fr_integration_window
            instantaneous_firing_rate = instantaneous_firing_rate*1000 # into Hz
            if len(instantaneous_firing_rate) != len(position_data): # correct the length of instantaneous firing rate
                instantaneous_firing_rate = np.append(instantaneous_firing_rate, np.zeros(len(position_data)-len(instantaneous_firing_rate)))

            # add firing rate and elapsed_distance to the position_data dataframe
            position_data["instantaneous_firing_rate"] = instantaneous_firing_rate.tolist()
            lap_distance_covered = (np.array(position_data["trial_number"])*track_length)-track_length #total elapsed distance
            elapsed_distance = np.array(position_data["x_position_cm"])+lap_distance_covered
            position_data["elapsed_distance"] = elapsed_distance.tolist()

            # get and apply set_mask (we are only concerned with the trial numbers in processed_position_data)
            set_mask = np.isin(trial_numbers, trial_number_to_use)
            set_fr = instantaneous_firing_rate[set_mask]
            set_trial_numbers = trial_numbers[set_mask]
            set_elapsed_distance = elapsed_distance[set_mask]

            #set_fr = signal.resample(set_fr, int(len(set_fr)/10))
            #set_fr[set_fr < 0] = 0
            #set_elapsed_distance = signal.resample(set_elapsed_distance, int(len(set_elapsed_distance)/10))

            ls = LombScargle(set_elapsed_distance, set_fr)
            frequency = np.linspace(1/400, 1, 100000)
            power = ls.power(frequency)
            #frequency, power = ls.autopower(minimum_frequency=1/800, maximum_frequency=1/1, samples_per_peak=10)
            best_frequency = frequency[np.argmax(power)]
            best_period = 1/best_frequency
            phase = (set_elapsed_distance / best_period) % 1
            t_fit = np.linspace(0, 1)
            y_fit = ls.model(t_fit, best_frequency)

            distance = 1/frequency
            distance = np.flip(distance)
            power = np.flip(power)

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.scatter(phase, set_fr, color="black", marker="o", alpha =0.01)
            ax.plot(t_fit, y_fit, color="blue")
            plt.ylabel('Firing Rate', fontsize=20, labelpad = 10)
            plt.xlabel('Phase', fontsize=20, labelpad = 10)
            plt.xlim(0,max(phase))
            plt.ylim(0,100)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_phase_at_maxpeak_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
            plt.close()

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            far = ls.false_alarm_level(0.05, method="bootstrap")
            ax.axhline(y=far, xmin=0, xmax=1, linestyle="dashed") # change method to "bootstrap" when you have time
            #ax.axhline(y= ls.false_alarm_level(0.000001, method="baluev"), xmin=0, xmax=1, linestyle="dotted")
            ax.plot(distance, power)
            plt.ylabel('Power', fontsize=20, labelpad = 10)
            plt.xlabel('Distance (cm)', fontsize=20, labelpad = 10)
            plt.xlim(0,400)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_lomb_scargle_periodogram_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
            plt.close()

    return spike_data

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
            EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
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
            EdmondHC.plot_utility2.style_track_plot(ax, track_length)
            EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
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
                ax.plot(cluster_spike_data.iloc[0].nonbeaconed_position_cm, cluster_spike_data.iloc[0].nonbeaconed_trial_number, '|', color='Red', markersize=4)
            if "probe" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].probe_position_cm, cluster_spike_data.iloc[0].probe_trial_number, '|', color='Blue', markersize=4)

            plt.ylabel('Spikes on trials', fontsize=20, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            style_track_plot(ax, track_length)
            tick_spacing = 50
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
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

            plt.ylabel('Trial Number', fontsize=20, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
            plt.xlim(0, track_length)
            tick_spacing = 50
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            EdmondHC.plot_utility2.style_track_plot(ax, track_length)
            EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '.png', dpi=200)
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

            EdmondHC.plot_utility2.style_track_plot(ax, track_length)
            EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
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
        EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
        EdmondHC.plot_utility2.style_track_plot(ax, track_length)
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
    EdmondHC.plot_utility2.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
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

    beaconed_stops = EdmondHC.plot_utility2.pandas_collumn_to_numpy_array(beaconed_trials["stop_location_cm"])
    non_beaconed_stops = EdmondHC.plot_utility2.pandas_collumn_to_numpy_array(non_beaconed_trials["stop_location_cm"])
    probe_stops = EdmondHC.plot_utility2.pandas_collumn_to_numpy_array(probe_trials["stop_location_cm"])

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
    EdmondHC.plot_utility2.style_track_plot(ax, track_length)
    tick_spacing = 50
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    x_max = max(beaconed_stop_hist/len(beaconed_trials))+0.1
    EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
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

    trial_speeds = EdmondHC.plot_utility2.pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned"])
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
    EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
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
        EdmondHC.plot_utility2.style_track_plot(ax, track_length)
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

        EdmondHC.plot_utility2.style_vr_plot(ax, x_max)
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
    EdmondHC.plot_utility2.style_vr_plot(ax, x_max=1)
    plt.savefig(save_path + '/allo_vs_ego_grid_cells.png', dpi=200)
    plt.close()


def add_time_elapsed_collumn(position_data):
    time_seconds = np.array(position_data['time_seconds'].to_numpy())
    time_elapsed = np.diff(time_seconds)
    time_elapsed = np.append(time_elapsed[0], time_elapsed)
    position_data["time_in_bin_seconds"] = time_elapsed.tolist()
    return position_data

def process_recordings(vr_recording_path_list, of_recording_path_list):

    for recording in vr_recording_path_list:
        print("processing ", recording)
        recording = "/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36"
        #recording = "/mnt/datastore/Harry/cohort8_may2021/vr/M14_D31_2021-06-21_12-07-01"
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            position_data = add_time_elapsed_collumn(position_data)
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            processed_position_data = add_hit_miss_try(processed_position_data, track_length=get_track_length(recording))

            #spike_data = process_vr_grid(spike_data, position_data, track_length=get_track_length(recording))
            #plot_speed_histogram(processed_position_data[processed_position_data["hit_miss_try"] == "hit"], output_path, track_length=get_track_length(recording), suffix="hit")
            #plot_speed_histogram(processed_position_data[processed_position_data["hit_miss_try"] == "miss"], output_path, track_length=get_track_length(recording), suffix="miss")
            #plot_speed_histogram(processed_position_data[processed_position_data["hit_miss_try"] == "try"], output_path, track_length=get_track_length(recording), suffix="try")

            #spike_data = process_vr_field_stats(spike_data, processed_position_data)
            #spike_data = process_vr_field_distances(spike_data, track_length=get_track_length(recording))

            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data, output_path, track_length=get_track_length(recording), suffix="")
            spike_data = plot_lomb_scargle_periodogram(spike_data, processed_position_data, position_data, output_path, track_length=get_track_length(recording), suffix="", GaussianKernelSTD_ms=5, fr_integration_window=2)
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[processed_position_data["hit_miss_try"] == "hit"], output_path, track_length=get_track_length(recording), suffix="hit")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[processed_position_data["hit_miss_try"] == "try"], output_path, track_length=get_track_length(recording), suffix="try")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[processed_position_data["hit_miss_try"] == "miss"], output_path, track_length=get_track_length(recording), suffix="miss")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[(processed_position_data["hit_miss_try"] == "hit") & (processed_position_data["trial_type"] == 0)], output_path, track_length=get_track_length(recording), suffix="hit_beaconed")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[(processed_position_data["hit_miss_try"] == "try") & (processed_position_data["trial_type"] == 0)], output_path, track_length=get_track_length(recording), suffix="try_beaconed")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[(processed_position_data["hit_miss_try"] == "miss") & (processed_position_data["trial_type"] == 0)], output_path, track_length=get_track_length(recording), suffix="miss_beaconed")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[(processed_position_data["hit_miss_try"] == "hit") & (processed_position_data["trial_type"] == 1)], output_path, track_length=get_track_length(recording), suffix="hit_nonbeaconed")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[(processed_position_data["hit_miss_try"] == "try") & (processed_position_data["trial_type"] == 1)], output_path, track_length=get_track_length(recording), suffix="try_nonbeaconed")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[(processed_position_data["hit_miss_try"] == "miss") & (processed_position_data["trial_type"] == 1)], output_path, track_length=get_track_length(recording), suffix="miss_nonbeaconed")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[(processed_position_data["hit_miss_try"] == "hit") & (processed_position_data["trial_type"] == 2)], output_path, track_length=get_track_length(recording), suffix="hit_probe")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[(processed_position_data["hit_miss_try"] == "try") & (processed_position_data["trial_type"] == 2)], output_path, track_length=get_track_length(recording), suffix="try_probe")
            #spike_data = plot_spatial_autocorrelogram(spike_data, processed_position_data[(processed_position_data["hit_miss_try"] == "miss") & (processed_position_data["trial_type"] == 2)], output_path, track_length=get_track_length(recording), suffix="miss_probe")



            #spike_data = calculate_allocentric_correlation(spike_data, position_data, output_path, track_length=get_track_length(recording))
            #spike_data = calculate_egocentric_correlation(spike_data, position_data, output_path, track_length=get_track_length(recording))
            #plot_inter_field_distance_histogram(spike_data=spike_data, output_path=output_path, , track_length=get_track_length(recording))
            #plot_field_com_histogram(spike_data=spike_data, output_path=output_path, , track_length=get_track_length(recording))
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
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()]
    process_recordings(vr_path_list, of_path_list)

    print("look now`")


if __name__ == '__main__':
    main()
