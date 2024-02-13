import os
import traceback
import warnings
import sys
import settings

import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pandas as pd

from scipy import stats
from scipy import signal

from astropy.timeseries import LombScargle
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.nddata import block_reduce

import control_sorting_analysis
import PostSorting.post_process_sorted_data_vr
import PostSorting.parameters
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from PostSorting.vr_spatial_firing import bin_fr_in_space, bin_fr_in_time, add_position_x, add_trial_number, add_trial_type

from Edmond.eLife_Grid_anchoring_2024.Helpers.remake_position_data import syncronise_position_data
from Edmond.eLife_Grid_anchoring_2024.Helpers.array_manipulations import *
import Edmond.eLife_Grid_anchoring_2024.Helpers.plot_utility as plot_utility
import Edmond.eLife_Grid_anchoring_2024.analysis_settings as Settings

warnings.filterwarnings('ignore')
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

def add_avg_track_speed(processed_position_data, position_data, track_length):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30
    track_start = 30
    track_end = track_length-30

    avg_speed_on_tracks = []
    avg_speed_in_RZs = []
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number]
        trial_position_data = position_data[position_data["trial_number"] == trial_number]
        speeds_in_time = np.array(trial_position_data["speed_per_100ms"])
        pos_in_time = np.array(trial_position_data["x_position_cm"])
        in_rz_mask = (pos_in_time > reward_zone_start) & (pos_in_time <= reward_zone_end)
        speeds_in_time_outside_RZ = speeds_in_time[~in_rz_mask]
        speeds_in_time_inside_RZ = speeds_in_time[in_rz_mask]

        if len(speeds_in_time_outside_RZ)==0:
            avg_speed_on_track = np.nan
        else:
            avg_speed_on_track = np.nanmean(speeds_in_time_outside_RZ)

        if len(speeds_in_time_inside_RZ) == 0:
            avg_speed_in_RZ = np.nan
        else:
            avg_speed_in_RZ= np.nanmean(speeds_in_time_inside_RZ)

        avg_speed_on_tracks.append(avg_speed_on_track)
        avg_speed_in_RZs.append(avg_speed_in_RZ)

    processed_position_data["avg_speed_on_track"] = avg_speed_on_tracks
    processed_position_data["avg_speed_in_RZ"] = avg_speed_in_RZs
    return processed_position_data

def add_hit_miss_try(processed_position_data):
    # first get the avg speeds in the reward zone for all hit trials
    rewarded_processed_position_data = processed_position_data[processed_position_data["hit_blender"] == True]
    speeds_in_rz = np.array(rewarded_processed_position_data["avg_speed_in_RZ"])

    mean, sigma = np.nanmean(speeds_in_rz), np.nanstd(speeds_in_rz)
    interval = stats.norm.interval(0.95, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]

    hit_miss_try =[]
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_process_position_data = processed_position_data[(processed_position_data.trial_number == trial_number)]
        track_speed = trial_process_position_data["avg_speed_on_track"].iloc[0]
        speed_in_rz = trial_process_position_data["avg_speed_in_RZ"].iloc[0]

        if (trial_process_position_data["hit_blender"].iloc[0] == True) and (track_speed>Settings.track_speed_threshold):
            hit_miss_try.append("hit")
        elif (speed_in_rz >= lower) and (speed_in_rz <= upper) and (track_speed>Settings.track_speed_threshold):
            hit_miss_try.append("try")
        elif (speed_in_rz < lower) or (speed_in_rz > upper) and (track_speed>Settings.track_speed_threshold):
            hit_miss_try.append("miss")
        else:
            hit_miss_try.append("rejected")

    processed_position_data["hit_miss_try"] = hit_miss_try
    return processed_position_data, upper


def find_paired_recording(recording_path, of_recording_path_list):
    mouse=recording_path.split("/")[-1].split("_")[0]
    training_day=recording_path.split("/")[-1].split("_")[1]

    for paired_recording in of_recording_path_list:
        paired_mouse=paired_recording.split("/")[-1].split("_")[0]
        paired_training_day=paired_recording.split("/")[-1].split("_")[1]

        if (mouse == paired_mouse) and (training_day == paired_training_day):
            return paired_recording, True
    return None, False

def plot_spatial_autocorrelogram_fr(spike_data, save_path, track_length, suffix=""):

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_rates = np.array(cluster_spike_data['fr_binned_in_space_smoothed'].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times_vr"].iloc[0])

        if len(firing_times_cluster)>1:
            fr = firing_rates.flatten()
            fr[np.isnan(fr)] = 0; fr[np.isinf(fr)] = 0
            autocorr_window_size = track_length*10
            lags = np.arange(0, autocorr_window_size, 1) # were looking at 10 timesteps back and 10 forward
            autocorrelogram = []
            for i in range(len(lags)):
                fr_lagged = fr[i:]
                corr = stats.pearsonr(fr_lagged, fr[:len(fr_lagged)])[0]
                autocorrelogram.append(corr)
            autocorrelogram= np.array(autocorrelogram)
            fig = plt.figure(figsize=(5,2.5))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            for f in range(1,11):
                ax.axvline(x=track_length*f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
            ax.axhline(y=0, color="black", linewidth=2,linestyle="dashed")
            ax.plot(lags, autocorrelogram, color="black", linewidth=3)
            plt.ylabel('Spatial Autocorr', fontsize=25, labelpad = 10)
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
            fig.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/spatial_autocorrelogram_' + spike_data.session_id.iloc[cluster_index] + '_' + str(int(cluster_id)) + suffix + '.png', dpi=200)
            plt.close()

def calculate_moving_lomb_scargle_periodogram(spike_data, processed_position_data, track_length, shuffled_trials=False):
    print('calculating moving lomb_scargle periodogram...')

    if shuffled_trials:
        suffix="_shuffled_trials"
    else:
        suffix=""

    n_trials = len(processed_position_data)
    elapsed_distance_bins = np.arange(0, (track_length*n_trials)+1, 1)
    elapsed_distance = 0.5*(elapsed_distance_bins[1:]+elapsed_distance_bins[:-1])/track_length

    shuffled_rate_maps = []
    freqs = []
    SNRs = []
    avg_powers = []
    all_powers = []
    all_centre_trials=[]
    all_centre_distances=[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_rates = np.array(cluster_spike_data["fr_binned_in_space_smoothed"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            if shuffled_trials:
                np.random.shuffle(firing_rates)

            fr = firing_rates.flatten()

            # construct the lomb-scargle periodogram
            frequency = Settings.frequency
            sliding_window_size=track_length*Settings.window_length_in_laps
            powers = []
            centre_distances = []
            indices_to_test = np.arange(0, len(fr)-sliding_window_size, 1, dtype=np.int64)[::Settings.power_estimate_step]
            for m in indices_to_test:
                ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr[m:m+sliding_window_size])
                power = ls.power(frequency)
                powers.append(power.tolist())
                centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
            powers = np.array(powers)
            centre_trials = np.round(np.array(centre_distances)).astype(np.int64)
            centre_distances = np.array(centre_distances)
            avg_power = np.nanmean(powers, axis=0)
            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)

            freqs.append(max_SNR_freq)
            SNRs.append(max_SNR)
            avg_powers.append(avg_power)
            all_powers.append(powers)
            all_centre_trials.append(centre_trials)
            shuffled_rate_maps.append(firing_rates)
            all_centre_distances.append(centre_distances)
        else:
            freqs.append(np.nan)
            SNRs.append(np.nan)
            avg_powers.append(np.nan)
            all_powers.append(np.nan)
            all_centre_trials.append(np.nan)
            shuffled_rate_maps.append(np.nan)
            all_centre_distances.append(np.nan)

    spike_data["MOVING_LOMB_freqs"+suffix] = freqs
    spike_data["MOVING_LOMB_avg_power"+suffix] = avg_powers
    spike_data["MOVING_LOMB_SNR"+suffix] = SNRs
    spike_data["MOVING_LOMB_all_powers"+suffix] = all_powers
    spike_data["MOVING_LOMB_all_centre_trials"+suffix] = all_centre_trials
    spike_data["MOVING_LOMB_all_centre_distances"+suffix] = all_centre_distances
    if shuffled_trials:
        spike_data["rate_maps"+suffix] = shuffled_rate_maps
    return spike_data


def analyse_lomb_powers(spike_data):
    frequency = Settings.frequency

    SNRs = [];
    Freqs = []
    for index, spike_row in spike_data.iterrows():
        cluster_spike_data = spike_row.to_frame().T.reset_index(drop=True)
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            avg_powers = np.nanmean(powers, axis=0)
            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_powers)
            SNRs.append(max_SNR)
            Freqs.append(max_SNR_freq)

    spike_data["ML_SNRs"] = SNRs
    spike_data["ML_Freqs"] = Freqs
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
    elif lomb_classifier_str == "P":
        return 0.5
    elif lomb_classifier_str == "D":
        return 1.5
    elif lomb_classifier_str == "N":
        return 2.5
    else:
        return 3.5

def get_lomb_classifier(lomb_SNR, lomb_freq, lomb_SNR_thres, lomb_freq_thres, numeric=False):
    lomb_distance_from_int = distance_from_integer(lomb_freq)[0]

    if lomb_SNR>lomb_SNR_thres:
        if lomb_distance_from_int<lomb_freq_thres:
            lomb_classifier = "Position"
        else:
            lomb_classifier = "Distance"
    else:
        if np.isnan(lomb_distance_from_int):
            lomb_classifier = "Unclassified"
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
        lomb_SNR_threshold = row["power_threshold"]
        lomb_SNR = row["ML_SNRs"+suffix]
        lomb_freq = row["ML_Freqs"+suffix]
        lomb_classifier = get_lomb_classifier(lomb_SNR, lomb_freq, lomb_SNR_threshold, 0.05, numeric=False)
        lomb_classifiers.append(lomb_classifier)

    spatial_firing["Lomb_classifier_"+suffix] = lomb_classifiers
    return spatial_firing

def distance_from_integer(frequencies):
    distance_from_zero = np.asarray(frequencies)%1
    distance_from_one = 1-(np.asarray(frequencies)%1)
    tmp = np.vstack((distance_from_zero, distance_from_one))
    return np.min(tmp, axis=0)

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
        if "firing_times_vr" in list(spike_data):
            firing_times_cluster = spike_data["firing_times_vr"].iloc[cluster_index]
        else:
            firing_times_cluster = spike_data["firing_times"].iloc[cluster_index]
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
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def plot_avg_spatial_periodograms_with_rolling_classifications(spike_data, processed_position_data, save_path, track_length, plot_for_all_trials=True, plot_avg = True):

    power_step = Settings.power_estimate_step
    step = Settings.frequency_step
    frequency = Settings.frequency

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times_vr"].iloc[0])#
        rolling_power_threshold =  cluster_spike_data["rolling_threshold"].iloc[0]
        power_threshold =  cluster_spike_data["power_threshold"].iloc[0]

        if len(firing_times_cluster)>1:
            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)

            rolling_lomb_classifier, rolling_lomb_classifier_numeric, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = \
                get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers, power_threshold=rolling_power_threshold, power_step=power_step, track_length=track_length)

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5/3, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            for f in range(1,6):
                ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)

            # add avg periodograms for position and distance coded trials
            for code, c in zip(["D", "P"], [Settings.egocentric_color, Settings.allocentric_color]):
                subset_trial_numbers = np.unique(rolling_points[rolling_lomb_classifier==code])

                # only plot if there if this is at least 15% of total trials
                if (len(subset_trial_numbers)/len(processed_position_data["trial_number"])>=0.15) or (plot_for_all_trials==True):
                    subset_mask = np.isin(centre_trials, subset_trial_numbers)
                    subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                    subset_powers = powers.copy()
                    subset_powers[subset_mask == False] = np.nan
                    avg_subset_powers = np.nanmean(subset_powers, axis=0)
                    sem_subset_powers = stats.sem(subset_powers, axis=0, nan_policy="omit")
                    ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color=c, alpha=0.3)
                    ax.plot(frequency, avg_subset_powers, color=c, linewidth=3)

            if plot_avg:
                subset_trial_numbers = np.asarray(processed_position_data["trial_number"])
                subset_mask = np.isin(centre_trials, subset_trial_numbers)
                subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                subset_powers = powers.copy()
                subset_powers[subset_mask == False] = np.nan
                avg_subset_powers = np.nanmean(subset_powers, axis=0)
                sem_subset_powers = stats.sem(subset_powers, axis=0, nan_policy="omit")
                ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color="black", alpha=0.3, zorder=-1)
                ax.plot(frequency, avg_subset_powers, color="black", linewidth=3, zorder=-1)

            ax.axhline(y=power_threshold, color="red", linewidth=3, linestyle="dashed")
            ax.set_ylabel('Periodic power', fontsize=30, labelpad = 10)
            #ax.set_xlabel("Spatial frequency", fontsize=25, labelpad = 10)
            ax.set_xlim([0.1,5.05])
            ax.set_xticks([1,2,3,4, 5])
            ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
            ax.set_ylim(bottom=0)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.xaxis.set_tick_params(labelsize=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/avg_spatial_periodograms_with_rolling_classifications_' + cluster_spike_data.session_id.iloc[0] + '' + str(int(cluster_id)) + '.png', dpi=300)
            plt.close()
    return

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


def make_field_array(firing_rate_map_by_trial, peaks_indices):
    field_array = np.zeros(len(firing_rate_map_by_trial))
    for i in range(len(peaks_indices)):
        field_array[peaks_indices[i][0]:peaks_indices[i][1]] = i+1
    return field_array.astype(np.int64)


def get_peak_indices(firing_rate_map, peaks_i):
    peak_indices =[]
    for j in range(len(peaks_i)):
        peak_index_tuple = find_neighbouring_minima(firing_rate_map, peaks_i[j])
        peak_indices.append(peak_index_tuple)
    return peak_indices

def plot_firing_rate_maps_short_with_rolling_classifications(spike_data, save_path, track_length, plot_avg=True, plot_codes=True):

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_times_cluster = spike_data.firing_times_vr.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            rolling_centre_trials = np.array(spike_data["rolling:rolling_centre_trials"].iloc[cluster_index])
            rolling_classifiers = np.array(spike_data["rolling:rolling_classifiers"].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = np.nan
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = np.nan

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5/3, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1) 
            locations = np.arange(0, len(cluster_firing_maps[0]))

            if plot_avg:
                ax.fill_between(locations, np.nanmean(cluster_firing_maps, axis=0)-stats.sem(cluster_firing_maps, axis=0, nan_policy="omit"), np.nanmean(cluster_firing_maps, axis=0)+stats.sem(cluster_firing_maps, axis=0, nan_policy="omit"), color="black", alpha=0.3)
                ax.plot(locations, np.nanmean(cluster_firing_maps, axis=0), color="black", linewidth=3)

            if plot_codes:
                for code, code_color in zip(["D", "P"], [Settings.egocentric_color, Settings.allocentric_color]):
                    trial_numbers = rolling_centre_trials[rolling_classifiers==code]
                    code_cluster_firing_maps = cluster_firing_maps[trial_numbers-1]
                    ax.fill_between(locations, np.nanmean(code_cluster_firing_maps, axis=0)-stats.sem(code_cluster_firing_maps, axis=0, nan_policy="omit"), np.nanmean(code_cluster_firing_maps, axis=0)+stats.sem(code_cluster_firing_maps, axis=0, nan_policy="omit"), color=code_color, alpha=0.3)
                    ax.plot(locations, np.nanmean(code_cluster_firing_maps, axis=0), color=code_color, linewidth=3)

            plt.ylabel('FR (Hz)', fontsize=25, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
            plt.xlim(0, track_length)
            ax.tick_params(axis='both', which='both', labelsize=20)
            ax.set_xlim([0, track_length])
            ax.set_yticks([0, np.round(ax.get_ylim()[1], 1)])
            ax.set_ylim(bottom=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/firing_rate_maps_short_with_rolling_classifications_' + spike_data.session_id.iloc[cluster_index] + '_' + str(int(cluster_id)) + '.png', dpi=300)
            plt.close()
    return


def get_spatial_information_score_for_trials(track_length, position_data, cluster_df, trial_ids):
    spikes_locations = np.array(cluster_df["x_position_cm"].iloc[0])
    spike_trial_numbers = np.array(cluster_df["trial_number"].iloc[0])

    spikes_locations = spikes_locations[np.isin(spike_trial_numbers, trial_ids)]
    spike_trial_numbers = spike_trial_numbers[np.isin(spike_trial_numbers, trial_ids)]

    number_of_spikes = len(spikes_locations)

    if number_of_spikes == 0:
        return np.nan

    position_data = position_data[position_data["trial_number"].isin(spike_trial_numbers)]

    position_heatmap = np.zeros(track_length)
    for x in np.arange(track_length):
        bin_occupancy = len(position_data[(position_data["x_position_cm"] > x) &
                                          (position_data["x_position_cm"] <= x+1)])
        position_heatmap[x] = bin_occupancy
    position_heatmap = position_heatmap*np.diff(position_data["time_seconds"])[-1] # convert to real time in seconds
    occupancy_probability_map = position_heatmap/np.sum(position_heatmap) # Pj

    vr_bin_size_cm = settings.vr_bin_size_cm
    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/vr_bin_size_cm)

    mean_firing_rate = number_of_spikes/np.sum(len(position_data)*np.diff(position_data["time_seconds"])[-1]) # λ
    spikes, _ = np.histogram(spikes_locations, bins=track_length, range=(0,track_length))
    rates = spikes/position_heatmap
    #rates = convolve(rates, gauss_kernel)
    mrate = mean_firing_rate
    Isec = np.sum(occupancy_probability_map * rates * np.log2((rates / mrate) + 0.0001))
    Ispike = Isec / mrate
    if np.isnan(Ispike):
        Ispike = 0

    if Ispike < 0:
        print("hello")

    return Isec

def plot_of_rate_map(spike_data, save_path):
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)]  # dataframe for that cluster
        firing_rate_map_original = cluster_df['firing_maps_of'].iloc[0]
        occupancy_map = cluster_df['occupancy_maps_of'].iloc[0]
        firing_rate_map_original[occupancy_map == 0] = np.nan
        firing_rate_map = np.rot90(firing_rate_map_original)

        firing_rate_map_fig = plt.figure()
        firing_rate_map_fig.set_size_inches(5, 5, forward=True)
        ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax = plot_utility.style_open_field_plot(ax)
        cmap = plt.get_cmap('jet')
        cmap.set_bad("white")
        rate_map_img = ax.imshow(firing_rate_map, cmap=cmap, interpolation='nearest')
        firing_rate_map_fig.colorbar(rate_map_img)
        plt.savefig(save_path + '/open_field_rate_map_' + spike_data.session_id.iloc[cluster_index] + '_' + str(int(cluster_id)) + '.png', dpi=300)
        plt.close()

def plot_of_autocorrelogram(spike_data, save_path):
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)]  # dataframe for that cluster
        rate_map_autocorr_fig = plt.figure()
        rate_map_autocorr_fig.set_size_inches(5, 5, forward=True)
        ax = rate_map_autocorr_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        rate_map_autocorr = cluster_df['rate_map_autocorrelogram_of'].iloc[0]
        if rate_map_autocorr.size:
            ax = plt.subplot(1, 1, 1)
            ax = plot_utility.style_open_field_plot(ax)
            autocorr_img = ax.imshow(rate_map_autocorr, cmap='jet', interpolation='nearest')
            rate_map_autocorr_fig.colorbar(autocorr_img)
            plt.tight_layout()
            plt.title('Autocorrelogram \n grid score: ' + str(round(cluster_df['grid_score'].iloc[0], 2)), fontsize=24)
            plt.savefig(save_path + '/open_field_rate_map_autocorrelogram_' + spike_data.session_id.iloc[cluster_index] + '_' + str(
                int(cluster_id)) + '.png', dpi=300)
        plt.close()


def plot_firing_rate_maps_short(spike_data, processed_position_data, save_path, track_length, by_trial_type=False, save_path_folder=True):

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        if "firing_times_vr" in list(spike_data):
            firing_times_cluster = spike_data["firing_times_vr"].iloc[cluster_index]
        else:
            firing_times_cluster = spike_data["firing_times"].iloc[cluster_index]

        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = np.nan
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = np.nan

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5/3, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            locations = np.arange(0, len(cluster_firing_maps[0]))

            if by_trial_type:
                for tt in [0,1,2]:
                    tt_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == tt]["trial_number"])
                    ax.fill_between(locations, np.nanmean(cluster_firing_maps[tt_trial_numbers-1], axis=0) - stats.sem(cluster_firing_maps[tt_trial_numbers-1], axis=0,nan_policy="omit"),
                                    np.nanmean(cluster_firing_maps[tt_trial_numbers-1], axis=0) + stats.sem(cluster_firing_maps[tt_trial_numbers-1], axis=0,nan_policy="omit"),
                                    color=get_trial_color(tt), alpha=0.2)
                    ax.plot(locations, np.nanmean(cluster_firing_maps[tt_trial_numbers-1], axis=0), color=get_trial_color(tt), linewidth=1)
            else:
                ax.fill_between(locations, np.nanmean(cluster_firing_maps, axis=0) - stats.sem(cluster_firing_maps, axis=0,nan_policy="omit"),
                                np.nanmean(cluster_firing_maps, axis=0) + stats.sem(cluster_firing_maps, axis=0,nan_policy="omit"), color="black",alpha=0.2)
                ax.plot(locations, np.nanmean(cluster_firing_maps, axis=0), color="black", linewidth=1)


            plt.ylabel('FR (Hz)', fontsize=25, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
            plt.xlim(0, track_length)
            ax.tick_params(axis='both', which='both', labelsize=20)
            ax.set_xlim([0, track_length])
            max_fr = max(np.nanmean(cluster_firing_maps, axis=0)+stats.sem(cluster_firing_maps, axis=0))
            max_fr = max_fr+(0.1*(max_fr))
            #ax.set_ylim([0, max_fr])
            ax.set_yticks([0, np.round(ax.get_ylim()[1], 1)])
            ax.set_ylim(bottom=0)
            #plot_utility.style_track_plot(ax, track_length, alpha=0.25)
            plot_utility.style_track_plot(ax, track_length, alpha=0.15)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            #cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            #cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
            #cbar.set_ticks([0,vmax])
            #cbar.set_ticklabels(["0", "Max"])
            #cbar.outline.set_visible(False)
            #cbar.ax.tick_params(labelsize=20)
            plt.savefig(save_path + '/avg_firing_rate_maps_short_' + spike_data.session_id.iloc[cluster_index] + '_' + str(int(cluster_id)) + '.png', dpi=300)
            plt.close()
    return

def plot_firing_rate_maps_per_trial(spike_data, processed_position_data, save_path, track_length, save_path_folder=False):
    if save_path_folder:
        save_path = save_path + '/Figures/rate_maps_by_trial'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        if "firing_times_vr" in list(spike_data):
            firing_times_cluster = spike_data["firing_times_vr"].iloc[cluster_index]
        else:
            firing_times_cluster = spike_data["firing_times"].iloc[cluster_index]

        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            percentile_99th_display = np.nanpercentile(cluster_firing_maps, 99);
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = plot_utility.get_vmin_vmax(cluster_firing_maps)

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            locations = np.arange(0, len(cluster_firing_maps[0]))
            ordered = np.arange(0, len(processed_position_data), 1)
            X, Y = np.meshgrid(locations, ordered)
            cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
            ax.pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
            plt.title(str(np.round(percentile_99th_display, decimals=1))+" Hz", fontsize=20)
            #plt.ylabel('Trial Number', fontsize=20, labelpad = 20)
            #plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
            plt.xlim(0, track_length)
            ax.tick_params(axis='both', which='both', labelsize=20)
            ax.set_xlim([0, track_length])
            ax.set_ylim([0, len(processed_position_data)-1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            tick_spacing = 100
            plt.locator_params(axis='y', nbins=3)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            spikes_on_track.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            #cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            #cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
            #cbar.set_ticks([0,np.max(cluster_firing_maps)])
            #cbar.set_ticklabels(["0", "Max"])
            #cbar.ax.tick_params(labelsize=20)
            plt.savefig(save_path + '/firing_rate_map_trials_' + spike_data.session_id.iloc[cluster_index] + '_' + str(int(cluster_id)) + '.png', dpi=300)
            plt.close()
    return

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
        plot_utility.style_vr_plot(ax, x_max)
        plot_utility.style_track_plot(ax, track_length)
        plt.tight_layout()
        plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_rate_map_Cluster_' + str(cluster_id) + '.png', dpi=200)
        plt.close()

def get_trial_color(trial_type):
    if trial_type == 0:
        return "tab:blue"
    elif trial_type == 1:
        return "tab:red"
    elif trial_type == 2:
        return "lightcoral"
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
    plot_utility.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    plot_utility.style_vr_plot(ax, x_max)
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

    beaconed_stops = pandas_collumn_to_numpy_array(beaconed_trials["stop_location_cm"])
    non_beaconed_stops = pandas_collumn_to_numpy_array(non_beaconed_trials["stop_location_cm"])
    #probe_stops = pandas_collumn_to_numpy_array(probe_trials["stop_location_cm"])

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
    plot_utility.style_track_plot(ax, track_length)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    plot_utility.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/stop_histogram' + '.png', dpi=200)
    plt.close()

def get_max_SNR(spatial_frequency, powers):
    max_SNR = powers[np.argmax(powers)]
    max_SNR_freq = spatial_frequency[np.argmax(powers)]
    return max_SNR, max_SNR_freq

def add_displayed_peak_firing(spike_data):
    peak_firing = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        if len(firing_times_cluster)>1:
            fr_binned_in_space = np.asarray(cluster_spike_data["fr_binned_in_space_smoothed"].iloc[0])
            fr_binned_in_space[np.isnan(fr_binned_in_space)] = 0
            fr_binned_in_space[np.isinf(fr_binned_in_space)] = 0
            peak_firing.append(np.nanpercentile(fr_binned_in_space.flatten(), 99))
        else:
            peak_firing.append(np.nan)
    spike_data["vr_peak_firing"] = peak_firing
    return spike_data

def get_rolling_lomb_classifier_for_centre_trial(centre_trials, powers, power_threshold, power_step, track_length, n_window_size=Settings.rolling_window_size_for_lomb_classifier, lomb_frequency_threshold=Settings.lomb_frequency_threshold):

    frequency = Settings.frequency

    trial_points = []
    peak_frequencies = []
    rolling_lomb_classifier = []
    rolling_lomb_classifier_numeric = []
    rolling_lomb_classifier_colors = []
    for i in range(len(centre_trials)):
        centre_trial = centre_trials[i]

        if n_window_size>1:
            if i<int(n_window_size/2):
                power_window = powers[:i+int(n_window_size/2), :]
            elif i+int(n_window_size/2)>len(centre_trials):
                power_window = powers[i-int(n_window_size/2):, :]
            else:
                power_window = powers[i-int(n_window_size/2):i+int(n_window_size/2), :]

            avg_power = np.nanmean(power_window, axis=0)
        else:
            avg_power = powers[i, :]

        max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)

        lomb_classifier = get_lomb_classifier(max_SNR, max_SNR_freq, power_threshold, lomb_frequency_threshold, numeric=False)
        peak_frequencies.append(max_SNR_freq)
        trial_points.append(centre_trial)
        if lomb_classifier == "Position":
            rolling_lomb_classifier.append("P")
            rolling_lomb_classifier_numeric.append(0.5)
            rolling_lomb_classifier_colors.append(Settings.allocentric_color)
        elif lomb_classifier == "Distance":
            rolling_lomb_classifier.append("D")
            rolling_lomb_classifier_numeric.append(1.5)
            rolling_lomb_classifier_colors.append(Settings.egocentric_color)
        elif lomb_classifier == "Null":
            rolling_lomb_classifier.append("N")
            rolling_lomb_classifier_numeric.append(2.5)
            rolling_lomb_classifier_colors.append(Settings.null_color)
        else:
            rolling_lomb_classifier.append("U")
            rolling_lomb_classifier_numeric.append(3.5)
            rolling_lomb_classifier_colors.append("black")
    return np.array(rolling_lomb_classifier), np.array(rolling_lomb_classifier_numeric), np.array(rolling_lomb_classifier_colors), np.array(peak_frequencies), np.array(trial_points)


def get_block_lengths_any_code(rolling_lomb_classifier):
    block_lengths = []
    current_block_length = 0
    current_code=rolling_lomb_classifier[0]

    for i in range(len(rolling_lomb_classifier)):
        if (rolling_lomb_classifier[i] == current_code):
            current_block_length+=1
        else:
            if (current_block_length != 0) and (current_code != "N"):
                block_lengths.append(current_block_length)
            current_block_length=0
            current_code=rolling_lomb_classifier[i]

    if (current_block_length != 0) and (current_code != "N"):
        block_lengths.append(current_block_length)

    block_lengths = np.array(block_lengths)/len(rolling_lomb_classifier) # normalise by length of session
    return block_lengths.tolist()



def get_block_lengths(rolling_lomb_classifier, modal_class_char):
    block_lengths = []
    current_block_length = 0
    for i in range(len(rolling_lomb_classifier)):
        if rolling_lomb_classifier[i] == modal_class_char:
            current_block_length+=1
        else:
            if current_block_length != 0:
                block_lengths.append(current_block_length)
            current_block_length=0
    block_lengths = np.array(block_lengths)/len(rolling_lomb_classifier) # normalise by length of session
    return block_lengths.tolist()

def get_block_ids(rolling_lomb_classifier):
    block_ids = np.zeros((len(rolling_lomb_classifier)))
    current_block_id=0
    current_block_classifier=rolling_lomb_classifier[0]
    for i in range(len(rolling_lomb_classifier)):
        if rolling_lomb_classifier[i] == current_block_classifier:
            block_ids[i] = current_block_id
        else:
            current_block_classifier = rolling_lomb_classifier[i]
            current_block_id+=1
    return block_ids

def shuffle_blocks(rolling_lomb_classifier):
    block_ids = get_block_ids(rolling_lomb_classifier)
    unique_block_ids = np.unique(block_ids)
    rolling_lomb_classifier_shuffled_by_blocks = np.empty((len(rolling_lomb_classifier)), dtype=np.str0)

    # shuffle unique ids
    np.random.shuffle(unique_block_ids)
    i=0
    for id in unique_block_ids:
        rolling_lomb_classifier_shuffled_by_blocks[i:i+len(block_ids[block_ids == id])] = rolling_lomb_classifier[block_ids == id]
        i+=len(block_ids[block_ids == id])
    return rolling_lomb_classifier_shuffled_by_blocks


def add_rolling_stats_shuffled_test(spike_data, processed_position_data, track_length):
    spike_data = calculate_moving_lomb_scargle_periodogram(spike_data, processed_position_data, track_length=track_length, shuffled_trials=True)

    power_step = Settings.power_estimate_step

    block_lengths_for_encoder=[]
    block_lengths_for_encoder_shuffled=[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])#
        rolling_power_threshold =  cluster_spike_data["rolling_threshold"].iloc[0]
        modal_class = cluster_spike_data['Lomb_classifier_'].iloc[0]

        if len(firing_times_cluster)>1:

            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            powers_shuffled =  np.array(cluster_spike_data["MOVING_LOMB_all_powers_shuffled_trials"].iloc[0])
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)

            powers[np.isnan(powers)] = 0; powers_shuffled[np.isnan(powers_shuffled)] = 0
            rolling_lomb_classifier, rolling_lomb_classifier_numeric, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers, power_threshold=rolling_power_threshold, power_step=power_step, track_length=track_length)
            rolling_lomb_classifier_shuffled, rolling_lomb_classifier_numeric_shuffled, rolling_lomb_classifier_colors_shuffled, rolling_frequencies_shuffled, rolling_points_shuffled = get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_shuffled, power_threshold=rolling_power_threshold, power_step=power_step, track_length=track_length)
            block_lengths = get_block_lengths_any_code(rolling_lomb_classifier)
            block_lengths_shuffled=get_block_lengths_any_code(rolling_lomb_classifier_shuffled)
        else:
            block_lengths=[]
            block_lengths_shuffled=[]

        block_lengths_for_encoder.append(block_lengths)
        block_lengths_for_encoder_shuffled.append(block_lengths_shuffled)

    spike_data["rolling:block_lengths"] = block_lengths_for_encoder
    spike_data["rolling:block_lengths_shuffled"] = block_lengths_for_encoder_shuffled

    # delete unwanted rows relating to the shuffled of the trials
    del spike_data["MOVING_LOMB_freqs_shuffled_trials"]
    del spike_data["MOVING_LOMB_avg_power_shuffled_trials"]
    del spike_data["MOVING_LOMB_SNR_shuffled_trials"]
    del spike_data["MOVING_LOMB_all_powers_shuffled_trials"]
    del spike_data["MOVING_LOMB_all_centre_trials_shuffled_trials"]
    return spike_data

def compress_rolling_stats(rolling_centre_trials, rolling_classifiers):
    rolling_trials = np.unique(rolling_centre_trials)
    rolling_modes = []
    rolling_modes_numeric = []
    for tn in rolling_trials:
        rolling_class = rolling_classifiers[rolling_centre_trials == tn]
        mode = stats.mode(rolling_class, axis=None)[0][0]
        rolling_modes.append(mode)

        if mode == "P":
            rolling_modes_numeric.append(0.5)
        elif mode == "D":
            rolling_modes_numeric.append(1.5)
        elif mode == "N":
            rolling_modes_numeric.append(2.5)
        else:
            rolling_modes_numeric.append(3.5)

    rolling_classifiers = np.array(rolling_modes)
    rolling_modes_numeric=np.array(rolling_modes_numeric)
    rolling_centre_trials = rolling_trials
    return rolling_centre_trials, rolling_classifiers, rolling_modes_numeric




def add_coding_by_trial_number(spike_data, processed_position_data):
    cluster_codes = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>0:
            rolling_centre_trials = cluster_spike_data["rolling:rolling_centre_trials"].iloc[0]
            rolling_classifiers = cluster_spike_data["rolling:rolling_classifiers"].iloc[0]

            rolling_centre_trials, rolling_classifiers,_ = compress_rolling_stats(rolling_centre_trials, rolling_classifiers)

            rolling_classifier_by_trial_number=[]
            for index, row in processed_position_data.iterrows():
                trial_number = row["trial_number"]
                rolling_class = rolling_classifiers[rolling_centre_trials == trial_number]
                if len(rolling_class)==1:
                    rolling_class = rolling_class[0]
                else:
                    rolling_class = np.nan

                rolling_classifier_by_trial_number.append(rolling_class)
            cluster_codes.append(rolling_classifier_by_trial_number)
        else:
            cluster_codes.append(np.nan)

    spike_data["rolling:classifier_by_trial_number"] = cluster_codes
    return spike_data

def add_rolling_stats(spike_data, track_length):
    power_step = Settings.power_estimate_step

    rolling_centre_trials=[]
    rolling_peak_frequencies=[]
    rolling_classifiers=[]
    proportion_encoding_position=[]
    proportion_encoding_distance=[]
    proportion_encoding_null=[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])#
        rolling_power_threshold = cluster_spike_data["rolling_threshold"].iloc[0]

        if len(firing_times_cluster)>1:
            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)
            powers[np.isnan(powers)] = 0
            rolling_lomb_classifier, rolling_lomb_classifier_numeric, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(centre_trials, powers, rolling_power_threshold, power_step, track_length)

            proportion_encoding_P = len(rolling_lomb_classifier[rolling_lomb_classifier=="P"])/len(rolling_lomb_classifier)
            proportion_encoding_D = len(rolling_lomb_classifier[rolling_lomb_classifier=="D"])/len(rolling_lomb_classifier)
            proportion_encoding_N = len(rolling_lomb_classifier[rolling_lomb_classifier=="N"])/len(rolling_lomb_classifier)

        else:
            rolling_lomb_classifier = np.array([])
            rolling_frequencies = np.array([])
            rolling_points = np.array([])#
            proportion_encoding_P = np.nan
            proportion_encoding_D = np.nan
            proportion_encoding_N = np.nan

        rolling_centre_trials.append(rolling_points)
        rolling_peak_frequencies.append(rolling_frequencies)
        rolling_classifiers.append(rolling_lomb_classifier)
        proportion_encoding_position.append(proportion_encoding_P)
        proportion_encoding_distance.append(proportion_encoding_D)
        proportion_encoding_null.append(proportion_encoding_N)

    spike_data["rolling:proportion_encoding_position"] = proportion_encoding_position
    spike_data["rolling:proportion_encoding_distance"] = proportion_encoding_distance
    spike_data["rolling:proportion_encoding_null"] = proportion_encoding_null
    spike_data["rolling:rolling_peak_frequencies"] = rolling_peak_frequencies
    spike_data["rolling:rolling_centre_trials"] = rolling_centre_trials
    spike_data["rolling:rolling_classifiers"] = rolling_classifiers
    return spike_data

def add_stop_location_trial_numbers(processed_position_data):
    trial_numbers=[]
    for index, row in processed_position_data.iterrows():
        trial_number = row["trial_number"]
        trial_stops = row["stop_location_cm"]
        trial_numbers.append(np.repeat(trial_number, len(trial_stops)).tolist())
    processed_position_data["stop_trial_numbers"] = trial_numbers
    return processed_position_data

def curate_stops_spike_data(spike_data, track_length):
    # stops are calculated as being below the stop threshold per unit time bin,
    # this function removes successive stops

    stop_locations_clusters = []
    stop_trials_clusters = []
    for index, row in spike_data.iterrows():
        row = row.to_frame().T.reset_index(drop=True)
        stop_locations=np.array(row["stop_locations"].iloc[0])
        stop_trials=np.array(row["stop_trial_numbers"].iloc[0])
        stop_locations_elapsed=(track_length*(stop_trials-1))+stop_locations

        curated_stop_locations=[]
        curated_stop_trials=[]
        for i, stop_loc in enumerate(stop_locations_elapsed):
            if (i==0): # take first stop always
                add_stop=True
            elif ((stop_locations_elapsed[i]-stop_locations_elapsed[i-1]) > 1): # only include stop if the last stop was at least 1cm away
                add_stop=True
            else:
                add_stop=False

            if add_stop:
                curated_stop_locations.append(stop_locations_elapsed[i])
                curated_stop_trials.append(stop_trials[i])

        # revert back to track positions
        curated_stop_locations = (np.array(curated_stop_locations)%track_length).tolist()

        stop_locations_clusters.append(curated_stop_locations)
        stop_trials_clusters.append(curated_stop_trials)

    spike_data["stop_locations"] = stop_locations_clusters
    spike_data["stop_trial_numbers"] = stop_trials_clusters
    return spike_data

def get_stop_histogram(cells_df, tt, coding_scheme=None, shuffle=False, track_length=None, use_first_stops=False, drop_bb_stops=False, trial_classification_column_name=None):
    if shuffle:
        iterations = 10
    else:
        iterations = 1
    gauss_kernel = Gaussian1DKernel(2)

    stop_histograms=[]
    stop_histogram_sems=[]
    number_of_trials_cells=[]
    number_of_stops_cells=[]
    stop_variability_cells=[]
    for index, cluster_df in cells_df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        if track_length is None:
            track_length = cluster_df["track_length"].iloc[0]

        stops_location_cm = np.array(cluster_df["stop_locations"].iloc[0])
        stop_trial_numbers = np.array(cluster_df["stop_trial_numbers"].iloc[0])

        if drop_bb_stops:
            stop_trial_numbers = stop_trial_numbers[(stops_location_cm >= 30) & (stops_location_cm <= 170)]
            stops_location_cm = stops_location_cm[(stops_location_cm >= 30) & (stops_location_cm <= 170)]

        if use_first_stops:
            stops_location_cm = stops_location_cm[np.unique(stop_trial_numbers, return_index=True)[1]]
            stop_trial_numbers = stop_trial_numbers[np.unique(stop_trial_numbers, return_index=True)[1]]

        hit_miss_try = np.array(cluster_df["behaviour_hit_try_miss"].iloc[0])
        trial_numbers = np.array(cluster_df["behaviour_trial_numbers"].iloc[0])
        trial_types = np.array(cluster_df["behaviour_trial_types"].iloc[0])

        # mask out only the trial numbers based on the trial type
        # and the coding scheme if that argument is given
        trial_type_mask = np.isin(trial_types, tt)
        hit_miss_try_mask = hit_miss_try!="rejected"
        if coding_scheme is not None:
            rolling_classifiers = np.array(cluster_df[trial_classification_column_name].iloc[0])
            classifier_mask = np.isin(rolling_classifiers, coding_scheme)
            tt_trial_numbers = trial_numbers[trial_type_mask & classifier_mask & hit_miss_try_mask]
        else:
            tt_trial_numbers = trial_numbers[trial_type_mask]

        number_of_bins = track_length
        number_of_trials = len(tt_trial_numbers)
        all_stop_locations = stops_location_cm[np.isin(stop_trial_numbers, tt_trial_numbers)]
        stop_variability = np.nanstd(all_stop_locations)

        stop_counts = np.zeros((iterations, number_of_trials, number_of_bins)); stop_counts[:,:,:] = np.nan

        for j in np.arange(iterations):
            if shuffle:
                stops_location_cm = np.random.uniform(low=0, high=track_length, size=len(stops_location_cm))

            for i, tn in enumerate(tt_trial_numbers):
                stop_locations_on_trial = stops_location_cm[stop_trial_numbers == tn]
                stop_in_trial_bins, bin_edges = np.histogram(stop_locations_on_trial, bins=number_of_bins, range=[0,track_length])
                stop_counts[j,i,:] = stop_in_trial_bins

        number_of_stops = np.sum(stop_counts)
        stop_counts = np.nanmean(stop_counts, axis=0)
        average_stops = np.nanmean(stop_counts, axis=0)
        average_stops_se = stats.sem(stop_counts, axis=0, nan_policy="omit")

        # only smooth histograms with trials
        if np.sum(np.isnan(average_stops))>0:
            average_stops = average_stops
            average_stops_se = average_stops_se
        else:
            average_stops = convolve(average_stops, gauss_kernel)
            average_stops_se = convolve(average_stops_se, gauss_kernel)

        stop_histograms.append(average_stops)
        stop_histogram_sems.append(average_stops_se)
        number_of_trials_cells.append(number_of_trials)
        number_of_stops_cells.append(number_of_stops)
        stop_variability_cells.append(stop_variability)

        bin_centres = np.arange(0.5, track_length+0.5, track_length/number_of_bins)

    return stop_histograms, stop_histogram_sems, bin_centres, number_of_trials_cells, number_of_stops_cells, stop_variability_cells

def add_stops(spike_data, processed_position_data, track_length):
    processed_position_data = add_stop_location_trial_numbers(processed_position_data)
    stop_locations = pandas_collumn_to_numpy_array(processed_position_data["stop_location_cm"])
    stop_trial_numbers = pandas_collumn_to_numpy_array(processed_position_data["stop_trial_numbers"])

    cluster_stop_locations=[]
    cluster_stop_trial_number=[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_stop_locations.append(stop_locations.tolist())
        cluster_stop_trial_number.append(stop_trial_numbers.tolist())
    spike_data["stop_locations"] = cluster_stop_locations
    spike_data["stop_trial_numbers"] = cluster_stop_trial_number

    spike_data = curate_stops_spike_data(spike_data, track_length)
    return spike_data


def add_trial_info(spike_data, processed_position_data):
    trial_types = np.array(processed_position_data["trial_type"])
    trial_numbers = np.array(processed_position_data["trial_number"])
    hit_try_miss = np.array(processed_position_data["hit_miss_try"])
    cluster_trial_numbers=[]
    cluster_hit_try_miss=[]
    cluster_trial_types=[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_trial_numbers.append(trial_numbers.tolist())
        cluster_hit_try_miss.append(hit_try_miss.tolist())
        cluster_trial_types.append(trial_types.tolist())
    spike_data["behaviour_trial_numbers"] = cluster_trial_numbers
    spike_data["behaviour_hit_try_miss"] = cluster_hit_try_miss
    spike_data["behaviour_trial_types"] = cluster_trial_types
    return spike_data


def spatial_info(mrate, occupancy_probability_map, rates):

    '''
    Calculates the spatial information score in bits per spike as in Skaggs et al.,
    1996, 1993).

    To estimate the spatial information contained in the
    firing rate of each cell we used Ispike and Isec – the standard
    approaches used for selecting place cells (Skaggs et al.,
    1996, 1993). We computed the Isec metric from the average firing rate (over trials) in
    the space bins using the following definition:

    Isec = sum(Pj*λj*log2(λj/λ))

    where λj is the mean firing rate in the j-th space bin and Pj
    the occupancy ratio of the bin (in other words, the probability of finding
    the animal in that bin), while λ is the overall
    mean firing rate of the cell. The Ispike metric is a normalization of Isec,
    defined as:

    Ispike = Isec / λ

    This normalization yields values in bits per spike,
    while Isec is in bits per second.
    '''
    Isec = np.sum(occupancy_probability_map * rates * np.log2((rates / mrate) + 0.0001))
    Ispike = Isec / mrate
    return Isec, Ispike

def calculate_spatial_information(spatial_firing, position_data, track_length):
    position_heatmap = np.zeros(track_length)
    for x in np.arange(track_length):
        bin_occupancy = len(position_data[(position_data["x_position_cm"] > x) &
                                                (position_data["x_position_cm"] <= x+1)])
        position_heatmap[x] = bin_occupancy
    position_heatmap = position_heatmap*np.diff(position_data["time_seconds"])[-1] # convert to real time in seconds
    occupancy_probability_map = position_heatmap/np.sum(position_heatmap) # Pj

    vr_bin_size_cm = settings.vr_bin_size_cm
    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/vr_bin_size_cm)

    spatial_information_scores_Ispike = []
    spatial_information_scores_Isec = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

        mean_firing_rate = cluster_df.iloc[0]["number_of_spikes"]/np.sum(len(position_data)*np.diff(position_data["time_seconds"])[-1]) # λ
        spikes, _ = np.histogram(np.array(cluster_df['x_position_cm'].iloc[0]), bins=track_length, range=(0,track_length))
        rates = spikes/position_heatmap

        Isec, Ispike = spatial_info(mean_firing_rate, occupancy_probability_map, rates)

        spatial_information_scores_Ispike.append(Ispike)
        spatial_information_scores_Isec.append(Isec)

    spatial_firing["spatial_information_score_Isec"] = spatial_information_scores_Isec
    spatial_firing["spatial_information_score_Ispike"] = spatial_information_scores_Ispike
    return spatial_firing

def get_session_weights(cells_df, tt, coding_scheme, rolling_classsifier_collumn_name="rolling:classifier_by_trial_number",session_id_column="session_id"):
    if session_id_column not in list(cells_df):
        session_id_column = "session_id_vr"

    weights = []
    for index, cluster_df in cells_df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        session_id = cluster_df[session_id_column].iloc[0]
        n_cells_for_session = len(cells_df[cells_df[session_id_column] == session_id])
        cells_by_session_weight = 1/n_cells_for_session
        hit_miss_try = np.array(cluster_df["behaviour_hit_try_miss"].iloc[0])
        trial_numbers = np.array(cluster_df["behaviour_trial_numbers"].iloc[0])
        trial_types = np.array(cluster_df["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cluster_df[rolling_classsifier_collumn_name].iloc[0])
        valid_trials_mask = (trial_types == tt) & (rolling_classifiers != "nan") & (hit_miss_try != "rejected")
        rolling_classifiers = rolling_classifiers[valid_trials_mask]
        if len(rolling_classifiers)>0:
            weight = len(rolling_classifiers[rolling_classifiers==coding_scheme])/len(rolling_classifiers)
            weight = weight*cells_by_session_weight
            weights.append(weight)
        else:
            weights.append(0)
    return np.array(weights)


def add_open_field_firing_rate(spike_data, paired_spike_data):
    open_field_firing_rates = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        if ("classifier" in list(paired_spike_data)) and (cluster_id in list(paired_spike_data["cluster_id"])):
            paired_cluster_df = paired_spike_data[(paired_spike_data.cluster_id == cluster_id)] # dataframe for that cluster
            open_field_firing_rates.append(paired_cluster_df["mean_firing_rate"].iloc[0])
        else:
            open_field_firing_rates.append(np.nan)
    spike_data["mean_firing_rate_of"] = open_field_firing_rates
    return spike_data

def add_open_field_classifier(spike_data, paired_spike_data):
    classifications = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        if ("classifier" in list(paired_spike_data)) and (cluster_id in list(paired_spike_data["cluster_id"])):
            paired_cluster_df = paired_spike_data[(paired_spike_data.cluster_id == cluster_id)] # dataframe for that cluster
            classifications.append(paired_cluster_df["classifier"].iloc[0])
        else:
            classifications.append("")
    spike_data["paired_cell_type_classification"] = classifications
    return spike_data


def add_spatial_information_during_position_and_distance_trials(spike_data, position_data, track_length):
    position_spatial_information_scores = []
    distance_spatial_information_scores = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        rolling_centre_trials = np.array(spike_data["rolling:rolling_centre_trials"].iloc[cluster_index])
        rolling_classifiers = np.array(spike_data["rolling:rolling_classifiers"].iloc[cluster_index])

        P_trial_numbers = rolling_centre_trials[rolling_classifiers=="P"]
        D_trial_numbers = rolling_centre_trials[rolling_classifiers=="D"]

        lowest_n_number = min([len(P_trial_numbers), len(D_trial_numbers)])
        # Subset trials using the lowest n number
        np.random.shuffle(P_trial_numbers)
        np.random.shuffle(D_trial_numbers)

        if lowest_n_number>0:
            position_spatial_information_scores.append(get_spatial_information_score_for_trials(track_length, position_data, cluster_df, P_trial_numbers[:lowest_n_number]))
            distance_spatial_information_scores.append(get_spatial_information_score_for_trials(track_length, position_data, cluster_df, D_trial_numbers[:lowest_n_number]))
        else:
            position_spatial_information_scores.append(np.nan)
            distance_spatial_information_scores.append(np.nan)

    spike_data["rolling:position_spatial_information_scores_Isec"] = position_spatial_information_scores
    spike_data["rolling:distance_spatial_information_scores_Isec"] = distance_spatial_information_scores
    return spike_data


def add_mean_firing_rate_during_position_and_distance_trials(spike_data, position_data, track_length):
    position_mean_firing_rates = []
    distance_mean_firing_rates = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        cluster_trial_numbers = np.asarray(cluster_df["trial_number"].iloc[0])
        cluster_firing = np.asarray(cluster_df["firing_times"].iloc[0])
        if len(cluster_firing)>0:
            rolling_centre_trials = np.array(spike_data["rolling:rolling_centre_trials"].iloc[cluster_index])
            rolling_classifiers = np.array(spike_data["rolling:rolling_classifiers"].iloc[cluster_index])

            P_trial_numbers = rolling_centre_trials[rolling_classifiers=="P"]
            D_trial_numbers = rolling_centre_trials[rolling_classifiers=="D"]

            P_firing = cluster_firing[np.isin(cluster_trial_numbers, P_trial_numbers)]
            D_firing = cluster_firing[np.isin(cluster_trial_numbers, D_trial_numbers)]

            P_position_data = position_data[np.isin(position_data["trial_number"], P_trial_numbers)]
            D_position_data = position_data[np.isin(position_data["trial_number"], D_trial_numbers)]
            P_mfr = len(P_firing) / np.sum(P_position_data["time_spent_in_bin"])
            D_mfr = len(D_firing) / np.sum(D_position_data["time_spent_in_bin"])

            position_mean_firing_rates.append(P_mfr)
            distance_mean_firing_rates.append(D_mfr)
        else:
            position_mean_firing_rates.append(np.nan)
            distance_mean_firing_rates.append(np.nan)

    spike_data["rolling:position_mean_firing_rate"] = position_mean_firing_rates
    spike_data["rolling:distance_mean_firing_rate"] = distance_mean_firing_rates
    return spike_data

def add_bin_time(position_data):
    time_spent_in_bin = np.diff(position_data["time_seconds"])[0]
    position_data["time_spent_in_bin"] = time_spent_in_bin
    return position_data

def add_speed_per_100ms(position_data, track_length):
    positions = np.array(position_data["x_position_cm"])
    trial_numbers = np.array(position_data["trial_number"])
    distance_travelled = positions+(track_length*(trial_numbers-1))

    change_in_distance_travelled = np.concatenate([np.zeros(1), np.diff(distance_travelled)], axis=0)

    speed_per_100ms = np.array(pd.Series(change_in_distance_travelled).rolling(100).sum())/(100/1000) # 0.1 seconds == 100ms
    position_data["speed_per_100ms"] = speed_per_100ms
    return position_data

def add_stopped_in_rz(position_data, track_length):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30
    track_start = 30
    track_end = track_length-30

    position_data["below_speed_threshold"] = position_data["speed_per_100ms"]<4.7
    position_data["stopped_in_rz"] = (position_data["below_speed_threshold"] == True) &\
                                     (position_data["x_position_cm"] <= reward_zone_end) & \
                                     (position_data["x_position_cm"] >= reward_zone_start)
    return position_data

def add_hit_according_to_blender(processed_position_data, position_data):
    hit_trial_numbers = np.unique(position_data[position_data["stopped_in_rz"] == True]["trial_number"])
    hit_array = np.zeros(len(processed_position_data), dtype=int)
    hit_array[hit_trial_numbers-1] = 1
    processed_position_data["hit_blender"] = hit_array

    return processed_position_data

def add_stops_according_to_blender(processed_position_data, position_data):
    stop_locations = []
    first_stop_locations = []
    for tn in processed_position_data["trial_number"]:
        trial_stop_locations = np.array(position_data[(position_data["below_speed_threshold"] == True) & (position_data["trial_number"] == tn)]['x_position_cm'])
        if len(trial_stop_locations)>0:
            trial_first_stop_location = trial_stop_locations[0]
        else:
            trial_first_stop_location = np.nan

        stop_locations.append(trial_stop_locations.tolist())
        first_stop_locations.append(trial_first_stop_location)

    processed_position_data["stop_location_cm"] = stop_locations
    processed_position_data["first_stop_location_cm"] = first_stop_locations
    return processed_position_data


def get_modal_color(modal_class):
    if modal_class == "Position":
        return Settings.allocentric_color
    elif modal_class == "Distance":
        return Settings.egocentric_color
    elif modal_class == "Null":
        return Settings.null_color


def get_numeric_from_trial_classifications(trial_classifications, track_length):
    trial_classifications = np.repeat(trial_classifications, track_length)
    numerics = []
    for i in range(len(trial_classifications)):
        numeric = get_numeric_lomb_classifer(trial_classifications[i])
        numerics.append(numeric)
    return np.array(numerics)


def plot_histogram_of_position_template_correlations(spike_data, save_path):
    correlations = pandas_collumn_to_numpy_array(spike_data["rolling:correlation_by_trial_number_t2tmethod"])
    fig, ax = plt.subplots(figsize=(6,6))
    #ax.hist(correlations[correlations > 0.3], range=(-1, 1), bins=100, color=Settings.allocentric_color)
    #ax.hist(correlations[correlations < 0.3], range=(-1, 1), bins=100, color=np.array([109/255, 217/255, 255/255]))
    for i in range(len(spike_data)):
        fig, ax = plt.subplots(figsize=(6, 6))
        id = spike_data["cluster_id"].iloc[i]
        classifier = spike_data["paired_cell_type_classification"].iloc[i]
        correlations = spike_data["rolling:correlation_by_trial_number_t2tmethod"].iloc[i]
        hist, bin_edges = np.histogram(correlations, range=(-1, 1), bins=20)
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        if classifier == "G":
            ax.plot(bin_centres, hist, color="orange")
        else:
            ax.plot(bin_centres, hist, color="blue")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlim(-1,1)
        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.2, right=0.87, top=0.92)
        plt.savefig(save_path + '/correlations_hist_against_position_template_cluster_'+str(id)+'.png', dpi=300)
        plt.close()
    return

def add_spatial_imformation_during_dominant_modes(spike_data, output_path, track_length, position_data,
                                                  n_window_size_for_rolling_window=Settings.rolling_window_size_for_lomb_classifier):
    spike_data["grid_cell"] = spike_data["paired_cell_type_classification"] == "G"
    spike_data = spike_data.sort_values(by=["grid_cell", "spatial_information_score"], ascending=False)

    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    if np.sum(spike_data["grid_cell"])<1:
        spike_data["spatial_information_during_P"] = np.nan
        spike_data["spatial_information_during_D"] = np.nan
        return spike_data

    rolling_classifiers_grid_cells = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster) > 1:
            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)
            powers[np.isnan(powers)] = 0

            rolling_centre_trials = cluster_spike_data["rolling:rolling_centre_trials"].iloc[0]
            rolling_classifiers = cluster_spike_data["rolling:rolling_classifiers"].iloc[0]

            rolling_centre_trials, rolling_classifiers, rolling_classifiers_numeric = compress_rolling_stats(
                centre_trials, rolling_classifiers)

            if cluster_spike_data["grid_cell"].iloc[0] == True:
                rolling_classifiers_grid_cells.append(rolling_classifiers_numeric.tolist())

    rolling_classifiers_grid_cells = np.array(rolling_classifiers_grid_cells)
    dominant_mode_grid_cells = stats.mode(rolling_classifiers_grid_cells, axis=0)[0][0]

    dominant_P_trials = np.unique(rolling_centre_trials[dominant_mode_grid_cells == 0.5])
    dominant_D_trials = np.unique(rolling_centre_trials[dominant_mode_grid_cells == 1.5])

    # take only lowest number of trials for calculation
    lowest_n_number = min([len(dominant_P_trials), len(dominant_D_trials)])
    # Subset trials using the lowest n number
    np.random.shuffle(dominant_P_trials)
    np.random.shuffle(dominant_D_trials)
    dominant_P_trials = dominant_P_trials[:lowest_n_number]
    dominant_D_trials = dominant_D_trials[:lowest_n_number]

    D_position_data = position_data[np.isin(position_data["trial_number"], dominant_D_trials)]
    P_position_data = position_data[np.isin(position_data["trial_number"], dominant_P_trials)]

    if (len(D_position_data) == 0) or (len(P_position_data) == 0):
        spike_data["spatial_information_during_P"] = np.nan
        spike_data["spatial_information_during_D"] = np.nan
        return spike_data

    P_position_heatmap = np.histogram(P_position_data["x_position_cm"], range=(0, track_length), bins=track_length)[
                             0] * np.diff(position_data["time_seconds"])[-1]  # convert to real time in seconds
    D_position_heatmap = np.histogram(D_position_data["x_position_cm"], range=(0, track_length), bins=track_length)[
                             0] * np.diff(position_data["time_seconds"])[-1]  # convert to real time in seconds

    P_occupancy_probability_map = P_position_heatmap / np.sum(P_position_heatmap)  #
    D_occupancy_probability_map = D_position_heatmap / np.sum(D_position_heatmap)  #

    spatial_info_P = []
    spatial_info_D = []
    fr_info_P = []
    fr_info_D = []
    for index, row in spike_data.iterrows():
        cluster_row = row.to_frame().T.reset_index(drop=True)
        spike_locations = np.array(cluster_row['x_position_cm'].iloc[0])
        spike_trial_numbers = np.array(cluster_row['trial_number'].iloc[0])

        P_spike_locations = spike_locations[np.isin(spike_trial_numbers, dominant_P_trials)]
        D_spike_locations = spike_locations[np.isin(spike_trial_numbers, dominant_D_trials)]

        P_mean_firing_rate = len(P_spike_locations) / np.sum(
            len(P_position_data) * np.diff(P_position_data["time_seconds"])[-1])
        D_mean_firing_rate = len(D_spike_locations) / np.sum(
            len(D_position_data) * np.diff(D_position_data["time_seconds"])[-1])

        P_spikes, _ = np.histogram(P_spike_locations, bins=track_length, range=(0, track_length))
        D_spikes, _ = np.histogram(D_spike_locations, bins=track_length, range=(0, track_length))

        P_rates = P_spikes / P_position_heatmap
        D_rates = D_spikes / D_position_heatmap

        P_Isec, P_Ispike = spatial_info(P_mean_firing_rate, P_occupancy_probability_map, P_rates)
        D_Isec, D_Ispike = spatial_info(D_mean_firing_rate, D_occupancy_probability_map, D_rates)

        spatial_info_P.append(P_Isec)
        spatial_info_D.append(D_Isec)
        fr_info_P.append(P_mean_firing_rate)
        fr_info_D.append(D_mean_firing_rate)

    spike_data["spatial_information_during_P"] = spatial_info_P
    spike_data["spatial_information_during_D"] = spatial_info_D
    spike_data["firing_rate_during_P"] = fr_info_P
    spike_data["firing_rate_during_D"] = fr_info_D
    return spike_data

def add_stops_to_spatial_firing(spike_data, processed_position_data, track_length):
    trial_numbers = []
    stop_locations = []
    for tn in processed_position_data["trial_number"]:
        trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == tn]
        trial_stops = np.array(trial_processed_position_data["stop_location_cm"].iloc[0])
        trial_numbers_repeated = np.repeat(tn, len(trial_stops))

        stop_locations.extend(trial_stops.tolist())
        trial_numbers.extend(trial_numbers_repeated.tolist())

    cluster_trial_numbers = []
    cluster_stop_locations = []
    for index, row in spike_data.iterrows():
        cluster_trial_numbers.append(trial_numbers)
        cluster_stop_locations.append(stop_locations)

    spike_data["stop_locations"] = cluster_stop_locations
    spike_data["stop_trial_numbers"] = cluster_trial_numbers

    spike_data = curate_stops_spike_data(spike_data, track_length)
    return spike_data

def add_agreement_between_cell_and_grid_global(spike_data, column_name="rolling:classifier_by_trial_number"):
    grid_cells = spike_data[spike_data["paired_cell_type_classification"]== "G"]

    agreements = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        if "firing_times_vr" in list(spike_data):
            firing_times_cluster = np.array(cluster_spike_data["firing_times_vr"].iloc[0])
        else:
            firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if (len(firing_times_cluster) > 1) and (len(grid_cells)>0):
            cell_code = np.array(spike_data[column_name].iloc[cluster_index])
            agreement = np.sum(cell_code == np.array(spike_data["rolling:grid_code_global"].iloc[0])) / len(cell_code)
        else:
            agreement = np.nan
        agreements.append(agreement)

    spike_data["agreement_between_cell_and_grid_global"] = agreements
    return spike_data

def code2_numeric(code):
    numerics = []
    for i in range(len(code)):
        if code[i] == "P":
            numerics.append(0.5)
        elif code[i] == "D":
            numerics.append(1.5)
        elif code[i] == "N":
            numerics.append(2.5)
        else:
            numerics.append(3.5)
    return np.array(numerics)

def add_grid_code_global(spike_data):
    spike_data_with_spikes = spike_data[spike_data["number_of_spikes"] > 1]

    spike_data_with_spikes["grid_cell"] = spike_data_with_spikes["paired_cell_type_classification"] == "G"
    grid_cells = spike_data_with_spikes[spike_data_with_spikes["grid_cell"] == True]

    if len(grid_cells)==0:
        spike_data["rolling:grid_code_global"] = np.nan
        return spike_data

    all_grid_cells_codes = pandas_collumn_to_2d_numpy_array(grid_cells["rolling:classifier_by_trial_number"])
    dominant_code_grid_cells = stats.mode(all_grid_cells_codes, axis=0)[0][0]
    proportion_encoding_position = len(dominant_code_grid_cells[dominant_code_grid_cells=="P"])/len(dominant_code_grid_cells)
    proportion_encoding_distance = len(dominant_code_grid_cells[dominant_code_grid_cells =="D"])/len(dominant_code_grid_cells)

    grid_code_global_encoding_position = []
    grid_code_global_encoding_distance = []
    grid_code_global = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        grid_code_global.append(dominant_code_grid_cells)
        grid_code_global_encoding_position.append(proportion_encoding_position)
        grid_code_global_encoding_distance.append(proportion_encoding_distance)

    spike_data["rolling:grid_code_global"] = grid_code_global
    spike_data["rolling:grid_code_global_encoding_position"] = grid_code_global_encoding_position
    spike_data["rolling:grid_code_global_encoding_distance"] = grid_code_global_encoding_distance
    return spike_data

def add_field_locations(spike_data, track_length):
    cluster_field_locations = []
    cluster_field_trial_numbers = []
    cluster_field_sizes = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)]  # dataframe for that cluster
        firing_times_cluster = cluster_df.firing_times.iloc[0]
        if len(firing_times_cluster) > 1:
            cluster_firing_maps_unsmoothed = np.array(cluster_df['fr_binned_in_space'].iloc[0])

            firing_rate_map_by_trial_flattened = cluster_firing_maps_unsmoothed.flatten()
            gauss_kernel_extra = Gaussian1DKernel(stddev=Settings.rate_map_extra_smooth_gauss_kernel_std)
            firing_rate_map_by_trial_flattened_extra_smooth = convolve(firing_rate_map_by_trial_flattened, gauss_kernel_extra)

            # find peaks and trough indices
            peaks_i = \
            signal.find_peaks(firing_rate_map_by_trial_flattened_extra_smooth, distance=Settings.minimum_peak_distance)[0]
            peaks_indices = get_peak_indices(firing_rate_map_by_trial_flattened_extra_smooth, peaks_i)
            field_coms, field_trial_numbers, field_sizes = get_field_centre_of_mass(firing_rate_map_by_trial_flattened, peaks_indices, track_length)

            cluster_field_locations.append(field_coms)
            cluster_field_trial_numbers.append(field_trial_numbers)
            cluster_field_sizes.append(field_sizes)
        else:
            cluster_field_locations.append(np.array([]))
            cluster_field_trial_numbers.append(np.array([]))
            cluster_field_sizes.append(np.array([]))

    spike_data["field_locations"] = cluster_field_locations
    spike_data["field_trial_numbers"] = cluster_field_trial_numbers
    spike_data["field_sizes"] = cluster_field_sizes
    return spike_data

def get_field_centre_of_mass(firing_rate_map_by_trial_flattened, peaks_indices, track_length):
    firing_rate_map_by_trial_flattened[np.isnan(firing_rate_map_by_trial_flattened)] = 0

    distance_travelled = np.arange(0.5, len(firing_rate_map_by_trial_flattened)+0.5, 1)
    field_coms = []
    field_trial_numbers = []
    field_sizes = []
    for i in range(len(peaks_indices)):
        field_left = peaks_indices[i][0]
        field_right = peaks_indices[i][1]
        field_firing_map = firing_rate_map_by_trial_flattened[field_left:field_right]
        field_distance_travelled = distance_travelled[field_left:field_right]
        field_com = (np.sum(field_firing_map*field_distance_travelled))/np.sum(field_firing_map)
        field_com_on_track = field_com%track_length
        field_trial_number = (field_com//track_length)+1

        field_coms.append(field_com_on_track)
        field_trial_numbers.append(field_trial_number)
        field_sizes.append(len(field_firing_map)*settings.vr_bin_size_cm)
    nan_mask = np.isnan(field_coms) & np.isnan(field_trial_numbers)

    field_coms = np.array(field_coms)[~nan_mask]
    field_trial_numbers = np.array(field_trial_numbers)[~nan_mask]
    field_trial_numbers = np.array(field_trial_numbers, dtype=np.int64)
    field_sizes = np.array(field_sizes)
    return field_coms, field_trial_numbers, field_sizes

def add_rolling_stats_by_trial_number_using_alternative_classification(spike_data):
    rolling_position_correlations_all_clusters = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_rate_map_smoothed = np.array(cluster_spike_data["fr_binned_in_space_smoothed"].iloc[0])
        rolling_classification_by_trial_number = np.array(cluster_spike_data["rolling:classifier_by_trial_number"].iloc[0], dtype=str) # for debugging purposes

        # extract template and TA template
        position_encoding_mask = rolling_classification_by_trial_number == "P"
        aperiodic_encoding_mask = rolling_classification_by_trial_number == "N"
        nan_encoding_mask = rolling_classification_by_trial_number == "nan"
        position_template = np.nanmean(firing_rate_map_smoothed[position_encoding_mask], axis=0)

        # calculate  trial to trial correlations and classify based on the 99th percentile
        t2t_position_correlations = []
        for i in range(len(firing_rate_map_smoothed)):
            if aperiodic_encoding_mask[i]: # if trial is already marked as aperiodic
                corr=np.nan
            elif ((np.sum(position_encoding_mask)/(len(firing_rate_map_smoothed)-np.sum(nan_encoding_mask))) >= 0.15): # check if theres enough position trials to take the template seriously
                position_nonnanmask = np.array(bool_to_int_array(~np.isnan(position_template))*bool_to_int_array(~np.isnan(firing_rate_map_smoothed[i])), dtype=np.bool8)
                if ((len(position_template[position_nonnanmask]) == len(firing_rate_map_smoothed[i][position_nonnanmask])) and (len(position_template[position_nonnanmask])>0)):
                    corr = stats.pearsonr(position_template[position_nonnanmask], firing_rate_map_smoothed[i][position_nonnanmask])[0]
                else:
                    corr = np.nan
            else:
                corr = np.nan
            t2t_position_correlations.append(corr)
        t2t_position_correlations=np.array(t2t_position_correlations)
        rolling_position_correlations_all_clusters.append(t2t_position_correlations)
    spike_data["rolling:position_correlation_by_trial_number_t2tmethod"] = rolling_position_correlations_all_clusters
    return spike_data

def process_recordings(vr_recording_path_list, of_recording_path_list):
    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            tags = control_sorting_analysis.get_tags_parameter_file(recording)
            sorter_name = control_sorting_analysis.check_for_tag_name(tags, "sorter_name")
            output_path = recording+'/'+sorter_name
            processed_position_data = pd.read_pickle(recording+"/"+sorter_name+"/DataFrames/processed_position_data.pkl")
            position_data = pd.read_pickle(recording+"/"+sorter_name+"/DataFrames/position_data.pkl")
            spike_data = pd.read_pickle(recording+"/"+sorter_name+"/DataFrames/spatial_firing.pkl")

            if len(spike_data) != 0:
                if paired_recording is not None:
                    paired_spike_data = pd.read_pickle(paired_recording+"/"+sorter_name+"/DataFrames/spatial_firing.pkl")
                    spike_data = add_open_field_classifier(spike_data, paired_spike_data)
                    spike_data = add_open_field_firing_rate(spike_data, paired_spike_data)
                    spike_data = add_spatial_imformation_during_dominant_modes(spike_data, output_path=output_path,
                                                                               track_length=get_track_length(recording),
                                                                               position_data=position_data)

                # remake the spike locations and firing rate maps
                raw_position_data, position_data = syncronise_position_data(recording, get_track_length(recording))
                position_data = add_speed_per_100ms(position_data, track_length=get_track_length(recording))
                position_data = add_stopped_in_rz(position_data, track_length=get_track_length(recording))
                position_data = add_bin_time(position_data)
                processed_position_data = add_hit_according_to_blender(processed_position_data, position_data)
                processed_position_data = add_avg_track_speed(processed_position_data, position_data, track_length=get_track_length(recording))
                processed_position_data = add_stops_according_to_blender(processed_position_data, position_data)
                processed_position_data, _ = add_hit_miss_try(processed_position_data)

                spike_data = add_position_x(spike_data, raw_position_data) # position per spike
                spike_data = add_trial_number(spike_data, raw_position_data) # trial number per spike
                spike_data = add_trial_type(spike_data, raw_position_data) # trial type per spike
                spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length=get_track_length(recording), smoothen=True)
                spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length=get_track_length(recording), smoothen=False)
                spike_data = bin_fr_in_time(spike_data, raw_position_data, smoothen=True)
                spike_data = bin_fr_in_time(spike_data, raw_position_data, smoothen=False)
                spike_data = add_displayed_peak_firing(spike_data) # for plotting
                spike_data = calculate_spatial_information(spike_data, position_data, track_length=get_track_length(recording))
                #spike_data = add_peak_power_from_classic_power_spectra(spike_data, output_path)
                spike_data = add_stops_to_spatial_firing(spike_data, processed_position_data, track_length=get_track_length(recording))
                spike_data = add_field_locations(spike_data, track_length=get_track_length(recording))
                spike_data = add_stops(spike_data, processed_position_data, track_length=get_track_length(recording))
                spike_data = add_trial_info(spike_data, processed_position_data) # info per trial

                # MOVING LOMB PERIODOGRAMS
                spike_data = calculate_moving_lomb_scargle_periodogram(spike_data, processed_position_data, track_length=get_track_length(recording))
                spike_data = analyse_lomb_powers(spike_data)
                spike_data = add_lomb_classifier(spike_data)
                spike_data = add_rolling_stats_shuffled_test(spike_data, processed_position_data, track_length=get_track_length(recording))
                spike_data = add_rolling_stats(spike_data, track_length=get_track_length(recording))
                spike_data = add_coding_by_trial_number(spike_data, processed_position_data)
                spike_data = add_mean_firing_rate_during_position_and_distance_trials(spike_data, position_data, track_length=get_track_length(recording))
                spike_data = add_spatial_information_during_position_and_distance_trials(spike_data, position_data, track_length=get_track_length(recording))
                spike_data = add_grid_code_global(spike_data)
                spike_data = add_agreement_between_cell_and_grid_global(spike_data)

                # Alternative classification
                spike_data = add_rolling_stats_by_trial_number_using_alternative_classification(spike_data)

                spike_data.to_pickle(recording+"/"+sorter_name+"/DataFrames/spatial_firing.pkl")
                position_data.to_pickle(recording+"/"+sorter_name+"/DataFrames/position_data.pkl")
                processed_position_data.to_pickle(recording + "/" + sorter_name + "/DataFrames/processed_position_data.pkl")
                print("successfully processed and saved vr_grid analysis on "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)


def main():
    print('-------------------------------------------------------------')
    vr_path_list = []
    of_path_list = []
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    process_recordings(vr_path_list, of_path_list)
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()
