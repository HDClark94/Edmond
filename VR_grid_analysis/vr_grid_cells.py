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
from PostSorting.vr_spatial_firing import bin_fr_in_space, bin_fr_in_time, add_position_x
from scipy import stats
import Edmond.VR_grid_analysis.analysis_settings as Settings
from scipy import signal
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
import os
import traceback
from astropy.nddata import block_reduce
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
        rewarded_bin_centres = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_rewarded_processed_position_data['pos_binned_in_space'])
        in_rz_mask = (rewarded_bin_centres > reward_zone_start) & (rewarded_bin_centres <= reward_zone_end)
        rewarded_speeds_in_space_in_reward_zone = rewarded_speeds_in_space[in_rz_mask]
        rewarded_speeds_in_space_in_reward_zone = rewarded_speeds_in_space_in_reward_zone[~np.isnan(rewarded_speeds_in_space_in_reward_zone)]
        speeds_in_rz.extend(rewarded_speeds_in_space_in_reward_zone.tolist())

    speeds_in_rz = np.array(speeds_in_rz)
    mean, sigma = np.nanmean(speeds_in_rz), np.nanstd(speeds_in_rz)
    interval = stats.norm.interval(0.95, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]

    hit_miss_try =[]
    avg_speed_in_rz =[]
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_process_position_data = processed_position_data[(processed_position_data.trial_number == trial_number)]
        track_speed = trial_process_position_data["avg_speed_on_track"].iloc[0]
        trial_speeds_in_space = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_process_position_data['speeds_binned_in_space'])
        trial_bin_centres = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_process_position_data['pos_binned_in_space'])
        in_rz_mask = (trial_bin_centres > reward_zone_start) & (trial_bin_centres <= reward_zone_end)
        trial_speeds_in_reward_zone = trial_speeds_in_space[in_rz_mask]
        trial_speeds_in_reward_zone = trial_speeds_in_reward_zone[~np.isnan(trial_speeds_in_reward_zone)]
        avg_trial_speed_in_reward_zone = np.mean(trial_speeds_in_reward_zone)

        if (trial_process_position_data["rewarded"].iloc[0] == True) and (track_speed>Settings.track_speed_threshold):
            hit_miss_try.append("hit")
        elif (avg_trial_speed_in_reward_zone >= lower) and (avg_trial_speed_in_reward_zone <= upper) and (track_speed>Settings.track_speed_threshold):
            hit_miss_try.append("try")
        elif (avg_trial_speed_in_reward_zone < lower) or (avg_trial_speed_in_reward_zone > upper) and (track_speed>Settings.track_speed_threshold):
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

def plot_spatial_autocorrelogram_fr(spike_data, output_path, track_length, suffix=""):
    print('plotting spike spatial autocorrelogram fr...')
    save_path = output_path + '/Figures/spatial_autocorrelograms_fr'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_rates = np.array(cluster_spike_data["fr_binned_in_space"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            fr = firing_rates.flatten()
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


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

def reduce_digits(numeric_float, n_digits=6):
    scientific_notation = "{:.1e}".format(numeric_float)
    return scientific_notation

def plot_moving_lomb_scargle_periodogram(spike_data, processed_position_data, output_path, track_length):
    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    n_trials = len(processed_position_data)
    elapsed_distance_bins = np.arange(0, (track_length*n_trials)+1, 1)
    elapsed_distance = 0.5*(elapsed_distance_bins[1:]+elapsed_distance_bins[:-1])/track_length

    freqs = []
    SNRs = []
    avg_powers = []
    all_powers = []
    all_centre_trials=[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_rates = np.array(cluster_spike_data["fr_binned_in_space"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
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

            avg_power = np.nanmean(powers, axis=0)
            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
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
            ax.plot(frequency, avg_subset_powers, color="black", linestyle="solid", linewidth=3)
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
    return spike_data


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
        step = Settings.frequency_step
        frequency = Settings.frequency

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

    step = Settings.frequency_step
    frequency = Settings.frequency

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

def get_vmin_vmax(cluster_firing_maps, bin_cm=8):
    cluster_firing_maps_reduced = []
    for i in range(len(cluster_firing_maps)):
        cluster_firing_maps_reduced.append(block_reduce(cluster_firing_maps[i], bin_cm, func=np.mean))
    cluster_firing_maps_reduced = np.array(cluster_firing_maps_reduced)
    vmin= 0
    vmax= np.max(cluster_firing_maps_reduced)
    if vmax==0:
        print("stop here")
    return vmin, vmax

def plot_firing_rate_maps_per_trial(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps2 = np.array(spike_data["fr_binned_in_space"].iloc[cluster_index])
            where_are_NaNs2 = np.isnan(cluster_firing_maps2)
            cluster_firing_maps2[where_are_NaNs2] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps2)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)


            x_max = len(processed_position_data)
            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            locations = np.arange(0, len(cluster_firing_maps[0]))
            ordered = np.arange(0, len(processed_position_data), 1)
            X, Y = np.meshgrid(locations, ordered)
            cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
            ax.pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
            plt.ylabel('Trial Number', fontsize=20, labelpad = 20)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
            plt.xlim(0, track_length)
            ax.tick_params(axis='both', which='both', labelsize=20)
            plt.xlabel('Location (cm)', fontsize=25, labelpad = 20)
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


def get_lomb_classifier_per_trial(rolling_points, rolling_classifier, processed_position_data):
    lomb_class_per_trial = []
    for tn in processed_position_data["trial_number"]:
        rolling_classifier_trial = rolling_classifier[rolling_points == tn]
        if len(rolling_classifier_trial) == 0:
            lomb_class_per_trial.append(np.nan)
        else:
            lomb_class_per_trial.append(stats.mode(rolling_classifier_trial)[0][0])
    return np.array(lomb_class_per_trial)

def add_rolling_lomb_classifier_percentages(spike_data, proceseed_position_data):

    allo_minus_ego_hit_proportions_b = []
    allo_minus_ego_try_proportions_b = []
    allo_minus_ego_miss_proportions_b = []
    allo_minus_ego_hit_proportions_nb = []
    allo_minus_ego_try_proportions_nb = []
    allo_minus_ego_miss_proportions_nb = []

    percentage_ego = []
    percentage_allo = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        if len(firing_times_cluster)>1:
            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)
            rolling_lomb_classifier, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial_not_numeric(centre_trials, powers)
            lomb_classifier_per_trial = get_lomb_classifier_per_trial(rolling_points, rolling_lomb_classifier, proceseed_position_data)

            allo = (len(rolling_lomb_classifier[rolling_lomb_classifier=="Position"])/len(centre_trials))*100
            ego = (len(rolling_lomb_classifier[rolling_lomb_classifier=="Distance"])/len(centre_trials))*100

            classifiers_during_b_hits = lomb_classifier_per_trial[proceseed_position_data[(proceseed_position_data["trial_type"]==0) & (proceseed_position_data["hit_miss_try"]=="hit")]["trial_number"]-1]
            classifiers_during_b_tries = lomb_classifier_per_trial[proceseed_position_data[(proceseed_position_data["trial_type"]==0) & (proceseed_position_data["hit_miss_try"]=="try")]["trial_number"]-1]
            classifiers_during_b_misses = lomb_classifier_per_trial[proceseed_position_data[(proceseed_position_data["trial_type"]==0) & (proceseed_position_data["hit_miss_try"]=="miss")]["trial_number"]-1]
            classifiers_during_nb_hits = lomb_classifier_per_trial[proceseed_position_data[(proceseed_position_data["trial_type"]==1) & (proceseed_position_data["hit_miss_try"]=="hit")]["trial_number"]-1]
            classifiers_during_nb_tries = lomb_classifier_per_trial[proceseed_position_data[(proceseed_position_data["trial_type"]==1) & (proceseed_position_data["hit_miss_try"]=="try")]["trial_number"]-1]
            classifiers_during_nb_misses = lomb_classifier_per_trial[proceseed_position_data[(proceseed_position_data["trial_type"]==1) & (proceseed_position_data["hit_miss_try"]=="miss")]["trial_number"]-1]

            if len(classifiers_during_b_hits)>0:
                allo_minus_ego_hit_proportion_b = (len(classifiers_during_b_hits[classifiers_during_b_hits=="Position"])/len(classifiers_during_b_hits))-(len(classifiers_during_b_hits[classifiers_during_b_hits=="Distance"])/len(classifiers_during_b_hits))
            else:
                allo_minus_ego_hit_proportion_b = np.nan

            if len(classifiers_during_b_tries)>0:
                allo_minus_ego_try_proportion_b = (len(classifiers_during_b_tries[classifiers_during_b_tries=="Position"])/len(classifiers_during_b_tries))-(len(classifiers_during_b_tries[classifiers_during_b_tries=="Distance"])/len(classifiers_during_b_tries))
            else:
                allo_minus_ego_try_proportion_b = np.nan

            if len(classifiers_during_b_misses)>0:
                allo_minus_ego_miss_proportion_b = (len(classifiers_during_b_misses[classifiers_during_b_misses=="Position"])/len(classifiers_during_b_misses))-(len(classifiers_during_b_misses[classifiers_during_b_misses=="Distance"])/len(classifiers_during_b_misses))
            else:
                allo_minus_ego_miss_proportion_b = np.nan

            if len(classifiers_during_nb_hits)>0:
                allo_minus_ego_hit_proportion_nb = (len(classifiers_during_nb_hits[classifiers_during_nb_hits=="Position"])/len(classifiers_during_nb_hits))-(len(classifiers_during_nb_hits[classifiers_during_nb_hits=="Distance"])/len(classifiers_during_nb_hits))
            else:
                allo_minus_ego_hit_proportion_nb = np.nan

            if len(classifiers_during_nb_tries):
                allo_minus_ego_try_proportion_nb = (len(classifiers_during_nb_tries[classifiers_during_nb_tries=="Position"])/len(classifiers_during_nb_tries))-(len(classifiers_during_nb_tries[classifiers_during_nb_tries=="Distance"])/len(classifiers_during_nb_tries))
            else:
                allo_minus_ego_try_proportion_nb = np.nan

            if len(classifiers_during_nb_misses)>0:
                allo_minus_ego_miss_proportion_nb = (len(classifiers_during_nb_misses[classifiers_during_nb_misses=="Position"])/len(classifiers_during_nb_misses))-(len(classifiers_during_nb_misses[classifiers_during_nb_misses=="Distance"])/len(classifiers_during_nb_misses))
            else:
                allo_minus_ego_miss_proportion_nb = np.nan

        else:
            allo = np.nan
            ego = np.nan
            allo_minus_ego_hit_proportion_b = np.nan
            allo_minus_ego_try_proportion_b = np.nan
            allo_minus_ego_miss_proportion_b = np.nan
            allo_minus_ego_hit_proportion_nb = np.nan
            allo_minus_ego_try_proportion_nb = np.nan
            allo_minus_ego_miss_proportion_nb = np.nan

        percentage_allo.append(allo)
        percentage_ego.append(ego)
        allo_minus_ego_hit_proportions_b.append(allo_minus_ego_hit_proportion_b)
        allo_minus_ego_try_proportions_b.append(allo_minus_ego_try_proportion_b)
        allo_minus_ego_miss_proportions_b.append(allo_minus_ego_miss_proportion_b)
        allo_minus_ego_hit_proportions_nb.append(allo_minus_ego_hit_proportion_nb)
        allo_minus_ego_try_proportions_nb.append(allo_minus_ego_try_proportion_nb)
        allo_minus_ego_miss_proportions_nb.append(allo_minus_ego_miss_proportion_nb)

    spike_data["percentage_allocentric"] = percentage_allo
    spike_data["percentage_egocentric"] = percentage_ego
    spike_data["allo_minus_ego_hit_proportions_b"] = allo_minus_ego_hit_proportions_b
    spike_data["allo_minus_ego_try_proportions_b"] = allo_minus_ego_try_proportions_b
    spike_data["allo_minus_ego_miss_proportions_b"] = allo_minus_ego_miss_proportions_b
    spike_data["allo_minus_ego_hit_proportions_nb"] = allo_minus_ego_hit_proportions_nb
    spike_data["allo_minus_ego_try_proportions_nb"] = allo_minus_ego_try_proportions_nb
    spike_data["allo_minus_ego_miss_proportions_nb"] = allo_minus_ego_miss_proportions_nb
    return spike_data


def get_rolling_lomb_classifier_for_centre_trial_not_numeric(centre_trials, powers, n_window_size=Settings.rolling_window_size_for_lomb_classifier):
    frequency = Settings.frequency

    trial_points = []
    peak_frequencies = []
    rolling_lomb_classifier = []
    rolling_lomb_classifier_colors = []
    for i in range(len(centre_trials)):
        centre_trial = centre_trials[i]

        if i<int(n_window_size/2):
            power_window = powers[:i+int(n_window_size/2), :]
        elif i+int(n_window_size/2)>len(centre_trials):
            power_window = powers[i-int(n_window_size/2):, :]
        else:
            power_window = powers[i-int(n_window_size/2):i+int(n_window_size/2), :]

        avg_power = np.nanmean(power_window, axis=0)
        max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)

        lomb_classifier = get_lomb_classifier(max_SNR, max_SNR_freq, 0.023, 0.05, numeric=False)
        peak_frequencies.append(max_SNR_freq)
        trial_points.append(centre_trial)
        if lomb_classifier == "Position":
            rolling_lomb_classifier.append("Position")
            rolling_lomb_classifier_colors.append(Settings.allocentric_color)
        elif lomb_classifier == "Distance":
            rolling_lomb_classifier.append("Distance")
            rolling_lomb_classifier_colors.append(Settings.egocentric_color)
        elif lomb_classifier == "Null":
            rolling_lomb_classifier.append("Null")
            rolling_lomb_classifier_colors.append(Settings.null_color)
        else:
            rolling_lomb_classifier.append(3.5)
            rolling_lomb_classifier_colors.append("black")
    return np.array(rolling_lomb_classifier), np.array(rolling_lomb_classifier_colors), np.array(peak_frequencies), np.array(trial_points)


def process_recordings(vr_recording_path_list, of_recording_path_list):
    vr_recording_path_list.sort()

    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            output_path = recording+'/'+settings.sorterName
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            raw_position_data, position_data = syncronise_position_data(recording, get_track_length(recording))
            spike_data = add_position_x(spike_data, raw_position_data)
            spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length=get_track_length(recording))
            position_data = add_time_elapsed_collumn(position_data)

            # BEHAVIOURAL
            processed_position_data = add_avg_track_speed(processed_position_data, track_length=get_track_length(recording))
            processed_position_data, _ = add_hit_miss_try3(processed_position_data, track_length=get_track_length(recording))
            spike_data = add_percentage_hits(spike_data, processed_position_data)

            # MOVING LOMB PERIODOGRAMS
            spike_data = plot_moving_lomb_scargle_periodogram(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))
            spike_data = add_rolling_lomb_classifier_percentages(spike_data, processed_position_data)
            spike_data = analyse_lomb_powers(spike_data, processed_position_data)
            spike_data = analyse_lomb_powers_ego_vs_allocentric(spike_data, processed_position_data)
            #shuffle_data = analyse_lomb_powers(shuffle_data, processed_position_data)

            # FIRING AND BEHAVIOURAL PLOTTING
            plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=get_track_length(recording),plot_trials=["beaconed", "non_beaconed", "probe"])
            plot_stops_on_track(processed_position_data, output_path, track_length=get_track_length(recording))
            plot_stop_histogram(processed_position_data, output_path, track_length=get_track_length(recording))

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
    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36']
    vr_path_list = []
    of_path_list = []
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort9_Junji/vr") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort9_Junji/of") if f.is_dir()])
    process_recordings(vr_path_list, of_path_list)


    print("look now")

if __name__ == '__main__':
    main()
