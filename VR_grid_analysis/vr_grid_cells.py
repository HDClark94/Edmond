import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from numpy import inf
from scipy.ndimage import uniform_filter1d
import PostSorting.parameters
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from Edmond.VR_grid_analysis.remake_position_data import syncronise_position_data
from PostSorting.vr_spatial_firing import bin_fr_in_space, bin_fr_in_time, add_position_x, add_trial_number, add_trial_type
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
from scipy import stats
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
        firing_rates = np.array(cluster_spike_data['fr_binned_in_space_smoothed'].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

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

def calculate_moving_lomb_scargle_periodogram(spike_data, processed_position_data, track_length, shuffled_trials=False):
    print('calculating moving lomb_scargle periodogram...')

    if shuffled_trials:
        suffix="_shuffled_trials"
    else:
        suffix=""

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
            avg_power = np.nanmean(powers, axis=0)
            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)

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

    spike_data["MOVING_LOMB_freqs"+suffix] = freqs
    spike_data["MOVING_LOMB_avg_power"+suffix] = avg_powers
    spike_data["MOVING_LOMB_SNR"+suffix] = SNRs
    spike_data["MOVING_LOMB_all_powers"+suffix] = all_powers
    spike_data["MOVING_LOMB_all_centre_trials"+suffix] = all_centre_trials
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
                            if code == "_allo":
                                _, _, code_i = get_allocentric_peak(frequency, avg_powers, tolerance=0.05)
                            elif code == "_ego":
                                _, _, code_i = get_egocentric_peak(frequency, avg_powers, tolerance=0.05)

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


def plot_avg_spatial_periodograms_with_rolling_classifications(spike_data, processed_position_data, output_path, track_length, plot_for_all_trials=True):

    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    power_step = Settings.power_estimate_step
    step = Settings.frequency_step
    frequency = Settings.frequency

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])#
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

            plot_avg = True
            # add avg periodograms for position and distance coded trials
            for code, c in zip(["P", "D"], [Settings.allocentric_color, Settings.egocentric_color]):
                subset_trial_numbers = np.unique(rolling_points[rolling_lomb_classifier==code])

                # only plot if there if this is at least 15% of total trials
                if (len(subset_trial_numbers)/len(processed_position_data["trial_number"])>=0.15) or (plot_for_all_trials==True):
                    #plot_avg = False
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
                ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color="black", alpha=0.3)
                ax.plot(frequency, avg_subset_powers, color="black", linewidth=3)
                #allocentric_peak_freq, allocentric_peak_power, allo_i = get_allocentric_peak(frequency, avg_subset_powers, tolerance=0.05)
                #egocentric_peak_freq, egocentric_peak_power, ego_i = get_egocentric_peak(frequency, avg_subset_powers, tolerance=0.05)
                #ax.scatter(allocentric_peak_freq, allocentric_peak_power, color=Settings.allocentric_color, marker="v", s=200, zorder=10)
                #ax.scatter(egocentric_peak_freq, egocentric_peak_power, color=Settings.egocentric_color, marker="v", s=200, zorder=10)

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
            plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_avg_spatial_periodograms_with_rolling_classifications_Cluster_' + str(cluster_id) +'.png', dpi=300)
            plt.close()
    return

def plot_firing_rate_maps_short_with_rolling_classifications(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            rolling_centre_trials = np.array(spike_data["rolling:rolling_centre_trials"].iloc[cluster_index])
            rolling_classifiers = np.array(spike_data["rolling:rolling_classifiers"].iloc[cluster_index])

            # remove first and last classification in streak
            remove_last_and_first_from_streak = True
            last_classifier=""
            streak = 1
            new_rolling_classifier = rolling_classifiers.copy()
            for j in range(len(rolling_classifiers)):
                if rolling_classifiers[j] == last_classifier:
                    streak += 1
                else:
                    streak = 1
                    new_rolling_classifier[j-1] = "nan"

                if streak == 1:
                    new_rolling_classifier[j] = "nan"

                last_classifier = rolling_classifiers[j]
            if remove_last_and_first_from_streak:
                rolling_classifiers = new_rolling_classifier

            cluster_firing_maps[np.isnan(cluster_firing_maps)] = np.nan
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = np.nan

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5/3, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1) 
            locations = np.arange(0, len(cluster_firing_maps[0]))
            ax.fill_between(locations, np.nanmean(cluster_firing_maps, axis=0)-stats.sem(cluster_firing_maps, axis=0, nan_policy="omit"), np.nanmean(cluster_firing_maps, axis=0)+stats.sem(cluster_firing_maps, axis=0, nan_policy="omit"), color="black", alpha=0.3)
            ax.plot(locations, np.nanmean(cluster_firing_maps, axis=0), color="black", linewidth=3)
            for code, code_color in zip(["P", "D"], [Settings.allocentric_color, Settings.egocentric_color]):
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
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_maps_short_with_rolling_classifications_' + str(cluster_id) + '.png', dpi=300)
            plt.close()
    return

def plot_firing_rate_maps_short(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = np.nan
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = np.nan
            #percentile_99th_display = np.nanpercentile(cluster_firing_maps, 99);
            #cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            #percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            #vmin, vmax = get_vmin_vmax(cluster_firing_maps)

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5/3, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            locations = np.arange(0, len(cluster_firing_maps[0]))
            ax.fill_between(locations, np.nanmean(cluster_firing_maps, axis=0)-stats.sem(cluster_firing_maps, axis=0, nan_policy="omit"), np.nanmean(cluster_firing_maps, axis=0)+stats.sem(cluster_firing_maps, axis=0, nan_policy="omit"), color="black", alpha=0.3)
            ax.plot(locations, np.nanmean(cluster_firing_maps, axis=0), color="black", linewidth=3)
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
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_maps_short_' + str(cluster_id) + '.png', dpi=300)
            plt.close()
    return

def moving_average(x, n):
    cumsum = np.cumsum(np.concatenate((np.array([x[0]]), x, np.array([x[-1]])),axis=0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def plot_firing_rate_maps_per_trial(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            percentile_99th_display = np.nanpercentile(cluster_firing_maps, 99);
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)

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
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '.png', dpi=300)
            plt.close()


            fig, ax = plt.subplots(1, 1, figsize=(1,5))
            cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
            avg_firing_rates_per_trial = np.nanmean(cluster_firing_maps, axis=1)
            avg_firing_rates_per_trial = uniform_filter1d(avg_firing_rates_per_trial, size=10)
            x_pos = 0
            legend_freq = np.linspace(x_pos, x_pos+0.2, 5)
            avg_firing_rates_per_trial_tiled = np.tile(avg_firing_rates_per_trial,(len(legend_freq),1))
            Y, X = np.meshgrid(np.array(processed_position_data["trial_number"]), legend_freq)
            ax.pcolormesh(X, Y, avg_firing_rates_per_trial_tiled, cmap=cmap, shading="flat")

            ax.set_ylim([0, len(processed_position_data)])
            ax.set_xlim(0, 0.3)
            plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
            ax.xaxis.set_tick_params(labelsize=30)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.xaxis.set_tick_params(labelsize=20)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/avg_firing_rates_across_trials_'+ str(cluster_id) + '.png', dpi=300)
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
        return "blue"
    elif trial_type == 1:
        return "red"
    elif trial_type == 2:
        return "deepskyblue"
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


def get_avg_correlation(firing_rate_map):
    corrs=[]
    for i in range(len(firing_rate_map)):
        for j in range(len(firing_rate_map)):
            if i!=j:
                corr = stats.pearsonr(firing_rate_map[i], firing_rate_map[j])[0]
                corrs.append(corr)
    return np.nanmean(corrs)

def add_realignement_shifts(spike_data, processed_position_data, track_length):
    realignments = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[spike_data["cluster_id"]==cluster_id]
        firing_times_cluster = spike_data.firing_times.iloc[0]
        # determine the max shift based on the spatial frequency
        try:
            putative_field_frequency = int(np.round(cluster_df["ML_Freqs"].iloc[0]))
            max_shift = int(track_length/putative_field_frequency)
        except:
            max_shift = track_length

        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)

            shifts=[]
            mean_cluster_firing_map = np.nanmean(cluster_firing_maps, axis=0)

            # best align to mean firing rate map
            for ti, tn in enumerate(processed_position_data["trial_number"]):
                A = mean_cluster_firing_map
                B = cluster_firing_maps[ti].flatten()
                A -= A.mean(); A /= A.std() # z score
                B -= B.mean(); B /= B.std() # z score
                xcorr = signal.correlate(A, B, mode="same")
                lags = signal.correlation_lags(A.size, B.size, mode="same")
                xcorr /= np.max(xcorr)

                xcorr = xcorr[np.abs(lags).argsort()][0:max_shift]
                lags = lags[np.abs(lags).argsort()][0:max_shift]
                recovered_shift = lags[xcorr.argmax()]
                shifts.append(recovered_shift)

            realignments.append(np.array(shifts))
        else:
            realignments.append(np.nan)

    spike_data["map_realignments"] = realignments
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

def add_n_PI_trial(spike_data, processed_position_data):
    processed_position_data = processed_position_data[(processed_position_data["trial_type"] == 1) | (processed_position_data["trial_type"] == 2)]
    n_hits = len(processed_position_data[processed_position_data["hit_miss_try"] == "hit"])
    n_tries = len(processed_position_data[processed_position_data["hit_miss_try"] == "try"])
    n_misses = len(processed_position_data[processed_position_data["hit_miss_try"] == "miss"])
    new_column = [[n_hits, n_misses, n_tries]]

    new = pd.DataFrame()
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        cluster_spike_data["n_pi_trials_by_hmt"] = new_column
        new = pd.concat([new, cluster_spike_data], ignore_index=True)
    return new

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


def get_rolling_lomb_classifier_for_centre_trial(centre_trials, powers, power_threshold, power_step, track_length, n_window_size=Settings.rolling_window_size_for_lomb_classifier):

    frequency = Settings.frequency

    trial_points = []
    peak_frequencies = []
    rolling_lomb_classifier = []
    rolling_lomb_classifier_numeric = []
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

        lomb_classifier = get_lomb_classifier(max_SNR, max_SNR_freq, power_threshold, 0.05, numeric=False)
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

def get_modal_class_char(modal_class):
    if modal_class == "Position":
        classifier = "P"
    elif modal_class == "Distance":
        classifier = "D"
    elif modal_class == "Null":
        classifier = "N"
    else:
        classifier = "U"
    return classifier

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

def add_rolling_stats_shuffled_blocks(spike_data, track_length):
    power_step = Settings.power_estimate_step

    rolling_lomb_classifiers=[]
    rolling_lomb_classifiers_shuffled_blocks=[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])#
        rolling_power_threshold =  cluster_spike_data["rolling_threshold"].iloc[0]

        if len(firing_times_cluster)>1:
            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)

            powers[np.isnan(powers)] = 0
            rolling_lomb_classifier, rolling_lomb_classifier_numeric, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers, power_threshold=rolling_power_threshold, power_step=power_step, track_length=track_length)
            rolling_lomb_classifier_shuffled_by_blocks = shuffle_blocks(rolling_lomb_classifier)
        else:
            rolling_lomb_classifier = np.array([])
            rolling_lomb_classifier_shuffled_by_blocks=np.array([])

        rolling_lomb_classifiers.append(rolling_lomb_classifier)
        rolling_lomb_classifiers_shuffled_blocks.append(rolling_lomb_classifier_shuffled_by_blocks)

    spike_data["rolling:rolling_lomb_classifiers"] = rolling_lomb_classifiers
    spike_data["rolling:rolling_lomb_classifiers_shuffled_blocks"] = rolling_lomb_classifiers_shuffled_blocks
    return spike_data

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
    for tn in rolling_trials:
        rolling_class = rolling_classifiers[rolling_centre_trials == tn]
        mode = stats.mode(rolling_class, axis=None)[0][0]
        rolling_modes.append(mode)
    rolling_classifiers = np.array(rolling_modes)
    rolling_centre_trials = rolling_trials
    return rolling_centre_trials, rolling_classifiers

def add_rolling_stats_percentage_hits(spike_data, processed_position_data):
    encoding_position = []
    encoding_distance = []
    encoding_null = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])#
        rolling_centre_trials = cluster_spike_data["rolling:rolling_centre_trials"].iloc[0]
        rolling_classifiers = cluster_spike_data["rolling:rolling_classifiers"].iloc[0]

        rolling_centre_trials, rolling_classifiers = compress_rolling_stats(rolling_centre_trials, rolling_classifiers)

        # make empty array for trial types and trial outcomes
        P = [[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]
        D = [[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]
        N = [[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]

        if len(firing_times_cluster)>0:
            for i, tt in enumerate([0,1,2]):
                tt_processed_position_data = processed_position_data[(processed_position_data["trial_type"] == tt)]
                tt_trial_numbers = np.array(tt_processed_position_data["trial_number"])

                p_trials = rolling_centre_trials[np.isin(rolling_centre_trials, tt_trial_numbers) & (rolling_classifiers=="P")]
                d_trials = rolling_centre_trials[np.isin(rolling_centre_trials, tt_trial_numbers) & (rolling_classifiers=="D")]
                n_trials = rolling_centre_trials[np.isin(rolling_centre_trials, tt_trial_numbers) & (rolling_classifiers=="N")]

                for j, hmt in enumerate(["hit", "try", "miss"]):
                    subset_processed_position_data = tt_processed_position_data[(tt_processed_position_data["hit_miss_try"] == hmt)]
                    subset_trial_numbers = np.array(subset_processed_position_data["trial_number"])

                    if len(p_trials)>0:
                        P[i][j] = len(p_trials[np.isin(p_trials, subset_trial_numbers)])/len(p_trials)
                    if len(d_trials)>0:
                        D[i][j] = len(d_trials[np.isin(d_trials, subset_trial_numbers)])/len(d_trials)
                    if len(n_trials)>0:
                        N[i][j] = len(n_trials[np.isin(n_trials, subset_trial_numbers)])/len(n_trials)

        encoding_position.append(P)
        encoding_distance.append(D)
        encoding_null.append(N)

    spike_data["rolling:percentage_trials_encoding_position"] = encoding_position
    spike_data["rolling:percentage_trials_encoding_distance"] = encoding_distance
    spike_data["rolling:percentage_trials_encoding_null"] = encoding_null
    return spike_data


def add_coding_by_trial_number(spike_data, processed_position_data, remove_first_and_last_in_streak=False):
    cluster_codes = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>0:
            rolling_centre_trials = cluster_spike_data["rolling:rolling_centre_trials"].iloc[0]
            rolling_classifiers = cluster_spike_data["rolling:rolling_classifiers"].iloc[0]

            rolling_centre_trials, rolling_classifiers = compress_rolling_stats(rolling_centre_trials, rolling_classifiers)

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

def add_rolling_stats_hmt(spike_data, processed_position_data):
    encoding_position = []
    encoding_distance = []
    encoding_null = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])#
        rolling_centre_trials = cluster_spike_data["rolling:rolling_centre_trials"].iloc[0]
        rolling_classifiers = cluster_spike_data["rolling:rolling_classifiers"].iloc[0]

        # make empty array for trial types and trial outcomes
        P = [[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]
        D = [[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]
        N = [[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]

        if len(firing_times_cluster)>0:
            for i, tt in enumerate([0,1,2]):
                for j, hmt in enumerate(["hit", "try", "miss"]):
                    subset_processed_position_data = processed_position_data[(processed_position_data["trial_type"] == tt) &
                                                                             (processed_position_data["hit_miss_try"] == hmt)]
                    subset_trial_numbers = np.array(subset_processed_position_data["trial_number"])


                    subset_rolling_classifiers = rolling_classifiers[np.isin(rolling_centre_trials, subset_trial_numbers)]

                    if len(subset_rolling_classifiers)>0:
                        P[i][j] = len(subset_rolling_classifiers[subset_rolling_classifiers == "P"])/len(subset_rolling_classifiers)
                        D[i][j] = len(subset_rolling_classifiers[subset_rolling_classifiers == "D"])/len(subset_rolling_classifiers)
                        N[i][j] = len(subset_rolling_classifiers[subset_rolling_classifiers == "N"])/len(subset_rolling_classifiers)

        encoding_position.append(P)
        encoding_distance.append(D)
        encoding_null.append(N)

    spike_data["rolling:encoding_position_by_trial_category"] = encoding_position
    spike_data["rolling:encoding_distance_by_trial_category"] = encoding_distance
    spike_data["rolling:encoding_null_by_trial_category"] = encoding_null
    return spike_data

def add_rolling_stats(spike_data, track_length):
    power_step = Settings.power_estimate_step

    rolling_centre_trials=[]
    rolling_peak_frequencies=[]
    rolling_classifiers=[]
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
        else:
            rolling_lomb_classifier = np.array([])
            rolling_frequencies = np.array([])
            rolling_points = np.array([])

        rolling_centre_trials.append(rolling_points)
        rolling_peak_frequencies.append(rolling_frequencies)
        rolling_classifiers.append(rolling_lomb_classifier)

    spike_data["rolling:rolling_peak_frequencies"] = rolling_peak_frequencies
    spike_data["rolling:rolling_centre_trials"] = rolling_centre_trials
    spike_data["rolling:rolling_classifiers"] = rolling_classifiers
    return spike_data

def add_rolling_stats_encoding_x(spike_data, track_length):
    power_step = Settings.power_estimate_step

    proportion_encoding_position=[]
    proportion_encoding_distance=[]
    proportion_encoding_null=[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])#
        rolling_power_threshold =  cluster_spike_data["rolling_threshold"].iloc[0]

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
            proportion_encoding_P = np.nan
            proportion_encoding_D = np.nan
            proportion_encoding_N = np.nan

        proportion_encoding_position.append(proportion_encoding_P)
        proportion_encoding_distance.append(proportion_encoding_D)
        proportion_encoding_null.append(proportion_encoding_N)

    spike_data["rolling:proportion_encoding_position"] = proportion_encoding_position
    spike_data["rolling:proportion_encoding_distance"] = proportion_encoding_distance
    spike_data["rolling:proportion_encoding_null"] = proportion_encoding_null
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

def get_stop_histogram(cells_df, tt, coding_scheme=None, shuffle=False, track_length=None, remove_last_and_first_from_streak=True):
    if shuffle:
        iterations = 1000
    else:
        iterations = 1
    gauss_kernel = Gaussian1DKernel(1)

    stop_histograms=[]
    stop_histogram_sems=[]
    for index, cluster_df in cells_df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        if track_length is None:
            track_length = cluster_df["track_length"].iloc[0]

        stops_location_cm = np.array(cluster_df["stop_locations"].iloc[0])
        stop_trial_numbers = np.array(cluster_df["stop_trial_numbers"].iloc[0])

        trial_numbers = np.array(cluster_df["behaviour_trial_numbers"].iloc[0])
        trial_types = np.array(cluster_df["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cluster_df["rolling:classifier_by_trial_number"].iloc[0])


        # remove first and last classification in streak
        last_classifier=""
        streak = 1
        new_rolling_classifier = rolling_classifiers.copy()
        for j in range(len(rolling_classifiers)):
            if rolling_classifiers[j] == last_classifier:
                streak +=1
            else:
                streak = 1
                new_rolling_classifier[j-1] = "nan"

            if streak == 1:
                new_rolling_classifier[j] = "nan"

            last_classifier = rolling_classifiers[j]
        if remove_last_and_first_from_streak:
            rolling_classifiers = new_rolling_classifier


        # mask out only the trial numbers based on the trial type
        # and the coding scheme if that argument is given
        trial_type_mask = np.isin(trial_types, tt)
        if coding_scheme is not None:
            classifier_mask = np.isin(rolling_classifiers, coding_scheme)
            tt_trial_numbers = trial_numbers[trial_type_mask & classifier_mask]
        else:
            tt_trial_numbers = trial_numbers[trial_type_mask]

        number_of_bins = track_length
        number_of_trials = len(tt_trial_numbers)

        stop_counts = np.zeros((iterations, number_of_trials, number_of_bins)); stop_counts[:,:,:] = np.nan

        for j in np.arange(iterations):
            if shuffle:
                stops_location_cm = np.random.uniform(low=0, high=track_length, size=len(stops_location_cm))

            for i, tn in enumerate(tt_trial_numbers):
                stop_locations_on_trial = stops_location_cm[stop_trial_numbers == tn]
                stop_in_trial_bins, bin_edges = np.histogram(stop_locations_on_trial, bins=number_of_bins, range=[0,track_length])
                stop_counts[j,i,:] = stop_in_trial_bins

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

        bin_centres = np.arange(0.5, track_length+0.5, track_length/number_of_bins)

    return stop_histograms, stop_histogram_sems, bin_centres, number_of_trials


def get_stop_histogram_cluster(cluster_df, tt, coding_scheme=None, shuffle=False, track_length=None):
    gauss_kernel = Gaussian1DKernel(1)

    if track_length is None:
        track_length = cluster_df["track_length"].iloc[0]

    stops_location_cm = np.array(cluster_df["stop_locations"].iloc[0])
    stop_trial_numbers = np.array(cluster_df["stop_trial_numbers"].iloc[0])

    trial_numbers = np.array(cluster_df["behaviour_trial_numbers"].iloc[0])
    trial_types = np.array(cluster_df["behaviour_trial_types"].iloc[0])
    rolling_classifiers = np.array(cluster_df["rolling:classifier_by_trial_number"].iloc[0])

    # mask out only the trial numbers based on the trial type
    # and the coding scheme if that argument is given
    trial_type_mask = np.isin(trial_types, tt)
    if coding_scheme is not None:
        classifier_mask = np.isin(rolling_classifiers, coding_scheme)
        tt_trial_numbers = trial_numbers[trial_type_mask & classifier_mask]
    else:
        tt_trial_numbers = trial_numbers[trial_type_mask]

    if shuffle:
        stops_location_cm = np.random.uniform(low=0, high=track_length, size=len(stops_location_cm))

    number_of_bins = track_length
    number_of_trials = len(tt_trial_numbers)
    stop_counts = np.zeros((number_of_trials, number_of_bins))
    stop_counts_og_shape = np.shape(stop_counts)
    for i, tn in enumerate(tt_trial_numbers):
        stop_locations_on_trial = stops_location_cm[stop_trial_numbers == tn]
        stop_in_trial_bins, bin_edges = np.histogram(stop_locations_on_trial, bins=track_length, range=[0,track_length])
        stop_counts[i,:] = stop_in_trial_bins

    stop_counts = stop_counts.flatten()
    if np.sum(stop_counts)>0:
        stop_counts = convolve(stop_counts, gauss_kernel)
    stop_counts = stop_counts.reshape(stop_counts_og_shape)

    average_stops = np.nanmean(stop_counts, axis=0)
    average_stops_se = stats.sem(stop_counts, axis=0, nan_policy="omit")

    #average_stops = convolve(average_stops, gauss_kernel)
    #average_stops_se = convolve(average_stops_se, gauss_kernel)

    #bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    bin_centres = np.arange(0.5, track_length+0.5, 1)
    return stop_counts, bin_centres
    #return average_stops, average_stops_se, bin_centres

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

def delete_unused_columns(spike_data):
    #print(list(spike_data))
    for column in ['ML_SNRs_all_beaconed', 'ML_Freqs_all_beaconed', 'ML_SNRs_all_nonbeaconed', 'ML_Freqs_all_nonbeaconed',
                   'ML_SNRs_all_probe', 'ML_Freqs_all_probe', 'ML_SNRs_beaconed_hits', 'ML_Freqs_beaconed_hits',
                   'ML_SNRs_nonbeaconed_hits', 'ML_Freqs_nonbeaconed_hits', 'ML_SNRs_probe_hits',
                   'ML_Freqs_probe_hits', 'ML_SNRs_all_hits', 'ML_Freqs_all_hits', 'ML_SNRs_beaconed_tries',
                   'ML_Freqs_beaconed_tries', 'ML_SNRs_nonbeaconed_tries', 'ML_Freqs_nonbeaconed_tries',
                   'ML_SNRs_probe_tries', 'ML_Freqs_probe_tries', 'ML_SNRs_all_tries', 'ML_Freqs_all_tries',
                   'ML_SNRs_beaconed_misses', 'ML_Freqs_beaconed_misses', 'ML_SNRs_nonbeaconed_misses',
                   'ML_Freqs_nonbeaconed_misses', 'ML_SNRs_probe_misses', 'ML_Freqs_probe_misses',
                   'ML_SNRs_all_misses', 'ML_Freqs_all_misses', 'ML_SNRs_allo', 'ML_Freqs_allo',
                   'ML_SNRs_all_beaconed_allo', 'ML_Freqs_all_beaconed_allo', 'ML_SNRs_all_nonbeaconed_allo',
                   'ML_Freqs_all_nonbeaconed_allo', 'ML_SNRs_all_probe_allo', 'ML_Freqs_all_probe_allo',
                   'ML_SNRs_beaconed_hits_allo', 'ML_Freqs_beaconed_hits_allo', 'ML_SNRs_nonbeaconed_hits_allo',
                   'ML_Freqs_nonbeaconed_hits_allo', 'ML_SNRs_probe_hits_allo', 'ML_Freqs_probe_hits_allo',
                   'ML_SNRs_all_hits_allo', 'ML_Freqs_all_hits_allo', 'ML_SNRs_beaconed_tries_allo',
                   'ML_Freqs_beaconed_tries_allo', 'ML_SNRs_nonbeaconed_tries_allo', 'ML_Freqs_nonbeaconed_tries_allo',
                   'ML_SNRs_probe_tries_allo', 'ML_Freqs_probe_tries_allo', 'ML_SNRs_all_tries_allo',
                   'ML_Freqs_all_tries_allo', 'ML_SNRs_beaconed_misses_allo', 'ML_Freqs_beaconed_misses_allo',
                   'ML_SNRs_nonbeaconed_misses_allo', 'ML_Freqs_nonbeaconed_misses_allo', 'ML_SNRs_probe_misses_allo',
                   'ML_Freqs_probe_misses_allo', 'ML_SNRs_all_misses_allo', 'ML_Freqs_all_misses_allo', 'ML_SNRs_ego',
                   'ML_Freqs_ego', 'ML_SNRs_all_beaconed_ego', 'ML_Freqs_all_beaconed_ego', 'ML_SNRs_all_nonbeaconed_ego',
                   'ML_Freqs_all_nonbeaconed_ego', 'ML_SNRs_all_probe_ego', 'ML_Freqs_all_probe_ego',
                   'ML_SNRs_beaconed_hits_ego', 'ML_Freqs_beaconed_hits_ego', 'ML_SNRs_nonbeaconed_hits_ego',
                   'ML_Freqs_nonbeaconed_hits_ego', 'ML_SNRs_probe_hits_ego', 'ML_Freqs_probe_hits_ego',
                   'ML_SNRs_all_hits_ego', 'ML_Freqs_all_hits_ego', 'ML_SNRs_beaconed_tries_ego',
                   'ML_Freqs_beaconed_tries_ego', 'ML_SNRs_nonbeaconed_tries_ego', 'ML_Freqs_nonbeaconed_tries_ego',
                   'ML_SNRs_probe_tries_ego', 'ML_Freqs_probe_tries_ego', 'ML_SNRs_all_tries_ego',
                   'ML_Freqs_all_tries_ego', 'ML_SNRs_beaconed_misses_ego', 'ML_Freqs_beaconed_misses_ego',
                   'ML_SNRs_nonbeaconed_misses_ego', 'ML_Freqs_nonbeaconed_misses_ego', 'ML_SNRs_probe_misses_ego',
                   'ML_Freqs_probe_misses_ego', 'ML_SNRs_all_misses_ego', 'ML_Freqs_all_misses_ego',
                   'avg_correlations_hmt_by_trial_type', 'field_realignments_hmt_by_trial_type',
                   'n_pi_trials_by_hmt', 'mean_fr_tt_all_hmt_all', 'mean_fr_tt_all_hmt_hit',
                   'mean_fr_tt_all_hmt_miss', 'mean_fr_tt_all_hmt_try', 'mean_fr_tt_0_hmt_all',
                   'mean_fr_tt_0_hmt_hit', 'mean_fr_tt_0_hmt_miss', 'mean_fr_tt_0_hmt_try',
                   'mean_fr_tt_1_hmt_all', 'mean_fr_tt_1_hmt_hit', 'mean_fr_tt_1_hmt_miss',
                   'mean_fr_tt_1_hmt_try', 'mean_fr_tt_2_hmt_all', 'mean_fr_tt_2_hmt_hit',
                   'mean_fr_tt_2_hmt_miss', 'mean_fr_tt_2_hmt_try',
                   'allo_minus_ego_hit_proportions_b', 'allo_minus_ego_try_proportions_b', 'allo_minus_ego_miss_proportions_b',
                   'allo_minus_ego_hit_proportions_nb', 'allo_minus_ego_try_proportions_nb', 'allo_minus_ego_miss_proportions_nb',
                   'ML_SNRs_by_trial_number', 'ML_Freqs_by_trial_number',]:
        if column in list(spike_data):
            del spike_data[column]
    return spike_data

def add_trials(spike_data, processed_position_data):
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


def calculate_spatial_information(spatial_firing, position_data, track_length):

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

    position_heatmap = np.zeros(track_length)
    for x in np.arange(track_length):
        bin_occupancy = len(position_data[(position_data["x_position_cm"] > x) &
                                                (position_data["x_position_cm"] <= x+1)])
        position_heatmap[x] = bin_occupancy
    position_heatmap = position_heatmap*np.diff(position_data["time_seconds"])[-1] # convert to real time in seconds
    occupancy_probability_map = position_heatmap/np.sum(position_heatmap) # Pj

    vr_bin_size_cm = settings.vr_bin_size_cm
    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/vr_bin_size_cm)

    spatial_information_scores = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

        mean_firing_rate = cluster_df.iloc[0]["number_of_spikes"]/np.sum(len(position_data)*np.diff(position_data["time_seconds"])[-1]) # λ
        spikes, _ = np.histogram(np.array(cluster_df['x_position_cm'].iloc[0]), bins=track_length, range=(0,track_length))
        rates = spikes/position_heatmap
        rates = convolve(rates, gauss_kernel)
        mrate = mean_firing_rate
        index = rates>0
        Ispike = np.sum(occupancy_probability_map[index] * (rates[index]/mrate) * np.log2(rates[index]/mrate))
        if Ispike < 0:
            print("hello")

        spatial_information_scores.append(Ispike)

    spatial_firing["spatial_information_score"] = spatial_information_scores

    return spatial_firing


def process_recordings(vr_recording_path_list, of_recording_path_list):
    vr_recording_path_list.sort()
    #vr_recording_path_list = vr_recording_path_list[::-1]
    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            output_path = recording+'/'+settings.sorterName
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            if len(spike_data) != 0:
                spike_data = delete_unused_columns(spike_data)
                # remake the spike locations and firing rate maps
                #raw_position_data, position_data = syncronise_position_data(recording, get_track_length(recording))
                #spike_data = add_position_x(spike_data, raw_position_data)
                #spike_data = add_trial_number(spike_data, raw_position_data)
                #spike_data = add_trial_type(spike_data, raw_position_data)
                #spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length=get_track_length(recording), smoothen=True)
                #spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length=get_track_length(recording), smoothen=False)
                #spike_data = bin_fr_in_time(spike_data, raw_position_data, smoothen=True)
                #spike_data = bin_fr_in_time(spike_data, raw_position_data, smoothen=False)
                #spike_data = add_displayed_peak_firing(spike_data)

                #
                #spike_data = calculate_spatial_information(spike_data, position_data, track_length=get_track_length(recording))

                # BEHAVIOURAL
                processed_position_data = add_avg_track_speed(processed_position_data, track_length=get_track_length(recording))
                processed_position_data, _ = add_hit_miss_try3(processed_position_data, track_length=get_track_length(recording))
                spike_data = add_percentage_hits(spike_data, processed_position_data)
                spike_data = add_n_PI_trial(spike_data, processed_position_data)
                spike_data = add_stops(spike_data, processed_position_data, track_length=get_track_length(recording))
                spike_data = add_trials(spike_data, processed_position_data)

                # MOVING LOMB PERIODOGRAMS
                spike_data = calculate_moving_lomb_scargle_periodogram(spike_data, processed_position_data, track_length=get_track_length(recording))
                #spike_data = analyse_lomb_powers(spike_data, processed_position_data)
                #spike_data = add_lomb_classifier(spike_data)

                # Rolling classifications
                #spike_data = add_rolling_stats_shuffled_blocks(spike_data, track_length=get_track_length(recording))
                #spike_data = add_rolling_stats_shuffled_test(spike_data, processed_position_data, track_length=get_track_length(recording))
                #spike_data = add_rolling_stats_encoding_x(spike_data, track_length=get_track_length(recording))
                spike_data = add_rolling_stats(spike_data, track_length=get_track_length(recording))
                #spike_data = add_rolling_stats_hmt(spike_data, processed_position_data) # requires add_rolling_stats
                spike_data = add_coding_by_trial_number(spike_data, processed_position_data)
                spike_data = add_rolling_stats_percentage_hits(spike_data, processed_position_data) # requires add_rolling_stats

                # Joint activity analysis
                #spike_data = add_realignement_shifts(spike_data=spike_data, processed_position_data=processed_position_data, track_length=get_track_length(recording))

                # FIRING AND BEHAVIOURAL PLOTTING
                #plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=get_track_length(recording),plot_trials=["beaconed", "non_beaconed", "probe"])
                #plot_firing_rate_maps_per_trial(spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=get_track_length(recording))
                #plot_stops_on_track(processed_position_data, output_path, track_length=get_track_length(recording))
                #plot_stop_histogram(processed_position_data, output_path, track_length=get_track_length(recording))

                spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
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
    vr_path_list = []
    of_path_list = []
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])

    #vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D18_2021-06-02_10-36-39']
    #vr_path_list = ["/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D23_2020-11-28_15-13-28",
    #                "/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D18_2020-11-21_14-29-49",
    #                "/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D22_2020-11-27_15-01-24"]
    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36']
    #vr_path_list = ['/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D44_2021-07-08_12-03-21']
    #vr_path_list = ['/mnt/datastore/Harry/cohort6_july2020/vr/M1_D5_2020-08-07_14-27-26','/mnt/datastore/Harry/cohort8_may2021/vr/M13_D27_2021-06-15_11-43-42','/mnt/datastore/Harry/cohort8_may2021/vr/M11_D17_2021-06-01_10-36-53','/mnt/datastore/Harry/cohort8_may2021/vr/M11_D14_2021-05-27_10-34-15',
    #                '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D17_2021-06-01_11-45-20','/mnt/datastore/Harry/cohort8_may2021/vr/M11_D15_2021-05-28_10-42-15','/mnt/datastore/Harry/cohort8_may2021/vr/M11_D12_2021-05-25_09-49-23','/mnt/datastore/Harry/cohort8_may2021/vr/M11_D3_2021-05-12_09-37-41',
    #                '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D16_2021-05-31_10-21-05', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D13_2021-05-26_09-46-36']
    #vr_path_list = ["/mnt/datastore/Harry/cohort8_may2021/vr/M11_D22_2021-06-08_10-55-28"]
    #vr_path_list=["/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D21_2021-06-07_10-26-21"]
    #vr_path_list= ["/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D14_2021-05-27_10-34-15"]
    #vr_path_list=["/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D11_2020-08-17_14-57-20"]
    process_recordings(vr_path_list, of_path_list)


    print("look now")

if __name__ == '__main__':
    main()
