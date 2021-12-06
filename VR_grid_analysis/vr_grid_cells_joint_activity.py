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
from Edmond.VR_grid_analysis.vr_grid_cells import add_lomb_classifier
from joblib import Parallel, delayed
import multiprocessing
import open_ephys_IO
warnings.filterwarnings('ignore')
from scipy.stats.stats import pearsonr
from scipy.stats import shapiro
import samplerate
plt.rc('axes', linewidth=1)

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

def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array

def get_track_length(recording_path):
    parameter_file_path = control_sorting_analysis.get_tags_parameter_file(recording_path)
    stop_threshold, track_length, cue_conditioned_goal = PostSorting.post_process_sorted_data_vr.process_running_parameter_tag(parameter_file_path)
    return track_length

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

def get_max_SNR(spatial_frequency, powers):
    max_SNR = np.abs(np.max(powers)/np.min(powers))
    max_SNR = powers[np.argmax(powers)]
    max_SNR_freq = spatial_frequency[np.argmax(powers)]
    return max_SNR, max_SNR_freq


def plot_joint_cell_cross_correlations(spike_data, output_path):
    spike_data = add_lomb_classifier(spike_data)
    print('plotting joint cell correlations...')
    save_path = output_path + '/Figures/joint_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    cluster_ids = pandas_collumn_to_numpy_array(spike_data["cluster_id"])
    peak_powers = pandas_collumn_to_2d_numpy_array(spike_data["peak_powers"])
    lomb_classes = pandas_collumn_to_numpy_array(spike_data["Lomb_classifier_"])

    cluster_ids = cluster_ids[np.argsort(lomb_classes)]
    peak_powers = peak_powers[np.argsort(lomb_classes)]
    lomb_classes = lomb_classes[np.argsort(lomb_classes)]

    cross_correlation_matrix = np.zeros((len(cluster_ids), len(cluster_ids)))
    for i, cluster_id_i in enumerate(cluster_ids):
        for j, cluster_id_j in enumerate(cluster_ids):
            if (len(peak_powers[i])>1) and (len(peak_powers[j])>1):
                corr = pearsonr(peak_powers[i], peak_powers[j])[0]
                cross_correlation_matrix[i, j] = corr
            else:
                corr = np.nan
                cross_correlation_matrix[i, j] = corr

    fig, ax = plt.subplots()
    im= ax.imshow(cross_correlation_matrix, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cluster_ids)))
    ax.set_yticks(np.arange(len(cluster_ids)))
    ax.set_yticklabels(cluster_ids)
    ax.set_xticklabels(cluster_ids)

    colors = []
    for i in range(len(lomb_classes)):
        colors.append(get_lomb_class_color(lomb_classes[i]))

    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)

    ax.set_ylabel("Cluster ID", fontsize=5)
    ax.set_xlabel("Cluster IDs", fontsize=5)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()
    fig.colorbar(im, ax=ax)
    plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + '_joint_peak_powers_cross_correlations.png', dpi=300)
    plt.close()
    return

def plot_joint_cell_correlations(spike_data, of_spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length):
    spike_data = add_lomb_classifier(spike_data)
    print('plotting joint cell correlations...')
    save_path = output_path + '/Figures/joint_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    gauss_kernel_peak_powers = Gaussian1DKernel(stddev=30)

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

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)


    peak_powers = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        cluster_spike_data_of = of_spike_data[of_spike_data["cluster_id"] == cluster_id]

        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        firing_locations_cluster = np.array(cluster_spike_data["x_position_cm"].iloc[0])
        firing_trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])
        lomb_class = cluster_spike_data["Lomb_classifier_"].iloc[0]
        #cell_type = cluster_spike_data_of["classifier"].iloc[0]
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
                power[np.isnan(power)] = 0
                powers.append(power.tolist())
                centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
            powers = np.array(powers)
            centre_trials = np.round(np.array(centre_distances)).astype(np.int64)

            avg_power = np.nanmean(powers, axis=0)
            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
            max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
            max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))

            peak_powers_cluster = powers[:, np.argmin(np.abs(frequency-max_SNR_freq))]
            peak_powers_cluster = convolve(peak_powers_cluster, gauss_kernel_peak_powers)
            peak_powers_cluster = moving_sum(peak_powers_cluster, window=20)/20
            peak_powers_cluster = np.append(peak_powers_cluster, np.zeros(len(centre_trials)-len(peak_powers_cluster)))

            peak_powers_cluster = min_max_normalize(peak_powers_cluster)
            peak_powers.append(peak_powers_cluster.tolist())

            ax.plot(centre_trials, peak_powers_cluster, color=get_lomb_class_color(lomb_class), alpha=0.3, linewidth=1)
        else:
            peak_powers.append(np.nan)

    spike_data["peak_powers"] = peak_powers
    plt.xlabel('Centre Trial', fontsize=20, labelpad = 10)
    plt.ylabel('Peak Power', fontsize=20, labelpad = 10)
    plt.xlim(0,max(centre_trials))
    plt.ylim(bottom=0)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_joint_peak_powers.png', dpi=300)
    plt.close()
    return spike_data

def get_lomb_class_color(lomb_class):
    if lomb_class == "Position":
        return "turquoise"
    elif lomb_class == "Distance":
        return "orange"
    elif lomb_class == "Null":
        return "gray"
    else:
        return ""

def get_cell_type_color(cell_type):
    if cell_type == "G":
        return "solid"
    else:
        return "dashed"

def process_recordings(vr_recording_path_list, of_recording_path_list):

    for recording in vr_recording_path_list:
        print("processing ", recording)
        #recording = "/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36"
        #recording = "/mnt/datastore/Harry/cohort8_may2021/vr/M11_D19_2021-06-03_10-50-41"
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
            processed_position_data, _ = add_hit_miss_try(processed_position_data, track_length=get_track_length(recording))

            if paired_recording is not None:
                of_spike_data = pd.read_pickle(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl")
                spike_data = plot_joint_cell_correlations(spike_data, of_spike_data, processed_position_data, position_data, raw_position_data, output_path, get_track_length(recording))
                plot_joint_cell_cross_correlations(spike_data, output_path)

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


    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D11_2021-05-24_10-00-53', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D12_2021-05-25_09-49-23',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D13_2021-05-26_09-46-36', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D15_2021-05-28_10-42-15',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D16_2021-05-31_10-21-05', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D17_2021-06-01_10-36-53',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D18_2021-06-02_10-36-39', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D22_2021-06-08_10-55-28',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D26_2021-06-14_10-34-14', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D29_2021-06-17_10-35-48',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D30_2021-06-18_10-46-48', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D33_2021-06-23_11-08-03',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D35_2021-06-25_12-02-52', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D37_2021-06-29_11-50-02', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D38_2021-06-30_11-54-56',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D3_2021-05-12_09-37-41', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D41_2021-07-05_12-05-02',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D44_2021-07-08_12-03-21', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D45_2021-07-09_11-39-02',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M12_D6_2021-05-17_10-26-15', '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D17_2021-06-01_11-45-20',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D24_2021-06-10_12-01-54', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D12_2021-05-25_11-03-39',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D15_2021-05-28_12-29-15', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D16_2021-05-31_12-01-35',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D20_2021-06-04_12-20-57', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D27_2021-06-15_12-21-58',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D31_2021-06-21_12-07-01', '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D35_2021-06-25_12-41-16',
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M14_D37_2021-06-29_12-33-24']

    process_recordings(vr_path_list, of_path_list)

    print("look now")

if __name__ == '__main__':
    main()
