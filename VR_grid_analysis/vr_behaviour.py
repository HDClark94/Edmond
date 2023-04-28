import numpy as np
import pandas as pd
import numbers
import matplotlib
import PostSorting.parameters
import PostSorting.vr_stop_analysis
from statsmodels.stats.anova import AnovaRM
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
import matplotlib.colors as colors
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import cycler
from sklearn.cluster import DBSCAN
from scipy import stats
import matplotlib.cm as cm
from scipy import signal
from astropy.convolution import convolve, Gaussian1DKernel
import os
import traceback
import warnings
import matplotlib.ticker as ticker
import sys
import Edmond.plot_utility2
import Edmond.VR_grid_analysis.vr_grid_stability_plots
import Edmond.VR_grid_analysis.hit_miss_try_firing_analysis
from Edmond.VR_grid_analysis.vr_grid_cells import *
import settings
from scipy import stats
import Edmond.VR_grid_analysis.analysis_settings as Settings
import matplotlib.pylab as plt
import matplotlib as mpl
import control_sorting_analysis
import PostSorting.post_process_sorted_data_vr
warnings.filterwarnings('ignore')
from Edmond.VR_grid_analysis.remake_position_data import syncronise_position_data
from scipy.stats.stats import pearsonr

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

def add_time_elapsed_collumn(position_data):
    time_seconds = np.array(position_data['time_seconds'].to_numpy())
    time_elapsed = np.diff(time_seconds)
    time_elapsed = np.append(time_elapsed[0], time_elapsed)
    position_data["time_in_bin_seconds"] = time_elapsed.tolist()
    return position_data

def assay_maximum_spatial_frequency_pairwise(spike_data, processed_position_data, position_data, raw_position_data, output_path, track_length, suffix="", GaussianKernelSTD_ms=5, fr_integration_window=2):
    print('plotting pairwise frequency assay...')
    save_path = output_path + '/Figures/pairwise_frequency_assay'
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
    corrs = []
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
            elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])
            trial_numbers_by_bin=((0.5*(bin_edges[1:]+bin_edges[:-1])//track_length)+1).astype(np.int32)
            gauss_kernel = Gaussian1DKernel(stddev=1)

            # remove nan values that coincide with start and end of the track before convolution
            fr[fr==np.inf] = np.nan
            fr[np.isnan(fr)] = 0

            fr = convolve(fr, gauss_kernel)
            fr = moving_sum(fr, window=2)/2
            fr = np.append(fr, np.zeros(len(elapsed_distance)-len(fr)))

            # make and apply the set mask
            set_mask = np.isin(trial_numbers_by_bin, trial_number_to_use)
            fr = fr[set_mask]
            elapsed_distance = elapsed_distance[set_mask]

            step = 0.01
            frequency = np.arange(0.1, 10+step, step)
            pairwise_correlations = []
            for freq_j in frequency:
                #print(freq_j)
                period = 1/freq_j
                realligned_elapsed_distances = elapsed_distance/(period*track_length)
                realligned_track_positions = realligned_elapsed_distances%1

                # pad array with nans
                new_track_length =int(track_length*period)
                if len(fr)%new_track_length == 0:
                    fr = fr
                else:
                    n_samples_to_remove = len(fr)%new_track_length
                    fr = fr[:-n_samples_to_remove]

                # take a 2d array for each new track lap.
                n_rows = int(len(fr)/new_track_length)
                fr_reshape = fr.reshape(n_rows, int(len(fr)/n_rows))

                trial_correlations = []
                for i in range(len(fr_reshape)-1):
                    pearson_r = stats.pearsonr(fr_reshape[i].flatten(),fr_reshape[i+1].flatten())
                    trial_correlations.append(pearson_r[0])
                pairwise_score = np.nanmean(np.array(trial_correlations))
                pairwise_correlations.append(pairwise_score)
            pairwise_correlations=np.array(pairwise_correlations)
            best_freq = frequency[np.nanargmax(pairwise_correlations)]
            corr_at_freq = pairwise_correlations[np.nanargmax(pairwise_correlations)]

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.plot(frequency, pairwise_correlations, color="blue")
            corr_at_freq_text = "Corr: "+ str(np.round(corr_at_freq, 2))
            best_freq_text = "Freq: "+ str(np.round(best_freq, 1))
            ax.text(0.9, 0.9, corr_at_freq_text, ha='right', va='center', transform=ax.transAxes, fontsize=10)
            ax.text(0.9, 0.8, best_freq_text, ha='right', va='center', transform=ax.transAxes, fontsize=10)
            plt.ylabel('Pairwise Correlation', fontsize=20, labelpad = 10)
            plt.xlabel('Spatial Frequency', fontsize=20, labelpad = 10)
            plt.xlim(0,max(frequency))
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_pairwise_freqency_assay_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
            plt.close()

            '''
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
            '''

            freqs.append(best_freq)
            corrs.append(corr_at_freq)
            #SNR_thresholds.append(SNR_threshold)
            #freq_thresholds.append(freq_threshold)
        else:
            freqs.append(np.nan)
            corrs.append(np.nan)
            #SNR_thresholds.append(np.nan)
            #freq_thresholds.append(np.nan)

    spike_data["pairwise_freq"+suffix] = freqs
    spike_data["pairwise_corr"+suffix] = corrs
    #spike_data["shuffleSNR"+suffix] = SNR_thresholds
    #spike_data["shufflefreqs"+suffix] = freq_thresholds #TODO change these collumn names, they're misleading
    return spike_data

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

def get_mouse_color(mouse_array, mouse_ids, mouse_colors):
    colors=[]
    for i in range(len(mouse_array)):
        mouse_mask = np.array(mouse_ids) == mouse_array[i]
        color = mouse_colors[mouse_mask]
        colors.append(color)
    return colors


def get_reward_colors(hmt):
    colors=[]
    for i in range(len(hmt)):
        if (hmt[i] == "hit"):
            colors.append("green")
        elif (hmt[i] == "try"):
            colors.append("orange")
        elif (hmt[i] == "miss"):
            colors.append("red")
        elif (hmt[i] == "rejected"):
            colors.append("grey")
        else:
            colors.append("white")
    return colors

def get_colors_hmt(hmt):
    colors = []
    for i in range(len(hmt)):
        if (hmt[i] == "hit"):
            colors.append("green")
        elif (hmt[i] == "miss"):
            colors.append("red")
        elif (hmt[i] == "try"):
            colors.append("orange")
        elif (hmt[i] == "rejected"):
            colors.append("grey")
    return colors

def get_reward_colors_hmt(rewarded, TI):
    colors = []
    for i in range(len(rewarded)):
        if (rewarded[i] == 1):
            colors.append("green")
        elif (rewarded[i] == 0) and (TI[i]<1):
            colors.append("orange")
        elif (rewarded[i] == 0) and (TI[i]<1):
            colors.append("red")
        else:
            colors.append("red")
    return colors

def plot_trial_discriminant_schematic(save_path):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.fill([0, 120, 120, 0], [0,   0, 120, 0], color="plum")
    ax.fill([0,   0, 120, 0], [0, 120, 120, 0], color="lavender")
    ax.plot([0,120], [0,120], linestyle="dashed", color="black",linewidth=2)
    ax.set_ylabel("Avg Trial Speed in RZ", fontsize=20)
    ax.set_xlabel("Avg Trial Speed on track", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    ax.text(x=20, y=90, s="TI < 1", fontsize=35)
    ax.text(x=60, y=20, s="TI > 1", fontsize=35)
    ax.set_xticks([0,20,40,60,80,100,120])
    ax.set_yticks([0,20,40,60,80,100,120])
    ax.set_ylim(bottom=0, top=120)
    ax.set_xlim(left=0, right=120)
    plt.subplots_adjust(right=0.95, top=0.95)
    plt.savefig(save_path + '/task_discriminant_schematic.png', dpi=300)
    plt.close()

def plot_trial_discriminant_histogram(processed_position_data, save_path):
    discriminants = pandas_collumn_to_numpy_array(processed_position_data["RZ_stop_bias"])
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(discriminants)
    ax.set_ylabel("Trial Counts", fontsize=20)
    ax.set_xlabel("TI", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.subplots_adjust(right=0.95, top=0.95)
    plt.savefig(save_path + '/task_discriminant_histogram.png', dpi=300)
    plt.close()


def plot_trial_speeds(processed_position_data, save_path):
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]
    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    processed_position_data = processed_position_data.sample(frac=1).reset_index()

    avg_RZ_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_in_RZ"])
    avg_track_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_on_track"])
    hmt = pandas_collumn_to_numpy_array(processed_position_data["hit_miss_try"])
    rewarded_colors = get_reward_colors(hmt)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(avg_track_speed, avg_RZ_speed, color=rewarded_colors, marker="x", alpha=0.3)
    ax.plot([0,150], [0,150], linestyle="dashed", color="black")
    ax.set_ylabel("Avg Trial Speed in RZ", fontsize=25, labelpad=10)
    ax.set_xlabel("Avg Trial Speed on track", fontsize=25, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    ax.set_xticks([0,50,100,150])
    ax.set_yticks([0,50,100,150])
    ax.set_ylim(bottom=0, top=150)
    ax.set_xlim(left=0, right=150)
    plt.subplots_adjust(right=0.95, top=0.95)
    plt.savefig(save_path + '/RZ_speed_vs_track_speed.png', dpi=300)
    plt.close()

    mouse_array = pandas_collumn_to_numpy_array(processed_position_data["mouse_id"])
    rewarded_colors = get_mouse_color(mouse_array, mouse_ids, colors)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(avg_track_speed, avg_RZ_speed, color=rewarded_colors, marker="x", alpha=0.3)
    ax.plot([0,150], [0,150], linestyle="dashed", color="black")
    ax.set_ylabel("Avg Trial Speed in RZ", fontsize=25, labelpad=10)
    ax.set_xlabel("Avg Trial Speed on track", fontsize=25, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    ax.set_xticks([0,50,100,150])
    ax.set_yticks([0,50,100,150])
    ax.set_ylim(bottom=0, top=150)
    ax.set_xlim(left=0, right=150)
    plt.subplots_adjust(right=0.95, top=0.95)
    plt.savefig(save_path + '/RZ_speed_vs_track_speed_by_mouse.png', dpi=300)
    plt.close() 
    plt.close()


def plot_trial_speeds_hmt(processed_position_data, save_path):
    # remove low speed trials
    #processed_position_data = processed_position_data[(processed_position_data["avg_speed_on_track"] > 20)]
    avg_RZ_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_in_RZ"])
    avg_track_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_on_track"])
    hmt = pandas_collumn_to_numpy_array(processed_position_data["hit_miss_try"])
    rewarded_colors = get_colors_hmt(hmt)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(avg_track_speed, avg_RZ_speed, color=rewarded_colors, edgecolor=rewarded_colors, marker="o", alpha=0.3)
    ax.plot([0,120], [0,120], linestyle="dashed", color="black")
    ax.set_ylabel("Avg Trial Speed in RZ", fontsize=25, labelpad=10)
    ax.set_xlabel("Avg Trial Speed on track", fontsize=25, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    ax.set_xticks([0,20,40,60,80,100])
    ax.set_yticks([0,20,40,60,80,100])
    ax.set_ylim(bottom=0, top=120)
    ax.set_xlim(left=0, right=120)
    plt.subplots_adjust(right=0.95, top=0.95)
    plt.savefig(save_path + '/RZ_speed_vs_track_speed_hmt.png', dpi=300)
    plt.close()

def plot_hit_avg_speeds_by_block(processed_position_data, save_path):
    start=0; end=199
    processed_position_data_block1 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 0) & (processed_position_data["avg_speed_on_track"] <= 10) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block2 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 10) & (processed_position_data["avg_speed_on_track"] <= 20) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block3 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 20) & (processed_position_data["avg_speed_on_track"] <= 30) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block4 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 30) & (processed_position_data["avg_speed_on_track"] <= 40) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block5 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 40) & (processed_position_data["avg_speed_on_track"] <= 50) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block6 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 50) & (processed_position_data["avg_speed_on_track"] <= 60) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block7 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 60) & (processed_position_data["avg_speed_on_track"] <= 70) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block8 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 70) & (processed_position_data["avg_speed_on_track"] <= 80) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block9 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 80) & (processed_position_data["avg_speed_on_track"] <= 90) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block10 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 90) & (processed_position_data["avg_speed_on_track"] <= 100) & (processed_position_data["rewarded"] == 1)]

    avg_speeds_block1 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block1["speeds_binned_in_space"]), axis=0)
    avg_speeds_block2 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block2["speeds_binned_in_space"]), axis=0)
    avg_speeds_block3 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block3["speeds_binned_in_space"]), axis=0)
    avg_speeds_block4 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block4["speeds_binned_in_space"]), axis=0)
    avg_speeds_block5 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block5["speeds_binned_in_space"]), axis=0)
    avg_speeds_block6 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block6["speeds_binned_in_space"]), axis=0)
    avg_speeds_block7 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block7["speeds_binned_in_space"]), axis=0)
    avg_speeds_block8 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block8["speeds_binned_in_space"]), axis=0)
    avg_speeds_block9 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block9["speeds_binned_in_space"]), axis=0)
    avg_speeds_block10 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block10["speeds_binned_in_space"]), axis=0)

    locations = np.asarray(processed_position_data['position_bin_centres'].iloc[0])

    fig, ax = plt.subplots(figsize=(6,4))
    colors = cm.rainbow(np.linspace(0, 1, 10))
    ax.plot(locations[start:end], avg_speeds_block1[start:end], color=colors[0], label="0-10")
    ax.plot(locations[start:end], avg_speeds_block2[start:end], color=colors[1], label="10-20")
    ax.plot(locations[start:end], avg_speeds_block3[start:end], color=colors[2], label="20-30")
    ax.plot(locations[start:end], avg_speeds_block4[start:end], color=colors[3], label="30-40")
    ax.plot(locations[start:end], avg_speeds_block5[start:end], color=colors[4], label="40-50")
    ax.plot(locations[start:end], avg_speeds_block6[start:end], color=colors[5], label="50-60")
    ax.plot(locations[start:end], avg_speeds_block7[start:end], color=colors[6], label="60-70")
    ax.plot(locations[start:end], avg_speeds_block8[start:end], color=colors[7], label="70-80")
    ax.plot(locations[start:end], avg_speeds_block9[start:end], color=colors[8], label="80-90")
    ax.plot(locations[start:end], avg_speeds_block10[start:end], color=colors[9], label="90-100")
    #ax.legend(title='Avg Track Speed (cm/s)')
    ax.set_ylabel("Speed (cm/s)", fontsize=25, labelpad=10)
    ax.set_xlabel("Location (cm)", fontsize=25, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    style_track_plot(ax, 200)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    Edmond.plot_utility2.style_vr_plot(ax, x_max=max(avg_speeds_block10[start:end]))
    fig.tight_layout()
    plt.subplots_adjust(right=0.9)
    ax.set_ylim(bottom=0, top=130)
    ax.set_xlim(left=0, right=200)
    ax.set_yticks([0,50, 100])
    plt.savefig(save_path + '/hits_avg_speeds_by_speed_block.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.bar(5, len(processed_position_data_block1), color=colors[0], width=10)
    ax.bar(15, len(processed_position_data_block2), color=colors[1], width=10)
    ax.bar(25, len(processed_position_data_block3), color=colors[2], width=10)
    ax.bar(35, len(processed_position_data_block4), color=colors[3], width=10)
    ax.bar(45, len(processed_position_data_block5), color=colors[4], width=10)
    ax.bar(55, len(processed_position_data_block6), color=colors[5], width=10)
    ax.bar(65, len(processed_position_data_block7), color=colors[6], width=10)
    ax.bar(75, len(processed_position_data_block8), color=colors[7], width=10)
    ax.bar(85, len(processed_position_data_block9), color=colors[8], width=10)
    ax.bar(95, len(processed_position_data_block10), color=colors[9], width=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.axvline(x=Settings.track_speed_threshold, linestyle="dashed", color="black")
    tick_spacing = 1000
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    ax.set_ylabel("Trial count", fontsize=25, labelpad=5)
    ax.set_xlabel("Avg Track Speed (cm/s)", fontsize=25, labelpad=10)
    plt.savefig(save_path + '/distribution_of_hits_avg_track_speed.png', dpi=300)
    plt.close()

    # make the same plot but for the misses
    processed_position_data_block1 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 0) & (processed_position_data["avg_speed_on_track"] <= 10) & (processed_position_data["rewarded"] == 0)]
    processed_position_data_block2 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 10) & (processed_position_data["avg_speed_on_track"] <= 20) & (processed_position_data["rewarded"] == 0)]
    processed_position_data_block3 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 20) & (processed_position_data["avg_speed_on_track"] <= 30) & (processed_position_data["rewarded"] == 0)]
    processed_position_data_block4 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 30) & (processed_position_data["avg_speed_on_track"] <= 40) & (processed_position_data["rewarded"] == 0)]
    processed_position_data_block5 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 40) & (processed_position_data["avg_speed_on_track"] <= 50) & (processed_position_data["rewarded"] == 0)]
    processed_position_data_block6 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 50) & (processed_position_data["avg_speed_on_track"] <= 60) & (processed_position_data["rewarded"] == 0)]
    processed_position_data_block7 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 60) & (processed_position_data["avg_speed_on_track"] <= 70) & (processed_position_data["rewarded"] == 0)]
    processed_position_data_block8 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 70) & (processed_position_data["avg_speed_on_track"] <= 80) & (processed_position_data["rewarded"] == 0)]
    processed_position_data_block9 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 80) & (processed_position_data["avg_speed_on_track"] <= 90) & (processed_position_data["rewarded"] == 0)]
    processed_position_data_block10 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 90) & (processed_position_data["avg_speed_on_track"] <= 100) & (processed_position_data["rewarded"] == 0)]
    avg_speeds_block1 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block1["speeds_binned_in_space"]), axis=0)
    avg_speeds_block2 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block2["speeds_binned_in_space"]), axis=0)
    avg_speeds_block3 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block3["speeds_binned_in_space"]), axis=0)
    avg_speeds_block4 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block4["speeds_binned_in_space"]), axis=0)
    avg_speeds_block5 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block5["speeds_binned_in_space"]), axis=0)
    avg_speeds_block6 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block6["speeds_binned_in_space"]), axis=0)
    avg_speeds_block7 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block7["speeds_binned_in_space"]), axis=0)
    avg_speeds_block8 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block8["speeds_binned_in_space"]), axis=0)
    avg_speeds_block9 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block9["speeds_binned_in_space"]), axis=0)
    avg_speeds_block10 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block10["speeds_binned_in_space"]), axis=0)


    fig, ax = plt.subplots(figsize=(6,4))
    colors = cm.rainbow(np.linspace(0, 1, 10))
    ax.plot(locations[start:end], avg_speeds_block1[start:end], color=colors[0], label="0-10")
    ax.plot(locations[start:end], avg_speeds_block2[start:end], color=colors[1], label="10-20")
    ax.plot(locations[start:end], avg_speeds_block3[start:end], color=colors[2], label="20-30")
    ax.plot(locations[start:end], avg_speeds_block4[start:end], color=colors[3], label="30-40")
    ax.plot(locations[start:end], avg_speeds_block5[start:end], color=colors[4], label="40-50")
    ax.plot(locations[start:end], avg_speeds_block6[start:end], color=colors[5], label="50-60")
    ax.plot(locations[start:end], avg_speeds_block7[start:end], color=colors[6], label="60-70")
    ax.plot(locations[start:end], avg_speeds_block8[start:end], color=colors[7], label="70-80")
    ax.plot(locations[start:end], avg_speeds_block9[start:end], color=colors[8], label="80-90")
    ax.plot(locations[start:end], avg_speeds_block10[start:end], color=colors[9], label="90-100")
    #ax.legend(title='Avg Track Speed (cm/s)')
    ax.set_ylabel("Speed (cm/s)", fontsize=25, labelpad=10)
    ax.set_xlabel("Location (cm)", fontsize=25, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    style_track_plot(ax, 200)
    Edmond.plot_utility2.style_vr_plot(ax, x_max=max(avg_speeds_block10[start:end]))
    fig.tight_layout()
    plt.subplots_adjust(right=0.9)
    ax.set_ylim(bottom=0, top=130)
    ax.set_xlim(left=0, right=200)
    ax.set_yticks([0,50, 100])
    plt.savefig(save_path + '/misses_avg_speeds_by_speed_block.png', dpi=300)
    plt.close()


    fig, ax = plt.subplots(figsize=(6,6))
    ax.bar(5, len(processed_position_data_block1), color=colors[0], width=10)
    ax.bar(15, len(processed_position_data_block2), color=colors[1], width=10)
    ax.bar(25, len(processed_position_data_block3), color=colors[2], width=10)
    ax.bar(35, len(processed_position_data_block4), color=colors[3], width=10)
    ax.bar(45, len(processed_position_data_block5), color=colors[4], width=10)
    ax.bar(55, len(processed_position_data_block6), color=colors[5], width=10)
    ax.bar(65, len(processed_position_data_block7), color=colors[6], width=10)
    ax.bar(75, len(processed_position_data_block8), color=colors[7], width=10)
    ax.bar(85, len(processed_position_data_block9), color=colors[8], width=10)
    ax.bar(95, len(processed_position_data_block10), color=colors[9], width=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    tick_spacing = 1000
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.axvline(x=Settings.track_speed_threshold, linestyle="dashed", color="black")
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    ax.set_ylabel("Trial count", fontsize=25, labelpad=5)
    ax.set_xlabel("Avg Track Speed (cm/s)", fontsize=25, labelpad=10)
    plt.savefig(save_path + '/distribution_of_misses_avg_track_speed.png', dpi=300)
    plt.close()

def add_RZ_bias(processed_position_data):
    avg_RZ_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_in_RZ"])
    avg_track_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_on_track"])
    RZ_stop_bias = avg_track_speed/avg_RZ_speed
    processed_position_data["RZ_stop_bias"] =RZ_stop_bias
    return processed_position_data


def compute_p_map(save_path):
    x = np.linspace(0.0001, 120, 1000)
    y = np.linspace(0.0001, 120, 1000)

    TI_map = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            TI_map[i,j] = x[i]/y[j]

    fig, ax = plt.subplots(figsize=(6,6))
    X, Y = np.meshgrid(x, y)
    pcm = ax.pcolormesh(X, Y, TI_map, norm=colors.LogNorm(vmin=0.1, vmax=10), shading="nearest", cmap='jet')
    #ax.set_ylabel("Trial Counts", fontsize=20)
    ax.plot([0,120], [0,120], linestyle="dashed", color="black")
    #ax.set_xlabel("TI", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    #ax.set_ylim(bottom=0)
    #ax.set_xlim(left=0)
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.1, extend="both")
    cbar.set_ticks([0.1,1,10])
    cbar.set_ticklabels(["0.1","1","10"])
    cbar.ax.tick_params(labelsize=20)
    ax.set_xticks([0,120])
    ax.set_yticks([0,120])
    ax.set_ylabel("Avg Trial Speed in RZ", fontsize=20)
    ax.set_xlabel("Avg Trial Speed on track", fontsize=20)
    ax.set_xticklabels([0,120], fontsize=20)
    ax.set_yticklabels([0,120], fontsize=20)
    cbar.set_label('Task Index', rotation=270, fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
    plt.savefig(save_path + '/TI_map.png', dpi=300)
    plt.close()

def get_hmt_color(hmt):
    if hmt=="hit":
        return "green"
    elif hmt=="try":
        return "orange"
    elif hmt =="miss":
        return "red"
    else:
        return "SOMETING IS WRONGG"

def plot_average_hmt_speed_trajectories_by_trial_type_by_mouse(processed_position_data, hmt, save_path):
    start=0;end=199
    hmt_processed = processed_position_data[processed_position_data["hit_miss_try"] == hmt]

    for tt, tt_string in zip([0,1,2], ["b", "nb", "p"]):
        t_processed = hmt_processed[hmt_processed["trial_type"] == tt]

        speed_histogram = plt.figure(figsize=(6,4))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for _, mouse_id in enumerate(np.unique(t_processed["mouse_id"])):
            mouse_processed = t_processed[t_processed["mouse_id"] == mouse_id]
            trajectories = pandas_collumn_to_2d_numpy_array(mouse_processed["speeds_binned_in_space"])
            trajectories_avg = np.nanmean(trajectories, axis=0)[start:end]
            trajectories_sem = np.nanstd(trajectories, axis=0)[start:end]
            locations = np.asarray(processed_position_data['position_bin_centres'].iloc[0])[start:end]
            #ax.fill_between(locations, trajectories_avg-trajectories_sem, trajectories_avg+trajectories_sem, color=get_hmt_color(hmt), alpha=0.3)
            ax.plot(locations, trajectories_avg, label=mouse_id)

        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.legend(title='Mouse')
        plt.ylabel('Speed (cm/s)', fontsize=25, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        tick_spacing = 100
        ax.set_yticks([0, 50, 100])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        Edmond.plot_utility2.style_vr_plot(ax, x_max=115)
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=200)
        plt.savefig(save_path + '/average_speed_trajectory_by_mouse_hmt_'+hmt+"_tt_"+tt_string+'.png', dpi=300)
        plt.close()

def plot_average_hmt_speed_trajectories_by_trial_type(processed_position_data, hmt, save_path):
    start=0;end=199
    hmt_processed = processed_position_data[processed_position_data["hit_miss_try"] == hmt]

    for tt, tt_string in zip([0,1,2], ["b", "nb", "p"]):
        tt_processed = hmt_processed[hmt_processed["trial_type"] == tt]
        trajectories = pandas_collumn_to_2d_numpy_array(tt_processed["speeds_binned_in_space"])
        trajectories_avg = np.nanmean(trajectories, axis=0)[start:end]
        #trajectories_sem = np.nanstd(trajectories, axis=0)[start:end]
        trajectories_sem = stats.sem(trajectories, axis=0, nan_policy="omit")[start:end]
        locations = np.asarray(processed_position_data['position_bin_centres'].iloc[0])[start:end]

        speed_histogram = plt.figure(figsize=(6,4))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.fill_between(locations, trajectories_avg-trajectories_sem, trajectories_avg+trajectories_sem, color=get_hmt_color(hmt), alpha=0.3)
        ax.plot(locations, trajectories_avg, color="black")
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.ylabel('Speed (cm/s)', fontsize=25, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        tick_spacing = 100
        ax.set_yticks([0, 50, 100])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        Edmond.plot_utility2.style_vr_plot(ax, x_max=115)
        plt.subplots_adjust(bottom = 0.2, left=0.2)

        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=200)
        plt.savefig(save_path + '/average_speed_trajectory_'+hmt+"_tt_"+tt_string+'.png', dpi=300)
        plt.close()

def cluster_speed_profiles(processed_position_data, save_path):
    trajectories = pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned_in_space"])
    hmt = pandas_collumn_to_numpy_array(processed_position_data["hit_miss_try"])
    tt = pandas_collumn_to_numpy_array(processed_position_data["trial_type"])

    hmts = []
    for i in range(len(hmt)):
        hmts.append(get_hmt_color(hmt[i]))
    hmts=np.array(hmts)

    tts = []
    for i in range(len(tt)):
        tts.append(get_trial_color(tt[i]))
    tts=np.array(tts)

    for i in range(len(trajectories)):
        trajectories[i] = stats.zscore(trajectories[i], axis=0, ddof=0, nan_policy='omit')
    #trajectories = stats.zscore(trajectories, axis=0, ddof=0, nan_policy='omit')
    z_scored_trajectories = trajectories[:, 30:170]

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(z_scored_trajectories)
    z_scored_trajectories = imp_mean.transform(z_scored_trajectories)

    pca = PCA(n_components=10)
    pca.fit(z_scored_trajectories)
    print(pca.explained_variance_ratio_)
    z_scored_trajectories = pca.transform(z_scored_trajectories)
    z_scored_trajectories = z_scored_trajectories[:, :4]

    xy = z_scored_trajectories
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(xy[:, 0], xy[:, 1], marker="o", alpha=0.3, color=tts)
    plt.savefig(save_path+"/DBSCAN_clusters.png", dpi=300)


    db = DBSCAN(eps=0.5, min_samples=50).fit(z_scored_trajectories)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        if k != -1:
            xy = z_scored_trajectories[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = z_scored_trajectories[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

    plt.savefig(save_path+"/DBSCAN_clusters.png", dpi=300)
    return

def plot_n_trial_per_session_by_mouse(processed_position_data, save_path):
    max_session = 30
    all_behaviour200cm_tracks = processed_position_data[processed_position_data["session_number"] <= max_session]
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]
    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))
    mouse_array = np.zeros((len(mouse_ids), max_session)); mouse_array[:, :] = np.nan

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    mouse_i = 0
    for mouse_id, mouse_color in zip(mouse_ids, colors):
        mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
        n_trials = []
        session_numbers = []
        for session_number in np.unique(mouse_df.session_number):
            session_df = mouse_df[mouse_df["session_number"] == session_number]
            n_trials.append(len(session_df))
            session_numbers.append(session_number)
            mouse_array[mouse_i, session_number-1] = len(session_df)
        mouse_i +=1
        # plot per mouse
        ax.plot(session_numbers, n_trials, '-', label=mouse_id, color=mouse_color)

    plt.ylabel('Number of trials', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    #plt.ylim(0, 100)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/n_trials_by_mouse.png', dpi=200)
    plt.close()

    # do stats test
    df = pd.DataFrame({'mouse_id': np.repeat(np.arange(0,len(mouse_ids)), len(mouse_array[0])),
                       'session_id': np.tile(np.arange(0,max_session),  len(mouse_array)),
                       'n_trials': mouse_array.flatten()})
    df = df.dropna()
    df = reassign_session_numbers(df)
    # Conduct the repeated measures ANOVA
    df = df[df["mouse_id"]!=0]
    df = df[df["mouse_id"]!=4]
    df = df[df["mouse_id"]!=6]
    df = df[df["mouse_id"]!=11]
    df = df[df["session_id"]<25]
    a = AnovaRM(data=df, depvar='n_trials', subject='mouse_id', within=['session_id'], aggregate_func='mean').fit()
    print("repeated measures test for n trials")
    print("p= ", str(a.anova_table["Pr > F"].iloc[0]), ", Num DF= ", str(a.anova_table["Num DF"].iloc[0]), ", Num DF= ", str(a.anova_table["Den DF"].iloc[0]), "F value= ", str(a.anova_table["F Value"].iloc[0]))

    # plot average across mouse
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    nan_mask = ~np.isnan(np.nanmean(mouse_array, axis=0))
    ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(mouse_array, axis=0)-np.nanstd(mouse_array, axis=0))[nan_mask], (np.nanmean(mouse_array, axis=0)+np.nanstd(mouse_array, axis=0))[nan_mask], color="black", alpha=0.3)
    ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(mouse_array, axis=0)[nan_mask], color="black")
    plt.ylabel('Number of trials', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    #plt.ylim(0, 100)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/n_trials_across_animals.png', dpi=200)
    plt.close()
    return


def plot_percentage_trial_per_session_by_mouse_short_plot(processed_position_data, hmt, save_path):

    for tt, tt_string in zip([0,1,2], ["b", "nb", "p"]):
        t_processed = processed_position_data[processed_position_data["trial_type"] == tt]

        speed_histogram = plt.figure(figsize=(8,1.5))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for _, mouse_id in enumerate(np.unique(t_processed["mouse_id"])):
            mouse_processed = t_processed[t_processed["mouse_id"] == mouse_id]

            percent_trials = []
            for session_number in np.unique(mouse_processed["session_number"]):
                session_processed = mouse_processed[mouse_processed["session_number"] == session_number]
                percent_trials.append(100*(len(session_processed[session_processed["hit_miss_try"]=="hit"])/len(session_processed)))

            #ax.scatter(np.unique(mouse_processed["session_number"]), percent_trials, marker="o", label=mouse_id, zorder=10, clip_on=False)
            ax.plot(np.unique(mouse_processed["session_number"]), percent_trials, label=mouse_id, zorder=10, clip_on=False)

        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.legend(title='Mouse')
        plt.ylabel('% trials', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)

        tick_spacing = 50
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([1, 15, 30])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        Edmond.plot_utility2.style_vr_plot(ax, x_max=100)
        plt.subplots_adjust(bottom = 0.3, left = 0.2)
        #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=30)
        plt.savefig(save_path + '/percentage_trials_by_mouse_hmt_'+hmt+"_tt_"+tt_string+'_shortplot.png', dpi=300)
        plt.close()

    return

def plot_percentage_trial_per_session_by_mouse(processed_position_data, hmt, save_path):

    for tt, tt_string in zip([0,1,2], ["b", "nb", "p"]):
        t_processed = processed_position_data[processed_position_data["trial_type"] == tt]

        speed_histogram = plt.figure(figsize=(8,3.3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for _, mouse_id in enumerate(np.unique(t_processed["mouse_id"])):
            mouse_processed = t_processed[t_processed["mouse_id"] == mouse_id]

            percent_trials = []
            for session_number in np.unique(mouse_processed["session_number"]):
                session_processed = mouse_processed[mouse_processed["session_number"] == session_number]
                percent_trials.append(100*(len(session_processed[session_processed["hit_miss_try"]=="hit"])/len(session_processed)))

            #ax.scatter(np.unique(mouse_processed["session_number"]), percent_trials, marker="o", label=mouse_id, zorder=10, clip_on=False)
            ax.plot(np.unique(mouse_processed["session_number"]), percent_trials, label=mouse_id, zorder=10, clip_on=False)

        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.legend(title='Mouse')
        plt.ylabel('% trials', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)

        tick_spacing = 50
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([1, 15, 30])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        Edmond.plot_utility2.style_vr_plot(ax, x_max=100)
        plt.subplots_adjust(bottom = 0.3, left = 0.2)
        #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=30)
        plt.savefig(save_path + '/percentage_trials_by_mouse_hmt_'+hmt+"_tt_"+tt_string+'.png', dpi=300)
        plt.close()

    return


def plot_percentage_trial_per_session_all_mice(processed_position_data, hmt, save_path):

    for tt, tt_string in zip([0,1,2], ["b", "nb", "p"]):

        t_processed = processed_position_data[processed_position_data["trial_type"] == tt]
        speed_histogram = plt.figure(figsize=(8,3.3))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        percentages_array = np.zeros((len(np.unique(processed_position_data["mouse_id"])), max(processed_position_data["session_number"])))*np.nan
        for mouse_i, mouse_id in enumerate(np.unique(t_processed["mouse_id"])):
            mouse_processed = t_processed[t_processed["mouse_id"] == mouse_id]

            for session_number in np.unique(mouse_processed["session_number"]):
                session_processed = mouse_processed[mouse_processed["session_number"] == session_number]
                percentages_array[mouse_i, session_number-1] = 100*(len(session_processed[session_processed["hit_miss_try"]=="hit"])/len(session_processed))

        ax.fill_between(np.arange(1, len(percentages_array[0])+1), np.nanmean(percentages_array, axis=0)-stats.sem(percentages_array, axis=0, nan_policy='omit'), np.nanmean(percentages_array, axis=0)+stats.sem(percentages_array, axis=0, nan_policy='omit'), zorder=10, clip_on=False, alpha=0.3, color=get_hmt_color(hmt))
        ax.plot(np.arange(1, len(percentages_array[0])+1), np.nanmean(percentages_array, axis=0), zorder=10, clip_on=False, color=get_hmt_color(hmt))

        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.legend(title='Mouse')
        plt.ylabel('% trials', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)

        tick_spacing = 50
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([1, 15, 30])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        Edmond.plot_utility2.style_vr_plot(ax, x_max=100)
        plt.subplots_adjust(bottom = 0.3, left = 0.2)
        #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=30)
        plt.savefig(save_path + '/percentage_trials_all_mice_hmt_'+hmt+"_tt_"+tt_string+'.png', dpi=300)
        plt.close()

    return


def plot_percentage_trial_per_session_all_mice_b_vs_nb(processed_position_data, hmt, save_path):
    speed_histogram = plt.figure(figsize=(8,3.3))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for tt, tt_string, c in zip([0,1], ["b", "nb"], ["black", "blue"]):
        t_processed = processed_position_data[processed_position_data["trial_type"] == tt]
        percentages_array = np.zeros((len(np.unique(processed_position_data["mouse_id"])), max(processed_position_data["session_number"])))*np.nan
        for mouse_i, mouse_id in enumerate(np.unique(t_processed["mouse_id"])):
            mouse_processed = t_processed[t_processed["mouse_id"] == mouse_id]

            for session_number in np.unique(mouse_processed["session_number"]):
                session_processed = mouse_processed[mouse_processed["session_number"] == session_number]
                percentages_array[mouse_i, session_number-1] = 100*(len(session_processed[session_processed["hit_miss_try"]=="hit"])/len(session_processed))

        ax.fill_between(np.arange(1, len(percentages_array[0])+1), np.nanmean(percentages_array, axis=0)-stats.sem(percentages_array, axis=0, nan_policy='omit'), np.nanmean(percentages_array, axis=0)+stats.sem(percentages_array, axis=0, nan_policy='omit'), zorder=10, clip_on=False, alpha=0.3, color=c)
        ax.plot(np.arange(1, len(percentages_array[0])+1), np.nanmean(percentages_array, axis=0), zorder=10, clip_on=False, color=c)

    ax.tick_params(axis='both', which='major', labelsize=20)
    #ax.legend(title='Mouse')
    plt.ylabel('% trials', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    tick_spacing = 50
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([1, 15, 30])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    Edmond.plot_utility2.style_vr_plot(ax, x_max=100)
    plt.subplots_adjust(bottom = 0.3, left = 0.2)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=30)
    plt.savefig(save_path + '/percentage_trials_all_mice_hmt_'+hmt+'_tt_b_vs_nb.png', dpi=300)
    plt.close()

    return


def plot_percentage_trial_per_session_all_mice_h_vs_t_vs_m(processed_position_data, tt, save_path):
    processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
    speed_histogram = plt.figure(figsize=(8,3.3))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for hmt, c in zip(["hit", "try", "miss", "rejected"], ["green", "orange", "red", "black"]):
        percentages_array = np.zeros((len(np.unique(processed_position_data["mouse_id"])), max(processed_position_data["session_number"])))*np.nan
        for mouse_i, mouse_id in enumerate(np.unique(processed_position_data["mouse_id"])):
            mouse_processed = processed_position_data[processed_position_data["mouse_id"] == mouse_id]

            for session_number in np.unique(mouse_processed["session_number"]):
                session_processed = mouse_processed[mouse_processed["session_number"] == session_number]
                percentages_array[mouse_i, session_number-1] = 100*(len(session_processed[session_processed["hit_miss_try"]==hmt])/len(session_processed))

        ax.fill_between(np.arange(1, len(percentages_array[0])+1), np.nanmean(percentages_array, axis=0)-stats.sem(percentages_array, axis=0, nan_policy='omit'), np.nanmean(percentages_array, axis=0)+stats.sem(percentages_array, axis=0, nan_policy='omit'), zorder=10, clip_on=False, alpha=0.3, color=c)
        ax.plot(np.arange(1, len(percentages_array[0])+1), np.nanmean(percentages_array, axis=0), zorder=10, clip_on=False, color=c)

    ax.tick_params(axis='both', which='major', labelsize=20)
    #ax.legend(title='Mouse')
    plt.ylabel('% trials', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    tick_spacing = 50
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([1, 15, 30])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    Edmond.plot_utility2.style_vr_plot(ax, x_max=100)
    plt.subplots_adjust(bottom = 0.3, left = 0.2)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=30)
    plt.savefig(save_path + '/percentage_trials_all_mice_tt_'+str(tt)+'_h_vs_t_vs_m.png', dpi=300)
    plt.close()

    return

def get_stop_trial_ids(session_df):
    stop_trial_ids = []
    for index, trial_row in session_df.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        tn = trial_row["trial_number"].iloc[0]
        trial_stop_locations = Edmond.plot_utility2.pandas_collumn_to_numpy_array(trial_row["stop_location_cm"])
        trial_stops = np.ones(len(trial_stop_locations))*tn
        stop_trial_ids.extend(trial_stops)
    return np.array(stop_trial_ids)

def curate_stops(session_df, track_length):
    # stops are calculated as being below the stop threshold per unit time bin,
    # this function removes successive stops

    curated_stop_locations = []
    for index, trial_row in session_df.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        stop_locations = trial_row["stop_location_cm"].iloc[0]

        curated_stops = []
        for i, stop in enumerate(stop_locations):
            if i == 0:
                curated_stops.append(stop)
            elif ((stop_locations[i]-stop_locations[i-1]) > 1):
                curated_stops.append(stop)

        curated_stop_locations.append(curated_stops)

    session_df["stop_location_cm"] = curated_stop_locations
    return session_df

def drop_first_and_last_trial(session_df):
    first_trial_number = 1
    last_trial_number = max(session_df["trial_number"])
    session_df = session_df[session_df["trial_number"] != first_trial_number]
    session_df = session_df[session_df["trial_number"] != last_trial_number]
    return session_df

def plot_all_speeds(mouse_df, save_path):
    stops_on_track = plt.figure(figsize=(6,30))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    mouse_df = mouse_df[mouse_df["session_number"]<=30]
    mouse_df = mouse_df.sort_values(by=['session_number', 'trial_number'])

    track_length = mouse_df.track_length.iloc[0]
    fig = plt.figure(figsize=(6,30))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    trial_speeds = Edmond.plot_utility2.pandas_collumn_to_2d_numpy_array(mouse_df["speeds_binned_in_space"])
    where_are_NaNs = np.isnan(trial_speeds)
    trial_speeds[where_are_NaNs] = 0
    locations = np.arange(0, len(trial_speeds[0]))
    ordered = np.arange(0, len(trial_speeds), 1)
    X, Y = np.meshgrid(locations, ordered)
    cmap = plt.cm.get_cmap("jet")
    pcm = ax.pcolormesh(X, Y, trial_speeds, cmap=cmap, shading="auto")
    n_trials = len(mouse_df)
    x_max = n_trials+0.5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.14)
    cbar.mappable.set_clim(0, 100)
    cbar.outline.set_visible(False)
    cbar.set_ticks([0,100])
    cbar.set_ticklabels(["0", "100"])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Speed (cm/s)', fontsize=20, rotation=270)
    plt.ylabel('Trial Number', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    mouse_df["elapsed_trial_number"] = np.arange(1, len(mouse_df)+1, 1)
    first_trials = mouse_df[mouse_df["trial_number"] == 1]
    for i in range(len(first_trials)):
        ax.axhline(y=first_trials["elapsed_trial_number"].iloc[i], color="black", linewidth=3, zorder=2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/all_speeds.png', dpi=200)


def plot_all_stops(mouse_df, save_path, track_length=200):
    stops_on_track = plt.figure(figsize=(6,30))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    mouse_df = curate_stops(mouse_df, track_length)
    mouse_df = mouse_df[mouse_df["session_number"]<=30]
    mouse_df = mouse_df.sort_values(by=['session_number', 'trial_number'])

    trial_number=1
    for session_number in np.unique(mouse_df["session_number"]):
        session_df = mouse_df[mouse_df["session_number"] == session_number]
        session_df = session_df.sort_values(by=['trial_number'])
        for index, trial_row in session_df.iterrows():
            trial_row = trial_row.to_frame().T.reset_index(drop=True)
            trial_type = trial_row["trial_type"].iloc[0]
            trial_stop_color = get_trial_color(trial_type)
            ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]), trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])), 'o', color=trial_stop_color, markersize=2, alpha=1)
            trial_number+=1

    mouse_df["elapsed_trial_number"] = np.arange(1, len(mouse_df)+1, 1)
    first_trials = mouse_df[mouse_df["trial_number"] == 1]
    for i in range(len(first_trials)):
        ax.axhline(y=first_trials["elapsed_trial_number"].iloc[i], color="red", linewidth=3, zorder=2)

    plt.ylabel('Stops on trials', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_track_plot(ax, track_length)
    n_trials = len(mouse_df)
    x_max = n_trials+0.5
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/all_stop_raster.png', dpi=200)

def plot_n_trials(mouse_df, save_path, track_length=200):
    max_session = 30
    stops_on_track = plt.figure(figsize=(6,4))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    mouse_df = mouse_df[mouse_df["session_number"]<=30]
    mouse_df = mouse_df.sort_values(by=['session_number'])

    n_trials = []
    session_numbers = []
    for session_number in np.unique(mouse_df.session_number):
        session_df = mouse_df[mouse_df["session_number"] == session_number]

        n = len(session_df)
        n_trials.append(n)
        session_numbers.append(session_number)

    ax.plot(session_numbers, n_trials, '-', color="black")

    plt.ylabel('n trials', fontsize=25, labelpad = 10)
    plt.xlabel('Session number)', fontsize=25, labelpad = 10)
    plt.xlim(0, max_session)
    #plt.ylim(0,100)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/n_trials.png', dpi=200)

def plot_percentage_hits(mouse_df, save_path, track_length=200):
    stops_on_track = plt.figure(figsize=(6,4))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    mouse_df = mouse_df[mouse_df["session_number"]<=30]
    mouse_df = mouse_df.sort_values(by=['session_number'])

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]
        percent_hits = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]

            percent = (len(session_df[session_df["hit_miss_try"] == "hit"])/len(session_df))*100
            percent_hits.append(percent)
            session_numbers.append(session_number)

        ax.plot(session_numbers, percent_hits, '-', color=tt_color)

    plt.ylabel('% hits', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(0,30)
    plt.ylim(0,100)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax, 100)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/hit_percentage.png', dpi=200)

def plot_all_first_stops(mouse_df, save_path, track_length=200):
    stops_on_track = plt.figure(figsize=(6,30))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    mouse_df = mouse_df[mouse_df["session_number"]<=30]
    mouse_df = mouse_df.sort_values(by=['session_number', 'trial_number'])
    mouse_df = curate_stops(mouse_df, track_length)

    trial_number = 1
    for index, trial_row in mouse_df.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_stop_color = get_trial_color(trial_type)
        if len(np.array(trial_row["stop_location_cm"].iloc[0]))>0:
            ax.plot(np.array(trial_row["stop_location_cm"].iloc[0])[0], trial_number, 'o', color=trial_stop_color, markersize=2, alpha=1)
        #ax.plot(trial_row["first_stop_location_cm"].iloc[0], trial_number, 'o', color=trial_stop_color, markersize=2, alpha=1)
        trial_number+=1

    mouse_df["elapsed_trial_number"] = np.arange(1, len(mouse_df)+1, 1)
    first_trials = mouse_df[mouse_df["trial_number"] == 1]
    for i in range(len(first_trials)):
        ax.axhline(y=first_trials["elapsed_trial_number"].iloc[i], color="red", linewidth=3, zorder=2)

    plt.ylabel('First stops on trials', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_track_plot(ax, track_length)
    n_trials = len(mouse_df)
    x_max = n_trials+0.5
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/all_first_stop_raster.png', dpi=200)
    return

def plot_stops_on_track(mouse_df, session_number, save_path, track_length=200):
    session_df = mouse_df[mouse_df["session_number"] == session_number]

    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    processed_position_data = curate_stops(session_df, track_length)

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_number = trial_row["trial_number"].iloc[0]
        trial_stop_color = get_trial_color(trial_type)
        ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]), trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])), 'o', color=trial_stop_color, markersize=4)

    plt.ylabel('Stops on trials', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    if len(processed_position_data)<10:
        tick_spacing = 5
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/stop_raster'+str(session_number)+'.png', dpi=200)

def plot_stops_on_track_fs(mouse_df, session_number, save_path, track_length=200):
    session_df = mouse_df[mouse_df["session_number"] == session_number]

    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    processed_position_data = curate_stops(session_df, track_length)

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_number = trial_row["trial_number"].iloc[0]
        trial_stop_color = get_trial_color(trial_type)
        ax.plot(trial_row["first_stop_location_cm"].iloc[0], trial_number, 'o', color=trial_stop_color, markersize=4)

    plt.ylabel('First stops on trials', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    if len(processed_position_data)<10:
        tick_spacing = 5
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/stop_raster_fs_'+str(session_number)+'.png', dpi=200)

def number_stops_in_rz_vs_training_day(mouse_df, save_path, percentile=99):
    max_session_number = max(mouse_df["session_number"])
    bin_size = 5

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        percent_in_RZ = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)
            track_length = session_df.track_length.iloc[0]
            rz_start = track_length-60-30-20
            rz_end = track_length-60-30

            session_df = curate_stops(session_df, track_length) # filter stops
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["stop_location_cm"])

            tt_stops_in_RZ = tt_stops[(tt_stops >= rz_start) & (tt_stops <= rz_end)]
            percent = len(tt_stops_in_RZ)/len(tt_session_df)
            percent_in_RZ.append(percent)
            session_numbers.append(session_number)

        ax.plot(session_numbers, percent_in_RZ, '-', color=tt_color)

    plt.ylabel('stops/trial in RZ', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,30)
    #plt.ylim(0,100)
    #ax.set_yticks([0, 1])
    #ax.set_yticklabels(["0", "1"])
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/number_stops_in_RZ_vs_training_day.png', dpi=200)
    plt.close()

def percentage_stops_in_rz_vs_training_day(mouse_df, save_path, percentile=99):
    max_session_number = max(mouse_df["session_number"])
    bin_size = 5

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        percent_in_RZ = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)
            track_length = session_df.track_length.iloc[0]
            rz_start = track_length-60-30-20
            rz_end = track_length-60-30

            session_df = curate_stops(session_df, track_length) # filter stops
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["stop_location_cm"])

            tt_stops_in_RZ = tt_stops[(tt_stops >= rz_start) & (tt_stops <= rz_end)]
            percent = (len(tt_stops_in_RZ)/len(tt_stops))*100
            percent_in_RZ.append(percent)
            session_numbers.append(session_number)

        ax.plot(session_numbers, percent_in_RZ, '-', color=tt_color)

    plt.ylabel('% stops in RZ', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,30)
    plt.ylim(0,100)
    #ax.set_yticks([0, 1])
    #ax.set_yticklabels(["0", "1"])
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/percentage_stops_in_RZ_vs_training_day.png', dpi=200)
    plt.close()

def number_first_stops_in_rz_vs_training_day(mouse_df, save_path, percentile=99):
    max_session_number = max(mouse_df["session_number"])
    bin_size = 5

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        percent_in_RZ = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)
            track_length = session_df.track_length.iloc[0]
            rz_start = track_length-60-30-20
            rz_end = track_length-60-30

            session_df = curate_stops(session_df, track_length) # filter stops
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["first_stop_location_cm"])

            tt_stops_in_RZ = tt_stops[(tt_stops >= rz_start) & (tt_stops <= rz_end)]
            percent = len(tt_stops_in_RZ)/len(tt_session_df)
            percent_in_RZ.append(percent)
            session_numbers.append(session_number)

        ax.plot(session_numbers, percent_in_RZ, '-', color=tt_color)

    plt.ylabel('first stops / trial in RZ', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,30)
    #plt.ylim(0,100)
    #ax.set_yticks([0, 1])
    #ax.set_yticklabels(["0", "1"])
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/number_first_stops_in_RZ_vs_training_day.png', dpi=200)
    plt.close()

def percentage_first_stops_in_rz_vs_training_day(mouse_df, save_path, percentile=99):
    max_session_number = max(mouse_df["session_number"])
    bin_size = 5

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        percent_in_RZ = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)
            track_length = session_df.track_length.iloc[0]
            rz_start = track_length-60-30-20
            rz_end = track_length-60-30

            session_df = curate_stops(session_df, track_length) # filter stops
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["first_stop_location_cm"])

            tt_stops_in_RZ = tt_stops[(tt_stops >= rz_start) & (tt_stops <= rz_end)]
            percent = (len(tt_stops_in_RZ)/len(tt_stops))*100
            percent_in_RZ.append(percent)
            session_numbers.append(session_number)

        ax.plot(session_numbers, percent_in_RZ, '-', color=tt_color)

    plt.ylabel('% first stops in RZ', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,30)
    plt.ylim(0,100)
    #ax.set_yticks([0, 1])
    #ax.set_yticklabels(["0", "1"])
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/percentage_first_stops_in_RZ_vs_training_day.png', dpi=200)
    plt.close()

def shuffled_vs_training_day(mouse_df, save_path, percentile=99):
    max_session_number = max(mouse_df["session_number"])
    bin_size = 5

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        shuffled_vs_peaks = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)
            track_length = session_df.track_length.iloc[0]
            rz_start = track_length-60-30-20
            rz_end = track_length-60-30

            session_df = curate_stops(session_df, track_length) # filter stops
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["stop_location_cm"])

            # calculate trial type stops per trial
            tt_hist, bin_edges = np.histogram(tt_stops, bins=int(track_length/bin_size), range=(0, track_length))
            bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            tt_hist_RZ = tt_hist[(bin_centres > rz_start) & (bin_centres < rz_end)]
            measured_peak = max(tt_hist_RZ/len(session_df))

            # calculate changce level peak
            shuffle_peaks = []
            for i in enumerate(np.arange(1000)):
                shuffled_stops = np.random.uniform(low=0, high=track_length, size=len(tt_stops))
                shuffled_stop_hist, bin_edges = np.histogram(shuffled_stops, bins=int(track_length/bin_size), range=(0, track_length))
                bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
                shuffled_stop_hist_RZ = shuffled_stop_hist[(bin_centres > rz_start) & (bin_centres < rz_end)]

                peak = max(shuffled_stop_hist_RZ/len(session_df))
                shuffle_peaks.append(peak)
            shuffle_peaks = np.array(shuffle_peaks)
            threshold = np.nanpercentile(shuffle_peaks, percentile)

            peak_vs_shuffle = measured_peak-threshold

            shuffled_vs_peaks.append(peak_vs_shuffle)
            session_numbers.append(session_number)

        ax.plot(session_numbers, shuffled_vs_peaks, '-', color=tt_color)

    plt.ylabel('Peak stops / trial\n vs shuffle', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,30)
    #plt.ylim(0,y_max)
    #ax.set_yticks([0, 1])
    #ax.set_yticklabels(["0", "1"])
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/shuffle_vs_training_day.png', dpi=200)
    plt.close()


def shuffled_vs_training_day_fs(mouse_df, save_path, percentile=99):
    max_session_number = max(mouse_df["session_number"])
    bin_size = 5

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        shuffled_vs_peaks = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)
            track_length = session_df.track_length.iloc[0]
            rz_start = track_length-60-30-20
            rz_end = track_length-60-30

            session_df = curate_stops(session_df, track_length) # filter stops
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df['first_stop_location_cm'])

            # calculate trial type stops per trial
            tt_hist, bin_edges = np.histogram(tt_stops, bins=int(track_length/bin_size), range=(0, track_length))
            bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            tt_hist_RZ = tt_hist[(bin_centres > rz_start) & (bin_centres < rz_end)]
            measured_peak = max(tt_hist_RZ/len(session_df))

            # calculate changce level peak
            shuffle_peaks = []
            for i in enumerate(np.arange(1000)):
                shuffled_stops = np.random.uniform(low=0, high=track_length, size=len(tt_stops))
                shuffled_stop_hist, bin_edges = np.histogram(shuffled_stops, bins=int(track_length/bin_size), range=(0, track_length))
                bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
                shuffled_stop_hist_RZ = shuffled_stop_hist[(bin_centres > rz_start) & (bin_centres < rz_end)]

                peak = max(shuffled_stop_hist_RZ/len(session_df))
                shuffle_peaks.append(peak)
            shuffle_peaks = np.array(shuffle_peaks)
            threshold = np.nanpercentile(shuffle_peaks, percentile)

            peak_vs_shuffle = measured_peak-threshold

            shuffled_vs_peaks.append(peak_vs_shuffle)
            session_numbers.append(session_number)

        ax.plot(session_numbers, shuffled_vs_peaks, '-', color=tt_color)

    plt.ylabel('First stop peak / trial\n vs shuffle', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,30)
    #plt.ylim(0,y_max)
    #ax.set_yticks([0, 1])
    #ax.set_yticklabels(["0", "1"])
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/shuffle_vs_training_day_fs.png', dpi=200)
    plt.close()

def plot_shuffled_stops(mouse_df, session_number, save_path, percentile=99, y_max=1):
    session_df = mouse_df[mouse_df["session_number"] == session_number]
    session_df = drop_first_and_last_trial(session_df)
    track_length = session_df.track_length.iloc[0]
    bin_size = 5
    rz_start = track_length-60-30-20
    rz_end = track_length-60-30

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = session_df[session_df["trial_type"] == tt]

        # only run if there are trials
        if len(tt_session_df) != 0:
            tt_session_df = curate_stops(tt_session_df, track_length) # filter stops
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(tt_session_df["stop_location_cm"])

            # calculate trial type stops per trial
            tt_hist, bin_edges = np.histogram(tt_stops, bins=int(track_length/bin_size), range=(0, track_length))
            peak = max(tt_hist/len(tt_session_df))

            # calculate changce level peak for reward zone
            shuffle_peaks = []
            for i in enumerate(np.arange(1000)):
                shuffled_stops = np.random.uniform(low=0, high=track_length, size=len(tt_stops))
                shuffled_stop_hist, bin_edges = np.histogram(shuffled_stops, bins=int(track_length/bin_size), range=(0, track_length))
                peak = (max(shuffled_stop_hist))/len(tt_session_df)
                shuffle_peaks.append(peak)
            shuffle_peaks = np.array(shuffle_peaks)
            threshold = np.nanpercentile(shuffle_peaks, percentile)

            # plot figure
            stop_histogram = plt.figure(figsize=(6,2))
            ax = stop_histogram.add_subplot(1, 1, 1)
            bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            ax.plot(bin_centres, tt_hist/len(tt_session_df), '-', color=tt_color)
            ax.axhline(y=threshold, color="Grey", linestyle="dashed", linewidth=2)
            plt.ylabel('Stops/trial', fontsize=25, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
            plt.xlim(0,track_length)
            plt.ylim(0,y_max)
            #ax.set_yticks([0, 1])
            #ax.set_yticklabels(["0", "1"])
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
            plt.savefig(save_path + '/stop_histogram_day'+str(session_number)+'_tt_'+str(tt)+'.png', dpi=200)
            plt.close()
    return

def correct_for_time_binned_teleport(trial_pos_in_time, track_length):
    # check if any of the first 5 or last 5 bins are too high or too low respectively
    first_5 = trial_pos_in_time[:5]
    last_5 = trial_pos_in_time[-5:]

    first_5[first_5>(track_length/2)] = first_5[first_5>(track_length/2)]-track_length
    last_5[last_5<(track_length/2)] = last_5[last_5<(track_length/2)]+track_length

    trial_pos_in_time[:5] = first_5
    trial_pos_in_time[-5:] = last_5
    return trial_pos_in_time

def plot_speed_profile(mouse_df, session_number, save_path):
    session_df = mouse_df[mouse_df["session_number"] == session_number]
    track_length = session_df.track_length.iloc[0]

    for tt in [0,1,2]:
        speed_histogram = plt.figure(figsize=(6,4))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        subset_processed_position_data = session_df[(session_df["trial_type"] == tt)]

        if len(subset_processed_position_data)>0:
            bin_centres = np.array(subset_processed_position_data["position_bin_centres"].iloc[0])

            trial_speeds = pandas_collumn_to_2d_numpy_array(subset_processed_position_data["speeds_binned_in_space"])
            bin_centres = np.array(subset_processed_position_data["position_bin_centres"].iloc[0])
            trial_speeds_avg = np.nanmean(trial_speeds, axis=0)

            # to plot by trial using the time binned data we need the n-1, n and n+1 trials so we can plot around the track limits
            # here we extract the n-1, n and n+1 trials, correct for any time binned teleports and concatenated the positions and speeds for each trial
            for i, tn in enumerate(session_df["trial_number"]):
                trial_processed_position_data = session_df[session_df["trial_number"] == tn]
                tt_trial = trial_processed_position_data["trial_type"].iloc[0]
                hmt_trial = trial_processed_position_data["hit_miss_try"].iloc[0]
                trial_speeds_in_time = np.asarray(trial_processed_position_data['speeds_binned_in_time'].iloc[0])
                trial_pos_in_time = np.asarray(trial_processed_position_data['pos_binned_in_time'].iloc[0])

                # cases above trial number 1
                if tn != min(session_df["trial_number"]):
                    trial_processed_position_data_1down = session_df[session_df["trial_number"] == tn-1]
                    trial_speeds_in_time_1down = np.asarray(trial_processed_position_data_1down['speeds_binned_in_time'].iloc[0])
                    trial_pos_in_time_1down = np.asarray(trial_processed_position_data_1down['pos_binned_in_time'].iloc[0])
                else:
                    trial_speeds_in_time_1down = np.array([])
                    trial_pos_in_time_1down = np.array([])

                # cases below trial number n
                if tn != max(session_df["trial_number"]):
                    trial_processed_position_data_1up = session_df[session_df["trial_number"] == tn+1]
                    trial_speeds_in_time_1up = np.asarray(trial_processed_position_data_1up['speeds_binned_in_time'].iloc[0])
                    trial_pos_in_time_1up = np.asarray(trial_processed_position_data_1up['pos_binned_in_time'].iloc[0])
                else:
                    trial_speeds_in_time_1up = np.array([])
                    trial_pos_in_time_1up = np.array([])

                trial_pos_in_time = np.concatenate((trial_pos_in_time_1down[-2:], trial_pos_in_time, trial_pos_in_time_1up[:2]))
                trial_speeds_in_time = np.concatenate((trial_speeds_in_time_1down[-2:], trial_speeds_in_time, trial_speeds_in_time_1up[:2]))

                if tt_trial == tt:
                    trial_pos_in_time = correct_for_time_binned_teleport(trial_pos_in_time, track_length)
                    ax.plot(trial_pos_in_time, trial_speeds_in_time, color="grey", alpha=0.4)

            ax.plot(bin_centres, trial_speeds_avg, color=get_trial_color(tt), linewidth=4)
            ax.axhline(y=4.7, color="black", linestyle="dashed", linewidth=2)
            plt.ylabel('Speed (cm/s)', fontsize=25, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
            plt.xlim(0,track_length)
            ax.set_yticks([0, 50, 100])
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            if tt == 0:
                style_track_plot(ax, track_length)
            else:
                style_track_plot_no_RZ(ax, track_length)
            tick_spacing = 100
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            x_max = 115
            Edmond.plot_utility2.style_vr_plot(ax, x_max)
            plt.subplots_adjust(bottom = 0.2, left=0.2)
            plt.savefig(save_path+'/trial_speeds_d'+str(session_number)+'tt_'+str(tt)+'.png', dpi=300)
            plt.close()

def plot_speeds_vs_days(mouse_df, save_path):
    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        avg_trial_speeds = []
        avg_RZ_speeds = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)

            avg_RZ_speeds.append(np.nanmean(session_df["avg_speed_in_RZ"]))
            avg_trial_speeds.append(np.nanmean(session_df["avg_trial_speed"]))
            session_numbers.append(session_number)

        ax.plot(session_numbers, avg_RZ_speeds, '-', color=tt_color)
        ax.plot(session_numbers, avg_trial_speeds, '--', color=tt_color)

    plt.ylabel('Speed (cm/s)', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/training_day_vs_speeds.png', dpi=200)
    plt.close()


def plot_speed_diff_vs_days(mouse_df, save_path):
    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        avg_trial_speeds = []
        avg_RZ_speeds = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)

            avg_RZ_speeds.append(np.nanmean(session_df["avg_speed_in_RZ"]))
            avg_trial_speeds.append(np.nanmean(session_df["avg_trial_speed"]))
            session_numbers.append(session_number)

        avg_RZ_speeds = np.array(avg_RZ_speeds)
        avg_trial_speeds = np.array(avg_trial_speeds)
        session_numbers = np.array(session_numbers)

        ax.plot(session_numbers, avg_RZ_speeds-avg_trial_speeds, color=tt_color)

    plt.ylabel(r'$\Delta$ speed (cm/s)', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,30)
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/training_day_vs_speed_diff.png', dpi=200)
    plt.close()

def plot_hit_percentage_beaconed_and_non_beaconed_for_example_mouse(mouse_df, meta_behaviour_data, save_path):
    max_session = 14
    mouse_df = mouse_df[mouse_df["session_number"] <= max_session]

    #mouse_df = mouse_df[mouse_df["session_number"] != 21]
    #mouse_df = mouse_df[mouse_df["session_number"] != 27]
    #mouse_df = mouse_df[mouse_df["session_number"] != 28]

    mouse_array = np.zeros((3, max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1], ["Black", "Blue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        percent_hits = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)

            if len(session_df)==0:
                percent = 0
            else:
                percent = (len(session_df[session_df["hit_miss_try"] == "hit"])/len(session_df))*100

            percent_hits.append(percent)
            session_numbers.append(session_number)

            mouse_array[tt, session_number-1] = percent

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    # plot per mouse
    nan_mask =  ~np.isnan(mouse_array[0])
    ax.plot(np.arange(1,max_session+1)[nan_mask], mouse_array[0][nan_mask], '-', color="black")
    ax.plot(np.arange(1,max_session+1)[nan_mask], mouse_array[1][nan_mask], '-', color="blue")
    #ax.axvline(x=15, color="red", linestyle="dashed")
    plt.title("M14", fontsize=25)
    plt.ylabel('% Fast hit trials', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    plt.ylim(0, 100)
    #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.9)
    plt.savefig(save_path + '/percentage_hits_M14.png', dpi=200)
    plt.close()


    mouse_array = np.zeros((3, max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1], ["Black", "Blue"]):
        tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

        percent_hits = []
        session_numbers = []
        for session_number in np.unique(tt_session_df.session_number):
            session_df = tt_session_df[tt_session_df["session_number"] == session_number]
            session_df = drop_first_and_last_trial(session_df)

            if len(session_df)==0:
                percent = 0
            else:
                percent = len(session_df[session_df["hit_miss_try"] == "hit"])

            percent_hits.append(percent)
            session_numbers.append(session_number)

            mouse_array[tt, session_number-1] = percent

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    # plot per mouse
    nan_mask =  ~np.isnan(mouse_array[0])
    ax.plot(np.arange(1,max_session+1)[nan_mask], mouse_array[0][nan_mask], '-', color="black")
    ax.plot(np.arange(1,max_session+1)[nan_mask], mouse_array[1][nan_mask], '-', color="blue")
    #ax.axvline(x=15, color="red", linestyle="dashed")
    plt.title("M14", fontsize=25)
    plt.ylabel('N fast hit trials', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    #plt.ylim(0, 100)
    #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.9)
    plt.savefig(save_path + '/number_hits_M14.png', dpi=200)
    plt.close()


    mouse_meta_behaviour_data = meta_behaviour_data[meta_behaviour_data["mouse_id"] == "M14"]
    mouse_meta_behaviour_data = mouse_meta_behaviour_data[mouse_meta_behaviour_data["session_number"] <= max_session]
    mouse_meta_behaviour_data = mouse_meta_behaviour_data[mouse_meta_behaviour_data["session_number"] != 21]
    mouse_meta_behaviour_data = mouse_meta_behaviour_data[mouse_meta_behaviour_data["session_number"] != 27]
    mouse_meta_behaviour_data = mouse_meta_behaviour_data[mouse_meta_behaviour_data["session_number"] != 28]

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    # plot per mouse
    trial_type_ratios = []
    for i in np.arange(1,max_session+1):
        session_meta = mouse_meta_behaviour_data[mouse_meta_behaviour_data["session_number"] == i]
        if len(session_meta)==1:
            trial_type_ratios.append(session_meta["trial_type_ratio_numeric"].iloc[0])
        else:
            trial_type_ratios.append(np.nan)

    nan_mask =  ~np.isnan(np.array(trial_type_ratios))
    ax.plot(np.arange(1,max_session+1)[nan_mask], np.array(trial_type_ratios)[nan_mask], '-', color="red")

    plt.title("M14", fontsize=25)
    plt.ylabel('Beaconed: non-beaconed trial ratio', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    ax.axhline(y=1, color="black", linestyle="dashed")
    #ax.axvline(x=15, color="red", linestyle="dashed")
    plt.ylim(0, 5)
    #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.9)
    plt.savefig(save_path + '/trial_type_ratio_M14.png', dpi=200)
    plt.close()
    return

def plot_speed_heat_map(mouse_df, session_number, save_path):
    session_df = mouse_df[mouse_df["session_number"] == session_number]
    track_length = session_df.track_length.iloc[0]
    x_max = len(session_df)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    trial_speeds = Edmond.plot_utility2.pandas_collumn_to_2d_numpy_array(session_df["speeds_binned_in_space"])
    where_are_NaNs = np.isnan(trial_speeds)
    trial_speeds[where_are_NaNs] = 0
    locations = np.arange(0, len(trial_speeds[0]))
    ordered = np.arange(0, len(trial_speeds), 1)
    X, Y = np.meshgrid(locations, ordered)
    cmap = plt.cm.get_cmap("jet")
    pcm = ax.pcolormesh(X, Y, trial_speeds, cmap=cmap, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.14)
    cbar.mappable.set_clim(0, 100)
    cbar.outline.set_visible(False)
    cbar.set_ticks([0,100])
    cbar.set_ticklabels(["0", "100"])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Speed (cm/s)', fontsize=20, rotation=270)
    plt.ylabel('Trial Number', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/speed_heat_map_d' +str(session_number)+ '.png', dpi=200)
    plt.close()


def plot_shuffled_stops_fs(mouse_df, session_number, save_path, percentile=99, y_max=1):
    session_df = mouse_df[mouse_df["session_number"] == session_number]
    session_df = drop_first_and_last_trial(session_df)
    track_length = session_df.track_length.iloc[0]
    bin_size = 5
    rz_start = track_length-60-30-20
    rz_end = track_length-60-30

    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_session_df = session_df[session_df["trial_type"] == tt]

        # only run if there are trials
        if len(tt_session_df) != 0:
            tt_session_df = curate_stops(tt_session_df, track_length) # filter stops
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(tt_session_df["first_stop_location_cm"])

            # calculate trial type stops per trial
            tt_hist, bin_edges = np.histogram(tt_stops, bins=int(track_length/bin_size), range=(0, track_length))
            peak = max(tt_hist/len(tt_session_df))

            # calculate changce level peak for reward zone
            shuffle_peaks = []
            for i in enumerate(np.arange(1000)):
                shuffled_stops = np.random.uniform(low=0, high=track_length, size=len(tt_stops))
                shuffled_stop_hist, bin_edges = np.histogram(shuffled_stops, bins=int(track_length/bin_size), range=(0, track_length))
                peak = (max(shuffled_stop_hist))/len(tt_session_df)
                shuffle_peaks.append(peak)
            shuffle_peaks = np.array(shuffle_peaks)
            threshold = np.nanpercentile(shuffle_peaks, percentile)

            # plot figure
            stop_histogram = plt.figure(figsize=(6,2))
            ax = stop_histogram.add_subplot(1, 1, 1)
            bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            ax.plot(bin_centres, tt_hist/len(tt_session_df), '-', color=tt_color)
            ax.axhline(y=threshold, color="Grey", linestyle="dashed", linewidth=2)
            plt.ylabel('Stops/trial', fontsize=25, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
            plt.xlim(0,track_length)
            plt.ylim(0,y_max)
            #ax.set_yticks([0, 1])
            #ax.set_yticklabels(["0", "1"])
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
            plt.savefig(save_path + '/stop_histogram_fs_day'+str(session_number)+'_tt_'+str(tt)+'.png', dpi=200)
            plt.close()
    return


def population_shuffled_vs_training_day_numbers_first_stops(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    bin_size = 5
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            percent_in_RZ = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)>0:
                    track_length = session_df.track_length.iloc[0]
                    rz_start = track_length-60-30-20
                    rz_end = track_length-60-30

                    session_df = curate_stops(session_df, track_length) # filter stops
                    tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df['first_stop_location_cm'])
                    if len(tt_stops)==0:
                        percent=0
                    else:
                        tt_stops_in_RZ = tt_stops[(tt_stops >= rz_start) & (tt_stops <= rz_end)]
                        percent = len(tt_stops_in_RZ)/len(session_df)
                    percent_in_RZ.append(percent)
                    session_numbers.append(session_number)

                    mouse_array[tt, mouse_i, session_number-1] = percent

            mouse_i +=1

            # plot per mouse
            ax.plot(session_numbers, percent_in_RZ, '-', label=mouse_id, color=mouse_color)
        plt.ylabel('first stops / trial in RZ', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        #plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/number_first_stops_vs_training_day_fs_tt_'+str(tt)+'_by_mouse.png', dpi=200)
        plt.close()

        # plot average across mouse
        tt_mouse_array = mouse_array[tt]


        # do stats test
        df = pd.DataFrame({'mouse_id': np.repeat(np.arange(0,len(mouse_ids)), len(tt_mouse_array[0])),
                           'session_id': np.tile(np.arange(0,max_session),  len(tt_mouse_array)),
                           'n_first_stops_in_rz': tt_mouse_array.flatten()})
        df = df.dropna()
        df = reassign_session_numbers(df)
        # Conduct the repeated measures ANOVA
        df = df[df["mouse_id"]!=0]
        df = df[df["mouse_id"]!=4]
        df = df[df["mouse_id"]!=6]
        df = df[df["mouse_id"]!=11]
        df = df[df["session_id"]<25]
        if tt == 0 or tt==1:
            a = AnovaRM(data=df, depvar='n_first_stops_in_rz', subject='mouse_id', within=['session_id'], aggregate_func='mean').fit()
            print("tt = ", str(tt), ", test for n_first_stops_in_rz")
            print("p= ", str(a.anova_table["Pr > F"].iloc[0]), ", Num DF= ", str(a.anova_table["Num DF"].iloc[0]), ", Num DF= ", str(a.anova_table["Den DF"].iloc[0]), "F value= ", str(a.anova_table["F Value"].iloc[0]))


        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
        plt.ylabel('n first stops in RZ', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        #plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/number_first_stops_vs_training_day_fs_tt_'+str(tt)+'.png', dpi=200)
        plt.close()

    # plot average across mouse for all trial types
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_mouse_array = mouse_array[tt]
        nan_mask = ~np.isnan(np.nanmean(tt_mouse_array, axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)

    for session in np.arange(1,max_session+1)-1:
        # do pairwise comparison
        f, p = stats.wilcoxon(mouse_array[0,:,session], mouse_array[1,:,session], nan_policy="omit")
        if p<0.05:
            ax.text(session, 0.65, "*", fontsize=20, color="red")
    plt.ylabel('n first stops in RZ', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    #plt.ylim(0, 100)
    #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/number_first_stops_vs_training_day_fs_all_tt.png', dpi=200)
    plt.close()


def reassign_session_numbers(df):
    new_df = pd.DataFrame()
    for mouse_id in np.unique(df["mouse_id"]):
        mouse_df = df[df["mouse_id"] == mouse_id]
        i=0
        for index, session_df in mouse_df.iterrows():
            session_df = session_df.to_frame().T.reset_index(drop=True)
            session_df["session_id"] = [i]
            new_df = pd.concat([new_df, session_df], ignore_index=True)
            i+=1
    return new_df

def plot_track_speeds_by_mouse(processed_position_data, save_path):
    max_session = 30
    all_behaviour200cm_tracks = processed_position_data[processed_position_data["session_number"] <= max_session]
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]
    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))
    mouse_array = np.zeros((len(mouse_ids), max_session)); mouse_array[:, :] = np.nan

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    mouse_i = 0
    for mouse_id, mouse_color in zip(mouse_ids, colors):
        mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
        avg_trial_speeds = []
        session_numbers = []
        for session_number in np.unique(mouse_df.session_number):
            session_df = mouse_df[mouse_df["session_number"] == session_number]
            avg_trial_speed = np.nanmean(session_df["avg_trial_speed"])
            avg_trial_speeds.append(avg_trial_speed)
            session_numbers.append(session_number)
            mouse_array[mouse_i, session_number-1] = avg_trial_speed
        mouse_i +=1
        # plot per mouse
        ax.plot(session_numbers, avg_trial_speeds, '-', label=mouse_id, color=mouse_color)

    plt.ylabel('Avg trial speed', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    plt.ylim(0, 80)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/trial_speed_by_mouse.png', dpi=200)
    plt.close()

    # plot average across mouse
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    nan_mask = ~np.isnan(np.nanmean(mouse_array, axis=0))
    ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(mouse_array, axis=0)-np.nanstd(mouse_array, axis=0))[nan_mask], (np.nanmean(mouse_array, axis=0)+np.nanstd(mouse_array, axis=0))[nan_mask], color="black", alpha=0.3)
    ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(mouse_array, axis=0)[nan_mask], color="black")
    plt.ylabel('Avg trial speed', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    plt.ylim(0, 80)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/trial_speeds_across_animals.png', dpi=200)
    plt.close()
    return

def plot_average_hit_try_run_profile(all_behaviour200cm_tracks, save_path):
    max_session = np.nanmax(all_behaviour200cm_tracks["session_number"])
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            percent_hits = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)==0:
                    percent = 0
                else:
                    percent = (len(session_df[session_df["hit_miss_try"] == "hit"])/len(session_df))*100

                percent_hits.append(percent)
                session_numbers.append(session_number)

                mouse_array[tt, mouse_i, session_number-1] = percent

            mouse_i +=1

    for i, mouse_id in enumerate(mouse_ids):
        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        # plot per mouse
        ax.plot(np.arange(1,max_session+1), mouse_array[0][i], '-', color="black")
        ax.plot(np.arange(1,max_session+1), mouse_array[1][i], '-', color="blue")
        ax.plot(np.arange(1,max_session+1), mouse_array[2][i], '-', color="deepskyblue")
        plt.title(mouse_id, fontsize=25)
        plt.ylabel('% hits', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.9)
        plt.savefig(save_path + '/percentage_hits_'+mouse_id+'.png', dpi=200)
        plt.close()

def plot_percentage_hits_by_mouse_individual_plots(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            percent_hits = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)==0:
                    percent = 0
                else:
                    percent = (len(session_df[session_df["hit_miss_try"] == "hit"])/len(session_df))*100

                percent_hits.append(percent)
                session_numbers.append(session_number)

                mouse_array[tt, mouse_i, session_number-1] = percent

            mouse_i +=1

    for i, mouse_id in enumerate(mouse_ids):
        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        # plot per mouse
        ax.plot(np.arange(1,max_session+1), mouse_array[0][i], '-', color="black")
        ax.plot(np.arange(1,max_session+1), mouse_array[1][i], '-', color="blue")
        ax.plot(np.arange(1,max_session+1), mouse_array[2][i], '-', color="deepskyblue")
        plt.title(mouse_id, fontsize=25)
        plt.ylabel('% hits', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.9)
        plt.savefig(save_path + '/percentage_hits_'+mouse_id+'.png', dpi=200)
        plt.close()


def plot_percentage_hits_by_mouse(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):

        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            percent_hits = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)==0:
                    percent = 0
                else:
                    percent = (len(session_df[session_df["hit_miss_try"] == "hit"])/len(session_df))*100

                percent_hits.append(percent)
                session_numbers.append(session_number)

                mouse_array[tt, mouse_i, session_number-1] = percent

            mouse_i +=1

            # plot per mouse
            ax.plot(session_numbers, percent_hits, '-', label=mouse_id, color=mouse_color)
        plt.ylabel('% hits', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/percentage_hits_tt_'+str(tt)+'_by_mouse.png', dpi=200)
        plt.close()

        # plot average across mouse
        tt_mouse_array = mouse_array[tt]

        # do stats test
        df = pd.DataFrame({'mouse_id': np.repeat(np.arange(0,len(mouse_ids)), len(tt_mouse_array[0])),
                           'session_id': np.tile(np.arange(0,max_session),  len(tt_mouse_array)),
                           'percentage_hits': tt_mouse_array.flatten()})
        df = df.dropna()
        df = reassign_session_numbers(df)
        # Conduct the repeated measures ANOVA
        df = df[df["mouse_id"]!=0]
        df = df[df["mouse_id"]!=4]
        df = df[df["mouse_id"]!=6]
        df = df[df["mouse_id"]!=11]
        df = df[df["session_id"]<25]
        if tt == 0 or tt==1:
            a = AnovaRM(data=df, depvar='percentage_hits', subject='mouse_id', within=['session_id'], aggregate_func='mean').fit()
            print("repeated measures test for percentage_hits for tt ", str(tt))
            print("p= ", str(a.anova_table["Pr > F"].iloc[0]), ", Num DF= ", str(a.anova_table["Num DF"].iloc[0]), ", Num DF= ", str(a.anova_table["Den DF"].iloc[0]), "F value= ", str(a.anova_table["F Value"].iloc[0]))

        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
        plt.ylabel('% hits', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/percentage_hits_tt_'+str(tt)+'.png', dpi=200)
        plt.close()

    # plot average across mouse for all trial types
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_mouse_array = mouse_array[tt]
        nan_mask = ~np.isnan(np.nanmean(tt_mouse_array, axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)

    for session in np.arange(1,max_session+1)-1:
        # do pairwise comparison
        f, p = stats.wilcoxon(mouse_array[0,:,session], mouse_array[1,:,session], nan_policy="omit")
        if p<0.05:
            ax.text(session, 95, "*", fontsize=20, color="red")

    plt.ylabel('% hits', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    plt.ylim(0, 100)
    #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/percentage_hits_all_tt.png', dpi=200)
    plt.close()

    f, p = stats.wilcoxon(mouse_array[0,:,:].flatten(), mouse_array[1,:,:].flatten(), nan_policy="omit")
    print("Comparing proportion of hit trials between beaconed and non beaconed trials, F = ", str(f), ", p = ", str(p), ", N = ", str(len(mouse_array[0,:,:].flatten()[~np.isnan(mouse_array[0,:,:].flatten())])))
    return

def population_shuffled_vs_training_day_percentage_first_stops(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    bin_size = 5
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            percent_in_RZ = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)>0:
                    track_length = session_df.track_length.iloc[0]
                    rz_start = track_length-60-30-20
                    rz_end = track_length-60-30

                    session_df = curate_stops(session_df, track_length) # filter stops
                    tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df['first_stop_location_cm'])
                    if len(tt_stops)==0:
                        percent=0
                    else:
                        tt_stops_in_RZ = tt_stops[(tt_stops >= rz_start) & (tt_stops <= rz_end)]
                        percent = (len(tt_stops_in_RZ)/len(tt_stops))*100
                    percent_in_RZ.append(percent)
                    session_numbers.append(session_number)

                    mouse_array[tt, mouse_i, session_number-1] = percent

            mouse_i +=1

            # plot per mouse
            ax.plot(session_numbers, percent_in_RZ, '-', label=mouse_id, color=mouse_color)
        plt.ylabel('% first stops in RZ', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/percentage_first_stops_vs_training_day_fs_tt_'+str(tt)+'_by_mouse.png', dpi=200)
        plt.close()

        # plot average across mouse
        tt_mouse_array = mouse_array[tt]

        # do stats test
        df = pd.DataFrame({'mouse_id': np.repeat(np.arange(0,len(mouse_ids)), len(tt_mouse_array[0])),
                           'session_id': np.tile(np.arange(0,max_session),  len(tt_mouse_array)),
                           'proportion_first_stops_in_rz': tt_mouse_array.flatten()})
        df = df.dropna()
        df = reassign_session_numbers(df)
        # Conduct the repeated measures ANOVA
        df = df[df["mouse_id"]!=0]
        df = df[df["mouse_id"]!=4]
        df = df[df["mouse_id"]!=6]
        df = df[df["mouse_id"]!=11]
        df = df[df["session_id"]<25]
        if tt == 0 or tt==1:
            a = AnovaRM(data=df, depvar='proportion_first_stops_in_rz', subject='mouse_id', within=['session_id'], aggregate_func='mean').fit()
            print("tt = ", str(tt), ", test for proportion_first_stops_in_rz")
            print("p= ", str(a.anova_table["Pr > F"].iloc[0]), ", Num DF= ", str(a.anova_table["Num DF"].iloc[0]), ", Num DF= ", str(a.anova_table["Den DF"].iloc[0]), "F value= ", str(a.anova_table["F Value"].iloc[0]))


        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
        plt.ylabel('% first stops in RZ', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/percentage_first_stops_vs_training_day_fs_tt_'+str(tt)+'.png', dpi=200)
        plt.close()

    # plot average across mouse for all trial types
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_mouse_array = mouse_array[tt]
        nan_mask = ~np.isnan(np.nanmean(tt_mouse_array, axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)

    for session in np.arange(1,max_session+1)-1:
        # do pairwise comparison
        f, p = stats.wilcoxon(mouse_array[0,:,session], mouse_array[1,:,session], nan_policy="omit")
        if p<0.05:
            ax.text(session, 95, "*", fontsize=20, color="red")
    plt.ylabel('% first stops in RZ', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    plt.ylim(0, 100)
    #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/percentage_first_stops_vs_training_day_fs_all_tt.png', dpi=200)
    plt.close() 

def plot_trial_ratio_vs_percentage_correct_across_time(meta_df, save_path):


    for mouse_id in np.unique(meta_df["mouse_id"]):
        mouse_meta_df = meta_df[meta_df["mouse_id"] == mouse_id]
        mouse_meta_df = mouse_meta_df.sort_values(by=["session_number"], ascending=True)

        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        ax.plot(mouse_meta_df["session_number"], mouse_meta_df["percent_beaconed_trials_correct"], color="black")
        ax.plot(mouse_meta_df["session_number"], mouse_meta_df["percent_nonbeaconed_trials_correct"], color="blue")
        ax.set_ylim([0,100])
        ax2=ax.twinx()
        ax.plot(mouse_meta_df["session_number"], mouse_meta_df["trial_type_ratio_numeric"], color="green")

        ax.set_xlabel("Session number", fontsize=25, labelpad = 10)
        ax.set_ylabel("% correct", fontsize=25, labelpad = 10)
        ax2.set_ylabel("trial type ratio", fontsize=25, labelpad = 10)
        ax.xaxis.set_tick_params(labelsize=17)
        ax.yaxis.set_tick_params(labelsize=17)
        ax2.yaxis.set_tick_params(labelsize=17)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/trial_ratio_vs_percentage_success_across_sessions_mouse'+str(mouse_id)+'.png', dpi=200)
        plt.close()


def plot_trial_ratios_against_success(meta_df, save_path):
    X = ['4:1','3:1','2:1','1:1', '1:2','3:7','1:3']
    X_axis = np.arange(len(X))

    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    success_u = meta_df.groupby(by=["trial_type_ratio_numeric"], as_index=False).mean()
    success_u= success_u.sort_values(by="trial_type_ratio_numeric", ascending=False)
    success_sem = meta_df.groupby(by=["trial_type_ratio_numeric"], as_index=False).sem()
    success_sem= success_sem.sort_values(by="trial_type_ratio_numeric", ascending=False)
    ax.bar(X_axis-0.2, success_u["percent_beaconed_trials_correct"], 0.4, color="black",alpha=0.6)
    ax.bar(X_axis+0.2, success_u["percent_nonbeaconed_trials_correct"], 0.4, color ='blue',alpha=0.6)
    ax.errorbar(X_axis-0.2, success_u["percent_beaconed_trials_correct"], yerr=success_sem["percent_beaconed_trials_correct"], color="black",fmt='none', capsize=5)
    ax.errorbar(X_axis+0.2, success_u["percent_nonbeaconed_trials_correct"], yerr=success_sem["percent_nonbeaconed_trials_correct"], color="black",fmt='none', capsize=5)
    plt.xticks(X_axis, X)
    plt.xlabel("B/NB ratio", fontsize=25, labelpad = 10)
    plt.ylabel("% correct", fontsize=25, labelpad = 10)
    plt.ylim(0,100)
    ax.xaxis.set_tick_params(labelsize=17)
    ax.yaxis.set_tick_params(labelsize=17)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/trial_ratio_vs_percentage_success.png', dpi=200)
    plt.close()

    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    success_u = meta_df.groupby(by=["trial_type_ratio_numeric"], as_index=False).mean()
    success_u= success_u.sort_values(by="trial_type_ratio_numeric", ascending=False)
    success_sem = meta_df.groupby(by=["trial_type_ratio_numeric"], as_index=False).sem()
    success_sem= success_sem.sort_values(by="trial_type_ratio_numeric", ascending=False)
    ax.bar(X_axis-0.2, success_u["n_beaconed_trials_correct"], 0.4, color="black",alpha=0.6)
    ax.bar(X_axis+0.2, success_u["n_nonbeaconed_trials_correct"], 0.4, color ='blue',alpha=0.6)
    ax.errorbar(X_axis-0.2, success_u["n_beaconed_trials_correct"], yerr=success_sem["n_beaconed_trials_correct"], color="black",fmt='none', capsize=5)
    ax.errorbar(X_axis+0.2, success_u["n_nonbeaconed_trials_correct"], yerr=success_sem["n_nonbeaconed_trials_correct"], color="black",fmt='none', capsize=5)
    plt.xticks(X_axis, X)
    plt.xlabel("B/NB ratio", fontsize=25, labelpad = 10)
    plt.ylabel("Number of trials", fontsize=25, labelpad = 10)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_tick_params(labelsize=17)
    ax.yaxis.set_tick_params(labelsize=17)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/trial_ratio_vs_number_of_trials_success.png', dpi=200)
    plt.close()

def population_shuffled_vs_training_day_fs(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    bin_size = 5
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            shuffled_vs_peaks = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)>0:
                    track_length = session_df.track_length.iloc[0]
                    rz_start = track_length-60-30-20
                    rz_end = track_length-60-30

                    session_df = curate_stops(session_df, track_length) # filter stops
                    tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df['first_stop_location_cm'])

                    # calculate trial type stops per trial
                    tt_hist, bin_edges = np.histogram(tt_stops, bins=int(track_length/bin_size), range=(0, track_length))
                    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
                    tt_hist_RZ = tt_hist[(bin_centres > rz_start) & (bin_centres < rz_end)]
                    measured_peak = max(tt_hist_RZ/len(session_df))

                    # calculate changce level peak
                    shuffle_peaks = []
                    for i in enumerate(np.arange(shuffles)):
                        shuffled_stops = np.random.uniform(low=0, high=track_length, size=len(tt_stops))
                        shuffled_stop_hist, bin_edges = np.histogram(shuffled_stops, bins=int(track_length/bin_size), range=(0, track_length))
                        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
                        shuffled_stop_hist_RZ = shuffled_stop_hist[(bin_centres > rz_start) & (bin_centres < rz_end)]

                        peak = max(shuffled_stop_hist_RZ/len(session_df))
                        shuffle_peaks.append(peak)
                    shuffle_peaks = np.array(shuffle_peaks)
                    threshold = np.nanpercentile(shuffle_peaks, percentile)

                    peak_vs_shuffle = measured_peak-threshold

                    shuffled_vs_peaks.append(peak_vs_shuffle)
                    session_numbers.append(session_number)

                    mouse_array[tt, mouse_i, session_number-1] = peak_vs_shuffle

            mouse_i +=1

            # plot per mouse
            ax.plot(session_numbers, shuffled_vs_peaks, '-', label=mouse_id, color=mouse_color)
        plt.ylabel('First stop peak / trial\n vs shuffle', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(-1,0.5)
        ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/shuffle_vs_training_day_fs_tt_'+str(tt)+'_by_mouse.png', dpi=200)
        plt.close()

        # plot average across mouse
        tt_mouse_array = mouse_array[tt]

        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
        plt.ylabel('First stop peak / trial\n vs shuffle', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(-1,0.5)
        ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/shuffle_vs_training_day_fs_tt_'+str(tt)+'.png', dpi=200)
        plt.close()

    # plot average across mouse for all trial types
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_mouse_array = mouse_array[tt]
        nan_mask = ~np.isnan(np.nanmean(tt_mouse_array, axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
    plt.ylabel('First stop peak / trial\n vs shuffle', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    plt.ylim(-1,0.5)
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/shuffle_vs_training_day_fs_all_tt.png', dpi=200)
    plt.close()

def population_shuffled_vs_training_day_numbers_stops(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    bin_size = 5
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            percent_in_RZ = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)>0:
                    track_length = session_df.track_length.iloc[0]
                    rz_start = track_length-60-30-20
                    rz_end = track_length-60-30

                    session_df = curate_stops(session_df, track_length) # filter stops
                    tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["stop_location_cm"])
                    tt_stops_in_RZ = tt_stops[(tt_stops >= rz_start) & (tt_stops <= rz_end)]
                    if len(tt_stops)==0:
                        percent = 0
                    else:
                        percent = len(tt_stops_in_RZ)/len(session_df)
                    percent_in_RZ.append(percent)
                    session_numbers.append(session_number)

                    mouse_array[tt, mouse_i, session_number-1] = percent

            mouse_i +=1

            # plot per mouse
            ax.plot(session_numbers, percent_in_RZ, '-', label=mouse_id, color=mouse_color)
        plt.ylabel('stops / trial in RZ', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        #plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/number_stops_vs_training_day_tt_'+str(tt)+'_by_mouse.png', dpi=200)
        plt.close()

        # plot average across mouse
        tt_mouse_array = mouse_array[tt]

        # do stats test
        df = pd.DataFrame({'mouse_id': np.repeat(np.arange(0,len(mouse_ids)), len(tt_mouse_array[0])),
                           'session_id': np.tile(np.arange(0,max_session),  len(tt_mouse_array)),
                           'n_stops_per_trial': tt_mouse_array.flatten()})
        df = df.dropna()
        df = reassign_session_numbers(df)
        # Conduct the repeated measures ANOVA
        df = df[df["mouse_id"]!=0]
        df = df[df["mouse_id"]!=4]
        df = df[df["mouse_id"]!=6]
        df = df[df["mouse_id"]!=11]
        df = df[df["session_id"]<25]
        if tt == 0 or tt==1:
            a = AnovaRM(data=df, depvar='n_stops_per_trial', subject='mouse_id', within=['session_id'], aggregate_func='mean').fit()
            print("tt = ", str(tt), ", test for n_stops_per_trial in RZ")
            print("p= ", str(a.anova_table["Pr > F"].iloc[0]), ", Num DF= ", str(a.anova_table["Num DF"].iloc[0]), ", Num DF= ", str(a.anova_table["Den DF"].iloc[0]), "F value= ", str(a.anova_table["F Value"].iloc[0]))

        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
        plt.ylabel('n stops in RZ', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        #plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/number_stops_vs_training_day_tt_'+str(tt)+'.png', dpi=200)
        plt.close()

    # plot average across mouse for all trial types
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_mouse_array = mouse_array[tt]
        nan_mask = ~np.isnan(np.nanmean(tt_mouse_array, axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)


    for session in np.arange(1,max_session+1)-1:
        # do pairwise comparison
        f, p = stats.wilcoxon(mouse_array[0,:,session], mouse_array[1,:,session], nan_policy="omit")
        if p<0.05:
            ax.text(session, 3.5, "*", fontsize=20, color="red")
    plt.ylabel('n stops in RZ', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    #plt.ylim(0, 100)
    #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/number_stops_vs_training_day_all_tt.png', dpi=200)
    plt.close()


def population_shuffled_vs_training_day_numbers_stops_all_track(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    bin_size = 5
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            n_in_track = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)>0:
                    track_length = session_df.track_length.iloc[0]
                    rz_start = track_length-60-30-20
                    rz_end = track_length-60-30

                    session_df = curate_stops(session_df, track_length) # filter stops
                    tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["stop_location_cm"])
                    if len(tt_stops)==0:
                        percent = 0
                    else:
                        percent = len(tt_stops)/len(session_df)
                    n_in_track.append(percent)
                    session_numbers.append(session_number)

                    mouse_array[tt, mouse_i, session_number-1] = percent

            mouse_i +=1

            # plot per mouse
            ax.plot(session_numbers, n_in_track, '-', label=mouse_id, color=mouse_color)
        plt.ylabel('stops / trial', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        #plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/all_track_number_stops_vs_training_day_tt_'+str(tt)+'_by_mouse.png', dpi=200)
        plt.close()

        # plot average across mouse
        tt_mouse_array = mouse_array[tt]

        # do stats test
        df = pd.DataFrame({'mouse_id': np.repeat(np.arange(0,len(mouse_ids)), len(tt_mouse_array[0])),
                           'session_id': np.tile(np.arange(0,max_session),  len(tt_mouse_array)),
                           'n_stops_per_trial_whole_track': tt_mouse_array.flatten()})
        df = df.dropna()
        df = reassign_session_numbers(df)
        # Conduct the repeated measures ANOVA
        df = df[df["mouse_id"]!=0]
        df = df[df["mouse_id"]!=4]
        df = df[df["mouse_id"]!=6]
        df = df[df["mouse_id"]!=11]
        df = df[df["session_id"]<25]
        if tt == 0 or tt==1:
            a = AnovaRM(data=df, depvar='n_stops_per_trial_whole_track', subject='mouse_id', within=['session_id'], aggregate_func='mean').fit()
            print("tt = ", str(tt), ", test for n_stops_per_trial_whole_track")
            print("p= ", str(a.anova_table["Pr > F"].iloc[0]), ", Num DF= ", str(a.anova_table["Num DF"].iloc[0]), ", Num DF= ", str(a.anova_table["Den DF"].iloc[0]), "F value= ", str(a.anova_table["F Value"].iloc[0]))

        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
        plt.ylabel('stops / trial', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        #plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/all_track_number_stops_vs_training_day_tt_'+str(tt)+'.png', dpi=200)
        plt.close()

    # plot average across mouse for all trial types
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_mouse_array = mouse_array[tt]
        nan_mask = ~np.isnan(np.nanmean(tt_mouse_array, axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
    plt.ylabel('stops / trial', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)

    for session in np.arange(1,max_session+1)-1:
        # do pairwise comparison
        f, p = stats.wilcoxon(mouse_array[0,:,session], mouse_array[1,:,session], nan_policy="omit")
        if p<0.05:
            ax.text(session, 30, "*", fontsize=20, color="red")
    plt.xlim(1,max_session)
    #plt.ylim(0, 100)
    #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/all_track_number_stops_vs_training_day_all_tt.png', dpi=200)
    plt.close()

def population_shuffled_vs_training_day_percentage_stops(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    bin_size = 5
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            percent_in_RZ = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)>0:
                    track_length = session_df.track_length.iloc[0]
                    rz_start = track_length-60-30-20
                    rz_end = track_length-60-30

                    session_df = curate_stops(session_df, track_length) # filter stops
                    tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["stop_location_cm"])
                    tt_stops_in_RZ = tt_stops[(tt_stops >= rz_start) & (tt_stops <= rz_end)]
                    if len(tt_stops)==0:
                        percent = 0
                    else:
                        percent = (len(tt_stops_in_RZ)/len(tt_stops))*100
                    percent_in_RZ.append(percent)
                    session_numbers.append(session_number)

                    mouse_array[tt, mouse_i, session_number-1] = percent

            mouse_i +=1

            # plot per mouse
            ax.plot(session_numbers, percent_in_RZ, '-', label=mouse_id, color=mouse_color)
        plt.ylabel('% stops in RZ', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/percentage_stops_vs_training_day_tt_'+str(tt)+'_by_mouse.png', dpi=200)
        plt.close()

        # plot average across mouse
        tt_mouse_array = mouse_array[tt]

        # do stats test
        df = pd.DataFrame({'mouse_id': np.repeat(np.arange(0,len(mouse_ids)), len(tt_mouse_array[0])),
                           'session_id': np.tile(np.arange(0,max_session),  len(tt_mouse_array)),
                           'proportion_stops_in_rz': tt_mouse_array.flatten()})
        df = df.dropna()
        df = reassign_session_numbers(df)
        # Conduct the repeated measures ANOVA
        df = df[df["mouse_id"]!=0]
        df = df[df["mouse_id"]!=4]
        df = df[df["mouse_id"]!=6]
        df = df[df["mouse_id"]!=11]
        df = df[df["session_id"]<25]
        if tt == 0 or tt==1:
            a = AnovaRM(data=df, depvar='proportion_stops_in_rz', subject='mouse_id', within=['session_id'], aggregate_func='mean').fit()
            print("tt = ", str(tt), ", test for proportion_stops_in_rz")
            print("p= ", str(a.anova_table["Pr > F"].iloc[0]), ", Num DF= ", str(a.anova_table["Num DF"].iloc[0]), ", Num DF= ", str(a.anova_table["Den DF"].iloc[0]), "F value= ", str(a.anova_table["F Value"].iloc[0]))

        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
        plt.ylabel('% stops in RZ', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(0, 100)
        #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/percentage_stops_vs_training_day_tt_'+str(tt)+'.png', dpi=200)
        plt.close()

    # plot average across mouse for all trial types
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_mouse_array = mouse_array[tt]
        nan_mask = ~np.isnan(np.nanmean(tt_mouse_array, axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)

    for session in np.arange(1,max_session+1)-1:
        # do pairwise comparison
        f, p = stats.wilcoxon(mouse_array[0,:,session], mouse_array[1,:,session], nan_policy="omit")
        if p<0.05:
            ax.text(session, 95, "*", fontsize=20, color="red")
    plt.ylabel('% stops in RZ', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    plt.ylim(0, 100)
    #ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/percentage_stops_vs_training_day_all_tt.png', dpi=200)
    plt.close()

def plot_population_n_trials(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    bin_size = 5
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((len(mouse_ids), max_session)); mouse_array[:, :] = np.nan

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)

    mouse_i = 0
    for mouse_id, mouse_color in zip(mouse_ids, colors):
        mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]

        trial_numbers = []
        session_numbers = []
        for session_number in np.unique(mouse_df.session_number):
            session_df = mouse_df[mouse_df["session_number"] == session_number]

            trial_numbers.append(len(session_df))
            session_numbers.append(session_number)
            mouse_array[mouse_i, session_number-1] = len(session_df)
        mouse_i +=1

        # plot per mouse
        ax.plot(session_numbers, trial_numbers, '-', label=mouse_id, color=mouse_color)
    plt.ylabel('n trials', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/n_trials_by_mouse.png', dpi=200)
    plt.close()

    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    nan_mask = ~np.isnan(np.nanmean(mouse_array, axis=0))
    ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(mouse_array, axis=0)-stats.sem(mouse_array, axis=0, nan_policy="omit"))[nan_mask], (np.nanmean(mouse_array, axis=0)+stats.sem(mouse_array, axis=0, nan_policy="omit"))[nan_mask], color="black", alpha=0.3)
    ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(mouse_array, axis=0)[nan_mask], color="black")
    plt.ylabel('n trials', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/n_trials_by_day.png', dpi=200)
    plt.close()



def population_shuffled_vs_training_day(all_behaviour200cm_tracks, save_path, percentile=99, shuffles=1000):
    max_session = 30
    all_behaviour200cm_tracks = all_behaviour200cm_tracks[all_behaviour200cm_tracks["session_number"] <= max_session]
    bin_size = 5
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]

    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    mouse_array = np.zeros((3, len(mouse_ids), max_session)); mouse_array[:, :] = np.nan
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        # plot figure
        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)

        mouse_i = 0
        for mouse_id, mouse_color in zip(mouse_ids, colors):
            mouse_df = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == mouse_id]
            tt_session_df = mouse_df[mouse_df["trial_type"] == tt]

            shuffled_vs_peaks = []
            session_numbers = []
            for session_number in np.unique(tt_session_df.session_number):
                session_df = tt_session_df[tt_session_df["session_number"] == session_number]
                session_df = drop_first_and_last_trial(session_df)

                if len(session_df)>0:
                    track_length = session_df.track_length.iloc[0]
                    rz_start = track_length-60-30-20
                    rz_end = track_length-60-30

                    session_df = curate_stops(session_df, track_length) # filter stops
                    tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["stop_location_cm"])

                    # calculate trial type stops per trial
                    tt_hist, bin_edges = np.histogram(tt_stops, bins=int(track_length/bin_size), range=(0, track_length))
                    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
                    tt_hist_RZ = tt_hist[(bin_centres > rz_start) & (bin_centres < rz_end)]
                    measured_peak = max(tt_hist_RZ/len(session_df))

                    # calculate changce level peak
                    shuffle_peaks = []
                    for i in enumerate(np.arange(shuffles)):
                        shuffled_stops = np.random.uniform(low=0, high=track_length, size=len(tt_stops))
                        shuffled_stop_hist, bin_edges = np.histogram(shuffled_stops, bins=int(track_length/bin_size), range=(0, track_length))
                        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
                        shuffled_stop_hist_RZ = shuffled_stop_hist[(bin_centres > rz_start) & (bin_centres < rz_end)]

                        peak = max(shuffled_stop_hist_RZ/len(session_df))
                        shuffle_peaks.append(peak)
                    shuffle_peaks = np.array(shuffle_peaks)
                    threshold = np.nanpercentile(shuffle_peaks, percentile)

                    peak_vs_shuffle = measured_peak-threshold

                    shuffled_vs_peaks.append(peak_vs_shuffle)
                    session_numbers.append(session_number)

                    mouse_array[tt, mouse_i, session_number-1] = peak_vs_shuffle

            mouse_i +=1

            # plot per mouse
            ax.plot(session_numbers, shuffled_vs_peaks, '-', label=mouse_id, color=mouse_color)
        plt.ylabel('Peak stops / trial\n vs shuffle', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(-4,2)
        ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/shuffle_vs_training_day_tt_'+str(tt)+'_by_mouse.png', dpi=200)
        plt.close()

        # plot average across mouse
        tt_mouse_array = mouse_array[tt]

        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)
        plt.ylabel('Peak stops / trial\n vs shuffle', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)
        plt.xlim(1,max_session)
        plt.ylim(-4,2)
        ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        tick_spacing = 10
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(fontsize=20)
        Edmond.plot_utility2.style_vr_plot(ax)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/shuffle_vs_training_day_tt_'+str(tt)+'.png', dpi=200)
        plt.close()

    # plot average across mouse for all trial types
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_mouse_array = mouse_array[tt]
        nan_mask = ~np.isnan(np.nanmean(tt_mouse_array, axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-np.nanstd(tt_mouse_array, axis=0))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+np.nanstd(tt_mouse_array, axis=0))[nan_mask], color=tt_color, alpha=0.3)
        ax.plot(np.arange(1,max_session+1)[nan_mask], np.nanmean(tt_mouse_array, axis=0)[nan_mask], color=tt_color)


    for session in np.arange(1,max_session+1)-1:
        # do pairwise comparison
        f, p = stats.wilcoxon(mouse_array[0,:,session], mouse_array[1,:,session], nan_policy="omit")
        if p<0.05:
            ax.text(session, 2, "*", fontsize=20, color="red")

    plt.ylabel('Peak stops / trial\n vs shuffle', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,max_session)
    plt.ylim(-4,2)
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/shuffle_vs_training_day_all_tt.png', dpi=200)
    plt.close()

def print_population_stats(all_behaviour200cm_tracks):
    print("hello there")


def process_recordings(vr_recording_path_list, all_behaviour, cohort):
    print(" ")
    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            session_id = recording.split("/")[-1]
            mouse_id = session_id.split("_")[0]
            track_length = get_track_length(recording)
            session_number = int(session_id.split("_")[1].split("D")[-1])

            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            processed_position_data = add_avg_trial_speed(processed_position_data)
            processed_position_data = add_avg_RZ_speed(processed_position_data, track_length=track_length)
            processed_position_data = add_avg_track_speed(processed_position_data, track_length=track_length)
            processed_position_data, _ = add_hit_miss_try3(processed_position_data, track_length=track_length)
            processed_position_data["session_number"] = session_number
            processed_position_data["mouse_id"] = mouse_id
            processed_position_data["track_length"] = track_length
            processed_position_data["cohort"] = cohort

            all_behaviour = pd.concat([all_behaviour, processed_position_data], ignore_index=True)
            print("successfully processed and saved vr_grid analysis on "+recording)
        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)

    return all_behaviour

def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found

def add_reward_ratio(session_data):
    mouse_id = session_data.mouse_id.iloc[0]
    session_number = session_data.session_number.iloc[0]
    if mouse_id in ["M10", "M11", "M12", "M13", "M14", "M15"]:
        if session_number >= 15:
            reward_ratio = 3
        else:
            reward_ratio = 1
    else:
        reward_ratio = 1
    return reward_ratio

def add_nb_expected_reward_from_the_session(ratio_numeric, reward_ratio):
    trial_type_ratio = ratio_numeric
    beaconed_dispense = 1
    non_beaconed_dispense = beaconed_dispense*reward_ratio
    n_b_trials = 1*trial_type_ratio
    n_nb_trials = 1*(1-trial_type_ratio)
    beaconed_reward = beaconed_dispense*n_b_trials
    non_beaconed_reward = non_beaconed_dispense*n_nb_trials
    nb_expected_reward = 100*non_beaconed_reward/(beaconed_reward+non_beaconed_reward)
    return nb_expected_reward

def generate_metadata(behaviour_df):
    meta = pd.DataFrame()

    for cohort in np.unique(behaviour_df["cohort"]):
        cohort_data = behaviour_df[behaviour_df["cohort"] == cohort]
        for mouse_id in np.unique(cohort_data["mouse_id"]):
            mouse_data = cohort_data[cohort_data["mouse_id"] == mouse_id]
            for session_number in np.unique(mouse_data["session_number"]):
                session_data = mouse_data[(mouse_data["session_number"] == session_number)]

                n_trials = len(session_data)
                n_probe_trials = len(session_data[session_data["trial_type"] == 2])
                n_beaconed_trials = len(session_data[session_data["trial_type"] == 0])
                n_nonbeaconed_trials = len(session_data[session_data["trial_type"] == 1])
                n_probe_trials_correct = len(session_data[(session_data["trial_type"] == 2) & (session_data["hit_miss_try"] == "hit")])
                n_beaconed_trials_correct = len(session_data[(session_data["trial_type"] == 0) & (session_data["hit_miss_try"] == "hit")])
                n_nonbeaconed_trials_correct = len(session_data[(session_data["trial_type"] == 1) & (session_data["hit_miss_try"] == "hit")])
                track_length = session_data["track_length"].iloc[0]
                mouse_id = session_data["mouse_id"].iloc[0]
                session_number = session_data["session_number"].iloc[0]
                reward_ratio = add_reward_ratio(session_data)
                cohort = session_data["cohort"].iloc[0]
                cohort_mouse = str(cohort)+"_"+mouse_id

                # search for the ratio used out of 4:1, 3:1, 2:1, 3:2
                trial_types = np.array(session_data["trial_type"])
                trial_types[trial_types==2] =1 # replace 2s with 1s for the trial type ratio
                if len(search_sequence_numpy(trial_types, np.array([0,0,0,0,1])))>0:
                    ratio= "4:1"
                    ratio_numeric = 4/5
                elif len(search_sequence_numpy(trial_types, np.array([0,0,0,1])))>0:
                    ratio= "3:1"
                    ratio_numeric = 3/4
                elif len(search_sequence_numpy(trial_types, np.array([0,0,1,0])))>0:
                    ratio= "2:1"
                    ratio_numeric = 2/3
                elif len(search_sequence_numpy(trial_types, np.array([0,1,0,1,0,1])))>0:
                    ratio = "1:1"
                    ratio_numeric = 1/2
                elif len(search_sequence_numpy(trial_types, np.array([0,0,1,1,1])))>0:
                    ratio = "3:7"
                    ratio_numeric =3/10
                elif len(search_sequence_numpy(trial_types, np.array([0,1,1,0,1,1,0])))>0:
                    ratio = "1:2"
                    ratio_numeric = 1/3
                elif len(search_sequence_numpy(trial_types, np.array([0,1,1,1])))>0:
                    ratio = "1:3"
                    ratio_numeric = 1/4
                else:
                    ratio= ""
                    ratio_numeric = np.nan

                non_beaconed_expected_reward_session = add_nb_expected_reward_from_the_session(ratio_numeric, reward_ratio)

                # make the session meta data and concatenate
                session_meta = pd.DataFrame()
                session_meta["n_trials"] = [n_trials]
                session_meta["n_probe_trials"] = [n_probe_trials]
                session_meta["n_beaconed_trials"] = [n_beaconed_trials]
                session_meta["n_nonbeaconed_trials"] = [n_nonbeaconed_trials]
                session_meta["n_probe_trials_correct"] =[n_probe_trials_correct]
                session_meta["n_beaconed_trials_correct"] = [n_beaconed_trials_correct]
                session_meta["n_nonbeaconed_trials_correct"] = [n_nonbeaconed_trials_correct]
                session_meta["percent_probe_trials_correct"] =[100*pass_div_by_zero(n_probe_trials_correct, n_probe_trials)]
                session_meta["percent_beaconed_trials_correct"] = [100*pass_div_by_zero(n_beaconed_trials_correct, n_beaconed_trials)]
                session_meta["percent_nonbeaconed_trials_correct"] = [100*pass_div_by_zero(n_nonbeaconed_trials_correct, n_nonbeaconed_trials)]
                session_meta["trial_type_ratio"] = [ratio]
                session_meta["trial_type_ratio_numeric"] = [ratio_numeric]
                session_meta["reward_ratio"] = [reward_ratio]
                session_meta["non_beaconed_expected_reward_session"] = [non_beaconed_expected_reward_session]
                session_meta["track_length"] = [track_length]
                session_meta["mouse_id"] = [mouse_id]
                session_meta["session_number"] = [session_number]
                session_meta["cohort"] = [cohort]
                session_meta["cohort_mouse"] = [cohort_mouse]

                meta = pd.concat([meta, session_meta], ignore_index=True)

    for track_length in np.unique(meta["track_length"]):
        track_length_meta = meta[meta["track_length"] == track_length]
        print("for track length", str(track_length))
        print("N mice = ", str(len(np.unique(track_length_meta["cohort_mouse"]))))
        print("N sessions = ", str(len(track_length_meta)))
    return meta

def plot_percentage_fast_hit_differential_for_more_frequent_beaconed_trials(meta_behaviour_data, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/meta_analysis"):
    meta_behaviour_data = meta_behaviour_data[meta_behaviour_data["trial_type_ratio_numeric"]>1]

    # plot figure
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    mouse_ids = ["M1", "M2", "M3", "M4", "M6", "M7",  "M10",  "M11",  "M12", "M13", "M14", "M15"]
    colors = cm.Paired(np.linspace(0, 1, len(mouse_ids)))

    for mouse_id, color in zip(mouse_ids, colors):
        mouse_meta = meta_behaviour_data[meta_behaviour_data["mouse_id"] == mouse_id]

        # plot per mouse
        ax.plot(np.array(mouse_meta["session_number"]), np.array(mouse_meta["percent_beaconed_trials_correct"])-np.array(mouse_meta["percent_nonbeaconed_trials_correct"]), '-', color=color)

    plt.ylabel('B-NB % fast hits', fontsize=25, labelpad = 10)
    plt.xlabel('Session number', fontsize=25, labelpad = 10)
    plt.xlim(1,30)
    plt.ylim(-100, 100)
    ax.axhline(y=0, linestyle="dashed", linewidth=3, color="black")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_visible(False)
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    Edmond.plot_utility2.style_vr_plot(ax)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.9)
    plt.savefig(save_path + '/fast_hit_differential_for_more_frequent_beaconed_trials.png', dpi=200)
    plt.close()
    return

def plot_hit_percentage_beaconed_and_non_beaconed_for_all_mice(meta_behaviour_data, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/meta_analysis"):
    return

def pass_div_by_zero(a, b):
    if b == 0:
        return 0
    else:
        return a/b

def plot_trial_type_correct_vs_trial_type_expected_reward(meta_behaviour_data, save_path):
    meta_behaviour_data = meta_behaviour_data[meta_behaviour_data["session_number"] <= 30]

    plt.close()
    for cohort in np.unique(meta_behaviour_data["cohort"]):
        meta_cohort = meta_behaviour_data[meta_behaviour_data["cohort"] == cohort]
        for mouse_id in np.unique(meta_cohort["mouse_id"]):
            mouse_meta = meta_cohort[meta_cohort["mouse_id"] == mouse_id]
            fig, ax = plt.subplots(figsize=(6,6))
            plt.title(mouse_id, fontsize=20)
            ax.plot(mouse_meta["session_number"], mouse_meta["percent_beaconed_trials_correct"], "-", color="black")
            ax.plot(mouse_meta["session_number"], mouse_meta["percent_nonbeaconed_trials_correct"], "-", color="blue")
            cmap = matplotlib.cm.get_cmap('PuOr')
            for x, color in zip(mouse_meta["session_number"], mouse_meta["non_beaconed_expected_reward_session"]):
                ax.scatter(x, 105, color=cmap(color/100), marker="s", s=50)
            ax.tick_params(axis='both', which='major', labelsize=15)
            fig.tight_layout()
            #ax.set_ylim(bottom=0, top=100)
            ax.set_xlim(left=0, right=30)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel("% Fast hit trials", fontsize=20)
            ax.set_xlabel("Training day", fontsize=20)
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
            plt.savefig(save_path+'/emp_'+str(cohort)+"_"+mouse_id+'.png', dpi=300)
            plt.close()

    for cohort in np.unique(meta_behaviour_data["cohort"]):
        meta_cohort = meta_behaviour_data[meta_behaviour_data["cohort"] == cohort]
        for mouse_id in np.unique(meta_cohort["mouse_id"]):
            mouse_meta = meta_cohort[meta_cohort["mouse_id"] == mouse_id]
            fig, ax = plt.subplots(figsize=(6,6))
            plt.title(mouse_id, fontsize=20)
            ax.plot(mouse_meta["session_number"], mouse_meta["n_beaconed_trials_correct"], "-", color="black")
            ax.plot(mouse_meta["session_number"], mouse_meta["n_nonbeaconed_trials_correct"], "-", color="blue")
            cmap = matplotlib.cm.get_cmap('PuOr')
            for x, color in zip(mouse_meta["session_number"], mouse_meta["non_beaconed_expected_reward_session"]):
                ax.scatter(x, 300, color=cmap(color/100), marker="s", s=50)
            ax.tick_params(axis='both', which='major', labelsize=15)
            fig.tight_layout()
            #ax.set_ylim(bottom=0, top=100)
            ax.set_xlim(left=0, right=30)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel("N Fast hit trials", fontsize=20)
            ax.set_xlabel("Training day", fontsize=20)
            plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
            plt.savefig(save_path+'/emp_n_trials_'+str(cohort)+"_"+mouse_id+'.png', dpi=300)
            plt.close()
    return

def plot_expected_reward_schematic(save_path, probe=False):
    y_trial_type_ratios = np.array([1/5, 1/4, 1/3, 1/2, 2/3, 3/4, 4/5])
    x_reward_ratios = np.array([1/4, 1/3, 1/2, 1/1, 2/1, 3/1, 4/1])
    y = np.arange(0, 7)
    x = np.arange(0, 7)

    n_trials = 1
    beaconed_dispense = 1
    # calculate the expected reward from beaconed and non beaconed trials
    # and then calculate the relative weight of non beaconed trials to the total expected reward
    Z = np.zeros((len(x_reward_ratios), len(y_trial_type_ratios)))
    for i in range(len(Z[0])):
        for j in range(len(Z)):

            trial_type_ratio = y_trial_type_ratios[i]
            reward_ratio = x_reward_ratios[j]

            non_beaconed_dispense = beaconed_dispense*reward_ratio

            n_b_trials = n_trials*trial_type_ratio
            n_nb_trials = n_trials*(1-trial_type_ratio)

            beaconed_reward = beaconed_dispense*n_b_trials
            non_beaconed_reward = non_beaconed_dispense*n_nb_trials

            if probe: # if probe every 2 non beaconed trials then rewards from non cued trials are halved
                non_beaconed_reward = non_beaconed_reward/2

            Z[i, j] = 100*non_beaconed_reward/(beaconed_reward+non_beaconed_reward)


    fig, ax = plt.subplots(figsize=(6,6))
    X, Y = np.meshgrid(x, y)
    pcm = ax.pcolormesh(X, Y, Z, shading="nearest", vmin=0, vmax=100, cmap='PuOr',zorder=-1)

    # annotate cells
    for i in range(len(Z[0])):
        for j in range(len(Z)):
            plt.text(j, i, str(int(np.round(Z[i, j]))), horizontalalignment='center', verticalalignment='center', fontsize=12, zorder=2)

    #ax.set_ylabel("Trial Counts", fontsize=20)
    #ax.set_xlabel("TI", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()
    #ax.set_ylim(bottom=0)
    #ax.set_xlim(left=0)
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.1)
    cbar.set_ticks([0,100])
    cbar.set_ticklabels(["0","100"])
    cbar.ax.tick_params(labelsize=20)
    ax.set_xticks(np.arange(0, 7))
    ax.set_yticks(np.arange(0, 7))
    ax.set_ylabel(" ", fontsize=20)
    ax.set_xlabel(" ", fontsize=20)
    ax.set_xticklabels([".25", ".33", ".5", "1", "2", "3", "4"], fontsize=20)
    ax.set_yticklabels(["1:4", "1:3", "1:2", "1:1", "2:1", "3:1", "4:1"], fontsize=20)
    #cbar.set_label('relative weight of ER', rotation=270, fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
    if probe:
        probe_string = "after_probe"
    else:
        probe_string = "before_probe"

    plt.savefig(save_path+'/expected_reward_schematic_'+probe_string+'.png', dpi=300)
    plt.close()

    return

def plot_expected_reward_on_nb_vs_non_beaconed_performance(meta_behaviour_data, save_path):
    meta_behaviour_data = meta_behaviour_data[(meta_behaviour_data["session_number"] <= 30)]

    # filter out mice that did not learn the task
    for mouse_id in np.unique(meta_behaviour_data["mouse_id"]):
        mouse_meta = meta_behaviour_data[meta_behaviour_data["mouse_id"] == mouse_id]


    fig, ax = plt.subplots(figsize=(6,6))
    cmap = matplotlib.cm.get_cmap('bwr')
    for x, color in zip(mouse_meta["session_number"], mouse_meta["non_beaconed_expected_reward_session"]):
        ax.scatter(x, 105, color=cmap(color/100), marker="s", s=50)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.tight_layout()
    #ax.set_ylim(bottom=0, top=100)
    ax.set_xlim(left=0, right=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("% Fast hit trials", fontsize=20)
    ax.set_xlabel("Training day", fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
    plt.savefig(save_path+'/emp.png', dpi=300)
    plt.close()
    return

def plot_performance_against_improvements(meta_behaviour_data, save_path):
    meta_behaviour_data = meta_behaviour_data[(meta_behaviour_data["session_number"] <= 30)]

    # filter out last 5 days of training for all mice
    filtered_meta_behaviour_data = pd.DataFrame()
    for cohort in np.unique(meta_behaviour_data["cohort"]):
        meta_cohort = meta_behaviour_data[meta_behaviour_data["cohort"] == cohort]
        for mouse_id in np.unique(meta_cohort["mouse_id"]):
            meta_mouse = meta_cohort[meta_cohort["mouse_id"] == mouse_id]
            last_day = max(meta_mouse["session_number"])
            meta_mouse_filtered = meta_mouse[(meta_mouse["session_number"] <= last_day) &
                                             (meta_mouse["session_number"] >= last_day-4)]
            filtered_meta_behaviour_data = pd.concat([filtered_meta_behaviour_data, meta_mouse_filtered], ignore_index=True)

    # remove mice that did not learn
    for cohort_mouse in ["2_245", "3_M6", "6_M1", "6_M2", "8_M13", "8_M15"]:
        filtered_meta_behaviour_data = filtered_meta_behaviour_data[(filtered_meta_behaviour_data["cohort_mouse"] != cohort_mouse)]

    # take last 5 day average for all the relavent stats
    averaged_df = pd.DataFrame()
    for cohort_mouse in np.unique(filtered_meta_behaviour_data["cohort_mouse"]):
        cohort_mouse_meta = filtered_meta_behaviour_data[filtered_meta_behaviour_data["cohort_mouse"] == cohort_mouse]

        row = pd.DataFrame()
        for row_name in list(cohort_mouse_meta):
            # check the row name is numeric
            if isinstance(cohort_mouse_meta[row_name].iloc[0], numbers.Number):
                avg = np.nanmean(cohort_mouse_meta[row_name])
            else:
                avg = cohort_mouse_meta[row_name].iloc[0]
            row[row_name] = [avg]
        averaged_df = pd.concat([averaged_df, row], ignore_index=True)

    print("there are this many mice in this comparison")
    print(len(averaged_df))

    x_pos = [0,1,2]
    x_labels = ["Default", "TTR", "TTR+R"]
    Default = averaged_df[(averaged_df["cohort"]>=2) & (averaged_df["cohort"]<=5)]
    TTR = averaged_df[(averaged_df["cohort"]==7)]
    TTR_R = averaged_df[averaged_df["cohort"]==8]

    print("ER for nb, Default = ", str(np.mean(Default["non_beaconed_expected_reward_session"])))
    print("ER for nb, TTR = ", str(np.mean(TTR["non_beaconed_expected_reward_session"])))
    print("ER for nb, TTR_R = ", str(np.mean(TTR_R["non_beaconed_expected_reward_session"])))
    for column_str, color, ylim in zip(["percent_nonbeaconed_trials_correct", "percent_beaconed_trials_correct",
                                  "n_nonbeaconed_trials_correct", "n_beaconed_trials_correct"], ["blue", "black", "blue", "black"],
                                       [100, 100, None, None]):

        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(np.ones(len(Default))*0, np.array(Default[column_str]), color=color, alpha=0.5)
        ax.scatter(np.ones(len(TTR))*1, np.array(TTR[column_str]), color=color, alpha=0.5)
        ax.scatter(np.ones(len(TTR_R))*2, np.array(TTR_R[column_str]), color=color, alpha=0.5)
        ax.errorbar(x=0, y=np.nanmean(Default[column_str]), yerr=stats.sem(Default[column_str], nan_policy="omit"), color=color, capsize=20)
        ax.errorbar(x=1, y=np.nanmean(TTR[column_str]), yerr=stats.sem(TTR[column_str], nan_policy="omit"), color=color, capsize=20)
        ax.errorbar(x=2, y=np.nanmean(TTR_R[column_str]), yerr=stats.sem(TTR_R[column_str], nan_policy="omit"), color=color, capsize=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='both', which='major', labelsize=15)
        fig.tight_layout()
        ax.set_ylim(bottom=0, top=ylim)
        ax.set_xlim(left=-0.5, right=2.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(column_str, fontsize=20)
        #ax.set_xlabel("Training day", fontsize=20)
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)
        plt.savefig(save_path+'/expected_reward_vs_performance_'+column_str+'.png', dpi=300)
        plt.close()

        print("comparing column "+column_str+" across expected_reward conditions Default vs TTR, df=",str(len(Default)+len(TTR)-2), ", p= ", str(stats.mannwhitneyu(np.array(Default[column_str]), np.array(TTR[column_str]))[1]), ", t= ", str(stats.mannwhitneyu(np.array(Default[column_str]), np.array(TTR[column_str]))[0]))
        print("comparing column "+column_str+" across expected_reward conditions Default vs TTR_R, df=",str(len(Default)+len(TTR_R)-2), ", p= ", str(stats.mannwhitneyu(np.array(Default[column_str]), np.array(TTR_R[column_str]))[1]), ", t= ", str(stats.mannwhitneyu(np.array(Default[column_str]), np.array(TTR_R[column_str]))[0]))
        print("comparing column "+column_str+" across expected_reward conditions TTR vs TTR_R, df=",str(len(TTR)+len(TTR_R)-2), ", p= ", str(stats.mannwhitneyu(np.array(TTR[column_str]), np.array(TTR_R[column_str]))[1]), ", t= ", str(stats.mannwhitneyu(np.array(TTR[column_str]), np.array(TTR_R[column_str]))[0]))


    return

def plot_histogram_all_track_lengths(behaviour_df,save_path, first_stops=False):
    if first_stops:
        suffix="fs"
    else:
        suffix=""
    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/1)

    for track_length in np.unique(behaviour_df["track_length"]):
        track_length_behaviour_df = behaviour_df[behaviour_df["track_length"]==track_length]

        if track_length == 200:
            track_length_behaviour_df = track_length_behaviour_df[track_length_behaviour_df["session_number"] <= 30]
            track_length_behaviour_df = track_length_behaviour_df[track_length_behaviour_df["session_number"] >= 25]

        bin_size = 1
        track_length_behaviour_df = curate_stops(track_length_behaviour_df, 200)  # filter stops

        fig, ax = plt.subplots(figsize=(6, 4))
        for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
            tt_df = track_length_behaviour_df[track_length_behaviour_df["trial_type"]==tt]
            if first_stops:
                tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(tt_df['first_stop_location_cm'])
            else:
                tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(tt_df["stop_location_cm"])
            tt_hist, bin_edges = np.histogram(tt_stops,range=(0,track_length), bins=int(track_length/bin_size))
            bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            tt_hist = convolve(tt_hist, gauss_kernel)

            ax.plot(bin_centres, tt_hist/np.sum(tt_hist), color=tt_color)
        ax.tick_params(axis='both', which='major', labelsize=20)
        style_track_plot(ax, track_length)
        ax.set_ylim(bottom=0)
        ax.axvline(x=93, color="red", linestyle="solid", linewidth=3)
        ax.set_xlim(left=0, right=track_length)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("Stop density", fontsize=25, labelpad=10)
        ax.set_xlabel("Location (cm)", fontsize=25, labelpad=10)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/trained_stop_hist_'+suffix+"_tracklength_"+str(track_length)+'.png', dpi=300)
        plt.close()
    return


def plot_histogram_first_and_last_5_days_stops(behaviour_df,save_path, first_stops=False):
    if first_stops:
        suffix="fs"
    else:
        suffix=""

    max_session = 30
    behaviour_df = behaviour_df[behaviour_df["session_number"] <= max_session]
    bin_size = 1
    behaviour_df = curate_stops(behaviour_df, 200)  # filter stops
    first_five_days_df = behaviour_df[behaviour_df["session_number"] <= 5]
    last_five_days_df = behaviour_df[behaviour_df["session_number"] >= max_session-5]

    fig, ax = plt.subplots(figsize=(6, 4))
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_df = first_five_days_df[first_five_days_df["trial_type"]==tt]
        if first_stops:
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(tt_df['first_stop_location_cm'])
        else:
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(tt_df["stop_location_cm"])
        tt_hist, bin_edges = np.histogram(tt_stops,range=(0,200), bins=int(200/bin_size))
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        ax.plot(bin_centres, tt_hist/np.sum(tt_hist), color=tt_color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    style_track_plot(ax, 200)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=200)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Stop density", fontsize=25, labelpad=10)
    ax.set_xlabel("Location (cm)", fontsize=25, labelpad=10)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/first_five_days_stop_hist_'+suffix+'.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    for tt, tt_color in zip([0,1,2], ["Black", "Blue", "deepskyblue"]):
        tt_df = last_five_days_df[last_five_days_df["trial_type"]==tt]
        if first_stops:
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(tt_df['first_stop_location_cm'])
        else:
            tt_stops = Edmond.plot_utility2.pandas_collumn_to_numpy_array(tt_df["stop_location_cm"])
        tt_hist, bin_edges = np.histogram(tt_stops,range=(0,200), bins=int(200/bin_size))
        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        ax.plot(bin_centres, tt_hist/np.sum(tt_hist), color=tt_color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    style_track_plot(ax, 200)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=200)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Stop density", fontsize=25, labelpad=10)
    ax.set_xlabel("Location (cm)", fontsize=25, labelpad=10)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/last_five_days_stop_hist_'+suffix+'.png', dpi=300)
    plt.close()
    return



def main():
    print('-------------------------------------------------------------')

    '''
    # give a path for a directory of recordings or path of a single recording
    all_behaviour = pd.DataFrame()
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/VirtualReality") if f.is_dir()]
    all_behaviour = process_recordings(vr_path_list, all_behaviour, cohort=2)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality") if f.is_dir()]
    all_behaviour = process_recordings(vr_path_list, all_behaviour, cohort=3)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality") if f.is_dir()]
    all_behaviour = process_recordings(vr_path_list, all_behaviour, cohort=4)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality") if f.is_dir()]
    all_behaviour = process_recordings(vr_path_list, all_behaviour, cohort=5)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()]
    all_behaviour = process_recordings(vr_path_list, all_behaviour, cohort=6)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()]
    all_behaviour = process_recordings(vr_path_list, all_behaviour, cohort=7)
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()]
    all_behaviour = process_recordings(vr_path_list, all_behaviour, cohort=8)
        
    # save dataframes
    all_behaviour[all_behaviour["track_length"] == 200].to_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_behaviour_200cm.pkl")
    all_behaviour.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_behaviour.pkl") 
    '''

    # load dataframe
    all_behaviour_all_tracks = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_behaviour.pkl")
    _ = generate_metadata(all_behaviour_all_tracks)

    all_behaviour200cm_tracks = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_behaviour_200cm.pkl")
    meta_behaviour_data = generate_metadata(all_behaviour200cm_tracks)

    plot_histogram_all_track_lengths(all_behaviour_all_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population", first_stops=True)
    plot_histogram_all_track_lengths(all_behaviour_all_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population", first_stops=False)

    # plot performance by default, improvement (trial type ratio), improvement (trial type ratio and reward)
    plot_performance_against_improvements(meta_behaviour_data, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/meta_analysis")

    #plot_trial_ratios_against_success(meta_behaviour_data, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/meta_analysis")
    #plot_trial_ratio_vs_percentage_correct_across_time(meta_behaviour_data, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/meta_analysis")

    # plot trial_type correct ratio vs non_beaconed_expected_reward session
    plot_trial_type_correct_vs_trial_type_expected_reward(meta_behaviour_data, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/meta_analysis")

    # plot expected reward schematic
    plot_expected_reward_schematic(save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/meta_analysis", probe=True)
    plot_expected_reward_schematic(save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/meta_analysis", probe=False)

    # plot differential between fast hit trials from beaconed and non-beaconed trials in all animals where the beaconed trials are more frequent
    #plot_percentage_fast_hit_differential_for_more_frequent_beaconed_trials(meta_behaviour_data, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/meta_analysis")

    # plot to show non beaconed trials success quickly diminishes
    example_mouse = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == "M14"]
    plot_hit_percentage_beaconed_and_non_beaconed_for_example_mouse(example_mouse, meta_behaviour_data, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")

    #population_speed_per_session_by_mouse(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")

    # Example plots for spatial learning. Using M11.
    example_mouse = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == "M11"]
    example_mouse = example_mouse.sort_values(by=['session_number', 'trial_number'])
    #remove probe trials before day 20
    all_behaviour200cm_tracks = all_behaviour200cm_tracks.drop(all_behaviour200cm_tracks[(all_behaviour200cm_tracks.trial_type == 2) & (all_behaviour200cm_tracks.session_number < 20)].index)

    plot_histogram_first_and_last_5_days_stops(all_behaviour200cm_tracks,save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population", first_stops=True)
    plot_histogram_first_and_last_5_days_stops(all_behaviour200cm_tracks,save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population", first_stops=False)

    """
    # speeds 
    plot_speed_heat_map(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_speed_heat_map(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_speed_profile(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_speed_profile(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_speeds_vs_days(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_speed_diff_vs_days(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_all_speeds(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    # all stops
    shuffled_vs_training_day(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    percentage_stops_in_rz_vs_training_day(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    number_stops_in_rz_vs_training_day(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_all_stops(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_shuffled_stops(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse", y_max=10)
    plot_shuffled_stops(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse", y_max=1)
    plot_stops_on_track(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_stops_on_track(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    # first stops
    plot_all_first_stops(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    shuffled_vs_training_day_fs(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    percentage_first_stops_in_rz_vs_training_day(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    number_first_stops_in_rz_vs_training_day(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_shuffled_stops_fs(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse", y_max=1)
    plot_shuffled_stops_fs(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse", y_max=1)
    plot_stops_on_track_fs(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_stops_on_track_fs(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    # hits
    plot_percentage_hits(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_n_trials(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    """

    # population level statistics
    #print_population_stats(all_behaviour200cm_tracks)

    #plot_trial_speeds_hmt(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_percentage_trial_per_session_by_mouse(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_percentage_trial_per_session_by_mouse_short_plot(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_percentage_trial_per_session_all_mice_b_vs_nb(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_percentage_trial_per_session_all_mice_h_vs_t_vs_m(all_behaviour200cm_tracks, tt=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_percentage_trial_per_session_all_mice(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")

    #plot_hit_avg_speeds_by_block(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_average_hmt_speed_trajectories_by_trial_type(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_average_hmt_speed_trajectories_by_trial_type(all_behaviour200cm_tracks, hmt="try", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_average_hmt_speed_trajectories_by_trial_type(all_behaviour200cm_tracks, hmt="miss", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_average_hmt_speed_trajectories_by_trial_type_by_mouse(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_average_hmt_speed_trajectories_by_trial_type_by_mouse(all_behaviour200cm_tracks, hmt="try", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_average_hmt_speed_trajectories_by_trial_type_by_mouse(all_behaviour200cm_tracks, hmt="miss", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")

    #plot_trial_speeds(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_average_hit_try_run_profile(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    #plot_percentage_hits_by_mouse_individual_plots(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    plot_n_trial_per_session_by_mouse(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    plot_track_speeds_by_mouse(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    plot_percentage_hits_by_mouse(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day_numbers_stops_all_track(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day_numbers_stops(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day_percentage_stops(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day_fs(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day_numbers_first_stops(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day_percentage_first_stops(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")

    print("look now")


if __name__ == '__main__':
    main()
