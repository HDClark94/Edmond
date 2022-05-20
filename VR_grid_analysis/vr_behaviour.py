import numpy as np
import pandas as pd
import PostSorting.parameters
import PostSorting.vr_stop_analysis
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



def get_reward_colors(hmt, avg_track_speed):
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
    avg_RZ_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_in_RZ"])
    avg_track_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_on_track"])
    hmt = pandas_collumn_to_numpy_array(processed_position_data["hit_miss_try"])
    rewarded_colors = get_reward_colors(hmt, avg_track_speed)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(avg_track_speed, avg_RZ_speed, color=rewarded_colors, marker="x", alpha=0.3)
    ax.plot([0,100], [0,100], linestyle="dashed", color="black")
    ax.set_ylabel("Avg Trial Speed in RZ", fontsize=25, labelpad=10)
    ax.set_xlabel("Avg Trial Speed on track", fontsize=25, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    ax.set_xticks([0,20,40,60,80,100])
    ax.set_yticks([0,20,40,60,80,100])
    ax.set_ylim(bottom=0, top=100)
    ax.set_xlim(left=0, right=100)
    plt.subplots_adjust(right=0.95, top=0.95)
    plt.savefig(save_path + '/RZ_speed_vs_track_speed.png', dpi=300)
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
        trajectories_sem = np.nanstd(trajectories, axis=0)[start:end]
        #trajectories_sem = stats.sem(trajectories, axis=0, nan_policy="omit")[start:end]
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

def plot_n_trial_per_session_by_mouse(processed_position_data, hmt, save_path):
    hmt_processed = processed_position_data[processed_position_data["hit_miss_try"] == hmt]

    for tt, tt_string in zip([0,1,2], ["b", "nb", "p"]):
        t_processed = hmt_processed[hmt_processed["trial_type"] == tt]

        max_n = 0
        speed_histogram = plt.figure(figsize=(6,6))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for _, mouse_id in enumerate(np.unique(t_processed["mouse_id"])):
            mouse_processed = t_processed[t_processed["mouse_id"] == mouse_id]

            n_trials = []
            for session_number in np.unique(mouse_processed["session_number"]):
                session_processed = mouse_processed[mouse_processed["session_number"] == session_number]
                n_trials.append(len(session_processed))

            ax.scatter(np.unique(mouse_processed["session_number"]), n_trials, marker="o", label=mouse_id, zorder=10, clip_on=False)

            if max(n_trials)>max_n:
                max_n = max(n_trials)

        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.legend(title='Mouse')
        plt.ylabel('Number of trials', fontsize=25, labelpad = 10)
        plt.xlabel('Session number', fontsize=25, labelpad = 10)

        tick_spacing = 50
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([1, 25, 50])
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        Edmond.plot_utility2.style_vr_plot(ax, x_max=max_n)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=50)
        plt.savefig(save_path + '/n_trials_by_mouse_hmt_'+hmt+"_tt_"+tt_string+'.png', dpi=300)
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
    stop_trials = get_stop_trial_ids(session_df)
    stop_locations = Edmond.plot_utility2.pandas_collumn_to_numpy_array(session_df["stop_location_cm"])

    # stops are calculated as being below the stop threshold per unit time bin,
    # this function removes successive stops

    stop_locations_elapsed=(track_length*(stop_trials-1))+stop_locations

    curated_stop_locations=[]
    curated_stop_trial_numbers=[]
    for i, stop_loc in enumerate(stop_locations_elapsed):
        if (i==0): # take first stop always
            add_stop=True
        elif ((stop_locations_elapsed[i]-stop_locations_elapsed[i-1]) > 1): # only include stop if the last stop was at least 1cm away
            add_stop=True
        else:
            add_stop=False

        if add_stop:
            curated_stop_locations.append(stop_locations_elapsed[i])
            curated_stop_trial_numbers.append(stop_trials[i])
    curated_stop_locations = np.array(curated_stop_locations)
    curated_stop_trial_numbers = np.array(curated_stop_trial_numbers)

    # add back curated stops
    stop_locations = []
    for tn in session_df.trial_number:
        stop_locations.append(curated_stop_locations[curated_stop_trial_numbers == tn]%track_length)
    session_df["stop_location_cm"] = stop_locations

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
    mouse_df = mouse_df.sort_values(by=['session_number'])
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
    trial_number=1
    for session_number in np.unique(mouse_df["session_number"]):
        session_df = mouse_df[mouse_df["session_number"] == session_number]
        session_df = session_df.sort_values(by=['trial_number'])
        for index, trial_row in session_df.iterrows():
            trial_row = trial_row.to_frame().T.reset_index(drop=True)
            trial_type = trial_row["trial_type"].iloc[0]
            trial_stop_color = get_trial_color(trial_type)
            ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]), trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])), 'o', color=trial_stop_color, markersize=2, alpha=0.2)
            trial_number+=1

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


def plot_all_first_stops(mouse_df, save_path, track_length=200):
    stops_on_track = plt.figure(figsize=(6,30))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    mouse_df = curate_stops(mouse_df, track_length)
    mouse_df = mouse_df[mouse_df["session_number"]<=30]
    trial_number=1
    for session_number in np.unique(mouse_df["session_number"]):
        session_df = mouse_df[mouse_df["session_number"] == session_number]
        session_df = session_df.sort_values(by=['trial_number'])
        for index, trial_row in session_df.iterrows():
            trial_row = trial_row.to_frame().T.reset_index(drop=True)
            trial_type = trial_row["trial_type"].iloc[0]
            trial_stop_color = get_trial_color(trial_type)
            ax.plot(trial_row["first_stop_location_cm"].iloc[0], trial_number, 'o', color=trial_stop_color, markersize=2, alpha=1)
            trial_number+=1

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

        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], color=tt_color, alpha=0.3)
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
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], color=tt_color, alpha=0.3)
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
    plt.savefig(save_path + '/percentage_first_stops_vs_training_day_fs_all_tt.png', dpi=200)
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
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], color=tt_color, alpha=0.3)
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
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], color=tt_color, alpha=0.3)
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

        stop_histogram = plt.figure(figsize=(6,4))
        ax = stop_histogram.add_subplot(1, 1, 1)
        nan_mask = ~np.isnan(np.nanmean(mouse_array[tt], axis=0))
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], color=tt_color, alpha=0.3)
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
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], color=tt_color, alpha=0.3)
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
    plt.savefig(save_path + '/percentage_stops_vs_training_day_all_tt.png', dpi=200)
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
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], color=tt_color, alpha=0.3)
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
        ax.fill_between(np.arange(1,max_session+1)[nan_mask], (np.nanmean(tt_mouse_array, axis=0)-stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], (np.nanmean(tt_mouse_array, axis=0)+stats.sem(tt_mouse_array, axis=0, nan_policy="omit"))[nan_mask], color=tt_color, alpha=0.3)
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
    plt.savefig(save_path + '/shuffle_vs_training_day_all_tt.png', dpi=200)
    plt.close()

def print_population_stats(all_behaviour200cm_tracks):
    print("hello there")


def process_recordings(vr_recording_path_list):
    print(" ")
    all_behaviour = pd.DataFrame()
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

            all_behaviour = pd.concat([all_behaviour, processed_position_data], ignore_index=True)
            print("successfully processed and saved vr_grid analysis on "+recording)
        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)

    return all_behaviour


def main():
    print('-------------------------------------------------------------')

    '''
    # give a path for a directory of recordings or path of a single recording
    vr_path_list = []
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort9_Junji/vr") if f.is_dir()])
    all_behaviour = process_recordings(vr_path_list)

    # save dataframes
    all_behaviour[all_behaviour["track_length"] == 200].to_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_behaviour_200cm.pkl")
    all_behaviour.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_behaviour.pkl")
    '''

    # load dataframe
    all_behaviour200cm_tracks = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_behaviour_200cm.pkl")


    # Example plots for spatial learning. Using M11.
    example_mouse = all_behaviour200cm_tracks[all_behaviour200cm_tracks["mouse_id"] == "M11"]
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
    plot_all_stops(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_shuffled_stops(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse", y_max=10)
    plot_shuffled_stops(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse", y_max=1)
    plot_stops_on_track(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_stops_on_track(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    # first stops
    plot_all_first_stops(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    shuffled_vs_training_day_fs(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    percentage_first_stops_in_rz_vs_training_day(example_mouse, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_shuffled_stops_fs(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse", y_max=1)
    plot_shuffled_stops_fs(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse", y_max=1)
    plot_stops_on_track_fs(example_mouse, session_number=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    plot_stops_on_track_fs(example_mouse, session_number=24, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/example_mouse")
    # hits



    # population level statistics
    population_shuffled_vs_training_day(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day_percentage_stops(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day_fs(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")
    population_shuffled_vs_training_day_percentage_first_stops(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/population")

    print_population_stats(all_behaviour200cm_tracks)
    plot_trial_speeds(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_trial_speeds_hmt(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_n_trial_per_session_by_mouse(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_percentage_trial_per_session_by_mouse(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_percentage_trial_per_session_by_mouse_short_plot(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_percentage_trial_per_session_all_mice_b_vs_nb(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_percentage_trial_per_session_all_mice_h_vs_t_vs_m(all_behaviour200cm_tracks, tt=1, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_percentage_trial_per_session_all_mice(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")


    plot_hit_avg_speeds_by_block(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories_by_trial_type(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories_by_trial_type(all_behaviour200cm_tracks, hmt="try", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories_by_trial_type(all_behaviour200cm_tracks, hmt="miss", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories_by_trial_type_by_mouse(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories_by_trial_type_by_mouse(all_behaviour200cm_tracks, hmt="try", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories_by_trial_type_by_mouse(all_behaviour200cm_tracks, hmt="miss", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")

    #compute_p_map(save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #cluster_speed_profiles(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour/cluster_analysis")
    print("look now")


if __name__ == '__main__':
    main()
