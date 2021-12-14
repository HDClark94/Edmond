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



def get_reward_colors(rewarded, avg_track_speed, ejection=False):
    colors = []

    if ejection:
        for i in range(len(rewarded)):
            if (rewarded[i] == 1) and (avg_track_speed[i] > 20):
                colors.append("green")
            elif (rewarded[i] == 1) and (avg_track_speed[i] < 20):
                colors.append("gray")
            elif (rewarded[i] == 0) and (avg_track_speed[i] > 20):
                colors.append("red")
            elif (rewarded[i] == 0) and (avg_track_speed[i] < 20):
                colors.append("gray")
            else:
                colors.append("red")

    else:
        for i in range(len(rewarded)):
            if rewarded[i] == 1:
                colors.append("green")
            else:
                colors.append("red")
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
    rewarded = pandas_collumn_to_numpy_array(processed_position_data["rewarded"])
    rewarded_colors = get_reward_colors(rewarded, avg_track_speed, ejection=False)
    rewarded_colors_with_ejections = get_reward_colors(rewarded, avg_track_speed, ejection=True)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(avg_track_speed, avg_RZ_speed, color=rewarded_colors_with_ejections, edgecolor=rewarded_colors, marker="o", alpha=0.3)
    ax.plot([0,120], [0,120], linestyle="dashed", color="black")
    ax.set_ylabel("Avg Trial Speed in RZ", fontsize=20)
    ax.set_xlabel("Avg Trial Speed on track", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    ax.set_xticks([0,20,40,60,80,100,120])
    ax.set_yticks([0,20,40,60,80,100,120])
    ax.set_ylim(bottom=0, top=120)
    ax.set_xlim(left=0, right=120)
    plt.subplots_adjust(right=0.95, top=0.95)
    plt.savefig(save_path + '/RZ_speed_vs_track_speed.png', dpi=300)
    plt.close()

def plot_trial_speeds_hmt(processed_position_data, save_path):
    # remove low speed trials
    processed_position_data = processed_position_data[(processed_position_data["avg_speed_on_track"] > 20)]
    avg_RZ_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_in_RZ"])
    avg_track_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_on_track"])
    hmt = pandas_collumn_to_numpy_array(processed_position_data["hit_miss_try"])
    TI = pandas_collumn_to_numpy_array(processed_position_data["RZ_stop_bias"])
    rewarded = pandas_collumn_to_numpy_array(processed_position_data["rewarded"])
    rewarded_colors = get_reward_colors_hmt(rewarded, TI)
    rewarded_colors = get_colors_hmt(hmt)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(avg_track_speed, avg_RZ_speed, color=rewarded_colors, edgecolor=rewarded_colors, marker="o", alpha=0.3)
    ax.plot([0,120], [0,120], linestyle="dashed", color="black")
    ax.set_ylabel("Avg Trial Speed in RZ", fontsize=20)
    ax.set_xlabel("Avg Trial Speed on track", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    ax.set_xticks([0,20,40,60,80,100,120])
    ax.set_yticks([0,20,40,60,80,100,120])
    ax.set_ylim(bottom=0, top=120)
    ax.set_xlim(left=0, right=120)
    plt.subplots_adjust(right=0.95, top=0.95)
    plt.savefig(save_path + '/RZ_speed_vs_track_speed_hmt.png', dpi=300)
    plt.close()

def plot_hit_avg_speeds_by_block(processed_position_data, save_path):
    processed_position_data_block1 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 0) & (processed_position_data["avg_speed_on_track"] <= 10) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block2 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 10) & (processed_position_data["avg_speed_on_track"] <= 20) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block3 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 20) & (processed_position_data["avg_speed_on_track"] <= 30) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block4 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 30) & (processed_position_data["avg_speed_on_track"] <= 40) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block5 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 40) & (processed_position_data["avg_speed_on_track"] <= 50) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block6 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 50) & (processed_position_data["avg_speed_on_track"] <= 60) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block7 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 60) & (processed_position_data["avg_speed_on_track"] <= 70) & (processed_position_data["rewarded"] == 1)]
    processed_position_data_block8 = processed_position_data[(processed_position_data["avg_speed_on_track"] > 70) & (processed_position_data["avg_speed_on_track"] <= 80) & (processed_position_data["rewarded"] == 1)]

    avg_speeds_block1 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block1["speeds_binned_in_space"]), axis=0)
    avg_speeds_block2 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block2["speeds_binned_in_space"]), axis=0)
    avg_speeds_block3 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block3["speeds_binned_in_space"]), axis=0)
    avg_speeds_block4 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block4["speeds_binned_in_space"]), axis=0)
    avg_speeds_block5 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block5["speeds_binned_in_space"]), axis=0)
    avg_speeds_block6 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block6["speeds_binned_in_space"]), axis=0)
    avg_speeds_block7 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block7["speeds_binned_in_space"]), axis=0)
    avg_speeds_block8 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block8["speeds_binned_in_space"]), axis=0)

    locations = np.asarray(processed_position_data['position_bin_centres'].iloc[0])

    fig, ax = plt.subplots(figsize=(6,6))
    colors = cm.rainbow(np.linspace(0, 1, 8))
    ax.plot(locations[30:170], avg_speeds_block1[30:170], color=colors[0], label="0-10")
    ax.plot(locations[30:170], avg_speeds_block2[30:170], color=colors[1], label="10-20")
    ax.plot(locations[30:170], avg_speeds_block3[30:170], color=colors[2], label="20-30")
    ax.plot(locations[30:170], avg_speeds_block4[30:170], color=colors[3], label="30-40")
    ax.plot(locations[30:170], avg_speeds_block5[30:170], color=colors[4], label="40-50")
    ax.plot(locations[30:170], avg_speeds_block6[30:170], color=colors[5], label="50-60")
    ax.plot(locations[30:170], avg_speeds_block7[30:170], color=colors[6], label="60-70")
    ax.plot(locations[30:170], avg_speeds_block8[30:170], color=colors[7], label="70-80")
    ax.legend(title='Avg Track Speed (cm/s)')
    ax.set_ylabel("Speed (cm/s)", fontsize=25)
    ax.set_xlabel("Track Position", fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    style_track_plot(ax, 200)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    Edmond.plot_utility2.style_vr_plot(ax, x_max=max(avg_speeds_block8[30:170]))
    fig.tight_layout()
    plt.subplots_adjust(right=0.9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=200)
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
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=80)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.axvline(x=20, linestyle="dashed", color="black")
    fig.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    ax.set_ylabel("Number of Hit Trials", fontsize=25)
    ax.set_xlabel("Avg Track Speed (cm/s)", fontsize=25)
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
    avg_speeds_block1 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block1["speeds_binned_in_space"]), axis=0)
    avg_speeds_block2 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block2["speeds_binned_in_space"]), axis=0)
    avg_speeds_block3 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block3["speeds_binned_in_space"]), axis=0)
    avg_speeds_block4 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block4["speeds_binned_in_space"]), axis=0)
    avg_speeds_block5 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block5["speeds_binned_in_space"]), axis=0)
    avg_speeds_block6 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block6["speeds_binned_in_space"]), axis=0)
    avg_speeds_block7 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block7["speeds_binned_in_space"]), axis=0)
    avg_speeds_block8 = np.nanmean(pandas_collumn_to_2d_numpy_array(processed_position_data_block8["speeds_binned_in_space"]), axis=0)


    fig, ax = plt.subplots(figsize=(6,6))
    colors = cm.rainbow(np.linspace(0, 1, 8))
    ax.plot(locations[30:170], avg_speeds_block1[30:170], color=colors[0], label="0-10")
    ax.plot(locations[30:170], avg_speeds_block2[30:170], color=colors[1], label="10-20")
    ax.plot(locations[30:170], avg_speeds_block3[30:170], color=colors[2], label="20-30")
    ax.plot(locations[30:170], avg_speeds_block4[30:170], color=colors[3], label="30-40")
    ax.plot(locations[30:170], avg_speeds_block5[30:170], color=colors[4], label="40-50")
    ax.plot(locations[30:170], avg_speeds_block6[30:170], color=colors[5], label="50-60")
    ax.plot(locations[30:170], avg_speeds_block7[30:170], color=colors[6], label="60-70")
    ax.plot(locations[30:170], avg_speeds_block8[30:170], color=colors[7], label="70-80")
    ax.legend(title='Avg Track Speed (cm/s)')
    ax.set_ylabel("Speed (cm/s)", fontsize=25)
    ax.set_xlabel("Track Position", fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    style_track_plot(ax, 200)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    Edmond.plot_utility2.style_vr_plot(ax, x_max=max(avg_speeds_block8[30:170]))
    fig.tight_layout()
    plt.subplots_adjust(right=0.9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=200)
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
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=80)
    ax.set_xticks([0, 20, 40, 60, 80])
    fig.tight_layout()
    ax.axvline(x=20, linestyle="dashed", color="black")
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    ax.set_ylabel("Number of Miss Trials", fontsize=25)
    ax.set_xlabel("Avg Track Speed (cm/s)", fontsize=25)
    plt.savefig(save_path + '/distribution_of_misses_avg_track_speed.png', dpi=300)
    plt.close()

def add_RZ_bias(processed_position_data):
    avg_RZ_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_in_RZ"])
    avg_track_speed = pandas_collumn_to_numpy_array(processed_position_data["avg_speed_on_track"])
    RZ_stop_bias = avg_track_speed/avg_RZ_speed
    processed_position_data["RZ_stop_bias"] =RZ_stop_bias
    return processed_position_data

def plot_trial_discriminant(processed_position_data, save_path):
    hits = processed_position_data[(processed_position_data["rewarded"] == 1) & (processed_position_data["avg_speed_on_track"] > 20)]
    ejected_hits = processed_position_data[(processed_position_data["rewarded"] == 1) & (processed_position_data["avg_speed_on_track"] < 20)]
    misses = processed_position_data[(processed_position_data["rewarded"] == 0) & (processed_position_data["avg_speed_on_track"] > 20)]

    misses_plus = misses[misses["RZ_stop_bias"]<1]
    misses_minus = misses[misses["RZ_stop_bias"]>1]

    hits_discrim = pandas_collumn_to_numpy_array(hits["RZ_stop_bias"])
    ejected_hits_discrim = pandas_collumn_to_numpy_array(ejected_hits["RZ_stop_bias"])
    misses_discrim = pandas_collumn_to_numpy_array(misses["RZ_stop_bias"])
    misses_plus_discrim = pandas_collumn_to_numpy_array(misses_plus["RZ_stop_bias"])
    misses_minus_discrim = pandas_collumn_to_numpy_array(misses_minus["RZ_stop_bias"])

    hits_discrim = hits_discrim[~np.isnan(hits_discrim)]
    #ejected_hits_discrim = ejected_hits_discrim[~np.isnan(ejected_hits_discrim)]
    misses_discrim = misses_discrim[~np.isnan(misses_discrim)]
    misses_plus_discrim = misses_plus_discrim[~np.isnan(misses_plus_discrim)]
    misses_minus_discrim = misses_minus_discrim[~np.isnan(misses_minus_discrim)]

    data = [hits_discrim, misses_discrim]
    fig, ax = plt.subplots(figsize=(3,6))
    parts = ax.violinplot(data, positions=[1,2], showmeans=False, showmedians=False,showextrema=False)
    hm_colors=["green", "red"]
    for pc, hm_color in zip(parts['bodies'], hm_colors):
        pc.set_facecolor(hm_color)
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0, top=2)
    ax.set_xlim(left=0.5, right=2.5)
    quartile1, medians, quartile3 = np.percentile(data[0], [25, 50, 75], axis=0)
    ax.vlines(1, medians-0.01, medians+0.01, color='k', linestyle='-', lw=35)
    quartile1, medians, quartile3 = np.percentile(data[1], [25, 50, 75], axis=0)
    ax.vlines(2, medians-0.01, medians+0.01, color='k', linestyle='-', lw=35)
    ax.axhline(y=1, linestyle="dashed", color="black", linewidth=2)
    ax.set_xticks([1,2])
    ax.set_yticks([0,1,2])
    ax.set_xticklabels(["Hit", "Miss"])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("Trial Blocks", fontsize=25)
    ax.set_ylabel("Task Index", fontsize=25)
    plt.savefig(save_path + '/dicrimination_index_hit_and_miss.png', dpi=300)
    plt.close()

    data = [misses_plus_discrim, misses_minus_discrim]
    fig, ax = plt.subplots(figsize=(3,6))
    parts = ax.violinplot(data, positions=[1,2], showmeans=False, showmedians=False,showextrema=False)
    hm_colors=["orange", "red"]
    for pc, hm_color in zip(parts['bodies'], hm_colors):
        pc.set_facecolor(hm_color)
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(bottom=0, top=2)
    ax.set_xlim(left=0.5, right=2.5)
    ax.set_yticks([])
    quartile1, medians, quartile3 = np.percentile(data[0], [25, 50, 75], axis=0)
    ax.vlines(1, medians-0.01, medians+0.01, color='k', linestyle='-', lw=35)
    quartile1, medians, quartile3 = np.percentile(data[1], [25, 50, 75], axis=0)
    ax.vlines(2, medians-0.01, medians+0.01, color='k', linestyle='-', lw=35)
    ax.axhline(y=1, linestyle="dashed", color="black", linewidth=2)
    ax.set_xticks([1,2])
    ax.set_xticklabels(["Miss+", "Miss-"])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("Trial Blocks", fontsize=25)
    plt.savefig(save_path + '/dicrimination_index_miss_and_miss.png', dpi=300)
    plt.close()

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


def plot_average_hmt_speed_trajectories(processed_position_data, hmt, save_path):
    hmt_processed = processed_position_data[processed_position_data["hit_miss_try"] == hmt]

    trajectories = pandas_collumn_to_2d_numpy_array(hmt_processed["speeds_binned_in_space"])
    trajectories_avg = np.nanmean(trajectories, axis=0)[30:170]
    trajectories_sem = np.nanstd(trajectories, axis=0)[30:170]
    #trajectories_sem = stats.sem(trajectories, axis=0, nan_policy="omit")[30:170]
    locations = np.asarray(processed_position_data['position_bin_centres'].iloc[0])[30:170]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.fill_between(locations, trajectories_avg-trajectories_sem, trajectories_avg+trajectories_sem, color=get_hmt_color(hmt), alpha=0.3)
    ax.plot(locations, trajectories_avg, color="black")
    ax.set_ylabel("Speed (cm/s)", fontsize=25)
    ax.set_xlabel("Track Position", fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    style_track_plot(ax, 200)
    tick_spacing = 100
    ax.set_yticks([0,40, 80])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    Edmond.plot_utility2.style_vr_plot(ax, x_max=80)
    fig.tight_layout()
    plt.subplots_adjust(right=0.9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=200)
    plt.savefig(save_path + '/average_speed_trajectory_'+hmt+'.png', dpi=300)
    plt.close()

def plot_average_hmt_speed_trajectories_by_trial_type(processed_position_data, hmt, save_path):
    hmt_processed = processed_position_data[processed_position_data["hit_miss_try"] == hmt]

    for tt, tt_string in zip([0,1,2], ["b", "nb", "p"]):
        tt_processed = hmt_processed[hmt_processed["trial_type"] == tt]
        trajectories = pandas_collumn_to_2d_numpy_array(tt_processed["speeds_binned_in_space"])
        trajectories_avg = np.nanmean(trajectories, axis=0)[30:170]
        trajectories_sem = np.nanstd(trajectories, axis=0)[30:170]
        #trajectories_sem = stats.sem(trajectories, axis=0, nan_policy="omit")[30:170]
        locations = np.asarray(processed_position_data['position_bin_centres'].iloc[0])[30:170]

        fig, ax = plt.subplots(figsize=(6,6))
        ax.fill_between(locations, trajectories_avg-trajectories_sem, trajectories_avg+trajectories_sem, color=get_hmt_color(hmt), alpha=0.3)
        ax.plot(locations, trajectories_avg, color="black")
        ax.set_ylabel("Speed (cm/s)", fontsize=25)
        ax.set_xlabel("Track Position", fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=20)
        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        tick_spacing = 100
        ax.set_yticks([0,40, 80])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        Edmond.plot_utility2.style_vr_plot(ax, x_max=80)
        fig.tight_layout()
        plt.subplots_adjust(right=0.9)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=200)
        plt.savefig(save_path + '/average_speed_trajectory_'+hmt+"_tt_"+tt_string+'.png', dpi=300)
        plt.close()

def process_recordings(vr_recording_path_list, of_recording_path_list):
    print(" ")
    all_behaviour = pd.DataFrame()
    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            position_data = add_time_elapsed_collumn(position_data)
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            processed_position_data, _ = add_hit_miss_try(processed_position_data, track_length=get_track_length(recording))
            processed_position_data = add_avg_trial_speed(processed_position_data)
            processed_position_data = add_avg_RZ_speed(processed_position_data, track_length=get_track_length(recording))
            processed_position_data = add_avg_track_speed(processed_position_data, track_length=get_track_length(recording))
            PI_hits_processed_position_data = extract_PI_trials(processed_position_data, hmt="hit")
            PI_misses_processed_position_data = extract_PI_trials(processed_position_data, hmt="miss")
            PI_tries_position_data = extract_PI_trials(processed_position_data, hmt="try")
            #raw_position_data, position_data = syncronise_position_data(recording, get_track_length(recording))

            #spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            if (get_track_length(recording)) == 200:
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

    # give a path for a directory of recordings or path of a single recording
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()]
    #vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36']
    #all_behaviour200cm_tracks = process_recordings(vr_path_list, of_path_list)
    #all_behaviour200cm_tracks.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_behaviour_cohort8_200cm.pkl")
    all_behaviour200cm_tracks = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_behaviour_cohort8_200cm.pkl")
    all_behaviour200cm_tracks = add_RZ_bias(all_behaviour200cm_tracks)
    plot_trial_discriminant(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_trial_discriminant_schematic(save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #plot_trial_discriminant_histogram(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_trial_speeds(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_trial_speeds_hmt(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_hit_avg_speeds_by_block(all_behaviour200cm_tracks, save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    #all_behaviour200cm_tracks = add_hit_miss_try2(all_behaviour200cm_tracks, track_length=200)

    plot_average_hmt_speed_trajectories(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories(all_behaviour200cm_tracks, hmt="try", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories(all_behaviour200cm_tracks, hmt="miss", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories_by_trial_type(all_behaviour200cm_tracks, hmt="hit", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories_by_trial_type(all_behaviour200cm_tracks, hmt="try", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    plot_average_hmt_speed_trajectories_by_trial_type(all_behaviour200cm_tracks, hmt="miss", save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")

    compute_p_map(save_path="/mnt/datastore/Harry/Vr_grid_cells/behaviour")
    print("look now")


if __name__ == '__main__':
    main()
