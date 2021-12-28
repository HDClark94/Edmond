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
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from Edmond.VR_grid_analysis.hit_miss_try_firing_analysis import hmt2collumn
import matplotlib.patches as patches
import matplotlib.colors as colors
from sklearn.linear_model import LinearRegression
from PostSorting.vr_spatial_firing import bin_fr_in_space, add_position_x
from scipy import signal
from astropy.convolution import convolve, Gaussian1DKernel
import os
import traceback
import warnings
import matplotlib.ticker as ticker
import sys
import scipy
import Edmond.plot_utility2
import Edmond.VR_grid_analysis.hit_miss_try_firing_analysis
from Edmond.VR_grid_analysis.vr_grid_cells import *
import settings
import matplotlib.pylab as plt
import matplotlib as mpl
import control_sorting_analysis
import PostSorting.post_process_sorted_data_vr
from Edmond.utility_functions.array_manipulations import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
from scipy.stats.stats import pearsonr
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from Edmond.VR_grid_analysis.hit_miss_try_firing_analysis import significance_bar, get_p_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline

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

def plot_spatial_info_vs_pearson(combined_df, output_path):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    stable_grid_cells = grid_cells[grid_cells["avg_pairwise_trial_pearson_r_stable"] == True]
    non_stable_grid_cells = grid_cells[grid_cells["avg_pairwise_trial_pearson_r_stable"] == False]
    stable_non_grid_cells = non_grid_cells[non_grid_cells["avg_pairwise_trial_pearson_r_stable"] == True]
    non_stable_non_grid_cells = non_grid_cells[non_grid_cells["avg_pairwise_trial_pearson_r_stable"] == False]

    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(np.asarray(stable_grid_cells["avg_pairwise_trial_pearson_r"]), np.asarray(stable_grid_cells["hmt_all_tt_all"]), edgecolor="red", marker="o", facecolors='none', alpha=0.3)
    ax.scatter(np.asarray(non_stable_grid_cells["avg_pairwise_trial_pearson_r"]), np.asarray(non_stable_grid_cells["hmt_all_tt_all"]), edgecolor="red", marker="x", facecolors='none', alpha=0.3)
    ax.scatter(np.asarray(stable_non_grid_cells["avg_pairwise_trial_pearson_r"]), np.asarray(stable_non_grid_cells["hmt_all_tt_all"]), edgecolor="black", marker="o", facecolors='none', alpha=0.3)
    ax.scatter(np.asarray(non_stable_non_grid_cells["avg_pairwise_trial_pearson_r"]), np.asarray(non_stable_non_grid_cells["hmt_all_tt_all"]), edgecolor="black", marker="x", facecolors='none', alpha=0.3)
    ax.set_xlabel("Avg Trial-pair Pearson R", fontsize=20, labelpad=10)
    ax.set_ylabel("Spatial Information", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path+"spatial_info_vs_pairwise_pearson.png", dpi=200)
    plt.close()

def hmt2numeric(hmt):
    # takes numpy array of "hit", "miss" and "try" and translates them into 1, 0 and 0.5 otherwise nan
    hmt_numeric = []
    for i in range(len(hmt)):
        if hmt[i] == "hit":
            numeric = 1
        elif hmt[i] == "try":
            numeric = 0.5
        elif hmt[i] == "miss":
            numeric = 0
        else:
            numeric = np.nan
        hmt_numeric.append(numeric)
    return np.array(hmt_numeric)

def hmt2color(hmt):
    # takes numpy array of "hit", "miss" and "try" and translates them into 1, 0 and 0.5 otherwise nan
    hmt_colors = []
    for i in range(len(hmt)):
        if hmt[i] == "hit":
            color = "green"
        elif hmt[i] == "try":
            color = "orange"
        elif hmt[i] == "miss":
            color = "red"
        else:
            color = np.nan
        hmt_colors.append(color)
    return np.array(hmt_colors)

def get_trial_type_colors(trial_types):
    # takes numpy array of 0, 1 and 2 and translates them into black, red and blue
    type_colors = []
    for i in range(len(trial_types)):
        if trial_types[i] == 0: # beaconed
            color_i = "black"
        elif trial_types[i] == 1: # non-beaconed
            color_i = "red"
        elif trial_types[i] == 2: # probe
            color_i = "blue"
        else:
            print("do nothing")
        type_colors.append(color_i)

    return np.array(type_colors)


def plot_hmt_against_pairwise(spike_data, processed_position_data, output_path, track_length, suffix=""):
    print('plotting hmt against pairwise correlations...')
    save_path = output_path + '/Figures/hmt_pairwise'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    save_path2 = output_path + '/Figures/avg_speed_against_pairwise'
    if os.path.exists(save_path2) is False:
        os.makedirs(save_path2)

    hmt = np.array(processed_position_data["hit_miss_try"])
    trial_types = np.array(processed_position_data["trial_type"])
    avg_speeds = np.array(processed_position_data["avg_trial_speed"])
    hmt_numeric = hmt2numeric(hmt)
    hit_mask = hmt_numeric==1
    try_mask = hmt_numeric==0.5
    miss_mask = hmt_numeric==0
    hmt_colors = hmt2color(hmt)
    trial_type_colors = get_trial_type_colors(trial_types)
    trial_numbers = np.array(processed_position_data["trial_number"])
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        pairwise_trial_pearson_r = np.array(cluster_spike_data['pairwise_trial_pearson_r'].iloc[0])

        fig, axs = plt.subplots(2,1,figsize=(4*len(processed_position_data)/75,4), gridspec_kw={'height_ratios': [1, 9]})
        i=0
        for x1, x2 in zip(trial_numbers-1, trial_numbers):
            axs[1].fill_between([x1, x2, x2, x1],[-1,-1, 1, 1], color = hmt_colors[i], alpha=0.4)
            axs[0].fill_between([x1, x2, x2, x1],[0, 0, 1, 1], color = trial_type_colors[i], alpha=1)
            i+=1
        axs[1].plot(trial_numbers[1:], pairwise_trial_pearson_r, color="black")
        axs[1].set_ylabel("Tn-1/Tn Correlation",color="black",fontsize=15)
        axs[1].set_xlabel('Trial Number', fontsize=15, labelpad = 10)
        axs[1].set_xlim(0,len(processed_position_data))
        axs[0].set_xlim(0,len(processed_position_data))
        axs[1].set_ylim(-1,1)
        axs[0].set_ylim(0,1)
        axs[0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
        axs[1].spines['top'].set_visible(False)
        plt.tick_params(top=False)
        axs[1].yaxis.set_ticks_position('left')
        axs[1].xaxis.set_ticks_position('bottom')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_hmt_against_pairwise_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
        plt.close()


        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_ylabel("Tn-1/Tn Correlation",color="black",fontsize=15, labelpad=10)
        ax.set_xlabel("Avg Trial Speed", color="black", fontsize=15, labelpad=10)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.scatter(x=avg_speeds[1:], y=pairwise_trial_pearson_r, color=hmt_colors[1:], marker="o", alpha=0.3)
        ax.errorbar(x=np.nanmean(avg_speeds[1:][hit_mask[1:]]), y=np.nanmean(pairwise_trial_pearson_r[hit_mask[1:]]), xerr=stats.sem(avg_speeds[1:][hit_mask[1:]], nan_policy='omit') , yerr=stats.sem(pairwise_trial_pearson_r[hit_mask[1:]], nan_policy='omit'), color="green", capsize=2)
        ax.errorbar(x=np.nanmean(avg_speeds[1:][try_mask[1:]]), y=np.nanmean(pairwise_trial_pearson_r[try_mask[1:]]), xerr=stats.sem(avg_speeds[1:][try_mask[1:]], nan_policy='omit') , yerr=stats.sem(pairwise_trial_pearson_r[try_mask[1:]], nan_policy='omit'), color="orange", capsize=2)
        ax.errorbar(x=np.nanmean(avg_speeds[1:][miss_mask[1:]]), y=np.nanmean(pairwise_trial_pearson_r[miss_mask[1:]]), xerr=stats.sem(avg_speeds[1:][miss_mask[1:]], nan_policy='omit') , yerr=stats.sem(pairwise_trial_pearson_r[miss_mask[1:]], nan_policy='omit'), color="red", capsize=2)
        ax.set_xlim([0,max(avg_speeds)])
        ax.set_ylim([-1,1])
        plt.tight_layout()
        plt.savefig(save_path2 + '/' + spike_data.session_id.iloc[cluster_index] + '_avg_speed_against_pairwise_Cluster_' + str(cluster_id) + suffix + '.png', dpi=200)
        plt.close()

def get_hmt_color(hmt):
    if hmt == "hit":
        return "green"
    elif hmt == "miss":
        return "red"
    elif hmt == "try":
        return "orange"
    else:
        return "black"

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

        ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]), trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])), 'o', color="black", markersize=4)

    plt.ylabel('Stops on trials', fontsize=20, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
    plt.xlim(0,track_length)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
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
    plt.savefig(output_path + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()

def plot_avg_speed_in_rz_hist(processed_position_data, output_path, percentile_speed):
    print('I am plotting avg speed histogram...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    hits = processed_position_data[processed_position_data["hit_miss_try"] == "hit"]
    misses = processed_position_data[processed_position_data["hit_miss_try"] == "miss"]
    tries = processed_position_data[processed_position_data["hit_miss_try"] == "try"]

    ax.hist(pandas_collumn_to_numpy_array(hits["avg_speed_in_rz"]), range=(0, 100), bins=25, alpha=0.3, color="green", histtype="bar", density=False, cumulative=False, linewidth=4)
    ax.hist(pandas_collumn_to_numpy_array(tries["avg_speed_in_rz"]), range=(0, 100), bins=25, alpha=0.3, color="orange", histtype="bar", density=False, cumulative=False, linewidth=4)
    ax.hist(pandas_collumn_to_numpy_array(misses["avg_speed_in_rz"]), range=(0, 100), bins=25, alpha=0.3, color="red", histtype="bar", density=False, cumulative=False, linewidth=4)

    plt.ylabel('Counts', fontsize=20, labelpad = 10)
    plt.xlabel('Avg Speed in RZ (cm/s)', fontsize=20, labelpad = 10)
    plt.xlim(0,100)
    tick_spacing = 50
    ax.axvline(x=percentile_speed, color="black", linestyle="dotted", linewidth=4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    n_trials = len(processed_position_data)
    ax.set_yticks([0, 10, 20, 30])
    Edmond.plot_utility2.style_vr_plot(ax)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/avg_speed_hist' + '.png', dpi=200)
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

def plot_speed_histogram_with_error(processed_position_data, output_path, track_length=200, suffix=""):
    if len(processed_position_data)>0:
        trial_speeds = pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned_in_space"])

        trial_speeds_sem = scipy.stats.sem(trial_speeds, axis=0, nan_policy="omit")
        trial_speeds_avg = np.nanmean(trial_speeds, axis=0)

        print('plotting avg speeds')
        save_path = output_path + '/Figures/behaviour'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        speed_histogram = plt.figure(figsize=(6,4))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])

        start_idx = 7
        end_idx = 193

        #ax.fill_between(bin_centres, trial_speeds_avg+trial_speeds_sem,  trial_speeds_avg-trial_speeds_sem, color=get_hmt_color(suffix), alpha=0.3)
        for i in range(len(trial_speeds)):
            #ax.plot(bin_centres[start_idx : end_idx], trial_speeds[i][start_idx : end_idx], color=get_hmt_color(suffix), alpha=0.3)
            ax.plot(bin_centres[start_idx : end_idx], trial_speeds[i][start_idx : end_idx], color="grey", alpha=0.4)

        ax.plot(bin_centres[start_idx : end_idx], trial_speeds_avg[start_idx : end_idx], color=get_hmt_color(suffix), linewidth=4)
        ax.axhline(y=4.7, color="black", linestyle="dashed", linewidth=2)
        plt.ylabel('Speed (cm/s)', fontsize=20, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
        plt.xlim(0,track_length)
        ax.set_yticks([0, 50, 100])
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Edmond.plot_utility2.style_track_plot(ax, track_length)
        tick_spacing = 100
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        x_max = max(trial_speeds_avg+trial_speeds_sem)
        x_max = 115
        Edmond.plot_utility2.style_vr_plot(ax, x_max)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.savefig(output_path + '/Figures/behaviour/speed_histogram_sem_' +suffix+ '.png', dpi=200)
        plt.close()


def plot_max_freq_histogram(combined_df, combined_shuffle_df, save_path):
    max_powers = pandas_collumn_to_numpy_array(combined_df["MOVING_LOMB_SNR"])
    max_freqs = pandas_collumn_to_numpy_array(combined_df["MOVING_LOMB_freqs"])

    shuffle_max_powers = pandas_collumn_to_2d_numpy_array(combined_shuffle_df["max_shuffle_power"])
    shuffle_max_freqs = pandas_collumn_to_numpy_array(combined_shuffle_df["max_shuffle_freq"])

    threshold = np.nanpercentile(shuffle_max_powers, 95)

    fig, ax = plt.subplots(figsize=(3,6))
    _, _, patches2 = ax.hist(shuffle_max_powers, bins=100, color="gray", density=False, linewidth=2)
    _, _, patches1 = ax.hist(max_powers, bins=100, color="blue", density=False, linewidth=2, alpha=0.5)
    ax.axvline(x=threshold, linestyle="solid", color="red")
    plt.ylabel("Counts",  fontsize=20)
    plt.xlabel("Max Power",  fontsize=15)
    ax.set_xlim([0,0.2])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_max_powers.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(3,6))
    _, _, patches2 = ax.hist(shuffle_max_freqs, bins=100, color="gray", density=False, linewidth=2)
    _, _, patches1 = ax.hist(max_freqs, bins=100, color="blue", density=False, linewidth=2, alpha=0.5)
    plt.ylabel("Counts",  fontsize=20)
    plt.xlabel("Max Frequency",  fontsize=15)
    ax.set_xlim([0,10])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_max_frequencies.png', dpi=300)
    plt.close()
    return


def plot_lomb_overview_ordered(concantenated_dataframe, save_path, cmap_string="viridis"):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe)
    print('plotting lomb classifers...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]
    g_position_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    g_null_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Null"]

    ng_distance_cells = non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Distance"]
    ng_position_cells = non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Position"]
    ng_null_cells = non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Null"]

    g_p_powers = pandas_collumn_to_2d_numpy_array(g_position_cells["MOVING_LOMB_avg_power"])
    g_p_max_freqs = pandas_collumn_to_numpy_array(g_position_cells["MOVING_LOMB_freqs"])
    g_d_powers = pandas_collumn_to_2d_numpy_array(g_distance_cells["MOVING_LOMB_avg_power"])
    g_d_max_freqs = pandas_collumn_to_numpy_array(g_distance_cells["MOVING_LOMB_freqs"])
    g_n_powers = pandas_collumn_to_2d_numpy_array(g_null_cells["MOVING_LOMB_avg_power"])
    g_n_max_freqs = pandas_collumn_to_numpy_array(g_null_cells["MOVING_LOMB_freqs"])

    ng_p_powers = pandas_collumn_to_2d_numpy_array(ng_position_cells["MOVING_LOMB_avg_power"])
    ng_p_max_freqs = pandas_collumn_to_numpy_array(ng_position_cells["MOVING_LOMB_freqs"])
    ng_d_powers = pandas_collumn_to_2d_numpy_array(ng_distance_cells["MOVING_LOMB_avg_power"])
    ng_d_max_freqs = pandas_collumn_to_numpy_array(ng_distance_cells["MOVING_LOMB_freqs"])
    ng_n_powers = pandas_collumn_to_2d_numpy_array(ng_null_cells["MOVING_LOMB_avg_power"])
    ng_n_max_freqs = pandas_collumn_to_numpy_array(ng_null_cells["MOVING_LOMB_freqs"])


    # this step corrects the array lengths that correspond to cells that didn't fire at all
    step = 0.02
    frequency = np.arange(0.1, 5+step, step)
    for i in range(len(ng_n_powers)):
        if isinstance(ng_n_powers[i], list):
            ng_n_powers[i] = np.ones(len(frequency))*np.nan
    ng_n_powers = np.stack(ng_n_powers)
    for i in range(len(ng_n_max_freqs)):
        if isinstance(ng_n_max_freqs[i], list):
            ng_n_max_freqs[i] = np.nan

    g_p_powers = g_p_powers[np.argsort(g_p_max_freqs)]
    g_d_powers = g_d_powers[np.argsort(g_d_max_freqs)]
    g_n_powers = g_n_powers[np.argsort(g_n_max_freqs)]

    ng_p_powers = ng_p_powers[np.argsort(ng_p_max_freqs)]
    ng_d_powers = ng_d_powers[np.argsort(ng_d_max_freqs)]
    ng_n_powers = ng_n_powers[np.argsort(ng_n_max_freqs)]

    fig, ax = plt.subplots(figsize=(6,8))
    groups = ["Position", "Distance", "Null"]
    colors_lm = ["turquoise", "orange", "gray"]

    # x is the spatial frequency
    color_legend_offset = 1
    frequency = np.arange(0.1, 5+step, step)+color_legend_offset
    ordered = np.arange(0, len(grid_cells)+1, 1)
    X, Y = np.meshgrid(frequency, ordered)
    powers = np.concatenate([g_p_powers, g_d_powers, g_n_powers], axis=0)
    powers = np.flip(powers, axis=0)

    for i in range(len(powers)):
        #powers[i, :] = min_max_normalize(powers[i, :])
        powers[i, :] = scipy.stats.zscore(powers[i, :], ddof=0, nan_policy='omit')
    pcm = ax.pcolormesh(X, Y, powers, vmin=np.min(powers), vmax=np.max(powers), cmap=cmap_string)
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Z-Scored Power', rotation=270)
    group_legend_array = np.arange(0, 0.5+step, step)
    color_legend_tmp = np.concatenate([np.ones((len(g_position_cells), len(group_legend_array)))*0,
                                       np.ones((len(g_distance_cells), len(group_legend_array)))*1.5,
                                       np.ones((len(g_null_cells), len(group_legend_array)))*2.5])
    color_legend_tmp = np.flip(color_legend_tmp, axis=0)

    X, Y = np.meshgrid(group_legend_array, ordered)
    cmap = colors.ListedColormap(['turquoise', 'orange','gray'])
    boundaries = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.pcolormesh(X, Y, color_legend_tmp, norm=norm, cmap=cmap)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels(["0","", " ", ""," ","5", " ", ""," ", "", "10"])
    ax.set_xticks(np.arange(0,11)+color_legend_offset)
    ax.set_xlabel("Frequency", fontsize=10)
    #plt.tight_layout()
    plt.xticks(fontsize=4)
    plt.subplots_adjust(left=0.4)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(save_path + '/grid_lomb_power_ordered.png', dpi=200)
    plt.close()


    fig, ax = plt.subplots(figsize=(6,8))
    groups = ["Position", "Distance", "Null"]
    colors_lm = ["turquoise", "orange", "gray"]

    # x is the spatial frequency
    color_legend_offset = 1
    step = 0.02
    frequency = np.arange(0.1, 5+step, step)+color_legend_offset
    ordered = np.arange(0, len(non_grid_cells)+1, 1)
    X, Y = np.meshgrid(frequency, ordered)
    powers = np.concatenate([ng_p_powers, ng_d_powers, ng_n_powers], axis=0)
    powers = np.flip(powers, axis=0)
    powers[np.isnan(powers)] = 0
    for i in range(len(powers)):
        #powers[i, :] = min_max_normalize(powers[i, :])
        powers[i, :] = scipy.stats.zscore(powers[i, :], ddof=0, nan_policy='omit')
    powers[np.isnan(powers)] = 0

    pcm = ax.pcolormesh(X, Y, powers, vmin=np.min(powers), vmax=np.max(powers), cmap=cmap_string)
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Z-Scored Power', rotation=270)

    group_legend_array = np.arange(0, 0.5+step, step)
    color_legend_tmp = np.concatenate([np.ones((len(ng_position_cells), len(group_legend_array)))*0,
                                       np.ones((len(ng_distance_cells), len(group_legend_array)))*1.5,
                                       np.ones((len(ng_null_cells), len(group_legend_array)))*2.5])
    color_legend_tmp = np.flip(color_legend_tmp, axis=0)

    X, Y = np.meshgrid(group_legend_array, ordered)
    cmap = colors.ListedColormap(['turquoise', 'orange','gray'])
    boundaries = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.pcolormesh(X, Y, color_legend_tmp, norm=norm, cmap=cmap)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels(["0","", " ", ""," ","5", " ", ""," ", "", "10"])
    ax.set_xticks(np.arange(0,11)+color_legend_offset)
    ax.set_xlabel("Frequency", fontsize=10)
    plt.xticks(fontsize=4)
    #plt.tight_layout()
    plt.subplots_adjust(left=0.4)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(save_path + '/nongrid_lomb_power_ordered.png', dpi=200)
    plt.close()

    return


def plot_lomb_classifiers(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]
    distance_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Distance"]
    position_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Position"]
    null_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Null"]

    avg_SNR_ratio_threshold = np.nanmean(concantenated_dataframe["shuffleSNR"+suffix])
    avg_distance_from_integer_threshold = np.nanmean(concantenated_dataframe["shufflefreqs"+suffix])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    ax1.set_ylabel("Peak Power",color="black",fontsize=15, labelpad=10)
    ax1.set_xlabel("Spatial Frequency", color="black", fontsize=15, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax2.set_xticks([0,0.25, 0.5])
    plt.setp(ax1.get_xticklabels(), fontsize=15)
    plt.setp(ax2.get_xticklabels(), fontsize=10)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.xaxis.grid() # vertical lines
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax1.scatter(x=non_grid_cells["ML_Freqs"+suffix], y=non_grid_cells["ML_SNRs"+suffix], color="black", marker="o", alpha=0.3)
    ax1.scatter(x=grid_cells["ML_Freqs"+suffix], y=grid_cells["ML_SNRs"+suffix], color="r", marker="o", alpha=0.3)
    #ax1.axhline(y=avg_SNR_ratio_threshold, xmin=0, xmax=10, color="black", linestyle="dashed")
    ax1.set_xlim([0,10])
    ax1.set_ylim([0,0.4])
    ax2.set_xlim([-0.1,0.6])
    ax2.set_ylim([0,0.4])
    ax2.set_xlabel(r'$\Delta$ from Integer', color="black", fontsize=15, labelpad=10)
    ax2.scatter(x=distance_from_integer(non_grid_cells["ML_Freqs"+suffix]), y=non_grid_cells["ML_SNRs"+suffix], color="black", marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(grid_cells["ML_Freqs"+suffix]), y=grid_cells["ML_SNRs"+suffix], color="r", marker="o", alpha=0.3)
    #ax2.axvline(x=avg_distance_from_integer_threshold, color="black", linestyle="dashed")
    #ax2.axhline(y=avg_SNR_ratio_threshold, color="black", linestyle="dashed")
    #ax1.set_yscale('log')
    #ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_GC_'+suffix+'.png', dpi=200)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    ax1.set_ylabel("Peak Power", color="black",fontsize=15, labelpad=10)
    ax1.set_xlabel("Spatial Frequency", color="black", fontsize=15, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax2.set_xticks([0,0.25, 0.5])
    plt.setp(ax1.get_xticklabels(), fontsize=15)
    plt.setp(ax2.get_xticklabels(), fontsize=10)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.xaxis.grid() # vertical lines
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax1.scatter(x=null_cells["ML_Freqs"+suffix], y=null_cells["ML_SNRs"+suffix], color="black", marker="o", alpha=0.3)
    ax1.scatter(x=distance_cells["ML_Freqs"+suffix], y=distance_cells["ML_SNRs"+suffix], color="orange", marker="o", alpha=0.3)
    ax1.scatter(x=position_cells["ML_Freqs"+suffix], y=position_cells["ML_SNRs"+suffix], color="turquoise", marker="o", alpha=0.3)
    #ax1.axhline(y=avg_SNR_ratio_threshold, xmin=0, xmax=10, color="black", linestyle="dashed")
    ax1.set_xlim([0,10])
    ax1.set_ylim([0,0.4])
    ax2.set_xlim([-0.1,0.6])
    ax2.set_ylim([0,0.4])
    ax2.set_xlabel(r'$\Delta$ from Integer', color="black", fontsize=15, labelpad=10)
    ax2.scatter(x=distance_from_integer(null_cells["ML_Freqs"+suffix]), y=null_cells["ML_SNRs"+suffix], color="black", marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(distance_cells["ML_Freqs"+suffix]), y=distance_cells["ML_SNRs"+suffix], color="orange", marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(position_cells["ML_Freqs"+suffix]), y=position_cells["ML_SNRs"+suffix], color="turquoise", marker="o", alpha=0.3)
    #ax2.axvline(x=avg_distance_from_integer_threshold, color="black", linestyle="dashed")
    #ax2.axhline(y=avg_SNR_ratio_threshold, color="black", linestyle="dashed")
    #ax1.set_yscale('log')
    #ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_DPN_'+suffix+'.png', dpi=200)
    plt.close()
    return

def plot_pairwise_classifiers(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = add_pairwise_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting pairwise classifers...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]
    distance_cells = concantenated_dataframe[concantenated_dataframe["Pairwise_classifier_"+suffix] == "Distance"]
    position_cells = concantenated_dataframe[concantenated_dataframe["Pairwise_classifier_"+suffix] == "Position"]
    null_cells = concantenated_dataframe[concantenated_dataframe["Pairwise_classifier_"+suffix] == "Null"]

    avg_SNR_ratio_threshold = 0.05
    avg_distance_from_integer_threshold =  0.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    ax1.set_ylabel("Pairwise Correlation",color="black",fontsize=15, labelpad=10)
    ax1.set_xlabel("Spatial Frequency", color="black", fontsize=15, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax2.set_xticks([0,0.25, 0.5])
    plt.setp(ax1.get_xticklabels(), fontsize=15)
    plt.setp(ax2.get_xticklabels(), fontsize=10)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.xaxis.grid() # vertical lines
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax1.scatter(x=non_grid_cells["pairwise_freq"+suffix], y=non_grid_cells["pairwise_corr"+suffix], color="black", marker="o", alpha=0.3)
    ax1.scatter(x=grid_cells["pairwise_freq"+suffix], y=grid_cells["pairwise_corr"+suffix], color="r", marker="o", alpha=0.3)
    ax1.axhline(y=avg_SNR_ratio_threshold, xmin=0, xmax=10, color="black", linestyle="dashed")
    ax1.set_xlim([0,10])
    ax1.set_ylim([-0.2,0.2])
    ax2.set_xlim([-0.1,0.6])
    ax2.set_ylim([-0.2,0.2])
    ax2.set_xlabel(r'$\Delta$ from Integer', color="black", fontsize=15, labelpad=10)
    ax2.scatter(x=distance_from_integer(non_grid_cells["pairwise_freq"+suffix]), y=non_grid_cells["pairwise_corr"+suffix], color="black", marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(grid_cells["pairwise_freq"+suffix]), y=grid_cells["pairwise_corr"+suffix], color="r", marker="o", alpha=0.3)
    ax2.axvline(x=avg_distance_from_integer_threshold, color="black", linestyle="dashed")
    ax2.axhline(y=avg_SNR_ratio_threshold, color="black", linestyle="dashed")
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path + '/pairwise_classifiers_GC_'+suffix+'.png', dpi=200)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,6), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    ax1.set_ylabel("SNR",color="black",fontsize=15, labelpad=10)
    ax1.set_xlabel("Spatial Frequency", color="black", fontsize=15, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax2.set_xticks([0,0.25, 0.5])
    plt.setp(ax1.get_xticklabels(), fontsize=15)
    plt.setp(ax2.get_xticklabels(), fontsize=10)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.xaxis.grid() # vertical lines
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax1.scatter(x=null_cells["pairwise_freq"+suffix], y=null_cells["pairwise_corr"+suffix], color="black", marker="o", alpha=0.3)
    ax1.scatter(x=distance_cells["pairwise_freq"+suffix], y=distance_cells["pairwise_corr"+suffix], color="orange", marker="o", alpha=0.3)
    ax1.scatter(x=position_cells["pairwise_freq"+suffix], y=position_cells["pairwise_corr"+suffix], color="turquoise", marker="o", alpha=0.3)
    ax1.axhline(y=avg_SNR_ratio_threshold, xmin=0, xmax=10, color="black", linestyle="dashed")
    ax1.set_xlim([0,10])
    ax1.set_ylim([-0.2,0.2])
    ax2.set_xlim([-0.1,0.6])
    ax2.set_ylim([-0.2,0.2])
    ax2.set_xlabel(r'$\Delta$ from Integer', color="black", fontsize=15, labelpad=10)
    ax2.scatter(x=distance_from_integer(null_cells["pairwise_freq"+suffix]), y=null_cells["pairwise_corr"+suffix], color="black", marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(distance_cells["pairwise_freq"+suffix]), y=distance_cells["pairwise_corr"+suffix], color="orange", marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(position_cells["pairwise_freq"+suffix]), y=position_cells["pairwise_corr"+suffix], color="turquoise", marker="o", alpha=0.3)
    ax2.axvline(x=avg_distance_from_integer_threshold, color="black", linestyle="dashed")
    ax2.axhline(y=avg_SNR_ratio_threshold, color="black", linestyle="dashed")
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path + '/pairwise_classifiers_DPN_'+suffix+'.png', dpi=200)
    plt.close()
    return


def plot_lomb_classifiers_proportions(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers proportions...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    fig, ax = plt.subplots(figsize=(4,6))
    groups = ["Position", "Distance", "Null"]
    colors_lm = ["turquoise", "orange", "gray"]
    objects = ["G", "NG"]
    x_pos = np.arange(len(objects))
    for object, x in zip(objects, x_pos):
        if object == "G":
            df = grid_cells
        elif object == "NG":
            df = non_grid_cells
        bottom=0
        for color, group in zip(colors_lm, groups):
            count = len(df[(df["Lomb_classifier_"+suffix] == group)])
            percent = (count/len(df))*100
            ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
            ax.text(x,bottom, str(count), color="k", fontsize=10, ha="center")
            bottom = bottom+percent
    plt.xticks(x_pos, objects, fontsize=15)
    plt.ylabel("Percent of neurons",  fontsize=25)
    plt.xlim((-0.5, len(objects)-0.5))
    plt.ylim((0,100))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.tight_layout()
    plt.subplots_adjust(left=0.4)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(save_path + '/lomb_classifiers_proportions_'+suffix+'.png', dpi=200)
    plt.close()
    return

def plot_lomb_classifiers_proportions_hmt(concantenated_dataframe, save_path=""):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix="PI")
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix="PI_try")
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix="PI_miss")
    print('plotting lomb classifers proportions...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    for df, c in zip([grid_cells, non_grid_cells], ["gc", "ngc"]):
        fig, ax = plt.subplots(figsize=(4,6))
        groups = ["Position", "Distance", "Null"]
        colors_lm = ["turquoise", "orange", "gray"]
        objects = ["Hit", "Try", "Miss"]
        x_pos = np.arange(len(objects))

        for object, x in zip(objects, x_pos):
            if object == "Hit":
                collumn = "Lomb_classifier_PI"
            elif group == "Try":
                collumn = "Lomb_classifier_PI_try"
            elif group == "Miss":
                collumn = "Lomb_classifier_PI_miss"
            bottom=0

            for color, group in zip(colors_lm, groups):
                count = len(df[(df[collumn] == group)])
                percent = (count/len(df))*100
                ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
                ax.text(x,bottom, str(count), color="k", fontsize=10, ha="center")
                bottom = bottom+percent

        plt.xticks(x_pos, objects, fontsize=15)
        plt.ylabel("Percent of neurons",  fontsize=25)
        plt.xlim((-0.5, len(objects)-0.5))
        plt.ylim((0,100))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.subplots_adjust(left=0.4)
        ax.tick_params(axis='both', which='major', labelsize=25)
        if c == "gc":
            plt.savefig(save_path + '/lomb_classifiers_proportions_hmt_'+c+'_.png', dpi=200)
        elif c == "ngc":
            plt.savefig(save_path + '/lomb_classifiers_proportions_hmt_'+c+'_.png', dpi=200)
    plt.close()
    return

def plot_grid_scores_by_classifier(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers grid scores...')
    classifier_collumn = "Lomb_classifier_"+suffix

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    G_P = grid_cells[grid_cells[classifier_collumn] == "Position"]
    G_D = grid_cells[grid_cells[classifier_collumn] == "Distance"]
    G_N = grid_cells[grid_cells[classifier_collumn] == "Null"]
    NG_P = non_grid_cells[non_grid_cells[classifier_collumn] == "Position"]
    NG_D = non_grid_cells[non_grid_cells[classifier_collumn] == "Distance"]
    NG_N = non_grid_cells[non_grid_cells[classifier_collumn] == "Null"]

    fig, ax = plt.subplots(figsize=(6,6))

    _, _, patches1 = ax.hist(np.asarray(G_P["grid_score"]), bins=500, color="turquoise", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches2 = ax.hist(np.asarray(G_D["grid_score"]), bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches3 = ax.hist(np.asarray(G_N["grid_score"]), bins=500, color="gray", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches4 = ax.hist(np.asarray(NG_P["grid_score"]), bins=500, color="turquoise", histtype="step", density=True, cumulative=True, linewidth=2, linestyle="dotted")
    _, _, patches5 = ax.hist(np.asarray(NG_D["grid_score"]), bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=2, linestyle="dotted")
    _, _, patches6 = ax.hist(np.asarray(NG_N["grid_score"]), bins=500, color="gray", histtype="step", density=True, cumulative=True, linewidth=2, linestyle="dotted")

    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    patches2[0].set_xy(patches2[0].get_xy()[:-1])
    patches3[0].set_xy(patches3[0].get_xy()[:-1])
    patches4[0].set_xy(patches4[0].get_xy()[:-1])
    patches5[0].set_xy(patches5[0].get_xy()[:-1])
    patches6[0].set_xy(patches6[0].get_xy()[:-1])

    plt.ylabel("Density",  fontsize=20)
    plt.xlabel("Grid Score",  fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_grid_scores_'+suffix+'.png', dpi=300)
    plt.close()
    return

def plot_of_stability_by_classifier(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers half session scores...')
    classifier_collumn = "Lomb_classifier_"+suffix

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    G_P = grid_cells[grid_cells[classifier_collumn] == "Position"]
    G_D = grid_cells[grid_cells[classifier_collumn] == "Distance"]
    G_N = grid_cells[grid_cells[classifier_collumn] == "Null"]
    NG_P = non_grid_cells[non_grid_cells[classifier_collumn] == "Position"]
    NG_D = non_grid_cells[non_grid_cells[classifier_collumn] == "Distance"]
    NG_N = non_grid_cells[non_grid_cells[classifier_collumn] == "Null"]

    fig, ax = plt.subplots(figsize=(6,6))

    _, _, patches1 = ax.hist(np.asarray(G_P["rate_map_correlation_first_vs_second_half"]), bins=500, color="turquoise", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches2 = ax.hist(np.asarray(G_D["rate_map_correlation_first_vs_second_half"]), bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches3 = ax.hist(np.asarray(G_N["rate_map_correlation_first_vs_second_half"]), bins=500, color="gray", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches4 = ax.hist(np.asarray(NG_P["rate_map_correlation_first_vs_second_half"]), bins=500, color="turquoise", histtype="step", density=True, cumulative=True, linewidth=2, linestyle="dotted")
    _, _, patches5 = ax.hist(np.asarray(NG_D["rate_map_correlation_first_vs_second_half"]), bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=2, linestyle="dotted")
    _, _, patches6 = ax.hist(np.asarray(NG_N["rate_map_correlation_first_vs_second_half"]), bins=500, color="gray", histtype="step", density=True, cumulative=True, linewidth=2, linestyle="dotted")

    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    patches2[0].set_xy(patches2[0].get_xy()[:-1])
    patches3[0].set_xy(patches3[0].get_xy()[:-1])
    patches4[0].set_xy(patches4[0].get_xy()[:-1])
    patches5[0].set_xy(patches5[0].get_xy()[:-1])
    patches6[0].set_xy(patches6[0].get_xy()[:-1])

    plt.ylabel("Density",  fontsize=20)
    plt.xlabel("Half-session Stability",  fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_hs_stability_'+suffix+'.png', dpi=300)
    plt.close()
    return

def plot_of_stability_vs_grid_score_by_classifier(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers half session scores...')
    classifier_collumn = "Lomb_classifier_"+suffix

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]

    G_P = grid_cells[grid_cells[classifier_collumn] == "Position"]
    G_D = grid_cells[grid_cells[classifier_collumn] == "Distance"]
    G_N = grid_cells[grid_cells[classifier_collumn] == "Null"]

    g_p_grid_score = np.asarray(G_P["grid_score"])
    g_p_spatial_score = np.asarray(G_P["rate_map_correlation_first_vs_second_half"])
    g_d_grid_score = np.asarray(G_D["grid_score"])
    g_d_spatial_score = np.asarray(G_D["rate_map_correlation_first_vs_second_half"])
    g_n_grid_score = np.asarray(G_N["grid_score"])
    g_n_spatial_score = np.asarray(G_N["rate_map_correlation_first_vs_second_half"])

    g_p_grid_score_m, g_p_grid_score_s = (np.nanmean(g_p_grid_score), np.nanstd(g_p_grid_score))
    g_p_spatial_score_m, g_p_spatial_score_s = (np.nanmean(g_p_spatial_score), np.nanstd(g_p_spatial_score))
    g_d_grid_score_m, g_d_grid_score_s = (np.nanmean(g_d_grid_score), np.nanstd(g_d_grid_score))
    g_d_spatial_score_m, g_d_spatial_score_s = (np.nanmean(g_d_spatial_score),np.nanstd(g_d_spatial_score))
    g_n_grid_score_m, g_n_grid_score_s = (np.nanmean(g_n_grid_score), np.nanstd(g_n_grid_score))
    g_n_spatial_score_m, g_n_spatial_score_s = (np.nanmean(g_n_spatial_score), np.nanstd(g_n_spatial_score))

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(g_n_grid_score, g_n_spatial_score, marker="o", alpha=0.3, color="gray")
    ax.scatter(g_p_grid_score, g_p_spatial_score, marker="o", alpha=0.3, color="turquoise")
    ax.scatter(g_d_grid_score, g_d_spatial_score, marker="o", alpha=0.3, color="orange")

    ec1 = patches.Ellipse((g_p_grid_score_m, g_p_spatial_score_m), g_p_grid_score_s, g_p_spatial_score_s, angle=0, alpha=0.3, color="turquoise")
    ec2 = patches.Ellipse((g_d_grid_score_m, g_d_spatial_score_m), g_d_grid_score_s, g_d_spatial_score_s, angle=0, alpha=0.3, color="orange")
    ec3 = patches.Ellipse((g_n_grid_score_m, g_n_spatial_score_m), g_n_grid_score_s, g_n_spatial_score_s, angle=0, alpha=0.3, color="gray")

    ax.add_patch(ec1)
    ax.add_patch(ec2)
    ax.add_patch(ec3)
    plt.ylabel("Half-session Stability",  fontsize=20)
    plt.xlabel("Grid Score",  fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_hs_stability_vs_gs_'+suffix+'.png', dpi=300)
    plt.close()
    return

def plot_lomb_classifiers_proportions_by_mouse(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers proportions by mouse...')

    for mouse in np.unique(concantenated_dataframe["mouse"]):
        mouse_concatenated_dataframe = concantenated_dataframe[(concantenated_dataframe["mouse"] == mouse)]
        grid_cells = mouse_concatenated_dataframe[mouse_concatenated_dataframe["classifier"] == "G"]
        non_grid_cells = mouse_concatenated_dataframe[mouse_concatenated_dataframe["classifier"] != "G"]

        fig, ax = plt.subplots(figsize=(4,6))
        groups = ["Position", "Distance", "Null"]
        colors_lm = ["turquoise", "orange", "gray"]
        objects = ["G", "NG"]
        x_pos = np.arange(len(objects))

        for object, x in zip(objects, x_pos):
            if object == "G":
                df = grid_cells
            elif object == "NG":
                df = non_grid_cells
            bottom=0
            for color, group in zip(colors_lm, groups):
                count = len(df[(df["Lomb_classifier_"+suffix] == group)])
                percent = (count/len(df))*100
                ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
                ax.text(x,bottom, str(count), color="k", fontsize=10, ha="center")
                bottom = bottom+percent

        plt.xticks(x_pos, objects, fontsize=15)
        plt.ylabel("Percent of neurons",  fontsize=25)
        plt.xlim((-0.5, len(objects)-0.5))
        plt.ylim((0,100))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.tight_layout()
        plt.subplots_adjust(left=0.4)
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.savefig(save_path + '/lomb_classifiers_proportions_'+str(mouse)+"_"+suffix+'.png', dpi=200)
        plt.close()
    return

def plot_snr_vs_RZbias_regression(spike_data, processed_position_data, output_path, track_length):
    print('plotting TI vs SNR regressions...')
    save_path = output_path + '/Figures/TI_regressions'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    step = 0.02
    frequency = np.arange(0.1, 5+step, step)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            SNRs = []
            TIs= []
            TTs= []
            for trial in np.unique(centre_trials):
                trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial]
                trial_powers = powers[centre_trials == trial]
                avg_powers = np.nanmean(trial_powers, axis=0)
                max_SNR, max_freq = get_max_SNR(frequency, avg_powers)
                SNRs.append(max_SNR)
                TTs.append(trial_processed_position_data["trial_type"].iloc[0])
                TIs.append(trial_processed_position_data["RZ_stop_bias"].iloc[0])
            SNRs = np.array(SNRs)
            trials = np.unique(centre_trials)
            TIs = np.array(TIs)
            TTs= np.array(TTs)


            stops_on_track = plt.figure(figsize=(6,6))
            ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            for tt, c, y_text in zip([0,1,2], ["black", "red", "blue"], [0.9, 0.8, 0.7]):
                ax.scatter(SNRs[TTs==tt], np.log(TIs)[TTs==tt], edgecolor=c, facecolor="none", marker="o")
                plot_regression(ax, y=np.log(TIs)[TTs==tt], x=SNRs[TTs==tt], c=c, y_text_pos=y_text)
            plt.xlabel('Power', fontsize=20, labelpad = 10)
            plt.ylabel("log(Task Index)", fontsize=20, labelpad = 10)
            #ax.set_xticks([0, np.round(ax.get_xlim()[1], 2)])
            #ax.set_xlim(left=0, right=np.round(ax.get_xlim()[1], 2))
            #ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
            #ax.set_ylim(bottom=0, top=np.round(ax.get_ylim()[1], 2))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/TI_regressions_'+str(cluster_id)+'.png', dpi=200)
            plt.close()
    return

def plot_regression(ax, x, y, c, y_text_pos):
    # x  and y are pandas collumn
    try:
        x = x.values
        y = y.values
    except Exception as ex:
        print("")

    x = x[~np.isnan(y)].reshape(-1, 1)
    y = y[~np.isnan(y)].reshape(-1, 1)

    pearson_r = stats.pearsonr(x.flatten(),y.flatten())

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x,y)  # perform linear regression

    x_test = np.linspace(min(x), max(x), 100)

    Y_pred = linear_regressor.predict(x_test.reshape(-1, 1))  # make predictions
    #ax.text(6, 0.65, "R= "+str(np.round(pearson_r[0], decimals=2))+ ", p = "+str(np.round(pearson_r[1], decimals=2)))

    ax.text(  # position text relative to Axes
        0.05, y_text_pos, "R= "+str(np.round(pearson_r[0], decimals=2))+ ", p = "+str(np.round(pearson_r[1], decimals=4)),
        ha='left', va='top', color=c,
        transform=ax.transAxes, fontsize=10)

    ax.plot(x_test, Y_pred, color=c)

def plot_snr_by_hmt_tt(spike_data, processed_position_data, output_path, track_length):
    print('plotting the power by hmt...')
    save_path = output_path + '/Figures/moving_lomb_power_by_hmt'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = 0.02
    frequency = np.arange(0.1, 5+step, step)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])


        stops_on_track = plt.figure(figsize=(6,6))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for f in range(1,6):
            ax.axvline(x=f, color="gray", linewidth=2,linestyle="dashed")

        for tt in [0, 1, 2]:
            for hmt in ["hit"]:
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
                        sem_subset_powers = scipy.stats.sem(subset_powers, axis=0, nan_policy="omit")
                        ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color=get_trial_color(tt), alpha=0.3)
                        ax.plot(frequency, avg_subset_powers, color=get_trial_color(tt))

        plt.ylabel('Periodic Power', fontsize=20, labelpad = 10)
        plt.xlabel("Track Frequency", fontsize=20, labelpad = 10)
        plt.xlim(0,5.05)
        ax.set_xticks([0,5])
        ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
        ax.set_ylim(bottom=0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #ax.set_yticks([0, 10, 20, 30])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/hit_tt_powers_'+str(cluster_id)+'.png', dpi=200)
        plt.close()
    return

def plot_snr_by_hmt(spike_data, processed_position_data, output_path, track_length):
    print('plotting the power by hmt...')
    save_path = output_path + '/Figures/moving_lomb_power_by_hmt'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = 0.02
    frequency = np.arange(0.1, 5+step, step)
    #processed_position_data = processed_position_data[(processed_position_data["trial_type"] == 1) | (processed_position_data["trial_type"] == 2)]

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        centre_trials = np.round(centre_trials).astype(np.int64)
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        stops_on_track = plt.figure(figsize=(6,6))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for f in range(1,6):
            ax.axvline(x=f, color="gray", linewidth=2,linestyle="dashed")

        for tt in ["all"]:
            for hmt in ["hit", "miss", "try"]:
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
                        sem_subset_powers = scipy.stats.sem(subset_powers, axis=0, nan_policy="omit")
                        ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color=get_hmt_color(hmt), alpha=0.3)
                        ax.plot(frequency, avg_subset_powers, color=get_hmt_color(hmt))

        plt.ylabel('Periodic Power', fontsize=20, labelpad = 10)
        plt.xlabel("Track Frequency", fontsize=20, labelpad = 10)
        plt.xlim(0,5.05)
        ax.set_xticks([0,5])
        ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
        ax.set_ylim(bottom=0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #ax.set_yticks([0, 10, 20, 30])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/hmt_powers_'+str(cluster_id)+'.png', dpi=200)
        plt.close()
    return

def plot_power_by_hmt(spike_data, processed_position_data, output_path, track_length):
    print('plotting the power by hmt...')
    save_path = output_path + '/Figures/moving_lomb_power_by_hmt'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = 0.02
    frequency = np.arange(0.1, 5+step, step)

    hit_miss_p_ = []
    hit_try_p_ = []
    try_miss_p_ = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>0:

            powers[np.isnan(powers)] = 0
            SNRs = []
            HMTs = []
            for trial in np.unique(centre_trials):
                trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial]
                hmt = trial_processed_position_data["hit_miss_try"].iloc[0]
                trial_powers = powers[centre_trials == trial]
                avg_powers = np.nanmean(trial_powers, axis=0)
                max_SNR, max_freq = get_max_SNR(frequency, avg_powers)
                SNRs.append(max_SNR)
                HMTs.append(hmt)
            SNRs=np.array(SNRs)
            HMTs=np.array(HMTs)

            fig, ax = plt.subplots(figsize=(4,4))
            ax.set_ylabel("Power", fontsize=30, labelpad=10)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            objects = ["Hit", "Try", "Miss"]
            x_pos = np.arange(len(objects))

            if len(SNRs[HMTs=="hit"])>1:
                ax.errorbar(x_pos[0], np.nanmean(SNRs[HMTs=="hit"]), yerr=stats.sem(SNRs[HMTs=="hit"], nan_policy='omit'), ecolor='green', capsize=20, fmt="o", color="green")
                ax.scatter(x_pos[0]*np.ones(len(SNRs[HMTs=="hit"])), np.asarray(SNRs[HMTs=="hit"]), edgecolor="green", marker="o", facecolors='none')
            if len(SNRs[HMTs=="try"])>1:
                ax.errorbar(x_pos[1], np.nanmean(SNRs[HMTs=="try"]), yerr=stats.sem(SNRs[HMTs=="try"], nan_policy='omit'), ecolor='orange', capsize=20, fmt="o", color="orange")
                ax.scatter(x_pos[1]*np.ones(len(SNRs[HMTs=="try"])), np.asarray(SNRs[HMTs=="try"]), edgecolor="orange", marker="o", facecolors='none')
            if len(SNRs[HMTs=="miss"])>1:
                ax.errorbar(x_pos[2], np.nanmean(SNRs[HMTs=="miss"]), yerr=stats.sem(SNRs[HMTs=="miss"], nan_policy='omit'), ecolor='red', capsize=20, fmt="o", color="red")
                ax.scatter(x_pos[2]*np.ones(len(SNRs[HMTs=="miss"])), np.asarray(SNRs[HMTs=="miss"]), edgecolor="red", marker="o", facecolors='none')

            if (len(SNRs[HMTs=="hit"])>0) and (len(SNRs[HMTs=="miss"])>0):
                hit_miss_p = stats.ttest_ind(SNRs[HMTs=="hit"], SNRs[HMTs=="miss"])[1]
            else:
                hit_miss_p = np.nan

            if (len(SNRs[HMTs=="hit"])>0) and (len(SNRs[HMTs=="try"])>0):
                hit_try_p =  stats.ttest_ind(SNRs[HMTs=="hit"], SNRs[HMTs=="try"])[1]
            else:
                hit_try_p = np.nan

            if (len(SNRs[HMTs=="try"])>0) and (len(SNRs[HMTs=="miss"])>0):
                try_miss_p = stats.ttest_ind(SNRs[HMTs=="try"], SNRs[HMTs=="miss"])[1]
            else:
                try_miss_p = np.nan

            all_behaviour = []; all_behaviour.extend(SNRs[HMTs=="hit"].tolist()); all_behaviour.extend(SNRs[HMTs=="try"].tolist()); all_behaviour.extend(SNRs[HMTs=="miss"].tolist())
            significance_bar(start=x_pos[0], end=x_pos[1], height=np.nanmax(all_behaviour)+0, displaystring=get_p_text(hit_try_p))
            significance_bar(start=x_pos[1], end=x_pos[2], height=np.nanmax(all_behaviour)+0.1, displaystring=get_p_text(try_miss_p))
            significance_bar(start=x_pos[0], end=x_pos[2], height=np.nanmax(all_behaviour)+0.2, displaystring=get_p_text(hit_miss_p))

            plt.xticks(x_pos, objects, fontsize=30)
            plt.xlim((-0.5, len(objects)-0.5))
            #plt.xticks(rotation=-45)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(save_path + '/hmt_powers_test_'+str(cluster_id)+'.png', dpi=200)
            plt.close()

            hit_miss_p_.append(hit_miss_p)
            hit_try_p_.append(hit_try_p)
            try_miss_p_.append(try_miss_p)
        else:
            hit_miss_p_.append(np.nan)
            hit_try_p_.append(np.nan)
            try_miss_p_.append(np.nan)

    spike_data["power_test_hit_miss_p"] = hit_miss_p_
    spike_data["power_test_hit_try_p"] = hit_try_p_
    spike_data["power_test_try_miss_p"] = try_miss_p_
    return spike_data


def plot_power_trajectories(spike_data, processed_position_data, output_path, track_length):
    print('plotting power trajectories...')
    save_path = output_path + '/Figures/power_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = 0.02
    frequency = np.arange(0.1, 5+step, step)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        cluster_spike_data = add_lomb_classifier(cluster_spike_data)
        Lomb_classifier_ = cluster_spike_data["Lomb_classifier_"].iloc[0]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        powers[np.isnan(powers)] = 0
        SNRs = []
        freqs = []
        classes = []
        for trial in np.unique(centre_trials):
            trial_powers = powers[centre_trials == trial]
            avg_powers = np.nanmean(trial_powers, axis=0)
            max_SNR, max_freq = get_max_SNR(frequency, avg_powers)
            lomb_class = get_lomb_classifier(max_SNR, max_freq, 0.03, 0.05, numeric=False)
            classes.append(lomb_class)
            SNRs.append(max_SNR)
            freqs.append(max_freq)
        SNRs=np.array(SNRs)
        freqs= np.array(freqs)

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        step = 0.02
        frequency = np.arange(0.1, 5+step, step)
        ax.plot(np.unique(centre_trials), freqs, color="black", linewidth=1, alpha=0.2)
        avg_powers = np.nanmean(powers, axis=0)
        #ax.plot(frequency, avg_powers, color="blue")
        plt.xlabel('Centre Trial', fontsize=20, labelpad = 10)
        plt.ylabel('Hz @ Max Power', fontsize=20, labelpad = 10)
        plt.xlim(0,max(np.unique(centre_trials)))
        plt.ylim(bottom=0)#
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_trajectory_Cluster_' + str(cluster_id) + '.png', dpi=300)
        plt.close()

    return

def plot_pca_space(spike_data, processed_position_data, output_path, track_length):
    print('plotting pca...')
    save_path = output_path + '/Figures/pca_space'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = 0.02
    frequency = np.arange(0.1, 5+step, step)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        cluster_spike_data = add_lomb_classifier(cluster_spike_data)
        Lomb_classifier_ = cluster_spike_data["Lomb_classifier_"].iloc[0]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        powers[np.isnan(powers)] = 0
        targets = []
        features = []
        for trial in np.unique(centre_trials):
            trial_powers = powers[centre_trials == trial]
            avg_powers = np.nanmean(trial_powers, axis=0)
            max_SNR, max_freq = get_max_SNR(frequency, avg_powers)
            lomb_class = get_lomb_classifier(max_SNR, max_freq, 0.03, 0.05, numeric=True)
            targets.append(lomb_class)
            features.append(avg_powers.tolist())
        target=np.array(targets)
        features= np.array(features)

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        step = 0.02
        frequency = np.arange(0.1, 5+step, step)
        for i in range(len(features)):
            ax.plot(frequency, features[i], color="black", linewidth=2, alpha=0.05)
        avg_powers = np.nanmean(powers, axis=0)
        ax.plot(frequency, avg_powers, color="blue")
        max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_powers)
        max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
        max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))
        ax.text(0.9, 0.9, max_SNR_text, ha='right', va='center', transform=ax.transAxes, fontsize=10)
        ax.text(0.9, 0.8, max_SNR_freq_test, ha='right', va='center', transform=ax.transAxes, fontsize=10)
        plt.xlabel('Spatial Frequency', fontsize=20, labelpad = 10)
        plt.ylabel('Power', fontsize=20, labelpad = 10)
        plt.xlim(0,max(frequency))
        plt.ylim(bottom=0)#
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_scargle_avg_periodogram_Cluster_' + str(cluster_id) + '.png', dpi=300)
        plt.close()

        pca = PCA(n_components=10)
        pca.fit(powers)
        var = pca.explained_variance_ratio_

        RANDOM_STATE = 42
        FIG_SIZE = (10, 7)

        # Make a train/test split using 30% test size
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.30, random_state=RANDOM_STATE
        )

        # Fit to data and predict using pipelined GNB and PCA.
        unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
        unscaled_clf.fit(X_train, y_train)
        pred_test = unscaled_clf.predict(X_test)

        # Fit to data and predict using pipelined scaling, GNB and PCA.
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
        std_clf.fit(X_train, y_train)
        pred_test_std = std_clf.predict(X_test)

        # Show prediction accuracies in scaled and unscaled data.
        print("\nPrediction accuracy for the normal test dataset with PCA")
        print("{:.2%}\n".format(metrics.accuracy_score(y_test, pred_test)))

        print("\nPrediction accuracy for the standardized test dataset with PCA")
        print("{:.2%}\n".format(metrics.accuracy_score(y_test, pred_test_std)))

        # Extract PCA from pipeline
        pca = unscaled_clf.named_steps["pca"]
        pca_std = std_clf.named_steps["pca"]

        # Show first principal components
        #print("\nPC 1 without scaling:\n", pca.components_[0])
        #print("\nPC 1 with scaling:\n", pca_std.components_[0])

        # Use PCA without and with scale on X_train data for visualization.
        X_train_transformed = pca.transform(X_train)
        scaler = std_clf.named_steps["standardscaler"]
        X_train_std_transformed = pca_std.transform(scaler.transform(X_train))

        # visualize standardized vs. untouched dataset with PCA performed
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)


        for l, c, m in zip(range(0,3), ("blue", "red", "green"), ("^", "s", "o")):
            ax1.scatter(
                X_train_transformed[y_train == l, 0],
                X_train_transformed[y_train == l, 1],
                color=c,
                label=get_class_str(l),
                alpha=0.5,
                marker=m,
            )

        for l, c, m in zip(range(0,3), ("blue", "red", "green"), ("^", "s", "o")):
            ax2.scatter(
                X_train_std_transformed[y_train == l, 0],
                X_train_std_transformed[y_train == l, 1],
                color=c,
                label=get_class_str(l),
                alpha=0.5,
                marker=m,
            )

        ax1.set_title("Training dataset after PCA")
        ax2.set_title("Standardized training dataset after PCA")

        for ax in (ax1, ax2):
            ax.set_xlabel("1st principal component")
            ax.set_ylabel("2nd principal component")
            ax.legend(loc="upper right")
            ax.grid()

        plt.tight_layout()
        plt.savefig(save_path + '/pca_powers_'+str(cluster_id)+'.png', dpi=200)
        plt.show()
    return

def get_class_str(lomb_numeric_class):
    if lomb_numeric_class == 0:
        return "Position"
    elif lomb_numeric_class == 1:
        return "Distance"
    elif lomb_numeric_class == 2:
        return "Null"

def plot_snr_by_trial_type(spike_data, processed_position_data, output_path, track_length):
    print('plotting the power by trial_type...')
    save_path = output_path + '/Figures/moving_lomb_power_by_tt'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = 0.02
    frequency = np.arange(0.1, 5+step, step)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])


        stops_on_track = plt.figure(figsize=(6,6))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for f in range(1,6):
            ax.axvline(x=f, color="gray", linewidth=2,linestyle="dashed")

        for tt in [0, 1, 2]:
            for hmt in ["all"]:
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
                        sem_subset_powers = scipy.stats.sem(subset_powers, axis=0, nan_policy="omit")
                        ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color=get_trial_color(tt), alpha=0.3)
                        ax.plot(frequency, avg_subset_powers, color=get_trial_color(tt))

        plt.ylabel('Periodic Power', fontsize=20, labelpad = 10)
        plt.xlabel("Track Frequency", fontsize=20, labelpad = 10)
        plt.xlim(0,5.05)
        ax.set_xticks([0,5])
        ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
        ax.set_ylim(bottom=0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #ax.set_yticks([0, 10, 20, 30])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/tt_powers_'+str(cluster_id)+'.png', dpi=200)
        plt.close()
    return


def plot_avg_lomb(spike_data, output_path):
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        avg_power = cluster_spike_data["MOVING_LOMB_avg_power"].iloc[0]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            step = 0.02
            frequency = np.arange(0.1, 5+step, step)
            ax.plot(frequency, avg_power, color="black", linewidth=2)
            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
            max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
            max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))
            ax.text(0.9, 0.9, max_SNR_text, ha='right', va='center', transform=ax.transAxes, fontsize=10)
            ax.text(0.9, 0.8, max_SNR_freq_test, ha='right', va='center', transform=ax.transAxes, fontsize=10)
            plt.xlabel('Spatial Frequency', fontsize=20, labelpad = 10)
            plt.ylabel('Power', fontsize=20, labelpad = 10)
            plt.xlim(0,max(frequency))
            plt.ylim(bottom=0)#
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_scargle_avg_periodogram_Cluster_' + str(cluster_id) + '.png', dpi=300)
            plt.close()

def plot_peak_histogram(spike_data, processed_position_data, output_path, track_length):
    spike_data = add_lomb_classifier(spike_data)
    print('plotting joint cell correlations...')
    save_path = output_path + '/Figures/lomb_peak_histograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = 0.02
    frequency = np.arange(0.1, 5+step, step)
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        MOVING_LOMB_all_powers = cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0]
        MOVING_LOMB_all_centre_trials = cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            trial_averaged_powers = []
            SNRs = []
            freqs = []
            trial_numbers = np.unique(MOVING_LOMB_all_centre_trials)
            for trial in trial_numbers:
                trial_powers = MOVING_LOMB_all_powers[MOVING_LOMB_all_centre_trials == trial]
                avg_powers = np.nanmean(trial_powers, axis=0)
                max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_powers)
                freqs.append(max_SNR_freq)
                SNRs.append(max_SNR)
                trial_averaged_powers.append(avg_powers.tolist())
            trial_averaged_powers = np.array(trial_averaged_powers)
            SNRs=np.array(SNRs)
            freqs=np.array(freqs)

            fig, ax = plt.subplots(figsize=(4,4))
            n_bins=100
            weights, bin_edges = np.histogram(freqs, bins=n_bins, range=(0, 10), weights=SNRs)
            bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            ax.bar(bin_centres, weights, width=10/n_bins, color="blue")
            ax.set_xlim([0,10])
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.set_title("ID: "+str(cluster_id), fontsize= 25)
            ax.set_ylabel("", fontsize=20)
            ax.set_xlabel("Spatial Frequency", fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=15)
            fig.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + '_peaks_cluster_'+str(cluster_id)+'.png', dpi=300)
            plt.close()
    return



def plot_trial_fr_cross_correlations(spike_data, processed_position_data, output_path, track_length):
    spike_data = add_lomb_classifier(spike_data)
    print('plotting joint cell correlations...')
    save_path = output_path + '/Figures/trial_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_rates = np.array(cluster_spike_data["fr_binned_in_space"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            cross_correlation_matrix = np.zeros((len(firing_rates), len(firing_rates)))
            for i in range(len(firing_rates)):
                for j in range(len(firing_rates)):
                    corr = pearsonr(firing_rates[i], firing_rates[j])[0]
                    cross_correlation_matrix[i, j] = corr

            cross_correlation_matrix[np.isnan(cross_correlation_matrix)] = 0
            fig, ax = plt.subplots()
            im= ax.imshow(cross_correlation_matrix, cmap="inferno", vmin=0, vmax=1)
            ax.set_title("ID: "+str(cluster_id), fontsize= 25)
            ax.set_ylabel("Trial", fontsize=20)
            ax.set_xlabel("Trial", fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=10)
            fig.tight_layout()
            fig.colorbar(im, ax=ax)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + '_fr_trial_correlations_cluster_'+str(cluster_id)+'.png', dpi=300)
            plt.close()
    return

def plot_trial_cross_correlations(spike_data, processed_position_data, output_path, track_length):
    spike_data = add_lomb_classifier(spike_data)
    print('plotting joint cell correlations...')
    save_path = output_path + '/Figures/trial_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        MOVING_LOMB_all_powers = cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0]
        MOVING_LOMB_all_centre_trials = cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            trial_averaged_powers = []
            trial_numbers = np.unique(MOVING_LOMB_all_centre_trials)
            for trial in trial_numbers:
                trial_powers = MOVING_LOMB_all_powers[MOVING_LOMB_all_centre_trials == trial]
                avg_powers = np.nanmean(trial_powers, axis=0)
                trial_averaged_powers.append(avg_powers.tolist())
            trial_averaged_powers = np.array(trial_averaged_powers)

            cross_correlation_matrix = np.zeros((len(trial_numbers), len(trial_numbers)))
            for i, trial_centre_i in enumerate(trial_numbers):
                for j, trial_centre_j in enumerate(trial_numbers):
                    corr = pearsonr(trial_averaged_powers[i], trial_averaged_powers[j])[0]
                    cross_correlation_matrix[i, j] = corr

            cross_correlation_matrix[np.isnan(cross_correlation_matrix)] = 0
            fig, ax = plt.subplots()
            im= ax.imshow(cross_correlation_matrix, cmap="inferno", vmin=0, vmax=1)
            #ax.set_xticks([])
            #ax.set_yticks([])
            #ax.set_yticklabels([])
            #ax.set_xticklabels([])

            colors = []
            #for i in range(len(cross_correlation_matrix)):
            #    colors.append("sdsdsd")
            #for xtick, color in zip(ax.get_xticklabels(), colors):
            #    xtick.set_color(color)
            #for ytick, color in zip(ax.get_yticklabels(), colors):
            #    ytick.set_color(color)
            ax.set_title("ID: "+str(cluster_id), fontsize= 25)
            ax.set_ylabel("Centre Trials", fontsize=20)
            ax.set_xlabel("Centre Trials", fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=10)
            fig.tight_layout()
            fig.colorbar(im, ax=ax)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + '_trial_correlations_cluster_'+str(cluster_id)+'.png', dpi=300)
            plt.close()
    return

def print_grid_cells_cluster_ids_and_scores(combined_df):
    recordings = np.unique(combined_df["session_id"])

    session_ids = []
    grid_scores = []
    cluster_ids = []
    classifiers =[]
    for session_id in recordings:
        recording_df = combined_df[combined_df["session_id"]==session_id]
        grid_cells = recording_df[recording_df["classifier"] == "G"]
        n_grid_cells = len(grid_cells)
        if n_grid_cells>0:
            for index, grid_cell in grid_cells.iterrows():
                grid_cell = grid_cell.to_frame().T.reset_index(drop=True)
                grid_score = grid_cell["grid_score"].iloc[0]
                cluster_id = grid_cell["cluster_id"].iloc[0]
                session_id = grid_cell["session_id"].iloc[0]
                lomb_class = grid_cell["Lomb_classifier_"].iloc[0]

                session_ids.append(session_id)
                grid_scores.append(grid_score)
                cluster_ids.append(cluster_id)
                classifiers.append(lomb_class)

    print("Session IDs")
    for i in range(len(session_ids)):
        print(session_ids[i])
    print("grid_scores")
    for i in range(len(session_ids)):
        print(grid_scores[i])
    print("classifiers")
    for i in range(len(session_ids)):
        print(classifiers[i])
    print("cluster_ids")
    for i in range(len(session_ids)):
        print(cluster_ids[i])
    return

def print_grid_cells_per_recording(combined_df):
    recordings = np.unique(combined_df["session_id"])

    Ns = []
    Joint_grid_recordings = []
    for session_id in recordings:
        recording_df = combined_df[combined_df["session_id"]==session_id]
        grid_cells = recording_df[recording_df["classifier"] == "G"]
        n_grid_cells = len(grid_cells)
        if n_grid_cells>0:
            Ns.append(n_grid_cells)
            Joint_grid_recordings.append("/mnt/datastore/Harry/cohort8_may2021/vr/"+session_id)

            print(session_id, " has ",n_grid_cells , " jointly recorded grid cells")
            print("cluster IDs: ")
            print(grid_cells["cluster_id"])
            print("are grid cells")

    print(Joint_grid_recordings)
    return

def read_df(combined_df):
    print_grid_cells_per_recording(combined_df)
    print_grid_cells_cluster_ids_and_scores(combined_df)

def sort_lomb_class_colors(lomb_classes_colors):

    lomb_classes_colors = np.hstack((lomb_classes_colors, np.arange(len(lomb_classes_colors)).reshape(1, len(lomb_classes_colors)).T))
    returned_array = []
    clusters_already_added = []
    uniques_I = np.unique(lomb_classes_colors[:, 0])
    for unique_I in uniques_I:
        temp_I = lomb_classes_colors[lomb_classes_colors[:,0]==unique_I]
        uniques_II = np.unique(temp_I[:, 1])
        for unique_II in uniques_II:
            temp_II = temp_I[temp_I[:,1]==unique_II]
            uniques_III = np.unique(temp_II[:, 2])
            for unique_III in uniques_III:
                temp_III = temp_II[temp_II[:,2]==unique_III]
                uniques_IV = np.unique(temp_III[:, 3])
                for unique_IV in uniques_IV:
                    temp_IV = temp_III[temp_III[:,3]==unique_IV]
                    for i in range(len(temp_IV)):
                        cluster_idx = temp_IV[i, -1]
                        if cluster_idx not in clusters_already_added:
                            clusters_already_added.append(cluster_idx)
                            returned_array.append(temp_IV[i].tolist()[0:-1])

    return np.array(returned_array)

def plot_lomb_classifiers_by_trial_type(concantenated_dataframe, save_path):

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    lomb_classifications = pandas_collumn_to_2d_numpy_array(grid_cells["Lomb_classifier_"])
    lomb_classifications_b = pandas_collumn_to_2d_numpy_array(grid_cells["Lomb_classifier__all_beaconed"])
    lomb_classifications_nb = pandas_collumn_to_2d_numpy_array(grid_cells["Lomb_classifier__all_nonbeaconed"])
    lomb_classifications_p = pandas_collumn_to_2d_numpy_array(grid_cells["Lomb_classifier__all_probe"])
    lomb_classes = np.concatenate([lomb_classifications, lomb_classifications_b, lomb_classifications_nb, lomb_classifications_p], axis=1)

    lomb_classes_colors = lomb_classes.copy()
    for i in range(len(lomb_classes_colors)):
        for j in range(len(lomb_classes_colors[0])):
            lomb_class_str = lomb_classes[i, j]
            if lomb_class_str == "Position":
                lomb_classes_colors[i, j] = 0.5
            elif lomb_class_str == "Distance":
                lomb_classes_colors[i, j] = 1.5
            elif lomb_class_str == "Null":
                lomb_classes_colors[i, j] = 2.5
            else:
                lomb_classes_colors[i, j] = 3.5

    lomb_classes_colors = lomb_classes_colors.astype(np.float64)
    lomb_classes_colors = lomb_classes_colors[np.argsort(lomb_classes_colors[:, 0])]
    lomb_classes_colors = sort_lomb_class_colors(lomb_classes_colors)
    lomb_classes_colors = np.flip(lomb_classes_colors, axis=0)

    PDN = np.array([1,2,3,4])
    fig, ax = plt.subplots(figsize=(1.5,8))
    ordered = np.arange(1, len(grid_cells)+1, 1)
    X, Y = np.meshgrid(PDN, ordered)
    cmap = colors.ListedColormap(['turquoise', 'orange', 'gray', 'white'])
    boundaries = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.pcolormesh(X, Y, lomb_classes_colors, norm=norm, cmap=cmap, edgecolors="k", linewidth=1, shading="nearest")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticks(PDN)
    ax.set_xticklabels(["All", "B", "NB", "P"])
    #ax.set_xlabel("Frequency", fontsize=10)
    #plt.tight_layout()
    plt.xticks(fontsize=4)
    plt.subplots_adjust(left=0.4)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(save_path + '/lomb_classes_by_trial_type.png', dpi=200)
    plt.close()

    return

def plot_proportion_significant_to_trial_outcome(concatenated_dataframe, save_path):
    grid_cells = concatenated_dataframe[concatenated_dataframe["classifier"] == "G"]
    non_grid_cells = concatenated_dataframe[concatenated_dataframe["classifier"] != "G"]

    for group in ["Position", "Distance", "Null"]:
        grids = grid_cells[grid_cells["Lomb_classifier_"] == group]
        sig_hm = pandas_collumn_to_numpy_array(grids["power_test_hit_miss_p"])
        sig_ht = pandas_collumn_to_numpy_array(grids["power_test_hit_try_p"])
        sig_tm = pandas_collumn_to_numpy_array(grids["power_test_try_miss_p"])

        sig_hm = sig_hm[~np.isnan(sig_hm)]
        sig_ht = sig_ht[~np.isnan(sig_ht)]
        sig_tm = sig_tm[~np.isnan(sig_tm)]

        sig_hm[sig_hm<=0.05] = 0
        sig_hm[sig_hm>0.05] = 1
        sig_ht[sig_ht<=0.05] = 0
        sig_ht[sig_ht>0.05] = 1
        sig_tm[sig_tm<=0.05] = 0
        sig_tm[sig_tm>0.05] = 1

        labels = ["Not Modulated", "Modulated"]
        sizes = [len(sig_hm[sig_hm==1]), len(sig_hm[sig_hm==0])]
        explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig(save_path + '/proportion_significant_modulated_by_trial_outcome_'+group+'.png', dpi=200)
        plt.close()
    return

def plot_lomb_classifiers_by_trial_outcome(concantenated_dataframe, save_path):

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    lomb_classifications = pandas_collumn_to_2d_numpy_array(grid_cells["Lomb_classifier_"])
    lomb_classifications_hits = pandas_collumn_to_2d_numpy_array(grid_cells["Lomb_classifier__all_hits"])
    lomb_classifications_tries = pandas_collumn_to_2d_numpy_array(grid_cells["Lomb_classifier__all_tries"])
    lomb_classifications_misses = pandas_collumn_to_2d_numpy_array(grid_cells["Lomb_classifier__all_misses"])
    lomb_classes = np.concatenate([lomb_classifications, lomb_classifications_hits, lomb_classifications_tries, lomb_classifications_misses], axis=1)

    lomb_classes_colors = lomb_classes.copy()
    for i in range(len(lomb_classes_colors)):
        for j in range(len(lomb_classes_colors[0])):
            lomb_class_str = lomb_classes[i, j]
            if lomb_class_str == "Position":
                lomb_classes_colors[i, j] = 0.5
            elif lomb_class_str == "Distance":
                lomb_classes_colors[i, j] = 1.5
            elif lomb_class_str == "Null":
                lomb_classes_colors[i, j] = 2.5
            else:
                lomb_classes_colors[i, j] = 3.5

    lomb_classes_colors = lomb_classes_colors.astype(np.float64)
    lomb_classes_colors = lomb_classes_colors[np.argsort(lomb_classes_colors[:, 0])]
    lomb_classes_colors = sort_lomb_class_colors(lomb_classes_colors)
    lomb_classes_colors = np.flip(lomb_classes_colors, axis=0)

    PDN = np.array([1,2,3,4])
    fig, ax = plt.subplots(figsize=(1.5,8))
    ordered = np.arange(1, len(grid_cells)+1, 1)
    X, Y = np.meshgrid(PDN, ordered)
    cmap = colors.ListedColormap(['turquoise', 'orange', 'gray', 'white'])
    boundaries = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.pcolormesh(X, Y, lomb_classes_colors, norm=norm, cmap=cmap, edgecolors="k", linewidth=1, shading="nearest")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticks(PDN)
    ax.set_xticklabels(["All", "H", "T", "M"])
    #ax.set_xlabel("Frequency", fontsize=10)
    #plt.tight_layout()
    plt.xticks(fontsize=4)
    plt.subplots_adjust(left=0.4)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(save_path + '/lomb_classes_by_trial_outcome.png', dpi=200)
    plt.close()

    return

def plot_firing_rate_maps(spike_data, processed_position_data, raw_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data = processed_position_data[(processed_position_data["trial_type"] == 1) | (processed_position_data["trial_type"] == 2)]

    spike_data = add_position_x(spike_data, raw_position_data)
    spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        firing_trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])
        firing_locations_cluster = np.array(cluster_spike_data["x_position_cm"].iloc[0])

        if len(firing_times_cluster)>1:
            fr_binned_in_space = np.array(cluster_spike_data["fr_binned_in_space"].iloc[0])
            fr_binned_in_space_bin_centres = np.array(cluster_spike_data['fr_binned_in_space_bin_centres'].iloc[0])[0]

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            ax.axvspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
            ax.axvspan(track_length-30, track_length, facecolor='k', linewidth =0, alpha=.25)# black box
            ax.axvline(x=track_length-60-30-20, color="black", linestyle="dotted", linewidth=1)
            ax.axvline(x=track_length-60-30, color="black", linestyle="dotted", linewidth=1)
            y_max=0

            for hmt, c in zip(["hit", "try", "miss"], ["green", "orange", "red"]):
                hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
                hmt_trial_numbers = np.asarray(hmt_processed_position_data["trial_number"])
                hmt_fr_binned_in_space = fr_binned_in_space[hmt_trial_numbers-1]
                ax.fill_between(fr_binned_in_space_bin_centres, np.nanmean(hmt_fr_binned_in_space, axis=0)-stats.sem(hmt_fr_binned_in_space, axis=0), np.nanmean(hmt_fr_binned_in_space, axis=0)+stats.sem(hmt_fr_binned_in_space, axis=0), color=c, alpha=0.3)
                ax.plot(fr_binned_in_space_bin_centres, np.nanmean(hmt_fr_binned_in_space, axis=0), color=c)

                hmt_max = max(np.nanmean(hmt_fr_binned_in_space, axis=0)+stats.sem(hmt_fr_binned_in_space, axis=0))
                y_max = max([y_max, hmt_max])
                y_max = np.ceil(y_max)

            plt.ylabel('Firing Rate', fontsize=20, labelpad = 20)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
            plt.xlim(0, track_length)
            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            Edmond.plot_utility2.style_vr_plot(ax, x_max=y_max)
            ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_by_trial_outcome_' + str(cluster_id) + '.png', dpi=300)
            plt.close()

    return


def plot_firing_rate_maps_per_trial_by_hmt(spike_data, processed_position_data, output_path, track_length, trial_types):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data_ = pd.DataFrame()
    for tt in trial_types:
        processed_position_data_ = pd.concat([processed_position_data_, processed_position_data[processed_position_data["trial_type"] == tt]], ignore_index=True)
    processed_position_data = processed_position_data_
    string_tts = [str(int) for int in trial_types]
    string_tts = "-".join(string_tts)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data["firing_rate_maps"].iloc[cluster_index])
            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0

            if len(cluster_firing_maps) == 0:
                print("stop here")

            cluster_firing_maps = min_max_normalize(cluster_firing_maps)


            tmp = []
            for i in range(len(cluster_firing_maps[0])):
                for j in range(int(settings.vr_grid_analysis_bin_size)):
                    tmp.append(cluster_firing_maps[:, i].tolist())
            cluster_firing_maps = np.array(tmp).T

            for hmt in ["hit", "miss", "try", "all"]:
                if hmt != "all":
                    hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
                elif hmt == "all":
                    hmt_processed_position_data = pd.DataFrame()
                    hmt_processed_position_data = pd.concat([hmt_processed_position_data, processed_position_data[processed_position_data["hit_miss_try"] == "hit"]], ignore_index=True)
                    hmt_processed_position_data = pd.concat([hmt_processed_position_data, processed_position_data[processed_position_data["hit_miss_try"] == "try"]], ignore_index=True)
                    hmt_processed_position_data = pd.concat([hmt_processed_position_data, processed_position_data[processed_position_data["hit_miss_try"] == "miss"]], ignore_index=True)

                if len(hmt_processed_position_data)>1:
                    hmt_trial_numbers = pandas_collumn_to_numpy_array(hmt_processed_position_data["trial_number"])
                    hmt_cluster_firing_maps = cluster_firing_maps[hmt_trial_numbers-1]

                    x_max = len(hmt_processed_position_data)+0.5
                    spikes_on_track = plt.figure()
                    spikes_on_track.set_size_inches(5, 5, forward=True)
                    ax = spikes_on_track.add_subplot(1, 1, 1)

                    locations = np.arange(0, len(hmt_cluster_firing_maps[0]))
                    ordered = np.arange(0, len(hmt_processed_position_data), 1)
                    X, Y = np.meshgrid(locations, ordered)
                    hmt_cluster_firing_maps = np.flip(hmt_cluster_firing_maps, axis=0)

                    cmap = plt.cm.get_cmap("jet")
                    ax.pcolormesh(X, Y, hmt_cluster_firing_maps, cmap=cmap, shading="flat")

                    #c = ax.imshow(hmt_cluster_firing_maps, interpolation='none', cmap=cmap, vmin=0, vmax=np.max(cluster_firing_maps), origin='lower', aspect="auto")

                    group_legend_array = np.arange(0, 5+0.01, 0.01)+track_length+5
                    ordered = np.arange(0, len(hmt_processed_position_data), 1)
                    color_legend_tmp = np.concatenate([np.ones((len(hmt_processed_position_data[hmt_processed_position_data["hit_miss_try"] == "hit"]), len(group_legend_array)))*0,
                                                       np.ones((len(hmt_processed_position_data[hmt_processed_position_data["hit_miss_try"] == "try"]), len(group_legend_array)))*1.5,
                                                       np.ones((len(hmt_processed_position_data[hmt_processed_position_data["hit_miss_try"] == "miss"]), len(group_legend_array)))*2.5])
                    color_legend_tmp = np.flip(color_legend_tmp, axis=0)

                    plt.ylabel('Trial', fontsize=20, labelpad = 20)
                    plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
                    plt.xlim(0, track_length+10)
                    tick_spacing = 100
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                    ax.yaxis.set_ticks_position('left')
                    ax.xaxis.set_ticks_position('bottom')
                    Edmond.plot_utility2.style_vr_plot(ax, x_max-2)
                    plt.locator_params(axis = 'y', nbins  = 4)
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    #cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
                    #cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
                    #cbar.set_ticks([0,np.max(cluster_firing_maps)])
                    #cbar.set_ticklabels(["0", "Max"])
                    #cbar.ax.tick_params(labelsize=20)

                    X, Y = np.meshgrid(group_legend_array, ordered)
                    cmap = colors.ListedColormap(['green', 'orange', 'red'])
                    boundaries = [0, 1, 2, 3]
                    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
                    ax.pcolormesh(X, Y, color_legend_tmp, norm=norm, cmap=cmap, shading="flat")

                    plt.tight_layout()
                    plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + 'hmt_' + hmt + '_tt_'+string_tts+'.png', dpi=300)
                    plt.close()
    return


def plot_firing_rate_maps_per_trial(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
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

            x_max = len(processed_position_data)
            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            c = ax.imshow(cluster_firing_maps, interpolation='none', cmap=cmap, vmin=0, vmax=np.max(cluster_firing_maps), origin='lower', aspect="auto")

            plt.ylabel('Trial Number', fontsize=20, labelpad = 20)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
            plt.xlim(0, track_length)
            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            #Edmond.plot_utility2.style_track_plot(ax, track_length)
            Edmond.plot_utility2.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
            cbar.set_ticks([0,np.max(cluster_firing_maps)])
            cbar.set_ticklabels(["0", "Max"])
            cbar.ax.tick_params(labelsize=20)

            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '.png', dpi=300)
            plt.close()

def plot_mean_firing_rates_hmt(concantenated_dataframe, save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    for group in ["Position", "Distance", "Null"]:
        grids = grid_cells[grid_cells["Lomb_classifier_"] == group]

        for tt in ["all", 0, 1, 2]:
            hits = pandas_collumn_to_numpy_array(grids['mean_fr_tt_'+str(tt)+'_hmt_hit'])
            tries = pandas_collumn_to_numpy_array(grids['mean_fr_tt_'+str(tt)+'_hmt_try'])
            misses = pandas_collumn_to_numpy_array(grids['mean_fr_tt_'+str(tt)+'_hmt_miss'])

            fig, ax = plt.subplots(figsize=(4,4))
            ax.set_ylabel("Mean Firing Rate (Hz)", fontsize=20, labelpad=10)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=20)
            objects = ["Hit", "Try", "Miss"]
            x_pos = np.arange(len(objects))
            for i in range(len(hits)):
                ax.plot(x_pos, [hits[i], tries[i], misses[i]], color="black", alpha=0.1)

            ax.errorbar(x_pos[0], np.nanmean(hits), yerr=stats.sem(hits, nan_policy='omit'), ecolor='green', capsize=20, fmt="o", color="green")
            ax.scatter(x_pos[0]*np.ones(len(hits)), np.asarray(hits), edgecolor="green", marker="o", facecolors='none')

            ax.errorbar(x_pos[1], np.nanmean(tries), yerr=stats.sem(tries, nan_policy='omit'), ecolor='orange', capsize=20, fmt="o", color="orange")
            ax.scatter(x_pos[1]*np.ones(len(tries)), np.asarray(tries), edgecolor="orange", marker="o", facecolors='none')

            ax.errorbar(x_pos[2], np.nanmean(misses), yerr=stats.sem(misses, nan_policy='omit'), ecolor='red', capsize=20, fmt="o", color="red")
            ax.scatter(x_pos[2]*np.ones(len(misses)), np.asarray(misses), edgecolor="red", marker="o", facecolors='none')

            bad_hm = ~np.logical_or(np.isnan(hits), np.isnan(misses))
            bad_ht = ~np.logical_or(np.isnan(hits), np.isnan(tries))
            bad_tm = ~np.logical_or(np.isnan(tries), np.isnan(misses))
            hit_miss_p = stats.wilcoxon(np.compress(bad_hm, hits), np.compress(bad_hm, misses))[1]
            hit_try_p = stats.wilcoxon(np.compress(bad_ht, hits), np.compress(bad_ht, tries))[1]
            try_miss_p = stats.wilcoxon(np.compress(bad_tm, tries), np.compress(bad_tm, misses))[1]

            all_behaviour = []; all_behaviour.extend(hits.tolist()); all_behaviour.extend(misses.tolist()); all_behaviour.extend(tries.tolist())
            significance_bar(start=x_pos[0], end=x_pos[1], height=np.nanmax(all_behaviour)+0, displaystring=get_p_text(hit_try_p))
            significance_bar(start=x_pos[1], end=x_pos[2], height=np.nanmax(all_behaviour)+0.1, displaystring=get_p_text(try_miss_p))
            significance_bar(start=x_pos[0], end=x_pos[2], height=np.nanmax(all_behaviour)+0.2, displaystring=get_p_text(hit_miss_p))

            plt.xticks(x_pos, objects, fontsize=30)
            plt.locator_params(axis='y', nbins=3)
            plt.xlim((-0.5, len(objects)-0.5))
            #plt.xticks(rotation=-45)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(save_path + '/mean_firing_rate_hmt_'+group+'_tt'+str(tt)+'.png', dpi=200)
            plt.close()

    return

def add_mean_firing_rate_hmt(spike_data, processed_position_data, position_data, track_length):
    position_data_trial_numbers = np.asarray(position_data["trial_number"])
    position_data_time_in_bin_sec = np.asarray(position_data["time_in_bin_seconds"])

    new = pd.DataFrame()
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.asarray(spike_data.firing_times.iloc[cluster_index])
        firing_trial_numbers = np.asarray(spike_data.trial_number.iloc[cluster_index])

        for tt in ["all", 0, 1, 2]:
            for hmt in ["all", "hit", "miss", "try"]:

                if len(firing_times_cluster)>1:
                    subset_processed_position_data = processed_position_data.copy()
                    if (tt == 0) or (tt == 1) or (tt == 2):
                        subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["trial_type"] == tt)]
                    if (hmt == "hit") or (hmt == "try") or (hmt == "try"):
                        subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["hit_miss_try"] == hmt)]

                    hmt_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])
                    time_seconds = np.sum(position_data_time_in_bin_sec[np.isin(position_data_trial_numbers, hmt_trial_numbers)])
                    n_spikes = len(firing_trial_numbers[np.isin(firing_trial_numbers, hmt_trial_numbers)])
                    mean_firing_rate = n_spikes/time_seconds

                    cluster_spike_data["mean_fr_tt_"+str(tt)+"_hmt_"+hmt] = mean_firing_rate
                else:
                    cluster_spike_data["mean_fr_tt_"+str(tt)+"_hmt_"+hmt] = 0

        new = pd.concat([new, cluster_spike_data], ignore_index=True)
    return new

def get_hits_preceded_by_misses_trial_numbers(processed_position_data, tt):
    trial_numbers = []
    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_number = trial_row["trial_number"].iloc[0]
        hmt = trial_row["hit_miss_try"].iloc[0]

        if (trial_type == tt) and (trial_number!=1) and (hmt=="hit"): # dont look at the first trial
            previous_trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number-1]
            previous_trial_hmt = previous_trial_processed_position_data["hit_miss_try"].iloc[0]

            if previous_trial_hmt == "miss":
                trial_numbers.append(trial_number)

    return np.array(trial_numbers)

def analyse_miss_hit_transitions(spike_data, processed_position_data, position_data, track_length):
    cluster_tt_tmps =[]
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.asarray(spike_data.firing_times.iloc[cluster_index])
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        centre_trials_rounded = np.round(np.array(centre_trials)).astype(np.int64)

        tt_tmp = []
        for tt in [0, 1, 2]:
            tmp = []
            hit_trial_numbers = get_hits_preceded_by_misses_trial_numbers(processed_position_data, tt)

            if (len(firing_times_cluster)>1) and (len(hit_trial_numbers)>0):
                for t_i in range(len(hit_trial_numbers)):
                    subset_powers = powers.copy()
                    two_trials_powers = np.concatenate((subset_powers[centre_trials_rounded == hit_trial_numbers[t_i]-1],
                                                        subset_powers[centre_trials_rounded == hit_trial_numbers[t_i]]))

                    tmp.append(np.max(two_trials_powers, axis=1).tolist())
            else:
                tmp.append([])
            tt_tmp.append(tmp)
        cluster_tt_tmps.append(tt_tmp)

    spike_data["miss_hit_transition_tt_012"] = cluster_tt_tmps
    return spike_data

def plot_trial_type_hmt_difference(concantenated_dataframe, save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    for group in ["Position", "Distance", "Null", "all"]:
        if group == "all":
            grids = grid_cells
        else:
            grids = grid_cells[grid_cells["Lomb_classifier_"] == group]

        hits_beaconed = pandas_collumn_to_numpy_array(grids['ML_SNRs_beaconed_hits'])
        hits_non_beaconed = pandas_collumn_to_numpy_array(grids['ML_SNRs_nonbeaconed_hits'])
        hits_probe = pandas_collumn_to_numpy_array(grids['ML_SNRs_probe_hits'])

        misses_beaconed = pandas_collumn_to_numpy_array(grids['ML_SNRs_beaconed_misses'])
        misses_non_beaconed = pandas_collumn_to_numpy_array(grids['ML_SNRs_nonbeaconed_misses'])
        misses_probe = pandas_collumn_to_numpy_array(grids['ML_SNRs_probe_misses'])

        diff_beaconed = hits_beaconed - misses_beaconed
        diff_non_beaconed = hits_non_beaconed - misses_non_beaconed
        diff_probe = hits_probe - misses_probe

        data = [diff_beaconed[~np.isnan(diff_beaconed)], diff_non_beaconed[~np.isnan(diff_non_beaconed)], diff_probe[~np.isnan(diff_probe)]]
        fig, ax = plt.subplots(figsize=(6,6))
        parts = ax.violinplot(data, positions=[0, 1, 2], showmeans=False, showmedians=False,showextrema=False)

        x1 = 0 * np.ones(len(diff_beaconed[~np.isnan(diff_beaconed)]))
        x2 = 1 * np.ones(len(diff_non_beaconed[~np.isnan(diff_non_beaconed)]))
        x3 = 2 * np.ones(len(diff_probe[~np.isnan(diff_probe)]))

        y1 = diff_beaconed[~np.isnan(diff_beaconed)]
        y2 = diff_non_beaconed[~np.isnan(diff_non_beaconed)]
        y3 = diff_probe[~np.isnan(diff_probe)]

        #Combine the sampled data together
        x = np.concatenate((x1, x2, x3), axis=0)
        y = np.concatenate((y1, y2, y3), axis=0)
        sns.swarmplot(x, y, ax=ax, color="black")

        hm_colors=["gray", "red", "blue"]
        for pc, hm_color in zip(parts['bodies'], hm_colors):
            pc.set_facecolor(hm_color)
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=-0.2, top=0.2)

        p_value_b = stats.ttest_1samp(y1, 0.0)[1]
        p_value_nb = stats.ttest_1samp(y2, 0.0)[1]
        p_value_p = stats.ttest_1samp(y3, 0.0)[1]

        p_value_b_str = get_p_text(p_value_b, ns=True)
        p_value_nb_str = get_p_text(p_value_nb, ns=True)
        p_value_p_str = get_p_text(p_value_p, ns=True)

        ax.text(x=0, y=0.2, s=p_value_b_str, ha="center", va="center", fontsize=20)
        ax.text(x=1, y=0.2, s=p_value_nb_str, ha="center", va="center", fontsize=20)
        ax.text(x=2, y=0.2, s=p_value_p_str, ha="center", va="center", fontsize=20)

        ax.axhline(y=0, linestyle="dashed", color="black", linewidth=2)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([-0.2, 0, 0.2])
        ax.set_xticklabels(["B", "NB", "P"])
        fig.tight_layout()

        plt.subplots_adjust(left=0.25, bottom=0.2)
        ax.set_ylabel('Hit/Miss '+ r'$\Delta$Power', fontsize=25)
        ax.set_ylabel('Hit - Miss Power', fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.savefig(save_path + '/trial_outcome_difference_by_trialtype_'+group+'.png', dpi=200)
        plt.close()
    return

def plot_trial_type_hmt_difference_hist(concantenated_dataframe, save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    for group, group_str in zip([grid_cells, non_grid_cells], ["G", "NG"]):
        p = group[group["Lomb_classifier_"] == "Position"]
        d = group[group["Lomb_classifier_"] == "Distance"]
        a = group[(group["Lomb_classifier_"] == "Position") | (group["Lomb_classifier_"] == "Distance")]

        for tt in ["beaconed", "non_beaconed", "probe"]:
            p_hit_miss_diff = np.asarray(p[hmt2collumn(hmt="hit", tt=tt)] - p[hmt2collumn(hmt="miss", tt=tt)])
            d_hit_miss_diff = np.asarray(d[hmt2collumn(hmt="hit", tt=tt)] - d[hmt2collumn(hmt="miss", tt=tt)])
            a_hit_miss_diff = np.asarray(a[hmt2collumn(hmt="hit", tt=tt)] - a[hmt2collumn(hmt="miss", tt=tt)])

            fig, ax = plt.subplots(figsize=(6,4))

            p_value_position = stats.ttest_1samp(p_hit_miss_diff[~np.isnan(p_hit_miss_diff)], 0.0)[1]
            p_value_distance = stats.ttest_1samp(d_hit_miss_diff[~np.isnan(d_hit_miss_diff)], 0.0)[1]
            p_value_all_cells = stats.ttest_1samp(a_hit_miss_diff[~np.isnan(a_hit_miss_diff)], 0.0)[1]

            p_value_position_str = get_p_text(p_value_position, ns=True)
            p_value_distance_str = get_p_text(p_value_distance, ns=True)
            p_value_all_cells_str = get_p_text(p_value_all_cells, ns=True)

            ax.hist([p_hit_miss_diff, d_hit_miss_diff], bins=20, range=(-0.2, 0.2), density=False, histtype='bar', stacked=True, color =["turquoise", "orange"], edgecolor='black', linewidth=2)
            ax.text(x=0.1, y=ax.get_ylim()[1], s=p_value_all_cells_str, ha="center", va="center", fontsize=20, color="black")
            ax.axvline(x=0, linestyle="solid", color="black", linewidth=3)
            ax.axvline(x=np.nanmedian(a_hit_miss_diff), linestyle="dashed", color="red", linewidth=2)
            #ax.set_xticks([0, 1, 2])
            #ax.set_yticks([-0.2, 0, 0.2])
            #ax.set_xticklabels(["B", "NB", "P"])
            fig.tight_layout()
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.locator_params(axis='y', nbins=4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(bottom=0)
            plt.subplots_adjust(left=0.25, bottom=0.2)
            ax.set_ylabel("Number of Cells", fontsize=25)
            ax.set_xlabel(r'$\Delta$ Periodic Power '+'\n'+'(Hit - Miss)', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=25)
            plt.savefig(save_path + '/trial_outcome_difference_by_trialtype_hist_'+group_str+'_tt'+tt+'.png', dpi=200)
            plt.close()
    return

def plot_hit_miss_transitions(concantenated_dataframe, save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    for group in ["Position", "Distance", "Null", "all"]:
        if group == "all":
            grids = grid_cells
        else:
            grids = grid_cells[grid_cells["Lomb_classifier_"] == group]

        fig, ax = plt.subplots(figsize=(6,6))

        for tt_i, tt_c in zip([0, 1, 2], ["black", "red", "blue"]): # trial type indices
            tt_means = []
            for i in range(len(grids)):
                transitions = grids['miss_hit_transition_tt_012'].iloc[i][tt_i]
                if len(transitions)>0:
                    if len(transitions[0])>0:
                        for j in range(len(transitions)):
                            length_of_transitions = len(transitions[j])
                            if length_of_transitions != 40:
                                for m in range(40-length_of_transitions):
                                    transitions[j].append(np.nan)
                        tt_means.append(np.nanmean(np.array(transitions), axis=0).tolist())

            tt_means = np.array(tt_means)
            ax.errorbar(x=np.arange(0,len(tt_means[0])), y=np.nanmean(tt_means, axis=0), yerr=stats.sem(tt_means, axis=0, nan_policy="omit"), color=tt_c)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axvline(x=20, linestyle="dashed", color="black", linewidth=2)
        ax.set_xticks([])
        fig.tight_layout()
        plt.subplots_adjust(left=0.25, bottom=0.2)
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.savefig(save_path + '/miss_hit_transition_by_trial_types_'+group+'.png', dpi=200)
        plt.close()
    return


def process_recordings(vr_recording_path_list, of_recording_path_list):

    shuffle_df = pd.DataFrame()
    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            session_id = recording.split("/")[-1]
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            position_data = add_time_elapsed_collumn(position_data)
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            shuffle_data = pd.read_pickle(recording+"/MountainSort/DataFrames/lomb_shuffle_powers.pkl")

            if len(spike_data)>0:
                #raw_position_data, position_data = syncronise_position_data(recording, get_track_length(recording))

                processed_position_data = add_avg_RZ_speed(processed_position_data, track_length=get_track_length(recording))
                processed_position_data = add_avg_track_speed(processed_position_data, track_length=get_track_length(recording))
                processed_position_data = add_RZ_bias(processed_position_data)
                processed_position_data, percentile_speed = add_hit_miss_try3(processed_position_data, track_length=get_track_length(recording))

                #plot_firing_rate_maps(spike_data=spike_data, processed_position_data=processed_position_data, raw_position_data=raw_position_data, output_path=output_path, track_length=get_track_length(recording))
                #spike_data = analyse_miss_hit_transitions(spike_data, processed_position_data, position_data,  track_length=get_track_length(recording))
                #spike_data = add_mean_firing_rate_hmt(spike_data, processed_position_data, position_data, track_length=get_track_length(recording))
                #plot_firing_rate_maps_per_trial_by_hmt(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=get_track_length(recording), trial_types=[1, 2])
                #plot_firing_rate_maps_per_trial(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=get_track_length(recording))
                #plot_speed_histogram_with_error(processed_position_data[processed_position_data["hit_miss_try"] == "hit"], output_path, track_length=get_track_length(recording), suffix="hit")
                #plot_speed_histogram_with_error(processed_position_data[processed_position_data["hit_miss_try"] == "miss"], output_path, track_length=get_track_length(recording), suffix="miss")
                #plot_speed_histogram_with_error(processed_position_data[processed_position_data["hit_miss_try"] == "try"], output_path, track_length=get_track_length(recording), suffix="try")
                #plot_speed_histogram_with_error(processed_position_data, output_path, track_length=get_track_length(recording), suffix="")
                #plot_stops_on_track(processed_position_data, output_path, track_length=get_track_length(recording))
                #plot_avg_speed_in_rz_hist(processed_position_data, output_path, percentile_speed=percentile_speed)
                plot_snr_by_hmt(spike_data, processed_position_data, output_path, track_length = get_track_length(recording))
                #plot_snr_by_hmt_tt(spike_data, processed_position_data, output_path, track_length = get_track_length(recording))
                #spike_data = plot_power_by_hmt(spike_data, processed_position_data, output_path, track_length = get_track_length(recording))
                #plot_snr_by_trial_type(spike_data, processed_position_data, output_path, track_length = get_track_length(recording))
                #plot_snr_vs_RZbias_regression(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))
                #plot_avg_lomb(spike_data, output_path)
                #plot_power_trajectories(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))
                #plot_trial_cross_correlations(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))
                #plot_trial_fr_cross_correlations(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))
                #plot_peak_histogram(spike_data, processed_position_data, output_path, track_length=get_track_length(recording))

                shuffle_data["session_id"] = session_id
                shuffle_df = pd.concat([shuffle_df, shuffle_data], ignore_index=True)
                #spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
                print("")

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)

    #shuffle_df.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8_lomb_shuffle.pkl")



def main():
    print('-------------------------------------------------------------')

    # give a path for a directory of recordings or path of a single recording
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()]
    #of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()]
    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M14_D18_2021-06-02_12-27-22', '/mnt/datastore/Harry/cohort8_may2021/vr/M10_D4_2021-05-13_09-20-38', '/mnt/datastore/Harry/cohort8_may2021/vr/M10_D5_2021-05-14_08-59-54', '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D11_2021-05-24_10-00-53',
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
                    '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D17_2021-06-01_11-45-20', '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D24_2021-06-10_12-01-54', '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D25_2021-06-11_12-03-07',
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
    process_recordings(vr_path_list, of_path_list)

    combined_shuffle_df = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8_lomb_shuffle.pkl")
    combined_df = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")
    combined_df = add_lomb_classifier(combined_df,suffix="")
    #read_df(combined_df)

    #plot_mean_firing_rates_hmt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/mean_firing_rate")
    plot_trial_type_hmt_difference(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/trial_type_differences")
    plot_trial_type_hmt_difference_hist(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/trial_type_differences")
    #plot_hit_miss_transitions(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/hit_miss_transitions")
    plot_proportion_significant_to_trial_outcome(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt")
    #plot_max_freq_histogram(combined_df, combined_shuffle_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")
    #plot_lomb_overview_ordered(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")
    #plot_spatial_info_vs_pearson(combined_df, output_path="/mnt/datastore/Harry/Vr_grid_cells/")
    #plot_lomb_classifiers(combined_df, suffix="", save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")
    #for suffix in ["", "_all_beaconed", "_all_nonbeaconed", "_all_probe", "_all_hits", "_all_tries", "_all_misses"]:
        #combined_df = add_lomb_classifier(combined_df,suffix=suffix)
        #plot_lomb_classifiers_proportions(combined_df, suffix=suffix, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")

    #plot_lomb_classifiers_by_trial_type(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")
    #plot_lomb_classifiers_by_trial_outcome(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")
    #plot_lomb_classifiers_proportions_by_mouse(combined_df, suffix="", save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")
    #plot_grid_scores_by_classifier(combined_df, suffix="", save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")
    #plot_of_stability_by_classifier(combined_df, suffix="", save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")
    #plot_of_stability_vs_grid_score_by_classifier(combined_df, suffix="", save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")
    print("look now")

if __name__ == '__main__':
    main()