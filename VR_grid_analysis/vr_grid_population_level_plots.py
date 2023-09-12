import numpy as np
import pandas as pd
import pickle
import shutil
from statsmodels.stats.multitest import fdrcorrection as fdrcorrection
import Edmond.VR_grid_analysis.analysis_settings as Settings
from matplotlib.markers import TICKDOWN
import PostSorting.parameters
from astropy.nddata import block_reduce
from scipy.signal import correlate
from matplotlib import colors
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from scipy import interpolate
from scipy import stats
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from Edmond.VR_grid_analysis.hit_miss_try_firing_analysis import hmt2collumn
import matplotlib.patches as patches
import matplotlib.colors as colors
from sklearn.linear_model import LinearRegression
from PostSorting.vr_spatial_firing import bin_fr_in_space, bin_fr_in_time, add_position_x
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
import seaborn as sns
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
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

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

    plt.ylabel('Stops on trials', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
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
    plt.savefig(output_path + '/Figures/behaviour/stop_raster.png', dpi=200)
    plt.close()

def plot_speed_per_trial(processed_position_data, output_path, track_length=200):
    print('plotting speed heatmap...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    x_max = len(processed_position_data)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    trial_speeds = Edmond.plot_utility2.pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned_in_space"])
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
    plt.savefig(output_path + '/Figures/behaviour/speed_heat_map' + '.png', dpi=200)
    plt.close()

def plot_avg_speed_in_rz_hist(processed_position_data, output_path, percentile_speed):
    print('I am plotting avg speed histogram...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    g = colors.colorConverter.to_rgb("green")
    r = colors.colorConverter.to_rgb("red")
    o = colors.colorConverter.to_rgb("orange")

    fig, axes = plt.subplots(2, 1, figsize=(6,4), sharex=True)

    hits = processed_position_data[processed_position_data["hit_miss_try"] == "hit"]
    misses = processed_position_data[processed_position_data["hit_miss_try"] == "miss"]
    tries = processed_position_data[processed_position_data["hit_miss_try"] == "try"]

    axes[0].hist(pandas_collumn_to_numpy_array(hits["avg_speed_in_rz"]), range=(0, 100), bins=25, alpha=0.3, facecolor=(g[0],g[1],g[2], 0.3), edgecolor=(g[0],g[1],g[2], 1), histtype="bar", density=False, cumulative=False, linewidth=1)
    axes[1].hist(pandas_collumn_to_numpy_array(tries["avg_speed_in_rz"]), range=(0, 100), bins=25, alpha=0.3, facecolor=(o[0],o[1],o[2], 0.3), edgecolor=(o[0],o[1],o[2], 1), histtype="bar", density=False, cumulative=False, linewidth=1)
    axes[1].hist(pandas_collumn_to_numpy_array(misses["avg_speed_in_rz"]), range=(0, 100), bins=25, alpha=0.3, facecolor=(r[0],r[1],r[2], 0.3), edgecolor=(r[0],r[1],r[2], 1), histtype="bar", density=False, cumulative=False, linewidth=1)

    #plt.ylabel('Trial', fontsize=20, labelpad = 10)
    plt.xlabel('Avg Speed in RZ (cm/s)', fontsize=20, labelpad = 10)
    plt.xlim(0,100)
    tick_spacing = 50
    axes[0].axvline(x=percentile_speed, color="red", linestyle="dotted", linewidth=4)
    axes[1].axvline(x=percentile_speed, color="red", linestyle="dotted", linewidth=4)
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    axes[0].set_yticks([0, 15, 30])
    axes[1].set_yticks([0, 15, 30])
    fig.text(0.2, 0.5, "       Trials", va='center', rotation='vertical', fontsize=20)
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

def correct_for_time_binned_teleport(trial_pos_in_time, track_length):
    # check if any of the first 5 or last 5 bins are too high or too low respectively
    first_5 = trial_pos_in_time[:5]
    last_5 = trial_pos_in_time[-5:]

    first_5[first_5>(track_length/2)] = first_5[first_5>(track_length/2)]-track_length
    last_5[last_5<(track_length/2)] = last_5[last_5<(track_length/2)]+track_length

    trial_pos_in_time[:5] = first_5
    trial_pos_in_time[-5:] = last_5
    return trial_pos_in_time

def plot_speed_histogram_with_error(processed_position_data, output_path, track_length=200, tt="", hmt=""):
    subset_processed_position_data = processed_position_data[(processed_position_data["trial_type"] == tt) &
                                                             (processed_position_data["hit_miss_try"] == hmt)]
    if len(subset_processed_position_data)>0:
        trial_speeds = pandas_collumn_to_2d_numpy_array(subset_processed_position_data["speeds_binned_in_space"])
        bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])
        trial_speeds_sem = scipy.stats.sem(trial_speeds, axis=0, nan_policy="omit")
        trial_speeds_avg = np.nanmean(trial_speeds, axis=0)

        print('plotting avg speeds')
        save_path = output_path + '/Figures/behaviour'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        speed_histogram = plt.figure(figsize=(6,4))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        # to plot by trial using the time binned data we need the n-1, n and n+1 trials so we can plot around the track limits
        # here we extract the n-1, n and n+1 trials, correct for any time binned teleports and concatenated the positions and speeds for each trial
        for i, tn in enumerate(processed_position_data["trial_number"]):
            trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == tn]
            tt_trial = trial_processed_position_data["trial_type"].iloc[0]
            hmt_trial = trial_processed_position_data["hit_miss_try"].iloc[0]
            trial_speeds_in_time = np.asarray(trial_processed_position_data['speeds_binned_in_time'].iloc[0])
            trial_pos_in_time = np.asarray(trial_processed_position_data['pos_binned_in_time'].iloc[0])

            # cases above trial number 1
            if tn != min(processed_position_data["trial_number"]):
                trial_processed_position_data_1down = processed_position_data[processed_position_data["trial_number"] == tn-1]
                trial_speeds_in_time_1down = np.asarray(trial_processed_position_data_1down['speeds_binned_in_time'].iloc[0])
                trial_pos_in_time_1down = np.asarray(trial_processed_position_data_1down['pos_binned_in_time'].iloc[0])
            else:
                trial_speeds_in_time_1down = np.array([])
                trial_pos_in_time_1down = np.array([])

            # cases below trial number n
            if tn != max(processed_position_data["trial_number"]):
                trial_processed_position_data_1up = processed_position_data[processed_position_data["trial_number"] == tn+1]
                trial_speeds_in_time_1up = np.asarray(trial_processed_position_data_1up['speeds_binned_in_time'].iloc[0])
                trial_pos_in_time_1up = np.asarray(trial_processed_position_data_1up['pos_binned_in_time'].iloc[0])
            else:
                trial_speeds_in_time_1up = np.array([])
                trial_pos_in_time_1up = np.array([])

            trial_pos_in_time = np.concatenate((trial_pos_in_time_1down[-2:], trial_pos_in_time, trial_pos_in_time_1up[:2]))
            trial_speeds_in_time = np.concatenate((trial_speeds_in_time_1down[-2:], trial_speeds_in_time, trial_speeds_in_time_1up[:2]))

            if tt_trial == tt and hmt_trial == hmt:
                trial_pos_in_time = correct_for_time_binned_teleport(trial_pos_in_time, track_length)
                ax.plot(trial_pos_in_time, trial_speeds_in_time, color="grey", alpha=0.4)

        ax.plot(bin_centres, trial_speeds_avg, color=get_hmt_color(hmt), linewidth=4)
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
        x_max = max(trial_speeds_avg+trial_speeds_sem)
        x_max = 115
        Edmond.plot_utility2.style_vr_plot(ax, x_max)
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.savefig(output_path + '/Figures/behaviour/trial_speeds_tt_'+str(tt)+"_"+hmt+'.png', dpi=300)
        plt.close()


def plot_speed_histogram_with_error_all_trials(processed_position_data, output_path, track_length=200, hmt=""):
    subset_processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hmt)]
    if len(subset_processed_position_data)>0:
        trial_speeds = pandas_collumn_to_2d_numpy_array(subset_processed_position_data["speeds_binned_in_space"])
        bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])
        trial_speeds_sem = scipy.stats.sem(trial_speeds, axis=0, nan_policy="omit")
        trial_speeds_avg = np.nanmean(trial_speeds, axis=0)

        print('plotting avg speeds')
        save_path = output_path + '/Figures/behaviour'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        speed_histogram = plt.figure(figsize=(6,4))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        # to plot by trial using the time binned data we need the n-1, n and n+1 trials so we can plot around the track limits
        # here we extract the n-1, n and n+1 trials, correct for any time binned teleports and concatenated the positions and speeds for each trial
        for i, tn in enumerate(processed_position_data["trial_number"]):
            trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == tn]
            hmt_trial = trial_processed_position_data["hit_miss_try"].iloc[0]
            trial_speeds_in_time = np.asarray(trial_processed_position_data['speeds_binned_in_time'].iloc[0])
            trial_pos_in_time = np.asarray(trial_processed_position_data['pos_binned_in_time'].iloc[0])

            # cases above trial number 1
            if tn != min(processed_position_data["trial_number"]):
                trial_processed_position_data_1down = processed_position_data[processed_position_data["trial_number"] == tn-1]
                trial_speeds_in_time_1down = np.asarray(trial_processed_position_data_1down['speeds_binned_in_time'].iloc[0])
                trial_pos_in_time_1down = np.asarray(trial_processed_position_data_1down['pos_binned_in_time'].iloc[0])
            else:
                trial_speeds_in_time_1down = np.array([])
                trial_pos_in_time_1down = np.array([])

            # cases below trial number n
            if tn != max(processed_position_data["trial_number"]):
                trial_processed_position_data_1up = processed_position_data[processed_position_data["trial_number"] == tn+1]
                trial_speeds_in_time_1up = np.asarray(trial_processed_position_data_1up['speeds_binned_in_time'].iloc[0])
                trial_pos_in_time_1up = np.asarray(trial_processed_position_data_1up['pos_binned_in_time'].iloc[0])
            else:
                trial_speeds_in_time_1up = np.array([])
                trial_pos_in_time_1up = np.array([])

            trial_pos_in_time = np.concatenate((trial_pos_in_time_1down[-2:], trial_pos_in_time, trial_pos_in_time_1up[:2]))
            trial_speeds_in_time = np.concatenate((trial_speeds_in_time_1down[-2:], trial_speeds_in_time, trial_speeds_in_time_1up[:2]))

            if hmt_trial == hmt:
                trial_pos_in_time = correct_for_time_binned_teleport(trial_pos_in_time, track_length)
                ax.plot(trial_pos_in_time, trial_speeds_in_time, color="grey", alpha=0.4)

        ax.plot(bin_centres, trial_speeds_avg, color=get_hmt_color(hmt), linewidth=4)
        ax.axhline(y=4.7, color="black", linestyle="dashed", linewidth=2)
        plt.ylabel('Speed (cm/s)', fontsize=25, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
        plt.xlim(0,track_length)
        ax.set_yticks([0, 50, 100])
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        style_track_plot(ax, track_length)
        tick_spacing = 100
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        x_max = 115
        Edmond.plot_utility2.style_vr_plot(ax, x_max)
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        plt.savefig(output_path + '/Figures/behaviour/trial_speeds_all_trial_types_'+hmt+'.png', dpi=300)
        plt.close()

def plot_max_freq_histogram(combined_df, combined_shuffle_df, save_path):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]
    gc_max_powers = pandas_collumn_to_numpy_array(grid_cells["ML_SNRs"])
    gc_max_freqs = pandas_collumn_to_numpy_array(grid_cells["ML_Freqs"])

    ng_max_powers = pandas_collumn_to_numpy_array(non_grid_cells["ML_SNRs"])
    ng_max_freqs = pandas_collumn_to_numpy_array(non_grid_cells["ML_Freqs"])

    shuffle_max_powers = pandas_collumn_to_2d_numpy_array(combined_shuffle_df['ML_SNRs'])
    shuffle_max_freqs = pandas_collumn_to_numpy_array(combined_shuffle_df['ML_Freqs'])

    threshold = np.nanpercentile(shuffle_max_powers, 95)

    fig, axes = plt.subplots(3, 1, figsize=(6,6), sharex=True)
    _, _, patches2 = axes[0].hist(shuffle_max_powers, bins=100, range=(0, 0.3), facecolor=(0.5,0.5,0.5, 0.5), edgecolor=(0.5,0.5,0.5, 1), density=False, linewidth=1)
    _, _, patches1 = axes[1].hist(ng_max_powers, bins=100, range=(0, 0.3), facecolor=(0,0,1, 0.5), edgecolor=(0,0,1, 1), density=False, linewidth=1)
    _, _, patches1 = axes[2].hist(gc_max_powers, bins=100, range=(0, 0.3), facecolor=(1,0,0, 0.5), edgecolor=(1,0,0, 1), density=False, linewidth=1)
    axes[0].axvline(x=threshold, linestyle="dashed", color="red", linewidth=2)
    axes[1].axvline(x=threshold, linestyle="dashed", color="red", linewidth=2)
    axes[2].axvline(x=threshold, linestyle="dashed", color="red", linewidth=2)
    offset=0.01
    axes[0].text(x=0.2, y=450, s="Shuffled Data", color="black", fontsize=18)
    axes[1].text(x=0.2, y=90, s="Real Data", color="black", fontsize=18)
    axes[0].text(x=threshold+offset, y=380, s="95th percentile value: \n"+str(np.round(threshold, decimals=3)), color="red", fontsize=12)
    axes[1].text(x=threshold+offset, y=90, s="(NG: "+str(int(np.round((len(non_grid_cells[non_grid_cells["ML_SNRs"]>threshold])/len(non_grid_cells))*100, decimals=0)))+"%)", color="blue", fontsize=12)
    axes[2].text(x=threshold+offset, y=9, s="(GC: "+str(int(np.round((len(grid_cells[grid_cells["ML_SNRs"]>threshold])/len(grid_cells))*100, decimals=0)))+"%)", color="red", fontsize=12)
    axes[1].set_ylabel("Neurons",  fontsize=20, labelpad=10)
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Periodic Power",  fontsize=20)
    axes[0].set_xlim([0,0.3])
    axes[1].set_xlim([0,0.3])
    axes[2].set_xlim([0,0.3])
    axes[2].set_xticks([0, 0.1, 0.2, 0.3])
    axes[0].set_ylim([0, 500])
    axes[2].set_ylim([0, 10])
    axes[1].set_ylim([0, 100])
    axes[0].set_yticks([0, 500])
    axes[1].set_yticks([0, 50, 100])
    axes[2].set_yticks([0, 5, 10])
    axes[0].tick_params(axis='both', which='major', labelsize=15)
    axes[1].tick_params(axis='both', which='major', labelsize=15)
    axes[2].tick_params(axis='both', which='major', labelsize=15)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_max_powers.png', dpi=300)
    plt.close()
    return


def plot_lomb_overview_ordered(concantenated_dataframe, save_path, cmap_string="plasma"):
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
    step = Settings.frequency_step
    frequency = Settings.frequency
    #for i in range(len(ng_n_powers)):
    #    if isinstance(ng_n_powers[i], list):
    #        ng_n_powers[i] = np.ones(len(frequency))*np.nan
    #ng_n_powers = np.stack(ng_n_powers)
    #for i in range(len(ng_n_max_freqs)):
    #    if isinstance(ng_n_max_freqs[i], list):
    #        ng_n_max_freqs[i] = np.nan

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
    frequency = frequency+color_legend_offset
    ordered = np.arange(0, len(grid_cells)+1, 1)
    X, Y = np.meshgrid(frequency, ordered)
    powers = np.concatenate([g_p_powers, g_d_powers, g_n_powers], axis=0)
    powers = np.flip(powers, axis=0)

    for i in range(len(powers)):
        powers[i, :] = min_max_normalize(powers[i, :])
        #powers[i, :] = scipy.stats.zscore(powers[i, :], ddof=0, nan_policy='omit')
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
    ax.set_xticklabels(["0","", " ", ""," ","5"])
    ax.set_xticks(np.arange(0,6)+color_legend_offset)
    ax.set_xlabel("      Track Frequency", fontsize=15)
    #plt.tight_layout()
    plt.xticks(fontsize=4)
    plt.subplots_adjust(left=0.4)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(save_path + '/grid_lomb_power_ordered.png', dpi=200)
    plt.close()

    '''
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
    '''
    return


def plot_lomb_classifiers_vs_shuffle(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers...')

    distance_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Distance"]
    position_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Position"]
    null_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Null"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,4), gridspec_kw={'width_ratios': [1, 0.3]})
    ax1.set_ylabel("Peak power vs \n false alarm rate",color="black",fontsize=25, labelpad=10)
    ax1.set_xlabel("Track frequency", color="black", fontsize=25, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax1.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
    ax2.set_xticks([0, 0.5])
    ax2.set_xticklabels(["0", "0.5"])
    ax2.set_yticks([])
    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax2.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    for f in range(1,6):
        ax1.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.scatter(x=null_cells["ML_Freqs"+suffix], y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax1.scatter(x=distance_cells["ML_Freqs"+suffix], y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax1.scatter(x=position_cells["ML_Freqs"+suffix], y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax1.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set_xlim([0,5.02])
    ax1.set_ylim([-0.1,0.4])
    ax2.set_xlim([-0.05,0.55])
    ax2.set_ylim([-0.1,0.4])
    ax2.set_xlabel(r'$\Delta$ from Int', color="black", fontsize=25, labelpad=10)
    ax2.scatter(x=distance_from_integer(null_cells["ML_Freqs"+suffix]), y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(distance_cells["ML_Freqs"+suffix]), y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(position_cells["ML_Freqs"+suffix]), y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax2.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_vs_shuffle_PDN_'+suffix+'.png', dpi=200)
    plt.close()



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,4), gridspec_kw={'width_ratios': [1, 0.3]})
    ax1.set_ylabel("Peak power vs \n false alarm rate",color="black",fontsize=25, labelpad=10)
    ax1.set_xlabel("Track frequency", color="black", fontsize=25, labelpad=10)
    #ax1.set_xticks(np.arange(0, 11, 1.0))
    ax1.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
    #ax2.set_xticks([0, 0.5])
    #ax2.set_xticklabels(["0", "0.5"])
    ax2.set_yticks([])
    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax2.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    for f in range(0,6):
        ax1.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.set_ylim([-0.1,0.4])
    ax1.set_xlim(left=0.02)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\Delta$ from Int', color="black", fontsize=25, labelpad=10)
    ax1.scatter(x=distance_from_integer(null_cells["ML_Freqs"+suffix]), y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax1.scatter(x=distance_from_integer(distance_cells["ML_Freqs"+suffix]), y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax1.scatter(x=distance_from_integer(position_cells["ML_Freqs"+suffix]), y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax1.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set_xlim([0,5.02])
    ax1.set_ylim([-0.1,0.4])
    #ax2.set_xlim([-0.05,0.55])
    ax2.set_ylim([-0.1,0.4])
    ax2.set_xscale('log')
    ax2.set_xlim(left=0.02)
    ax2.set_xlabel(r'$\Delta$ from Int', color="black", fontsize=25, labelpad=10)
    ax2.scatter(x=distance_from_integer(null_cells["ML_Freqs"+suffix]), y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(distance_cells["ML_Freqs"+suffix]), y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(position_cells["ML_Freqs"+suffix]), y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax2.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_vs_shuffle_PDN_'+suffix+'_2.png', dpi=200)
    plt.close()

    return


def plot_lomb_classifiers_vs_shuffle_grid_cells(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = concantenated_dataframe[concantenated_dataframe["grid_cell"] == 1]

    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers...')

    distance_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Distance"]
    position_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Position"]
    null_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Null"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,4), gridspec_kw={'width_ratios': [1, 0.3]})
    ax1.set_ylabel("Peak power vs \n false alarm rate",color="black",fontsize=25, labelpad=10)
    ax1.set_xlabel("Track frequency", color="black", fontsize=25, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax1.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
    ax2.set_xticks([0, 0.5])
    ax2.set_xticklabels(["0", "0.5"])
    ax2.set_yticks([])
    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax2.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    for f in range(1,6):
        ax1.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.scatter(x=null_cells["ML_Freqs"+suffix], y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax1.scatter(x=distance_cells["ML_Freqs"+suffix], y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax1.scatter(x=position_cells["ML_Freqs"+suffix], y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax1.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set_xlim([0,5.02])
    ax1.set_ylim([-0.1,0.4])
    ax2.set_xlim([-0.05,0.55])
    ax2.set_ylim([-0.1,0.4])
    ax2.set_xlabel(r'$\Delta$ from Int', color="black", fontsize=25, labelpad=10)
    ax2.scatter(x=distance_from_integer(null_cells["ML_Freqs"+suffix]), y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(distance_cells["ML_Freqs"+suffix]), y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(position_cells["ML_Freqs"+suffix]), y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax2.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_vs_shuffle_PDN_'+suffix+'_grid_cell.png', dpi=200)
    plt.close()
    return


def plot_lomb_classifiers_vs_shuffle_non_grid_cells(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = concantenated_dataframe[concantenated_dataframe["grid_cell"] == 0]

    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers...')

    distance_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Distance"]
    position_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Position"]
    null_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Null"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,4), gridspec_kw={'width_ratios': [1, 0.3]})
    ax1.set_ylabel("Peak power vs \n false alarm rate",color="black",fontsize=25, labelpad=10)
    ax1.set_xlabel("Track frequency", color="black", fontsize=25, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax1.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
    ax2.set_xticks([0, 0.5])
    ax2.set_xticklabels(["0", "0.5"])
    ax2.set_yticks([])
    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax2.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    for f in range(1,6):
        ax1.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.scatter(x=null_cells["ML_Freqs"+suffix], y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax1.scatter(x=distance_cells["ML_Freqs"+suffix], y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax1.scatter(x=position_cells["ML_Freqs"+suffix], y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax1.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set_xlim([0,5.02])
    ax1.set_ylim([-0.1,0.4])
    ax2.set_xlim([-0.05,0.55])
    ax2.set_ylim([-0.1,0.4])
    ax2.set_xlabel(r'$\Delta$ from Int', color="black", fontsize=25, labelpad=10)
    ax2.scatter(x=distance_from_integer(null_cells["ML_Freqs"+suffix]), y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(distance_cells["ML_Freqs"+suffix]), y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(position_cells["ML_Freqs"+suffix]), y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax2.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_vs_shuffle_PDN_'+suffix+'_non_grid_cell.png', dpi=200)
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

    #avg_SNR_ratio_threshold = np.nanmean(concantenated_dataframe["shuffleSNR"+suffix])
    #avg_distance_from_integer_threshold = np.nanmean(concantenated_dataframe["shufflefreqs"+suffix])

    #fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(7,6), gridspec_kw={'width_ratios': [1, 0.3]})
    fig, ((ax4, ax5, ax6), (ax1, ax2, ax3)) = plt.subplots(2, 3, figsize=(9,9), gridspec_kw={'width_ratios': [1, 0.3, 0.3], 'height_ratios': [0.3, 1]})
    ax1.set_ylabel("Peak Power",color="black",fontsize=25, labelpad=10)
    ax1.set_xlabel("Track Frequency", color="black", fontsize=25, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax4.set_xticks(np.arange(0, 11, 1.0))
    ax1.set_yticks([0, 0.1, 0.2, 0.3])
    ax2.set_xticks([0, 0.5])
    ax4.set_xticks([0, 0.5])
    ax2.set_yticks([])
    ax4.set_ylabel("Density", color="black", fontsize=25, labelpad=10)
    ax3.set_xlabel("Density", color="black", fontsize=25, labelpad=10)
    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax2.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    for f in range(1,6):
        ax1.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
        ax4.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.scatter(x=null_cells["ML_Freqs"+suffix], y=null_cells["ML_SNRs"+suffix], color=Settings.null_color, marker="o")
    ax1.scatter(x=distance_cells["ML_Freqs"+suffix], y=distance_cells["ML_SNRs"+suffix], color=Settings.egocentric_color, marker="o")
    ax1.scatter(x=position_cells["ML_Freqs"+suffix], y=position_cells["ML_SNRs"+suffix], color=Settings.allocentric_color, marker="o")
    ax1.axhline(y=0.024, color="red", linewidth=3,linestyle="dashed")
    ax3.hist(position_cells["ML_SNRs"+suffix], density=True, range=(0,0.4), bins=40, histtype="stepfilled", alpha=0.5, color=Settings.allocentric_color, orientation="horizontal")
    ax3.hist(distance_cells["ML_SNRs"+suffix], density=True, range=(0,0.4), bins=40, histtype="stepfilled", alpha=0.5, color=Settings.egocentric_color, orientation="horizontal")
    ax3.hist(null_cells["ML_SNRs"+suffix], density=True, range=(0,0.4), bins=40, histtype="stepfilled", alpha=0.5, color=Settings.null_color, orientation="horizontal")
    ax4.hist(position_cells["ML_Freqs"+suffix], density=True, range=(0,5), bins=100, histtype="stepfilled", alpha=0.5, color=Settings.allocentric_color)
    ax4.hist(distance_cells["ML_Freqs"+suffix], density=True, range=(0,5), bins=100, histtype="stepfilled", alpha=0.5, color=Settings.egocentric_color)
    ax4.hist(null_cells["ML_Freqs"+suffix], density=True, range=(0,5), bins=100, histtype="stepfilled", alpha=0.5, color=Settings.null_color)
    ax5.hist(distance_from_integer(position_cells["ML_Freqs"+suffix]), density=True, range=(0,0.5), bins=10, histtype="stepfilled", alpha=0.5, color=Settings.allocentric_color)
    ax5.hist(distance_from_integer(distance_cells["ML_Freqs"+suffix]), density=True, range=(0,0.5), bins=10, histtype="stepfilled", alpha=0.5, color=Settings.egocentric_color)
    ax5.hist(distance_from_integer(null_cells["ML_Freqs"+suffix]), density=True, range=(0,0.5), bins=10, histtype="stepfilled", alpha=0.5, color=Settings.null_color)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax1.set_xlim([0,5.02])
    ax4.set_xlim([0,5.02])
    ax1.set_ylim([0,0.3])
    ax2.set_xlim([-0.05,0.55])
    ax2.set_ylim([0,0.3])
    ax3.set_ylim([0,0.3])
    ax5.set_xlim([-0.05,0.55])
    ax2.set_xlabel(r'$\Delta$ from Integer', color="black", fontsize=25, labelpad=10)
    ax2.scatter(x=distance_from_integer(position_cells["ML_Freqs"+suffix]), y=position_cells["ML_SNRs"+suffix], color=Settings.allocentric_color, marker="o")
    ax2.scatter(x=distance_from_integer(distance_cells["ML_Freqs"+suffix]), y=distance_cells["ML_SNRs"+suffix], color=Settings.egocentric_color, marker="o")
    ax2.scatter(x=distance_from_integer(null_cells["ML_Freqs"+suffix]), y=null_cells["ML_SNRs"+suffix], color=Settings.null_color, marker="o")
    ax2.axhline(y=0.024, color="red", linewidth=3,linestyle="dashed")
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_PDN_'+suffix+'.png', dpi=200)
    plt.close()
    return


def significance_bar(start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 15,color = 'k'):
    # draw a line with downticks at the ends
    plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)
    # draw the text with a bounding box covering up the line
    plt.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize)

def plot_lomb_classifier_powers_vs_groups(concantenated_dataframe, suffix="", save_path="", fig_size=(6,6)):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Distance"]["ML_SNRs"+suffix])
    g_position_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Position"]["ML_SNRs"+suffix])
    g_null_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Null"]["ML_SNRs"+suffix])
    ng_distance_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Distance"]["ML_SNRs"+suffix])
    ng_position_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Position"]["ML_SNRs"+suffix])
    ng_null_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Null"]["ML_SNRs"+suffix])

    fig, ax = plt.subplots(figsize=fig_size)
    data = [g_position_cells[~np.isnan(g_position_cells)],
            g_distance_cells[~np.isnan(g_distance_cells)],
            g_null_cells[~np.isnan(g_null_cells)],
            ng_position_cells[~np.isnan(ng_position_cells)],
            ng_distance_cells[~np.isnan(ng_distance_cells)],
            ng_null_cells[~np.isnan(ng_null_cells)]]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3,5,6,7], boxprops=boxprops, medianprops=medianprops,
               whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(bottom=0, top=0.3)
    ax.set_xlim(left=0.5, right=3.5)
    #ax.set_xticks([2, 6])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    #ax.set_xticklabels(["G", "NG"])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Peak power", fontsize=20)
    significance_bar(start=1, end=2, height=0.292, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[1])[1]))
    #significance_bar(start=5, end=6, height=0.3, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[4], data[5])[1]))
    #significance_bar(start=2, end=6, height=0.3125, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[1], data[5])[1]))
    #significance_bar(start=1, end=5, height=0.325, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[4])[1]))
    plt.savefig(save_path + '/lomb_classifier_powers_vs_groups.png', dpi=300)
    plt.close()


    fig, ax = plt.subplots(figsize=(4,4))
    data = [g_position_cells[~np.isnan(g_position_cells)],
            g_distance_cells[~np.isnan(g_distance_cells)],
            g_null_cells[~np.isnan(g_null_cells)],
            ng_position_cells[~np.isnan(ng_position_cells)],
            ng_distance_cells[~np.isnan(ng_distance_cells)],
            ng_null_cells[~np.isnan(ng_null_cells)]]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    parts = ax.violinplot(data, positions=[1,2,3,5,6,7], showmeans=False, showmedians=False, showextrema=False)
    for patch, color, data_x, pos_x in zip(parts['bodies'], colors, data, [1,2,3,5,6,7]):
        patch.set_facecolor(color)
        patch.set_alpha(1)
        ax.scatter((np.ones(len(data_x))*pos_x)+np.random.uniform(low=-0.3, high=0.3, size=len(data_x)), data_x, color="black", marker="o", zorder=-1, alpha=0.3, s=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0, top=0.3)
    ax.set_xlim(left=0.5, right=7.5)
    ax.set_xticks([2, 6])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["G", "NG"])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Peak power", fontsize=20)
    significance_bar(start=1, end=2, height=0.3, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[1])[1]))
    significance_bar(start=5, end=6, height=0.3, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[4], data[5])[1]))
    significance_bar(start=2, end=6, height=0.3125, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[1], data[5])[1]))
    significance_bar(start=1, end=5, height=0.325, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[4])[1]))
    plt.savefig(save_path + '/lomb_classifier_powers_vs_groups_voliin.png', dpi=300)
    plt.close()

    print("comping peak powers between postion and distance encoding grid cells, df=",str(len(data[0])+len(data[1])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0], data[1])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0], data[1])[0]))


    # plot cumulative histogram
    fig, ax = plt.subplots(figsize=(3,3))
    _, _, patches1 = ax.hist(data[0], bins=500, color=Settings.allocentric_color, histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches2 = ax.hist(data[1], bins=500, color=Settings.egocentric_color, histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches3 = ax.hist(data[2], bins=500, color=Settings.null_color, histtype="step", density=True, cumulative=True, linewidth=2)
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    patches2[0].set_xy(patches2[0].get_xy()[:-1])
    patches3[0].set_xy(patches3[0].get_xy()[:-1])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0, top=1)
    ax.set_yticks([0,1])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("peak power", fontsize=20)
    ax.set_ylabel("Cumulative prob", fontsize=20)
    plt.savefig(save_path + '/lomb_classifier_power_vs_groups_cumhist.png', dpi=300)
    plt.close()

    return

def plot_lomb_classifier_mfr_vs_groups_vs_open_field(concantenated_dataframe, suffix="", save_path="", fig_size=(6,6)):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Distance"]['mean_firing_rate_vr'])
    g_position_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Position"]['mean_firing_rate_vr'])
    g_null_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Null"]['mean_firing_rate_vr'])
    ng_distance_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Distance"]['mean_firing_rate_vr'])
    ng_position_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Position"]['mean_firing_rate_vr'])
    ng_null_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Null"]['mean_firing_rate_vr'])

    g_distance_cells_of = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Distance"]['mean_firing_rate_of'])
    g_position_cells_of = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Position"]['mean_firing_rate_of'])
    g_null_cells_of = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Null"]['mean_firing_rate_of'])
    ng_distance_cells_of = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Distance"]['mean_firing_rate_of'])
    ng_position_cells_of = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Position"]['mean_firing_rate_of'])
    ng_null_cells_of = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Null"]['mean_firing_rate_of'])

    print("mean firing rate in the vr for position encoding grid cells is ", np.nanmean(g_position_cells), " +- ", np.nanstd(g_position_cells))
    print("mean firing rate in the of for position encoding grid cells is ", np.nanmean(g_position_cells_of), " +- ", np.nanstd(g_position_cells_of))
    print("mean firing rate in the vr for distance encoding grid cells is ", np.nanmean(g_distance_cells), " +- ", np.nanstd(g_distance_cells))
    print("mean firing rate in the of for distance encoding grid cells is ", np.nanmean(g_distance_cells_of), " +- ", np.nanstd(g_distance_cells_of))
    print("mean firing rate in the vr for null encoding grid cells is ", np.nanmean(g_null_cells), " +- ", np.nanstd(g_null_cells))
    print("mean firing rate in the of for null encoding grid cells is ", np.nanmean(g_null_cells_of), " +- ", np.nanstd(g_null_cells_of))

    print("comping mean firing rates between postion vr and position of encoding grid cells, p = ", str(scipy.stats.wilcoxon(x=g_position_cells, y=g_position_cells_of)[1]), ", t = ", str(scipy.stats.wilcoxon(x=g_position_cells, y=g_position_cells_of)[0]),", df = ",str(len(g_position_cells)-1))
    print("comping mean firing rates between distnace vr and distnace of encoding grid cells, p = ", str(scipy.stats.wilcoxon(x=g_distance_cells, y=g_distance_cells_of)[1]), ", t = ", str(scipy.stats.wilcoxon(x=g_distance_cells, y=g_distance_cells_of)[0]),", df = ",str(len(g_distance_cells)-1))
    print("comping mean firing rates between null vr and null of encoding grid cells, p = ", str(scipy.stats.wilcoxon(x=g_null_cells, y=g_null_cells_of)[1]), ", t = ", str(scipy.stats.wilcoxon(x=g_null_cells, y=g_null_cells_of)[0]),", df = ",str(len(g_null_cells)-1))

    # plot scatter plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(g_distance_cells, g_distance_cells_of, color=Settings.egocentric_color)
    ax.scatter(g_position_cells, g_position_cells_of, color=Settings.allocentric_color)
    ax.plot(np.arange(0,25), np.arange(0,25), color="black")
    ax.scatter(g_null_cells, g_null_cells_of, color=Settings.null_color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    #ax.set_yticks([0,1])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("VR Mean Firing Rate (Hz)", fontsize=20)
    ax.set_ylabel("OF Mean Firing Rate (Hz)", fontsize=20)
    plt.savefig(save_path + '/mean_firing_rate_vs_mean_firing_rate_of_for_groups.png', dpi=300)
    plt.close()
    return

def plot_lomb_classifier_spatinfo_vs_groups(concantenated_dataframe, suffix="", save_path="", fig_size=(6,6), score="spatial_information_score_Ispike_vr"):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Distance"][score])
    g_position_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Position"][score])
    g_null_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Null"][score])
    ng_distance_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Distance"][score])
    ng_position_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Position"][score])
    ng_null_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Null"][score])

    fig, ax = plt.subplots(figsize=fig_size)
    data = [g_position_cells[~np.isnan(g_position_cells)],
            g_distance_cells[~np.isnan(g_distance_cells)],
            g_null_cells[~np.isnan(g_null_cells)],
            ng_position_cells[~np.isnan(ng_position_cells)],
            ng_distance_cells[~np.isnan(ng_distance_cells)],
            ng_null_cells[~np.isnan(ng_null_cells)]]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #ax.set_ylim(bottom=0, top=4)
    ax.set_xlim(left=0.5, right=3.5)
    #ax.set_xticks([2, 6])
    #ax.set_yticks([0, 1,2,3,4])
    #ax.set_xticklabels(["G", "NG"])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Spatial info", fontsize=20)
    #significance_bar(start=1, end=2, height=0.8, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[1])[1]))
    #significance_bar(start=1, end=3, height=0.9, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[2])[1]))
    #significance_bar(start=2, end=3, height=0.7, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[1], data[2])[1]))
    #significance_bar(start=2, end=6, height=0.3125, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[1], data[5])[1]))
    #significance_bar(start=1, end=5, height=0.325, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[4])[1]))
    plt.savefig(save_path + '/lomb_classifier_spatinfo_vs_groups_'+score+'.png', dpi=300)
    plt.close()

    print("comping spatinfo between postion and distance encoding grid cells, df=",str(len(data[0])+len(data[1])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0], data[1])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0], data[1])[0]))
    print("comping spatinfo postion and null encoding grid cells, df=",str(len(data[0])+len(data[2])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0], data[2])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0], data[2])[0]))
    print("comping spatinfo null and distance encoding grid cells, df=",str(len(data[2])+len(data[1])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[2], data[1])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[2], data[1])[0]))

    # plot cumulative histogram
    fig, ax = plt.subplots(figsize=(3,3))
    _, _, patches1 = ax.hist(data[0], bins=500, color=Settings.allocentric_color, histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches2 = ax.hist(data[1], bins=500, color=Settings.egocentric_color, histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches3 = ax.hist(data[2], bins=500, color=Settings.null_color, histtype="step", density=True, cumulative=True, linewidth=2)
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    patches2[0].set_xy(patches2[0].get_xy()[:-1])
    patches3[0].set_xy(patches3[0].get_xy()[:-1])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0, top=1)
    ax.set_yticks([0,1])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("Spatial info", fontsize=20)
    ax.set_ylabel("Cumulative prob", fontsize=20)
    plt.savefig(save_path + '/lomb_classifier_spatinfo_vs_groups_cumhist'+score+'.png', dpi=300)
    plt.close()
    return

def plot_lomb_classifier_mfr_vs_groups(concantenated_dataframe, suffix="", save_path="", fig_size=(6,6)):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Distance"]['mean_firing_rate_vr'])
    g_position_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Position"]['mean_firing_rate_vr'])
    g_null_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Null"]['mean_firing_rate_vr'])
    ng_distance_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Distance"]['mean_firing_rate_vr'])
    ng_position_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Position"]['mean_firing_rate_vr'])
    ng_null_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Null"]['mean_firing_rate_vr'])

    fig, ax = plt.subplots(figsize=fig_size)
    data = [g_position_cells[~np.isnan(g_position_cells)],
            g_distance_cells[~np.isnan(g_distance_cells)],
            g_null_cells[~np.isnan(g_null_cells)],
            ng_position_cells[~np.isnan(ng_position_cells)],
            ng_distance_cells[~np.isnan(ng_distance_cells)],
            ng_null_cells[~np.isnan(ng_null_cells)]]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(bottom=0, top=15)
    ax.set_xlim(left=0.5, right=3.5)
    #ax.set_xticks([2, 6])
    ax.set_yticks([0, 5, 10, 15])
    #ax.set_xticklabels(["G", "NG"])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Mean firing rate", fontsize=20)
    significance_bar(start=1, end=2, height=15, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[1])[1]))
    significance_bar(start=1, end=3, height=14, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[2])[1]))
    significance_bar(start=2, end=3, height=7, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[1], data[2])[1]))
    #significance_bar(start=2, end=6, height=0.3125, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[1], data[5])[1]))
    #significance_bar(start=1, end=5, height=0.325, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[4])[1]))
    plt.savefig(save_path + '/lomb_classifier_mfr_vs_groups.png', dpi=300)
    plt.close()

    print("comping mean firing rates between postion and distance encoding grid cells, df=",str(len(data[0])+len(data[1])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0], data[1])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0], data[1])[0]))
    print("comping mean firing rates postion and null encoding grid cells, df=",str(len(data[0])+len(data[2])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0], data[2])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0], data[2])[0]))
    print("comping mean firing rates null and distance encoding grid cells, df=",str(len(data[2])+len(data[1])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[2], data[1])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[2], data[1])[0]))

    # plot cumulative histogram
    fig, ax = plt.subplots(figsize=(3,3))
    _, _, patches1 = ax.hist(data[0], bins=500, color=Settings.allocentric_color, histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches2 = ax.hist(data[1], bins=500, color=Settings.egocentric_color, histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches3 = ax.hist(data[2], bins=500, color=Settings.null_color, histtype="step", density=True, cumulative=True, linewidth=2)
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    patches2[0].set_xy(patches2[0].get_xy()[:-1])
    patches3[0].set_xy(patches3[0].get_xy()[:-1])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0, top=1)
    ax.set_yticks([0,1])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("Mean Firing Rate (Hz)", fontsize=20)
    ax.set_ylabel("Cumulative prob", fontsize=20)
    plt.savefig(save_path + '/lomb_classifier_mfr_vs_groups_cumhist.png', dpi=300)
    plt.close()
    return


def plot_lomb_classifier_peak_width_vs_groups(concantenated_dataframe, suffix="", save_path="", fig_size=(6,6)):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    print('plotting lomb classifers...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Distance"]['ML_peak_width'])
    g_position_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Position"]['ML_peak_width'])
    g_null_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"+suffix] == "Null"]['ML_peak_width'])
    ng_distance_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Distance"]['ML_peak_width'])
    ng_position_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Position"]['ML_peak_width'])
    ng_null_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"+suffix] == "Null"]['ML_peak_width'])

    fig, ax = plt.subplots(figsize=fig_size)
    data = [g_position_cells[~np.isnan(g_position_cells)],
            g_distance_cells[~np.isnan(g_distance_cells)],
            g_null_cells[~np.isnan(g_null_cells)],
            ng_position_cells[~np.isnan(ng_position_cells)],
            ng_distance_cells[~np.isnan(ng_distance_cells)],
            ng_null_cells[~np.isnan(ng_null_cells)]]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(bottom=0, top=1.5)
    ax.set_xlim(left=0.5, right=3.5)
    #ax.set_xticks([2, 6])
    ax.set_yticks([0, 0.5, 1, 1.5])
    #ax.set_xticklabels(["G", "NG"])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Peak width", fontsize=20)
    significance_bar(start=1, end=2, height=1.5, displaystring=get_p_text(scipy.stats.ks_2samp(data[0], data[1])[1]))
    significance_bar(start=1, end=3, height=1.4, displaystring=get_p_text(scipy.stats.ks_2samp(data[0], data[2])[1]))
    significance_bar(start=2, end=3, height=1.3, displaystring=get_p_text(scipy.stats.ks_2samp(data[1], data[2])[1]))
    #significance_bar(start=2, end=6, height=0.3125, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[1], data[5])[1]))
    #significance_bar(start=1, end=5, height=0.325, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[4])[1]))
    plt.savefig(save_path + '/lomb_classifier_peak_width_vs_groups.png', dpi=300)
    plt.close()


    print("comping peak widths between postion and distance encoding grid cells, df=",str(len(data[0])+len(data[1])-2), ", p= ", str(scipy.stats.ks_2samp(data[0], data[1])[1]), ", ks= ", str(scipy.stats.ks_2samp(data[0], data[1])[0]))



    # plot cumulative histogram
    fig, ax = plt.subplots(figsize=(3,3))
    _, _, patches1 = ax.hist(data[0], bins=500, color=Settings.allocentric_color, histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches2 = ax.hist(data[1], bins=500, color=Settings.egocentric_color, histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patches3 = ax.hist(data[2], bins=500, color=Settings.null_color, histtype="step", density=True, cumulative=True, linewidth=2)
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    patches2[0].set_xy(patches2[0].get_xy()[:-1])
    patches3[0].set_xy(patches3[0].get_xy()[:-1])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0, top=1)
    ax.set_yticks([0,1])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("peak width", fontsize=20)
    ax.set_ylabel("Cumulative prob", fontsize=20)
    plt.savefig(save_path + '/lomb_classifier_peakwidth_vs_groups_cumhist.png', dpi=300)
    plt.close()
    return

def get_rolling_percent_encoding(df, code="P", tt=1, hmt="hit"):
    if code == "P":
        result_column = "rolling:encoding_position_by_trial_category"
    elif code == "D":
        result_column = "rolling:encoding_distance_by_trial_category"
    elif code == "N":
        result_column = "rolling:encoding_null_by_trial_category"

    # tt and i index are shared
    i = int(tt)

    if hmt == "hit":
        j =0
    elif hmt == "try":
        j=1
    elif hmt == "miss":
        j=2

    result=[]
    for m in range(len(df)):
        if isinstance(df[result_column].iloc[m], list):
            result.append(df[result_column].iloc[m][i][j])
        else:
            print(df["session_id"].iloc[m], " is missing an entry in ", result_column)
    return np.array(result)*100

def plot_percentage_encoding_by_trial_category_each_mouse_from_global(combined_df, save_path="", suffix=""):
    print('plotting lomb classifers proportions...')
    for tt in [0, 1, 2]:
        fig, ax = plt.subplots(figsize=(6, 6))
        for code, code_color, code_z_order in zip(["P", "D", "N"], [Settings.allocentric_color, Settings.egocentric_color, Settings.null_color], [1,2,3]):
            for i, mouse_id in enumerate(np.unique(combined_df["mouse"])):
                mouse_cells = combined_df[combined_df["mouse"] == mouse_id]
                mouse_hit = get_percentage_from_global_code(mouse_cells, code=code, tt=tt, hmt="hit")
                mouse_try = get_percentage_from_global_code(mouse_cells, code=code, tt=tt, hmt="try")
                mouse_miss = get_percentage_from_global_code(mouse_cells, code=code, tt=tt, hmt="miss")

                mask = ~np.isnan(mouse_hit) & ~np.isnan(mouse_try) & ~np.isnan(mouse_miss)
                ax.plot([2, 6, 10], [np.nanmean(mouse_hit[mask]), np.nanmean(mouse_try[mask]), np.nanmean(mouse_miss[mask])], clip_on=False, color=code_color, linewidth=1.5, alpha=0.3, zorder=-1)

            All_hit = get_percentage_from_global_code(combined_df, code=code, tt=tt, hmt="hit")
            All_try = get_percentage_from_global_code(combined_df, code=code, tt=tt, hmt="try")
            All_miss = get_percentage_from_global_code(combined_df, code=code, tt=tt, hmt="miss")
            mask = ~np.isnan(All_hit) & ~np.isnan(All_try) & ~np.isnan(All_miss)

            #ax.plot([2, 6, 10],[np.nanmean(All_hit[mask]), np.nanmean(All_try[mask]), np.nanmean(All_miss[mask])],clip_on=False, linewidth=3, color=code_color, zorder=3)
            ax.errorbar(x=[2, 6, 10], y=[np.nanmean(All_hit[mask]), np.nanmean(All_try[mask]), np.nanmean(All_miss[mask])],
                        yerr=[scipy.stats.sem(All_hit[mask], axis=0), scipy.stats.sem(All_try[mask], axis=0), scipy.stats.sem(All_miss[mask], axis=0)], capsize=25, linewidth=3, color=code_color, zorder=code_z_order)

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.set_xticks([2,6,10])
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_xticks([])
        #ax.set_xticklabels(["Hit", "Try", "Run"])
        ax.set_xlim(left=0, right=12)
        ax.set_ylim(bottom=0, top=100)
        #ax.legend(loc="best")
        plt.savefig(save_path + '/rolling_percent_encooding_from_global_'+str(tt)+'_for_each_mouse.png', dpi=300)
        plt.close()
    return

def plot_percentage_encoding_by_trial_category_each_mouse(combined_df, save_path="", suffix=""):
    print('plotting lomb classifers proportions...')
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    combined_df = combined_df[combined_df["classifier"] == "G"]
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for tt in [0, 1, 2]:
        fig, ax = plt.subplots(figsize=(6, 6))
        for code, code_color, code_z_order in zip(["P", "D"], [Settings.allocentric_color, Settings.egocentric_color], [1,2]):
            for i, mouse_id in enumerate(np.unique(combined_df["mouse"])):
                mouse_cells = combined_df[combined_df["mouse"] == mouse_id]
                mouse_hit = get_percentage_from_rolling_classification2(mouse_cells, code=code, tt=tt, hmt="hit")
                mouse_try = get_percentage_from_rolling_classification2(mouse_cells, code=code, tt=tt, hmt="try")
                mouse_miss = get_percentage_from_rolling_classification2(mouse_cells, code=code, tt=tt, hmt="miss")

                mask = ~np.isnan(mouse_hit) & ~np.isnan(mouse_try) & ~np.isnan(mouse_miss)
                ax.plot([2, 6, 10], [np.nanmean(mouse_hit[mask]), np.nanmean(mouse_try[mask]), np.nanmean(mouse_miss[mask])], clip_on=False, color=code_color, linewidth=1.5, alpha=0.3, zorder=-1)

            All_hit = get_percentage_from_rolling_classification2(combined_df, code=code, tt=tt, hmt="hit")
            All_try = get_percentage_from_rolling_classification2(combined_df, code=code, tt=tt, hmt="try")
            All_miss = get_percentage_from_rolling_classification2(combined_df, code=code, tt=tt, hmt="miss")
            mask = ~np.isnan(All_hit) & ~np.isnan(All_try) & ~np.isnan(All_miss)

            #ax.plot([2, 6, 10],[np.nanmean(All_hit[mask]), np.nanmean(All_try[mask]), np.nanmean(All_miss[mask])],clip_on=False, linewidth=3, color=code_color, zorder=3)
            ax.errorbar(x=[2, 6, 10], y=[np.nanmean(All_hit[mask]), np.nanmean(All_try[mask]), np.nanmean(All_miss[mask])],
                        yerr=[scipy.stats.sem(All_hit[mask], axis=0), scipy.stats.sem(All_try[mask], axis=0), scipy.stats.sem(All_miss[mask], axis=0)], capsize=25, linewidth=3, color=code_color, zorder=code_z_order)

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.set_xticks([2,6,10])
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.set_xticks([])
        #ax.set_xticklabels(["Hit", "Try", "Run"])
        ax.set_xlim(left=0, right=12)
        ax.set_ylim(bottom=0, top=100)
        #ax.legend(loc="best")
        plt.savefig(save_path + '/rolling_percent_encooding_'+str(tt)+'_for_each_mouse.png', dpi=300)
        plt.close()
    return

def plot_percentage_encoding_by_trial_category_non_grid_cells(combined_df, save_path="", suffix="_non_grid_cells"):
    print('plotting lomb classifers proportions...')
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    # we only want to look at the remapped coding cells (<85% encoding)
    # grid_cells = grid_cells[(grid_cells["rolling:proportion_encoding_position"] < 0.85) &
    #                        (grid_cells["rolling:proportion_encoding_distance"] < 0.85)]

    p_b_hit = get_rolling_percent_encoding(non_grid_cells, code="P", tt=0, hmt="hit")
    d_b_hit = get_rolling_percent_encoding(non_grid_cells, code="D", tt=0, hmt="hit")
    n_b_hit = get_rolling_percent_encoding(non_grid_cells, code="N", tt=0, hmt="hit")
    p_b_try = get_rolling_percent_encoding(non_grid_cells, code="P", tt=0, hmt="try")
    d_b_try = get_rolling_percent_encoding(non_grid_cells, code="D", tt=0, hmt="try")
    n_b_try = get_rolling_percent_encoding(non_grid_cells, code="N", tt=0, hmt="try")
    p_b_miss = get_rolling_percent_encoding(non_grid_cells, code="P", tt=0, hmt="miss")
    d_b_miss = get_rolling_percent_encoding(non_grid_cells, code="D", tt=0, hmt="miss")
    n_b_miss = get_rolling_percent_encoding(non_grid_cells, code="N", tt=0, hmt="miss")
    p_nb_hit = get_rolling_percent_encoding(non_grid_cells, code="P", tt=1, hmt="hit")
    d_nb_hit = get_rolling_percent_encoding(non_grid_cells, code="D", tt=1, hmt="hit")
    n_nb_hit = get_rolling_percent_encoding(non_grid_cells, code="N", tt=1, hmt="hit")
    p_nb_try = get_rolling_percent_encoding(non_grid_cells, code="P", tt=1, hmt="try")
    d_nb_try = get_rolling_percent_encoding(non_grid_cells, code="D", tt=1, hmt="try")
    n_nb_try = get_rolling_percent_encoding(non_grid_cells, code="N", tt=1, hmt="try")
    p_nb_miss = get_rolling_percent_encoding(non_grid_cells, code="P", tt=1, hmt="miss")
    d_nb_miss = get_rolling_percent_encoding(non_grid_cells, code="D", tt=1, hmt="miss")
    n_nb_miss = get_rolling_percent_encoding(non_grid_cells, code="N", tt=1, hmt="miss")
    p_p_hit = get_rolling_percent_encoding(non_grid_cells, code="P", tt=2, hmt="hit")
    d_p_hit = get_rolling_percent_encoding(non_grid_cells, code="D", tt=2, hmt="hit")
    n_p_hit = get_rolling_percent_encoding(non_grid_cells, code="N", tt=2, hmt="hit")
    p_p_try = get_rolling_percent_encoding(non_grid_cells, code="P", tt=2, hmt="try")
    d_p_try = get_rolling_percent_encoding(non_grid_cells, code="D", tt=2, hmt="try")
    n_p_try = get_rolling_percent_encoding(non_grid_cells, code="N", tt=2, hmt="try")
    p_p_miss = get_rolling_percent_encoding(non_grid_cells, code="P", tt=2, hmt="miss")
    d_p_miss = get_rolling_percent_encoding(non_grid_cells, code="D", tt=2, hmt="miss")
    n_p_miss = get_rolling_percent_encoding(non_grid_cells, code="N", tt=2, hmt="miss")

    p_b_hit = get_percentage_from_rolling_classification2(non_grid_cells, code="P", tt=0, hmt="hit")
    d_b_hit = get_percentage_from_rolling_classification2(non_grid_cells, code="D", tt=0, hmt="hit")
    n_b_hit = get_percentage_from_rolling_classification2(non_grid_cells, code="N", tt=0, hmt="hit")
    p_b_try = get_percentage_from_rolling_classification2(non_grid_cells, code="P", tt=0, hmt="try")
    d_b_try = get_percentage_from_rolling_classification2(non_grid_cells, code="D", tt=0, hmt="try")
    n_b_try = get_percentage_from_rolling_classification2(non_grid_cells, code="N", tt=0, hmt="try")
    p_b_miss = get_percentage_from_rolling_classification2(non_grid_cells, code="P", tt=0, hmt="miss")
    d_b_miss = get_percentage_from_rolling_classification2(non_grid_cells, code="D", tt=0, hmt="miss")
    n_b_miss = get_percentage_from_rolling_classification2(non_grid_cells, code="N", tt=0, hmt="miss")
    p_nb_hit = get_percentage_from_rolling_classification2(non_grid_cells, code="P", tt=1, hmt="hit")
    d_nb_hit = get_percentage_from_rolling_classification2(non_grid_cells, code="D", tt=1, hmt="hit")
    n_nb_hit = get_percentage_from_rolling_classification2(non_grid_cells, code="N", tt=1, hmt="hit")
    p_nb_try = get_percentage_from_rolling_classification2(non_grid_cells, code="P", tt=1, hmt="try")
    d_nb_try = get_percentage_from_rolling_classification2(non_grid_cells, code="D", tt=1, hmt="try")
    n_nb_try = get_percentage_from_rolling_classification2(non_grid_cells, code="N", tt=1, hmt="try")
    p_nb_miss = get_percentage_from_rolling_classification2(non_grid_cells, code="P", tt=1, hmt="miss")
    d_nb_miss = get_percentage_from_rolling_classification2(non_grid_cells, code="D", tt=1, hmt="miss")
    n_nb_miss = get_percentage_from_rolling_classification2(non_grid_cells, code="N", tt=1, hmt="miss")
    p_p_hit = get_percentage_from_rolling_classification2(non_grid_cells, code="P", tt=2, hmt="hit")
    d_p_hit = get_percentage_from_rolling_classification2(non_grid_cells, code="D", tt=2, hmt="hit")
    n_p_hit = get_percentage_from_rolling_classification2(non_grid_cells, code="N", tt=2, hmt="hit")
    p_p_try = get_percentage_from_rolling_classification2(non_grid_cells, code="P", tt=2, hmt="try")
    d_p_try = get_percentage_from_rolling_classification2(non_grid_cells, code="D", tt=2, hmt="try")
    n_p_try = get_percentage_from_rolling_classification2(non_grid_cells, code="N", tt=2, hmt="try")
    p_p_miss = get_percentage_from_rolling_classification2(non_grid_cells, code="P", tt=2, hmt="miss")
    d_p_miss = get_percentage_from_rolling_classification2(non_grid_cells, code="D", tt=2, hmt="miss")
    n_p_miss = get_percentage_from_rolling_classification2(non_grid_cells, code="N", tt=2, hmt="miss")

    b_mask = ~np.isnan(p_b_hit) & ~np.isnan(p_b_try) & ~np.isnan(p_b_miss)
    nb_mask = ~np.isnan(p_nb_hit) & ~np.isnan(p_nb_try) & ~np.isnan(p_nb_miss)
    p_mask = ~np.isnan(p_p_hit) & ~np.isnan(p_p_try) & ~np.isnan(p_p_miss)

    colors = [Settings.allocentric_color, Settings.egocentric_color, Settings.null_color,
              Settings.allocentric_color, Settings.egocentric_color, Settings.null_color,
              Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]

    data = [p_b_hit[b_mask], d_b_hit[b_mask], n_b_hit[b_mask],
            p_b_try[b_mask], d_b_try[b_mask], n_b_try[b_mask],
            p_b_miss[b_mask], d_b_miss[b_mask], n_b_miss[b_mask]]

    fig, ax = plt.subplots(figsize=(6, 6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1, 2, 3, 5, 6, 7, 9, 10, 11], widths=1, boxprops=boxprops,
                     medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # ax.set_yticks([0,0.5,1])
    ax.set_xticks([2, 6, 10])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0, right=12)
    # fig.tight_layout()
    # plt.subplots_adjust(left=0.25, bottom=0.2)
    # ax.set_ylabel("% encoding", fontsize=20)
    plt.savefig(save_path + '/rolling_percent_encooding_beaconed' + suffix + '.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[0]), np.mean(data[3]), np.mean(data[6])],
                yerr=[scipy.stats.sem(data[0], axis=0), scipy.stats.sem(data[3], axis=0),
                      scipy.stats.sem(data[6], axis=0)], color=Settings.allocentric_color, capsize=25, linewidth=3)
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[1]), np.mean(data[4]), np.mean(data[7])],
                yerr=[scipy.stats.sem(data[1], axis=0), scipy.stats.sem(data[4], axis=0),
                      scipy.stats.sem(data[7], axis=0)], color=Settings.egocentric_color, capsize=25, linewidth=3)
    # ax.errorbar(x=[1,2,3], y=[np.mean(data[2]),  np.mean(data[5]), np.mean(data[8])], yerr=[scipy.stats.sem(data[2], axis=0), scipy.stats.sem(data[5], axis=0), scipy.stats.sem(data[8], axis=0)], color=Settings.null_color, capsize=25, linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[0]), np.mean(data[3]), np.mean(data[6])], color=Settings.allocentric_color,
            linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[1]), np.mean(data[4]), np.mean(data[7])], color=Settings.egocentric_color,
            linewidth=3)
    # ax.plot([1,2,3], [np.mean(data[2]), np.mean(data[5]), np.mean(data[8])], color=Settings.null_color, linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/rolling_percent_encooding_beaconed2' + suffix + '.png', dpi=300)
    plt.close()

    data = [p_nb_hit[nb_mask], d_nb_hit[nb_mask], n_nb_hit[nb_mask],
            p_nb_try[nb_mask], d_nb_try[nb_mask], n_nb_try[nb_mask],
            p_nb_miss[nb_mask], d_nb_miss[nb_mask], n_nb_miss[nb_mask]]

    fig, ax = plt.subplots(figsize=(6, 6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1, 2, 3, 5, 6, 7, 9, 10, 11], widths=1, boxprops=boxprops,
                     medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # ax.set_yticks([0,0.5,1])
    ax.set_xticks([2, 6, 10])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0, right=12)
    # fig.tight_layout()
    # plt.subplots_adjust(left=0.25, bottom=0.2)
    # ax.set_ylabel("% encoding", fontsize=20)
    plt.savefig(save_path + '/rolling_percent_encooding_non_beaconed' + suffix + '.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[0]), np.mean(data[3]), np.mean(data[6])],
                yerr=[scipy.stats.sem(data[0], axis=0), scipy.stats.sem(data[3], axis=0),
                      scipy.stats.sem(data[6], axis=0)], color=Settings.allocentric_color, capsize=25, linewidth=3)
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[1]), np.mean(data[4]), np.mean(data[7])],
                yerr=[scipy.stats.sem(data[1], axis=0), scipy.stats.sem(data[4], axis=0),
                      scipy.stats.sem(data[7], axis=0)], color=Settings.egocentric_color, capsize=25, linewidth=3)
    # ax.errorbar(x=[1,2,3], y=[np.mean(data[2]),  np.mean(data[5]), np.mean(data[8])], yerr=[scipy.stats.sem(data[2], axis=0), scipy.stats.sem(data[5], axis=0), scipy.stats.sem(data[8], axis=0)], color=Settings.null_color, capsize=25, linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[0]), np.mean(data[3]), np.mean(data[6])], color=Settings.allocentric_color,
            linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[1]), np.mean(data[4]), np.mean(data[7])], color=Settings.egocentric_color,
            linewidth=3)
    # ax.plot([1,2,3], [np.mean(data[2]), np.mean(data[5]), np.mean(data[8])], color=Settings.null_color, linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/rolling_percent_encooding_non_beaconed2' + suffix + '.png', dpi=300)
    plt.close()

    data = [p_p_hit[p_mask], d_p_hit[p_mask], n_p_hit[p_mask],
            p_p_try[p_mask], d_p_try[p_mask], n_p_try[p_mask],
            p_p_miss[p_mask], d_p_miss[p_mask], n_p_miss[p_mask]]

    fig, ax = plt.subplots(figsize=(6, 6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1, 2, 3, 5, 6, 7, 9, 10, 11], widths=1, boxprops=boxprops,
                     medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # ax.set_yticks([0,0.5,1])
    ax.set_xticks([2, 6, 10])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0, right=12)
    # fig.tight_layout()
    # plt.subplots_adjust(left=0.25, bottom=0.2)
    # ax.set_ylabel("% encoding", fontsize=20)
    plt.savefig(save_path + '/rolling_percent_encooding_probe' + suffix + '.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[0]), np.mean(data[3]), np.mean(data[6])],
                yerr=[scipy.stats.sem(data[0], axis=0), scipy.stats.sem(data[3], axis=0),
                      scipy.stats.sem(data[6], axis=0)], color=Settings.allocentric_color, capsize=25, linewidth=3)
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[1]), np.mean(data[4]), np.mean(data[7])],
                yerr=[scipy.stats.sem(data[1], axis=0), scipy.stats.sem(data[4], axis=0),
                      scipy.stats.sem(data[7], axis=0)], color=Settings.egocentric_color, capsize=25, linewidth=3)
    # ax.errorbar(x=[1,2,3], y=[np.mean(data[2]),  np.mean(data[5]), np.mean(data[8])], yerr=[scipy.stats.sem(data[2], axis=0), scipy.stats.sem(data[5], axis=0), scipy.stats.sem(data[8], axis=0)], color=Settings.null_color, capsize=25, linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[0]), np.mean(data[3]), np.mean(data[6])], color=Settings.allocentric_color,
            linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[1]), np.mean(data[4]), np.mean(data[7])], color=Settings.egocentric_color,
            linewidth=3)
    # ax.plot([1,2,3], [np.mean(data[2]), np.mean(data[5]), np.mean(data[8])], color=Settings.null_color, linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/rolling_percent_encooding_probe2' + suffix + '.png', dpi=300)
    plt.close()

    print("=========for position encoding trials =========")
    # do stats test
    df_hit = pd.DataFrame(
        {'trial_outcome': np.tile(0, len(p_b_hit[b_mask])), 'percentage_position_encoding': p_b_hit[b_mask],
         'unique_id': np.arange(len(p_b_hit[b_mask]))})
    df_try = pd.DataFrame(
        {'trial_outcome': np.tile(1, len(p_b_hit[b_mask])), 'percentage_position_encoding': p_b_try[b_mask],
         'unique_id': np.arange(len(p_b_hit[b_mask]))})
    df_miss = pd.DataFrame(
        {'trial_outcome': np.tile(2, len(p_b_hit[b_mask])), 'percentage_position_encoding': p_b_miss[b_mask],
         'unique_id': np.arange(len(p_b_hit[b_mask]))})
    df = pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True);
    df = pd.concat([df, df_try], ignore_index=True);
    df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for beacoend trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_position_encoding', within='trial_outcome', subject='unique_id', data=df,
                      detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(p_b_hit[b_mask]) - 1))

    # do stats test
    df_hit = pd.DataFrame(
        {'trial_outcome': np.tile(0, len(p_nb_hit[nb_mask])), 'percentage_position_encoding': p_nb_hit[nb_mask],
         'unique_id': np.arange(len(p_nb_hit[nb_mask]))})
    df_try = pd.DataFrame(
        {'trial_outcome': np.tile(1, len(p_nb_hit[nb_mask])), 'percentage_position_encoding': p_nb_try[nb_mask],
         'unique_id': np.arange(len(p_nb_hit[nb_mask]))})
    df_miss = pd.DataFrame(
        {'trial_outcome': np.tile(2, len(p_nb_hit[nb_mask])), 'percentage_position_encoding': p_nb_miss[nb_mask],
         'unique_id': np.arange(len(p_nb_hit[nb_mask]))})
    df = pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True);
    df = pd.concat([df, df_try], ignore_index=True);
    df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for non_beaconed trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_position_encoding', within='trial_outcome', subject='unique_id', data=df,
                      detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(p_nb_hit[nb_mask]) - 1))

    # do stats test
    df_hit = pd.DataFrame(
        {'trial_outcome': np.tile(0, len(p_p_hit[p_mask])), 'percentage_position_encoding': p_p_hit[p_mask],
         'unique_id': np.arange(len(p_p_hit[p_mask]))})
    df_try = pd.DataFrame(
        {'trial_outcome': np.tile(1, len(p_p_hit[p_mask])), 'percentage_position_encoding': p_p_try[p_mask],
         'unique_id': np.arange(len(p_p_hit[p_mask]))})
    df_miss = pd.DataFrame(
        {'trial_outcome': np.tile(2, len(p_p_hit[p_mask])), 'percentage_position_encoding': p_p_miss[p_mask],
         'unique_id': np.arange(len(p_p_hit[p_mask]))})
    df = pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True);
    df = pd.concat([df, df_try], ignore_index=True);
    df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for probe trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_position_encoding', within='trial_outcome', subject='unique_id', data=df,
                      detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(p_p_hit[p_mask]) - 1))

    print("=========for distance encoding trials =========")
    # do stats test
    df_hit = pd.DataFrame(
        {'trial_outcome': np.tile(0, len(d_b_hit[b_mask])), 'percentage_distance_encoding': d_b_hit[b_mask],
         'unique_id': np.arange(len(d_b_hit[b_mask]))})
    df_try = pd.DataFrame(
        {'trial_outcome': np.tile(1, len(d_b_hit[b_mask])), 'percentage_distance_encoding': d_b_try[b_mask],
         'unique_id': np.arange(len(d_b_hit[b_mask]))})
    df_miss = pd.DataFrame(
        {'trial_outcome': np.tile(2, len(d_b_hit[b_mask])), 'percentage_distance_encoding': d_b_miss[b_mask],
         'unique_id': np.arange(len(d_b_hit[b_mask]))})
    df = pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True);
    df = pd.concat([df, df_try], ignore_index=True);
    df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for beacoend trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_distance_encoding', within='trial_outcome', subject='unique_id', data=df,
                      detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(d_b_hit[b_mask]) - 1))

    # do stats test
    df_hit = pd.DataFrame(
        {'trial_outcome': np.tile(0, len(d_nb_hit[nb_mask])), 'percentage_distance_encoding': d_nb_hit[nb_mask],
         'unique_id': np.arange(len(d_nb_hit[nb_mask]))})
    df_try = pd.DataFrame(
        {'trial_outcome': np.tile(1, len(d_nb_hit[nb_mask])), 'percentage_distance_encoding': d_nb_try[nb_mask],
         'unique_id': np.arange(len(d_nb_hit[nb_mask]))})
    df_miss = pd.DataFrame(
        {'trial_outcome': np.tile(2, len(d_nb_hit[nb_mask])), 'percentage_distance_encoding': d_nb_miss[nb_mask],
         'unique_id': np.arange(len(d_nb_hit[nb_mask]))})
    df = pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True);
    df = pd.concat([df, df_try], ignore_index=True);
    df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for non_beaconed trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_distance_encoding', within='trial_outcome', subject='unique_id', data=df,
                      detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(d_nb_hit[nb_mask]) - 1))

    # do stats test
    df_hit = pd.DataFrame(
        {'trial_outcome': np.tile(0, len(d_p_hit[p_mask])), 'percentage_distance_encoding': d_p_hit[p_mask],
         'unique_id': np.arange(len(d_p_hit[p_mask]))})
    df_try = pd.DataFrame(
        {'trial_outcome': np.tile(1, len(d_p_hit[p_mask])), 'percentage_distance_encoding': d_p_try[p_mask],
         'unique_id': np.arange(len(d_p_hit[p_mask]))})
    df_miss = pd.DataFrame(
        {'trial_outcome': np.tile(2, len(d_p_hit[p_mask])), 'percentage_distance_encoding': d_p_miss[p_mask],
         'unique_id': np.arange(len(d_p_hit[p_mask]))})
    df = pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True);
    df = pd.concat([df, df_try], ignore_index=True);
    df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for probe trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_distance_encoding', within='trial_outcome', subject='unique_id', data=df,
                      detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(d_p_hit[p_mask]) - 1))

    print("=========for aperiodic encoding trials =========")
    # do stats test
    df_hit = pd.DataFrame(
        {'trial_outcome': np.tile(0, len(n_b_hit[b_mask])), 'percentage_null_encoding': n_b_hit[b_mask],
         'unique_id': np.arange(len(n_b_hit[b_mask]))})
    df_try = pd.DataFrame(
        {'trial_outcome': np.tile(1, len(n_b_hit[b_mask])), 'percentage_null_encoding': n_b_try[b_mask],
         'unique_id': np.arange(len(n_b_hit[b_mask]))})
    df_miss = pd.DataFrame(
        {'trial_outcome': np.tile(2, len(n_b_hit[b_mask])), 'percentage_null_encoding': n_b_miss[b_mask],
         'unique_id': np.arange(len(n_b_hit[b_mask]))})
    df = pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True);
    df = pd.concat([df, df_try], ignore_index=True);
    df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for beacoend trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_null_encoding', within='trial_outcome', subject='unique_id', data=df,
                      detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(n_b_hit[b_mask]) - 1))

    # do stats test
    df_hit = pd.DataFrame(
        {'trial_outcome': np.tile(0, len(n_nb_hit[nb_mask])), 'percentage_null_encoding': n_nb_hit[nb_mask],
         'unique_id': np.arange(len(n_nb_hit[nb_mask]))})
    df_try = pd.DataFrame(
        {'trial_outcome': np.tile(1, len(n_nb_hit[nb_mask])), 'percentage_null_encoding': n_nb_try[nb_mask],
         'unique_id': np.arange(len(n_nb_hit[nb_mask]))})
    df_miss = pd.DataFrame(
        {'trial_outcome': np.tile(2, len(n_nb_hit[nb_mask])), 'percentage_null_encoding': n_nb_miss[nb_mask],
         'unique_id': np.arange(len(n_nb_hit[nb_mask]))})
    df = pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True);
    df = pd.concat([df, df_try], ignore_index=True);
    df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for non_beaconed trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_null_encoding', within='trial_outcome', subject='unique_id', data=df,
                      detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(n_nb_hit[nb_mask]) - 1))

    # do stats test
    df_hit = pd.DataFrame(
        {'trial_outcome': np.tile(0, len(n_p_hit[p_mask])), 'percentage_null_encoding': n_p_hit[p_mask],
         'unique_id': np.arange(len(n_p_hit[p_mask]))})
    df_try = pd.DataFrame(
        {'trial_outcome': np.tile(1, len(n_p_hit[p_mask])), 'percentage_null_encoding': n_p_try[p_mask],
         'unique_id': np.arange(len(n_p_hit[p_mask]))})
    df_miss = pd.DataFrame(
        {'trial_outcome': np.tile(2, len(n_p_hit[p_mask])), 'percentage_null_encoding': n_p_miss[p_mask],
         'unique_id': np.arange(len(n_p_hit[p_mask]))})
    df = pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True);
    df = pd.concat([df, df_try], ignore_index=True);
    df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for probe trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_null_encoding', within='trial_outcome', subject='unique_id', data=df,
                      detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(n_p_hit[p_mask]) - 1))

    return

def get_percentage_from_global_code(df, code, tt, hmt):
    percentages = []
    for index, cell in df.iterrows():
        cell = cell.to_frame().T.reset_index(drop=True)

        behaviours = np.array(cell["behaviour_hit_try_miss"].iloc[0])
        trial_types = np.array(cell["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cell["rolling:code_global"].iloc[0])

        # mask by trial type
        tt_mask = trial_types == tt
        trial_types = trial_types[tt_mask]
        behaviours = behaviours[tt_mask]
        rolling_classifiers = rolling_classifiers[tt_mask]

        # mask by behaviour
        beh_mask = behaviours == hmt
        trial_types = trial_types[beh_mask]
        behaviours = behaviours[beh_mask]
        rolling_classifiers = rolling_classifiers[beh_mask]

        # mask by code
        code_mask = rolling_classifiers != "" # remove
        trial_types = trial_types[code_mask]
        behaviours = behaviours[code_mask]
        rolling_classifiers = rolling_classifiers[code_mask]

        print(np.unique(rolling_classifiers))
        code_mask = rolling_classifiers == code
        rolling_classifiers_masked = rolling_classifiers[code_mask]

        if len(rolling_classifiers) == 0:
            percentages.append(np.nan)
        else:
            percentage_encoding = (len(rolling_classifiers_masked)/len(rolling_classifiers))*100
            percentages.append(percentage_encoding)

    return np.array(percentages)

def plot_percentage_encoding_for_trials(combined_df, save_path="", suffix=""):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    combined_df = combined_df[combined_df["classifier"] == "G"]
    combined_df = drop_duplicate_sessions(combined_df)

    print('plotting lomb classifers proportions...')

    p_b_hit = get_percentage_from_global_code(combined_df, code="P", tt=0, hmt="hit")
    d_b_hit = get_percentage_from_global_code(combined_df, code="D", tt=0, hmt="hit")
    n_b_hit = get_percentage_from_global_code(combined_df, code="N", tt=0, hmt="hit")
    p_b_try = get_percentage_from_global_code(combined_df, code="P", tt=0, hmt="try")
    d_b_try = get_percentage_from_global_code(combined_df, code="D", tt=0, hmt="try")
    n_b_try = get_percentage_from_global_code(combined_df, code="N", tt=0, hmt="try")
    p_b_miss = get_percentage_from_global_code(combined_df, code="P", tt=0, hmt="miss")
    d_b_miss = get_percentage_from_global_code(combined_df, code="D", tt=0, hmt="miss")
    n_b_miss = get_percentage_from_global_code(combined_df, code="N", tt=0, hmt="miss")
    p_nb_hit = get_percentage_from_global_code(combined_df, code="P", tt=1, hmt="hit")
    d_nb_hit = get_percentage_from_global_code(combined_df, code="D", tt=1, hmt="hit")
    n_nb_hit = get_percentage_from_global_code(combined_df, code="N", tt=1, hmt="hit")
    p_nb_try = get_percentage_from_global_code(combined_df, code="P", tt=1, hmt="try")
    d_nb_try = get_percentage_from_global_code(combined_df, code="D", tt=1, hmt="try")
    n_nb_try = get_percentage_from_global_code(combined_df, code="N", tt=1, hmt="try")
    p_nb_miss = get_percentage_from_global_code(combined_df, code="P", tt=1, hmt="miss")
    d_nb_miss = get_percentage_from_global_code(combined_df, code="D", tt=1, hmt="miss")
    n_nb_miss = get_percentage_from_global_code(combined_df, code="N", tt=1, hmt="miss")
    p_p_hit = get_percentage_from_global_code(combined_df, code="P", tt=2, hmt="hit")
    d_p_hit = get_percentage_from_global_code(combined_df, code="D", tt=2, hmt="hit")
    n_p_hit = get_percentage_from_global_code(combined_df, code="N", tt=2, hmt="hit")
    p_p_try = get_percentage_from_global_code(combined_df, code="P", tt=2, hmt="try")
    d_p_try = get_percentage_from_global_code(combined_df, code="D", tt=2, hmt="try")
    n_p_try = get_percentage_from_global_code(combined_df, code="N", tt=2, hmt="try")
    p_p_miss = get_percentage_from_global_code(combined_df, code="P", tt=2, hmt="miss")
    d_p_miss = get_percentage_from_global_code(combined_df, code="D", tt=2, hmt="miss")
    n_p_miss = get_percentage_from_global_code(combined_df, code="N", tt=2, hmt="miss")

    b_mask = ~np.isnan(p_b_hit) & ~np.isnan(p_b_try) & ~np.isnan(p_b_miss)
    nb_mask = ~np.isnan(p_nb_hit) & ~np.isnan(p_nb_try) & ~np.isnan(p_nb_miss)
    p_mask = ~np.isnan(p_p_hit) & ~np.isnan(p_p_try) & ~np.isnan(p_p_miss)

    colors = [Settings.allocentric_color, Settings.egocentric_color, Settings.null_color,
              Settings.allocentric_color, Settings.egocentric_color, Settings.null_color,
              Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]

    data = [p_b_hit[b_mask], d_b_hit[b_mask], n_b_hit[b_mask],
            p_b_try[b_mask], d_b_try[b_mask], n_b_try[b_mask],
            p_b_miss[b_mask], d_b_miss[b_mask], n_b_miss[b_mask]]



    fig, ax = plt.subplots(figsize=(6,6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3, 5,6,7, 9,10,11], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([2,6,10])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0, right=12)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% encoding", fontsize=20)
    plt.savefig(save_path + '/rolling_percent_encooding_beaconed_from_global_code2'+suffix+'.png', dpi=300)
    plt.close()


    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[0]), np.mean(data[3]), np.mean(data[6])],
                yerr=[scipy.stats.sem(data[0], axis=0), scipy.stats.sem(data[3], axis=0),
                      scipy.stats.sem(data[6], axis=0)], color=Settings.allocentric_color, capsize=25, linewidth=3)
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[1]), np.mean(data[4]), np.mean(data[7])],
                yerr=[scipy.stats.sem(data[1], axis=0), scipy.stats.sem(data[4], axis=0),
                      scipy.stats.sem(data[7], axis=0)], color=Settings.egocentric_color, capsize=25, linewidth=3)
    # ax.errorbar(x=[1,2,3], y=[np.mean(data[2]),  np.mean(data[5]), np.mean(data[8])], yerr=[scipy.stats.sem(data[2], axis=0), scipy.stats.sem(data[5], axis=0), scipy.stats.sem(data[8], axis=0)], color=Settings.null_color, capsize=25, linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[0]), np.mean(data[3]), np.mean(data[6])], color=Settings.allocentric_color,
            linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[1]), np.mean(data[4]), np.mean(data[7])], color=Settings.egocentric_color,
            linewidth=3)
    # ax.plot([1,2,3], [np.mean(data[2]), np.mean(data[5]), np.mean(data[8])], color=Settings.null_color, linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/rolling_percent_encooding_beaconed_from_global_code' + suffix + '.png', dpi=300)
    plt.close()

    data = [p_nb_hit[nb_mask], d_nb_hit[nb_mask], n_nb_hit[nb_mask],
            p_nb_try[nb_mask], d_nb_try[nb_mask], n_nb_try[nb_mask],
            p_nb_miss[nb_mask], d_nb_miss[nb_mask], n_nb_miss[nb_mask]]

    fig, ax = plt.subplots(figsize=(6,6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3, 5,6,7, 9,10,11], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([2,6,10])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0, right=12)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% encoding", fontsize=20)
    plt.savefig(save_path + '/rolling_percent_encooding_non_beaconed_from_global_code2'+suffix+'.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[0]), np.mean(data[3]), np.mean(data[6])],
                yerr=[scipy.stats.sem(data[0], axis=0), scipy.stats.sem(data[3], axis=0),
                      scipy.stats.sem(data[6], axis=0)], color=Settings.allocentric_color, capsize=25, linewidth=3)
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[1]), np.mean(data[4]), np.mean(data[7])],
                yerr=[scipy.stats.sem(data[1], axis=0), scipy.stats.sem(data[4], axis=0),
                      scipy.stats.sem(data[7], axis=0)], color=Settings.egocentric_color, capsize=25, linewidth=3)
    # ax.errorbar(x=[1,2,3], y=[np.mean(data[2]),  np.mean(data[5]), np.mean(data[8])], yerr=[scipy.stats.sem(data[2], axis=0), scipy.stats.sem(data[5], axis=0), scipy.stats.sem(data[8], axis=0)], color=Settings.null_color, capsize=25, linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[0]), np.mean(data[3]), np.mean(data[6])], color=Settings.allocentric_color,
            linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[1]), np.mean(data[4]), np.mean(data[7])], color=Settings.egocentric_color,
            linewidth=3)
    # ax.plot([1,2,3], [np.mean(data[2]), np.mean(data[5]), np.mean(data[8])], color=Settings.null_color, linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/rolling_percent_encooding_non_beaconed_from_global_code' + suffix + '.png', dpi=300)
    plt.close()

    data = [p_p_hit[p_mask], d_p_hit[p_mask], n_p_hit[p_mask],
            p_p_try[p_mask], d_p_try[p_mask], n_p_try[p_mask],
            p_p_miss[p_mask], d_p_miss[p_mask], n_p_miss[p_mask]]

    fig, ax = plt.subplots(figsize=(6,6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3, 5,6,7, 9,10,11], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([2,6,10])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0, right=12)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% encoding", fontsize=20)
    plt.savefig(save_path + '/rolling_percent_encooding_probe_from_global_code2'+suffix+'.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[0]), np.mean(data[3]), np.mean(data[6])],
                yerr=[scipy.stats.sem(data[0], axis=0), scipy.stats.sem(data[3], axis=0),
                      scipy.stats.sem(data[6], axis=0)], color=Settings.allocentric_color, capsize=25, linewidth=3)
    ax.errorbar(x=[1, 2, 3], y=[np.mean(data[1]), np.mean(data[4]), np.mean(data[7])],
                yerr=[scipy.stats.sem(data[1], axis=0), scipy.stats.sem(data[4], axis=0),
                      scipy.stats.sem(data[7], axis=0)], color=Settings.egocentric_color, capsize=25, linewidth=3)
    # ax.errorbar(x=[1,2,3], y=[np.mean(data[2]),  np.mean(data[5]), np.mean(data[8])], yerr=[scipy.stats.sem(data[2], axis=0), scipy.stats.sem(data[5], axis=0), scipy.stats.sem(data[8], axis=0)], color=Settings.null_color, capsize=25, linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[0]), np.mean(data[3]), np.mean(data[6])], color=Settings.allocentric_color,
            linewidth=3)
    ax.plot([1, 2, 3], [np.mean(data[1]), np.mean(data[4]), np.mean(data[7])], color=Settings.egocentric_color,
            linewidth=3)
    # ax.plot([1,2,3], [np.mean(data[2]), np.mean(data[5]), np.mean(data[8])], color=Settings.null_color, linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/rolling_percent_encooding_probe_from_global_code' + suffix + '.png', dpi=300)
    plt.close()
    return

def get_first_stops(stop_locations, stop_trial_numbers, trial_numbers):
    first_stop_locations = []
    for tn in trial_numbers:
        stops_on_trial = stop_locations[stop_trial_numbers == tn]
        if len(stops_on_trial) == 0:
            first_stop_locations.append(np.nan)
        else:
            first_stop_locations.append(stops_on_trial[0])
    return np.array(first_stop_locations)

def plot_position_coding_predictions_of_first_stop_locations(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    combined_df = combined_df[combined_df["classifier"] == "G"]
    lags = np.arange(-100, 100, 1)

    for tt in [0,1,2]:
        true_scores = []
        shuffled_scores = []
        for session_id in np.unique(combined_df["session_id_vr"]):
            session_df = combined_df[combined_df["session_id_vr"] == session_id]

            trial_types = np.array(session_df["behaviour_trial_types"].iloc[0])
            trial_numbers = np.array(session_df["behaviour_trial_numbers"].iloc[0])
            rolling_classifiers = np.array(session_df["rolling:code_global"].iloc[0])
            firing_rates = pandas_collumn_to_2d_numpy_array(session_df["fr_binned_in_space_smoothed"])
            firing_rates = scipy.stats.zscore(firing_rates, axis=1, nan_policy='omit')
            firing_rates[np.isnan(firing_rates)] = 0
            stop_trial_numbers = np.array(session_df["stop_trial_numbers"].iloc[0])
            stop_locations = np.array(session_df["stop_locations"].iloc[0])
            first_stop_locations = get_first_stops(stop_locations, stop_trial_numbers, trial_numbers=np.arange(1,len(firing_rates[0,:,0])+1))

            tt_mask = trial_types == tt
            trial_numbers = trial_numbers[tt_mask]
            rolling_classifiers = rolling_classifiers[tt_mask]

            position_trial_numbers = trial_numbers[rolling_classifiers == "P"]

            if len(position_trial_numbers) >= 10:
                reference_position_trials = np.random.choice(position_trial_numbers, size=int(len(position_trial_numbers)/3))
                test_position_trials = np.array(list(set(position_trial_numbers) - set(reference_position_trials)))

                avg_firing_rates = np.nanmean(firing_rates[:,reference_position_trials-1,:], axis=1)
                avg_first_stop_location = np.nanmean(first_stop_locations[reference_position_trials-1])

                zero_padded_avg_firing_rates = np.hstack((np.zeros((len(avg_firing_rates), int(len(lags) / 2))),
                                                           avg_firing_rates,
                                                           np.zeros((len(avg_firing_rates), int(len(lags) / 2)))))
                delta_frs = []
                delta_first_stops = []
                for test_trial_number in test_position_trials:
                    test_firing_rate_map = firing_rates[:,test_trial_number-1,:]
                    test_first_stop = first_stop_locations[test_trial_number-1]
                    delta_first_stop = avg_first_stop_location-test_first_stop
                    delta_first_stops.append(delta_first_stop)

                    correlations = []
                    for i in range(len(lags)):
                        zero_padded_test_firing_rates = np.hstack((np.zeros((len(test_firing_rate_map), i)),
                                                                        test_firing_rate_map,
                                                                        np.zeros((len(test_firing_rate_map), len(lags)-i))))

                        corr = stats.pearsonr(zero_padded_avg_firing_rates.flatten(), zero_padded_test_firing_rates.flatten())[0]
                        correlations.append(corr)
                    correlations =np.array(correlations)
                    delta_fr = lags[np.argmax(correlations)]
                    delta_frs.append(delta_fr)

                delta_frs = np.array(delta_frs)
                delta_first_stops = np.array(delta_first_stops)

                true_diff = np.nanmean(delta_frs-delta_first_stops)

                # shuffle the trial order
                np.random.shuffle(delta_frs)
                shuffled_diff = np.nanmean(delta_frs-delta_first_stops)

                """
                x = delta_frs[~np.isnan(delta_frs)].reshape(-1, 1)
                y = delta_first_stops[~np.isnan(delta_first_stops)].reshape(-1, 1)
                pearson_r = stats.pearsonr(x.flatten(), y.flatten())
                R=pearson_r[0]
                true_scores.append(R)

                # shuffle the trial order
                np.random.shuffle(delta_frs)
                shuffled_diff = np.nanmean(delta_frs-delta_first_stops)

                x = delta_frs[~np.isnan(delta_frs)].reshape(-1, 1)
                y = delta_first_stops[~np.isnan(delta_first_stops)].reshape(-1, 1)
                pearson_r = stats.pearsonr(x.flatten(), y.flatten())
                R = pearson_r[0]
                shuffled_scores.append(R)"""

                true_scores.append(true_diff)
                shuffled_scores.append(shuffled_diff)
            else:
                true_scores.append(np.nan)
                shuffled_scores.append(np.nan)

        true_scores = np.array(true_scores)
        shuffled_scores = np.array(shuffled_scores)
        # make a plot here showing true vs shuffled

        x_pos = np.array([0, 1])
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.bar(x_pos, [np.nanmean(true_scores), np.nanmean(shuffled_scores)], 0.4, color="white", edgecolor="black", linewidth=3)
        ax.errorbar(x_pos, [np.nanmean(true_scores), np.nanmean(shuffled_scores)], yerr=[stats.tstd(true_scores), stats.tstd(shuffled_scores)], capsize=10,
                    color="black", linestyle='', elinewidth=3)
        for i in range(len(true_scores)):
            ax.plot(x_pos, [true_scores[i], shuffled_scores[i]], alpha=0.1,
                    color="black")

        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.set_ylim(bottom=0, top=1)
        ax.set_xlabel("", fontsize=20)
        ax.set_xticks([0, 1])
        #ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        #ax.set_yticklabels([0, 25, 50, 75, 100])
        ax.set_xticklabels(["T", "S"])
        # ax.set_ylabel("Agreement", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='both', labelsize=25)
        plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.3, right=0.87, top=0.92)
        plt.savefig(save_path + '/position_coding_predictions_of_first_stop_locations_tt'+str(tt)+'.png', dpi=300)
        plt.close()
    return


def plot_percentage_encoding_by_trial_category(combined_df, save_path="", suffix=""):
    print('plotting lomb classifers proportions...')
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    # we only want to look at the remapped coding cells (<85% encoding)
    #grid_cells = grid_cells[(grid_cells["rolling:proportion_encoding_position"] < 0.85) &
    #                        (grid_cells["rolling:proportion_encoding_distance"] < 0.85)]


    p_b_hit = get_rolling_percent_encoding(grid_cells, code="P", tt=0, hmt="hit")
    d_b_hit = get_rolling_percent_encoding(grid_cells, code="D", tt=0, hmt="hit")
    n_b_hit = get_rolling_percent_encoding(grid_cells, code="N", tt=0, hmt="hit")
    p_b_try = get_rolling_percent_encoding(grid_cells, code="P", tt=0, hmt="try")
    d_b_try = get_rolling_percent_encoding(grid_cells, code="D", tt=0, hmt="try")
    n_b_try = get_rolling_percent_encoding(grid_cells, code="N", tt=0, hmt="try")
    p_b_miss = get_rolling_percent_encoding(grid_cells, code="P", tt=0, hmt="miss")
    d_b_miss = get_rolling_percent_encoding(grid_cells, code="D", tt=0, hmt="miss")
    n_b_miss = get_rolling_percent_encoding(grid_cells, code="N", tt=0, hmt="miss")
    p_nb_hit = get_rolling_percent_encoding(grid_cells, code="P", tt=1, hmt="hit")
    d_nb_hit = get_rolling_percent_encoding(grid_cells, code="D", tt=1, hmt="hit")
    n_nb_hit = get_rolling_percent_encoding(grid_cells, code="N", tt=1, hmt="hit")
    p_nb_try = get_rolling_percent_encoding(grid_cells, code="P", tt=1, hmt="try")
    d_nb_try = get_rolling_percent_encoding(grid_cells, code="D", tt=1, hmt="try")
    n_nb_try = get_rolling_percent_encoding(grid_cells, code="N", tt=1, hmt="try")
    p_nb_miss = get_rolling_percent_encoding(grid_cells, code="P", tt=1, hmt="miss")
    d_nb_miss = get_rolling_percent_encoding(grid_cells, code="D", tt=1, hmt="miss")
    n_nb_miss = get_rolling_percent_encoding(grid_cells, code="N", tt=1, hmt="miss")
    p_p_hit = get_rolling_percent_encoding(grid_cells, code="P", tt=2, hmt="hit")
    d_p_hit = get_rolling_percent_encoding(grid_cells, code="D", tt=2, hmt="hit")
    n_p_hit = get_rolling_percent_encoding(grid_cells, code="N", tt=2, hmt="hit")
    p_p_try = get_rolling_percent_encoding(grid_cells, code="P", tt=2, hmt="try")
    d_p_try = get_rolling_percent_encoding(grid_cells, code="D", tt=2, hmt="try")
    n_p_try = get_rolling_percent_encoding(grid_cells, code="N", tt=2, hmt="try")
    p_p_miss = get_rolling_percent_encoding(grid_cells, code="P", tt=2, hmt="miss")
    d_p_miss = get_rolling_percent_encoding(grid_cells, code="D", tt=2, hmt="miss")
    n_p_miss = get_rolling_percent_encoding(grid_cells, code="N", tt=2, hmt="miss")


    p_b_hit = get_percentage_from_rolling_classification2(grid_cells, code="P", tt=0, hmt="hit")
    d_b_hit = get_percentage_from_rolling_classification2(grid_cells, code="D", tt=0, hmt="hit")
    n_b_hit = get_percentage_from_rolling_classification2(grid_cells, code="N", tt=0, hmt="hit")
    p_b_try = get_percentage_from_rolling_classification2(grid_cells, code="P", tt=0, hmt="try")
    d_b_try = get_percentage_from_rolling_classification2(grid_cells, code="D", tt=0, hmt="try")
    n_b_try = get_percentage_from_rolling_classification2(grid_cells, code="N", tt=0, hmt="try")
    p_b_miss = get_percentage_from_rolling_classification2(grid_cells, code="P", tt=0, hmt="miss")
    d_b_miss = get_percentage_from_rolling_classification2(grid_cells, code="D", tt=0, hmt="miss")
    n_b_miss = get_percentage_from_rolling_classification2(grid_cells, code="N", tt=0, hmt="miss")
    p_nb_hit = get_percentage_from_rolling_classification2(grid_cells, code="P", tt=1, hmt="hit")
    d_nb_hit = get_percentage_from_rolling_classification2(grid_cells, code="D", tt=1, hmt="hit")
    n_nb_hit = get_percentage_from_rolling_classification2(grid_cells, code="N", tt=1, hmt="hit")
    p_nb_try = get_percentage_from_rolling_classification2(grid_cells, code="P", tt=1, hmt="try")
    d_nb_try = get_percentage_from_rolling_classification2(grid_cells, code="D", tt=1, hmt="try")
    n_nb_try = get_percentage_from_rolling_classification2(grid_cells, code="N", tt=1, hmt="try")
    p_nb_miss = get_percentage_from_rolling_classification2(grid_cells, code="P", tt=1, hmt="miss")
    d_nb_miss = get_percentage_from_rolling_classification2(grid_cells, code="D", tt=1, hmt="miss")
    n_nb_miss = get_percentage_from_rolling_classification2(grid_cells, code="N", tt=1, hmt="miss")
    p_p_hit = get_percentage_from_rolling_classification2(grid_cells, code="P", tt=2, hmt="hit")
    d_p_hit = get_percentage_from_rolling_classification2(grid_cells, code="D", tt=2, hmt="hit")
    n_p_hit = get_percentage_from_rolling_classification2(grid_cells, code="N", tt=2, hmt="hit")
    p_p_try = get_percentage_from_rolling_classification2(grid_cells, code="P", tt=2, hmt="try")
    d_p_try = get_percentage_from_rolling_classification2(grid_cells, code="D", tt=2, hmt="try")
    n_p_try = get_percentage_from_rolling_classification2(grid_cells, code="N", tt=2, hmt="try")
    p_p_miss = get_percentage_from_rolling_classification2(grid_cells, code="P", tt=2, hmt="miss")
    d_p_miss = get_percentage_from_rolling_classification2(grid_cells, code="D", tt=2, hmt="miss")
    n_p_miss = get_percentage_from_rolling_classification2(grid_cells, code="N", tt=2, hmt="miss")


    b_mask = ~np.isnan(p_b_hit) & ~np.isnan(p_b_try) & ~np.isnan(p_b_miss)
    nb_mask = ~np.isnan(p_nb_hit) & ~np.isnan(p_nb_try) & ~np.isnan(p_nb_miss)
    p_mask = ~np.isnan(p_p_hit) & ~np.isnan(p_p_try) & ~np.isnan(p_p_miss)

    colors = [Settings.allocentric_color,  Settings.egocentric_color, Settings.null_color,
                 Settings.allocentric_color,  Settings.egocentric_color, Settings.null_color,
                 Settings.allocentric_color,  Settings.egocentric_color, Settings.null_color]

    data = [p_b_hit[b_mask], d_b_hit[b_mask], n_b_hit[b_mask],
            p_b_try[b_mask], d_b_try[b_mask], n_b_try[b_mask],
            p_b_miss[b_mask], d_b_miss[b_mask], n_b_miss[b_mask]]

    fig, ax = plt.subplots(figsize=(6,6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3, 5,6,7, 9,10,11], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([2,6,10])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0, right=12)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% encoding", fontsize=20)
    plt.savefig(save_path + '/rolling_percent_encooding_beaconed'+suffix+'.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(x=[1,2,3], y=[np.mean(data[0]),  np.mean(data[3]), np.mean(data[6])], yerr=[scipy.stats.sem(data[0], axis=0), scipy.stats.sem(data[3], axis=0), scipy.stats.sem(data[6], axis=0)], color=Settings.allocentric_color, capsize=25, linewidth=3)
    ax.errorbar(x=[1,2,3], y=[np.mean(data[1]),  np.mean(data[4]), np.mean(data[7])], yerr=[scipy.stats.sem(data[1], axis=0), scipy.stats.sem(data[4], axis=0), scipy.stats.sem(data[7], axis=0)], color=Settings.egocentric_color, capsize=25, linewidth=3)
    #ax.errorbar(x=[1,2,3], y=[np.mean(data[2]),  np.mean(data[5]), np.mean(data[8])], yerr=[scipy.stats.sem(data[2], axis=0), scipy.stats.sem(data[5], axis=0), scipy.stats.sem(data[8], axis=0)], color=Settings.null_color, capsize=25, linewidth=3)
    ax.plot([1,2,3], [np.mean(data[0]), np.mean(data[3]), np.mean(data[6])], color=Settings.allocentric_color, linewidth=3)
    ax.plot([1,2,3], [np.mean(data[1]), np.mean(data[4]), np.mean(data[7])], color=Settings.egocentric_color, linewidth=3)
    #ax.plot([1,2,3], [np.mean(data[2]), np.mean(data[5]), np.mean(data[8])], color=Settings.null_color, linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/rolling_percent_encooding_beaconed2'+suffix+'.png', dpi=300)
    plt.close()


    data = [p_nb_hit[nb_mask], d_nb_hit[nb_mask], n_nb_hit[nb_mask],
            p_nb_try[nb_mask], d_nb_try[nb_mask], n_nb_try[nb_mask],
            p_nb_miss[nb_mask], d_nb_miss[nb_mask], n_nb_miss[nb_mask]]

    fig, ax = plt.subplots(figsize=(6,6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3, 5,6,7, 9,10,11], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([2,6,10])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0, right=12)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% encoding", fontsize=20)
    plt.savefig(save_path + '/rolling_percent_encooding_non_beaconed'+suffix+'.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(x=[1,2,3], y=[np.mean(data[0]),  np.mean(data[3]), np.mean(data[6])], yerr=[scipy.stats.sem(data[0], axis=0), scipy.stats.sem(data[3], axis=0), scipy.stats.sem(data[6], axis=0)], color=Settings.allocentric_color, capsize=25, linewidth=3)
    ax.errorbar(x=[1,2,3], y=[np.mean(data[1]),  np.mean(data[4]), np.mean(data[7])], yerr=[scipy.stats.sem(data[1], axis=0), scipy.stats.sem(data[4], axis=0), scipy.stats.sem(data[7], axis=0)], color=Settings.egocentric_color, capsize=25, linewidth=3)
    #ax.errorbar(x=[1,2,3], y=[np.mean(data[2]),  np.mean(data[5]), np.mean(data[8])], yerr=[scipy.stats.sem(data[2], axis=0), scipy.stats.sem(data[5], axis=0), scipy.stats.sem(data[8], axis=0)], color=Settings.null_color, capsize=25, linewidth=3)
    ax.plot([1,2,3], [np.mean(data[0]), np.mean(data[3]), np.mean(data[6])], color=Settings.allocentric_color, linewidth=3)
    ax.plot([1,2,3], [np.mean(data[1]), np.mean(data[4]), np.mean(data[7])], color=Settings.egocentric_color, linewidth=3)
    #ax.plot([1,2,3], [np.mean(data[2]), np.mean(data[5]), np.mean(data[8])], color=Settings.null_color, linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/rolling_percent_encooding_non_beaconed2'+suffix+'.png', dpi=300)
    plt.close()

    data = [p_p_hit[p_mask], d_p_hit[p_mask], n_p_hit[p_mask],
            p_p_try[p_mask], d_p_try[p_mask], n_p_try[p_mask],
            p_p_miss[p_mask], d_p_miss[p_mask], n_p_miss[p_mask]]


    fig, ax = plt.subplots(figsize=(6,6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3, 5,6,7, 9,10,11], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([2,6,10])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0, right=12)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% encoding", fontsize=20)
    plt.savefig(save_path + '/rolling_percent_encooding_probe'+suffix+'.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(x=[1,2,3], y=[np.mean(data[0]),  np.mean(data[3]), np.mean(data[6])], yerr=[scipy.stats.sem(data[0], axis=0), scipy.stats.sem(data[3], axis=0), scipy.stats.sem(data[6], axis=0)], color=Settings.allocentric_color, capsize=25, linewidth=3)
    ax.errorbar(x=[1,2,3], y=[np.mean(data[1]),  np.mean(data[4]), np.mean(data[7])], yerr=[scipy.stats.sem(data[1], axis=0), scipy.stats.sem(data[4], axis=0), scipy.stats.sem(data[7], axis=0)], color=Settings.egocentric_color, capsize=25, linewidth=3)
    #ax.errorbar(x=[1,2,3], y=[np.mean(data[2]),  np.mean(data[5]), np.mean(data[8])], yerr=[scipy.stats.sem(data[2], axis=0), scipy.stats.sem(data[5], axis=0), scipy.stats.sem(data[8], axis=0)], color=Settings.null_color, capsize=25, linewidth=3)
    ax.plot([1,2,3], [np.mean(data[0]), np.mean(data[3]), np.mean(data[6])], color=Settings.allocentric_color, linewidth=3)
    ax.plot([1,2,3], [np.mean(data[1]), np.mean(data[4]), np.mean(data[7])], color=Settings.egocentric_color, linewidth=3)
    #ax.plot([1,2,3], [np.mean(data[2]), np.mean(data[5]), np.mean(data[8])], color=Settings.null_color, linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(["Hit", "Try", "Run"])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/rolling_percent_encooding_probe2'+suffix+'.png', dpi=300)
    plt.close()



    print("=========for position encoding trials =========")
    # do stats test
    df_hit = pd.DataFrame({'trial_outcome': np.tile(0,  len(p_b_hit[b_mask])),'percentage_position_encoding': p_b_hit[b_mask],'unique_id':np.arange(len(p_b_hit[b_mask]))})
    df_try = pd.DataFrame({'trial_outcome': np.tile(1,  len(p_b_hit[b_mask])),'percentage_position_encoding': p_b_try[b_mask],'unique_id':np.arange(len(p_b_hit[b_mask]))})
    df_miss = pd.DataFrame({'trial_outcome': np.tile(2,  len(p_b_hit[b_mask])),'percentage_position_encoding': p_b_miss[b_mask],'unique_id':np.arange(len(p_b_hit[b_mask]))})
    df =pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True); df = pd.concat([df, df_try], ignore_index=True); df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for beacoend trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_position_encoding', within='trial_outcome', subject='unique_id', data=df, detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(p_b_hit[b_mask])-1))

    # do stats test
    df_hit = pd.DataFrame({'trial_outcome': np.tile(0,  len(p_nb_hit[nb_mask])),'percentage_position_encoding': p_nb_hit[nb_mask],'unique_id':np.arange(len(p_nb_hit[nb_mask]))})
    df_try = pd.DataFrame({'trial_outcome': np.tile(1,  len(p_nb_hit[nb_mask])),'percentage_position_encoding': p_nb_try[nb_mask],'unique_id':np.arange(len(p_nb_hit[nb_mask]))})
    df_miss = pd.DataFrame({'trial_outcome': np.tile(2,  len(p_nb_hit[nb_mask])),'percentage_position_encoding': p_nb_miss[nb_mask],'unique_id':np.arange(len(p_nb_hit[nb_mask]))})
    df =pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True); df = pd.concat([df, df_try], ignore_index=True); df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for non_beaconed trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_position_encoding', within='trial_outcome', subject='unique_id', data=df, detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(p_nb_hit[nb_mask])-1))
 
    # do stats test
    df_hit = pd.DataFrame({'trial_outcome': np.tile(0,  len(p_p_hit[p_mask])),'percentage_position_encoding': p_p_hit[p_mask],'unique_id':np.arange(len(p_p_hit[p_mask]))})
    df_try = pd.DataFrame({'trial_outcome': np.tile(1,  len(p_p_hit[p_mask])),'percentage_position_encoding': p_p_try[p_mask],'unique_id':np.arange(len(p_p_hit[p_mask]))})
    df_miss = pd.DataFrame({'trial_outcome': np.tile(2,  len(p_p_hit[p_mask])),'percentage_position_encoding': p_p_miss[p_mask],'unique_id':np.arange(len(p_p_hit[p_mask]))})
    df =pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True); df = pd.concat([df, df_try], ignore_index=True); df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for probe trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_position_encoding', within='trial_outcome', subject='unique_id', data=df, detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(p_p_hit[p_mask])-1))

    print("=========for distance encoding trials =========")
    # do stats test
    df_hit = pd.DataFrame({'trial_outcome': np.tile(0,  len(d_b_hit[b_mask])),'percentage_distance_encoding': d_b_hit[b_mask],'unique_id':np.arange(len(d_b_hit[b_mask]))})
    df_try = pd.DataFrame({'trial_outcome': np.tile(1,  len(d_b_hit[b_mask])),'percentage_distance_encoding': d_b_try[b_mask],'unique_id':np.arange(len(d_b_hit[b_mask]))})
    df_miss = pd.DataFrame({'trial_outcome': np.tile(2,  len(d_b_hit[b_mask])),'percentage_distance_encoding': d_b_miss[b_mask],'unique_id':np.arange(len(d_b_hit[b_mask]))})
    df =pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True); df = pd.concat([df, df_try], ignore_index=True); df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for beacoend trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_distance_encoding', within='trial_outcome', subject='unique_id', data=df, detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(d_b_hit[b_mask])-1))

    # do stats test
    df_hit = pd.DataFrame({'trial_outcome': np.tile(0,  len(d_nb_hit[nb_mask])),'percentage_distance_encoding': d_nb_hit[nb_mask],'unique_id':np.arange(len(d_nb_hit[nb_mask]))})
    df_try = pd.DataFrame({'trial_outcome': np.tile(1,  len(d_nb_hit[nb_mask])),'percentage_distance_encoding': d_nb_try[nb_mask],'unique_id':np.arange(len(d_nb_hit[nb_mask]))})
    df_miss = pd.DataFrame({'trial_outcome': np.tile(2,  len(d_nb_hit[nb_mask])),'percentage_distance_encoding': d_nb_miss[nb_mask],'unique_id':np.arange(len(d_nb_hit[nb_mask]))})
    df =pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True); df = pd.concat([df, df_try], ignore_index=True); df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for non_beaconed trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_distance_encoding', within='trial_outcome', subject='unique_id', data=df, detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(d_nb_hit[nb_mask])-1))

    # do stats test
    df_hit = pd.DataFrame({'trial_outcome': np.tile(0,  len(d_p_hit[p_mask])),'percentage_distance_encoding': d_p_hit[p_mask],'unique_id':np.arange(len(d_p_hit[p_mask]))})
    df_try = pd.DataFrame({'trial_outcome': np.tile(1,  len(d_p_hit[p_mask])),'percentage_distance_encoding': d_p_try[p_mask],'unique_id':np.arange(len(d_p_hit[p_mask]))})
    df_miss = pd.DataFrame({'trial_outcome': np.tile(2,  len(d_p_hit[p_mask])),'percentage_distance_encoding': d_p_miss[p_mask],'unique_id':np.arange(len(d_p_hit[p_mask]))})
    df =pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True); df = pd.concat([df, df_try], ignore_index=True); df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for probe trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_distance_encoding', within='trial_outcome', subject='unique_id', data=df, detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(d_p_hit[p_mask])-1))


    print("=========for aperiodic encoding trials =========")
    # do stats test
    df_hit = pd.DataFrame({'trial_outcome': np.tile(0,  len(n_b_hit[b_mask])),'percentage_null_encoding': n_b_hit[b_mask],'unique_id':np.arange(len(n_b_hit[b_mask]))})
    df_try = pd.DataFrame({'trial_outcome': np.tile(1,  len(n_b_hit[b_mask])),'percentage_null_encoding': n_b_try[b_mask],'unique_id':np.arange(len(n_b_hit[b_mask]))})
    df_miss = pd.DataFrame({'trial_outcome': np.tile(2,  len(n_b_hit[b_mask])),'percentage_null_encoding': n_b_miss[b_mask],'unique_id':np.arange(len(n_b_hit[b_mask]))})
    df =pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True); df = pd.concat([df, df_try], ignore_index=True); df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for beacoend trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_null_encoding', within='trial_outcome', subject='unique_id', data=df, detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(n_b_hit[b_mask])-1))

    # do stats test
    df_hit = pd.DataFrame({'trial_outcome': np.tile(0,  len(n_nb_hit[nb_mask])),'percentage_null_encoding': n_nb_hit[nb_mask],'unique_id':np.arange(len(n_nb_hit[nb_mask]))})
    df_try = pd.DataFrame({'trial_outcome': np.tile(1,  len(n_nb_hit[nb_mask])),'percentage_null_encoding': n_nb_try[nb_mask],'unique_id':np.arange(len(n_nb_hit[nb_mask]))})
    df_miss = pd.DataFrame({'trial_outcome': np.tile(2,  len(n_nb_hit[nb_mask])),'percentage_null_encoding': n_nb_miss[nb_mask],'unique_id':np.arange(len(n_nb_hit[nb_mask]))})
    df =pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True); df = pd.concat([df, df_try], ignore_index=True); df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for non_beaconed trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_null_encoding', within='trial_outcome', subject='unique_id', data=df, detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(n_nb_hit[nb_mask])-1))

    # do stats test
    df_hit = pd.DataFrame({'trial_outcome': np.tile(0,  len(n_p_hit[p_mask])),'percentage_null_encoding': n_p_hit[p_mask],'unique_id':np.arange(len(n_p_hit[p_mask]))})
    df_try = pd.DataFrame({'trial_outcome': np.tile(1,  len(n_p_hit[p_mask])),'percentage_null_encoding': n_p_try[p_mask],'unique_id':np.arange(len(n_p_hit[p_mask]))})
    df_miss = pd.DataFrame({'trial_outcome': np.tile(2,  len(n_p_hit[p_mask])),'percentage_null_encoding': n_p_miss[p_mask],'unique_id':np.arange(len(n_p_hit[p_mask]))})
    df =pd.DataFrame()
    df = pd.concat([df, df_hit], ignore_index=True); df = pd.concat([df, df_try], ignore_index=True); df = pd.concat([df, df_miss], ignore_index=True)

    # Conduct the repeated measures ANOVA
    print("=========for probe trial outcomes =========")
    aov = pg.rm_anova(dv='percentage_null_encoding', within='trial_outcome', subject='unique_id', data=df, detailed=True)
    print(aov[['Source', 'DF', 'F', 'p-unc']])
    print("degrees of freedom: ", str(len(n_p_hit[p_mask])-1))

    return


def get_rolling_percent_encoding2(df, code="P", tt=1, hmt="hit"):
    if code == "P":
        result_column = "rolling:percentage_trials_encoding_position"
    elif code == "D":
        result_column = "rolling:percentage_trials_encoding_distance"
    elif code == "N":
        result_column = "rolling:percentage_trials_encoding_null"

    # tt and i index are shared
    i = int(tt)

    if hmt == "hit":
        j =0
    elif hmt == "try":
        j=1
    elif hmt == "miss":
        j=2

    result=[]
    for m in range(len(df)):
        if isinstance(df[result_column].iloc[m], list):
            result.append(df[result_column].iloc[m][i][j])
        else:
            print(df["session_id"].iloc[m], " is missing an entry in ", result_column)
    return np.array(result)

def plot_ROC(concantenated_dataframe, save_path=""):
    print('plotting lomb classifers proportions...')
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    grid_cells = add_ROC_stats(grid_cells)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(np.arange(0, 10), np.arange(0,10), linestyle="dashed", color="red")
    FPR = np.array(grid_cells["ROC:tt0_FPR"])
    TPR = np.array(grid_cells["ROC:tt0_TPR"])
    nan_mask = ~np.isnan(FPR) & ~np.isnan(TPR)
    ax.scatter(FPR[nan_mask], TPR[nan_mask], color="black")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='both', labelsize=25)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel("TPR", fontsize=20)
    ax.set_xlabel("FPR", fontsize=20)
    plt.savefig(save_path + '/ROC_plot_tt0.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(np.arange(0, 10), np.arange(0,10), linestyle="dashed", color="red")
    FPR = np.array(grid_cells["ROC:tt1_FPR"])
    TPR = np.array(grid_cells["ROC:tt1_TPR"])
    nan_mask = ~np.isnan(FPR) & ~np.isnan(TPR)
    ax.scatter(FPR[nan_mask], TPR[nan_mask], color="black")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='both', labelsize=25)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel("TPR", fontsize=20)
    ax.set_xlabel("FPR", fontsize=20)
    plt.savefig(save_path + '/ROC_plot_tt1.png', dpi=300)
    plt.close()
    return

def plot_percentage_hits_for_remapped_encoding_grid_cells_probe(combined_df, save_path=""):
    print('plotting lomb classifers proportions...')
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    fig, ax = plt.subplots(figsize=(6,6))

    p_b_hit = get_rolling_percent_encoding2(grid_cells, code="P", tt=0, hmt="hit")*100
    d_b_hit = get_rolling_percent_encoding2(grid_cells, code="D", tt=0, hmt="hit")*100
    n_b_hit = get_rolling_percent_encoding2(grid_cells, code="N", tt=0, hmt="hit")*100
    p_nb_hit = get_rolling_percent_encoding2(grid_cells, code="P", tt=1, hmt="hit")*100
    d_nb_hit = get_rolling_percent_encoding2(grid_cells, code="D", tt=1, hmt="hit")*100
    n_nb_hit = get_rolling_percent_encoding2(grid_cells, code="N", tt=1, hmt="hit")*100


    p_b_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=0)
    d_b_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=0)
    n_b_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=0)
    p_nb_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=1)
    d_nb_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=1)
    n_nb_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=1)
    p_p_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=2)
    d_p_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=2)
    n_p_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=2)

    b_mask = ~np.isnan(p_b_hit) & ~np.isnan(d_b_hit)
    nb_mask = ~np.isnan(p_nb_hit) & ~np.isnan(d_nb_hit)
    p_mask = ~np.isnan(p_p_hit) & ~np.isnan(d_p_hit)

    data = [p_p_hit[p_mask], d_p_hit[p_mask]]

    print("comparing % hits between position and distance encoding trials for probe trials, df=",str(len(p_p_hit[p_mask])-1), ", p= ", str(scipy.stats.wilcoxon(p_p_hit[p_mask], d_p_hit[p_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_p_hit[p_mask], d_p_hit[p_mask])[0]))

    colors = [Settings.allocentric_color, Settings.egocentric_color]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([1,2])
    ax.set_xticklabels(["P", "P"])
    ax.set_xlim(left=0, right=6)
    ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/percentage_hit_trials_in_coded_trials_probe.png', dpi=300)
    plt.close()
    return

def adjust_for_multiple_cells_in_session_for_percentage_hits(vals, cells_df):
    new_vals = []
    for session_id in np.unique(cells_df["session_id"]):
        all_vals_from_session = vals[np.asarray(cells_df["session_id"] == session_id)]
        avg_val = np.nanmean(all_vals_from_session)
        new_vals.append(avg_val)
    return np.array(new_vals)

def adjust_for_mouse_level_percentage_hits(vals, cells_df):
    new_vals = []
    for mouse_id in np.unique(cells_df["mouse"]):
        all_vals_from_session = vals[np.asarray(cells_df["mouse"] == mouse_id)]
        avg_val = np.nanmean(all_vals_from_session)
        new_vals.append(avg_val)
    return np.array(new_vals)


def plot_percentage_hits_for_remapped_encoding_grid_cells_mouse_level(combined_df, save_path="", suffix=""):

    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    fig, ax = plt.subplots(figsize=(6,6))

    p_b_hit = get_rolling_percent_encoding2(grid_cells, code="P", tt=0, hmt="hit")*100
    d_b_hit = get_rolling_percent_encoding2(grid_cells, code="D", tt=0, hmt="hit")*100
    n_b_hit = get_rolling_percent_encoding2(grid_cells, code="N", tt=0, hmt="hit")*100
    p_nb_hit = get_rolling_percent_encoding2(grid_cells, code="P", tt=1, hmt="hit")*100
    d_nb_hit = get_rolling_percent_encoding2(grid_cells, code="D", tt=1, hmt="hit")*100
    n_nb_hit = get_rolling_percent_encoding2(grid_cells, code="N", tt=1, hmt="hit")*100


    p_b_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=0)
    d_b_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=0)
    n_b_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=0)
    p_nb_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=1)
    d_nb_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=1)
    n_nb_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=1)
    p_p_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=2)
    d_p_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=2)
    n_p_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=2)

    p_b_hit = adjust_for_mouse_level_percentage_hits(p_b_hit, grid_cells)
    d_b_hit = adjust_for_mouse_level_percentage_hits(d_b_hit, grid_cells)
    p_nb_hit = adjust_for_mouse_level_percentage_hits(p_nb_hit, grid_cells)
    d_nb_hit = adjust_for_mouse_level_percentage_hits(d_nb_hit, grid_cells)

    b_mask = ~np.isnan(p_b_hit) & ~np.isnan(d_b_hit)
    nb_mask = ~np.isnan(p_nb_hit) & ~np.isnan(d_nb_hit)

    data = [p_b_hit[b_mask], d_b_hit[b_mask], p_nb_hit[nb_mask], d_nb_hit[nb_mask]]

    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    #for i in range(len(data[0])):
    #    ax.plot([1,2], [data[0][i], data[1][i]], color="black", alpha=0.3)
    #for i in range(len(data[2])):
    #    ax.plot([4,5], [data[2][i], data[3][i]], color="black", alpha=0.3)

    #for i in range(len(data[0])):
    #    ax.scatter([1,2], [data[0][i],data[1][i]], marker="o", color="black")
    #    ax.plot([1,2], [data[0][i],data[1][i]], color="black")
    #for i in range(len(data[2])):
    #    ax.scatter([4,5], [data[2][i],data[3][i]], marker="o", color="black")
    #    ax.plot([4,5], [data[2][i],data[3][i]], color="black")

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "B", "NB", "NB"])
    ax.set_xlim(left=0, right=6)
    ax.set_ylim(bottom=0, top=100)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% hit trials", fontsize=20)
    plt.savefig(save_path + '/percentage_hit_trials_in_coded_trials_animal'+suffix+'.png', dpi=300)
    plt.close()


    print("comparing % hits between position and distance encoding trials for beaconed trials at the level of mouse, df=",str(len(p_b_hit[b_mask])-1), ", p= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[0]))
    print("comparing % hits between position and distance encoding trials for non beaconed trials at the level of mouse, df=",str(len(p_nb_hit[nb_mask])-1), ", p= ",str(scipy.stats.wilcoxon(p_nb_hit[nb_mask], d_nb_hit[nb_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_nb_hit[nb_mask], d_nb_hit[nb_mask])[0]))

    return

def plot_p_hit_for_remapped_encoding_grid_cells(combined_df, save_path="", suffix=""):
    print('plotting lomb classifers proportions...')
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    for tt in [0,1,2]:
        fig, ax = plt.subplots(1,1, figsize=(3.5,6))

        p_b_hit = get_rolling_percent_encoding2(grid_cells, code="P", tt=tt, hmt="hit")
        d_b_hit = get_rolling_percent_encoding2(grid_cells, code="D", tt=tt, hmt="hit")
        n_b_hit = get_rolling_percent_encoding2(grid_cells, code="N", tt=tt, hmt="hit")

        p_b_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=tt)/100
        d_b_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=tt)/100
        n_b_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=tt)/100

        p_b_hit = adjust_for_multiple_cells_in_session_for_percentage_hits(p_b_hit, grid_cells)
        d_b_hit = adjust_for_multiple_cells_in_session_for_percentage_hits(d_b_hit, grid_cells)
        b_mask = ~np.isnan(p_b_hit) & ~np.isnan(d_b_hit)

        data = [p_b_hit[b_mask], d_b_hit[b_mask]]

        colors = [Settings.allocentric_color, Settings.egocentric_color]

        boxprops = dict(linewidth=3, color='k')
        medianprops = dict(linewidth=3, color='k')
        capprops = dict(linewidth=3, color='k')
        whiskerprops = dict(linewidth=3, color='k')
        box = ax.boxplot(data, positions=[1,2], widths=1, boxprops=boxprops, medianprops=medianprops,
                         whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
        for i, x in enumerate([1,2]):
            ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        #ax.set_yticks([0,0.5,1])
        ax.set_xticks([1,2])
        ax.set_xticklabels(["B", "B"])
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(left=0.4, right=2.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='both', labelsize=25)
        #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.subplots_adjust(left = 0.2)
        #fig.tight_layout()
        #plt.subplots_adjust(left=0.25, bottom=0.2)
        #ax.set_ylabel("% hit trials", fontsize=20)
        plt.savefig(save_path + '/percentage_hit_trials_in_coded_trials'+suffix+'_tt'+str(tt)+'.png', dpi=300)
        plt.close()

        print("comparing % hits between position and distance encoding trials for tt "+str(tt)+" trials, df=",str(len(p_b_hit[b_mask])-1), ", p= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[0]))

    return

def plot_percentage_hits_for_remapped_encoding_non_grid_cells(combined_df, save_path="", suffix=""):
    print('plotting lomb classifers proportions...')
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    fig, ax = plt.subplots(figsize=(8,6))

    p_b_hit = get_rolling_percent_encoding2(non_grid_cells, code="P", tt=0, hmt="hit")*100
    d_b_hit = get_rolling_percent_encoding2(non_grid_cells, code="D", tt=0, hmt="hit")*100
    n_b_hit = get_rolling_percent_encoding2(non_grid_cells, code="N", tt=0, hmt="hit")*100
    p_nb_hit = get_rolling_percent_encoding2(non_grid_cells, code="P", tt=1, hmt="hit")*100
    d_nb_hit = get_rolling_percent_encoding2(non_grid_cells, code="D", tt=1, hmt="hit")*100
    n_nb_hit = get_rolling_percent_encoding2(non_grid_cells, code="N", tt=1, hmt="hit")*100


    p_b_hit = get_percentage_from_rolling_classification(non_grid_cells, code="P", tt=0)
    d_b_hit = get_percentage_from_rolling_classification(non_grid_cells, code="D", tt=0)
    n_b_hit = get_percentage_from_rolling_classification(non_grid_cells, code="N", tt=0)
    p_nb_hit = get_percentage_from_rolling_classification(non_grid_cells, code="P", tt=1)
    d_nb_hit = get_percentage_from_rolling_classification(non_grid_cells, code="D", tt=1)
    n_nb_hit = get_percentage_from_rolling_classification(non_grid_cells, code="N", tt=1)
    p_p_hit = get_percentage_from_rolling_classification(non_grid_cells, code="P", tt=2)
    d_p_hit = get_percentage_from_rolling_classification(non_grid_cells, code="D", tt=2)
    n_p_hit = get_percentage_from_rolling_classification(non_grid_cells, code="N", tt=2)


    b_mask = ~np.isnan(p_b_hit) & ~np.isnan(d_b_hit)
    nb_mask = ~np.isnan(p_nb_hit) & ~np.isnan(d_nb_hit)
    p_mask = ~np.isnan(p_p_hit) & ~np.isnan(d_p_hit)

    data = [p_b_hit[b_mask], d_b_hit[b_mask], p_nb_hit[nb_mask], d_nb_hit[nb_mask], p_p_hit[p_mask], d_p_hit[p_mask]]

    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color, Settings.allocentric_color, Settings.egocentric_color]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5,7,8], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5,7,8]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([1,2,4,5,7,8])
    ax.set_xticklabels(["B", "B", "NB", "NB", "P", "P"])
    ax.set_xlim(left=0, right=9)
    #ax.set_ylim(bottom=0, top=100)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% hit trials", fontsize=20)
    plt.savefig(save_path + '/percentage_hit_trials_in_coded_trials'+suffix+'_non_grid_cells.png', dpi=300)
    plt.close()


    print("comparing % hits between position and distance encoding trials for beaconed trials non_grid_cells, df=",str(len(p_b_hit[b_mask])-1), ", p= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[0]))
    print("comparing % hits between position and distance encoding trials for non beaconed trials non_grid_cells, df=",str(len(p_nb_hit[nb_mask])-1), ", p= ",str(scipy.stats.wilcoxon(p_nb_hit[nb_mask], d_nb_hit[nb_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_nb_hit[nb_mask], d_nb_hit[nb_mask])[0]))
    print("comparing % hits between position and distance encoding trials for probe trials non_grid_cells, df=",str(len(p_p_hit[p_mask])-1), ", p= ",str(scipy.stats.wilcoxon(p_p_hit[p_mask], d_p_hit[p_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_p_hit[p_mask], d_p_hit[p_mask])[0]))

    return



#

def plot_percentage_hits_for_trials(combined_df, save_path="", suffix=""):
    print('plotting lomb classifers proportions...')
    #grid_cells = combined_df[combined_df["classifier"] == "G"]
    cells = combined_df
    cells = drop_duplicate_sessions(cells)
    fig, ax = plt.subplots(figsize=(8,6))

    p_b_hit = get_percentage_from_rolling_classification(cells, code="P", tt=0, rolling_classsifier_collumn_name="rolling:code_global")
    d_b_hit = get_percentage_from_rolling_classification(cells, code="D", tt=0, rolling_classsifier_collumn_name="rolling:code_global")
    n_b_hit = get_percentage_from_rolling_classification(cells, code="N", tt=0, rolling_classsifier_collumn_name="rolling:code_global")
    p_nb_hit = get_percentage_from_rolling_classification(cells, code="P", tt=1, rolling_classsifier_collumn_name="rolling:code_global")
    d_nb_hit = get_percentage_from_rolling_classification(cells, code="D", tt=1, rolling_classsifier_collumn_name="rolling:code_global")
    n_nb_hit = get_percentage_from_rolling_classification(cells, code="N", tt=1, rolling_classsifier_collumn_name="rolling:code_global")
    p_p_hit = get_percentage_from_rolling_classification(cells, code="P", tt=2, rolling_classsifier_collumn_name="rolling:code_global")
    d_p_hit = get_percentage_from_rolling_classification(cells, code="D", tt=2, rolling_classsifier_collumn_name="rolling:code_global")
    n_p_hit = get_percentage_from_rolling_classification(cells, code="N", tt=2, rolling_classsifier_collumn_name="rolling:code_global")

    b_mask = ~np.isnan(p_b_hit) & ~np.isnan(d_b_hit)
    nb_mask = ~np.isnan(p_nb_hit) & ~np.isnan(d_nb_hit)
    p_mask = ~np.isnan(p_p_hit) & ~np.isnan(d_p_hit)

    data = [p_b_hit[b_mask], d_b_hit[b_mask], p_nb_hit[nb_mask], d_nb_hit[nb_mask], p_p_hit[p_mask], d_p_hit[p_mask]]

    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color, Settings.allocentric_color, Settings.egocentric_color]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5,7,8], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5,7,8]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=25)
    ax.set_xticks([1,2,4,5,7,8])
    ax.set_xticklabels(["B", "B", "NB", "NB", "P", "P"])
    ax.set_xlim(left=0, right=9)
    plt.savefig(save_path + '/percentage_hit_trials_encoding_position_and_distance.png', dpi=300)
    plt.close()

    print("comparing % hits between position and distance encoding trials for beaconed trials, df=",str(len(p_b_hit[b_mask])-1), ", p= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[0]))
    print("comparing % hits between position and distance encoding trials for non beaconed trials, df=",str(len(p_nb_hit[nb_mask])-1), ", p= ",str(scipy.stats.wilcoxon(p_nb_hit[nb_mask], d_nb_hit[nb_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_nb_hit[nb_mask], d_nb_hit[nb_mask])[0]))
    print("comparing % hits between position and distance encoding trials for probe trials, df=",str(len(p_p_hit[p_mask])-1), ", p= ",str(scipy.stats.wilcoxon(p_p_hit[p_mask], d_p_hit[p_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_p_hit[p_mask], d_p_hit[p_mask])[0]))
    return

def plot_percentage_hits_for_remapped_encoding_grid_cells(combined_df, save_path="", suffix=""):
    print('plotting lomb classifers proportions...')
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    fig, ax = plt.subplots(figsize=(8,6))

    p_b_hit = get_rolling_percent_encoding2(grid_cells, code="P", tt=0, hmt="hit")*100
    d_b_hit = get_rolling_percent_encoding2(grid_cells, code="D", tt=0, hmt="hit")*100
    n_b_hit = get_rolling_percent_encoding2(grid_cells, code="N", tt=0, hmt="hit")*100
    p_nb_hit = get_rolling_percent_encoding2(grid_cells, code="P", tt=1, hmt="hit")*100
    d_nb_hit = get_rolling_percent_encoding2(grid_cells, code="D", tt=1, hmt="hit")*100
    n_nb_hit = get_rolling_percent_encoding2(grid_cells, code="N", tt=1, hmt="hit")*100


    p_b_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=0)
    d_b_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=0)
    n_b_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=0)
    p_nb_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=1)
    d_nb_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=1)
    n_nb_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=1)
    p_p_hit = get_percentage_from_rolling_classification(grid_cells, code="P", tt=2)
    d_p_hit = get_percentage_from_rolling_classification(grid_cells, code="D", tt=2)
    n_p_hit = get_percentage_from_rolling_classification(grid_cells, code="N", tt=2)

    #p_b_hit = adjust_for_multiple_cells_in_session_for_percentage_hits(p_b_hit, grid_cells)
    #d_b_hit = adjust_for_multiple_cells_in_session_for_percentage_hits(d_b_hit, grid_cells)
    #p_nb_hit = adjust_for_multiple_cells_in_session_for_percentage_hits(p_nb_hit, grid_cells)
    #d_nb_hit = adjust_for_multiple_cells_in_session_for_percentage_hits(d_nb_hit, grid_cells)
    #p_p_hit = adjust_for_multiple_cells_in_session_for_percentage_hits(p_p_hit, grid_cells)
    #d_p_hit = adjust_for_multiple_cells_in_session_for_percentage_hits(d_p_hit, grid_cells)

    b_mask = ~np.isnan(p_b_hit) & ~np.isnan(d_b_hit)
    nb_mask = ~np.isnan(p_nb_hit) & ~np.isnan(d_nb_hit)
    p_mask = ~np.isnan(p_p_hit) & ~np.isnan(d_p_hit)

    data = [p_b_hit[b_mask], d_b_hit[b_mask], p_nb_hit[nb_mask], d_nb_hit[nb_mask], p_p_hit[p_mask], d_p_hit[p_mask]]

    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color, Settings.allocentric_color, Settings.egocentric_color]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5,7,8], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5,7,8]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    #for i in range(len(data[0])):
    #    ax.plot([1,2], [data[0][i], data[1][i]], color="black", alpha=0.3)
    #for i in range(len(data[2])):
    #    ax.plot([4,5], [data[2][i], data[3][i]], color="black", alpha=0.3)

    #for i in range(len(data[0])):
    #    ax.scatter([1,2], [data[0][i],data[1][i]], marker="o", color="black")
    #    ax.plot([1,2], [data[0][i],data[1][i]], color="black")
    #for i in range(len(data[2])):
    #    ax.scatter([4,5], [data[2][i],data[3][i]], marker="o", color="black")
    #    ax.plot([4,5], [data[2][i],data[3][i]], color="black")

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([1,2,4,5,7,8])
    ax.set_xticklabels(["B", "B", "NB", "NB", "P", "P"])
    ax.set_xlim(left=0, right=9)
    #ax.set_ylim(bottom=0, top=100)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% hit trials", fontsize=20)
    plt.savefig(save_path + '/percentage_hit_trials_in_coded_trials'+suffix+'.png', dpi=300)
    plt.close()


    print("comparing % hits between position and distance encoding trials for beaconed trials, df=",str(len(p_b_hit[b_mask])-1), ", p= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_b_hit[b_mask], d_b_hit[b_mask])[0]))
    print("comparing % hits between position and distance encoding trials for non beaconed trials, df=",str(len(p_nb_hit[nb_mask])-1), ", p= ",str(scipy.stats.wilcoxon(p_nb_hit[nb_mask], d_nb_hit[nb_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_nb_hit[nb_mask], d_nb_hit[nb_mask])[0]))
    print("comparing % hits between position and distance encoding trials for probe trials, df=",str(len(p_p_hit[p_mask])-1), ", p= ",str(scipy.stats.wilcoxon(p_p_hit[p_mask], d_p_hit[p_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(p_p_hit[p_mask], d_p_hit[p_mask])[0]))

    return

def plot_lomb_classifiers_proportions_by_nonbeaconed_hmt(concantenated_dataframe, save_path=""):
    print('plotting lomb classifers proportions...')
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]

    fig, ax = plt.subplots(figsize=(4,6))
    groups = ["Position", "Distance", "Null"]
    colors_lm = [Settings.allocentric_color,  Settings.egocentric_color, Settings.null_color, "black"]
    suffixes = ["_all_nonbeaconed", "_nonbeaconed_hits", "_nonbeaconed_tries", "_nonbeaconed_misses"]
    labels = ["all", "hits", "tries", "runs"]
    x_pos = np.arange(len(suffixes))
    for suffix, x in zip(suffixes, x_pos):
        bottom=0
        for color, group in zip(colors_lm, groups):
            c_df = add_lomb_classifier(grid_cells, suffix=suffix)
            c_df = c_df[c_df["Lomb_classifier_"+suffix] != "Unclassified"]

            count = len(c_df[(c_df["Lomb_classifier_"+suffix] == group)])
            percent = (count/len(c_df))*100
            ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
            ax.text(x,bottom+0.5, str(count), color="k", fontsize=15, ha="center")
            bottom = bottom+percent
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=-45, ha="left", fontsize=25, rotation_mode='anchor')
    plt.ylabel("Percent of neurons", fontsize=25)
    plt.xlim((-0.5, len(labels)-0.5))
    plt.ylim((0,100))
    plt.xticks(rotation = -45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.4, bottom=0.2)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(save_path + '/lomb_classifiers_proportions_non_beaconed.png', dpi=200)
    plt.close()
    return

def plot_lomb_classifiers_proportions_by_hits_tt(concantenated_dataframe, save_path=""):
    print('plotting lomb classifers proportions...')
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]

    fig, ax = plt.subplots(figsize=(4,6))
    groups = ["Position", "Distance", "Null"]
    colors_lm = [Settings.allocentric_color,  Settings.egocentric_color, Settings.null_color, "black"]
    suffixes = ["_all_hits", "_beaconed_hits", "_nonbeaconed_hits", "_probe_hits"]
    labels = ["all", "beaconed", "non-beaconed", "probe"]
    x_pos = np.arange(len(suffixes))
    for suffix, x in zip(suffixes, x_pos):
        bottom=0
        for color, group in zip(colors_lm, groups):
            c_df = add_lomb_classifier(grid_cells, suffix=suffix)
            c_df = c_df[c_df["Lomb_classifier_"+suffix] != "Unclassified"]

            count = len(c_df[(c_df["Lomb_classifier_"+suffix] == group)])
            percent = (count/len(c_df))*100
            ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
            ax.text(x,bottom+0.5, str(count), color="k", fontsize=15, ha="center")
            bottom = bottom+percent
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=-45, ha="left", fontsize=25, rotation_mode='anchor')
    plt.ylabel("Percent of neurons",  fontsize=25)
    plt.xlim((-0.5, len(suffixes)-0.5))
    plt.ylim((0,100))
    plt.xticks(rotation = -45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.4, bottom=0.2)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(save_path + '/lomb_classifiers_proportions_hits.png', dpi=200)
    plt.close()
    return

def plot_lomb_classifiers_proportions(concantenated_dataframe, suffix="", save_path=""):
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix=suffix)
    concantenated_dataframe["Lomb_classifier_"+suffix] = concantenated_dataframe["Lomb_classifier_"+suffix].replace(["Unclassified"], "Null")

    #concantenated_dataframe = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] != "Unclassified"]

    print('plotting lomb classifers proportions...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    fig, ax = plt.subplots(figsize=(4,6))
    groups = ["Position", "Distance", "Null", "Unclassified"]
    groups = ["Position", "Distance", "Null"]
    colors_lm = [Settings.allocentric_color,  Settings.egocentric_color, Settings.null_color, "black"]
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
            ax.text(x,bottom+0.5, str(count), color="k", fontsize=15, ha="center")
            bottom = bottom+percent
    plt.xticks(x_pos, objects, fontsize=15)
    plt.ylabel("Percent of neurons",  fontsize=25)
    plt.xlim((-0.5, len(objects)-0.5))
    plt.ylim((0,100))
    ax.set_yticks([0,25,50,75,100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.tight_layout()
    plt.subplots_adjust(left=0.4)
    ax.tick_params(axis='both', which='major', labelsize=30)
    plt.savefig(save_path + '/lomb_classifiers_proportions_'+suffix+'.png', dpi=200)
    plt.close()

    # plot by pie chart
    groups = ["Position", "Distance", "Null"]
    colors_lm = [Settings.allocentric_color,  Settings.egocentric_color, Settings.null_color, "black"]
    objects = ["G", "NG"]
    for object, x in zip(objects, x_pos):
        if object == "G":
            df = grid_cells
        elif object == "NG":
            df = non_grid_cells
        sizes = [len(df[(df["Lomb_classifier_"+suffix] == "Position")]),
                 len(df[(df["Lomb_classifier_"+suffix] == "Distance")]),
                 len(df[(df["Lomb_classifier_"+suffix] == "Null")])]
        fig1, ax1 = plt.subplots(figsize=(4,4))
        ax1.pie(sizes, labels=groups, autopct='%1.1f%%', startangle=90, colors=colors_lm, pctdistance=1.2, labeldistance=1.5)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig(save_path + '/lomb_classifiers_proportions_'+suffix+'_pie_'+object+'.png', dpi=1000)
        plt.close()
    return

def add_lomb_classifier_for_hmt(spatial_firing):
    frequency = Settings.frequency

    for hmt in ["hit", "try", "miss"]:
        for tt in [0,1,2]:
            lomb_classifiers = []
            for index, row in spatial_firing.iterrows():
                lomb_SNR_threshold = row["power_threshold"]
                spatial_periodogram = row[str(hmt)+"_spatial_periodogram_tt"+str(tt)]
                if spatial_periodogram is not None and np.shape(spatial_periodogram)!=(0,):
                    #print(np.shape(spatial_periodogram))
                    avg_power = np.nanmean(spatial_periodogram, axis=0)

                    max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
                    lomb_classifier = get_lomb_classifier(max_SNR, max_SNR_freq, lomb_SNR_threshold, 0.05, numeric=False)
                    lomb_classifiers.append(lomb_classifier)
                else:
                    lomb_classifiers.append(None)

            spatial_firing["Lomb_classifier_"+str(hmt)+"_tt"+str(tt)] = lomb_classifiers
    return spatial_firing


def plot_lomb_classifiers_proportions_hmt(concantenated_dataframe, save_path=""):
    '''
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix="PI")
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix="PI_try")
    concantenated_dataframe = add_lomb_classifier(concantenated_dataframe, suffix="PI_miss")

    concantenated_dataframe = add_lomb_classifier_for_hmt(concantenated_dataframe)
    print('plotting lomb classifers proportions...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    for df, c in zip([grid_cells, non_grid_cells], ["gc", "ngc"]):
        fig, ax = plt.subplots(figsize=(4,6))
        groups = ["Position", "Distance", "Null"]
        colors_lm = [Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
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
    '''

    concantenated_dataframe = add_lomb_classifier_for_hmt(concantenated_dataframe)
    print('plotting lomb classifers proportions...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    # second try
    for df, c in zip([grid_cells, non_grid_cells], ["gc", "ngc"]):
        for tt in [0,1,2]:
            null_mask = np.invert(np.array(df["Lomb_classifier_hit_tt"+str(tt)].isnull())) & \
                        np.invert(np.array(df["Lomb_classifier_try_tt"+str(tt)].isnull())) & \
                        np.invert(np.array(df["Lomb_classifier_miss_tt"+str(tt)].isnull()))

            tt_df = df[null_mask]

            fig, ax = plt.subplots(figsize=(4,6))
            groups = ["Position", "Distance", "Null"]
            colors_lm = [Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
            objects = ["hit", "try", "miss"]
            x_pos = np.arange(len(objects))

            for object, x in zip(objects, x_pos):
                collumn = "Lomb_classifier_"+str(objects[x])+"_tt"+str(tt)
                bottom=0

                for color, group in zip(colors_lm, groups):
                    count = len(tt_df[(tt_df[collumn] == group)])
                    percent = (count/len(tt_df))*100
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
                plt.savefig(save_path + '/lomb_classifiers_proportions_hmt2_'+c+'_tt'+str(tt)+'.png', dpi=200)
            elif c == "ngc":
                plt.savefig(save_path + '/lomb_classifiers_proportions_hmt2_'+c+'_tt'+str(tt)+'.png', dpi=200)
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


def extract_hit_success(df):
    b=[];nb=[];p=[]
    for index, cell in df.iterrows():
        cell = cell.to_frame().T.reset_index(drop=True)
        b_hit = cell["percentage_hits"].iloc[0][0]
        nb_hit = cell["percentage_hits"].iloc[0][1]
        p_hit = cell["percentage_hits"].iloc[0][2]
        b.append(b_hit);nb.append(nb_hit); p.append(p_hit)
    df["percentage_b_hits"] = b
    df["percentage_nb_hits"] = nb
    df["percentage_p_hits"] = p
    return df

def add_session_number(df):
    session_number = []
    for index, cell in df.iterrows():
        cell = cell.to_frame().T.reset_index(drop=True)
        session_id = cell["session_id"].iloc[0]
        D = str(session_id.split("_")[1].split("D")[-1])
        session_number.append(D)
    df["session_number"] = session_number
    return df


def plot_lomb_classifiers_proportions_by_hit_success2(concantenated_dataframe, suffix="", save_path=""):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    grid_cells = add_lomb_classifier(grid_cells, suffix=suffix)
    grid_cells = add_session_number(grid_cells)
    grid_cells = extract_hit_success(grid_cells)
    p_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    d_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]
    n_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Null"]

    groups = ["Position", "Distance", "Null"]
    colors_lm = ["turquoise", "orange", "gray"]

    objects = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    obj_lim = [(0,10), (10,20), (20,30), (30,40), (40,50), (50,60), (60,70), (70,80), (80,90), (90,100)]
    objects = ["0-25", "25-50", "50-75", "75-100"]
    obj_lim = [(0,25), (25,50), (50,75), (75,100)]
    x_pos = np.arange(len(objects))

    for tt in [0,1]:
        fig, ax = plt.subplots(figsize=(6,6))
        for group_grid_cell, color in zip([p_grid_cells, d_grid_cells, n_grid_cells], [Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]):

            percentage_over_limits = []
            for limits in obj_lim:
                if tt ==0:
                    column = "percentage_b_hits"
                elif tt == 1:
                    column = "percentage_nb_hits"

                percentage_cells = (len(group_grid_cell[((group_grid_cell[column]>limits[0]) & (group_grid_cell[column]<limits[1]))])/\
                                   len(grid_cells[((grid_cells[column]>limits[0]) & (grid_cells[column]<limits[1]))]))*100
                percentage_over_limits.append(percentage_cells)
            percentage_over_limits = np.array(percentage_over_limits)
            ax.plot(x_pos, percentage_over_limits, "o-", color=color, linewidth=2)

        plt.xticks(x_pos, objects, fontsize=15)
        plt.yticks(fontsize=20)
        plt.ylabel("Percent of neurons",  fontsize=25)
        plt.xlabel("Percentage hit trials\n in session",  fontsize=25)
        plt.xlim((-0.5, len(objects)-0.5))
        plt.ylim((0,100))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.tight_layout()
        plt.subplots_adjust(left=0.4, bottom=0.2)
        #ax.tick_params(axis='both', which='major', labelsize=25)
        plt.savefig(save_path + "/lomb_classifiers_proportions_by_hits_tt_"+str(tt)+".png", dpi=200)
        plt.close()
    return

def plot_lomb_classifiers_proportions_by_hit_success(concantenated_dataframe, suffix="", save_path=""):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    grid_cells = add_lomb_classifier(grid_cells, suffix=suffix)
    grid_cells = add_session_number(grid_cells)
    grid_cells = extract_hit_success(grid_cells)

    groups = ["Position", "Distance", "Null"]
    colors_lm = [Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]

    objects = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    obj_lim = [(0,10), (10,20), (20,30), (30,40), (40,50), (50,60), (60,70), (70,80), (80,90), (90,100)]
    objects = ["0-25", "25-50", "50-75", "75-100"]
    obj_lim = [(0,25), (25,50), (50,75), (75,100)]
    x_pos = np.arange(len(objects))

    for tt in [0,1]:
        fig, ax = plt.subplots(figsize=(6,6))
        for object, limits, x in zip(objects, obj_lim, x_pos):
            if tt ==0:
                df = grid_cells[((grid_cells["percentage_b_hits"]>limits[0]) & (grid_cells["percentage_b_hits"]<limits[1]))]
            elif tt == 1:
                df = grid_cells[((grid_cells["percentage_nb_hits"]>limits[0]) & (grid_cells["percentage_nb_hits"]<limits[1]))]

            bottom=0
            for color, group in zip(colors_lm, groups):
                count = len(df[(df["Lomb_classifier_"] == group)])
                if count>0:
                    percent = (count/len(df))*100
                    ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
                    ax.text(x,bottom+2, str(count), color="k", fontsize=10, ha="center")
                    bottom = bottom+percent

        plt.xticks(x_pos, objects, fontsize=15)
        plt.yticks(fontsize=25)
        plt.ylabel("Percent of neurons",  fontsize=25)
        plt.xlabel("Percentage hit trials\n in session",  fontsize=25)
        plt.xlim((-0.5, len(objects)-0.5))
        plt.ylim((0,100))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.tight_layout()
        plt.subplots_adjust(left=0.4, bottom=0.2)
        #ax.tick_params(axis='both', which='major', labelsize=25)
        plt.savefig(save_path + "/lomb_classifiers_proportions_by_hits_tt_"+str(tt)+".png", dpi=200)
        plt.close()


        fig, ax = plt.subplots(figsize=(6,6))
        scalar=40
        for session_id in np.unique(grid_cells["session_id"]):
            grid_cell_session = grid_cells[grid_cells["session_id"]==session_id]
            grid_cell_session_number = int(grid_cell_session["session_number"].iloc[0])
            if tt==0:
                grid_cell_percentage = grid_cell_session["percentage_b_hits"].iloc[0]
            elif tt==1:
                grid_cell_percentage = grid_cell_session["percentage_nb_hits"].iloc[0]

            n_p = len(grid_cell_session[grid_cell_session["Lomb_classifier_"] == "Position"])
            n_d = len(grid_cell_session[grid_cell_session["Lomb_classifier_"] == "Distance"])
            n_n = len(grid_cell_session[grid_cell_session["Lomb_classifier_"] == "Null"])

            if n_p>0:
                jitter_x, jitter_y = getjitterr(scalar=1,grid_cell_percentage=grid_cell_percentage, grid_cell_session_number=grid_cell_session_number, grid_cells=grid_cells)
                ax.scatter(grid_cell_session_number+jitter_x, grid_cell_percentage+jitter_y, s=n_p*scalar, marker="o", alpha=0.6, color=Settings.allocentric_color, edgecolor='none', clip_on=False)
            if n_d>0:
                jitter_x, jitter_y = getjitterr(scalar=1,grid_cell_percentage=grid_cell_percentage, grid_cell_session_number=grid_cell_session_number, grid_cells=grid_cells)
                ax.scatter(grid_cell_session_number+jitter_x, grid_cell_percentage+jitter_y, s=n_d*scalar, marker="o", alpha=0.6, color=Settings.egocentric_color, edgecolor='none', clip_on=False)
            if n_n>0:
                jitter_x, jitter_y = getjitterr(scalar=1,grid_cell_percentage=grid_cell_percentage, grid_cell_session_number=Settings.null_color, grid_cells=grid_cells)
                ax.scatter(grid_cell_session_number+jitter_x, grid_cell_percentage+jitter_y, s=n_n*scalar, marker="o", alpha=0.6, color="gray", edgecolor='none', clip_on=False)

        ax.set_ylabel("Percentage hit trials\n in session",  fontsize=25)
        plt.xlabel("Session number",  fontsize=25)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        ax.set_xticks([0,10, 20, 30, 40])
        plt.ylim(0,100)
        plt.xlim(0, max(np.asarray(grid_cells["session_number"]).astype(np.int64)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.subplots_adjust(left=0.4, bottom=0.2)
        #ax.tick_params(axis='both', which='major', labelsize=25)
        plt.savefig(save_path + "/lomb_classifiers_session_by_hits_session_tt_"+str(tt)+".png", dpi=200)
        plt.close()
    return

def getjitterr(scalar,grid_cell_percentage, grid_cell_session_number,grid_cells):
    jitter_x=np.random.uniform(low=-0.3, high=0.3)*scalar
    jitter_y=np.random.uniform(low=-0.6, high=0.6)*scalar
    if grid_cell_percentage == 0:
        jitter_y=0
    if (grid_cell_session_number == max(np.asarray(grid_cells["session_number"]).astype(np.int64))):
        jitter_x=0
    return jitter_x, jitter_y

def plot_regression(ax, x, y, c, y_text_pos, x_test_min=None, x_test_max=None):
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

    if x_test_min is None:
        x_test_min = min(x)
    if x_test_max is None:
        x_test_max = max(x)

    x_test = np.linspace(x_test_min, x_test_max, 100)

    Y_pred = linear_regressor.predict(x_test.reshape(-1, 1))  # make predictions
    #ax.text(6, 0.65, "R= "+str(np.round(pearson_r[0], decimals=2))+ ", p = "+str(np.round(pearson_r[1], decimals=2)))

    ax.text(  # position text relative to Axes
        0.05, y_text_pos, "R= "+str(np.round(pearson_r[0], decimals=2))+ ", p = "+str(np.round(pearson_r[1], decimals=4)),
        ha='left', va='top', color=c,
        transform=ax.transAxes, fontsize=10)

    ax.plot(x_test, Y_pred, color=c)

def get_lomb_color(modal_class):
    if modal_class == "Position":
        return Settings.allocentric_color
    elif modal_class == "Distance":
        return Settings.egocentric_color
    elif modal_class == "Null":
        return Settings.null_color
    else:
        return "black"

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

def plot_allo_vs_ego_power(spike_data, processed_position_data, output_path, track_length):
    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = Settings.frequency_step
    frequency = Settings.frequency
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        modal_frequency = cluster_spike_data['ML_Freqs'].iloc[0]
        modal_class = cluster_spike_data['Lomb_classifier_'].iloc[0]

        fig, axes = plt.subplots(1, 3, figsize=(9,3), sharey=True)
        subset_trial_numbers = np.asarray(processed_position_data["trial_number"])

        if len(firing_times_cluster)>1:
            if len(subset_trial_numbers)>0:
                subset_mask = np.isin(centre_trials, subset_trial_numbers)
                subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                subset_powers = powers.copy()
                subset_powers[subset_mask == False] = np.nan
                avg_subset_powers = np.nanmean(subset_powers, axis=0)
                allocentric_peak_freq, allocentric_peak_power, allo_i = get_allocentric_peak(frequency, avg_subset_powers, tolerance=0.05)
                egocentric_peak_freq, egocentric_peak_power, ego_i = get_egocentric_peak(frequency, avg_subset_powers, tolerance=0.05)

                for ax, hmt in zip(axes, ["hit", "miss", "try"]):
                    subset_processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hmt)]
                    subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])
                    subset_mask = np.isin(centre_trials, subset_trial_numbers)
                    subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                    subset_powers = powers.copy()
                    subset_powers[subset_mask == False] = np.nan

                    allo_powers = subset_powers[:, allo_i]
                    ego_powers = subset_powers[:, ego_i]
                    ax.scatter(allo_powers, ego_powers, color=get_hmt_color(hmt), marker="o", alpha=0.1)

                    ax.set_xlabel('Allocentric Power', fontsize=20, labelpad = 10)
                    if hmt=="hit":
                        ax.set_ylabel("Egocentric Power", fontsize=20, labelpad = 10)
                    #ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
                    #ax.set_xticks([0, np.round(ax.get_xlim()[1], 2)])
                    ax.set_ylim(bottom=0, top=np.nanmax(subset_powers))
                    ax.set_xlim(left=0, right=np.nanmax(subset_powers))
                    ax.yaxis.set_ticks_position('left')
                    ax.xaxis.set_ticks_position('bottom')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.xaxis.set_tick_params(labelsize=20)
                    ax.yaxis.set_tick_params(labelsize=20)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_allo_vs_ego_power_' + str(cluster_id) + '.png', dpi=300)
        plt.close()
    return

def get_modal_color(modal_class):
    if modal_class == "Position":
        return Settings.allocentric_color
    elif modal_class == "Distance":
        return Settings.egocentric_color
    elif modal_class == "Null":
        return Settings.null_color

def plot_snr(spike_data, processed_position_data, output_path, track_length):
    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = Settings.frequency_step
    frequency = Settings.frequency
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        modal_frequency = cluster_spike_data['ML_Freqs'].iloc[0]
        modal_class = cluster_spike_data['Lomb_classifier_'].iloc[0]
        modal_class_color = get_modal_color(modal_class)

        stops_on_track = plt.figure(figsize=(6,6))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for f in range(1,6):
            ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
        ax.axvline(x=modal_frequency, color=modal_class_color, linewidth=3,linestyle="solid")

        subset_trial_numbers = np.asarray(processed_position_data["trial_number"])

        if len(firing_times_cluster)>1:
            if len(subset_trial_numbers)>0:
                subset_mask = np.isin(centre_trials, subset_trial_numbers)
                subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                subset_powers = powers.copy()
                subset_powers[subset_mask == False] = np.nan
                avg_subset_powers = np.nanmean(subset_powers, axis=0)
                sem_subset_powers = scipy.stats.sem(subset_powers, axis=0, nan_policy="omit")
                ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color="black", alpha=0.3)
                ax.plot(frequency, avg_subset_powers, color="black", linewidth=3)
                #ax.scatter(frequency[np.argmax(avg_subset_powers)], avg_subset_powers[np.argmax(avg_subset_powers)], color=get_trial_color(tt), marker="v")
                allocentric_peak_freq, allocentric_peak_power, allo_i = get_allocentric_peak(frequency, avg_subset_powers, tolerance=0.05)
                egocentric_peak_freq, egocentric_peak_power, ego_i = get_egocentric_peak(frequency, avg_subset_powers, tolerance=0.05)
                #ax.scatter(allocentric_peak_freq, allocentric_peak_power, color="turquoise", marker="x", zorder=10)
                #ax.scatter(egocentric_peak_freq, egocentric_peak_power, color="orange", marker="x", zorder=10)

        ax.axhline(y=Settings.measured_far, color="red", linewidth=3, linestyle="dashed")
        plt.ylabel('Periodic Power', fontsize=25, labelpad = 10)
        plt.xlabel("Track Frequency", fontsize=25, labelpad = 10)
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
        plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_moving_lomb_scargle_avg_periodogram_Cluster_' + str(cluster_id) + '.png', dpi=300)
        plt.close()
    return

def plot_snr_by_hmt_tt(spike_data, processed_position_data, output_path, track_length):
    print('plotting the power by hmt...')
    save_path = output_path + '/Figures/moving_lomb_power_by_hmt'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = Settings.frequency_step
    frequency = Settings.frequency
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        modal_frequency = cluster_spike_data['ML_Freqs'].iloc[0]
        modal_class = cluster_spike_data['Lomb_classifier_'].iloc[0]
        modal_class_color = get_modal_color(modal_class)

        stops_on_track = plt.figure(figsize=(6,6))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for f in range(1,6):
            ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
        ax.axvline(x=modal_frequency, color=modal_class_color, linewidth=3,linestyle="solid")

        for tt, tt_color in zip([0, 1], ["black", "blue"]):
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
                        ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color=tt_color, alpha=0.3)
                        ax.plot(frequency, avg_subset_powers, color=tt_color)
                        #ax.scatter(frequency[np.argmax(avg_subset_powers)], avg_subset_powers[np.argmax(avg_subset_powers)], color=get_trial_color(tt), marker="v")

        plt.ylabel('Periodic Power', fontsize=25, labelpad = 10)
        plt.xlabel("Track Frequency", fontsize=25, labelpad = 10)
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

    return local_min_left, local_min_right

def plot_snr_peaks_by_hmt(spike_data, processed_position_data, output_path, track_length):
    print('plotting the power by hmt...')
    save_path = output_path + '/Figures/moving_lomb_power_by_hmt'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    gauss_kernel = Gaussian1DKernel(stddev=3)
    step = Settings.frequency_step
    frequency = Settings.frequency
    processed_position_data = processed_position_data[(processed_position_data["trial_type"] == 1)]

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        centre_trials = np.round(centre_trials).astype(np.int64)
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        modal_frequency = cluster_spike_data['ML_Freqs'].iloc[0]
        modal_class = cluster_spike_data['Lomb_classifier_'].iloc[0]

        fig, axes = plt.subplots(3, 1, figsize=(6,6), sharex=True)
        for ax, hmt, ylab in zip(axes, ["hit", "try", "miss"], ["Hit", "Try", "Run"]):
            for f in range(1,6):
                ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
            ax.axvline(x=modal_frequency, color="black", linewidth=2,linestyle="solid")

            subset_processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hmt)]
            subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])

            if len(firing_times_cluster)>1:
                if len(subset_trial_numbers)>0:
                    subset_mask = np.isin(centre_trials, subset_trial_numbers)
                    subset_mask = np.vstack([subset_mask]*len(powers[0])).T
                    subset_powers = powers.copy()
                    subset_powers[subset_mask == False] = np.nan
                    nan_row_mask = np.all(np.isnan(subset_powers), axis=1)
                    subset_powers = subset_powers[~nan_row_mask]
                    freq_peaks = frequency[np.nanargmax(subset_powers, axis=1)]
                    freq_powers = np.nanmax(subset_powers, axis=1)
                    hist, bin_edges = np.histogram(freq_peaks, bins=len(frequency)+1, range=[min(frequency), max(frequency)])
                    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
                    smoothened_hist = convolve(hist, gauss_kernel)

                    local_maxima_idx, _ = signal.find_peaks(smoothened_hist, height=0, distance=5)
                    for i in local_maxima_idx:
                    #    min_left_i, min_right_i = find_neighbouring_minima(smoothened_hist, i)
                        ax.scatter(bin_centres[i], smoothened_hist[i], color="black", marker="x", zorder=10)
                    #    ax.scatter(bin_centres[min_left_i], smoothened_hist[min_left_i], color="blue", marker="x", zorder=10)
                    #    ax.scatter(bin_centres[min_right_i], smoothened_hist[min_right_i], color="blue", marker="x", zorder=10)

                    #ax.bar(bin_centres, hist, width=np.diff(bin_centres)[0], color=get_hmt_color(hmt))
                    ax.plot(bin_centres, smoothened_hist, color=get_hmt_color(hmt), linewidth=3, zorder=0)

                    ax.set_ylabel(ylab, fontsize=20, labelpad = 10)
                    ax.set_xlim([0, 5.05])
                    ax.set_xticks([1,2,3,4,5])
                    ax.set_xticklabels(["", "", "", "", ""])
                    ax.set_yticks([0, int(ax.get_ylim()[1])])
                    ax.set_ylim(bottom=0)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.yaxis.set_tick_params(labelsize=20)

        ax.set_xlabel("Track Frequency", fontsize=20, labelpad = 10)
        ax.set_xticklabels(["1", "2", "3", "4", "5"])
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(fontsize=20)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/hmt_peak_histogram_'+str(cluster_id)+'.png', dpi=200)
        plt.close()
    return

def plot_snr_by_hmt(spike_data, processed_position_data, output_path, track_length):
    print('plotting the power by hmt...')
    save_path = output_path + '/Figures/moving_lomb_power_by_hmt'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = Settings.frequency_step
    frequency = Settings.frequency
    processed_position_data = processed_position_data[(processed_position_data["trial_type"] == 1)]

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        centre_trials = np.round(centre_trials).astype(np.int64)
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        modal_frequency = cluster_spike_data['ML_Freqs'].iloc[0]
        modal_class = cluster_spike_data['Lomb_classifier_'].iloc[0]
        modal_class_color = get_modal_color(modal_class)

        stops_on_track = plt.figure(figsize=(6,6))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for f in range(1,6):
            ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
        ax.axvline(x=modal_frequency, color=modal_class_color, linewidth=3,linestyle="solid")

        for hmt in ["hit", "miss", "try"]:
            subset_processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hmt)]
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
                    #ax.scatter(frequency[np.argmax(avg_subset_powers)], avg_subset_powers[np.argmax(avg_subset_powers)], color=get_hmt_color(hmt), marker="v")

        plt.ylabel('Periodic Power', fontsize=25, labelpad = 10)
        plt.xlabel("Track Frequency", fontsize=25, labelpad = 10)
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


def plot_hit_power_by_tt(spike_data, processed_position_data, output_path, track_length):
    print('plotting the hit power by tt...')
    save_path = output_path + '/Figures/moving_lomb_power_by_hmt'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == "hit")]

    step = Settings.frequency_step
    frequency = Settings.frequency

    b_nb_p_ = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        modal_frequency = cluster_spike_data['ML_Freqs'].iloc[0]

        if len(firing_times_cluster)>0:

            powers[np.isnan(powers)] = 0
            SNRs = []
            TTs = []
            for trial in processed_position_data["trial_number"]:
                trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial]
                tt = trial_processed_position_data["trial_type"].iloc[0]
                trial_powers = powers[centre_trials == trial]
                avg_powers = np.nanmean(trial_powers, axis=0)
                max_SNR = avg_powers[frequency == modal_frequency][0]
                SNRs.append(max_SNR)
                TTs.append(tt)
            SNRs=np.array(SNRs)
            TTs=np.array(TTs)
            TTs = TTs[~np.isnan(SNRs)]
            SNRs = SNRs[~np.isnan(SNRs)]

            fig, ax = plt.subplots(figsize=(4,4))
            ax.set_ylabel("Trial Power", fontsize=30, labelpad=10)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            objects = ["Cue", "PI"]
            x_pos = np.arange(len(objects))

            if len(SNRs[TTs==0])>1:
                ax.errorbar(x_pos[0], np.nanmean(SNRs[TTs==0]), yerr=stats.sem(SNRs[TTs==0], nan_policy='omit'), ecolor='black', capsize=10, fmt='none', color="black", elinewidth=2)
                ax.bar(x_pos[0], np.nanmean(SNRs[TTs==0]), edgecolor="black", color="None", facecolor="black", linewidth=2, width=0.75, alpha=0.7)
            if len(SNRs[TTs==1])>1:
                ax.errorbar(x_pos[1], np.nanmean(SNRs[TTs==1]), yerr=stats.sem(SNRs[TTs==1], nan_policy='omit'), ecolor='black', capsize=10, fmt='none', color="black", elinewidth=2)
                ax.bar(x_pos[1], np.nanmean(SNRs[TTs==1]), edgecolor="black", color="None", facecolor="blue", linewidth=2, width=0.75)

            if (len(SNRs[TTs==0])>0) and (len(SNRs[TTs==1])>0):
                b_nb_p = stats.ttest_ind(SNRs[TTs==0], SNRs[TTs==1])[1]
            else:
                b_nb_p = np.nan

            plt.xticks(x_pos, objects, fontsize=30)
            ax.set_xlim([-0.6, 1.6])
            ax.set_ylim([0,0.3])
            ax.set_yticks([0, 0.1, 0.2, 0.3])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(save_path + '/hit_powers_by_tt_test_'+str(cluster_id)+'.png', dpi=200)
            plt.close()

            b_nb_p_.append(b_nb_p)
        else:
            b_nb_p_.append(np.nan)

    spike_data["hit_power_test_b_nb_p"] = b_nb_p_
    return spike_data

def plot_power_by_hmt(spike_data, processed_position_data, output_path, track_length):
    print('plotting the power by hmt...')
    save_path = output_path + '/Figures/moving_lomb_power_by_hmt'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data = processed_position_data[(processed_position_data["trial_type"] == 1) | (processed_position_data["trial_type"] == 2)]

    step = Settings.frequency_step
    frequency = Settings.frequency

    hit_miss_p_ = []
    hit_try_p_ = []
    try_miss_p_ = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        modal_frequency = cluster_spike_data['ML_Freqs'].iloc[0]

        if len(firing_times_cluster)>0:

            powers[np.isnan(powers)] = 0
            SNRs = []
            HMTs = []
            for trial in processed_position_data["trial_number"]:
                trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial]
                hmt = trial_processed_position_data["hit_miss_try"].iloc[0]
                trial_powers = powers[centre_trials == trial]
                avg_powers = np.nanmean(trial_powers, axis=0)
                max_SNR = avg_powers[frequency == modal_frequency][0]
                SNRs.append(max_SNR)
                HMTs.append(hmt)
            SNRs=np.array(SNRs)
            HMTs=np.array(HMTs)
            HMTs = HMTs[~np.isnan(SNRs)]
            SNRs = SNRs[~np.isnan(SNRs)]

            fig, ax = plt.subplots(figsize=(4,4))
            ax.set_ylabel("Trial Power", fontsize=30, labelpad=10)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            objects = ["Hit", "Try", "Run"]
            x_pos = np.arange(len(objects))

            pts = np.linspace(0, np.pi * 2, 24)
            circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
            vert = np.r_[circ, circ[::-1] * .7]
            open_circle = mpl.path.Path(vert)

            if len(SNRs[HMTs=="hit"])>1:
                ax.errorbar(x_pos[0], np.nanmean(SNRs[HMTs=="hit"]), yerr=stats.sem(SNRs[HMTs=="hit"], nan_policy='omit'), ecolor='black', capsize=10, fmt='none', color="black", elinewidth=2)
                ax.bar(x_pos[0], np.nanmean(SNRs[HMTs=="hit"]), edgecolor="black", color="None", facecolor="green", linewidth=2, width=0.75)
            if len(SNRs[HMTs=="try"])>1:
                ax.errorbar(x_pos[1], np.nanmean(SNRs[HMTs=="try"]), yerr=stats.sem(SNRs[HMTs=="try"], nan_policy='omit'), ecolor='black', capsize=10, fmt='none', color="black", elinewidth=2)
                ax.bar(x_pos[1], np.nanmean(SNRs[HMTs=="try"]), edgecolor="black", color="None", facecolor="orange", linewidth=2, width=0.75)
            if len(SNRs[HMTs=="miss"])>1:
                ax.errorbar(x_pos[2], np.nanmean(SNRs[HMTs=="miss"]), yerr=stats.sem(SNRs[HMTs=="miss"], nan_policy='omit'), ecolor='black', capsize=10, fmt='none', color="black", elinewidth=2)
                ax.bar(x_pos[2], np.nanmean(SNRs[HMTs=="miss"]), edgecolor="black", color="None", facecolor="red", linewidth=2, width=0.75)

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

            #all_behaviour = []; all_behaviour.extend(SNRs[HMTs=="hit"].tolist()); all_behaviour.extend(SNRs[HMTs=="try"].tolist()); all_behaviour.extend(SNRs[HMTs=="miss"].tolist())
            #significance_bar(start=x_pos[0], end=x_pos[1], height=np.nanmax(all_behaviour)+0, displaystring=get_p_text(hit_try_p))
            #significance_bar(start=x_pos[1], end=x_pos[2], height=np.nanmax(all_behaviour)+0.1, displaystring=get_p_text(try_miss_p))
            #significance_bar(start=x_pos[0], end=x_pos[2], height=np.nanmax(all_behaviour)+0.2, displaystring=get_p_text(hit_miss_p))

            plt.xticks(x_pos, objects, fontsize=30)
            ax.set_xlim([-0.6, 2.6])
            ax.set_ylim([0,0.3])
            ax.set_yticks([0, 0.1, 0.2, 0.3])
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

    step = Settings.frequency_step
    frequency = Settings.frequency

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])


        stops_on_track = plt.figure(figsize=(6,6))
        ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for f in range(1,6):
            ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)

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


def plot_peak_histogram(spike_data, processed_position_data, output_path, track_length):
    spike_data = add_lomb_classifier(spike_data)
    print('plotting joint cell correlations...')
    save_path = output_path + '/Figures/lomb_peak_histograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    step = Settings.frequency_step
    frequency = Settings.frequency
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

def filter_by_n_pi_trials_by_hmt(spike_data, n_trials=5):
    if n_trials ==0:
        return spike_data
    new = pd.DataFrame()
    for index, cluster_spike_data in spike_data.iterrows():
        cluster_spike_data = cluster_spike_data.to_frame().T.reset_index(drop=True)
        n_trials_h = cluster_spike_data["n_pi_trials_by_hmt"].iloc[0][0]
        n_trials_m = cluster_spike_data["n_pi_trials_by_hmt"].iloc[0][1]

        if (n_trials_h>=n_trials) & (n_trials_m>=n_trials):
            new = pd.concat([new, cluster_spike_data], ignore_index=True)
    return new

def plot_mean_firing_rates_vr_vs_of(concatenated_dataframe, save_path):
    grid_cells = concatenated_dataframe[concatenated_dataframe["classifier"] == "G"]
    non_grid_cells = concatenated_dataframe[concatenated_dataframe["classifier"] != "G"]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot([0,50], [0,50], color="black", linestyle="dashed", linewidth=2)
    for group in ["Position", "Distance", "Null"]:
        grids = grid_cells[grid_cells["Lomb_classifier_"] == group]
        for index, grid in grids.iterrows():
            grid = grid.to_frame().T.reset_index(drop=True)
            ax.scatter(grid["mean_firing_rate_vr"].iloc[0], grid["mean_firing_rate_of"].iloc[0], color=get_lomb_color(grid["Lomb_classifier_"].iloc[0]), marker="o", alpha=0.5)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    r = stats.pearsonr(grid_cells["mean_firing_rate_vr"], grid_cells["mean_firing_rate_of"])[0]
    ax.text(x=1, y=12, s="R = "+str(np.round(r, decimals=2)), color="black", fontsize=25)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xlim(left=0, right=15)
    ax.set_ylim(bottom=0, top=15)
    ax.set_xticks([0,5,10,15])
    ax.set_yticks([0,5,10,15])
    #ax.set_xticks([-1, 0, 1])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("VR Mean Rate (Hz)", fontsize=25)
    ax.set_ylabel("OF Mean Rate (Hz)", fontsize=25)
    plt.savefig(save_path + '/mean_firing_rate_vr_vs_of.png', dpi=300)
    plt.close()
    return

def plot_proportion_significant_to_trial_outcome(concatenated_dataframe, save_path):
    grid_cells = concatenated_dataframe[concatenated_dataframe["classifier"] == "G"]
    non_grid_cells = concatenated_dataframe[concatenated_dataframe["classifier"] != "G"]

    for group in ["Position", "Distance", "Null"]:
        grids = grid_cells[grid_cells["Lomb_classifier_"] == group]

        proportion_modulated =[]
        n_trials=[20]
        for n in n_trials:
            grids = filter_by_n_pi_trials_by_hmt(grids, n_trials=n)

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
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.savefig(save_path + '/proportion_significant_modulated_by_trial_outcome_'+group+'.png', dpi=200)
            plt.close()

            proportion_modulated.append(len(sig_hm[sig_hm==0])/(len(sig_hm)))

        fig1, ax1 = plt.subplots()
        ax1.plot(n_trials, proportion_modulated)
        ax1.set_ylim([0,1])
        #plt.savefig(save_path + '/proportion_significant_modulated_by_trial_outcome_'+group+'_by_n_trials.png', dpi=200)
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

def plot_firing_rate_maps_hits_between_trial_types(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == "hit")]

    # raw_position_data is needed for this
    #spike_data = add_position_x(spike_data, raw_position_data)
    #spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length)
    #spike_data = bin_fr_in_time(spike_data, raw_position_data)

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

            for tt, c in zip([0, 1], ["black", "blue"]):
                tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
                tt_trial_numbers = np.asarray(tt_processed_position_data["trial_number"])
                tt_fr_binned_in_space = fr_binned_in_space[tt_trial_numbers-1]
                ax.fill_between(fr_binned_in_space_bin_centres, np.nanmean(tt_fr_binned_in_space, axis=0)-stats.sem(tt_fr_binned_in_space, axis=0), np.nanmean(tt_fr_binned_in_space, axis=0)+stats.sem(tt_fr_binned_in_space, axis=0), color=c, alpha=0.3)
                ax.plot(fr_binned_in_space_bin_centres, np.nanmean(tt_fr_binned_in_space, axis=0), color=c)

                hmt_max = max(np.nanmean(tt_fr_binned_in_space, axis=0)+stats.sem(tt_fr_binned_in_space, axis=0))
                y_max = max([y_max, hmt_max])
                y_max = np.ceil(y_max)

            plt.ylabel('Firing Rate (Hz)', fontsize=20, labelpad = 20)
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
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_hits_by_trial_type_' + str(cluster_id) + '.png', dpi=300)
            plt.close()

    return spike_data

def plot_firing_rate_maps_hmt(spike_data, processed_position_data, output_path, track_length):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    processed_position_data = processed_position_data[(processed_position_data["trial_type"] == 1)]

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

            plt.ylabel('Firing Rate (Hz)', fontsize=20, labelpad = 20)
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

    return spike_data


def plot_firing_rate_maps_per_trial_by_tt(spike_data, processed_position_data, output_path, track_length, hmts):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data_ = pd.DataFrame()
    for hmt in hmts:
        processed_position_data_ = pd.concat([processed_position_data_, processed_position_data[processed_position_data["hit_miss_try"] == hmt]], ignore_index=True)
    processed_position_data = processed_position_data_
    string_hmts = [str(int) for int in hmts]
    string_hmts = "-".join(string_hmts)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data["fr_binned_in_space"].iloc[cluster_index])
            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)

            fig, axes = plt.subplots(3, 1, figsize=(6,6), sharex=True)
            for ax, tt, color, ytitle in zip(axes, [2,0,1], ["red", "black", "blue"], ["PI", "Cued", "PI"]):
                tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
                if len(tt_processed_position_data)>1:
                    hmt_trial_numbers = pandas_collumn_to_numpy_array(tt_processed_position_data["trial_number"])
                    hmt_cluster_firing_maps = cluster_firing_maps[hmt_trial_numbers-1]
                    locations = np.arange(0, len(hmt_cluster_firing_maps[0]))
                    ordered = np.arange(0, len(tt_processed_position_data), 1)
                    X, Y = np.meshgrid(locations, ordered)
                    cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
                    ax.pcolormesh(X, Y, hmt_cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
                if len(tt_processed_position_data)>0:
                    Edmond.plot_utility2.style_vr_plot(ax, len(tt_processed_position_data))
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.set_yticks([len(tt_processed_position_data)-1])
                ax.set_yticklabels([len(tt_processed_position_data)])
                ax.set_ylabel(ytitle, fontsize=25, labelpad = 15)
                plt.xlabel('Location (cm)', fontsize=25, labelpad = 20)
                ax.set_xlim([0, track_length])
                ax.set_ylim([0, len(tt_processed_position_data)-1])
            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            fig.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '_hmt_'+string_hmts+'.png', dpi=300)
            plt.close()
    return

def get_avg_correlation(firing_rate_map):
    corrs=[]
    for i in range(len(firing_rate_map)):
        for j in range(len(firing_rate_map)):
            if i!=j:
                corr = scipy.stats.pearsonr(firing_rate_map[i], firing_rate_map[j])[0]
                corrs.append(corr)
    return np.nanmean(corrs)

def get_reconstructed_trial_signal(recovered_shift, trial_firing_rate_map, min_shift, max_shift):
    zero_pad_1 = np.zeros(np.abs(min_shift - recovered_shift))
    zero_pad_2 = np.zeros(np.abs(max_shift - recovered_shift))
    return np.concatenate((zero_pad_1, trial_firing_rate_map, zero_pad_2))

def plot_realignment_matrix(spike_data, processed_position_data, output_path, track_length, trial_types):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/joint_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data_ = pd.DataFrame()
    for tt in trial_types:
        processed_position_data_ = pd.concat([processed_position_data_, processed_position_data[processed_position_data["trial_type"] == tt]], ignore_index=True)
    processed_position_data = processed_position_data_
    string_tts = [str(int) for int in trial_types]
    string_tts = "-".join(string_tts)

    ids = pandas_collumn_to_numpy_array(spike_data["cluster_id"])

    for m, hmt in enumerate(["hit", "try", "miss"]):
        hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
        hmt_trial_numbers = pandas_collumn_to_numpy_array(hmt_processed_position_data["trial_number"])
        if len(hmt_processed_position_data)>0:

            cross_correlations = np.zeros((len(ids), len(ids)))
            for i in range(len(cross_correlations)):
                for j in range(len(cross_correlations[0])):
                    cluster_j_df = spike_data[spike_data["cluster_id"] == ids[j]]
                    cluster_i_df = spike_data[spike_data["cluster_id"] == ids[i]]
                    shifts_i = cluster_i_df["realignment_by_hmt"].iloc[0][m]

                    cluster_firing_maps_j = np.array(cluster_j_df["fr_binned_in_space"].iloc[0])
                    where_are_NaNs2 = np.isnan(cluster_firing_maps_j)
                    cluster_firing_maps_j[where_are_NaNs2] = 0
                    cluster_firing_maps_j = min_max_normalize(cluster_firing_maps_j)
                    hmt_cluster_firing_maps_j = cluster_firing_maps_j[hmt_trial_numbers-1]
                    avg_correlation = get_avg_correlation(hmt_cluster_firing_maps_j)

                    # reconstruct avg spatial correlation using the newly aligned trials
                    reconstructed_signal = []
                    for ti, tn in enumerate(hmt_trial_numbers):
                        reconstructed_trial = get_reconstructed_trial_signal(shifts_i[ti], hmt_cluster_firing_maps_j[ti].flatten(),
                                                                             min_shift=min(shifts_i), max_shift=max(shifts_i))
                        reconstructed_signal.append(reconstructed_trial.tolist())
                    reconstructed_signal = np.array(reconstructed_signal)
                    reconstructed_signal_corr = get_avg_correlation(reconstructed_signal)

                    cross_correlations[i, j] = reconstructed_signal_corr - avg_correlation

            fig, ax = plt.subplots()
            im= ax.imshow(cross_correlations, cmap="coolwarm", vmin=-0.5, vmax=0.5)
            ax.set_xticks(np.arange(len(ids)))
            ax.set_yticks(np.arange(len(ids)))
            ax.set_yticklabels(ids)
            ax.set_xticklabels(ids)
            ax.set_ylabel("Cluster ID: Reference", fontsize=5)
            ax.set_xlabel("Cluster ID: Shifted", fontsize=5)
            ax.tick_params(axis='both', which='major', labelsize=8)
            fig.tight_layout()
            fig.colorbar(im, ax=ax)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + 'joint_alignment_cross_correlations_'+hmt+'.png', dpi=300)
            plt.close()

    return

def get_indices(hmt, tt):
    i = tt
    if hmt=="hit":
        j = 0
    elif hmt=="miss":
        j = 1
    elif hmt=="try":
        j = 2
    return i, j

def get_shifts(cluster_spike_data, hmt, tt):
    i, j= get_indices(hmt, tt)
    shifts = cluster_spike_data['field_realignments_hmt_by_trial_type'].iloc[0][i][j]
    return shifts


def plot_firing_rate_maps_per_trial_by_hmt_aligned_other_neuron(spike_data, processed_position_data, output_path, track_length, trial_types):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials_aligned_by_i'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data_ = pd.DataFrame()
    for tt in trial_types:
        processed_position_data_ = pd.concat([processed_position_data_, processed_position_data[processed_position_data["trial_type"] == tt]], ignore_index=True)
    processed_position_data = processed_position_data_
    string_tts = [str(int) for int in trial_types]
    string_tts = "-".join(string_tts)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        for cluster_index_j, cluster_id_j in enumerate(spike_data.cluster_id):
            cluster_j_df = spike_data[spike_data["cluster_id"]==cluster_id_j]
            firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

            if len(firing_times_cluster)>1:
                cluster_firing_maps2 = np.array(spike_data["fr_binned_in_space"].iloc[cluster_index])
                where_are_NaNs2 = np.isnan(cluster_firing_maps2)
                cluster_firing_maps2[where_are_NaNs2] = 0

                if len(cluster_firing_maps2) == 0:
                    print("stop here")
                cluster_firing_maps = min_max_normalize(cluster_firing_maps2)
                vmin, vmax = get_vmin_vmax(cluster_firing_maps)

                fig, axes = plt.subplots(3, 1, figsize=(6,6), sharex=True)
                for ax, hmt, color, ytitle in zip(axes, ["hit", "try", "miss"], ["green", "orange", "red"], ["Hits", "Tries", "Runs"]):
                    shifts = get_shifts(cluster_j_df, hmt=hmt, tt=tt)
                    hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
                    if len(hmt_processed_position_data)>0:
                        hmt_trial_numbers = pandas_collumn_to_numpy_array(hmt_processed_position_data["trial_number"])
                        hmt_cluster_firing_maps = cluster_firing_maps[hmt_trial_numbers-1]
                        hmt_mean_firing_rate_map = np.nanmean(hmt_cluster_firing_maps, axis=0)
                        avg_correlation = get_avg_correlation(hmt_cluster_firing_maps)

                        for i, tn in enumerate(hmt_trial_numbers):
                            B = hmt_cluster_firing_maps[i].flatten()
                            locations = np.arange(0, len(B))+shifts[i]
                            X, Y = np.meshgrid(locations, i)
                            cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
                            trial_firing_rate =  hmt_cluster_firing_maps[i].reshape(1, len(hmt_cluster_firing_maps[i]))
                            X = np.vstack((X,X)); Y = np.vstack((Y,Y+0.5)); trial_firing_rate = np.vstack((trial_firing_rate, trial_firing_rate))
                            ax.pcolormesh(X, Y, trial_firing_rate, cmap=cmap, shading="gourand", vmin=vmin, vmax=vmax)

                        # reconstruct avg spatial correlation using the newly aligned trials
                        reconstructed_signal=[]
                        for i, tn in enumerate(hmt_trial_numbers):
                            reconstructed_trial = get_reconstructed_trial_signal(shifts[i], hmt_cluster_firing_maps[i].flatten(),
                                                                                 min_shift=min(shifts), max_shift=max(shifts))
                            reconstructed_signal.append(reconstructed_trial.tolist())

                        reconstructed_signal = np.array(reconstructed_signal)
                        reconstructed_signal_corr = get_avg_correlation(reconstructed_signal)

                        Edmond.plot_utility2.style_vr_plot(ax, len(hmt_processed_position_data))
                    else:
                        avg_correlation = np.nan
                        reconstructed_signal_corr = np.nan

                    ax.tick_params(axis='both', which='both', labelsize=20)
                    ax.set_yticks([len(hmt_processed_position_data)-1])
                    ax.set_yticklabels([len(hmt_processed_position_data)])
                    ax.set_ylabel(ytitle, fontsize=25, labelpad = 15)
                    ax.set_title(("before align R:"+str(np.round(avg_correlation, decimals=2))+ ", after align R:"+str(np.round(reconstructed_signal_corr, decimals=2))), fontsize=13)
                    plt.xlabel('Location (cm)', fontsize=25, labelpad = 20)
                tick_spacing = 100
                ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                fig.tight_layout(pad=2.0)
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
                plt.savefig(save_path +'/'+spike_data.session_id.iloc[cluster_index]+'_firing_rate_map_trials_'+str(cluster_id)+"_by_"+str(cluster_id_j)+ '_tt_'+string_tts+'.png', dpi=300)
                plt.close()
    return


def add_avg_correlations_and_realignement_shifts(spike_data, processed_position_data, track_length):
    avg_correlations_hmt_by_trial_type = []
    field_realignments_hmt_by_trial_type = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[spike_data["cluster_id"]==cluster_id]
        firing_times_cluster = spike_data.firing_times.iloc[0]
        try:
            putative_field_frequency = int(np.round(cluster_df["ML_Freqs"].iloc[0]))
            max_shift = int(track_length/putative_field_frequency)
        except:
            max_shift=200

        avg_correlation_cluster = [[np.nan] * 3 for i in [1] * 3]
        field_realignments_cluster = [[np.nan] * 3 for i in [1] * 3]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data["fr_binned_in_space"].iloc[cluster_index])
            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)

            for i, tt in enumerate([0,1,2]):
                for j, hmt in enumerate(["hit", "miss", "try"]):
                    subset_processed_position_data = processed_position_data[(processed_position_data["trial_type"] == tt)]
                    subset_processed_position_data = subset_processed_position_data[(subset_processed_position_data["hit_miss_try"] == hmt)]
                    subset_trial_numbers = np.asarray(subset_processed_position_data["trial_number"])

                    shifts=[]
                    avg_shift = np.nan
                    avg_correlation = np.nan
                    if len(subset_trial_numbers)>0:
                        hmt_cluster_firing_maps = cluster_firing_maps[subset_trial_numbers-1]
                        hmt_mean_firing_rate_map = np.nanmean(hmt_cluster_firing_maps, axis=0)
                        avg_correlation = get_avg_correlation(hmt_cluster_firing_maps)

                        # best align to mean firing rate map
                        for ti, tn in enumerate(subset_trial_numbers):
                            A = hmt_mean_firing_rate_map
                            B = hmt_cluster_firing_maps[ti].flatten()
                            A -= A.mean(); A /= A.std()
                            B -= B.mean(); B /= B.std()
                            xcorr = signal.correlate(A, B, mode="same")
                            lags = signal.correlation_lags(A.size, B.size, mode="same")
                            xcorr /= np.max(xcorr)

                            xcorr = xcorr[np.abs(lags).argsort()][0:max_shift]
                            lags = lags[np.abs(lags).argsort()][0:max_shift]
                            recovered_shift = lags[xcorr.argmax()]
                            shifts.append(recovered_shift)

                    avg_correlation_cluster[i][j] = avg_correlation
                    field_realignments_cluster[i][j] = np.array(shifts)

        avg_correlations_hmt_by_trial_type.append(avg_correlation_cluster)
        field_realignments_hmt_by_trial_type.append(field_realignments_cluster)

    spike_data["avg_correlations_hmt_by_trial_type"] = avg_correlations_hmt_by_trial_type
    spike_data["field_realignments_hmt_by_trial_type"] = field_realignments_hmt_by_trial_type
    return spike_data

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


def plot_firing_rate_maps_per_trial_by_hmt_aligned(spike_data, processed_position_data, output_path, track_length, trial_types):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials_aligned'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data_ = pd.DataFrame()
    for tt in trial_types:
        processed_position_data_ = pd.concat([processed_position_data_, processed_position_data[processed_position_data["trial_type"] == tt]], ignore_index=True)
    processed_position_data = processed_position_data_
    string_tts = [str(int) for int in trial_types]
    string_tts = "-".join(string_tts)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_times_cluster = cluster_df.firing_times.iloc[0]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(cluster_df["fr_binned_in_space"].iloc[0])

            try:
                putative_field_frequency = int(np.round(cluster_df["ML_Freqs"].iloc[0]))
                max_shift = int(track_length/putative_field_frequency)
            except:
                max_shift=200
            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)

            fig, axes = plt.subplots(3, 1, figsize=(6,6), sharex=True)
            for ax, hmt, color, ytitle in zip(axes, ["hit", "try", "miss"], ["green", "orange", "red"], ["Hits", "Tries", "Runs"]):
                hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
                if len(hmt_processed_position_data)>0:
                    hmt_trial_numbers = pandas_collumn_to_numpy_array(hmt_processed_position_data["trial_number"])
                    hmt_cluster_firing_maps = cluster_firing_maps[hmt_trial_numbers-1]
                    hmt_mean_firing_rate_map = np.nanmean(hmt_cluster_firing_maps, axis=0)
                    avg_correlation = get_avg_correlation(hmt_cluster_firing_maps)

                    # best align to mean firing rate map
                    shifts = []
                    reconstructed_signal = []
                    for i, tn in enumerate(hmt_trial_numbers):
                        A = hmt_mean_firing_rate_map
                        B = hmt_cluster_firing_maps[i].flatten()

                        nsamples = A.size
                        A -= A.mean(); A /= A.std()
                        B -= B.mean(); B /= B.std()
                        xcorr = signal.correlate(A, B, mode="same")
                        lags = signal.correlation_lags(A.size, B.size, mode="same")
                        xcorr /= np.max(xcorr)

                        xcorr = xcorr[np.abs(lags).argsort()][0:max_shift]
                        lags = lags[np.abs(lags).argsort()][0:max_shift]
                        recovered_shift = lags[xcorr.argmax()]
                        shifts.append(recovered_shift)

                        locations = np.arange(0, len(B))+recovered_shift
                        X, Y = np.meshgrid(locations, i)
                        cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
                        trial_firing_rate =  hmt_cluster_firing_maps[i].reshape(1, len(hmt_cluster_firing_maps[i]))
                        X = np.vstack((X,X)); Y = np.vstack((Y,Y+0.5)); trial_firing_rate = np.vstack((trial_firing_rate, trial_firing_rate))
                        ax.pcolormesh(X, Y, trial_firing_rate, cmap=cmap, shading="gourand", vmin=vmin, vmax=vmax)

                    avg_shift = np.nanmean(np.abs(np.array(shifts)))
                    # reconstruct avg spatial correlation using the newly aligned trials
                    for i, tn in enumerate(hmt_trial_numbers):
                        reconstructed_trial = get_reconstructed_trial_signal(shifts[i], hmt_cluster_firing_maps[i].flatten(),
                                                                             min_shift=min(shifts), max_shift=max(shifts))
                        reconstructed_signal.append(reconstructed_trial.tolist())
                    reconstructed_signal = np.array(reconstructed_signal)

                    Edmond.plot_utility2.style_vr_plot(ax, len(hmt_processed_position_data))
                    ax.set_title(("Map Shift: "+str(np.round(avg_shift, decimals=2))+"cm"), fontsize=20)
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.set_yticks([len(hmt_processed_position_data)-1])
                ax.set_yticklabels([len(hmt_processed_position_data)])
                ax.set_ylabel(ytitle, fontsize=25, labelpad = 15)
                plt.xlabel('Location (cm)', fontsize=25, labelpad = 20)
                #ax.set_xlim([0, track_length])
                #ax.set_ylim([0, len(hmt_processed_position_data)-1])

            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            fig.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '_tt_'+string_tts+'.png', dpi=300)
            plt.close()




def plot_firing_rate_maps_per_trial_by_tt_aligned(spike_data, processed_position_data, output_path, track_length, hmts):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials_aligned'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    processed_position_data_ = pd.DataFrame()
    for hmt in hmts:
        processed_position_data_ = pd.concat([processed_position_data_, processed_position_data[processed_position_data["hit_miss_try"] == hmt]], ignore_index=True)
    processed_position_data = processed_position_data_
    string_hmts = [str(int) for int in hmts]
    string_hmts = "-".join(string_hmts)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_times_cluster = cluster_df.firing_times.iloc[0]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(cluster_df["fr_binned_in_space"].iloc[0])
            try:
                putative_field_frequency = int(np.round(cluster_df["ML_Freqs"].iloc[0]))
                max_shift = int(track_length/putative_field_frequency)
            except:
                max_shift=200

            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)

            fig, axes = plt.subplots(3, 1, figsize=(6,6), sharex=True)
            for ax, tt, color, ytitle in zip(axes, [2,0,1], ["red", "black", "blue"], ["PI", "Cued", "PI"]):
                tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
                if len(tt_processed_position_data)>0:
                    tt_trial_numbers = pandas_collumn_to_numpy_array(tt_processed_position_data["trial_number"])
                    tt_cluster_firing_maps = cluster_firing_maps[tt_trial_numbers-1]
                    tt_mean_firing_rate_map = np.nanmean(tt_cluster_firing_maps, axis=0)
                    avg_correlation = get_avg_correlation(tt_cluster_firing_maps)

                    # best align to mean firing rate map
                    shifts = []
                    reconstructed_signal = []
                    for i, tn in enumerate(tt_trial_numbers):
                        A = tt_mean_firing_rate_map
                        B = tt_cluster_firing_maps[i].flatten()

                        nsamples = A.size
                        A -= A.mean(); A /= A.std()
                        B -= B.mean(); B /= B.std()
                        xcorr = signal.correlate(A, B, mode="same")
                        lags = signal.correlation_lags(A.size, B.size, mode="same")
                        xcorr /= np.max(xcorr)

                        xcorr = xcorr[np.abs(lags).argsort()][0:max_shift]
                        lags = lags[np.abs(lags).argsort()][0:max_shift]
                        recovered_shift = lags[xcorr.argmax()]
                        shifts.append(recovered_shift)

                        locations = np.arange(0, len(B))+recovered_shift
                        X, Y = np.meshgrid(locations, i)
                        cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
                        trial_firing_rate =  tt_cluster_firing_maps[i].reshape(1, len(tt_cluster_firing_maps[i]))
                        X = np.vstack((X,X)); Y = np.vstack((Y,Y+0.5)); trial_firing_rate = np.vstack((trial_firing_rate, trial_firing_rate))
                        ax.pcolormesh(X, Y, trial_firing_rate, cmap=cmap, shading="gourand", vmin=vmin, vmax=vmax)

                    avg_shift = np.nanmean(np.abs(np.array(shifts)))
                    # reconstruct avg spatial correlation using the newly aligned trials
                    for i, tn in enumerate(tt_trial_numbers):
                        reconstructed_trial = get_reconstructed_trial_signal(shifts[i], tt_cluster_firing_maps[i].flatten(),
                                                                             min_shift=min(shifts), max_shift=max(shifts))
                        reconstructed_signal.append(reconstructed_trial.tolist())
                    reconstructed_signal = np.array(reconstructed_signal)

                    Edmond.plot_utility2.style_vr_plot(ax, len(tt_processed_position_data))
                    ax.set_title(("Map Shift: "+str(np.round(avg_shift, decimals=2))+"cm"), fontsize=20)
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.set_yticks([len(tt_processed_position_data)-1])
                ax.set_yticklabels([len(tt_processed_position_data)])
                ax.set_ylabel(ytitle, fontsize=25, labelpad = 15)
                plt.xlabel('Location (cm)', fontsize=25, labelpad = 20)
                #ax.set_xlim([0, track_length])
                #ax.set_ylim([0, len(hmt_processed_position_data)-1])

            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            fig.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '_hmt_'+string_hmts+'.png', dpi=300)
            plt.close()

def get_hmt_shifts(cluster_hmt_shifts):
    hit_shifts=[];  try_shifts=[]; miss_shifts=[]
    for i in range(len(cluster_hmt_shifts)):
        hit_shifts.append(cluster_hmt_shifts[i][0])
        try_shifts.append(cluster_hmt_shifts[i][1])
        miss_shifts.append(cluster_hmt_shifts[i][2])
    return np.array(hit_shifts), np.array(try_shifts), np.array(miss_shifts)

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
            cluster_firing_maps = np.array(spike_data["fr_binned_in_space"].iloc[cluster_index])
            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)

            fig, axes = plt.subplots(3, 1, figsize=(6,6), sharex=True)
            for ax, hmt, color, ytitle in zip(axes, ["hit", "try", "miss"], ["green", "orange", "red"], ["Hits", "Tries", "Runs"]):
                hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
                if len(hmt_processed_position_data)>1:
                    hmt_trial_numbers = pandas_collumn_to_numpy_array(hmt_processed_position_data["trial_number"])
                    hmt_cluster_firing_maps = cluster_firing_maps[hmt_trial_numbers-1]
                    locations = np.arange(0, len(hmt_cluster_firing_maps[0]))
                    ordered = np.arange(0, len(hmt_processed_position_data), 1)
                    X, Y = np.meshgrid(locations, ordered)
                    cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
                    ax.pcolormesh(X, Y, hmt_cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
                if len(hmt_processed_position_data)>0:
                    Edmond.plot_utility2.style_vr_plot(ax, len(hmt_processed_position_data))
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.set_yticks([len(hmt_processed_position_data)-1])
                ax.set_yticklabels([len(hmt_processed_position_data)])
                ax.set_ylabel(ytitle, fontsize=25, labelpad = 15)
                plt.xlabel('Location (cm)', fontsize=25, labelpad = 20)
                ax.set_xlim([0, track_length])
                ax.set_ylim([0, len(hmt_processed_position_data)-1])

            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            fig.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '_tt_'+string_tts+'.png', dpi=300)
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
            cluster_firing_maps = np.array(spike_data["fr_binned_in_space"].iloc[cluster_index])
            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0
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

def plot_firing_rates_tt(concantenated_dataframe, save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    b_mean_fr = pandas_collumn_to_numpy_array(grid_cells['mean_fr_tt_0_hmt_hit'])
    nb_mean_fr = pandas_collumn_to_numpy_array(grid_cells['mean_fr_tt_1_hmt_hit'])
    rate_discriminant_index = (b_mean_fr-nb_mean_fr)/(b_mean_fr+nb_mean_fr)
    rate_discriminant_index = rate_discriminant_index[~np.isnan(rate_discriminant_index)]
    bad_bnb = ~np.logical_or(np.isnan(b_mean_fr), np.isnan(nb_mean_fr))
    p = stats.wilcoxon(rate_discriminant_index)[1]
    p_str = get_p_text(p)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.axvline(x=0, linewidth=2, linestyle="dashed", color="blue")
    ax.axvline(x=np.median(rate_discriminant_index), linestyle="solid", linewidth=2, color="red")
    ax.hist(rate_discriminant_index, bins=100, range=(-1, 1), density=False, facecolor="white", linewidth=1, edgecolor="black")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=20)
    ax.set_xlim(left=-1, right=1)
    ax.set_xticks([-1, 0, 1])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("RDI", fontsize=20)
    ax.set_ylabel("Neurons", fontsize=20)
    plt.savefig(save_path + '/mean_firing_rate_hits_by_tt.png', dpi=300)
    plt.close()
    return


def plot_firing_rates_hmt(concantenated_dataframe, save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    h_mean_fr = pandas_collumn_to_numpy_array(grid_cells['mean_fr_tt_1_hmt_hit'])
    m_mean_fr = pandas_collumn_to_numpy_array(grid_cells['mean_fr_tt_1_hmt_miss'])

    rate_discriminant_index = (h_mean_fr-m_mean_fr)/(h_mean_fr+m_mean_fr)
    rate_discriminant_index = rate_discriminant_index[~np.isnan(rate_discriminant_index)]
    bad_hm = ~np.logical_or(np.isnan(h_mean_fr), np.isnan(m_mean_fr))
    p = stats.wilcoxon(rate_discriminant_index)[1]
    p_str = get_p_text(p)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.axvline(x=0, linewidth=2, linestyle="dashed", color="blue")
    ax.axvline(x=np.median(rate_discriminant_index), linestyle="solid", linewidth=2, color="red")
    ax.hist(rate_discriminant_index, bins=100, range=(-1, 1), density=False, facecolor="white", linewidth=1, edgecolor="black")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=20)
    ax.set_xlim(left=-1, right=1)
    ax.set_xticks([-1, 0, 1])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("RDI", fontsize=20)
    ax.set_ylabel("Neurons", fontsize=20)
    plt.savefig(save_path + '/mean_firing_rate_nb_by_hmt.png', dpi=300)
    plt.close()
    return

def plot_firing_rates_PDN(concantenated_dataframe, save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    p_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    d_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]
    n_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Null"]

    p_mean_fr = pandas_collumn_to_numpy_array(p_grids['mean_fr_tt_all_hmt_all'])
    d_mean_fr = pandas_collumn_to_numpy_array(d_grids['mean_fr_tt_all_hmt_all'])
    n_mean_fr = pandas_collumn_to_numpy_array(n_grids['mean_fr_tt_all_hmt_all'])

    fig, ax = plt.subplots(figsize=(4,4))
    objects = ["P", "D", "N"]
    data = [p_mean_fr[~np.isnan(p_mean_fr)], d_mean_fr[~np.isnan(d_mean_fr)], n_mean_fr[~np.isnan(n_mean_fr)]]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=20)
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(["P", "D", "N"])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Mean Rate (Hz)", fontsize=20)
    significance_bar(start=1, end=2, height=15, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[1])[1]))
    significance_bar(start=1, end=3, height=16, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[0], data[2])[1]))
    significance_bar(start=2, end=3, height=17, displaystring=get_p_text(scipy.stats.mannwhitneyu(data[1], data[2])[1]))
    plt.savefig(save_path + '/mean_firing_rate_PDN.png', dpi=300)
    plt.close()
    return



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

                    all_trials_fr = []
                    for tn in subset_processed_position_data["trial_number"]:
                        hmt_trial_numbers = np.array([tn])
                        time_seconds = np.sum(position_data_time_in_bin_sec[np.isin(position_data_trial_numbers, hmt_trial_numbers)])
                        n_spikes = len(firing_trial_numbers[np.isin(firing_trial_numbers, hmt_trial_numbers)])
                        firing_rate = n_spikes/time_seconds
                        all_trials_fr.append(firing_rate)
                    mean_firing_rate = np.nanmean(np.array(all_trials_fr))

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

def add_n_trial(spike_data, processed_position_data):
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

def calculate_rewarded_stops(processed_position_data, track_length):
    # this function assumes the reward zone is located
    # at specific coordinates relative to the track length
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30

    reward_stop_location_cm = []
    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        stop_location_cm = np.array(trial_row["stop_location_cm"].iloc[0])
        reward_locations = stop_location_cm[(stop_location_cm > reward_zone_start) & (stop_location_cm < reward_zone_end)]

        if len(reward_locations)==0:
            reward_locations = []
        reward_stop_location_cm.append(list(reward_locations))

    processed_position_data["reward_stop_location_cm"] = reward_stop_location_cm
    return processed_position_data

def calculate_rewarded_trials(processed_position_data):
    rewarded_trials = []
    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        reward_stop_location_cm = np.array(trial_row["reward_stop_location_cm"].iloc[0])
        if len(reward_stop_location_cm)==0:
            rewarded = False
        else:
            rewarded = True
        rewarded_trials.append(rewarded)

    processed_position_data["rewarded"] = rewarded_trials
    return processed_position_data

def get_hmt_for_centre_trial(centre_trials, processed_position_data, tt=1):

    hit_trials = np.asarray(processed_position_data[(processed_position_data["hit_miss_try"] == "hit") &
                                                    (processed_position_data["trial_type"] == tt)]["trial_number"])
    try_trials = np.asarray(processed_position_data[(processed_position_data["hit_miss_try"] == "try") &
                                                    (processed_position_data["trial_type"] == tt)]["trial_number"])
    miss_trials = np.asarray(processed_position_data[(processed_position_data["hit_miss_try"] == "miss") &
                                                    (processed_position_data["trial_type"] == tt)]["trial_number"])
    centre_trials_hmt = np.ones(len(centre_trials))*3

    hit_mask = np.isin(centre_trials, hit_trials)
    try_mask = np.isin(centre_trials, try_trials)
    miss_mask = np.isin(centre_trials, miss_trials)
    centre_trials_hmt[hit_mask] = 2
    centre_trials_hmt[try_mask] = 1
    centre_trials_hmt[miss_mask] = 0
    return centre_trials_hmt

def get_rolling_lomb_classifier_for_centre_trial(centre_trials, powers, n_window_size=Settings.rolling_window_size_for_lomb_classifier):
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
            rolling_lomb_classifier.append(0.5)
            rolling_lomb_classifier_colors.append(Settings.allocentric_color)
        elif lomb_classifier == "Distance":
            rolling_lomb_classifier.append(1.5)
            rolling_lomb_classifier_colors.append(Settings.egocentric_color)
        elif lomb_classifier == "Null":
            rolling_lomb_classifier.append(2.5)
            rolling_lomb_classifier_colors.append(Settings.null_color)
        else:
            rolling_lomb_classifier.append(3.5)
            rolling_lomb_classifier_colors.append("black")


    return np.array(rolling_lomb_classifier), np.array(rolling_lomb_classifier_colors), np.array(peak_frequencies), np.array(trial_points)

def get_peak_powers_across_trials(powers, centre_trials):
    tn_powers=[]
    for tn in np.unique(centre_trials):
        tn_power = powers[centre_trials==tn]
        peak_powers = np.max(tn_power, axis=1)
        avg_peak = np.mean(peak_powers)
        tn_powers.append(avg_peak)
    return np.array(tn_powers), np.unique(centre_trials)

def plot_hits_of_cued_and_pi_trials(processed_position_data, output_path):
    print('I am plotting stop rasta...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    fig, ax = plt.subplots(1, 1, figsize=(3,6))

    cued_hits_trial_numbers = processed_position_data[(processed_position_data["hit_miss_try"] == "hit") &
                                                      (processed_position_data["trial_type"] == 0)]["trial_number"]
    PI_hits_trial_numbers = processed_position_data[(processed_position_data["hit_miss_try"] == "hit") &
                                                      (processed_position_data["trial_type"] == 1)]["trial_number"]

    ax.scatter(np.ones(len(cued_hits_trial_numbers))*0, cued_hits_trial_numbers, marker="_", color="green", clip_on=False, s=4000)
    ax.scatter(np.ones(len(PI_hits_trial_numbers))*0.3, PI_hits_trial_numbers, marker="_", color="green", clip_on=False, s=4000)

    ax.set_ylim([0, len(processed_position_data)])
    ax.set_xticks([0, 0.3])
    ax.set_xlim(-0.2, 0.4)
    ax.set_xticklabels(["Cued", "PI"])
    plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_visible(False)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(bottom = 0.2)
    plt.savefig(save_path + '/hits_across_trial_types.png', dpi=300)
    plt.close()
    return

def plot_rolling_lomb_codes_across_cells(spike_data, paired_recording, processed_position_data, output_path, track_length):

    if paired_recording is not None:
        of_spike_data = pd.read_pickle(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl")
    spike_data = pd.merge(spike_data, of_spike_data[["cluster_id", "classifier"]], on="cluster_id")

    spike_data = spike_data[spike_data["classifier"] == "G"]
    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/rolling_classifiers'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = Settings.frequency_step
    frequency = Settings.frequency

    fig, axes = plt.subplots(4, 1, figsize=(6,12), sharex=True, gridspec_kw={'height_ratios': [2, 6, 6, 6]})

    spike_data = spike_data.sort_values(by=["classifier"])
    tab10 = plt.rcParams['axes.prop_cycle'].by_key()['color']

    y_pos = 0
    y_step = 0.4
    classifier_labels = []
    avg_firing_rates_cluster = []
    avg_peak_powers_cluster = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        firing_rates = np.array(cluster_spike_data["fr_binned_in_space"].iloc[0])
        where_are_NaNs = np.isnan(firing_rates)
        firing_rates[where_are_NaNs] = 0

        avg_firing_rates = np.mean(firing_rates, axis=1)
        avg_firing_rates = min_max_normalize(avg_firing_rates)

        gauss_kernel = Gaussian1DKernel(stddev=2)
        avg_firing_rates = convolve(avg_firing_rates, gauss_kernel)
        trial_numbers = np.asarray(processed_position_data.trial_number)
        classifier = cluster_spike_data["classifier"].iloc[0]
        label = classifier+str(cluster_index+1)
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        centre_trials = np.round(centre_trials).astype(np.int64)
        peak_power_across_trials, trial_centre_numbers = get_peak_powers_across_trials(powers, centre_trials)
        peak_power_across_trials = min_max_normalize(peak_power_across_trials)
        peak_power_across_trials = convolve(peak_power_across_trials, gauss_kernel)

        axes[1].plot(trial_numbers, avg_firing_rates, alpha=0.5, color=tab10[cluster_index])
        axes[2].plot(trial_centre_numbers, peak_power_across_trials, alpha=0.5, color=tab10[cluster_index])

        legend_freq = np.linspace(y_pos, y_pos+0.2, 5)
        rolling_lomb_classifier, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(centre_trials, powers)
        rolling_lomb_classifier_tiled = np.tile(rolling_lomb_classifier,(len(legend_freq),1))
        cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
        boundaries = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        X, Y = np.meshgrid(centre_trials, legend_freq)
        axes[3].pcolormesh(X, Y, rolling_lomb_classifier_tiled, cmap=cmap, norm=norm, shading="flat")


        avg_firing_rates_cluster.append(avg_firing_rates.tolist())
        avg_peak_powers_cluster.append(peak_power_across_trials.tolist())
        y_pos += y_step
        classifier_labels.append(label)

    avg_firing_rates = np.mean(np.array(avg_firing_rates_cluster), axis=0)
    avg_peak_powers = np.mean(np.array(avg_peak_powers_cluster), axis=0)
    axes[1].plot(trial_numbers, avg_firing_rates, color="black")
    axes[2].plot(trial_centre_numbers, avg_peak_powers, color="black")

    axes[3].set_yticks(np.arange(0, y_step*len(spike_data), y_step)+0.1)
    axes[3].set_yticklabels(classifier_labels)
    for i in range(len(classifier_labels)):
        axes[3].get_yticklabels()[i].set_color(tab10[i])
    #legend_freq = np.linspace(y_pos, y_pos+0.2, 5)
    #centre_trial_hmt_nb = get_hmt_for_centre_trial(centre_trials, processed_position_data, tt=1)
    #centre_trial_hmt_nb = np.tile(centre_trial_hmt_nb,(len(legend_freq),1))
    #cmap = colors.ListedColormap(['red', 'orange', 'green', 'white'])
    #boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5]
    #norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    #X, Y = np.meshgrid(centre_trials, legend_freq)
    #axes[0].pcolormesh(X, Y, centre_trial_hmt_nb, cmap=cmap, norm=norm, shading="flat")

    centre_trial_hmt_nb = get_hmt_for_centre_trial(centre_trials, processed_position_data, tt=1)
    centre_trial_hmt_b = get_hmt_for_centre_trial(centre_trials, processed_position_data, tt=0)
    centre_trial_hmt_nb[centre_trial_hmt_nb==3] = np.nan
    centre_trial_hmt_b[centre_trial_hmt_b==3] = np.nan

    axes[0].plot(centre_trials[~np.isnan(centre_trial_hmt_b)], centre_trial_hmt_b[~np.isnan(centre_trial_hmt_b)], color="black")
    axes[0].plot(centre_trials[~np.isnan(centre_trial_hmt_nb)], centre_trial_hmt_nb[~np.isnan(centre_trial_hmt_nb)], color="blue")

    axes[1].set_ylabel('Normalised FR', fontsize=20, labelpad = 10)
    axes[2].set_ylabel('Normalised Power', fontsize=20, labelpad = 10)
    axes[3].set_ylabel('Cell number', fontsize=20, labelpad = 10)
    axes[3].set_xlabel('CentreTrial', fontsize=20, labelpad = 10)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    n_x_ticks = int(max(centre_trials)//50)+1
    x_tick_locs= np.linspace(np.ceil(min(centre_trials)), max(centre_trials), n_x_ticks, dtype=np.int64)
    axes[3].set_xticks(x_tick_locs.tolist())
    axes[3].set_ylim([-0.1, (y_step*len(spike_data))])
    axes[3].set_xlim([min(centre_trials), max(centre_trials)])
    axes[0].set_yticks([0,1,2])
    axes[0].set_yticklabels(["Run","Try","Hit"])
    axes[0].set_ylim([0,2])
    axes[0].xaxis.set_visible(False)
    axes[3].yaxis.set_tick_params(labelsize=20)
    axes[2].yaxis.set_tick_params(labelsize=22)
    axes[1].yaxis.set_tick_params(labelsize=22)
    axes[0].yaxis.set_tick_params(labelsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(bottom = 0.2)
    plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_spatial_moving_lomb_scargle_periodogram_Cluster_' + str(cluster_id) +'.png', dpi=300)
    plt.close()
    return



def plot_spatial_periodogram(spike_data, processed_position_data, output_path, track_length):

    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = Settings.frequency_step
    frequency = Settings.frequency

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
        centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
        centre_trials = np.round(centre_trials).astype(np.int64)
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        modal_frequency = cluster_spike_data['ML_Freqs'].iloc[0]
        modal_class = cluster_spike_data['Lomb_classifier_'].iloc[0]
        modal_class_color = get_modal_color(modal_class)


        fig = plt.figure(figsize=(6,6))
        n_y_ticks = int(max(centre_trials)//50)+1
        y_tick_locs= np.linspace(np.ceil(min(centre_trials)), max(centre_trials), n_y_ticks, dtype=np.int64)
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        powers[np.isnan(powers)] = 0
        Y, X = np.meshgrid(centre_trials, frequency)
        cmap = plt.cm.get_cmap("inferno")
        ax.pcolormesh(X, Y, powers.T, cmap=cmap, shading="flat")

        #legend_freq = np.linspace(5.1, 5.3, 5)
        #centre_trial_hmt = get_hmt_for_centre_trial(centre_trials, processed_position_data, tt=1)
        #centre_trial_hmt = np.tile(centre_trial_hmt,(len(legend_freq),1))
        #cmap = colors.ListedColormap(['red', 'orange', 'green', 'white'])
        #boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5]
        #norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        #X, Y = np.meshgrid(centre_trials, legend_freq)
        #ax.pcolormesh(X, Y, centre_trial_hmt, cmap=cmap, norm=norm, shading="flat")

        legend_freq = np.linspace(5.3, 5.5, 5)
        rolling_lomb_classifier, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(centre_trials, powers)
        rolling_lomb_classifier_tiled = np.tile(rolling_lomb_classifier,(len(legend_freq),1))
        cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
        boundaries = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        Y, X = np.meshgrid(centre_trials, legend_freq)
        ax.pcolormesh(X, Y, rolling_lomb_classifier_tiled, cmap=cmap, norm=norm, shading="flat")

        for f in range(1,6):
            ax.axvline(x=f, color="white", linewidth=2,linestyle="dotted")
        #ax.scatter(rolling_points, rolling_frequencies, c=rolling_lomb_classifier_colors, marker="o")
        plt.xlabel('Track Frequency', fontsize=25, labelpad = 10)
        plt.ylabel('Centre Trial', fontsize=25, labelpad = 10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_yticks(y_tick_locs.tolist())
        ax.set_xlim([0.1,5])
        ax.set_ylim([min(centre_trials), max(centre_trials)])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_spatial_moving_lomb_scargle_periodogram_Cluster_' + str(cluster_id) +'.png', dpi=300)
        plt.close()
    return

def interpolate_by_trial(cluster_firing_maps, step_cm, track_length):
    x = np.arange(0, track_length)
    xnew = np.arange(step_cm/2, track_length, step_cm)

    interpolated_rate_map = []
    for i in range(len(cluster_firing_maps)):
        trial_cluster_firing_maps = cluster_firing_maps[i]
        y = trial_cluster_firing_maps
        f = interpolate.interp1d(x, y)

        ynew = f(xnew)
        interpolated_rate_map.append(ynew.tolist())

    return np.array(interpolated_rate_map)


def plot_both_spatial_periodograms(spike_data, processed_position_data, output_path, track_length):

    print('plotting moving lomb_scargle periodogram...')
    save_path = output_path + '/Figures/moving_lomb_scargle_periodograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    step = Settings.frequency_step
    frequency = Settings.frequency

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        if len(firing_times_cluster)>1:

            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)
            firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
            modal_frequency = cluster_spike_data['ML_Freqs'].iloc[0]
            modal_class = cluster_spike_data['Lomb_classifier_'].iloc[0]
            modal_class_color = get_modal_color(modal_class)

            fig, axes = plt.subplots(3,1,figsize=(6,18), gridspec_kw={'height_ratios': [1, 1, 1]})

            cluster_firing_maps2 = np.array(spike_data["fr_binned_in_space"].iloc[cluster_index])
            where_are_NaNs2 = np.isnan(cluster_firing_maps2)
            cluster_firing_maps2[where_are_NaNs2] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps2)
            step_cm = 2
            cluster_firing_maps = interpolate_by_trial(cluster_firing_maps, step_cm =step_cm, track_length=track_length)


            vmin, vmax = get_vmin_vmax(cluster_firing_maps)
            x_max = len(processed_position_data)
            locations = np.arange(0, track_length, step_cm)
            ordered = np.arange(0, len(processed_position_data), 1)
            X, Y = np.meshgrid(locations, ordered)
            cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
            axes[0].pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
            axes[0].set_ylabel('Trial Number', fontsize=30, labelpad = 20)
            axes[0].set_xlabel('Location (cm)', fontsize=25, labelpad = 20)
            axes[0].set_xlim([0, track_length])
            axes[0].tick_params(axis='both', which='both', labelsize=20)
            axes[0].set_ylim([0, len(processed_position_data)-1])
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['right'].set_visible(False)
            tick_spacing = 100
            axes[0].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            axes[0].yaxis.set_ticks_position('left')
            axes[0].xaxis.set_ticks_position('bottom')


            n_y_ticks = int(max(centre_trials)//50)+1
            y_tick_locs= np.linspace(np.ceil(min(centre_trials)), max(centre_trials), n_y_ticks, dtype=np.int64)
            powers[np.isnan(powers)] = 0
            Y, X = np.meshgrid(centre_trials, frequency)
            cmap = plt.cm.get_cmap("inferno")
            axes[1].pcolormesh(X, Y, powers.T, cmap=cmap, shading="flat")
            for f in range(1,5):
                axes[1].axvline(x=f, color="white", linewidth=2,linestyle="dotted")
            x_pos = 4.8
            legend_freq = np.linspace(x_pos, x_pos+0.2, 5)
            rolling_lomb_classifier, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(centre_trials, powers)
            rolling_lomb_classifier_tiled = np.tile(rolling_lomb_classifier,(len(legend_freq),1))
            cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
            boundaries = [0, 1, 2, 3, 4]
            norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
            Y, X = np.meshgrid(centre_trials, legend_freq)
            axes[1].pcolormesh(X, Y, rolling_lomb_classifier_tiled, cmap=cmap, norm=norm, shading="flat")
            axes[1].set_ylabel('Centre Trial', fontsize=30, labelpad = 10)
            axes[1].spines['top'].set_visible(False)
            axes[1].spines['right'].set_visible(False)
            axes[1].set_xticks([0, 1, 2, 3, 4, 5])
            axes[1].set_yticks(y_tick_locs.tolist())
            axes[1].set_xlim([0.1,5])
            axes[1].set_ylim([min(centre_trials), max(centre_trials)])
            axes[1].yaxis.set_tick_params(labelsize=20)
            axes[1].xaxis.set_tick_params(labelsize=20)




            for f in range(1,6):
                axes[2].axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
            #axes[1].axvline(x=modal_frequency, color=modal_class_color, linewidth=3,linestyle="solid")
            subset_trial_numbers = np.asarray(processed_position_data["trial_number"])
            subset_mask = np.isin(centre_trials, subset_trial_numbers)
            subset_mask = np.vstack([subset_mask]*len(powers[0])).T
            subset_powers = powers.copy()
            subset_powers[subset_mask == False] = np.nan
            avg_subset_powers = np.nanmean(subset_powers, axis=0)
            sem_subset_powers = scipy.stats.sem(subset_powers, axis=0, nan_policy="omit")
            axes[2].fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color="black", alpha=0.3)
            axes[2].plot(frequency, avg_subset_powers, color="black", linewidth=3)
            allocentric_peak_freq, allocentric_peak_power, allo_i = get_allocentric_peak(frequency, avg_subset_powers, tolerance=0.05)
            egocentric_peak_freq, egocentric_peak_power, ego_i = get_egocentric_peak(frequency, avg_subset_powers, tolerance=0.05)
            axes[2].scatter(allocentric_peak_freq, allocentric_peak_power, color=Settings.allocentric_color, marker="v", s=200, zorder=10)
            axes[2].scatter(egocentric_peak_freq, egocentric_peak_power, color=Settings.egocentric_color, marker="v", s=200, zorder=10)
            axes[2].axhline(y=Settings.measured_far, color="red", linewidth=3, linestyle="dashed")
            axes[2].set_ylabel('Periodic Power', fontsize=30, labelpad = 10)
            axes[2].set_xlabel("Track Frequency", fontsize=25, labelpad = 10)
            axes[2].set_xlim([0.1,5.05])
            axes[2].set_xticks([1,2,3,4, 5])
            axes[2].set_yticks([0, np.round(axes[2].get_ylim()[1], 2)])
            axes[2].set_ylim(bottom=0)
            axes[2].yaxis.set_tick_params(labelsize=20)
            axes[2].xaxis.set_tick_params(labelsize=20)
            axes[2].spines['top'].set_visible(False)
            axes[2].spines['right'].set_visible(False)


            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + cluster_spike_data.session_id.iloc[0] + '_spatial_moving_lomb_scargle_periodogram_combined_Cluster_' + str(cluster_id) +'.png', dpi=300)
            plt.close()
    return


def plot_allo_minus_ego_component4(concantenated_dataframe,  save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    grid_cells = add_lomb_classifier(grid_cells, suffix="")
    grid_cells = add_session_number(grid_cells)
    grid_cells = extract_hit_success(grid_cells)

    p_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    d_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]

    for grids, name, in zip([p_grids, d_grids], ["P_cells", "D_cells"]):

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        allo_minus_ego_hit_proportions_b = np.asarray(grids["allo_minus_ego_hit_proportions_b"]); b_mask = ~np.isnan(allo_minus_ego_hit_proportions_b)
        allo_minus_ego_hit_proportions_nb = np.asarray(grids["allo_minus_ego_hit_proportions_nb"]); nb_mask = ~np.isnan(allo_minus_ego_hit_proportions_nb)
        mask = b_mask & nb_mask

        allo_minus_ego_hit_proportions_b = allo_minus_ego_hit_proportions_b[mask]
        allo_minus_ego_hit_proportions_nb = allo_minus_ego_hit_proportions_nb[mask]

        bad_bnb = ~np.logical_or(np.isnan(allo_minus_ego_hit_proportions_b), np.isnan(allo_minus_ego_hit_proportions_nb))
        bnb_p = stats.wilcoxon(np.compress(bad_bnb, allo_minus_ego_hit_proportions_b), np.compress(bad_bnb, allo_minus_ego_hit_proportions_nb))[1]

        allo_minus_ego_hit_proportions_b = allo_minus_ego_hit_proportions_b[~np.isnan(allo_minus_ego_hit_proportions_b)]
        allo_minus_ego_hit_proportions_nb = allo_minus_ego_hit_proportions_nb[~np.isnan(allo_minus_ego_hit_proportions_nb)]




        data = [allo_minus_ego_hit_proportions_b, allo_minus_ego_hit_proportions_nb]

        pts = np.linspace(0, np.pi * 2, 24)
        circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
        vert = np.r_[circ, circ[::-1] * .7]
        open_circle = mpl.path.Path(vert)

        x4 = 3 * np.ones(len(allo_minus_ego_hit_proportions_b))
        x5 = 4 * np.ones(len(allo_minus_ego_hit_proportions_nb))
        x = np.concatenate((x4, x5), axis=0)
        y = np.concatenate((allo_minus_ego_hit_proportions_b, allo_minus_ego_hit_proportions_nb), axis=0)
        sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0, clip_on=False)

        ax.bar(0, np.nanmedian(allo_minus_ego_hit_proportions_b), edgecolor="black", color="None", facecolor="None", linewidth=3, width=0.5)
        ax.bar(1, np.nanmedian(allo_minus_ego_hit_proportions_nb), edgecolor="blue", color="None", facecolor="None", linewidth=3, width=0.5)

        significance_bar(start=0, end=1, height=0.8, displaystring=get_p_text(bnb_p))

        ax.axhline(y=0, linestyle="dashed", color="black")
        ax.set_xticks([0,1, 2 ])
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticklabels(["Cued", "PI", " "])
        ax.set_ylabel("% Allocentric - % Egocentric\nduring_hmt_trials", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='major', labelsize=25)

        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/allo_minus_ego_coding_tt_'+name+'.png', dpi=300)
        plt.close()

    return

def plot_allo_minus_ego_component3(concantenated_dataframe,  save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    grid_cells = add_lomb_classifier(grid_cells, suffix="")
    grid_cells = add_session_number(grid_cells)
    grid_cells = extract_hit_success(grid_cells)

    p_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    d_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]

    for grids, name, in zip([p_grids, d_grids], ["P_cells", "D_cells"]):

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        allo_minus_ego_hit_proportions_nb = np.asarray(grids["allo_minus_ego_hit_proportions_nb"]); hit_mask = ~np.isnan(allo_minus_ego_hit_proportions_nb)
        allo_minus_ego_try_proportions_nb = np.asarray(grids["allo_minus_ego_try_proportions_nb"]); try_mask = ~np.isnan(allo_minus_ego_try_proportions_nb)
        allo_minus_ego_miss_proportions_nb = np.asarray(grids["allo_minus_ego_miss_proportions_nb"]); miss_mask = ~np.isnan(allo_minus_ego_miss_proportions_nb)
        hmt_mask = hit_mask & try_mask & miss_mask

        allo_minus_ego_hit_proportions_nb = allo_minus_ego_hit_proportions_nb[hmt_mask]
        allo_minus_ego_try_proportions_nb = allo_minus_ego_try_proportions_nb[hmt_mask]
        allo_minus_ego_miss_proportions_nb = allo_minus_ego_miss_proportions_nb[hmt_mask]

        bad_ht = ~np.logical_or(np.isnan(allo_minus_ego_hit_proportions_nb), np.isnan(allo_minus_ego_try_proportions_nb))
        bad_hm = ~np.logical_or(np.isnan(allo_minus_ego_hit_proportions_nb), np.isnan(allo_minus_ego_miss_proportions_nb))
        hit_try_p = stats.wilcoxon(np.compress(bad_ht, allo_minus_ego_hit_proportions_nb), np.compress(bad_ht, allo_minus_ego_try_proportions_nb))[1]
        hit_miss_p = stats.wilcoxon(np.compress(bad_hm, allo_minus_ego_hit_proportions_nb), np.compress(bad_hm, allo_minus_ego_miss_proportions_nb))[1]

        allo_minus_ego_hit_proportions_nb = allo_minus_ego_hit_proportions_nb[~np.isnan(allo_minus_ego_hit_proportions_nb)]
        allo_minus_ego_try_proportions_nb = allo_minus_ego_try_proportions_nb[~np.isnan(allo_minus_ego_try_proportions_nb)]
        allo_minus_ego_miss_proportions_nb = allo_minus_ego_miss_proportions_nb[~np.isnan(allo_minus_ego_miss_proportions_nb)]

        data = [allo_minus_ego_hit_proportions_nb, allo_minus_ego_try_proportions_nb, allo_minus_ego_miss_proportions_nb]

        pts = np.linspace(0, np.pi * 2, 24)
        circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
        vert = np.r_[circ, circ[::-1] * .7]
        open_circle = mpl.path.Path(vert)

        x4 = 3 * np.ones(len(allo_minus_ego_hit_proportions_nb))
        x5 = 4 * np.ones(len(allo_minus_ego_try_proportions_nb))
        x6 = 5 * np.ones(len(allo_minus_ego_miss_proportions_nb))
        x = np.concatenate((x4, x5, x6), axis=0)
        y = np.concatenate((allo_minus_ego_hit_proportions_nb, allo_minus_ego_try_proportions_nb, allo_minus_ego_miss_proportions_nb), axis=0)
        sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0, clip_on=False)

        colors=["green", "orange", "red", "green", "orange", "red"]
        boxprops = dict(linewidth=3, color='k')
        medianprops = dict(linewidth=3, color='k')
        capprops = dict(linewidth=3, color='k')
        whiskerprops = dict(linewidth=3, color='k')
        #box = ax.boxplot(data, positions=[0,1,2,3,4,5], boxprops=boxprops, medianprops=medianprops,
        #                 whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
        #for patch, color in zip(box['boxes'], colors):
        #    patch.set_facecolor(color)

        ax.bar(0, np.nanmedian(allo_minus_ego_hit_proportions_nb), edgecolor="green", color="None", facecolor="None", linewidth=3, width=0.5)
        ax.bar(1, np.nanmedian(allo_minus_ego_try_proportions_nb), edgecolor="orange", color="None", facecolor="None", linewidth=3, width=0.5)
        ax.bar(2, np.nanmedian(allo_minus_ego_miss_proportions_nb), edgecolor="red", color="None", facecolor="None", linewidth=3, width=0.5)

        #ax.bar(3, np.nanmean(allo_minus_ego_hit_proportions_nb), edgecolor="green", color="None", facecolor="None", linewidth=3, width=0.5)
        #ax.bar(4, np.nanmean(allo_minus_ego_try_proportions_nb), edgecolor="orange", color="None", facecolor="None", linewidth=3, width=0.5)
        #ax.bar(5, np.nanmean(allo_minus_ego_miss_proportions_nb), edgecolor="red", color="None", facecolor="None", linewidth=3, width=0.5)

        #vp = ax.violinplot(data, [0, 1,2], widths=0.5,
        #                   showmeans=False, showmedians=True, showextrema=False)
        #for i, pc in enumerate(vp['bodies']):
        #    pc.set_facecolor(colors[i])
        #    pc.set_edgecolor('black')
        #    pc.set_alpha(1)

        significance_bar(start=0, end=1, height=0.8, displaystring=get_p_text(hit_try_p))
        significance_bar(start=0, end=2, height=1, displaystring=get_p_text(hit_miss_p))


        ax.axhline(y=0, linestyle="dashed", color="black")
        ax.set_xticks([0,1,2])
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticklabels(["Hit", "Try", "Miss"])
        ax.set_ylabel("% Allocentric - % Egocentric\nduring_hmt_trials", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='major', labelsize=25)

        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/allo_minus_ego_coding_'+name+'.png', dpi=300)
        plt.close()

    return

def plot_allo_minus_ego_component2(concantenated_dataframe,  save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    grid_cells = add_lomb_classifier(grid_cells, suffix="")
    grid_cells = add_session_number(grid_cells)
    grid_cells = extract_hit_success(grid_cells)

    p_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    d_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]

    for grids, name, in zip([grid_cells, p_grids, d_grids], ["all_cells", "P_cells", "D_cells"]):

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        allo_minus_ego_hit_proportions_b = np.asarray(grids["allo_minus_ego_hit_proportions_b"])
        allo_minus_ego_hit_proportions_nb = np.asarray(grids["allo_minus_ego_hit_proportions_nb"])
        allo_minus_ego_try_proportions_b = np.asarray(grids["allo_minus_ego_try_proportions_b"])
        allo_minus_ego_try_proportions_nb = np.asarray(grids["allo_minus_ego_try_proportions_nb"])
        allo_minus_ego_miss_proportions_b = np.asarray(grids["allo_minus_ego_miss_proportions_b"])
        allo_minus_ego_miss_proportions_nb = np.asarray(grids["allo_minus_ego_miss_proportions_nb"])


        allo_minus_ego_hit_proportions_b = allo_minus_ego_hit_proportions_b[~np.isnan(allo_minus_ego_hit_proportions_b)]
        allo_minus_ego_hit_proportions_nb = allo_minus_ego_hit_proportions_nb[~np.isnan(allo_minus_ego_hit_proportions_nb)]
        allo_minus_ego_try_proportions_b = allo_minus_ego_try_proportions_b[~np.isnan(allo_minus_ego_try_proportions_b)]
        allo_minus_ego_try_proportions_nb = allo_minus_ego_try_proportions_nb[~np.isnan(allo_minus_ego_try_proportions_nb)]
        allo_minus_ego_miss_proportions_b = allo_minus_ego_miss_proportions_b[~np.isnan(allo_minus_ego_miss_proportions_b)]
        allo_minus_ego_miss_proportions_nb = allo_minus_ego_miss_proportions_nb[~np.isnan(allo_minus_ego_miss_proportions_nb)]

        data = [allo_minus_ego_hit_proportions_b, allo_minus_ego_try_proportions_b, allo_minus_ego_miss_proportions_b,
                allo_minus_ego_hit_proportions_nb, allo_minus_ego_try_proportions_nb, allo_minus_ego_miss_proportions_nb]

        pts = np.linspace(0, np.pi * 2, 24)
        circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
        vert = np.r_[circ, circ[::-1] * .7]
        open_circle = mpl.path.Path(vert)

        x1 = 0 * np.ones(len(allo_minus_ego_hit_proportions_b))
        x2 = 1 * np.ones(len(allo_minus_ego_try_proportions_b))
        x3 = 2 * np.ones(len(allo_minus_ego_miss_proportions_b))
        x4 = 3 * np.ones(len(allo_minus_ego_hit_proportions_nb))
        x5 = 4 * np.ones(len(allo_minus_ego_try_proportions_nb))
        x6 = 5 * np.ones(len(allo_minus_ego_miss_proportions_nb))
        x = np.concatenate((x1, x2, x3, x4, x5, x6), axis=0)
        y = np.concatenate((allo_minus_ego_hit_proportions_b, allo_minus_ego_try_proportions_b, allo_minus_ego_miss_proportions_b,
                            allo_minus_ego_hit_proportions_nb, allo_minus_ego_try_proportions_nb, allo_minus_ego_miss_proportions_nb), axis=0)
        sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0, clip_on=False)

        colors=["green", "orange", "red", "green", "orange", "red"]
        boxprops = dict(linewidth=3, color='k')
        medianprops = dict(linewidth=3, color='k')
        capprops = dict(linewidth=3, color='k')
        whiskerprops = dict(linewidth=3, color='k')
        #box = ax.boxplot(data, positions=[0,1,2,3,4,5], boxprops=boxprops, medianprops=medianprops,
        #                 whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
        #for patch, color in zip(box['boxes'], colors):
        #    patch.set_facecolor(color)

        #ax.bar(0, np.nanmean(allo_minus_ego_hit_proportions_b), edgecolor="green", color="None", facecolor="None", linewidth=3, width=0.5)
        #ax.bar(1, np.nanmean(allo_minus_ego_try_proportions_b), edgecolor="orange", color="None", facecolor="None", linewidth=3, width=0.5)
        #ax.bar(2, np.nanmean(allo_minus_ego_miss_proportions_b), edgecolor="red", color="None", facecolor="None", linewidth=3, width=0.5)

        #ax.bar(3, np.nanmean(allo_minus_ego_hit_proportions_nb), edgecolor="green", color="None", facecolor="None", linewidth=3, width=0.5)
        #ax.bar(4, np.nanmean(allo_minus_ego_try_proportions_nb), edgecolor="orange", color="None", facecolor="None", linewidth=3, width=0.5)
        #ax.bar(5, np.nanmean(allo_minus_ego_miss_proportions_nb), edgecolor="red", color="None", facecolor="None", linewidth=3, width=0.5)

        vp = ax.violinplot(data, [0, 1,2,3,4,5], widths=0.5,
                           showmeans=False, showmedians=True, showextrema=False)
        #for i, pc in enumerate(vp['bodies']):
        #    pc.set_facecolor(colors[i])
        #    pc.set_edgecolor('black')
        #    pc.set_alpha(1)

        p = stats.mannwhitneyu(allo_minus_ego_hit_proportions_b, allo_minus_ego_hit_proportions_nb)[1]
        significance_bar(start=0, end=3, height=1, displaystring=get_p_text(p))
        p = stats.mannwhitneyu(allo_minus_ego_hit_proportions_nb, allo_minus_ego_try_proportions_nb)[1]
        significance_bar(start=3, end=4, height=0.8, displaystring=get_p_text(p))
        p = stats.mannwhitneyu(allo_minus_ego_hit_proportions_nb, allo_minus_ego_miss_proportions_nb)[1]
        significance_bar(start=3, end=5, height=0.9, displaystring=get_p_text(p))


        ax.axhline(y=0, linestyle="dashed", color="black")
        ax.set_xticks([1,5])
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticklabels(["Cued", "PI"])
        ax.set_ylabel("% Allocentric - % Egocentric\nduring_hmt_trials", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='major', labelsize=25)

        ax.tick_params(axis='both', which='both', labelsize=20)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/allo_minus_ego_coding_'+name+'.png', dpi=300)
        plt.close()

    return

def plot_allo_minus_ego_component(concantenated_dataframe,  save_path):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    grid_cells = add_lomb_classifier(grid_cells, suffix="")
    grid_cells = add_session_number(grid_cells)
    grid_cells = extract_hit_success(grid_cells)

    p_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    d_grids = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]

    for grids, name, in zip([grid_cells, p_grids, d_grids], ["all_cells", "P_cells", "D_cells"]):

        for hmt in ["hit", "miss", "try"]:
            fig, ax = plt.subplots(1,1, figsize=(6,6))

            allo_minus_ego_hmt_proportions_b = np.asarray(grids["allo_minus_ego_"+hmt+"_proportions_b"])
            allo_minus_ego_hmt_proportions_nb = np.asarray(grids["allo_minus_ego_"+hmt+"_proportions_nb"])

            allo_minus_ego_hmt_proportions_b = allo_minus_ego_hmt_proportions_b[~np.isnan(allo_minus_ego_hmt_proportions_b)]
            allo_minus_ego_hmt_proportions_nb = allo_minus_ego_hmt_proportions_nb[~np.isnan(allo_minus_ego_hmt_proportions_nb)]

            data = [allo_minus_ego_hmt_proportions_b, allo_minus_ego_hmt_proportions_nb]
            p = stats.mannwhitneyu(allo_minus_ego_hmt_proportions_b, allo_minus_ego_hmt_proportions_nb)[1]
            vp = ax.violinplot(data, [0, 1], widths=0.5,
                               showmeans=False, showmedians=True, showextrema=False)

            x1 = 0 * np.ones(len(allo_minus_ego_hmt_proportions_b))
            x2 = 1 * np.ones(len(allo_minus_ego_hmt_proportions_nb))
            #Combine the sampled data together
            x = np.concatenate((x1, x2), axis=0)
            y = np.concatenate((allo_minus_ego_hmt_proportions_b, allo_minus_ego_hmt_proportions_nb), axis=0)
            sns.stripplot(x, y, ax=ax, color="black", jitter=0.2)
            significance_bar(start=0, end=1, height=1, displaystring=get_p_text(p))

            ax.axhline(y=0, linestyle="dashed", color="black")
            ax.set_xticks([0,1])
            ax.set_yticks([-1, 0, 1])
            ax.set_ylim([-1, 1])
            ax.set_xticklabels(["Cued", "PI"])
            ax.set_ylabel("% Allocentric - % Egocentric\nduring "+hmt+" trials", fontsize=20, labelpad=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_tick_params(length=0)
            ax.tick_params(axis='both', which='major', labelsize=25)

            ax.tick_params(axis='both', which='both', labelsize=20)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            plt.savefig(save_path + '/allo_minus_ego_coding_'+name+'_'+hmt+'.png', dpi=300)
            plt.close()

    return

def get_percentage_hit_column(df, tt):
    percentage_hits = []
    for index, cluster_df in df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        percentage = cluster_df["percentage_hits"].iloc[0][tt]
        percentage_hits.append(percentage)

    percentage_hits = np.array(percentage_hits)
    return percentage_hits[~np.isnan(percentage_hits)]


def get_percentage_from_rolling_classification2(df, code, tt, hmt, remove_last_and_first_from_streak=False):

    percentages = []
    for index, cluster_df in df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        behaviours = np.array(cluster_df["behaviour_hit_try_miss"].iloc[0])
        trial_types = np.array(cluster_df["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cluster_df["rolling:classifier_by_trial_number"].iloc[0])


        # remove first and last classification in streak
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

        valid_trials = rolling_classifiers[(behaviours == hmt) & (trial_types == tt) & (rolling_classifiers != "nan")]

        if len(valid_trials) == 0:
            percentage = np.nan
        else:
            percentage = 100*(len(valid_trials[valid_trials==code])/len(valid_trials))
        percentages.append(percentage)

    return np.array(percentages)


def get_percentage_from_rolling_classification(df, code, tt,
                                               remove_last_and_first_from_streak=False,
                                               rolling_classsifier_collumn_name="rolling:classifier_by_trial_number"):

    percentage_hits = []
    for index, cluster_df in df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        behaviours = np.array(cluster_df["behaviour_hit_try_miss"].iloc[0])
        trial_types = np.array(cluster_df["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cluster_df[rolling_classsifier_collumn_name].iloc[0])

        # remove first and last classification in streak
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

        valid_trials = behaviours[(rolling_classifiers == code) & (trial_types == tt) & (rolling_classifiers != "nan")]

        if len(valid_trials) == 0:
            percentage = np.nan
        else:
            percentage = 100*(len(valid_trials[valid_trials=="hit"])/len(valid_trials))
        percentage_hits.append(percentage)

    return np.array(percentage_hits)



def compile_remapped_grid_cell_stop_histogram(df, tt):
    #for index, cluster_df in df.iterrows():
    #    cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
    #    full_session_id = cluster_df["full_session_id_vr"].iloc[0]
    #    print(full_session_id)

    for index, cluster_df in df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        session_id = cluster_df["session_id"].iloc[0]
        full_session_id = cluster_df["full_session_id_vr"].iloc[0]
        cluster_id = cluster_df["cluster_id"].iloc[0]
        path = full_session_id+"/MountainSort/Figures/stop_histogram_for_coding_epochs/stop_histogram_c_"+str(cluster_id)+"_tt_"+str(tt)+".png"
        if os.path.exists(path):
            shutil.copyfile(path, "/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/remapped_grid_cell_stop_histograms/"+session_id+"_"+str(cluster_id)+"_tt_"+str(tt)+".png")



def get_peak_amp_and_locs(stop_histogram_2d_numpy_array, bin_centres):
    peak_locs = []; peak_amps=[]; RZ_peak_amps = []
    for i in range(len(stop_histogram_2d_numpy_array)):
        # check for nans
        if np.sum(np.isnan(stop_histogram_2d_numpy_array[i]))>0:
            peak_locs.append(np.nan)
            peak_amps.append(np.nan)
            RZ_peak_amps.append(np.nan)
        else:
            RZ_peak_i = np.nanargmax(stop_histogram_2d_numpy_array[i][89:109])
            peak_i = np.nanargmax(stop_histogram_2d_numpy_array[i])
            peak_amp = stop_histogram_2d_numpy_array[i][peak_i]
            RZ_peak_amp = stop_histogram_2d_numpy_array[i][RZ_peak_i+89]
            peak_loc = bin_centres[peak_i]
            peak_locs.append(peak_loc)
            peak_amps.append(peak_amp)
            RZ_peak_amps.append(RZ_peak_amp)
    return np.array(peak_amps), np.array(peak_locs), np.array(RZ_peak_amps)

def plot_stop_variability_stable(combined_df, save_path, drop_bb_stops=True):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    Position_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    Distance_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]

    stable_position_grid_cells = Position_grid_cells[Position_grid_cells["rolling:proportion_encoding_position"] > 0.85]
    stable_distance_grid_cells = Distance_grid_cells[Distance_grid_cells["rolling:proportion_encoding_distance"] > 0.85]

    stable_position_grid_cells = drop_duplicate_sessions(stable_position_grid_cells)
    stable_distance_grid_cells = drop_duplicate_sessions(stable_distance_grid_cells)

    print("n session for stable position grid cells, n = ", str(len(stable_position_grid_cells)))
    print("n session for stable distance grid cells, n = ", str(len(stable_distance_grid_cells)))

    # trial type 0
    stable_position_grid_cells_stop_histogram_0, _, bin_centres, _, _, stop_var_p_0 = get_stop_histogram(stable_position_grid_cells, tt=0, coding_scheme=None, shuffle=False, drop_bb_stops=drop_bb_stops)
    stable_distance_grid_cells_stop_histogram_0, _, bin_centres, _, _, stop_var_d_0 = get_stop_histogram(stable_distance_grid_cells, tt=0, coding_scheme=None, shuffle=False, drop_bb_stops=drop_bb_stops)
    stop_var_p_0 = np.array(stop_var_p_0)
    stop_var_d_0 = np.array(stop_var_d_0)

    # trial type 1
    stable_position_grid_cells_stop_histogram_1, _, bin_centres, _, _, stop_var_p_1 = get_stop_histogram(stable_position_grid_cells, tt=1, coding_scheme=None, shuffle=False, drop_bb_stops=drop_bb_stops)
    stable_distance_grid_cells_stop_histogram_1, _, bin_centres, _, _, stop_var_d_1 = get_stop_histogram(stable_distance_grid_cells, tt=1, coding_scheme=None, shuffle=False, drop_bb_stops=drop_bb_stops)
    stop_var_p_1 = np.array(stop_var_p_1)
    stop_var_d_1 = np.array(stop_var_d_1)

    # plot and compare similarly to the peak amplitudes hits
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color]
    data = [stop_var_p_0, stop_var_d_0, stop_var_p_1, stop_var_d_1]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "B", "NB", "NB"])
    #ax.set_yticks([0,100])
    #ax.set_ylim([0, 0.35])
    ax.set_ylim(bottom=0)
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/stop_variablity_stable_grid_cells.png', dpi=300)
    plt.close()

    print("comparing variability of stops between postion and distance encoding grid cells for beaconed trials, df=",str(len(data[0])+len(data[1])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0],data[1])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0],data[1])[0]))
    print("comparing variability of stops between postion and distance encoding grid cells for non beaconed trials, df=",str(len(data[2])+len(data[3])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[2],data[3])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[2],data[3])[0]))


def plot_number_of_trials_stable(combined_df, save_path, drop_bb_stops=True):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    Position_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    Distance_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]

    stable_position_grid_cells = Position_grid_cells[Position_grid_cells["rolling:proportion_encoding_position"] > 0.85]
    stable_distance_grid_cells = Distance_grid_cells[Distance_grid_cells["rolling:proportion_encoding_distance"] > 0.85]

    stable_position_grid_cells = drop_duplicate_sessions(stable_position_grid_cells)
    stable_distance_grid_cells = drop_duplicate_sessions(stable_distance_grid_cells)

    print("n session for stable position grid cells, n = ", str(len(stable_position_grid_cells)))
    print("n session for stable distance grid cells, n = ", str(len(stable_distance_grid_cells)))

    # trial type 0
    stable_position_grid_cells_stop_histogram_0, _, bin_centres, n_trials_p_0, n_stops_p_0, _ = get_stop_histogram(stable_position_grid_cells, tt=0, coding_scheme=None, shuffle=False, drop_bb_stops=drop_bb_stops)
    stable_distance_grid_cells_stop_histogram_0, _, bin_centres, n_trials_d_0, n_stops_d_0, _ = get_stop_histogram(stable_distance_grid_cells, tt=0, coding_scheme=None, shuffle=False, drop_bb_stops=drop_bb_stops)
    stable_position_grid_cells_stop_histogram_0 = np.array(stable_position_grid_cells_stop_histogram_0)
    stable_distance_grid_cells_stop_histogram_0 = np.array(stable_distance_grid_cells_stop_histogram_0)
    n_trials_p_0 = np.array(n_trials_p_0)
    n_trials_d_0 = np.array(n_trials_d_0)
    n_stops_p_0 = np.array(n_stops_p_0)
    n_stops_d_0 = np.array(n_stops_d_0)

    # trial type 1
    stable_position_grid_cells_stop_histogram_1, _, bin_centres, n_trials_p_1, n_stops_p_1, _ = get_stop_histogram(stable_position_grid_cells, tt=1, coding_scheme=None, shuffle=False)
    stable_distance_grid_cells_stop_histogram_1, _, bin_centres, n_trials_d_1, n_stops_d_1, _ = get_stop_histogram(stable_distance_grid_cells, tt=1, coding_scheme=None, shuffle=False)
    stable_position_grid_cells_stop_histogram_1 = np.array(stable_position_grid_cells_stop_histogram_1)
    stable_distance_grid_cells_stop_histogram_1 = np.array(stable_distance_grid_cells_stop_histogram_1)
    n_trials_p_1 = np.array(n_trials_p_1)
    n_trials_d_1 = np.array(n_trials_d_1)
    n_stops_p_1 = np.array(n_stops_p_1)
    n_stops_d_1 = np.array(n_stops_d_1)

    # get n stops per trial in RZ
    p_1 = n_stops_p_1/n_trials_p_1
    d_1 = n_stops_d_1/n_trials_d_1
    p_0 = n_stops_p_0/n_trials_p_0
    d_0 = n_stops_d_0/n_trials_d_0

    # plot and compare similarly to the peak amplitudes hits
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color]
    data = [p_0, d_0, p_1, d_1]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "B", "NB", "NB"])
    #ax.set_yticks([0,100])
    #ax.set_ylim([0, 0.35])
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/n_stops_stable_grid_cells.png', dpi=300)
    plt.close()

    print("comparing n stops between postion and distance encoding grid cells for beaconed trials, df=",str(len(data[0])+len(data[1])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0],data[1])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0],data[1])[0]))
    print("comparing n stops between postion and distance encoding grid cells for non beaconed trials, df=",str(len(data[2])+len(data[3])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[2],data[3])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[2],data[3])[0]))

def plot_stop_variability_remapped(combined_df, save_path, suffix="", account_for="", drop_bb_stops=True):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    # trial type 0
    remapped_position_grid_cells_stop_histogram_0, _, bin_centres, _, _, stop_var_p_0 = get_stop_histogram(grid_cells, tt=0, coding_scheme="P", shuffle=False, account_for=account_for, drop_bb_stops=drop_bb_stops)
    remapped_distance_grid_cells_stop_histogram_0, _, bin_centres, _, _, stop_var_d_0 = get_stop_histogram(grid_cells, tt=0, coding_scheme="D", shuffle=False, account_for=account_for, drop_bb_stops=drop_bb_stops)
    stop_var_p_0 = np.array(stop_var_p_0)
    stop_var_d_0 = np.array(stop_var_d_0)

    # trial type 1
    remapped_position_grid_cells_stop_histogram_1, _, bin_centres, _, _, stop_var_p_1 = get_stop_histogram(grid_cells, tt=1, coding_scheme="P", shuffle=False, account_for=account_for, drop_bb_stops=drop_bb_stops)
    remapped_distance_grid_cells_stop_histogram_1, _, bin_centres, _, _, stop_var_d_1 = get_stop_histogram(grid_cells, tt=1, coding_scheme="D", shuffle=False, account_for=account_for, drop_bb_stops=drop_bb_stops)
    stop_var_p_1 = np.array(stop_var_p_1)
    stop_var_d_1 = np.array(stop_var_d_1)

    b_mask = ~np.isnan(stop_var_p_0) & ~np.isnan(stop_var_d_0)
    nb_mask = ~np.isnan(stop_var_p_1) & ~np.isnan(stop_var_d_1)

    # plot and compare similarly to the peak amplitudes hits
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color]
    data = [stop_var_p_0[b_mask], stop_var_d_0[b_mask], stop_var_p_1[nb_mask], stop_var_d_1[nb_mask]]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "B", "NB", "NB"])
    #ax.set_yticks([0, 0.1, 0.2, 0.3])
    #ax.set_ylim([0, 0.35])
    ax.set_ylim(bottom=0)
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/stop_variability_remapped_grid_cells'+account_for+suffix+'.png', dpi=300)
    plt.close()

    print("comparing stop variability between postion and distance remapped encoding grid cells for beaconed trials_",account_for,", df=",str(len(data[0])+len(data[1])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0],data[1])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0],data[1])[0]))
    print("comparing stop variability between postion and distance remapped encoding grid cells for non beaconed trials_",account_for,", df=",str(len(data[2])+len(data[3])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[2],data[3])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[2],data[3])[0]))
    return

def plot_number_of_trials_remapped(combined_df, save_path, suffix="", account_for="", drop_bb_stops=True):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    # trial type 0
    remapped_position_grid_cells_stop_histogram_0, _, bin_centres, n_trials_p_0, n_stops_p_0, _ = get_stop_histogram(grid_cells, tt=0, coding_scheme="P", shuffle=False, account_for=account_for, drop_bb_stops=drop_bb_stops)
    remapped_distance_grid_cells_stop_histogram_0, _, bin_centres, n_trials_d_0, n_stops_d_0, _ = get_stop_histogram(grid_cells, tt=0, coding_scheme="D", shuffle=False, account_for=account_for, drop_bb_stops=drop_bb_stops)
    remapped_position_grid_cells_stop_histogram_0 = np.array(remapped_position_grid_cells_stop_histogram_0)
    remapped_distance_grid_cells_stop_histogram_0 = np.array(remapped_distance_grid_cells_stop_histogram_0)
    n_trials_p_0 = np.array(n_trials_p_0)
    n_stops_p_0 = np.array(n_stops_p_0)
    n_trials_d_0 = np.array(n_trials_d_0)
    n_stops_d_0 = np.array(n_stops_d_0)

    # trial type 1
    remapped_position_grid_cells_stop_histogram_1, _, bin_centres, n_trials_p_1, n_stops_p_1, _ = get_stop_histogram(grid_cells, tt=1, coding_scheme="P", shuffle=False, account_for=account_for, drop_bb_stops=drop_bb_stops)
    remapped_distance_grid_cells_stop_histogram_1, _, bin_centres, n_trials_d_1, n_stops_d_1, _ = get_stop_histogram(grid_cells, tt=1, coding_scheme="D", shuffle=False, account_for=account_for, drop_bb_stops=drop_bb_stops)
    remapped_position_grid_cells_stop_histogram_1 = np.array(remapped_position_grid_cells_stop_histogram_1)
    remapped_distance_grid_cells_stop_histogram_1 = np.array(remapped_distance_grid_cells_stop_histogram_1)
    n_trials_p_1 = np.array(n_trials_p_1)
    n_stops_p_1 = np.array(n_stops_p_1)
    n_trials_d_1 = np.array(n_trials_d_1)
    n_stops_d_1 = np.array(n_stops_d_1)

    # get n stops per trial in RZ
    p_1 = n_stops_p_1/n_trials_p_1
    d_1 = n_stops_d_1/n_trials_d_1
    p_0 = n_stops_p_0/n_trials_p_0
    d_0 = n_stops_d_0/n_trials_d_0

    b_mask = ~np.isnan(p_0) & ~np.isnan(d_0)
    nb_mask = ~np.isnan(p_1) & ~np.isnan(d_1)

    # plot and compare similarly to the peak amplitudes hits
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color]
    data = [p_0[b_mask], d_0[b_mask], p_1[nb_mask], d_1[nb_mask]]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "B", "NB", "NB"])
    #ax.set_yticks([0, 0.1, 0.2, 0.3])
    #ax.set_ylim([0, 0.35])
    ax.set_ylim(bottom=0, top=10)
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/n_stops_remapped_grid_cells_'+account_for+str(suffix)+'.png', dpi=300)
    plt.close()

    print("comparing n stops between postion and distance remapped encoding grid cells for beaconed trials_",account_for,", df=",str(len(data[0])+len(data[1])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0],data[1])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0],data[1])[0]))
    print("comparing n stops between postion and distance remapped encoding grid cells for non beaconed trials_",account_for,", df=",str(len(data[2])+len(data[3])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[2],data[3])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[2],data[3])[0]))
    return

def plot_stop_peak_stop_location_and_height_stable(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    Position_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    Distance_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]

    stable_position_grid_cells = Position_grid_cells[Position_grid_cells["rolling:proportion_encoding_position"] > 0.85]
    stable_distance_grid_cells = Distance_grid_cells[Distance_grid_cells["rolling:proportion_encoding_distance"] > 0.85]

    stable_position_grid_cells = drop_duplicate_sessions(stable_position_grid_cells)
    stable_distance_grid_cells = drop_duplicate_sessions(stable_distance_grid_cells)

    print("n session for stable position grid cells, n = ", str(len(stable_position_grid_cells)))
    print("n session for stable distance grid cells, n = ", str(len(stable_distance_grid_cells)))

    # trial type 0
    stable_position_grid_cells_stop_histogram_0, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_grid_cells, tt=0, coding_scheme=None, shuffle=False)
    stable_distance_grid_cells_stop_histogram_0, _, bin_centres, _, _ , _= get_stop_histogram(stable_distance_grid_cells, tt=0, coding_scheme=None, shuffle=False)
    stable_position_grid_cells_shuffled_histogram_0, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_grid_cells, tt=0, coding_scheme=None, shuffle=True)
    stable_distance_grid_cells_shuffled_histogram_0, _, bin_centres, _, _, _ = get_stop_histogram(stable_distance_grid_cells, tt=0, coding_scheme=None, shuffle=True)
    stable_position_grid_cells_stop_histogram_0 = np.array(stable_position_grid_cells_stop_histogram_0)
    stable_distance_grid_cells_stop_histogram_0 = np.array(stable_distance_grid_cells_stop_histogram_0)
    stable_position_grid_cells_shuffled_histogram_0 = np.array(stable_position_grid_cells_shuffled_histogram_0)
    stable_distance_grid_cells_shuffled_histogram_0 = np.array(stable_distance_grid_cells_shuffled_histogram_0)

    # apply normalisation with baseline
    stable_position_grid_cells_stop_histogram_0 = stable_position_grid_cells_stop_histogram_0-stable_position_grid_cells_shuffled_histogram_0
    stable_distance_grid_cells_stop_histogram_0 = stable_distance_grid_cells_stop_histogram_0-stable_distance_grid_cells_shuffled_histogram_0

    # get amplitudes of peaks and locations of peaks
    PG_tt_0_peak_amps, PG_tt_0_peak_locs, PG_tt_0_RZ_peak_amps = get_peak_amp_and_locs(stable_position_grid_cells_stop_histogram_0, bin_centres)
    DG_tt_0_peak_amps, DG_tt_0_peak_locs, DG_tt_0_RZ_peak_amps = get_peak_amp_and_locs(stable_distance_grid_cells_stop_histogram_0, bin_centres)

    # trial type 1
    stable_position_grid_cells_stop_histogram_1, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_grid_cells, tt=1, coding_scheme=None, shuffle=False)
    stable_distance_grid_cells_stop_histogram_1, _, bin_centres, _, _, _ = get_stop_histogram(stable_distance_grid_cells, tt=1, coding_scheme=None, shuffle=False)
    stable_position_grid_cells_shuffled_histogram_1, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_grid_cells, tt=1, coding_scheme=None, shuffle=True)
    stable_distance_grid_cells_shuffled_histogram_1, _, bin_centres, _, _, _ = get_stop_histogram(stable_distance_grid_cells, tt=1, coding_scheme=None, shuffle=True)
    stable_position_grid_cells_stop_histogram_1 = np.array(stable_position_grid_cells_stop_histogram_1)
    stable_distance_grid_cells_stop_histogram_1 = np.array(stable_distance_grid_cells_stop_histogram_1)
    stable_position_grid_cells_shuffled_histogram_1 = np.array(stable_position_grid_cells_shuffled_histogram_1)
    stable_distance_grid_cells_shuffled_histogram_1 = np.array(stable_distance_grid_cells_shuffled_histogram_1)

    # apply normalisation with baseline
    stable_position_grid_cells_stop_histogram_1 = stable_position_grid_cells_stop_histogram_1-stable_position_grid_cells_shuffled_histogram_1
    stable_distance_grid_cells_stop_histogram_1 = stable_distance_grid_cells_stop_histogram_1-stable_distance_grid_cells_shuffled_histogram_1

    # get amplitudes of peaks and locations of peaks
    PG_tt_1_peak_amps, PG_tt_1_peak_locs, PG_tt_1_RZ_peak_amps = get_peak_amp_and_locs(stable_position_grid_cells_stop_histogram_1, bin_centres)
    DG_tt_1_peak_amps, DG_tt_1_peak_locs, DG_tt_1_RZ_peak_amps = get_peak_amp_and_locs(stable_distance_grid_cells_stop_histogram_1, bin_centres)

    # plot and compare similarly to the peak amplitudes hits
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color]
    data = [PG_tt_0_RZ_peak_amps, DG_tt_0_RZ_peak_amps, PG_tt_1_RZ_peak_amps, DG_tt_1_RZ_peak_amps]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "B", "NB", "NB"])
    #ax.set_yticks([0, 0.1, 0.2, 0.3])
    #ax.set_ylim([0, 0.35])
    ax.set_ylim(bottom=0)
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/peak_minus_baseline_for_stable_grid_cells.png', dpi=300)
    plt.close()

    print("comparing peak_amps between postion and distance encoding grid cells for beaconed trials, df=",str(len(data[0])+len(data[2])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0],data[2])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0],data[2])[0]))
    print("comparing peak_amps between postion and distance encoding grid cells for non beaconed trials, df=",str(len(data[1])+len(data[3])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[1],data[3])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[1],data[3])[0]))



    # plot and compare similarly to the peak amplitudes hits
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color, Settings.allocentric_color, Settings.egocentric_color]
    data = [PG_tt_0_peak_locs, DG_tt_0_peak_locs, PG_tt_1_peak_locs, DG_tt_1_peak_locs]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "NB", "B", "NB"])
    #ax.set_yticks([-1, 0, 1])
    ax.axhspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axhspan(90, 110, xmin=0, xmax=0.5, facecolor='DarkGreen', alpha=.25, linewidth =0)
    ax.axhline(y=90, xmin=0.5, xmax=1, color="black", linestyle="dotted", linewidth=1)
    ax.axhline(y=110, xmin=0.5, xmax=1, color="black", linestyle="dotted", linewidth=1)
    ax.set_ylim([0, 130])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/peak_location_for_stable_grid_cells.png', dpi=300)
    plt.close()

    print("comparing peak_locations between postion and distance encoding grid cells for beaconed trials, df=",str(len(data[0])+len(data[2])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[0],data[2])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[0],data[2])[0]))
    print("comparing peak_locations between postion and distance encoding grid cells for non beaconed trials, df=",str(len(data[1])+len(data[3])-2), ", p= ", str(scipy.stats.mannwhitneyu(data[1],data[3])[1]), ", t= ", str(scipy.stats.mannwhitneyu(data[1],data[3])[0]))
    return


def plot_stop_peak_stop_location_and_height_remapped(combined_df, save_path, suffix=""):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    # trial type 0
    remapped_position_grid_cells_stop_histogram_0, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=0, coding_scheme="P", shuffle=False)
    remapped_distance_grid_cells_stop_histogram_0, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=0, coding_scheme="D", shuffle=False)
    remapped_position_grid_cells_shuffled_histogram_0, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=0, coding_scheme="P", shuffle=True)
    remapped_distance_grid_cells_shuffled_histogram_0, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=0,coding_scheme="D", shuffle=True)
    remapped_position_grid_cells_stop_histogram_0 = np.array(remapped_position_grid_cells_stop_histogram_0)
    remapped_distance_grid_cells_stop_histogram_0 = np.array(remapped_distance_grid_cells_stop_histogram_0)
    remapped_position_grid_cells_shuffled_histogram_0 = np.array(remapped_position_grid_cells_shuffled_histogram_0)
    remapped_distance_grid_cells_shuffled_histogram_0 = np.array(remapped_distance_grid_cells_shuffled_histogram_0)

    # apply normalisation with baseline
    remapped_position_grid_cells_stop_histogram_0 = remapped_position_grid_cells_stop_histogram_0-remapped_position_grid_cells_shuffled_histogram_0
    remapped_distance_grid_cells_stop_histogram_0 = remapped_distance_grid_cells_stop_histogram_0-remapped_distance_grid_cells_shuffled_histogram_0

    # get amplitudes of peaks and locations of peaks
    PG_tt_0_peak_amps, PG_tt_0_peak_locs, PG_tt_0_RZ_peak_amps = get_peak_amp_and_locs(remapped_position_grid_cells_stop_histogram_0, bin_centres)
    DG_tt_0_peak_amps, DG_tt_0_peak_locs, DG_tt_0_RZ_peak_amps = get_peak_amp_and_locs(remapped_distance_grid_cells_stop_histogram_0, bin_centres)

    # trial type 1
    remapped_position_grid_cells_stop_histogram_1, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=1, coding_scheme="P", shuffle=False)
    remapped_distance_grid_cells_stop_histogram_1, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=1, coding_scheme="D", shuffle=False)
    remapped_position_grid_cells_shuffled_histogram_1, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=1, coding_scheme="P", shuffle=True)
    remapped_distance_grid_cells_shuffled_histogram_1, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=1, coding_scheme="D", shuffle=True)
    remapped_position_grid_cells_stop_histogram_1 = np.array(remapped_position_grid_cells_stop_histogram_1)
    remapped_distance_grid_cells_stop_histogram_1 = np.array(remapped_distance_grid_cells_stop_histogram_1)
    remapped_position_grid_cells_shuffled_histogram_1 = np.array(remapped_position_grid_cells_shuffled_histogram_1)
    remapped_distance_grid_cells_shuffled_histogram_1 = np.array(remapped_distance_grid_cells_shuffled_histogram_1)

    # apply normalisation with baseline
    remapped_position_grid_cells_stop_histogram_1 = remapped_position_grid_cells_stop_histogram_1-remapped_position_grid_cells_shuffled_histogram_1
    remapped_distance_grid_cells_stop_histogram_1 = remapped_distance_grid_cells_stop_histogram_1-remapped_distance_grid_cells_shuffled_histogram_1

    # get amplitudes of peaks and locations of peaks
    PG_tt_1_peak_amps, PG_tt_1_peak_locs, PG_tt_1_RZ_peak_amps = get_peak_amp_and_locs(remapped_position_grid_cells_stop_histogram_1, bin_centres)
    DG_tt_1_peak_amps, DG_tt_1_peak_locs, DG_tt_1_RZ_peak_amps = get_peak_amp_and_locs(remapped_distance_grid_cells_stop_histogram_1, bin_centres)

    b_mask = ~np.isnan(PG_tt_0_RZ_peak_amps) & ~np.isnan(DG_tt_0_RZ_peak_amps)
    nb_mask = ~np.isnan(PG_tt_1_RZ_peak_amps) & ~np.isnan(DG_tt_1_RZ_peak_amps)

    # plot and compare similarly to the peak amplitudes hits
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color]
    data = [PG_tt_0_RZ_peak_amps[b_mask], DG_tt_0_RZ_peak_amps[b_mask],
            PG_tt_1_RZ_peak_amps[nb_mask], DG_tt_1_RZ_peak_amps[nb_mask]]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    """
    for i in range(len(data[0])):
        ax.scatter([1,2], [data[0][i],data[1][i]], marker="o", color="black")
        ax.plot([1,2], [data[0][i],data[1][i]], color="black")
    for i in range(len(data[2])):
        ax.scatter([1,2], [data[2][i],data[3][i]], marker="o", color="black")
        ax.plot([1,2], [data[2][i],data[3][i]], color="black")
    """
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "B", "NB", "NB"])
    #ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_ylim(bottom=0)
    #ax.set_ylim([0, 0.35])
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/peak_minus_baseline_for_remapped_grid_cells'+suffix+'.png', dpi=300)
    plt.close()

    # plot and compare similarly to the peak amplitudes hits
    b_mask2 = ~np.isnan(PG_tt_0_peak_locs) & ~np.isnan(DG_tt_0_peak_locs)
    nb_mask2 = ~np.isnan(PG_tt_1_peak_locs) & ~np.isnan(DG_tt_1_peak_locs)
    
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color]
    data = [PG_tt_0_peak_locs[b_mask2], DG_tt_0_peak_locs[b_mask2],
            PG_tt_1_peak_locs[nb_mask2], DG_tt_1_peak_locs[nb_mask2]]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    """
    for i in range(len(data[0])):
        ax.scatter([1,2], [data[0][i],data[1][i]], marker="o", color="black")
        ax.plot([1,2], [data[0][i],data[1][i]], color="black")
    for i in range(len(data[2])):
        ax.scatter([1,2], [data[2][i],data[3][i]], marker="o", color="black")
        ax.plot([1,2], [data[2][i],data[3][i]], color="black")"""

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "B", "NB", "NB"])
    #ax.set_yticks([-1, 0, 1])
    ax.axhspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axhspan(90, 110, xmin=0, xmax=0.5, facecolor='DarkGreen', alpha=.25, linewidth =0)
    ax.axhline(y=90, xmin=0.5, xmax=1, color="black", linestyle="dotted", linewidth=1)
    ax.axhline(y=110, xmin=0.5, xmax=1, color="black", linestyle="dotted", linewidth=1)
    ax.set_ylim([0, 130])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/peak_location_for_remapped_grid_cells'+suffix+'.png', dpi=300)
    plt.close()


    print("comparing peak_amps between postion and distance remapped encoding grid cells for beaconed trials, df=",str(len(PG_tt_0_RZ_peak_amps[b_mask])-1), ", p= ", str(scipy.stats.wilcoxon(PG_tt_0_RZ_peak_amps[b_mask],DG_tt_0_RZ_peak_amps[b_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(PG_tt_0_RZ_peak_amps[b_mask],DG_tt_0_RZ_peak_amps[b_mask])[0]))
    print("comparing peak_amps between postion and distance remapped encoding grid cells for non beaconed trials, df=",str(len(PG_tt_1_RZ_peak_amps[nb_mask])-1), ", p= ", str(scipy.stats.wilcoxon(PG_tt_1_RZ_peak_amps[nb_mask],DG_tt_1_RZ_peak_amps[nb_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(PG_tt_1_RZ_peak_amps[nb_mask],DG_tt_1_RZ_peak_amps[nb_mask])[0]))
    print("comparing peak_locs between postion and distance remapped encoding grid cells for beaconed trials, df=",str(len(PG_tt_0_peak_locs[b_mask2])-1), ", p= ", str(scipy.stats.wilcoxon(PG_tt_0_peak_locs[b_mask2],DG_tt_0_peak_locs[b_mask2])[1]), ", t= ", str(scipy.stats.wilcoxon(PG_tt_0_peak_locs[b_mask2],DG_tt_0_peak_locs[b_mask2])[0]))
    print("comparing peak_locs between postion and distance remapped encoding grid cells for non beaconed trials, df=",str(len(PG_tt_1_peak_locs[nb_mask2])-1), ", p= ", str(scipy.stats.wilcoxon(PG_tt_1_peak_locs[nb_mask2],DG_tt_1_peak_locs[nb_mask2])[1]), ", t= ", str(scipy.stats.wilcoxon(PG_tt_1_peak_locs[nb_mask2],DG_tt_1_peak_locs[nb_mask2])[0]))
    return

def plot_stop_peak_stop_location_and_height_remapped_probe(combined_df, save_path, suffix=""):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    # trial type 0
    remapped_position_grid_cells_stop_histogram_2, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=2, coding_scheme="P", shuffle=False)
    remapped_distance_grid_cells_stop_histogram_2, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=2, coding_scheme="D", shuffle=False)
    remapped_grid_cells_shuffled_histogram_2, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=2, coding_scheme=None, shuffle=True)
    remapped_position_grid_cells_stop_histogram_2 = np.array(remapped_position_grid_cells_stop_histogram_2)
    remapped_distance_grid_cells_stop_histogram_2 = np.array(remapped_distance_grid_cells_stop_histogram_2)
    remapped_grid_cells_shuffled_histogram_2 = np.array(remapped_grid_cells_shuffled_histogram_2)

    # apply normalisation with baseline
    remapped_position_grid_cells_stop_histogram_2 = remapped_position_grid_cells_stop_histogram_2-remapped_grid_cells_shuffled_histogram_2
    remapped_distance_grid_cells_stop_histogram_2 = remapped_distance_grid_cells_stop_histogram_2-remapped_grid_cells_shuffled_histogram_2

    # get amplitudes of peaks and locations of peaks
    PG_tt_2_peak_amps, PG_tt_2_peak_locs, PG_tt_2_RZ_peak_amps = get_peak_amp_and_locs(remapped_position_grid_cells_stop_histogram_2, bin_centres)
    DG_tt_2_peak_amps, DG_tt_2_peak_locs, DG_tt_2_RZ_peak_amps = get_peak_amp_and_locs(remapped_distance_grid_cells_stop_histogram_2, bin_centres)

    p_mask = ~np.isnan(PG_tt_2_RZ_peak_amps) & ~np.isnan(DG_tt_2_RZ_peak_amps)

    # plot and compare similarly to the peak amplitudes hits
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color]
    data = [PG_tt_2_RZ_peak_amps[p_mask], DG_tt_2_RZ_peak_amps[p_mask]]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    """
    for i in range(len(data[0])):
        ax.scatter([1,2], [data[0][i],data[1][i]], marker="o", color="black")
        ax.plot([1,2], [data[0][i],data[1][i]], color="black")
    for i in range(len(data[2])):
        ax.scatter([1,2], [data[2][i],data[3][i]], marker="o", color="black")
        ax.plot([1,2], [data[2][i],data[3][i]], color="black")
    """
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2])
    ax.set_xticklabels(["P","P"])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    #ax.set_ylim([0, 0.35])
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/peak_minus_baseline_for_remapped_grid_cells_probe'+suffix+'.png', dpi=300)
    plt.close()

    # plot and compare similarly to the peak amplitudes hits
    p_mask2 = ~np.isnan(PG_tt_2_peak_locs) & ~np.isnan(DG_tt_2_peak_locs)

    fig, ax = plt.subplots(1,1, figsize=(6,6))
    colors = [Settings.allocentric_color, Settings.egocentric_color]
    data = [PG_tt_2_peak_locs[p_mask], DG_tt_2_peak_locs[p_mask]]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    """
    for i in range(len(data[0])):
        ax.scatter([1,2], [data[0][i],data[1][i]], marker="o", color="black")
        ax.plot([1,2], [data[0][i],data[1][i]], color="black")
    for i in range(len(data[2])):
        ax.scatter([1,2], [data[2][i],data[3][i]], marker="o", color="black")
        ax.plot([1,2], [data[2][i],data[3][i]], color="black")"""

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2])
    ax.set_xticklabels(["P","P"])
    #ax.set_yticks([-1, 0, 1])
    ax.axhspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axhspan(170, 200, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axhline(y=90, xmin=0, xmax=1, color="black", linestyle="dotted", linewidth=1)
    ax.axhline(y=110, xmin=0, xmax=1, color="black", linestyle="dotted", linewidth=1)
    ax.set_ylim([0, 200])
    ax.set_xlim([0, 6])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(save_path + '/peak_location_for_remapped_grid_cells_probe'+suffix+'.png', dpi=300)
    plt.close()


    print("comparing peak_amps between postion and distance remapped encoding grid cells for probe trials, df=",str(len(PG_tt_2_RZ_peak_amps[p_mask])-1), ", p= ", str(scipy.stats.wilcoxon(PG_tt_2_RZ_peak_amps[p_mask],DG_tt_2_RZ_peak_amps[p_mask])[1]), ", t= ", str(scipy.stats.wilcoxon(PG_tt_2_RZ_peak_amps[p_mask],DG_tt_2_RZ_peak_amps[p_mask])[0]))
    print("comparing peak_locs between postion and distance remapped encoding grid cells for probe trials, df=",str(len(PG_tt_2_peak_locs[p_mask2])-1), ", p= ", str(scipy.stats.wilcoxon(PG_tt_2_peak_locs[p_mask2],DG_tt_2_peak_locs[p_mask2])[1]), ", t= ", str(scipy.stats.wilcoxon(PG_tt_2_peak_locs[p_mask2],DG_tt_2_peak_locs[p_mask2])[0]))

def plot_stop_histogram_for_remapped_encoding_grid_cells_by_mouse(combined_df, save_path, suffix="", lock_y_axis=True):
    print("do stuff")
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #grid_cells = grid_cells[(grid_cells["rolling:proportion_encoding_position"] < 0.85) &
    #                        (grid_cells["rolling:proportion_encoding_distance"] < 0.85)]

    for tt in [0, 1, 2]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.axhline(y=0, linestyle="dashed", linewidth=2, color="black")

        for code, code_color in zip(["P", "D"], [Settings.allocentric_color, Settings.egocentric_color]):
            hists = []
            for m, mouse_id in enumerate(np.unique(grid_cells["mouse"])):
                mouse_grids=grid_cells[grid_cells["mouse"] == mouse_id]

                remapped_position_grid_cells_stop_histogram_tt, _, bin_centres, number_of_trials_cells, _, _ = get_stop_histogram(mouse_grids, tt=tt, coding_scheme=code, shuffle=False)
                remapped_position_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(mouse_grids, tt=tt, coding_scheme=code,shuffle=True)

                remapped_position_grid_cells_stop_histogram_tt = np.array(remapped_position_grid_cells_stop_histogram_tt)
                remapped_position_grid_cells_shuffled_histogram_tt = np.array(remapped_position_grid_cells_shuffled_histogram_tt)

                # normalise to baseline
                remapped_position_grid_cells_stop_histogram_tt = remapped_position_grid_cells_stop_histogram_tt-remapped_position_grid_cells_shuffled_histogram_tt

                if np.nanmean(number_of_trials_cells)>10:
                    hists.append(np.nanmean(remapped_position_grid_cells_stop_histogram_tt, axis=0))

                    #ax.plot(bin_centres, np.nanmean(remapped_position_grid_cells_stop_histogram_tt, axis=0), color=cycle[m], linewidth=1.5, label=mouse_id)
                    #ax.fill_between(bin_centres, np.nanmean(remapped_position_grid_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(remapped_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                    #                np.nanmean(remapped_position_grid_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(remapped_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.allocentric_color, alpha=0.3)

            hists = np.array(hists)
            ax.plot(bin_centres, np.nanmean(hists, axis=0), color=code_color,linewidth=3)
            ax.fill_between(bin_centres, np.nanmean(hists, axis=0)-scipy.stats.sem(hists, axis=0, nan_policy="omit"),
                            np.nanmean(hists, axis=0)+scipy.stats.sem(hists, axis=0, nan_policy="omit"), color=code_color, alpha=0.3)

        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        #plt.ylabel('Stops (/cm)', fontsize=20, labelpad = 20)
        #plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
        plt.xlim(0, 200)
        tick_spacing = 100
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Edmond.plot_utility2.style_vr_plot(ax)
        if lock_y_axis:
            if tt ==2:
                ax.set_ylim([-0.05,0.05])
                ax.set_yticks([0, 0.05])
            else:
                ax.set_ylim([-0.05,0.15])
                ax.set_yticks([0, 0.1])
        #plt.locator_params(axis = 'y', nbins  = 3)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()
        #ax.legend(loc="best")
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        plt.savefig(save_path + '/stop_histogram_for_remapped_grid_cells_encoding_position_and_distance_'+str(tt)+suffix+'_by_mouse.png', dpi=300)
        plt.close()
    return

def plot_stop_histogram_for_trials(combined_df, save_path, suffix="", lock_y_axis=True):
    print("do stuff")
    #grid_cells = combined_df[combined_df["classifier"] == "G"]
    cells = combined_df
    #grid_cells = grid_cells[(grid_cells["rolling:proportion_encoding_position"] < 0.85) &
    #                        (grid_cells["rolling:proportion_encoding_distance"] < 0.85)]

    #compile_remapped_grid_cell_stop_histogram(grid_cells, tt=0)
    #compile_remapped_grid_cell_stop_histogram(grid_cells, tt=1)

    cells = drop_duplicate_sessions(cells)

    for tt in [0,1,2]:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.axhline(y=0, linestyle="dashed", linewidth=2, color="black")
        remapped_null_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt, coding_scheme="N",shuffle=False, rolling_classsifier_collumn_name="rolling:code_global")
        remapped_position_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt, coding_scheme="P", shuffle=False, rolling_classsifier_collumn_name="rolling:code_global")
        remapped_distance_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt, coding_scheme="D", shuffle=False, rolling_classsifier_collumn_name="rolling:code_global")
        remapped_null_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt, coding_scheme="N",shuffle=True, rolling_classsifier_collumn_name="rolling:code_global")
        remapped_position_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt, coding_scheme="P",shuffle=True, rolling_classsifier_collumn_name="rolling:code_global")
        remapped_distance_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt, coding_scheme="D",shuffle=True, rolling_classsifier_collumn_name="rolling:code_global")

        remapped_null_cells_stop_histogram_tt = np.array(remapped_null_cells_stop_histogram_tt)
        remapped_position_cells_stop_histogram_tt = np.array(remapped_position_cells_stop_histogram_tt)
        remapped_distance_cells_stop_histogram_tt = np.array(remapped_distance_cells_stop_histogram_tt)
        remapped_position_cells_shuffled_histogram_tt = np.array(remapped_position_cells_shuffled_histogram_tt)
        remapped_distance_cells_shuffled_histogram_tt = np.array(remapped_distance_cells_shuffled_histogram_tt)
        remapped_null_cells_shuffled_histogram_tt = np.array(remapped_null_cells_shuffled_histogram_tt)

        # remove stop profiles where there isn't a stop profile for both postion and distance encoding trials
        #nan_mask = ~np.isnan(remapped_position_grid_cells_stop_histogram_tt) & ~np.isnan(remapped_distance_grid_cells_stop_histogram_tt)
        #nan_mask = nan_mask[:,0]
        #remapped_position_grid_cells_stop_histogram_tt = remapped_position_grid_cells_stop_histogram_tt[nan_mask,:]
        #remapped_distance_grid_cells_stop_histogram_tt = remapped_distance_grid_cells_stop_histogram_tt[nan_mask,:]
        #remapped_grid_cells_shuffled_histogram_tt = remapped_grid_cells_shuffled_histogram_tt[nan_mask,:]

        p_nan_mask = ~np.isnan(remapped_position_cells_stop_histogram_tt)[:,0]
        d_nan_mask = ~np.isnan(remapped_distance_cells_stop_histogram_tt)[:,0]
        n_nan_mask = ~np.isnan(remapped_null_cells_stop_histogram_tt)[:, 0]
        #remapped_position_grid_cells_stop_histogram_tt = remapped_position_grid_cells_stop_histogram_tt[p_nan_mask,:]
        #remapped_distance_grid_cells_stop_histogram_tt = remapped_distance_grid_cells_stop_histogram_tt[d_nan_mask,:]

        # normalise to baseline
        remapped_position_cells_stop_histogram_tt = remapped_position_cells_stop_histogram_tt-remapped_position_cells_shuffled_histogram_tt
        remapped_distance_cells_stop_histogram_tt = remapped_distance_cells_stop_histogram_tt-remapped_distance_cells_shuffled_histogram_tt
        remapped_null_cells_stop_histogram_tt = remapped_null_cells_stop_histogram_tt - remapped_null_cells_shuffled_histogram_tt

        # apply mask
        remapped_position_cells_stop_histogram_tt = remapped_position_cells_stop_histogram_tt[p_nan_mask,:]
        remapped_distance_cells_stop_histogram_tt = remapped_distance_cells_stop_histogram_tt[d_nan_mask,:]
        remapped_null_cells_stop_histogram_tt = remapped_null_cells_stop_histogram_tt[n_nan_mask, :]

        # plot position grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(remapped_position_cells_stop_histogram_tt, axis=0), color= Settings.allocentric_color, linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(remapped_position_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(remapped_position_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(remapped_position_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(remapped_position_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.allocentric_color, alpha=0.3)

        # plot distance grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(remapped_distance_cells_stop_histogram_tt, axis=0), color= Settings.egocentric_color, linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(remapped_distance_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(remapped_distance_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(remapped_distance_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(remapped_distance_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.egocentric_color, alpha=0.3)

        # plot null grid cell session stop histogram
        #ax.plot(bin_centres, np.nanmean(remapped_null_cells_stop_histogram_tt, axis=0), color= Settings.null_color, linewidth=3)
        #ax.fill_between(bin_centres, np.nanmean(remapped_null_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(remapped_null_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
        #                np.nanmean(remapped_null_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(remapped_null_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.null_color, alpha=0.3)

        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        #plt.ylabel('Stops (/cm)', fontsize=20, labelpad = 20)
        #plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
        plt.xlim(0, 200)
        tick_spacing = 100
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Edmond.plot_utility2.style_vr_plot(ax)
        if lock_y_axis:
            ax.set_ylim([-0.05,0.15])
            ax.set_yticks([0, 0.1])

        plt.locator_params(axis = 'y', nbins  = 3)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        plt.savefig(save_path + '/stop_histogram_for_trials_encoding_position_and_distance_'+str(tt)+suffix+'.png', dpi=300)
        plt.close()
    return

def plot_stop_histogram_for_remapped_encoding_grid_cells(combined_df, save_path, suffix="", lock_y_axis=True):
    print("do stuff")
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    #grid_cells = grid_cells[(grid_cells["rolling:proportion_encoding_position"] < 0.85) &
    #                        (grid_cells["rolling:proportion_encoding_distance"] < 0.85)]

    #compile_remapped_grid_cell_stop_histogram(grid_cells, tt=0)
    #compile_remapped_grid_cell_stop_histogram(grid_cells, tt=1)

    for tt in [0,1,2]:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.axhline(y=0, linestyle="dashed", linewidth=2, color="black")
        remapped_position_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=tt, coding_scheme="P", shuffle=False)
        remapped_distance_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=tt, coding_scheme="D", shuffle=False)
        remapped_position_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=0,coding_scheme="P",shuffle=True)
        remapped_distance_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(grid_cells, tt=0,coding_scheme="D",shuffle=True)

        remapped_position_grid_cells_stop_histogram_tt = np.array(remapped_position_grid_cells_stop_histogram_tt)
        remapped_distance_grid_cells_stop_histogram_tt = np.array(remapped_distance_grid_cells_stop_histogram_tt)
        remapped_position_grid_cells_shuffled_histogram_tt = np.array(remapped_position_grid_cells_shuffled_histogram_tt)
        remapped_distance_grid_cells_shuffled_histogram_tt = np.array(remapped_distance_grid_cells_shuffled_histogram_tt)

        # remove stop profiles where there isn't a stop profile for both postion and distance encoding trials
        #nan_mask = ~np.isnan(remapped_position_grid_cells_stop_histogram_tt) & ~np.isnan(remapped_distance_grid_cells_stop_histogram_tt)
        #nan_mask = nan_mask[:,0]
        #remapped_position_grid_cells_stop_histogram_tt = remapped_position_grid_cells_stop_histogram_tt[nan_mask,:]
        #remapped_distance_grid_cells_stop_histogram_tt = remapped_distance_grid_cells_stop_histogram_tt[nan_mask,:]
        #remapped_grid_cells_shuffled_histogram_tt = remapped_grid_cells_shuffled_histogram_tt[nan_mask,:]

        p_nan_mask = ~np.isnan(remapped_position_grid_cells_stop_histogram_tt)[:,0]
        d_nan_mask = ~np.isnan(remapped_distance_grid_cells_stop_histogram_tt)[:,0]
        #remapped_position_grid_cells_stop_histogram_tt = remapped_position_grid_cells_stop_histogram_tt[p_nan_mask,:]
        #remapped_distance_grid_cells_stop_histogram_tt = remapped_distance_grid_cells_stop_histogram_tt[d_nan_mask,:]

        # normalise to baseline
        remapped_position_grid_cells_stop_histogram_tt = remapped_position_grid_cells_stop_histogram_tt-remapped_position_grid_cells_shuffled_histogram_tt
        remapped_distance_grid_cells_stop_histogram_tt = remapped_distance_grid_cells_stop_histogram_tt-remapped_distance_grid_cells_shuffled_histogram_tt

        # apply mask
        remapped_position_grid_cells_stop_histogram_tt = remapped_position_grid_cells_stop_histogram_tt[p_nan_mask,:]
        remapped_distance_grid_cells_stop_histogram_tt = remapped_distance_grid_cells_stop_histogram_tt[d_nan_mask,:]

        # plot position grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(remapped_position_grid_cells_stop_histogram_tt, axis=0), color= Settings.allocentric_color, linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(remapped_position_grid_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(remapped_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(remapped_position_grid_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(remapped_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.allocentric_color, alpha=0.3)

        # plot distance grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(remapped_distance_grid_cells_stop_histogram_tt, axis=0), color= Settings.egocentric_color, linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(remapped_distance_grid_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(remapped_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(remapped_distance_grid_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(remapped_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.egocentric_color, alpha=0.3)

        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        #plt.ylabel('Stops (/cm)', fontsize=20, labelpad = 20)
        #plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
        plt.xlim(0, 200)
        tick_spacing = 100
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Edmond.plot_utility2.style_vr_plot(ax)
        if lock_y_axis:
            ax.set_ylim([-0.05,0.15])
            ax.set_yticks([0, 0.1])

        plt.locator_params(axis = 'y', nbins  = 3)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        plt.savefig(save_path + '/stop_histogram_for_remapped_grid_cells_encoding_position_and_distance_'+str(tt)+suffix+'.png', dpi=300)
        plt.close()
    return

def plot_stop_histogram_for_stable_sessions(combined_df, save_path):
    print("do stuff")
    cells = combined_df

    #position_cells = cells[cells["Lomb_classifier_"] == "Position"]
    #distance_cells = cells[cells["Lomb_classifier_"] == "Distance"]
    #stable_position_cells = position_cells[position_cells["rolling:code_global_encoding_position"] > 0.85]
    #stable_distance_cells = distance_cells[distance_cells["rolling:code_global_encoding_distance"] > 0.85]
    #stable_position_cells = drop_duplicate_sessions(stable_position_cells)
    #stable_distance_cells = drop_duplicate_sessions(stable_distance_cells)

    stable_position_cells = combined_df[combined_df["proportion_network_encoding_position"] > 0.85]
    stable_distance_cells = combined_df[combined_df["proportion_network_encoding_distance"] > 0.85]
    print("n session for stable position sessions, n = ", str(len(stable_position_cells)))
    print("n session for stable distance sessions, n = ", str(len(stable_distance_cells)))

    for tt in [0,1,2]:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.axhline(y=0, linestyle="dashed", linewidth=2, color="black")
        stable_position_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_cells, tt=tt, coding_scheme=None, shuffle=False)
        stable_distance_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_distance_cells, tt=tt, coding_scheme=None, shuffle=False)
        stable_position_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_cells, tt=tt, coding_scheme=None, shuffle=True)
        stable_distance_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_distance_cells, tt=tt, coding_scheme=None, shuffle=True)
        stable_position_grid_cells_stop_histogram_tt = np.array(stable_position_grid_cells_stop_histogram_tt)
        stable_distance_grid_cells_stop_histogram_tt = np.array(stable_distance_grid_cells_stop_histogram_tt)
        stable_position_grid_cells_shuffled_histogram_tt = np.array(stable_position_grid_cells_shuffled_histogram_tt)
        stable_distance_grid_cells_shuffled_histogram_tt = np.array(stable_distance_grid_cells_shuffled_histogram_tt)

        # apply normalisation with baseline
        stable_position_grid_cells_stop_histogram_tt = stable_position_grid_cells_stop_histogram_tt-stable_position_grid_cells_shuffled_histogram_tt
        stable_distance_grid_cells_stop_histogram_tt = stable_distance_grid_cells_stop_histogram_tt-stable_distance_grid_cells_shuffled_histogram_tt

        # plot position grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(stable_position_grid_cells_stop_histogram_tt, axis=0), color= Settings.allocentric_color,linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(stable_position_grid_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(stable_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(stable_position_grid_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(stable_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.allocentric_color, alpha=0.3)

        # plot distance grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(stable_distance_grid_cells_stop_histogram_tt, axis=0), color= Settings.egocentric_color,linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(stable_distance_grid_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(stable_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(stable_distance_grid_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(stable_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.egocentric_color, alpha=0.3)

        # plot_the_baseline_shuffle stop histogram
        #ax.plot(bin_centres, np.nanmean(stable_grid_cells_shuffled_histogram_tt, axis=0), color="black", linestyle="dashed")
        #ax.fill_between(bin_centres, np.nanmean(stable_grid_cells_shuffled_histogram_tt, axis=0)-scipy.stats.sem(stable_grid_cells_shuffled_histogram_tt, axis=0, nan_policy="omit"),
        #                np.nanmean(stable_grid_cells_shuffled_histogram_tt, axis=0)+scipy.stats.sem(stable_grid_cells_shuffled_histogram_tt, axis=0, nan_policy="omit"), color="black", alpha=0.3)

        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        #plt.ylabel('Stops (/cm)', fontsize=20, labelpad = 20)
        #plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
        plt.xlim(0, 200)
        tick_spacing = 100
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Edmond.plot_utility2.style_vr_plot(ax)
        ax.set_ylim([-0.05,0.15])
        ax.set_yticks([0, 0.1])

        #ax.set_ylim([-0.05,0.05])
        #ax.set_yticks([0, 0.05])

        plt.locator_params(axis = 'y', nbins  = 3)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        plt.savefig(save_path + '/stop_histogram_for_stable_session_encoding_position_and_distance_'+str(tt)+'.png', dpi=300)
        plt.close()
    return

def plot_percentage_hits_for_stable_sessions(combined_df, save_path):
    print("do stuff")
    cells = combined_df

    position_cells = cells[cells["Lomb_classifier_"] == "Position"]
    distance_cells = cells[cells["Lomb_classifier_"] == "Distance"]
    stable_position_cells = position_cells[position_cells["rolling:code_global_encoding_position"] > 0.85]
    stable_distance_cells = distance_cells[distance_cells["rolling:code_global_encoding_distance"] > 0.85]
    stable_position_cells = drop_duplicate_sessions(stable_position_cells)
    stable_distance_cells = drop_duplicate_sessions(stable_distance_cells)

    stable_position_cells = combined_df[combined_df["proportion_network_encoding_position"] > 0.85]
    stable_distance_cells = combined_df[combined_df["proportion_network_encoding_distance"] > 0.85]

    fig, ax = plt.subplots(1,1, figsize=(8,6))

    beaconed_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_cells, tt=0)
    non_beaconed_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_cells, tt=1)
    beaconed_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_cells, tt=0)
    non_beaconed_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_cells, tt=1)
    probe_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_cells, tt=2)
    probe_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_cells, tt=2)

    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color, Settings.allocentric_color, Settings.egocentric_color]

    data = [beaconed_percentage_hits_stable_position_grid_cells, beaconed_percentage_hits_stable_distance_grid_cells,
            non_beaconed_percentage_hits_stable_position_grid_cells, non_beaconed_percentage_hits_stable_distance_grid_cells,
            probe_percentage_hits_stable_position_grid_cells, probe_percentage_hits_stable_distance_grid_cells]

    print("comping % hits between postion and distance encoding grid cells for beaconed trials, df=",str(len(beaconed_percentage_hits_stable_position_grid_cells)+len(beaconed_percentage_hits_stable_distance_grid_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(beaconed_percentage_hits_stable_position_grid_cells,beaconed_percentage_hits_stable_distance_grid_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(beaconed_percentage_hits_stable_position_grid_cells,beaconed_percentage_hits_stable_distance_grid_cells)[0]))
    print("comping % hits between postion and distance encoding grid cells for non beaconed trials, df=",str(len(non_beaconed_percentage_hits_stable_position_grid_cells)+len(non_beaconed_percentage_hits_stable_distance_grid_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(non_beaconed_percentage_hits_stable_position_grid_cells,non_beaconed_percentage_hits_stable_distance_grid_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(non_beaconed_percentage_hits_stable_position_grid_cells,non_beaconed_percentage_hits_stable_distance_grid_cells)[0]))
    print("comping % hits between postion and distance encoding grid cells for probe trials, df=",str(len(probe_percentage_hits_stable_position_grid_cells)+len(probe_percentage_hits_stable_distance_grid_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(probe_percentage_hits_stable_position_grid_cells,probe_percentage_hits_stable_distance_grid_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(probe_percentage_hits_stable_position_grid_cells,probe_percentage_hits_stable_distance_grid_cells)[0]))

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5, 7,8], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5, 7,8]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5,7,8])
    ax.set_xticklabels(["B", "B", "NB", "NB", "P", "P"])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 9])
    #ax.set_xlabel("Encoding grid cells", fontsize=20)
    #ax.set_ylabel("Percentage hits", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/percentage_hits_for_stable_sessions.png', dpi=300)
    plt.close()
    return

def plot_stop_histogram_for_stable_encoding_grid_cells(combined_df, save_path):
    print("do stuff")
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    cells = combined_df
    cells = combined_df[combined_df["classifier"] == "G"]

    position_cells = cells[cells["Lomb_classifier_"] == "Position"]
    distance_cells = cells[cells["Lomb_classifier_"] == "Distance"]
    stable_position_cells = position_cells[position_cells["rolling:proportion_encoding_position"] > 0.85]
    stable_distance_cells = distance_cells[distance_cells["rolling:proportion_encoding_distance"] > 0.85]
    stable_position_cells = drop_duplicate_sessions(stable_position_cells)
    stable_distance_cells = drop_duplicate_sessions(stable_distance_cells)

    for tt in [0,1,2]:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.axhline(y=0, linestyle="dashed", linewidth=2, color="black")
        stable_position_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_cells, tt=tt, coding_scheme=None, shuffle=False)
        stable_distance_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_distance_cells, tt=tt, coding_scheme=None, shuffle=False)
        stable_position_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_cells, tt=tt, coding_scheme=None, shuffle=True)
        stable_distance_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_distance_cells, tt=tt, coding_scheme=None, shuffle=True)
        stable_position_grid_cells_stop_histogram_tt = np.array(stable_position_grid_cells_stop_histogram_tt)
        stable_distance_grid_cells_stop_histogram_tt = np.array(stable_distance_grid_cells_stop_histogram_tt)
        stable_position_grid_cells_shuffled_histogram_tt = np.array(stable_position_grid_cells_shuffled_histogram_tt)
        stable_distance_grid_cells_shuffled_histogram_tt = np.array(stable_distance_grid_cells_shuffled_histogram_tt)

        # apply normalisation with baseline
        stable_position_grid_cells_stop_histogram_tt = stable_position_grid_cells_stop_histogram_tt-stable_position_grid_cells_shuffled_histogram_tt
        stable_distance_grid_cells_stop_histogram_tt = stable_distance_grid_cells_stop_histogram_tt-stable_distance_grid_cells_shuffled_histogram_tt

        # plot position grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(stable_position_grid_cells_stop_histogram_tt, axis=0), color= Settings.allocentric_color,linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(stable_position_grid_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(stable_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(stable_position_grid_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(stable_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.allocentric_color, alpha=0.3)

        # plot distance grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(stable_distance_grid_cells_stop_histogram_tt, axis=0), color= Settings.egocentric_color,linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(stable_distance_grid_cells_stop_histogram_tt, axis=0)-scipy.stats.sem(stable_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(stable_distance_grid_cells_stop_histogram_tt, axis=0)+scipy.stats.sem(stable_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.egocentric_color, alpha=0.3)

        # plot_the_baseline_shuffle stop histogram
        #ax.plot(bin_centres, np.nanmean(stable_grid_cells_shuffled_histogram_tt, axis=0), color="black", linestyle="dashed")
        #ax.fill_between(bin_centres, np.nanmean(stable_grid_cells_shuffled_histogram_tt, axis=0)-scipy.stats.sem(stable_grid_cells_shuffled_histogram_tt, axis=0, nan_policy="omit"),
        #                np.nanmean(stable_grid_cells_shuffled_histogram_tt, axis=0)+scipy.stats.sem(stable_grid_cells_shuffled_histogram_tt, axis=0, nan_policy="omit"), color="black", alpha=0.3)

        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        #plt.ylabel('Stops (/cm)', fontsize=20, labelpad = 20)
        #plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
        plt.xlim(0, 200)
        tick_spacing = 100
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        Edmond.plot_utility2.style_vr_plot(ax)
        ax.set_ylim([-0.05,0.15])
        ax.set_yticks([0, 0.1])

        #ax.set_ylim([-0.05,0.05])
        #ax.set_yticks([0, 0.05])

        plt.locator_params(axis = 'y', nbins  = 3)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        plt.savefig(save_path + '/stop_histogram_for_stable_grid_cells_encoding_position_and_distance_'+str(tt)+'.png', dpi=300)
        plt.close()
    return


def drop_duplicate_sessions(cells_df):
    sessions = []
    new_df = pd.DataFrame()
    for index, cluster_df in cells_df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        session_id = cluster_df["session_id"].iloc[0]
        if session_id not in sessions:
            new_df = pd.concat([new_df, cluster_df], ignore_index=True)
            sessions.append(session_id)
    return new_df

def plot_p_hit_for_stable_encoding_grid_cells(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    Position_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    Distance_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]

    stable_position_grid_cells = Position_grid_cells[Position_grid_cells["rolling:proportion_encoding_position"] > 0.85]
    stable_distance_grid_cells = Distance_grid_cells[Distance_grid_cells["rolling:proportion_encoding_distance"] > 0.85]

    #stable_position_grid_cells = Position_grid_cells
    #stable_distance_grid_cells = Distance_grid_cells

    stable_position_grid_cells = drop_duplicate_sessions(stable_position_grid_cells)
    stable_distance_grid_cells = drop_duplicate_sessions(stable_distance_grid_cells)

    print("n session for stable position grid cells, n = ", str(len(stable_position_grid_cells)))
    print("n session for stable distance grid cells, n = ", str(len(stable_distance_grid_cells)))
    #stable_position_grid_cells = Position_grid_cells
    #stable_distance_grid_cells = Distance_grid_cells

    for tt in [0,1,2]:
        fig, ax = plt.subplots(1,1, figsize=(3.5,6))

        tt_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_grid_cells, tt=tt)/100
        tt_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_grid_cells, tt=tt)/100

        colors = [Settings.allocentric_color, Settings.egocentric_color]

        data = [tt_percentage_hits_stable_position_grid_cells, tt_percentage_hits_stable_distance_grid_cells]

        print("comping % hits between postion and distance encoding grid cells for tt "+str(tt)+" trials, df=",str(len(tt_percentage_hits_stable_position_grid_cells)+len(tt_percentage_hits_stable_distance_grid_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(tt_percentage_hits_stable_position_grid_cells,tt_percentage_hits_stable_distance_grid_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(tt_percentage_hits_stable_position_grid_cells,tt_percentage_hits_stable_distance_grid_cells)[0]))

        boxprops = dict(linewidth=3, color='k')
        medianprops = dict(linewidth=3, color='k')
        capprops = dict(linewidth=3, color='k')
        whiskerprops = dict(linewidth=3, color='k')
        box = ax.boxplot(data, positions=[1,2], widths=1, boxprops=boxprops, medianprops=medianprops,
                         whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
        for i, x in enumerate([1,2]):
            ax.scatter((np.ones(len(data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xticks([1,2])
        #ax.set_xticklabels(["B", "B"])
        #ax.set_yticks([-1, 0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlim(left=0.4, right=2.6)
        #ax.set_xlabel("Encoding grid cells", fontsize=20)
        #ax.set_ylabel("Percentage hits", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='both', labelsize=25)
        #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.subplots_adjust(left = 0.2)
        plt.savefig(save_path + '/percentage_hits_for_stable_grid_cells_tt'+str(tt)+'.png', dpi=300)
        plt.close()
    return

def plot_percentage_hits_for_stable_encoding_grid_cells(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    cells = combined_df
    cells = combined_df[combined_df["classifier"] == "G"]

    position_cells = cells[cells["Lomb_classifier_"] == "Position"]
    distance_cells = cells[cells["Lomb_classifier_"] == "Distance"]

    stable_position_cells = position_cells[position_cells["rolling:proportion_encoding_position"] > 0.85]
    stable_distance_cells = distance_cells[distance_cells["rolling:proportion_encoding_distance"] > 0.85]

    #stable_position_grid_cells = Position_grid_cells
    #stable_distance_grid_cells = Distance_grid_cells

    stable_position_cells = drop_duplicate_sessions(stable_position_cells)
    stable_distance_cells = drop_duplicate_sessions(stable_distance_cells)

    print("n session for stable position grid cells, n = ", str(len(stable_position_cells)))
    print("n session for stable distance grid cells, n = ", str(len(stable_distance_cells)))
    #stable_position_grid_cells = Position_grid_cells
    #stable_distance_grid_cells = Distance_grid_cells

    fig, ax = plt.subplots(1,1, figsize=(8,6))

    beaconed_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_cells, tt=0)
    non_beaconed_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_cells, tt=1)
    beaconed_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_cells, tt=0)
    non_beaconed_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_cells, tt=1)
    probe_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_cells, tt=2)
    probe_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_cells, tt=2)

    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color, Settings.allocentric_color, Settings.egocentric_color]

    data = [beaconed_percentage_hits_stable_position_grid_cells, beaconed_percentage_hits_stable_distance_grid_cells,
            non_beaconed_percentage_hits_stable_position_grid_cells, non_beaconed_percentage_hits_stable_distance_grid_cells,
            probe_percentage_hits_stable_position_grid_cells, probe_percentage_hits_stable_distance_grid_cells]

    print("comping % hits between postion and distance encoding grid cells for beaconed trials, df=",str(len(beaconed_percentage_hits_stable_position_grid_cells)+len(beaconed_percentage_hits_stable_distance_grid_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(beaconed_percentage_hits_stable_position_grid_cells,beaconed_percentage_hits_stable_distance_grid_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(beaconed_percentage_hits_stable_position_grid_cells,beaconed_percentage_hits_stable_distance_grid_cells)[0]))
    print("comping % hits between postion and distance encoding grid cells for non beaconed trials, df=",str(len(non_beaconed_percentage_hits_stable_position_grid_cells)+len(non_beaconed_percentage_hits_stable_distance_grid_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(non_beaconed_percentage_hits_stable_position_grid_cells,non_beaconed_percentage_hits_stable_distance_grid_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(non_beaconed_percentage_hits_stable_position_grid_cells,non_beaconed_percentage_hits_stable_distance_grid_cells)[0]))
    print("comping % hits between postion and distance encoding grid cells for probe trials, df=",str(len(probe_percentage_hits_stable_position_grid_cells)+len(probe_percentage_hits_stable_distance_grid_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(probe_percentage_hits_stable_position_grid_cells,probe_percentage_hits_stable_distance_grid_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(probe_percentage_hits_stable_position_grid_cells,probe_percentage_hits_stable_distance_grid_cells)[0]))



    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5, 7,8], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5, 7,8]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5,7,8])
    ax.set_xticklabels(["B", "B", "NB", "NB", "P", "P"])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 9])
    #ax.set_xlabel("Encoding grid cells", fontsize=20)
    #ax.set_ylabel("Percentage hits", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/percentage_hits_for_stable_grid_cells.png', dpi=300)
    plt.close()
    return

def plot_percentage_hits_for_stable_encoding_non_grid_cells(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    Position_cells = non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Position"]
    Distance_cells = non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Distance"]

    stable_position_cells = Position_cells[Position_cells["rolling:proportion_encoding_position"] > 0.85]
    stable_distance_cells = Distance_cells[Distance_cells["rolling:proportion_encoding_distance"] > 0.85]

    stable_position_cells = drop_duplicate_sessions(stable_position_cells)
    stable_distance_cells = drop_duplicate_sessions(stable_distance_cells)

    print("n session for stable position cells, n = ", str(len(stable_position_cells)))
    print("n session for stable distance cells, n = ", str(len(stable_distance_cells)))


    fig, ax = plt.subplots(1,1, figsize=(8,6))

    beaconed_percentage_hits_stable_position_cells = get_percentage_hit_column(stable_position_cells, tt=0)
    non_beaconed_percentage_hits_stable_position_cells = get_percentage_hit_column(stable_position_cells, tt=1)
    beaconed_percentage_hits_stable_distance_cells = get_percentage_hit_column(stable_distance_cells, tt=0)
    non_beaconed_percentage_hits_stable_distance_cells = get_percentage_hit_column(stable_distance_cells, tt=1)
    probe_percentage_hits_stable_position_cells = get_percentage_hit_column(stable_position_cells, tt=2)
    probe_percentage_hits_stable_distance_cells = get_percentage_hit_column(stable_distance_cells, tt=2)

    colors = [Settings.allocentric_color, Settings.egocentric_color,  Settings.allocentric_color, Settings.egocentric_color, Settings.allocentric_color, Settings.egocentric_color]

    data = [beaconed_percentage_hits_stable_position_cells, beaconed_percentage_hits_stable_distance_cells,
            non_beaconed_percentage_hits_stable_position_cells, non_beaconed_percentage_hits_stable_distance_cells,
            probe_percentage_hits_stable_position_cells, probe_percentage_hits_stable_distance_cells]

    print("comping % hits between postion and distance encoding cells for beaconed trials, df=",str(len(beaconed_percentage_hits_stable_position_cells)+len(beaconed_percentage_hits_stable_distance_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(beaconed_percentage_hits_stable_position_cells,beaconed_percentage_hits_stable_distance_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(beaconed_percentage_hits_stable_position_cells,beaconed_percentage_hits_stable_distance_cells)[0]))
    print("comping % hits between postion and distance encoding cells for non beaconed trials, df=",str(len(non_beaconed_percentage_hits_stable_position_cells)+len(non_beaconed_percentage_hits_stable_distance_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(non_beaconed_percentage_hits_stable_position_cells,non_beaconed_percentage_hits_stable_distance_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(non_beaconed_percentage_hits_stable_position_cells,non_beaconed_percentage_hits_stable_distance_cells)[0]))
    print("comping % hits between postion and distance encoding cells for probe trials, df=",str(len(probe_percentage_hits_stable_position_cells)+len(probe_percentage_hits_stable_distance_cells)-2), ", p= ", str(scipy.stats.mannwhitneyu(probe_percentage_hits_stable_position_cells,probe_percentage_hits_stable_distance_cells)[1]), ", t= ", str(scipy.stats.mannwhitneyu(probe_percentage_hits_stable_position_cells,probe_percentage_hits_stable_distance_cells)[0]))

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5, 7,8], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2,4,5, 7,8]):
        ax.scatter((np.ones(len( data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2,4,5,7,8])
    ax.set_xticklabels(["B", "B", "NB", "NB", "P", "P"])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 9])
    #ax.set_xlabel("Encoding grid cells", fontsize=20)
    #ax.set_ylabel("Percentage hits", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/percentage_hits_for_stable_non_grid_cells.png', dpi=300)
    plt.close()


def plot_rolling_lomb_block_sizes(combined_df, save_path):
    print("do stuff")
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    print("the average rolling threshold for grid cells is "+str(np.nanmean(grid_cells["rolling_threshold"])))

    Position_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    Distance_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]
    Null_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Null"]

    fig, ax = plt.subplots(1,1, figsize=(6,6))
    #ax.set_xticks([0,1])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([-1, 1])
    ax.set_xlim([0, 1])
    ax.hist(pandas_collumn_to_numpy_array(Position_grid_cells["rolling:proportion_encoding_encoder"]), density=True, bins=20, range=(0,1), alpha=0.5, color=Settings.allocentric_color)
    ax.hist(pandas_collumn_to_numpy_array(Distance_grid_cells["rolling:proportion_encoding_encoder"]), density=True, bins=20, range=(0,1), alpha=0.5, color=Settings.egocentric_color)
    ax.hist(pandas_collumn_to_numpy_array(Null_grid_cells["rolling:proportion_encoding_encoder"]), density=True, bins=20, range=(0,1), alpha=0.5, color=Settings.null_color)
    ax.set_xlabel("Frac. session", fontsize=20)
    ax.set_ylabel("Density", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/fraction_encoding.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(6,6))
    #ax.set_xticks([0,1])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([-1, 1])
    ax.set_xlim([0, 1])
    ax.hist(pandas_collumn_to_numpy_array(Position_grid_cells["rolling:block_lengths_for_encoder"]), density=True, bins=10, range=(0,1), alpha=0.5, color=Settings.allocentric_color)
    ax.hist(pandas_collumn_to_numpy_array(Distance_grid_cells["rolling:block_lengths_for_encoder"]), density=True, bins=10, range=(0,1), alpha=0.5, color=Settings.egocentric_color)
    ax.hist(pandas_collumn_to_numpy_array(Null_grid_cells["rolling:block_lengths_for_encoder"]), density=True, bins=10, range=(0,1), alpha=0.5, color=Settings.null_color)
    ax.set_xlabel("Block length (frac. session)", fontsize=20)
    ax.set_ylabel("Density", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/block_length_encoding.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_xticks([0,1])
    #ax.set_yticks([-1, 0, 1])
    ax.set_ylim([0, 20])
    ax.set_xlim([0, 1])
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    ax.hist([pandas_collumn_to_numpy_array(Position_grid_cells["rolling:proportion_encoding_position"]),
             pandas_collumn_to_numpy_array(Distance_grid_cells["rolling:proportion_encoding_position"]),
             pandas_collumn_to_numpy_array(Null_grid_cells["rolling:proportion_encoding_position"])], bins=20, range=(0,1), color=colors, stacked=True)
    #ax.set_xlabel("frac. session", fontsize=20)
    #ax.set_ylabel("Number of cells", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    plt.savefig(save_path + '/block_length_encoding_position.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_xticks([0,1])
    #ax.set_yticks([-1, 0, 1])
    ax.set_ylim([0, 20])
    ax.set_xlim([0, 1])
    ax.hist([pandas_collumn_to_numpy_array(Position_grid_cells["rolling:proportion_encoding_distance"]),
             pandas_collumn_to_numpy_array(Distance_grid_cells["rolling:proportion_encoding_distance"]),
             pandas_collumn_to_numpy_array(Null_grid_cells["rolling:proportion_encoding_distance"])], bins=20, range=(0,1), color=colors, stacked=True)
    #ax.set_xlabel("frac. session", fontsize=20)
    #ax.set_ylabel("Number of cells", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    plt.savefig(save_path + '/block_length_encoding_distance.png', dpi=300)
    plt.close() 

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_xticks([0,1])
    #ax.set_yticks([-1, 0, 1])
    ax.set_ylim([0, 20])
    ax.set_xlim([0, 1])
    ax.hist([pandas_collumn_to_numpy_array(Position_grid_cells["rolling:proportion_encoding_null"]),
             pandas_collumn_to_numpy_array(Distance_grid_cells["rolling:proportion_encoding_null"]),
             pandas_collumn_to_numpy_array(Null_grid_cells["rolling:proportion_encoding_null"])], bins=20, range=(0,1), color=colors, stacked=True)
    #ax.set_xlabel("frac. session", fontsize=20)
    #ax.set_ylabel("Number of cells", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    plt.savefig(save_path + '/block_length_encoding_null.png', dpi=300)
    plt.close()

    print("unstable cells: ", len(Position_grid_cells[Position_grid_cells["rolling:proportion_encoding_position"]<0.85])+
          len(Distance_grid_cells[Distance_grid_cells["rolling:proportion_encoding_distance"]<0.85])+
          len(Null_grid_cells[Null_grid_cells["rolling:proportion_encoding_null"]<0.85]), ", ",
          (len(Position_grid_cells[Position_grid_cells["rolling:proportion_encoding_position"]<0.85])/len(grid_cells))+
          (len(Distance_grid_cells[Distance_grid_cells["rolling:proportion_encoding_distance"]<0.85])/len(grid_cells))+
          (len(Null_grid_cells[Null_grid_cells["rolling:proportion_encoding_null"]<0.85])/len(grid_cells)), "%")

    print("stable cells: ", len(Position_grid_cells[Position_grid_cells["rolling:proportion_encoding_position"]>=0.85])+
          len(Distance_grid_cells[Distance_grid_cells["rolling:proportion_encoding_distance"]>=0.85])+
          len(Null_grid_cells[Null_grid_cells["rolling:proportion_encoding_null"]>=0.85]), ", ",
          (len(Position_grid_cells[Position_grid_cells["rolling:proportion_encoding_position"]>=0.85])/len(grid_cells))+
          (len(Distance_grid_cells[Distance_grid_cells["rolling:proportion_encoding_distance"]>=0.85])/len(grid_cells))+
          (len(Null_grid_cells[Null_grid_cells["rolling:proportion_encoding_null"]>=0.85])/len(grid_cells)), "%")

    return

def add_max_block_lengths(df, suffix=""):
    max_block_length = []
    max_block_length_shuffled = []
    for index, cluster_df in df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        block_lengths = pandas_collumn_to_numpy_array(cluster_df["rolling:block_lengths"+suffix])
        block_lengths_shuffled = pandas_collumn_to_numpy_array(cluster_df["rolling:block_lengths_shuffled"+suffix])
        if len(block_lengths)>0:
            max_block_length.append(np.nanmax(block_lengths))
        else:
            max_block_length.append(0)
        if len(block_lengths_shuffled)>0:
            max_block_length_shuffled.append(np.nanmax(block_lengths_shuffled))
        else:
            max_block_length_shuffled.append(0)

    df["rolling:max_block_lengths"+suffix] = max_block_length
    df["rolling:max_block_lengths_shuffled"+suffix] = max_block_length_shuffled
    return df

def add_mean_block_lengths(df, suffix=""):
    mean_block_length = []
    mean_block_length_shuffled = []
    for index, cluster_df in df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        block_lengths = pandas_collumn_to_numpy_array(cluster_df["rolling:block_lengths"+suffix])
        block_lengths_shuffled = pandas_collumn_to_numpy_array(cluster_df["rolling:block_lengths_shuffled"+suffix])
        if len(block_lengths)>0:
            mean_block_length.append(np.nanmean(block_lengths))
        else:
            mean_block_length.append(0)
        if len(block_lengths_shuffled)>0:
            mean_block_length_shuffled.append(np.nanmean(block_lengths_shuffled))
        else:
            mean_block_length_shuffled.append(0)

    df["rolling:mean_block_lengths"+suffix] = mean_block_length
    df["rolling:mean_block_lengths_shuffled"+suffix] = mean_block_length_shuffled
    return df

def add_median_block_lengths(df, suffix=""):
    median_block_length = []
    median_block_length_shuffled = []
    for index, cluster_df in df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        block_lengths = pandas_collumn_to_numpy_array(cluster_df["rolling:block_lengths"+suffix])
        block_lengths_shuffled = pandas_collumn_to_numpy_array(cluster_df["rolling:block_lengths_shuffled"+suffix])
        if len(block_lengths)>0:
            median_block_length.append(np.nanmedian(block_lengths))
        else:
            median_block_length.append(0)
        if len(block_lengths_shuffled)>0:
            median_block_length_shuffled.append(np.nanmedian(block_lengths_shuffled))
        else:
            median_block_length_shuffled.append(0)

    df["rolling:median_block_lengths"+suffix] = median_block_length
    df["rolling:median_block_lengths_shuffled"+suffix] = median_block_length_shuffled
    return df

def plot_rolling_lomb_block_lengths_vs_shuffled(combined_df, save_path):
    print("do stuff")
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    grid_cells = add_max_block_lengths(grid_cells)

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    #ax.set_xticks([0,1])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([-1, 1])
    #ax.set_xlim([0, 1])
    block_lengths = pandas_collumn_to_numpy_array(grid_cells["rolling:block_lengths"])
    block_lengths_shuffled = pandas_collumn_to_numpy_array(grid_cells["rolling:block_lengths_shuffled"])
    _, _, patches0 = ax.hist(block_lengths, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="red",linewidth=2)
    _, _, patches1 = ax.hist(block_lengths_shuffled, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="grey",linewidth=2)
    patches0[0].set_xy(patches0[0].get_xy()[:-1])
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    #ax.set_xlabel("Frac. session", fontsize=20)
    #ax.set_ylabel("Density", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/block_lengths_vs_shuffled_trials.png', dpi=300)
    plt.close()
    ks, p = scipy.stats.ks_2samp(block_lengths, block_lengths_shuffled)

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    #ax.set_xticks([0,1])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([-1, 1])
    #ax.set_xlim([0, 1])
    block_lengths = pandas_collumn_to_numpy_array(grid_cells["rolling:block_lengths_small_window"])
    block_lengths_shuffled = pandas_collumn_to_numpy_array(grid_cells["rolling:block_lengths_shuffled_small_window"])
    _, _, patches0 = ax.hist(block_lengths, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="red",linewidth=2)
    _, _, patches1 = ax.hist(block_lengths_shuffled, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="grey",linewidth=2)
    patches0[0].set_xy(patches0[0].get_xy()[:-1])
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    #ax.set_xlabel("Frac. session", fontsize=20)
    #ax.set_ylabel("Density", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/block_lengths_vs_shuffled_trials_small_window.png', dpi=300)
    plt.close()
    ks_small_window, p_small_window = scipy.stats.ks_2samp(block_lengths, block_lengths_shuffled)


    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_xticks([0,1])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([-1, 1])
    ax.set_xlim([0, 1])
    max_block_lengths = pandas_collumn_to_numpy_array(grid_cells["rolling:max_block_lengths"])
    max_block_lengths_shuffled = pandas_collumn_to_numpy_array(grid_cells["rolling:max_block_lengths_shuffled"])
    _, _, patches0 = ax.hist(max_block_lengths, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="red",linewidth=2)
    _, _, patches1 = ax.hist(max_block_lengths_shuffled, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="grey",linewidth=2)
    patches0[0].set_xy(patches0[0].get_xy()[:-1])
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    #ax.set_xlabel("Frac. session", fontsize=20)
    #ax.set_ylabel("Density", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_xscale('log')
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/max_block_lengths_vs_shuffled_trials.png', dpi=300)
    plt.close()
    ks, p = scipy.stats.ks_2samp(max_block_lengths, max_block_lengths_shuffled)

    suffix ="_small_window"
    grid_cells = add_max_block_lengths(grid_cells, suffix=suffix)
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_xticks([0,1])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([-1, 1])
    ax.set_xlim([0, 1])
    max_block_lengths = pandas_collumn_to_numpy_array(grid_cells["rolling:max_block_lengths"+suffix])
    max_block_lengths_shuffled = pandas_collumn_to_numpy_array(grid_cells["rolling:max_block_lengths_shuffled"+suffix])
    _, _, patches0 = ax.hist(max_block_lengths, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="red",linewidth=2)
    _, _, patches1 = ax.hist(max_block_lengths_shuffled, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="grey",linewidth=2)
    patches0[0].set_xy(patches0[0].get_xy()[:-1])
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    #ax.set_xlabel("Frac. session", fontsize=20)
    #ax.set_ylabel("Density", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_xscale('log')
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/max_block_lengths_vs_shuffled_trials'+suffix+'.png', dpi=300)
    plt.close()
    ks_small_window, p_small_window = scipy.stats.ks_2samp(max_block_lengths, max_block_lengths_shuffled)


    Position_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    Distance_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]
    Null_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Null"]
    unstable_cells = pd.concat([Position_grid_cells[Position_grid_cells["rolling:proportion_encoding_position"]<0.85],
                                Distance_grid_cells[Distance_grid_cells["rolling:proportion_encoding_distance"]<0.85],
                                Null_grid_cells[Null_grid_cells["rolling:proportion_encoding_null"]<0.85]], ignore_index=True)
    block_lengths = pandas_collumn_to_numpy_array(unstable_cells["rolling:block_lengths"])
    block_lengths_shuffled = pandas_collumn_to_numpy_array(unstable_cells["rolling:block_lengths_shuffled"])
    ks, p = scipy.stats.ks_2samp(block_lengths, block_lengths_shuffled)


    return


def get_allocentric_peak(frequency, avg_subset_powers, tolerance=0.05):
    local_maxima_idx, _ = signal.find_peaks(avg_subset_powers, height=0, distance=5)
    power = 0; freq=0; i_max=0
    for i in local_maxima_idx:
        if (distance_from_integer(frequency[i])<=tolerance) and (avg_subset_powers[i]>power):
            freq = frequency[i]
            power = avg_subset_powers[i]
            i_max = i
    return freq, power, i_max

def add_peak_width(combined_df):
    peak_widths = []
    for index, row in combined_df.iterrows():
        avg_powers = row["MOVING_LOMB_avg_power"]
        if np.sum(np.isnan(avg_powers))==0:
            width, _, _, _ = signal.peak_widths(avg_powers, np.array([np.nanargmax(avg_powers)]))
            width = width[0]*Settings.frequency_step
        else:
            width = np.nan
        peak_widths.append(width)
    combined_df["ML_peak_width"] = peak_widths
    return combined_df

def add_ROC_stats(df):
    tt0_TPRs = []; tt0_FPRs = []
    tt1_TPRs = []; tt1_FPRs = []
    for index, row in df.iterrows():
        cluster_df = row.to_frame().T.reset_index(drop=True)

        trial_numbers = np.array(cluster_df["behaviour_trial_numbers"].iloc[0])
        trial_types = np.array(cluster_df["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cluster_df["rolling:classifier_by_trial_number"].iloc[0])
        hit_try_run = np.array(cluster_df["behaviour_hit_try_miss"].iloc[0])

        nan_mask = rolling_classifiers != "nan"
        trial_numbers = trial_numbers[nan_mask]
        trial_types = trial_types[nan_mask]
        rolling_classifiers = rolling_classifiers[nan_mask]
        hit_try_run = hit_try_run[nan_mask]

        cluster_trial_ROC_classification = []
        for i, tn in enumerate(trial_numbers):
            if (hit_try_run[i] == "hit") and (rolling_classifiers[i] == "P"):
                cluster_trial_ROC_classification.append("TP")
            elif (hit_try_run[i] != "hit") and (rolling_classifiers[i] == "P"):
                cluster_trial_ROC_classification.append("FP")
            elif (hit_try_run[i] != "hit") and (rolling_classifiers[i] != "P"):
                cluster_trial_ROC_classification.append("TN")
            elif (hit_try_run[i] == "hit") and (rolling_classifiers[i] != "P"):
                cluster_trial_ROC_classification.append("FN")
            else:
                print("something went wrong")
        cluster_trial_ROC_classification = np.array(cluster_trial_ROC_classification)

        tt0_classifications = cluster_trial_ROC_classification[trial_types == 0]
        tt1_classifications = cluster_trial_ROC_classification[trial_types == 1]

        if (len(tt0_classifications[tt0_classifications=="TP"])+len(tt0_classifications[tt0_classifications=="FN"]))>0:
            tt0_TPR = len(tt0_classifications[tt0_classifications=="TP"])/(len(tt0_classifications[tt0_classifications=="TP"])+len(tt0_classifications[tt0_classifications=="FN"]))
        else:
            tt0_TPR = np.nan

        if (len(tt0_classifications[tt0_classifications=="FP"])+len(tt0_classifications[tt0_classifications=="TN"]))>0:
            tt0_FPR = len(tt0_classifications[tt0_classifications=="FP"])/(len(tt0_classifications[tt0_classifications=="FP"])+len(tt0_classifications[tt0_classifications=="TN"]))
        else:
            tt0_FPR = np.nan

        if (len(tt1_classifications[tt1_classifications=="TP"])+len(tt1_classifications[tt1_classifications=="FN"]))>0:
            tt1_TPR = len(tt1_classifications[tt1_classifications=="TP"])/(len(tt1_classifications[tt1_classifications=="TP"])+len(tt1_classifications[tt1_classifications=="FN"]))
        else:
            tt1_TPR = np.nan

        if (len(tt1_classifications[tt1_classifications=="FP"])+len(tt1_classifications[tt1_classifications=="TN"]))>0:
            tt1_FPR = len(tt1_classifications[tt1_classifications=="FP"])/(len(tt1_classifications[tt1_classifications=="FP"])+len(tt1_classifications[tt1_classifications=="TN"]))
        else:
            tt1_FPR = np.nan

        tt0_TPRs.append(tt0_TPR); tt0_FPRs.append(tt0_FPR)
        tt1_TPRs.append(tt1_TPR); tt1_FPRs.append(tt1_FPR)

    df["ROC:tt0_TPR"] = tt0_TPRs
    df["ROC:tt0_FPR"] = tt0_FPRs
    df["ROC:tt1_TPR"] = tt1_TPRs
    df["ROC:tt1_FPR"] = tt1_FPRs
    return df

def plot_spatial_information_on_hmt_trials(combined_df, save_path, fig_size):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    for tt, tt_string in zip([0,1,2], ["beaconed", "non_beaconed", "probe"]):
        hit_scores = grid_cells["hit_spatial_information_score_tt"+str(tt)]
        try_scores = grid_cells["try_spatial_information_score_tt"+str(tt)]
        miss_scores = grid_cells["miss_spatial_information_score_tt"+str(tt)]

        #hit_scores = grid_cells["hmt_hit_tt_"+tt_string]
        #try_scores = grid_cells["hmt_try_tt_"+tt_string]
        #miss_scores = grid_cells["hmt_miss_tt_"+tt_string]

        nan_mask = ~np.isnan(hit_scores) & ~np.isnan(try_scores) & ~np.isnan(miss_scores)
        hit_scores = hit_scores[nan_mask]
        try_scores = try_scores[nan_mask]
        miss_scores = miss_scores[nan_mask]

        fig, ax = plt.subplots(figsize=fig_size)
        #for i in range(len(hit_scores)):
        #    ax.scatter([0,1,2], [hit_scores.iloc[i], try_scores.iloc[i], miss_scores.iloc[i]], marker="o", color="black",alpha=0.1)
        #    ax.plot([0,1,2],  [hit_scores.iloc[i], try_scores.iloc[i], miss_scores.iloc[i]], color="black", alpha=0.1)

        data = [hit_scores, try_scores, miss_scores, hit_scores, hit_scores, hit_scores, hit_scores]
        colors=[get_hmt_color("hit"), get_hmt_color("try"), get_hmt_color("miss"), get_hmt_color("hit"), get_hmt_color("hit"), get_hmt_color("hit"), get_hmt_color("hit")]
        boxprops = dict(linewidth=3, color='k')
        medianprops = dict(linewidth=3, color='k')
        capprops = dict(linewidth=3, color='k')
        whiskerprops = dict(linewidth=3, color='k')
        box = ax.boxplot(data, positions=[1,2,3,4,5,6,7], boxprops=boxprops, medianprops=medianprops,
                         whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        #ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(left=0.5, right=3.5)
        #ax.bar(0, np.nanmean(position_scores))
        #ax.bar(1, np.nanmean(distance_scores))
        #ax.set_ylim([0,1])
        ax.set_xticks([1,2])
        plt.xticks(rotation = 30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("Spatial info bits/spike", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        fig.tight_layout()
        plt.savefig(save_path + '/' +'hmt_spatial_scores_tt'+str(tt)+'.png', dpi=300)
        plt.close()
    return


def plot_mean_firing_rates_on_position_and_distance_trials(combined_df, save_path, fig_size, suffix="", unlock_y=False):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    position_scores = grid_cells["rolling:position_mean_firing_rate"]
    distance_scores = grid_cells["rolling:distance_mean_firing_rate"]
    nan_mask = ~np.isnan(position_scores) & ~np.isnan(distance_scores)
    position_scores = position_scores[nan_mask]
    distance_scores = distance_scores[nan_mask]

    print("comping mean firing rates _on_position_and_distance_trials,"+
          " p = ", str(stats.wilcoxon(x=position_scores, y=distance_scores)[1]),
          " t = ", str(stats.wilcoxon(x=position_scores, y=distance_scores)[0]),
          ", df = ",str(len(position_scores)-1))
    _, p = stats.wilcoxon(position_scores, distance_scores)

    fig, ax = plt.subplots(figsize=fig_size)
    #for i in range(len(position_scores)):
    #    ax.scatter([0,1], [position_scores.iloc[i], distance_scores.iloc[i]], marker="o", color="black",alpha=0.1)
    #    ax.plot([0,1],  [position_scores.iloc[i], distance_scores.iloc[i]], color="black", alpha=0.1)

    data = [position_scores, distance_scores, position_scores, distance_scores, position_scores, distance_scores, position_scores]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2, 4,4,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if unlock_y:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(bottom=0, top=15)
        ax.set_yticks([0,5,10,15])
    ax.set_xlim(left=0.5, right=3.5)
    #ax.bar(0, np.nanmean(position_scores))
    #ax.bar(1, np.nanmean(distance_scores))
    ax.set_xticks([1,2])
    plt.xticks(rotation = 30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_ylabel("Mean firing rate", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=25)
    fig.tight_layout()
    plt.savefig(save_path + '/' +'position_vs_distance_trials_mfr_'+suffix+'.png', dpi=300)
    plt.close()

def plot_spatial_information_on_position_and_distance_trials(combined_df, save_path, fig_size, suffix="", unlock_y=False):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    position_scores = grid_cells["rolling:position_spatial_information_scores_Isec"]
    distance_scores = grid_cells["rolling:distance_spatial_information_scores_Isec"]
    nan_mask = ~np.isnan(position_scores) & ~np.isnan(distance_scores)
    position_scores = position_scores[nan_mask]
    distance_scores = distance_scores[nan_mask]

    print("comping _spatial_information_on_position_and_distance_trials,"+
          " p = ", str(stats.wilcoxon(x=position_scores, y=distance_scores)[1]),
          " t = ", str(stats.wilcoxon(x=position_scores, y=distance_scores)[0]),
          ", df = ",str(len(position_scores)-1))
    _, p = stats.wilcoxon(position_scores, distance_scores)



    fig, ax = plt.subplots(figsize=fig_size)
    #for i in range(len(position_scores)):
    #    ax.scatter([0,1], [position_scores.iloc[i], distance_scores.iloc[i]], marker="o", color="black",alpha=0.1)
    #    ax.plot([0,1],  [position_scores.iloc[i], distance_scores.iloc[i]], color="black", alpha=0.1)

    data = [position_scores, distance_scores, position_scores, distance_scores, position_scores, distance_scores, position_scores]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2, 4,4,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if unlock_y:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(bottom=0, top=3)
    ax.set_xlim(left=0.5, right=3.5)
    #ax.bar(0, np.nanmean(position_scores))
    #ax.bar(1, np.nanmean(distance_scores))
    ax.set_xticks([1,2])
    ax.set_yticks([0,1, 2, 3])
    ax.set_yticklabels(["00", "01", "02", "03"])
    plt.xticks(rotation = 30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_ylabel("Spatial info bits/s", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=25)
    fig.tight_layout()
    plt.savefig(save_path + '/' +'position_vs_distance_trials_spatial_scores_'+suffix+'.png', dpi=300)
    plt.close()

def behaviours_classification_to_numeric(behaviours_classifications):
    numeric_classifications = []
    for i in range(len(behaviours_classifications)):
        if behaviours_classifications[i] == "hit":
            numeric_classifications.append(1)
        elif behaviours_classifications[i] == "try":
            numeric_classifications.append(0.5)
        elif behaviours_classifications[i] == "miss":
            numeric_classifications.append(0)
        else:
            numeric_classifications.append(np.nan)
    return np.array(numeric_classifications)

def generate_short_form_for_R_by_session(combined_df):
    df = pd.DataFrame()
    # iterate over clusters
    for index, cluster_row in combined_df.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)
        cluster_id = cluster_row["cluster_id"].iloc[0]
        session_id = cluster_row["session_id"].iloc[0]
        mouse_id = cluster_row["mouse"].iloc[0]
        classifier = cluster_row["classifier"].iloc[0]
        grid_cell = cluster_row["grid_cell"].iloc[0]
        Lomb_classifier = cluster_row["Lomb_classifier_"].iloc[0]

        if Lomb_classifier == "Position":
            rolling_proportion_column = "rolling:proportion_encoding_position"
        if Lomb_classifier == "Distance":
            rolling_proportion_column = "rolling:proportion_encoding_distance"
        if Lomb_classifier == "Null":
            rolling_proportion_column = "rolling:proportion_encoding_null"
        proportion_encoding = cluster_row[rolling_proportion_column].iloc[0]

        behaviours = np.array(cluster_row["behaviour_hit_try_miss"].iloc[0])
        hits_binary = np.array(behaviours == "hit", dtype=np.int0)
        trial_types = np.array(cluster_row["behaviour_trial_types"].iloc[0])

        for trial_type in [0,1,2]:
            session_df = pd.DataFrame()
            session_df["mouse_id"] = [mouse_id]
            session_df["session_id"] = [session_id]
            session_df["cluster_id"] = [cluster_id]
            session_df["classifier"] = [classifier]
            session_df["grid_cell"] = [grid_cell]
            session_df["session_id_cluster_id"] = [str(session_id) + "_" + str(cluster_id)]
            session_df["trial_type"] = [trial_type]
            session_df["Lomb_classifier"] = [Lomb_classifier]
            session_df["proportion_encoding_Lomb_classifier"] = [proportion_encoding]

            n_hits = np.sum(hits_binary[(trial_types == trial_type)])
            n_trials = np.sum(trial_types == trial_type)
            session_df["n_hits"] = [n_hits]
            session_df["n_trials"] = [n_trials]
            df = pd.concat([df, session_df], ignore_index=True)

    return df

def generate_long_form_for_R_by_network_epochs(combined_df):
    df = pd.DataFrame()
    # iterate over position_sessions
    for index, session_row in combined_df.iterrows():
        session_row = session_row.to_frame().T.reset_index(drop=True)
        session_id = session_row["session_id"].iloc[0]
        mouse_id = session_row["mouse"].iloc[0]
        behaviours = np.array(session_row["behaviour_hit_try_miss"].iloc[0])
        hits_binary = np.array(behaviours == "hit", dtype=np.int0)
        trial_types = np.array(session_row["behaviour_trial_types"].iloc[0])
        trial_numbers = np.array(session_row["behaviour_trial_numbers"].iloc[0])
        rolling_classifier_network = np.array(session_row["rolling:code_global"].iloc[0])

        # remove rejected trials
        valid_trials_mask = behaviours != "rejected"
        behaviours = behaviours[valid_trials_mask]
        hits_binary = hits_binary[valid_trials_mask]
        trial_types = trial_types[valid_trials_mask]
        trial_numbers = trial_numbers[valid_trials_mask]
        rolling_classifier_network = rolling_classifier_network[valid_trials_mask]

        for i, tn in enumerate(trial_numbers):
            trial_df = pd.DataFrame()
            trial_df["mouse_id"] = [mouse_id]
            trial_df["session_id"] = [session_id]
            trial_df["trial_type"] = [trial_types[i]]
            trial_df["rolling_classifier"] = [rolling_classifier_network[i]]
            trial_df["trial_number"] = [tn]
            trial_df["hit"] = [hits_binary[i]]
            trial_df["behavioural_outcome"] = [behaviours[i]]
            df = pd.concat([df, trial_df], ignore_index=True)
    return df

def generate_short_form_for_R_by_network_epochs(combined_df):
    df = pd.DataFrame()
    # iterate over position_sessions
    for index, session_row in combined_df.iterrows():
        session_row = session_row.to_frame().T.reset_index(drop=True)
        session_id = session_row["session_id"].iloc[0]
        mouse_id = session_row["mouse"].iloc[0]
        behaviours = np.array(session_row["behaviour_hit_try_miss"].iloc[0])
        hits_binary = np.array(behaviours == "hit", dtype=np.int0)
        trial_types = np.array(session_row["behaviour_trial_types"].iloc[0])
        trial_numbers = np.array(session_row["behaviour_trial_numbers"].iloc[0])
        rolling_classifier_network = np.array(session_row["rolling:code_global"].iloc[0])

        # remove rejected trials
        valid_trials_mask = behaviours != "rejected"
        behaviours = behaviours[valid_trials_mask]
        hits_binary = hits_binary[valid_trials_mask]
        trial_types = trial_types[valid_trials_mask]
        trial_numbers = trial_numbers[valid_trials_mask]
        rolling_classifier_network = rolling_classifier_network[valid_trials_mask]

        for trial_type in [0,1,2]:
            for code in ["P", "D", "N"]:
                session_df = pd.DataFrame()
                session_df["mouse_id"] = [mouse_id]
                session_df["session_id"] = [session_id]
                session_df["trial_type"] = [trial_type]
                session_df["rolling_classifier"] = [code]
                n_hits = np.sum(hits_binary[(rolling_classifier_network == code) & (trial_types == trial_type)])
                n_trials = np.sum((rolling_classifier_network == code) & (trial_types == trial_type))
                session_df["n_hits"] = [n_hits]
                session_df["n_trials"] = [n_trials]
                df = pd.concat([df, session_df], ignore_index=True)
    return df

def generate_long_form_for_R_by_network_session(combined_df):
    position_sessions = combined_df[combined_df["proportion_network_encoding_position"] >= 0.85]
    distance_sessions = combined_df[combined_df["proportion_network_encoding_distance"] >= 0.85]

    df = pd.DataFrame()
    # iterate over position_sessions
    for session_dfs, session_type in zip([position_sessions, distance_sessions], ["Position", "Distance"]):
        for index, session_row in session_dfs.iterrows():
            session_row = session_row.to_frame().T.reset_index(drop=True)
            session_id = session_row["session_id"].iloc[0]
            mouse_id = session_row["mouse"].iloc[0]
            behaviours = np.array(session_row["behaviour_hit_try_miss"].iloc[0])
            hits_binary = np.array(behaviours == "hit", dtype=np.int0)
            trial_types = np.array(session_row["behaviour_trial_types"].iloc[0])
            trial_numbers = np.array(session_row["behaviour_trial_numbers"].iloc[0])

            # remove rejected trials
            valid_trials_mask = behaviours != "rejected"
            behaviours = behaviours[valid_trials_mask]
            hits_binary = hits_binary[valid_trials_mask]
            trial_types = trial_types[valid_trials_mask]
            trial_numbers = trial_numbers[valid_trials_mask]

            for i, tn in enumerate(trial_numbers):
                trial_df = pd.DataFrame()
                trial_df["mouse_id"] = [mouse_id]
                trial_df["session_id"] = [session_id]
                trial_df["trial_type"] = [trial_types[i]]
                trial_df["Lomb_classifier"] = [session_type]
                trial_df["trial_number"] = [tn]
                trial_df["hit"] = [hits_binary[i]]
                df = pd.concat([df, trial_df], ignore_index=True)
    return df


def generate_short_form_for_R_by_network_session(combined_df):
    position_sessions = combined_df[combined_df["proportion_network_encoding_position"] >= 0.85]
    distance_sessions = combined_df[combined_df["proportion_network_encoding_distance"] >= 0.85]

    df = pd.DataFrame()
    # iterate over position_sessions
    for session_dfs, session_type in zip([position_sessions, distance_sessions], ["Position", "Distance"]):
        for index, session_row in session_dfs.iterrows():
            session_row = session_row.to_frame().T.reset_index(drop=True)
            session_id = session_row["session_id"].iloc[0]
            mouse_id = session_row["mouse"].iloc[0]
            behaviours = np.array(session_row["behaviour_hit_try_miss"].iloc[0])
            hits_binary = np.array(behaviours == "hit", dtype=np.int0)
            trial_types = np.array(session_row["behaviour_trial_types"].iloc[0])

            # remove rejected trials
            valid_trials_mask = behaviours != "rejected"
            behaviours = behaviours[valid_trials_mask]
            hits_binary = hits_binary[valid_trials_mask]
            trial_types = trial_types[valid_trials_mask]

            for trial_type in [0,1,2]:
                session_df = pd.DataFrame()
                session_df["mouse_id"] = [mouse_id]
                session_df["session_id"] = [session_id]
                session_df["trial_type"] = [trial_type]
                session_df["Lomb_classifier"] = [session_type]
                n_hits = np.sum(hits_binary[(trial_types == trial_type)])
                n_trials = np.sum(trial_types == trial_type)
                session_df["n_hits"] = [n_hits]
                session_df["n_trials"] = [n_trials]
                df = pd.concat([df, session_df], ignore_index=True)
    return df

def generate_short_form_for_R_by_trial(combined_df):
    df = pd.DataFrame()
    # iterate over clusters
    for index, cluster_row in combined_df.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)
        cluster_id = cluster_row["cluster_id"].iloc[0]
        session_id = cluster_row["session_id"].iloc[0]
        classifier = cluster_row["classifier"].iloc[0]
        grid_cell = cluster_row["grid_cell"].iloc[0]
        mouse_id = cluster_row["mouse"].iloc[0]

        trial_numbers = np.array(cluster_row["behaviour_trial_numbers"].iloc[0])
        behaviours = np.array(cluster_row["behaviour_hit_try_miss"].iloc[0])
        behaviours_numeric = behaviours_classification_to_numeric(behaviours)
        hits_binary = np.array(behaviours == "hit", dtype=np.int0)
        trial_types = np.array(cluster_row["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cluster_row["rolling:classifier_by_trial_number"].iloc[0])

        for trial_type in [0,1,2]:
            for code in ["P", "D", "N"]:
                session_df = pd.DataFrame()
                session_df["mouse_id"] = [mouse_id]
                session_df["session_id"] = [session_id]
                session_df["cluster_id"] = [cluster_id]
                session_df["classifier"] = [classifier]
                session_df["grid_cell"] = [grid_cell]
                session_df["session_id_cluster_id"] = [str(session_id) + "_" + str(cluster_id)]
                session_df["trial_type"] = [trial_type]
                session_df["rolling_classifier"] = [code]

                n_hits = np.sum(hits_binary[(rolling_classifiers == code) & (trial_types == trial_type)])
                n_trials = np.sum((rolling_classifiers == code) & (trial_types == trial_type))
                session_df["n_hits"] = [n_hits]
                session_df["n_trials"] = [n_trials]
                df = pd.concat([df, session_df], ignore_index=True)

    return df

def generate_long_form_for_R_by_session(combined_df):
    position_sessions = combined_df[combined_df["proportion_network_encoding_position"] >= 0.85]
    distance_sessions = combined_df[combined_df["proportion_network_encoding_distance"] >= 0.85]

    df = pd.DataFrame()
    # iterate over position_sessions
    for session_dfs, session_type in zip([position_sessions, distance_sessions], ["Position", "Distance"]):
        for index, session_row in session_dfs.iterrows():
            session_row = session_row.to_frame().T.reset_index(drop=True)
            session_id = session_row["session_id"].iloc[0]
            cluster_id = session_row["cluster_id"].iloc[0]
            mouse_id = session_row["mouse"].iloc[0]
            behaviours = np.array(session_row["behaviour_hit_try_miss"].iloc[0])
            hits_binary = np.array(behaviours == "hit", dtype=np.int0)
            trial_types = np.array(session_row["behaviour_trial_types"].iloc[0])
            trial_numbers = np.array(session_row["behaviour_trial_numbers"].iloc[0])

            # remove rejected trials
            valid_trials_mask = behaviours != "rejected"
            behaviours = behaviours[valid_trials_mask]
            hits_binary = hits_binary[valid_trials_mask]
            trial_types = trial_types[valid_trials_mask]
            trial_numbers = trial_numbers[valid_trials_mask]

            for i, tn in enumerate(trial_numbers):
                trial_df = pd.DataFrame()
                trial_df["mouse_id"] = [mouse_id]
                trial_df["session_id"] = [session_id]
                trial_df["cluster_id"] = [cluster_id]
                trial_df["trial_type"] = [trial_types[i]]
                trial_df["Lomb_classifier"] = [session_type]
                trial_df["trial_number"] = [tn]
                trial_df["hit"] = [hits_binary[i]]
                trial_df["behavioural_outcome"] = [behaviours[i]]
                df = pd.concat([df, trial_df], ignore_index=True)
    return df

def generate_long_form_for_R(combined_df):
    df = pd.DataFrame()
    # iterate over clusters
    for index, cluster_row in combined_df.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)
        cluster_id = cluster_row["cluster_id"].iloc[0]
        session_id = cluster_row["session_id"].iloc[0]
        classifier = cluster_row["classifier"].iloc[0]
        grid_cell = cluster_row["grid_cell"].iloc[0]
        mouse_id = cluster_row["mouse"].iloc[0]
        trial_numbers = np.array(cluster_row["behaviour_trial_numbers"].iloc[0])
        behaviours = np.array(cluster_row["behaviour_hit_try_miss"].iloc[0])
        behaviours_numeric = behaviours_classification_to_numeric(behaviours)
        hits_binary = np.array(behaviours == "hit", dtype=np.int0)
        trial_types = np.array(cluster_row["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cluster_row["rolling:classifier_by_trial_number"].iloc[0])
        p_coding = np.array(rolling_classifiers == "P", dtype=np.int0)
        d_coding = np.array(rolling_classifiers == "D", dtype=np.int0)
        n_coding = np.array(rolling_classifiers == "N", dtype=np.int0)

        # remove rejected trials
        valid_trials_mask = behaviours != "rejected"
        trial_numbers = trial_numbers[valid_trials_mask]
        behaviours = behaviours[valid_trials_mask]
        behaviours_numeric = behaviours_numeric[valid_trials_mask]
        hits_binary = hits_binary[valid_trials_mask]
        trial_types = trial_types[valid_trials_mask]
        rolling_classifiers = rolling_classifiers[valid_trials_mask]
        p_coding = p_coding[valid_trials_mask]
        d_coding = d_coding[valid_trials_mask]
        n_coding = n_coding[valid_trials_mask]

        trial_df = pd.DataFrame()
        for tn, hit_binary, behaviour_numeric, behaviour, tt, code, p_code, d_code, n_code in \
                zip(trial_numbers, hits_binary, behaviours_numeric, behaviours, trial_types, rolling_classifiers, p_coding, d_coding, n_coding):
            trial_df = pd.DataFrame()
            trial_df["trial_number"] = [tn]
            trial_df["hit"] = [hit_binary]
            trial_df["behaviour_numeric"] = [behaviour_numeric]
            trial_df["behavioural_outcome"] = [behaviour]
            trial_df["trial_type"] = [tt]
            trial_df["rolling_classifier"] = [code]
            trial_df["p_coding"] = [p_code]
            trial_df["d_coding"] = [d_code]
            trial_df["n_coding"] = [n_code]
            trial_df["mouse_id"] = [mouse_id]
            trial_df["session_id"] = [session_id]
            trial_df["cluster_id"] = [cluster_id]
            trial_df["classifier"] = [classifier]
            trial_df["grid_cell"] = [grid_cell]
            df = pd.concat([df, trial_df], ignore_index=True)
    return df

def plot_odds_ratio(odds_ratio_data, save_path):
    beaconed_odds_ratio = odds_ratio_data[odds_ratio_data["trial_type"] == "beaconed"]
    non_beaconed_odds_ratio = odds_ratio_data[odds_ratio_data["trial_type"] == "non_beaconed"]
    probe_odds_ratio = odds_ratio_data[odds_ratio_data["trial_type"] == "probe"]

    fig, ax = plt.subplots(figsize=(4,6))
    y = 10
    y_increment = 0.3
    linewidth=3
    s=75
    cell_types = ["g", "ng", "s", "b", "hd", "ns"]
    cell_colors = [(255/255,127/255,14/255), (0/255, 154/255, 255/255),(188/255,189/255,35/255), (140/255, 86/255, 75/255), (227/255, 119/255, 194/255), (127/255,127/255,127/255)]

    for cell_type, cell_color in zip(cell_types, cell_colors):
        ax.plot([beaconed_odds_ratio[beaconed_odds_ratio["cell_type"] == cell_type]["ci_lower"].iloc[0],
                 beaconed_odds_ratio[beaconed_odds_ratio["cell_type"] == cell_type]["ci_upper"].iloc[0]], [y, y], color=cell_color, linewidth=linewidth)
        ax.scatter(y=y, x=beaconed_odds_ratio[beaconed_odds_ratio["cell_type"] == cell_type]["odds_ratio"].iloc[0],color=cell_color, s=s)
        y -= y_increment

    y -= 0.75

    for cell_type, cell_color in zip(cell_types, cell_colors):
        ax.plot([non_beaconed_odds_ratio[non_beaconed_odds_ratio["cell_type"] == cell_type]["ci_lower"].iloc[0],
                 non_beaconed_odds_ratio[non_beaconed_odds_ratio["cell_type"] == cell_type]["ci_upper"].iloc[0]], [y, y], color=cell_color, linewidth=linewidth)
        ax.scatter(y=y, x=non_beaconed_odds_ratio[non_beaconed_odds_ratio["cell_type"] == cell_type]["odds_ratio"].iloc[0],color=cell_color, s=s)
        y -= y_increment

    y -= 0.75

    for cell_type, cell_color in zip(cell_types, cell_colors):
        ax.plot([probe_odds_ratio[probe_odds_ratio["cell_type"] == cell_type]["ci_lower"].iloc[0],
                 probe_odds_ratio[probe_odds_ratio["cell_type"] == cell_type]["ci_upper"].iloc[0]], [y, y], color=cell_color, linewidth=linewidth)
        ax.scatter(y=y, x=probe_odds_ratio[probe_odds_ratio["cell_type"] == cell_type]["odds_ratio"].iloc[0],color=cell_color, s=s)
        y -= y_increment


    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xscale('log')
    ax.set_xticks([      0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1,  2,  3,  4,  5,   6,  7,  8,  9,  10])
    ax.set_xticklabels(["0.1",  "",  "",  "","0.5", "",  "",  "",  "", "1","", "", "", "5", "", "", "", "", "10"])
    #ax.set_ylim(0.5, 10.5)
    ax.set_yticks([4.9, 7.15, 9.4])
    ax.set_yticklabels(["P", "N", "B"])
    #ax.set_xlim(left=0, right=9)
    ax.axvline(x=1, color="black",linestyle="dotted")
    #ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/odds_ratio.png', dpi=300)
    plt.close()
    return

def plot_average_firing_rate_map_during_position_and_distance_epochs(combined_df, save_path, fig_size, z_scored=False):
    # remove putative interneurons
    combined_df = combined_df[combined_df["mean_firing_rate_vr"]<15]

    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    for cells, cell_name in zip([grid_cells, non_grid_cells], ["grid_cells", "non_grid_cells"]):
        for tt in [0,1,""]:
            for hmt in ["hit", "try", "miss", ""]:

                fig, ax = plt.subplots(figsize=fig_size)
                for code, code_color in zip(["P", "D"], [Settings.allocentric_color, Settings.egocentric_color]):

                    avg_firing_rate_maps = []
                    for index, cell_row in cells.iterrows():
                        cell_row = cell_row.to_frame().T.reset_index(drop=True)
                        fr_binned_in_space_smoothed = np.array(cell_row["fr_binned_in_space_smoothed"].iloc[0])
                        behaviour_trial_numbers = np.array(cell_row["behaviour_trial_numbers"].iloc[0])
                        behaviour_hit_try_miss = np.array(cell_row["behaviour_hit_try_miss"].iloc[0])
                        behaviour_trial_types = np.array(cell_row["behaviour_trial_types"].iloc[0])
                        code_from_grid_cells = np.array(cell_row["average_rolling_classification_for_multiple_grid_cells"].iloc[0]) # excludes some trials around the start and finish
                        code_centre_trials_from_grid_cells = np.array(cell_row["average_rolling_centre_trials_for_multiple_grid_cells"].iloc[0]) # excludes some trials around the start and finish

                        # correct the arrays to represent the trials with a valid grid code
                        trial_mask = np.isin(behaviour_trial_numbers, code_centre_trials_from_grid_cells)
                        behaviour_trial_numbers = behaviour_trial_numbers[trial_mask]
                        behaviour_hit_try_miss = behaviour_hit_try_miss[trial_mask]
                        behaviour_trial_types = behaviour_trial_types[trial_mask]

                        # remove non biologically likely firing rates
                        fr_binned_in_space_smoothed[fr_binned_in_space_smoothed>100] = np.nan

                        if tt is not "":
                            tt_mask = behaviour_trial_types == tt
                        else:
                            tt_mask = np.ones(len(behaviour_trial_types), dtype=bool)

                        if hmt is not "":
                            hmt_mask = behaviour_hit_try_miss == hmt
                        else:
                            hmt_mask = np.ones(len(behaviour_hit_try_miss), dtype=bool)

                        code_mask = code_from_grid_cells == code

                        mask = tt_mask & hmt_mask & code_mask
                        trial_numbers = behaviour_trial_numbers[mask]
                        fr = fr_binned_in_space_smoothed[trial_numbers-1]
                        avg_fr = np.nanmean(fr, axis=0)
                        if z_scored:
                            avg_fr = stats.zscore(avg_fr)
                        avg_firing_rate_maps.append(avg_fr.tolist())

                    locations = np.arange(0, len(avg_firing_rate_maps[0]))
                    # for i in range(len(position_scores)):
                    #    ax.scatter([0,1], [position_scores.iloc[i], distance_scores.iloc[i]], marker="o", color="black",alpha=0.1)
                    #    ax.plot([0,1],  [position_scores.iloc[i], distance_scores.iloc[i]], color="black", alpha=0.1)
                    ax.plot(locations, np.nanmean(np.array(avg_firing_rate_maps), axis=0), color=code_color, linewidth=3)
                    ax.fill_between(locations,
                                    np.nanmean(np.array(avg_firing_rate_maps), axis=0) - stats.sem(
                                        np.array(avg_firing_rate_maps), axis=0, nan_policy="omit"),
                                    np.nanmean(np.array(avg_firing_rate_maps), axis=0) + stats.sem(
                                        np.array(avg_firing_rate_maps), axis=0, nan_policy="omit"),
                                    color=code_color, alpha=0.2)

                plt.ylabel('FR (Hz)', fontsize=25, labelpad=10)
                plt.xlabel('Location (cm)', fontsize=25, labelpad=10)
                plt.xlim(0, 200)
                ax.tick_params(axis='both', which='both', labelsize=20)
                ax.set_xlim([0, 200])
                max_fr = max(np.nanmean(np.array(avg_firing_rate_maps), axis=0) + stats.sem(np.array(avg_firing_rate_maps), axis=0))
                max_fr = max_fr + (0.1 * (max_fr))
                # ax.set_ylim([0, max_fr])
                #ax.set_yticks([0, np.round(ax.get_ylim()[1], 1)])
                #ax.set_ylim(bottom=0)
                # Edmond.plot_utility2.style_track_plot(ax, track_length, alpha=0.25)
                if tt == 1:
                    style_track_plot_no_RZ(ax, 200)
                else:
                    style_track_plot(ax, 200)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
                # ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.3, right=0.87, top=0.92)
                plt.savefig(save_path + '/' + 'position_vs_distance_trials_from_grid_cells_FIRING_RATE_MAPS_tt_'+str(tt)+'_hmt_'+str(hmt)+'_'+str(cell_name)+'.png', dpi=300)
                plt.close()

    return


def plot_comparison_grids_vs_non_grids_during_position_and_distance_epochs(combined_df, save_path, fig_size):
    print("")
    # this function serves to remove all self references and any references with grid-cell-grid-cell interactions
    # this allows us to only look at grid-cell-non-grid-cell interactions
    def remove_grid_references(grid_cells):
        spatial_information_scores_P = []
        spatial_information_scores_D = []
        cluster_pairs = []
        for index, spike_row in grid_cells.iterrows():
            cluster_spike_data = spike_row.to_frame().T.reset_index(drop=True)
            session_id = cluster_spike_data["session_id"].iloc[0]
            cluster_id = cluster_spike_data["cluster_id"].iloc[0]
            grid_session_df = grid_cells[grid_cells["session_id"] == session_id]
            cluster_ids_matching_grid_cells = grid_session_df["cluster_id"].values

            spatial_information_scores_P_cluster = np.array(cluster_spike_data["rolling:spatial_info_by_other_P"].iloc[0])
            spatial_information_scores_D_cluster = np.array(cluster_spike_data["rolling:spatial_info_by_other_D"].iloc[0])
            cluster_pairs_cluster = np.array(cluster_spike_data["rolling:spatial_info_by_other_ids"].iloc[0])

            spatial_information_scores_P_cluster = spatial_information_scores_P_cluster[~np.isin(cluster_pairs_cluster, cluster_ids_matching_grid_cells)]
            spatial_information_scores_D_cluster = spatial_information_scores_D_cluster[~np.isin(cluster_pairs_cluster, cluster_ids_matching_grid_cells)]
            cluster_pairs_cluster = cluster_pairs_cluster[~np.isin(cluster_pairs_cluster, cluster_ids_matching_grid_cells)]

            spatial_information_scores_P.append(spatial_information_scores_P_cluster.tolist())
            spatial_information_scores_D.append(spatial_information_scores_D_cluster.tolist())
            cluster_pairs.append(cluster_pairs_cluster.tolist())

        grid_cells["rolling:spatial_info_by_other_P"] = spatial_information_scores_P
        grid_cells["rolling:spatial_info_by_other_D"] = spatial_information_scores_D
        grid_cells["rolling:spatial_info_by_other_ids"] = cluster_pairs
        return grid_cells

    grid_cells = combined_df[combined_df["classifier"] == "G"]
    grid_cells = remove_grid_references(grid_cells)
    position_scores = pandas_collumn_to_numpy_array(grid_cells["rolling:spatial_info_by_other_P"])
    distance_scores = pandas_collumn_to_numpy_array(grid_cells["rolling:spatial_info_by_other_D"])

    nan_mask = ~np.isnan(position_scores) & ~np.isnan(distance_scores)
    position_scores = position_scores[nan_mask]
    distance_scores = distance_scores[nan_mask]

    print("comping _spatial_information_on_position_and_distance_trials," +
          " p = ", str(stats.wilcoxon(x=position_scores, y=distance_scores)[1]),
          " t = ", str(stats.wilcoxon(x=position_scores, y=distance_scores)[0]),
          ", df = ", str(len(position_scores) - 1))
    _, p = stats.wilcoxon(position_scores, distance_scores)

    fig, ax = plt.subplots(figsize=fig_size)
    # for i in range(len(position_scores)):
    #    ax.scatter([0,1], [position_scores.iloc[i], distance_scores.iloc[i]], marker="o", color="black",alpha=0.1)
    #    ax.plot([0,1],  [position_scores.iloc[i], distance_scores.iloc[i]], color="black", alpha=0.1)

    data = [position_scores, distance_scores, position_scores, distance_scores, position_scores, distance_scores,
            position_scores]
    colors = [Settings.allocentric_color, Settings.egocentric_color, Settings.egocentric_color,
              Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color,
              Settings.egocentric_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1, 2, 4, 4, 5, 6, 7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0.5, right=3.5)
    # ax.bar(0, np.nanmean(position_scores))
    # ax.bar(1, np.nanmean(distance_scores))
    ax.set_xticks([1, 2])
    plt.xticks(rotation=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Spatial info bits/spike", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=25)
    fig.tight_layout()
    plt.savefig(save_path + '/' + 'position_vs_distance_trials_from_grid_cells_other_cells_spatial_scores.png', dpi=300)
    plt.close()

    return

def get_agreements(cells_1, cells_2, shuffled=False, n_shuffles=2):
    agreements = []
    for index, cell_row in cells_1.iterrows():
        cell_row = cell_row.to_frame().T.reset_index(drop=True)
        cluster_id = cell_row["cluster_id"].iloc[0]
        session_id = cell_row["session_id"].iloc[0]
        cell_rolling_classifiers = np.array(cell_row["rolling:classifier_by_trial_number"].iloc[0])

        cells_2_in_same_session = cells_2[cells_2["session_id"] == session_id]
        for paired_index, paired_cell_row in cells_2_in_same_session.iterrows():
            paired_cell_row = paired_cell_row.to_frame().T.reset_index(drop=True)
            paired_cluster_id = paired_cell_row["cluster_id"].iloc[0]
            paired_session_id = paired_cell_row["session_id"].iloc[0]
            paired_cell_rolling_classifiers = np.array(paired_cell_row["rolling:classifier_by_trial_number"].iloc[0])

            if cluster_id != paired_cluster_id:
                if shuffled:
                    shuffled_agreements = np.zeros(n_shuffles)
                    for i in range(n_shuffles):
                        paired_cell_rolling_classifiers = shuffle_blocks(paired_cell_rolling_classifiers)
                        shuffled_agreement = np.array(cell_rolling_classifiers==paired_cell_rolling_classifiers, dtype=np.int64)
                        shuffled_score = np.sum(shuffled_agreement)/len(shuffled_agreement)
                        shuffled_agreements[i] = shuffled_score

                    agreements.append(np.nanmean(shuffled_agreements))
                else:
                    agreement = np.array(cell_rolling_classifiers==paired_cell_rolling_classifiers, dtype=np.int64)
                    score = np.sum(agreement)/len(agreement)
                    agreements.append(score)

    return np.array(agreements)


def get_agreements_between_grid_and_non_grid_network(cells, shuffled=False, n_shuffles=2):
    unique_sessions_with_grid_cells = np.unique(cells[cells["classifier"] == "G"]["session_id"])

    agreements=[]
    for session_id in unique_sessions_with_grid_cells:
        session_df = cells[cells["session_id"] == session_id]
        grids = session_df[session_df["classifier"] == "G"]
        non_grids = session_df[session_df["classifier"] != "G"]

        if len(non_grids)>0:
            grid_modes = pandas_collumn_to_2d_numpy_array(grids["rolling:classifier_by_trial_number"])
            non_grid_modes = pandas_collumn_to_2d_numpy_array(non_grids["rolling:classifier_by_trial_number"])

            if shuffled:
                shuffled_agreements = np.zeros(n_shuffles)
                for i in range(n_shuffles):
                    shuffled_non_grid_modes = np.zeros((len(non_grid_modes), len(non_grid_modes[0])), dtype=np.str)
                    for j in range(len(non_grid_modes)):
                        shuffled_non_grid_modes[j,:] = shuffle_blocks(non_grid_modes[j])

                    dominant_mode_grid_cells = stats.mode(grid_modes, axis=0)[0][0]
                    dominant_mode_non_grid_cells_shuffled = stats.mode(shuffled_non_grid_modes, axis=0)[0][0]

                    shuffled_agreement = np.array(dominant_mode_grid_cells == dominant_mode_non_grid_cells_shuffled, dtype=np.int64)
                    shuffled_score = np.sum(shuffled_agreement) / len(shuffled_agreement)
                    shuffled_agreements[i] = shuffled_score
                agreements.append(np.nanmean(shuffled_agreements))
            else:
                dominant_mode_grid_cells = stats.mode(grid_modes, axis=0)[0][0]
                dominant_mode_non_grid_cells = stats.mode(non_grid_modes, axis=0)[0][0]

                agreement = np.array(dominant_mode_grid_cells == dominant_mode_non_grid_cells, dtype=np.int64)
                score = np.sum(agreement) / len(agreement)
                agreements.append(score)
    
    return np.array(agreements)

def plot_agreement_between_grid_and_non_grids(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]

    combined_df = combined_df[combined_df["rolling:max_proportion_encoding"] <= 0.85]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    n_shuffles = 10

    gc_ngc_network_agreement_true = get_agreements_between_grid_and_non_grid_network(cells=combined_df, shuffled=False, n_shuffles=n_shuffles)
    gc_ngc_network_agreement_shuffled = get_agreements_between_grid_and_non_grid_network(cells=combined_df, shuffled=True, n_shuffles=n_shuffles)

    gc_gc_agreement_true = get_agreements(cells_1=grid_cells, cells_2=grid_cells, shuffled=False, n_shuffles=n_shuffles)
    gc_gc_agreement_shuffle = get_agreements(cells_1=grid_cells, cells_2=grid_cells, shuffled=True, n_shuffles=n_shuffles)
    ngc_ngc_agreement_true = get_agreements(cells_1=non_grid_cells, cells_2=non_grid_cells, shuffled=False, n_shuffles=n_shuffles)
    ngc_ngc_agreement_shuffle = get_agreements(cells_1=non_grid_cells, cells_2=non_grid_cells, shuffled=True, n_shuffles=n_shuffles)
    gc_ngc_agreement_true = get_agreements(cells_1=grid_cells, cells_2=non_grid_cells, shuffled=False, n_shuffles=n_shuffles)
    gc_ngc_agreement_shuffle = get_agreements(cells_1=grid_cells, cells_2=non_grid_cells, shuffled=True, n_shuffles=n_shuffles)

    x_pos = np.array([0,1,2,4])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(x_pos-0.22, [np.nanmean(gc_gc_agreement_true), np.nanmean(ngc_ngc_agreement_true), np.nanmean(gc_ngc_agreement_true), np.nanmean(gc_ngc_network_agreement_true)], 0.4, color="white", edgecolor="black", linewidth=3)
    ax.bar(x_pos+0.22, [np.nanmean(gc_gc_agreement_shuffle), np.nanmean(ngc_ngc_agreement_shuffle), np.nanmean(gc_ngc_agreement_shuffle), np.nanmean(gc_ngc_network_agreement_shuffled)], 0.4, color="white", edgecolor="red", linewidth=3)
    ax.errorbar(x_pos-0.22, [np.nanmean(gc_gc_agreement_true), np.nanmean(ngc_ngc_agreement_true), np.nanmean(gc_ngc_agreement_true), np.nanmean(gc_ngc_network_agreement_true)],
                yerr=[stats.tstd(gc_gc_agreement_true), stats.tstd(ngc_ngc_agreement_true), stats.tstd(gc_ngc_agreement_true), stats.tstd(gc_ngc_network_agreement_true)], capsize=10, color="black", linestyle='', elinewidth=3)
    ax.errorbar(x_pos+0.22, [np.nanmean(gc_gc_agreement_shuffle), np.nanmean(ngc_ngc_agreement_shuffle),np.nanmean(gc_ngc_agreement_shuffle), np.nanmean(gc_ngc_network_agreement_shuffled)],
                yerr=[stats.tstd(gc_gc_agreement_shuffle), stats.tstd(ngc_ngc_agreement_shuffle), stats.tstd(gc_ngc_agreement_shuffle), stats.tstd(gc_ngc_network_agreement_shuffled)], capsize=10, color="black", linestyle='', elinewidth=3)

    for i in range(len(gc_gc_agreement_true)):
        ax.plot([0-0.22, 0+0.22], [gc_gc_agreement_true[i], gc_gc_agreement_shuffle[i]], alpha=0.1, color="black")
    for i in range(len(ngc_ngc_agreement_true)):
        ax.plot([1-0.22, 1+0.22], [ngc_ngc_agreement_true[i], ngc_ngc_agreement_shuffle[i]], alpha=0.005, color="black")
    for i in range(len(gc_ngc_agreement_true)):
        ax.plot([2-0.22, 2+0.22], [gc_ngc_agreement_true[i], gc_ngc_agreement_shuffle[i]], alpha=0.03, color="black")

    for i in range(len(gc_ngc_network_agreement_true)):
        ax.plot([4-0.22, 4+0.22], [gc_ngc_network_agreement_true[i], gc_ngc_network_agreement_shuffled[i]], alpha=0.1, color="black")


    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlabel("", fontsize=20)
    ax.set_xticks([0,1,2,4])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([0, 25, 50, 75, 100])
    ax.set_xticklabels(["","","",""])
    #ax.set_ylabel("Agreement", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/agreements_between_grid_cells_and_non_grid_cells.png',dpi=300)
    plt.close()



    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    colors = ["dimgrey", "lightcoral", "dimgrey", "lightcoral", "dimgrey", "lightcoral", "dimgrey", "lightcoral"]
    data = [gc_gc_agreement_true, gc_gc_agreement_shuffle, ngc_ngc_agreement_true, ngc_ngc_agreement_shuffle,
            gc_ngc_agreement_true, gc_ngc_agreement_shuffle, gc_ngc_network_agreement_true, gc_ngc_network_agreement_shuffled]
    #ax.axhline(y=0, color="black",linestyle="dashed")
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2, 4,5, 7,8, 10, 11], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=2)
    #for i, x in enumerate([1,2]):
    #    ax.scatter((np.ones(len(data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax.set_xticklabels(["", "", "", ""])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([0, 25, 50, 75, 100])
    #ax.set_xticklabels(["","","",""])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 12])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=20)
    #plt.subplots_adjust(hspace = None, wspace = None,  bottom = None, left = 0.4, right =None, top = None)
    plt.savefig(save_path + '/agreements_between_grid_cells_and_non_grid_cells_boxplot.png', dpi=300)
    plt.close()


    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.plot(np.arange(0,11), np.arange(0,11), linestyle="dashed", color="black")
    ax.scatter(gc_gc_agreement_true, gc_gc_agreement_shuffle, marker="o", s=20, facecolors=np.array([0/255, 154/255, 255/ 255, 1]).reshape(1, 4), linewidths=0.5,edgecolors="black")
    plot_regression(ax, gc_gc_agreement_true, gc_gc_agreement_shuffle, c="black", y_text_pos=0.9, x_test_min=0, x_test_max=10)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.tick_params(axis='both', which='major', labelsize=15)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.savefig(save_path + '/agreements_between_grid_cells_and_grid_cells_scatter.png', dpi=300)
    plt.close()

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.plot(np.arange(0,11), np.arange(0,11), linestyle="dashed", color="black")
    ax.scatter(ngc_ngc_agreement_true, ngc_ngc_agreement_shuffle, marker="o", s=20, facecolors=np.array([255/255,127/255,14/255, 1]).reshape(1,4), linewidths=0.5, edgecolors="black", alpha=0.2)
    plot_regression(ax, ngc_ngc_agreement_true, ngc_ngc_agreement_shuffle, c="black", y_text_pos=0.9, x_test_min=0, x_test_max=10)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.tick_params(axis='both', which='major', labelsize=15)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.savefig(save_path + '/agreements_between_non_grid_cells_and_non_grid_cells_scatter.png', dpi=300)
    plt.close()


    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.plot(np.arange(0,11), np.arange(0,11), linestyle="dashed", color="black")
    ax.scatter(gc_ngc_agreement_true, gc_ngc_agreement_shuffle, marker="o", s=20, facecolors="black", linewidths=0.5, edgecolors="black", alpha=0.2)
    plot_regression(ax, gc_ngc_agreement_true, gc_ngc_agreement_shuffle, c="black", y_text_pos=0.9, x_test_min=0, x_test_max=10)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.tick_params(axis='both', which='major', labelsize=15)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.savefig(save_path + '/agreements_between_grid_cells_and_non_grid_cells_scatter.png', dpi=300)
    plt.close()

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.plot(np.arange(0,11), np.arange(0,11), linestyle="dashed", color="black")
    ax.scatter(gc_ngc_network_agreement_true, gc_ngc_network_agreement_shuffled, marker="o", s=20, facecolors="black", linewidths=0.5, edgecolors="black", alpha=0.2)
    plot_regression(ax, gc_ngc_network_agreement_true, gc_ngc_network_agreement_shuffled, c="black", y_text_pos=0.9, x_test_min=0, x_test_max=10)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.tick_params(axis='both', which='major', labelsize=15)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.savefig(save_path + '/agreements_between_network_grid_cells_and_non_grid_cells_scatter.png', dpi=300)
    plt.close()
    return

def plot_firing_rate_position_to_distance_epochs(combined_df, save_path, transition_from="", transition_to=""):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    for cells, cells_str in zip([grid_cells, non_grid_cells], ["grid_cells", "non_grid_cells"]):

        fr_profiles_all_cells = []
        for index, cell_row in cells.iterrows():
            cell_row = cell_row.to_frame().T.reset_index(drop=True)
            rolling_classifiers = np.array(cell_row["rolling:classifier_by_trial_number"].iloc[0])
            firing_rates = np.array(cell_row["avg_z_scored_firing_rates"].iloc[0])

            if np.nansum(firing_rates)>0:
                transition_indices = get_transition_trial_numbers(rolling_classifiers, transition_from=transition_from, transition_to=transition_to)

                last_and_first_5_frs_after_transition = np.zeros((len(transition_indices), 10))
                last_and_first_5_frs_after_transition[:,:] = np.nan

                for i in range(len(transition_indices)):
                    for j in range(len(last_and_first_5_frs_after_transition[0])):
                        speed_index = transition_indices[i]+j-5
                        if (speed_index<len(firing_rates)) and (speed_index>0):
                            last_and_first_5_frs_after_transition[i,j] = firing_rates[speed_index]
                last_and_first_5_frs_after_transition = np.nanmean(last_and_first_5_frs_after_transition, axis=0)
                fr_profiles_all_cells.append(last_and_first_5_frs_after_transition.tolist())
        fr_profiles_all_cells = np.array(fr_profiles_all_cells)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(np.arange(-4,6), np.nanmean(fr_profiles_all_cells, axis=0), color="black", linewidth=3)
        ax.fill_between(np.arange(-4, 6), np.nanmean(fr_profiles_all_cells, axis=0)+stats.sem(fr_profiles_all_cells, axis=0, nan_policy="omit"),
                        np.nanmean(fr_profiles_all_cells, axis=0)-stats.sem(fr_profiles_all_cells, axis=0, nan_policy="omit"), color="black", alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.set_xlim([0, 9])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Trial relative to transition", fontsize=20)
        ax.set_ylabel("Firing rate (Hz)", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='both', labelsize=25)
        # plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/z_scored_firing_rates_between_transition_from_'+transition_from+'_to_'+transition_to+'_'+cells_str+'.png', dpi=300)
        plt.close()
    return

def get_transition_trial_numbers(rolling_classifiers, transition_from="", transition_to=""):
    transition_indices = []
    for i in range(len(rolling_classifiers)-1):
        if (rolling_classifiers[i] == transition_from) and (rolling_classifiers[i+1] == transition_to):
            transition_indices.append(i)
    return np.array(transition_indices)

def plot_speeds_between_position_and_distance_epochs(combined_df, save_path):
    print('plotting lomb classifers proportions...')
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    for cells, cells_str in zip([grid_cells, non_grid_cells], ["grid_cells", "non_grid_cells"]):

        P_speeds = []
        D_speeds = []
        for index, cell_row in cells.iterrows():
            cell_row = cell_row.to_frame().T.reset_index(drop=True)
            rolling_classifiers = np.array(cell_row["rolling:classifier_by_trial_number"].iloc[0])
            running_speeds = np.array(cell_row["avg_z_scored_running_speeds"].iloc[0])

            if np.nansum(running_speeds) > 0:
                P_speeds_cell = running_speeds[rolling_classifiers == "P"]
                D_speeds_cell = running_speeds[rolling_classifiers == "D"]

                P_speeds.append(np.nanmean(P_speeds_cell))
                D_speeds.append(np.nanmean(D_speeds_cell))

        P_speeds = np.array(P_speeds)
        D_speeds = np.array(D_speeds)
        P_not_nan_mask = ~np.isnan(P_speeds)
        D_not_nan_mask = ~np.isnan(D_speeds)
        non_nan_mask = P_not_nan_mask*D_not_nan_mask

        print("comping samples for agreement plot, p = ", str(stats.wilcoxon(x=P_speeds[non_nan_mask],
                                                                             y=D_speeds[non_nan_mask])[1]),
              ", t = ", str(stats.wilcoxon(x=P_speeds[non_nan_mask],
                                           y=D_speeds[non_nan_mask])[0]), ", df = ",
              str(len(P_speeds[non_nan_mask]) - 1))

        _, p = stats.wilcoxon(P_speeds[non_nan_mask],
                              D_speeds[non_nan_mask])

        fig, ax = plt.subplots(figsize=(4, 6))
        for i in range(len(P_speeds[non_nan_mask])):
            ax.scatter([0, 1], [P_speeds[non_nan_mask][i], D_speeds[non_nan_mask][i]], marker="o", color="black")
            ax.plot([0, 1], [P_speeds[non_nan_mask][i], D_speeds[non_nan_mask][i]], color="black")
        ax.set_xticks([0, 1])
        #ax.set_yticks([0, 0.5, 1])
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim(bottom=0)
        plt.xticks(rotation=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels(["P", "D"])
        ax.set_ylabel("Avg trial speed during epoch", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        fig.tight_layout()
        plt.savefig(save_path + '/' + 'Avg_speed_during_position_and_distance_coding_epoch_'+cells_str+'.png', dpi=300)
        plt.close()
    return

def plot_running_speed_position_to_distance_epochs(combined_df, save_path, transition_from="", transition_to=""):
    print('plotting lomb classifers proportions...')
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    for cells, cells_str in zip([grid_cells, non_grid_cells], ["grid_cells", "non_grid_cells"]):

        speed_profiles_all_cells = []
        for index, cell_row in cells.iterrows():
            cell_row = cell_row.to_frame().T.reset_index(drop=True)
            rolling_classifiers = np.array(cell_row["rolling:classifier_by_trial_number"].iloc[0])
            running_speeds = np.array(cell_row["avg_z_scored_running_speeds"].iloc[0])
            if np.nansum(running_speeds) > 0:
                transition_indices = get_transition_trial_numbers(rolling_classifiers, transition_from=transition_from, transition_to=transition_to)

                last_and_first_5_speeds_after_transition = np.zeros((len(transition_indices), 10))
                last_and_first_5_speeds_after_transition[:,:] = np.nan

                for i in range(len(transition_indices)):
                    for j in range(len(last_and_first_5_speeds_after_transition[0])):
                        speed_index = transition_indices[i]+j-5
                        if (speed_index<len(running_speeds)) and (speed_index>0):
                            last_and_first_5_speeds_after_transition[i,j] = running_speeds[speed_index]
                last_and_first_5_speeds_after_transition = np.nanmean(last_and_first_5_speeds_after_transition, axis=0)
                speed_profiles_all_cells.append(last_and_first_5_speeds_after_transition.tolist())
        speed_profiles_all_cells = np.array(speed_profiles_all_cells)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(np.arange(-4,6), np.nanmean(speed_profiles_all_cells, axis=0), color="black", linewidth=3)
        ax.fill_between(np.arange(-4, 6), np.nanmean(speed_profiles_all_cells, axis=0)+stats.sem(speed_profiles_all_cells, axis=0, nan_policy="omit"),
                        np.nanmean(speed_profiles_all_cells, axis=0)-stats.sem(speed_profiles_all_cells, axis=0, nan_policy="omit"), color="black", alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=20)
        #ax.set_xlim([0, 9])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Trial relative to transition", fontsize=20)
        ax.set_ylabel("Avg trial speed (cm/s)", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='both', labelsize=25)
        # plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/z_scored_speeds_between_transition_from_'+transition_from+'_to_'+transition_to+'_'+cells_str+'.png', dpi=300)
        plt.close()
    return

def plot_trial_types_vs_modes(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]

    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    for cells, cells_str in zip([grid_cells, non_grid_cells], ["grid_cells", "non_grid_cells"]):
        fig, ax = plt.subplots(figsize=(4,6))
        groups = ["P", "D", "N"]
        colors_lm = [Settings.allocentric_color,  Settings.egocentric_color, Settings.null_color]
        objects = [0, 1, 2]
        x_pos = np.arange(len(objects))
        for object, x in zip(objects, x_pos):
            bottom=0
            for color, group in zip(colors_lm, groups):
                trial_types = pandas_collumn_to_numpy_array(cells["behaviour_trial_types"])
                rolling_classifiers = pandas_collumn_to_numpy_array(cells["rolling:classifier_by_trial_number"])
                nan_mask = rolling_classifiers == "nan"
                trial_types = trial_types[~nan_mask]
                rolling_classifiers = rolling_classifiers[~nan_mask]

                valid_rolling_classifiers = rolling_classifiers[trial_types == object]

                count = len(valid_rolling_classifiers[valid_rolling_classifiers == group])
                percent = (count/len(valid_rolling_classifiers))*100
                ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
                bottom = bottom + percent

        plt.xticks(x_pos, np.array(objects, dtype=np.str0), fontsize=15)
        plt.ylabel("Percent of trials",  fontsize=25)
        plt.xlim((-0.5, len(objects)-0.5))
        plt.ylim((0,100))
        ax.set_yticks([0,25,50,75,100])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.tight_layout()
        plt.subplots_adjust(left=0.4)
        ax.tick_params(axis='both', which='major', labelsize=30)
        plt.savefig(save_path + '/trial_types_vs_modes_' + cells_str + '.png', dpi=200)
        plt.close()
    return


def add_correlations_of_avg_rate_maps_during_GCnetwork_modes(combined_df):
    correlations = []
    for i in range(len(combined_df)):
        global_grid_code = np.array(combined_df["rolling:grid_code_global"].iloc[i])
        p_mask = global_grid_code=="P"
        d_mask = global_grid_code=="D"
        firing_rates = np.asarray(combined_df["fr_binned_in_space_smoothed"].iloc[i])

        p_firing_rates_avg = np.nanmean(firing_rates[p_mask, :], axis=0)
        d_firing_rates_avg = np.nanmean(firing_rates[d_mask, :], axis=0)
        p_firing_rates_avg[np.isnan(p_firing_rates_avg)] = 0
        d_firing_rates_avg[np.isnan(d_firing_rates_avg)] = 0

        if np.shape(p_firing_rates_avg) == np.shape(d_firing_rates_avg):
            corr = stats.pearsonr(p_firing_rates_avg.flatten(), d_firing_rates_avg.flatten())[0]
            correlations.append(corr)
        else:
            correlations.append(np.nan)
    combined_df["correlations_of_avg_rate_maps_during_GCnetwork_modes"] = correlations
    return combined_df

def plot_correlations_of_avg_rate_maps_during_GCnetwork_modes(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    grid_fr_correlations = []
    non_grid_fr_correlations = []
    for session_id in np.unique(grid_cells["session_id_vr"]):
        session_df = combined_df[combined_df["session_id_vr"] == session_id]
        global_grid_code = np.array(session_df["rolling:grid_code_global"].iloc[0])
        p_mask = global_grid_code=="P"
        d_mask = global_grid_code=="D"
        session_grid_cells = session_df[session_df["classifier"] == "G"]
        session_non_grid_cells = session_df[session_df["classifier"] != "G"]

        grid_cell_firing_rates = pandas_collumn_to_2d_numpy_array(session_grid_cells["fr_binned_in_space_smoothed"])
        non_grid_cell_firing_rates = pandas_collumn_to_2d_numpy_array(session_non_grid_cells["fr_binned_in_space_smoothed"])


        for i in range(len(grid_cell_firing_rates)):
            p_firing_rates_avg = np.nanmean(grid_cell_firing_rates[i, p_mask, :], axis=0)
            d_firing_rates_avg = np.nanmean(grid_cell_firing_rates[i, d_mask, :], axis=0)
            p_firing_rates_avg[np.isnan(p_firing_rates_avg)] = 0
            d_firing_rates_avg[np.isnan(d_firing_rates_avg)] = 0

            if np.shape(p_firing_rates_avg) == np.shape(d_firing_rates_avg):
                corr = stats.pearsonr(p_firing_rates_avg.flatten(), d_firing_rates_avg.flatten())[0]
                grid_fr_correlations.append(corr)

        for i in range(len(non_grid_cell_firing_rates)):
            p_firing_rates_avg = np.nanmean(non_grid_cell_firing_rates[i, p_mask, :], axis=0)
            d_firing_rates_avg = np.nanmean(non_grid_cell_firing_rates[i, d_mask, :], axis=0)
            p_firing_rates_avg[np.isnan(p_firing_rates_avg)] = 0
            d_firing_rates_avg[np.isnan(d_firing_rates_avg)] = 0

            if np.shape(p_firing_rates_avg) == np.shape(d_firing_rates_avg):
                corr = stats.pearsonr(p_firing_rates_avg.flatten(), d_firing_rates_avg.flatten())[0]
                non_grid_fr_correlations.append(corr)

    grid_fr_correlations = np.array(grid_fr_correlations)
    non_grid_fr_correlations = np.array(non_grid_fr_correlations)

    grid_fr_correlations = grid_fr_correlations[~np.isnan(grid_fr_correlations)]
    non_grid_fr_correlations = non_grid_fr_correlations[~np.isnan(non_grid_fr_correlations)]

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.hist(grid_fr_correlations, bins=20, range=(-1,1), color="blue", density=True, alpha=0.5)
    ax.hist(non_grid_fr_correlations, bins=20, range=(-1, 1), color="red", density=True, alpha=0.5)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.xaxis.set_tick_params(labelbottom=False)
    #ax.yaxis.set_tick_params(labelleft=False)
    #ax.set_xticks([0,5,10])
    #ax.set_yticks([0,5,10])
    ax.set_xlim([-1,1])
    #ax.set_ylim([0,10])
    ax.tick_params(axis='both', which='major', labelsize=15)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.savefig(save_path + '/correlation_hist_avg_rate_maps_during_GCnetwork_mode.png', dpi=300)
    plt.close()

    print("comping _correlation of avg rate maps_on_position_and_distance_trials according to grid cells, between grid and non grid cells"+
          " p = ", str(stats.mannwhitneyu(x=grid_fr_correlations, y=non_grid_fr_correlations)[1]),
          " t = ", str(stats.mannwhitneyu(x=grid_fr_correlations, y=non_grid_fr_correlations)[0]),
          ", df1 = ",str(len(grid_fr_correlations)),
          ", df2 = ",str(len(non_grid_fr_correlations)))

    print("comping _correlation of avg rate maps_on_position_and_distance_trials according to grid cells, between grid and non grid cells, GC test vs 0"+
          " p = ", str(stats.wilcoxon(x=grid_fr_correlations)[1]),
          " t = ", str(stats.wilcoxon(x=grid_fr_correlations)[0]),
          ", df = ",str(len(grid_fr_correlations)-1))

    print(
        "comping _correlation of avg rate maps_on_position_and_distance_trials according to grid cells, between grid and non grid cells, NGC test vs 0" +
        " p = ", str(stats.wilcoxon(x=non_grid_fr_correlations)[1]),
        " t = ", str(stats.wilcoxon(x=non_grid_fr_correlations)[0]),
        ", df = ", str(len(non_grid_fr_correlations) - 1))

    return

def plot_proportion_change_in_si_between_PandD_vs_correlations_between_PandD(combined_df, save_path):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    spatial_info_P_G = grid_cells["spatial_information_during_P"]
    spatial_info_D_G = grid_cells["spatial_information_during_D"]
    spatial_info_P_NG = non_grid_cells["spatial_information_during_P"]
    spatial_info_D_NG = non_grid_cells["spatial_information_during_D"]

    rate_map_correlations_G = grid_cells["correlations_of_avg_rate_maps_during_GCnetwork_modes"]
    rate_map_correlations_NG = non_grid_cells["correlations_of_avg_rate_maps_during_GCnetwork_modes"]

    proportion_change_in_SI_G = (spatial_info_P_G - spatial_info_D_G) / spatial_info_P_G
    proportion_change_in_SI_NG = (spatial_info_P_NG - spatial_info_D_NG) / spatial_info_P_NG

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    #ax.plot(np.arange(-4,4), np.arange(-4,4), linestyle="dashed", color="black")
    ax.scatter(proportion_change_in_SI_NG, rate_map_correlations_NG, marker="o", s=40, facecolors=np.array([255/256, 127/256, 14/256, 0.75]).reshape(1,4), linewidths=0.5, edgecolors="black")
    ax.scatter(proportion_change_in_SI_G, rate_map_correlations_G, marker="o", s=40, facecolors=np.array([31/256, 119/256, 180/256, 0.75]).reshape(1,4), linewidths=0.5,  edgecolors="black")
    #plot_regression(ax, spatial_info_P_NG[not_nan_mask_NG], spatial_info_D_NG[not_nan_mask_NG], c="tab:orange", y_text_pos=0.9, x_test_min=0, x_test_max=10)
    #plot_regression(ax, spatial_info_P_G[not_nan_mask_G], spatial_info_D_G[not_nan_mask_G], c="tab:blue", y_text_pos=0.8, x_test_min=0, x_test_max=10)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.xaxis.set_tick_params(labelbottom=False)
    #ax.yaxis.set_tick_params(labelleft=False)
    #ax.set_xticks([0,5,10])
    #ax.set_yticks([0,5,10])
    #ax.set_xlim([0,10])
    ax.set_ylim([-1,1])
    ax.tick_params(axis='both', which='major', labelsize=15)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.savefig(save_path + '/proportion_change_in_si_between_PandD_vs_correlations_between_PandD.png', dpi=300)
    plt.close()
    return




def plot_spatial_information_during_dominant_PandD_grid_modes(combined_df, save_path):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    spatial_info_P_G = grid_cells["spatial_information_during_P"]
    spatial_info_D_G = grid_cells["spatial_information_during_D"]
    spatial_info_P_NG = non_grid_cells["spatial_information_during_P"]
    spatial_info_D_NG = non_grid_cells["spatial_information_during_D"]

    not_nan_mask_G = ~np.isnan(spatial_info_P_G) & ~np.isnan(spatial_info_D_G)
    not_nan_mask_NG = ~np.isnan(spatial_info_P_NG) & ~np.isnan(spatial_info_D_NG)

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.plot(np.arange(0,11), np.arange(0,11), linestyle="dashed", color="black")
    ax.scatter(spatial_info_P_NG[not_nan_mask_NG], spatial_info_D_NG[not_nan_mask_NG], marker="o", s=40, facecolors=np.array([255/255,127/255,14/255, 1]).reshape(1,4), linewidths=0.5, edgecolors="black")
    ax.scatter(spatial_info_P_G[not_nan_mask_G], spatial_info_D_G[not_nan_mask_G], marker="o", s=40, facecolors=np.array([0/255, 154/255, 255/255, 1]).reshape(1,4), linewidths=0.5,  edgecolors="black")
    #plot_regression(ax, spatial_info_P_NG[not_nan_mask_NG], spatial_info_D_NG[not_nan_mask_NG], c=np.array([255/255,127/255,14/255, 1]).reshape(1,4), y_text_pos=0.9, x_test_min=0, x_test_max=10)
    #plot_regression(ax, spatial_info_P_G[not_nan_mask_G], spatial_info_D_G[not_nan_mask_G], c=np.array([0/255, 154/255, 255/255, 1]).reshape(1,4), y_text_pos=0.8, x_test_min=0, x_test_max=10)

    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.xaxis.set_tick_params(labelbottom=False)
    #ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([0,5,10])
    ax.set_yticks([0,5,10])
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.tick_params(axis='both', which='major', labelsize=15)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.savefig(save_path + '/spatial_info_P_vs_spatial_info_D.png', dpi=300)
    plt.close()

    print("comping _spatial_information_on_position_and_distance_trials according to grid cells, NG"+
          " p = ", str(stats.wilcoxon(x=spatial_info_P_NG[not_nan_mask_NG], y=spatial_info_D_NG[not_nan_mask_NG])[1]),
          " t = ", str(stats.wilcoxon(x=spatial_info_P_NG[not_nan_mask_NG], y=spatial_info_D_NG[not_nan_mask_NG])[0]),
          ", df = ",str(len(spatial_info_P_NG[not_nan_mask_NG])-1))

    print("comping _spatial_information_on_position_and_distance_trials according to grid cells, G"+
          " p = ", str(stats.wilcoxon(x=spatial_info_P_G[not_nan_mask_G], y=spatial_info_D_G[not_nan_mask_G])[1]),
          " t = ", str(stats.wilcoxon(x=spatial_info_P_G[not_nan_mask_G], y=spatial_info_D_G[not_nan_mask_G])[0]),
          ", df = ",str(len(spatial_info_P_NG[not_nan_mask_NG])-1))


    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    hist_G, bin_edges = np.histogram((spatial_info_P_G[not_nan_mask_G]-spatial_info_D_G[not_nan_mask_G])/spatial_info_P_G[not_nan_mask_G], density=True, bins=20, range=(-2,2))
    hist_NG, bin_edges = np.histogram((spatial_info_P_NG[not_nan_mask_NG]-spatial_info_D_NG[not_nan_mask_NG])/spatial_info_P_NG[not_nan_mask_NG], density=True, bins=20, range=(-2,2))
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.axvline(x=0, color="black")
    ax.plot(bin_centres, hist_G, color="blue")
    ax.plot(bin_centres, hist_NG, color="red")
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig(save_path + '/proportion_of_spatial_information_during_D_over_P.png', dpi=300)
    plt.close()




    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(2.5, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)

    G_hist, bin_edges = np.histogram((spatial_info_P_G[not_nan_mask_G]-spatial_info_D_G[not_nan_mask_G])/spatial_info_P_G[not_nan_mask_G], bins=10000, density=True, range=(-5,5))
    G_hist, bin_edges = np.histogram(spatial_info_P_G[not_nan_mask_G]-spatial_info_D_G[not_nan_mask_G], bins=10000, density=True, range=(-5,5))
    dx = bin_edges[1] - bin_edges[0]
    G_hist = np.cumsum(G_hist) * dx
    ax.plot(bin_edges[1:], G_hist, linewidth=2, color=np.array([0/255, 154/255, 255/255, 1]).reshape(1,4))

    NG_hist, bin_edges = np.histogram((spatial_info_P_NG[not_nan_mask_NG]-spatial_info_D_NG[not_nan_mask_NG])/spatial_info_P_NG[not_nan_mask_NG], bins=10000, density=True, range=(-5,5))
    NG_hist, bin_edges = np.histogram(spatial_info_P_NG[not_nan_mask_NG]-spatial_info_D_NG[not_nan_mask_NG], bins=10000, density=True, range=(-5,5))
    dx = bin_edges[1] - bin_edges[0]
    NG_hist = np.cumsum(NG_hist) * dx
    ax.plot(bin_edges[1:], NG_hist, linewidth=2, color=np.array([255/255,127/255,14/255, 1]).reshape(1,4))

    ax.tick_params(axis='both', which='major', labelsize=15)
    #ax.set_xticks([0,1,2])
    ax.axvline(x=0, color="black", linestyle="dashed")
    ax.set_yticks([0,1])
    ax.set_xlim([-5,5])
    ax.set_ylim([0,1])
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_path + '/spatial_info_P_vs_spatial_info_D_hist.png', dpi=300)
    plt.close()


    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(2.5, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)

    colors = [np.array([0/255, 154/255, 255/255, 1]).reshape(1,4), np.array([255/255,127/255,14/255, 1]).reshape(1,4)]
    data = [np.asarray((spatial_info_P_G[not_nan_mask_G] - spatial_info_D_G[not_nan_mask_G])*100 / spatial_info_P_G[not_nan_mask_G]),
            np.asarray((spatial_info_P_NG[not_nan_mask_NG]-spatial_info_D_NG[not_nan_mask_NG])*100 / spatial_info_P_NG[not_nan_mask_NG])]
    ax.axhline(y=0, color="black",linestyle="dashed")
    boxprops = dict(linewidth=2, color='k')
    medianprops = dict(linewidth=2, color='k')
    capprops = dict(linewidth=2, color='k')
    whiskerprops = dict(linewidth=2, color='k')
    box = ax.boxplot(data, positions=[1,3], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=2)
    #for i, x in enumerate([1,2]):
    #    ax.scatter((np.ones(len(data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,3])
    ax.set_xticklabels(["G", "NG"])
    ax.set_yticks([-100, 0, 100])
    ax.set_ylim([-150, 150])
    ax.set_xlim([0, 4])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=15)
    plt.subplots_adjust(hspace = None, wspace = None,  bottom = None, left = 0.4, right =None, top = None)
    plt.savefig(save_path + '/spatial_info_P_vs_spatial_info_D_boxplot.png', dpi=300)
    plt.close()

    print("comparing % change in si position_vs_distance according to grid cells, G vs NG"+
          " p = ", str(stats.mannwhitneyu(x=data[0], y=data[1])[1]),
          " t = ", str(stats.mannwhitneyu(x=data[0], y=data[1])[0]),
          ", df1 = ",str(len(data[0])),
          ", df2 = ",str(len(data[0])))

    print("comparing % change in si position_vs_distance according to grid cells, G vs null"+
          " p = ", str(stats.wilcoxon(x=data[0])[1]),
          " t = ", str(stats.wilcoxon(x=data[0])[0]),
          ", df = ",str(len(data[0])-1))

    print("comparing % change in si position_vs_distance according to grid cells, NG vs null"+
          " p = ", str(stats.wilcoxon(x=data[1])[1]),
          " t = ", str(stats.wilcoxon(x=data[1])[0]),
          ", df = ",str(len(data[1])-1))
    return

def plot_field_distributions(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    combined_df = combined_df[combined_df["classifier"] == "G"]

    field_dist = []
    for i in range(len(combined_df)):
        field_locations = np.array(combined_df["field_locations"].iloc[i])
        hist, bin_edges = np.histogram(field_locations, bins=200, range=(0,200), density=True)
        field_dist.append(hist)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    field_dist = np.array(field_dist)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(bin_centres, np.nanmean(field_dist, axis=0), color="black", linewidth=3)
    ax.fill_between(bin_centres, np.nanmean(field_dist, axis=0) - stats.sem(field_dist, axis=0, nan_policy="omit"),
                                 np.nanmean(field_dist, axis=0) + stats.sem(field_dist, axis=0, nan_policy="omit"), color="black", alpha=0.3)
    style_track_plot(ax, 200)
    plt.xlim(0, 200)
    tick_spacing = 100
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    Edmond.plot_utility2.style_vr_plot(ax)
    #ax.set_ylim([-0.05, 0.15])
    #ax.set_yticks([0, 0.1])
    plt.locator_params(axis='y', nbins=3)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.savefig(save_path + '/field_dist.png',dpi=300)
    plt.close()
    return

def add_network_stability(cells_df):
    proportion_network_encoding_position = []
    proportion_network_encoding_distance = []
    proportion_network_encoding_null = []
    for index, cell_row in cells_df.iterrows():
        cell_row = cell_row.to_frame().T.reset_index(drop=True)
        network_code = cell_row["rolling:code_global"].iloc[0]
        network_code = network_code[network_code != ""]

        proportion_encoding_position = len(network_code[network_code == "P"]) / len(network_code)
        proportion_encoding_distance = len(network_code[network_code == "D"]) / len(network_code)
        proportion_encoding_null = len(network_code[network_code == "N"]) / len(network_code)

        proportion_network_encoding_position.append(proportion_encoding_position)
        proportion_network_encoding_distance.append(proportion_encoding_distance)
        proportion_network_encoding_null.append(proportion_encoding_null)

    cells_df["proportion_network_encoding_position"] = proportion_network_encoding_position
    cells_df["proportion_network_encoding_distance"] = proportion_network_encoding_distance
    cells_df["proportion_network_encoding_null"] = proportion_network_encoding_null
    return cells_df

def main():
    print('-------------------------------------------------------------')
    print("hello")

    odds_ratio_data = pd.read_csv("/mnt/datastore/Harry/Vr_grid_cells/odds_ratio_glmer_model.csv")
    plot_odds_ratio(odds_ratio_data, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/fast_hit_analysis")

    combined_df = pd.DataFrame()
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort6.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort7.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")], ignore_index=True)

    # remove artefacts, low firing rates in the open field, sessions with non 200cm track lengths and sessions with less than 10 trials
    combined_df = combined_df[combined_df["snippet_peak_to_trough"] < 500] # uV
    combined_df = combined_df[combined_df["mean_firing_rate_of"] > 0.2] # Hz
    combined_df = combined_df[combined_df["track_length"] == 200]
    combined_df = combined_df[combined_df["n_trials"] >= 10]
    combined_df = add_lomb_classifier(combined_df,suffix="")
    combined_df = add_peak_width(combined_df)
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]

    # remove mice without any grid cells
    combined_df = combined_df[combined_df["mouse"] != "M2"]
    combined_df = combined_df[combined_df["mouse"] != "M4"]
    combined_df = combined_df[combined_df["mouse"] != "M15"]

    # count how many clusters are in each recording
    for recording in np.unique(combined_df["session_id_vr"]):
        print(recording, " has ", str(len(combined_df[combined_df["session_id_vr"] == recording])), " clusters")

    grid_cells = combined_df[combined_df["classifier"] == "G"]
    print("There is ", len(np.unique(combined_df["mouse"])), " mice in the dataset")
    print(np.unique(combined_df["mouse"]))
    print("There is ", len(combined_df), " clusters in the dataset")
    print("There is ", len(np.unique(combined_df["session_id"])), " sessions in this dataset")

    print("There is ", len(grid_cells), " grid clusters in the dataset")
    print("There is ", len(np.unique(grid_cells["session_id"])), " sessions in this dataset with grid cells")

    combined_df = add_network_stability(combined_df)
    all_networks_df = drop_duplicate_sessions(combined_df)
    stable_networks_df = all_networks_df[((all_networks_df["proportion_network_encoding_position"] >= 0.85) |
                                          (all_networks_df["proportion_network_encoding_distance"] >= 0.85))]
    unstable_networks_df = all_networks_df[((all_networks_df["proportion_network_encoding_position"] < 0.85) &
                                          (all_networks_df["proportion_network_encoding_distance"] < 0.85))]

    #================ for R =============================================================#
    stable_networks_for_R_shortform = generate_short_form_for_R_by_network_session(stable_networks_df)
    stable_networks_for_R_longform = generate_long_form_for_R_by_network_session(stable_networks_df)
    stable_networks_for_R_shortform.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/stable_networks_for_R_shortform.pkl")
    stable_networks_for_R_longform.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/stable_networks_for_R_longform.pkl")
    unstable_networks_for_R_shortform = generate_short_form_for_R_by_network_epochs(unstable_networks_df)
    unstable_networks_for_R_longform = generate_long_form_for_R_by_network_epochs(unstable_networks_df)
    unstable_networks_for_R_shortform.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/unstable_networks_for_R_shortform.pkl")
    unstable_networks_for_R_longform.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/unstable_networks_for_R_longform.pkl")
    all_networks_for_R_shortform = generate_short_form_for_R_by_network_epochs(all_networks_df)
    all_networks_for_R_longform = generate_long_form_for_R_by_network_epochs(all_networks_df)
    all_networks_for_R_shortform.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_networks_for_R_shortform.pkl")
    all_networks_for_R_longform.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_networks_for_R_longform.pkl")

    all_cells_for_R_longform = generate_long_form_for_R(all_networks_df)
    all_cells_for_R_longform.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_cells_for_R_longform.pkl")
    all_cells_for_R_longform_by_session = generate_long_form_for_R_by_session(stable_networks_df)
    all_cells_for_R_longform_by_session.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/all_cells_for_R_longform_by_session.pkl")
    #=====================================================================================#

    '''
    # ===================== for R ========================================================= #
    combined_for_R = combined_df.copy()
    combined_for_R = combined_for_R[combined_for_R["Lomb_classifier_"] != "Unclassified"]
    #combined_for_R = combined_for_R[combined_for_R["classifier"] == "G"]
    combined_for_R = generate_long_form_for_R(combined_for_R)
    combined_for_R.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_df_for_R.pkl")

    combined_for_R = combined_df.copy()
    combined_for_R = combined_for_R[combined_for_R["Lomb_classifier_"] != "Unclassified"]
    #combined_for_R = combined_for_R[combined_for_R["classifier"] == "G"]
    combined_for_R = generate_short_form_for_R_by_session(combined_for_R)
    combined_for_R.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_df_for_R_shortform_by_session.pkl")

    combined_for_R = combined_df.copy()
    combined_for_R = combined_for_R[combined_for_R["Lomb_classifier_"] != "Unclassified"]
    #combined_for_R = combined_for_R[combined_for_R["classifier"] == "G"]
    combined_for_R = generate_short_form_for_R_by_trial(combined_for_R)
    combined_for_R.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_df_for_R_shortform_by_trial.pkl")
   
    
    combined_for_R["p_b_hit"] = get_percentage_from_rolling_classification(combined_for_R, code="P", tt=0)
    combined_for_R["d_b_hit"] = get_percentage_from_rolling_classification(combined_for_R, code="D", tt=0)
    combined_for_R["n_b_hit"] = get_percentage_from_rolling_classification(combined_for_R, code="N", tt=0)
    combined_for_R["p_nb_hit"] = get_percentage_from_rolling_classification(combined_for_R, code="P", tt=1)
    combined_for_R["d_nb_hit"] = get_percentage_from_rolling_classification(combined_for_R, code="D", tt=1)
    combined_for_R["n_nb_hit"] = get_percentage_from_rolling_classification(combined_for_R, code="N", tt=1)
    combined_for_R["p_p_hit"] = get_percentage_from_rolling_classification(combined_for_R, code="P", tt=2)
    combined_for_R["d_p_hit"] = get_percentage_from_rolling_classification(combined_for_R, code="D", tt=2)
    combined_for_R["n_p_hit"] = get_percentage_from_rolling_classification(combined_for_R, code="N", tt=2)
    #output short form for R
    combined_for_R.drop(columns=["miss_spatial_periodogram_tt2","miss_spatial_periodogram_tt1", "miss_spatial_periodogram_tt0",
                              "try_spatial_periodogram_tt2","try_spatial_periodogram_tt1", "try_spatial_periodogram_tt0",
                              "hit_spatial_periodogram_tt0","hit_spatial_periodogram_tt1", "hit_spatial_periodogram_tt2",
                              "MOVING_LOMB_avg_power", "n_pi_trials_by_hmt","fields_jitter_hmt_by_trial_type",
                              "fields_jitter_hmt_by_trial_type_pre_rz","fields_jitter_hmt_by_trial_type_post_rz",
                              "percentage_hits","stop_locations","stop_trial_numbers","behaviour_trial_numbers",
                              "behaviour_hit_try_miss","behaviour_trial_types","rolling:block_lengths_for_encoder",
                              "rolling:block_lengths_for_encoder_shuffled","rolling:encoding_position_by_trial_category",
                              "rolling:encoding_position_by_trial_category","rolling:encoding_position_by_trial_category",
                              "rolling:percentage_trials_encoding_position","rolling:percentage_trials_encoding_distance",
                              "rolling:percentage_trials_encoding_null","rolling:block_lengths_shuffled","rolling:block_lengths",
                              "rolling:block_lengths_small_window","rolling:block_lengths_shuffled_small_window","rolling:classifier_by_trial_number",
                              "MOVING_LOMB_all_powers", "MOVING_LOMB_all_centre_trials",'fields_com', 'fields_com_trial_number', 'fields_com_trial_type',
                              'minimum_distance_to_field_in_next_trial', 'fields_com_next_trial_type'])
    combined_for_R = combined_for_R[['cluster_id', 'primary_channel', 'session_id', 'recording_length_sampling_points', 'tetrode', "mouse","classifier",
                                     'number_of_spikes', 'mean_firing_rate_vr', 'recording_length_seconds', 'isolation', "Lomb_classifier_",
                                     'p_b_hit', 'd_b_hit', 'n_b_hit', 'p_nb_hit', 'd_nb_hit', 'n_nb_hit', 'p_p_hit', 'd_p_hit', 'n_p_hit']]
    combined_for_R.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_df_for_R.pkl")
    # ===================== for R ========================================================= #
    '''


    save_path = "/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/fast_hit_analysis"
    #plot_agreement_between_grid_and_non_grids(combined_df, save_path=save_path)

    #plot_correlations_of_avg_rate_maps_during_GCnetwork_modes(combined_df,save_path=save_path)
    #plot_spatial_information_during_dominant_PandD_grid_modes(combined_df, save_path=save_path)
    #combined_df = add_correlations_of_avg_rate_maps_during_GCnetwork_modes(combined_df)
    #plot_proportion_change_in_si_between_PandD_vs_correlations_between_PandD(combined_df, save_path=save_path)
    #plot_trial_types_vs_modes(combined_df,save_path=save_path)
    #plot_spatial_information_on_position_and_distance_trials(combined_df, save_path=save_path, fig_size=(3.5,6))
    #plot_mean_firing_rates_on_position_and_distance_trials(combined_df, save_path=save_path, fig_size=(3.5,6))



    #plot_firing_rate_position_to_distance_epochs(combined_df, save_path=save_path, transition_from="P", transition_to="D")
    #plot_firing_rate_position_to_distance_epochs(combined_df, save_path=save_path, transition_from="D", transition_to="P")
    #plot_running_speed_position_to_distance_epochs(combined_df, save_path=save_path, transition_from="P", transition_to="D")
    #plot_running_speed_position_to_distance_epochs(combined_df, save_path=save_path, transition_from="D", transition_to="P")
    #plot_speeds_between_position_and_distance_epochs(combined_df, save_path=save_path)
    #read_df(combined_df)
    # Figure 2 population level plots
    fig_size = (3.5,6)

    #plot_comparison_grids_vs_non_grids_during_position_and_distance_epochs(combined_df, save_path=save_path, fig_size=fig_size)
    #plot_average_firing_rate_map_during_position_and_distance_epochs(combined_df, save_path=save_path, fig_size=(6,4), z_scored=True)

    # compare

    #plot_lomb_classifiers_proportions(combined_df, suffix="", save_path=save_path)
    #plot_lomb_classifiers_vs_shuffle(combined_df, suffix="", save_path=save_path)
    #plot_lomb_classifiers_vs_shuffle_non_grid_cells(combined_df, suffix="", save_path=save_path)
    #plot_lomb_classifiers_vs_shuffle_grid_cells(combined_df, suffix="", save_path=save_path)

    #plot_lomb_classifier_powers_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size)
    #plot_lomb_classifier_mfr_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size)
    #plot_lomb_classifier_spatinfo_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size, score="spatial_information_score_Isec_vr")
    #plot_lomb_classifier_spatinfo_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size, score="spatial_information_score_Ispike_vr")
    #plot_lomb_classifier_peak_width_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size)
    #plot_lomb_classifier_mfr_vs_groups_vs_open_field(combined_df, suffix="", save_path=save_path, fig_size=fig_size)

    # Figure 3 plots remapping
    #plot_rolling_lomb_block_sizes(combined_df, save_path=save_path)
    #plot_rolling_lomb_block_lengths_vs_shuffled(combined_df, save_path=save_path)


    #plot_percentage_encoding_by_trial_category_each_mouse(combined_df, save_path=save_path)


    #plot_stop_histogram_for_remapped_encoding_grid_cells_by_mouse(combined_df, save_path=save_path)
    #plot_percentage_hits_for_remapped_encoding_grid_cells_mouse_level(combined_df, save_path=save_path)
    #plot_stop_variability_remapped(combined_df, save_path=save_path, account_for="animal")
    #plot_number_of_trials_remapped(combined_df, save_path=save_path, account_for="animal")

    # stable cell analysis
    plot_stop_histogram_for_stable_sessions(stable_networks_df, save_path=save_path)
    plot_percentage_hits_for_stable_sessions(stable_networks_df, save_path=save_path)
    # remap cell analysis
    #plot_field_distributions(combined_df, save_path=save_path)
    #plot_position_coding_predictions_of_first_stop_locations(combined_df,save_path=save_path)
    plot_percentage_encoding_by_trial_category_each_mouse_from_global(all_networks_df, save_path=save_path)
    #plot_percentage_encoding_for_trials(combined_df, save_path=save_path)

    plot_stop_histogram_for_trials(unstable_networks_df, save_path=save_path)
    plot_percentage_hits_for_trials(unstable_networks_df, save_path=save_path)

    #plot_stop_histogram_for_stable_encoding_grid_cells(combined_df, save_path=save_path)
    #plot_percentage_hits_for_stable_encoding_grid_cells(combined_df, save_path=save_path)
    #plot_percentage_hits_for_stable_encoding_non_grid_cells(combined_df, save_path=save_path)

    # plot_p_hit_for_stable_encoding_grid_cells(combined_df, save_path=save_path)
    #plot_stop_variability_stable(combined_df, save_path=save_path)
    #plot_number_of_trials_stable(combined_df, save_path=save_path)
    #plot_stop_peak_stop_location_and_height_stable(combined_df, save_path=save_path)
    #plot_stop_peak_stop_location_and_height_remapped_probe(combined_df, save_path=save_path)
    #plot_percentage_hits_for_remapped_encoding_grid_cells_probe(combined_df, save_path=save_path)

    # remap analysis
    #plot_p_hit_for_remapped_encoding_grid_cells(combined_df, save_path=save_path)
    plot_percentage_hits_for_remapped_encoding_grid_cells(combined_df, save_path=save_path)
    plot_percentage_hits_for_remapped_encoding_non_grid_cells(combined_df, save_path=save_path)
    plot_stop_histogram_for_remapped_encoding_grid_cells(combined_df, save_path=save_path)
    #plot_stop_variability_remapped(combined_df, save_path=save_path, account_for="cell")
    #plot_number_of_trials_remapped(combined_df, save_path=save_path, account_for="cell")
    #plot_stop_peak_stop_location_and_height_remapped(combined_df, save_path=save_path)

    plot_percentage_encoding_by_trial_category(combined_df, save_path=save_path)
    plot_percentage_encoding_by_trial_category_non_grid_cells(combined_df, save_path=save_path)
    ###
    #for mouse_id in np.unique(combined_df["mouse"]):
    #    print("analysing by mouse ", mouse_id)
    #    try:
    #        plot_spatial_information_on_position_and_distance_trials(combined_df[combined_df["mouse"] == mouse_id], save_path=save_path, fig_size=fig_size, suffix="_" + mouse_id, unlock_y=True)
    #    except Exception as ex:
    #        print('This is what Python says happened:'); print(ex)
    #    try:
    #        plot_percentage_encoding_by_trial_category(combined_df[combined_df["mouse"] == mouse_id], save_path=save_path, suffix="_" + mouse_id)
    #    except Exception as ex:
    #        print('This is what Python says happened:'); print(ex)

    #plot_lomb_classifiers_proportions_hmt(combined_df, save_path=save_path)
    #plot_spatial_information_on_hmt_trials(combined_df, save_path=save_path, fig_size=fig_size)

    # Figure 4 plots behaviours
    print("===================Figure 4==================")
    #plot_ROC(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers")

    #for mouse_id in np.unique(combined_df["mouse"]):
    #    print("analysing by mouse ", mouse_id)
    #    try:
    #        plot_spatial_information_on_position_and_distance_trials(combined_df[combined_df["mouse"] == mouse_id], save_path=save_path, fig_size=fig_size, suffix="_" + mouse_id)
    #    except Exception as ex:
    #        print('This is what Python says happened:'); print(ex)
    #    try:
    #        plot_percentage_hits_for_remapped_encoding_grid_cells(combined_df[combined_df["mouse"]==mouse_id], save_path=save_path, suffix="_"+mouse_id)
    #    except Exception as ex:
    #        print('This is what Python says happened:'); print(ex)
    #    try:
    #        plot_stop_peak_stop_location_and_height_remapped(combined_df[combined_df["mouse"]==mouse_id], save_path=save_path, suffix="_"+mouse_id)
    #    except Exception as ex:
    #        print('This is what Python says happened:'); print(ex)
    #    try:
    #        plot_stop_histogram_for_remapped_encoding_grid_cells(combined_df[combined_df["mouse"]==mouse_id], save_path=save_path, suffix="_"+mouse_id, lock_y_axis=False)
    #    except Exception as ex:
    #        print('This is what Python says happened:'); print(ex)
    #    try:
    #        plot_stop_peak_stop_location_and_height_remapped_probe(combined_df[combined_df["mouse"]==mouse_id], save_path=save_path, suffix="_"+mouse_id)
    #    except Exception as ex:
    #        print('This is what Python says happened:'); print(ex)
    #    try:
    #        plot_stop_variability_remapped(combined_df[combined_df["mouse"]==mouse_id], save_path=save_path, suffix="_"+mouse_id)
    #    except Exception as ex:
    #        print('This is what Python says happened:'); print(ex)
    #    try:
    #        plot_number_of_trials_remapped(combined_df[combined_df["mouse"]==mouse_id], save_path=save_path, suffix="_"+mouse_id)
    #    except Exception as ex:
    #        print('This is what Python says happened:'); print(ex)

    print("hello")


if __name__ == '__main__':
    main()