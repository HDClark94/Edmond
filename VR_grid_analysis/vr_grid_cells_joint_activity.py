import numpy as np
import pandas as pd
from scipy.signal import correlate
import Edmond.VR_grid_analysis.analysis_settings as Settings
import PostSorting.parameters
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from matplotlib.markers import TICKDOWN
from Edmond.VR_grid_analysis.remake_position_data import syncronise_position_data
from Edmond.VR_grid_analysis.vr_grid_stability_plots import add_hit_miss_try3, add_avg_track_speed, get_avg_correlation, \
    get_reconstructed_trial_signal, plot_firing_rate_maps_per_trial_by_hmt_aligned, plot_firing_rate_maps_per_trial_by_hmt_aligned_other_neuron, plot_firing_rate_maps_per_trial_aligned_other_neuron, get_shifts
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
plt.rc('axes', linewidth=3)

def get_p_text(p, ns=True):

    if p is not None:
        if np.isnan(p):
            return " "
        if p<0.00001:
            return 'p < 1e'+('{:.1e}'.format(p)).split("e")[-1]
        if p<0.0001:
            return "****"
        elif p<0.001:
            return "***"
        elif p<0.01:
            return "**"
        elif p<0.05:
            return "*"
        elif ns:
            return "ns"
        else:
            return " "
    else:
        return " "

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

def plot_joint_jitter_correlations(spike_data, of_spike_data, processed_position_data, position_data, output_path, track_length, matched_recording_df):

    spike_data = add_lomb_classifier(spike_data)
    print('plotting joint cell correlations...')
    save_path = output_path + '/Figures/joint_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    spike_data = pd.merge(spike_data, of_spike_data[["cluster_id", "grid_cell"]], on="cluster_id")
    spike_data = spike_data.sort_values(by=["grid_cell"], ascending=False)
    cluster_ids = pandas_collumn_to_numpy_array(spike_data["cluster_id"])
    ids = cluster_ids
    n_shuffles = 1
    shuffled = np.ones((n_shuffles+1), dtype=bool); shuffled[0] = False

    for m, hmt in enumerate(["hit", "try", "miss"]):
        for n, tt in enumerate([0,1]):
            hmt_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
            subset_processed_position_data = hmt_processed_position_data[hmt_processed_position_data["trial_type"] == tt]
            subset_trial_numbers = pandas_collumn_to_numpy_array(subset_processed_position_data["trial_number"])
            if len(subset_trial_numbers)>0:

                cross_correlations = np.zeros((len(shuffled), len(ids), len(ids)))
                for si, shuffle in enumerate(shuffled):
                    for i in range(len(cross_correlations[0])):
                        for j in range(len(cross_correlations[0][0])):
                            cluster_j_df = spike_data[spike_data["cluster_id"] == ids[j]]
                            cluster_i_df = spike_data[spike_data["cluster_id"] == ids[i]]
                            shifts_i = get_shifts(cluster_i_df, hmt=hmt, tt=tt)
                            if shuffle:
                                #np.random.seed(j*i)
                                np.random.shuffle(shifts_i)
                                #shifts_i = np.random.randint(-100, 100, size=len(shifts_i))

                            cluster_firing_maps_j = np.array(cluster_j_df["fr_binned_in_space_smoothed"].iloc[0])
                            where_are_NaNs2 = np.isnan(cluster_firing_maps_j)
                            cluster_firing_maps_j[where_are_NaNs2] = 0
                            cluster_firing_maps_j = min_max_normalize(cluster_firing_maps_j)
                            hmt_cluster_firing_maps_j = cluster_firing_maps_j[subset_trial_numbers-1]
                            avg_correlation = get_avg_correlation(hmt_cluster_firing_maps_j)

                            # reconstruct avg spatial correlation using the newly aligned trials
                            reconstructed_signal = []
                            for ti, tn in enumerate(subset_trial_numbers):
                                reconstructed_trial = get_reconstructed_trial_signal(shifts_i[ti], hmt_cluster_firing_maps_j[ti].flatten(),
                                                                                     min_shift=min(shifts_i), max_shift=max(shifts_i))
                                reconstructed_signal.append(reconstructed_trial.tolist())
                            reconstructed_signal = np.array(reconstructed_signal)
                            reconstructed_signal_corr = get_avg_correlation(reconstructed_signal)

                            cross_correlations[si, i, j] = reconstructed_signal_corr - avg_correlation

                fig, ax = plt.subplots()
                cross_correlation = cross_correlations[0]-np.nanmean(cross_correlations[1:], axis=0)
                np.fill_diagonal(cross_correlation, np.nan)

                cross_correlation_rank = np.zeros((len(cross_correlation), len(cross_correlation[0])))
                cross_correlation_values = cross_correlation.flatten()
                cross_correlation_values = cross_correlation_values[~np.isnan(cross_correlation_values)]
                for ii in range(len(cross_correlation)):
                    for jj in range(len(cross_correlation[0])):
                        cross_correlation_rank[ii, jj] = stats.percentileofscore(cross_correlation_values, cross_correlation[ii,jj])

                im= ax.imshow(cross_correlation_rank, cmap="Purples", vmin=50, vmax=100)
                ax.set_xticks(np.arange(len(ids)))
                ax.set_yticks(np.arange(len(ids)))
                ax.set_yticklabels(ids)
                ax.set_xticklabels(ids)
                ax.set_ylabel("Cluster ID: Reference", fontsize=20)
                ax.set_xlabel("Cluster ID: Shifted", fontsize=20)
                ax.tick_params(axis='both', which='major', labelsize=8)
                fig.tight_layout()
                fig.colorbar(im, ax=ax)
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + 'joint_alignment_cross_correlations_'+hmt+'_'+str(tt)+'.png', dpi=300)
                plt.close()
                print(hmt+" "+ str(np.nanmean(cross_correlations[0])))


                fig, ax = plt.subplots(figsize=(6,6))
                ax.axhline(y=0, linestyle="dashed", color="gray")
                ax.axvline(x=0, linestyle="dashed", color="gray")
                ax.plot(np.arange(-2, 2), np.arange(-2, 2), linestyle="solid", color="black")

                grid_pairs_vs_shuffle = []
                grid_non_grid_pairs_vs_shuffle = []
                non_grid_non_grid_pairs_vs_shuffle = []
                tetrode_level = []
                for i in range(len(cross_correlations[0])):
                    for j in range(len(cross_correlations[0][0])):
                        cluster_j_classifier = spike_data[spike_data["cluster_id"] == ids[j]]["grid_cell"].iloc[0]
                        cluster_i_classifier = spike_data[spike_data["cluster_id"] == ids[i]]["grid_cell"].iloc[0]
                        tetrode_j = spike_data[spike_data["cluster_id"] == ids[j]]["tetrode"].iloc[0]
                        tetrode_i = spike_data[spike_data["cluster_id"] == ids[i]]["tetrode"].iloc[0]

                        if tetrode_i == tetrode_j:
                            marker="x"
                            tetrode_level.append("same")
                        else:
                            marker="o"
                            tetrode_level.append("different")

                        if ((cluster_j_classifier == True) and (cluster_i_classifier == True)):
                            ax.scatter(cross_correlations[0, i, j], np.nanmean(cross_correlations[1:], axis=0)[i, j], marker=marker, color="red", zorder=10)
                            grid_pairs_vs_shuffle.append(cross_correlations[0, i, j]-np.nanmean(cross_correlations[1:]))
                        elif ((cluster_j_classifier == True) or (cluster_i_classifier == True)):
                            ax.scatter(cross_correlations[0, i, j], np.nanmean(cross_correlations[1:], axis=0)[i, j], marker=marker, color="blue", alpha=0.3, zorder=5)
                            grid_non_grid_pairs_vs_shuffle.append(cross_correlations[0, i, j]-np.nanmean(cross_correlations[1:]))
                        else:
                            ax.scatter(cross_correlations[0, i, j], np.nanmean(cross_correlations[1:], axis=0)[i, j], marker=marker, color="black", alpha=0.3, zorder=1)
                            non_grid_non_grid_pairs_vs_shuffle.append(cross_correlations[0, i, j]-np.nanmean(cross_correlations[1:]))

                ax.set_ylabel("Change in R (Shuffle)", fontsize=20)
                ax.set_xlabel("Change in R (Real)", fontsize=20)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylim([-0.5, 0.5])
                ax.set_xlim([-0.5, 0.5])
                ax.tick_params(axis='both', which='major', labelsize=15)
                fig.tight_layout()
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + 'joint_alignment_correlations_shuffle_vs_real_'+hmt+'_'+str(tt)+'png', dpi=300)
                plt.close()

                tetrode_level = np.array(tetrode_level)
                grid_pairs_vs_shuffle = np.array(grid_pairs_vs_shuffle); gg_p = stats.wilcoxon(grid_pairs_vs_shuffle)[1]; gg_p= get_p_text(gg_p, ns=False)
                grid_non_grid_pairs_vs_shuffle = np.array(grid_non_grid_pairs_vs_shuffle); gng_p = stats.wilcoxon(grid_non_grid_pairs_vs_shuffle)[1]; gng_p= get_p_text(gng_p, ns=False)
                non_grid_non_grid_pairs_vs_shuffle = np.array(non_grid_non_grid_pairs_vs_shuffle); ngng_p = stats.wilcoxon(non_grid_non_grid_pairs_vs_shuffle)[1]; ngng_p= get_p_text(ngng_p, ns=False)
                data = [grid_pairs_vs_shuffle, grid_non_grid_pairs_vs_shuffle, non_grid_non_grid_pairs_vs_shuffle];

                fig, ax = plt.subplots(figsize=(6,6))
                vp = ax.violinplot(data, [2, 4, 6], widths=2,
                                   showmeans=False, showmedians=True, showextrema=False)
                height =ax.get_ylim()[1]
                ax.text(2, height, gg_p, ha="center", fontsize=15)
                ax.text(4, height, gng_p, ha="center", fontsize=15)
                ax.text(6, height, ngng_p, ha="center", fontsize=15)
                ax.axhline(y=0, linestyle="dashed", color="black")
                ax.set_xticks([2,4,6])
                ax.set_yticks([-0.2, 0, 0.2, 0.4])
                ax.set_ylim([-0.4, 0.6])
                ax.set_xticklabels(["G-G", "G-NG", "NG-NG"])
                ax.set_ylabel("Change in spatial\ncorrelation vs shuffle", fontsize=20, labelpad=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.xaxis.set_tick_params(length=0)
                ax.tick_params(axis='both', which='major', labelsize=25)
                fig.tight_layout()
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + 'joint_alignment_correlations_violin_shuffle_vs_real_'+hmt+'_'+str(tt)+'png', dpi=300)
                plt.close()

                session_df = pd.DataFrame()
                session_df["session_id"] = [spike_data.session_id.iloc[0]]
                session_df["trial_type"] = [tt]
                session_df["hit_miss_try"] = [hmt]
                session_df["tetrode_level"] = [tetrode_level]
                session_df["grid_pairs_vs_shuffle"] = [grid_pairs_vs_shuffle]
                session_df["grid_non_grid_pairs_vs_shuffle"] = [grid_non_grid_pairs_vs_shuffle]
                session_df["non_grid_non_grid_pairs_vs_shuffle"] = [non_grid_non_grid_pairs_vs_shuffle]
                # save the session_df and append to the global matched recording df
                matched_recording_df = pd.concat([matched_recording_df, session_df], ignore_index=True)

    return matched_recording_df


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
            step = Settings.frequency_step
            frequency = Settings.frequency
            sliding_window_size=track_length*Settings.window_length_in_laps

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


def add_percentage_for_lomb_classes(combined_df):
    # this function calculates the percentage of concurrently recorded cells

    NG_props_D = []
    NG_props_P = []
    NG_props_N = []
    G_props_D = []
    G_props_P = []
    G_props_N = []
    for index, cluster_row in combined_df.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)
        session_id = cluster_row["session_id"].iloc[0]
        cluster_id = cluster_row["cluster_id"].iloc[0]
        NG_cells_in_same_recordings = combined_df[(combined_df["session_id"] == session_id) & (combined_df["classifier"] != "G") & (combined_df["cluster_id"] != cluster_id)]
        G_cells_in_same_recordings = combined_df[(combined_df["session_id"] == session_id) & (combined_df["classifier"] == "G") & (combined_df["cluster_id"] != cluster_id)]
        NG_lomb_classes = NG_cells_in_same_recordings["Lomb_classifier_"]
        G_lomb_classes = G_cells_in_same_recordings["Lomb_classifier_"]

        if len(NG_lomb_classes)>0:
            NG_proportion_D = np.sum(NG_lomb_classes=="Distance")/len(NG_lomb_classes)
            NG_proportion_P = np.sum(NG_lomb_classes=="Position")/len(NG_lomb_classes)
            NG_proportion_N = np.sum(NG_lomb_classes=="Null")/len(NG_lomb_classes)
        else:
            NG_proportion_D = np.nan
            NG_proportion_P = np.nan
            NG_proportion_N = np.nan

        if len(G_lomb_classes)>0:
            G_proportion_D = np.sum(G_lomb_classes=="Distance")/len(G_lomb_classes)
            G_proportion_P = np.sum(G_lomb_classes=="Position")/len(G_lomb_classes)
            G_proportion_N = np.sum(G_lomb_classes=="Null")/len(G_lomb_classes)
        else:
            G_proportion_D = np.nan
            G_proportion_P = np.nan
            G_proportion_N = np.nan

        NG_props_D.append(NG_proportion_D)
        NG_props_P.append(NG_proportion_P)
        NG_props_N.append(NG_proportion_N)
        G_props_D.append(G_proportion_D)
        G_props_P.append(G_proportion_P)
        G_props_N.append(G_proportion_N)

    combined_df["NG_props_D"] = NG_props_D
    combined_df["NG_props_P"] = NG_props_P
    combined_df["NG_props_N"] = NG_props_N
    combined_df["G_props_D"] = G_props_D
    combined_df["G_props_P"] = G_props_P
    combined_df["G_props_N"] = G_props_N
    return combined_df

def get_grid_cells_from_same_recording(spatial_firing):

    grid_cells = pd.DataFrame()
    for index, cluster_row in spatial_firing.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)
        session_id = cluster_row["session_id"].iloc[0]

        same_session_id_cells = spatial_firing[spatial_firing["session_id"] == session_id]
        if len(same_session_id_cells)>1:
            grid_cells = pd.concat([grid_cells, cluster_row], ignore_index=True)
    return grid_cells

def plot_class_prection_credence(spatial_firing, save_path):
    distance_cells = spatial_firing[spatial_firing["Lomb_classifier_"] == "Distance"]
    position_cells = spatial_firing[spatial_firing["Lomb_classifier_"] == "Position"]
    null_cells = spatial_firing[spatial_firing["Lomb_classifier_"] == "Null"]
    errorbarwidth =2

    fig, ax = plt.subplots(figsize=(6,6))
    ax.bar(x=0.4, height=np.nanmean(np.asarray(position_cells["G_props_P"]), dtype=np.float64), width=0.1, edgecolor="black", color="turquoise")
    ax.errorbar(x=0.4, y=np.nanmean(np.asarray(position_cells["G_props_P"]), dtype=np.float64), yerr=stats.sem(np.asarray(position_cells["G_props_P"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=0.5, height=np.nanmean(np.asarray(position_cells["G_props_D"]), dtype=np.float64), width=0.1, edgecolor="black", color="orange")
    ax.errorbar(x=0.5, y=np.nanmean(np.asarray(position_cells["G_props_D"]), dtype=np.float64), yerr=stats.sem(np.asarray(position_cells["G_props_D"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=0.6, height=np.nanmean(np.asarray(position_cells["G_props_N"]), dtype=np.float64), width=0.1, edgecolor="black", color="gray")
    ax.errorbar(x=0.6, y=np.nanmean(np.asarray(position_cells["G_props_N"]), dtype=np.float64), yerr=stats.sem(np.asarray(position_cells["G_props_N"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)

    ax.bar(x=0.9, height=np.nanmean(np.asarray(distance_cells["G_props_P"]), dtype=np.float64), width=0.1, edgecolor="black", color="turquoise")
    ax.errorbar(x=0.9, y=np.nanmean(np.asarray(distance_cells["G_props_P"]), dtype=np.float64), yerr=stats.sem(np.asarray(distance_cells["G_props_P"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=1.0, height=np.nanmean(np.asarray(distance_cells["G_props_D"]), dtype=np.float64), width=0.1, edgecolor="black", color="orange")
    ax.errorbar(x=1.0, y=np.nanmean(np.asarray(distance_cells["G_props_D"]), dtype=np.float64), yerr=stats.sem(np.asarray(distance_cells["G_props_D"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=1.1, height=np.nanmean(np.asarray(distance_cells["G_props_N"]), dtype=np.float64), width=0.1, edgecolor="black", color="gray")
    ax.errorbar(x=1.1, y=np.nanmean(np.asarray(distance_cells["G_props_N"]), dtype=np.float64), yerr=stats.sem(np.asarray(distance_cells["G_props_N"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)

    ax.bar(x=1.4, height=np.nanmean(np.asarray(null_cells["G_props_P"]), dtype=np.float64), width=0.1, edgecolor="black", color="turquoise")
    ax.errorbar(x=1.4, y=np.nanmean(np.asarray(null_cells["G_props_P"]), dtype=np.float64), yerr=stats.sem(np.asarray(null_cells["G_props_P"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=1.5, height=np.nanmean(np.asarray(null_cells["G_props_D"]), dtype=np.float64), width=0.1, edgecolor="black", color="orange")
    ax.errorbar(x=1.5, y=np.nanmean(np.asarray(null_cells["G_props_D"]), dtype=np.float64), yerr=stats.sem(np.asarray(null_cells["G_props_D"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=1.6, height=np.nanmean(np.asarray(null_cells["G_props_N"]), dtype=np.float64), width=0.1, edgecolor="black", color="gray")
    ax.errorbar(x=1.6, y=np.nanmean(np.asarray(null_cells["G_props_N"]), dtype=np.float64), yerr=stats.sem(np.asarray(null_cells["G_props_N"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)

    ax.axhline(y=0.33, color="black", linestyle="dashed", linewidth=2)
    plt.ylabel('Probability', fontsize=20, labelpad = 10)
    plt.xlabel("Cell Class", fontsize=20, labelpad = 10)
    ax.set_xticks([0.5, 1, 1.5])
    ax.set_xticklabels(["P", "D", "N"])
    ax.set_yticks([0, 0.5,  1])
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=2)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/prediction_credence_G.png', dpi=200)
    plt.close()


    fig, ax = plt.subplots(figsize=(6,6))
    ax.bar(x=0.4, height=np.nanmean(np.asarray(position_cells["NG_props_P"]), dtype=np.float64), width=0.1, edgecolor="black", color="turquoise", hatch="/")
    ax.errorbar(x=0.4, y=np.nanmean(np.asarray(position_cells["NG_props_P"]), dtype=np.float64), yerr=stats.sem(np.asarray(position_cells["NG_props_P"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=0.5, height=np.nanmean(np.asarray(position_cells["NG_props_D"]), dtype=np.float64), width=0.1, edgecolor="black", color="orange", hatch="/")
    ax.errorbar(x=0.5, y=np.nanmean(np.asarray(position_cells["NG_props_D"]), dtype=np.float64), yerr=stats.sem(np.asarray(position_cells["NG_props_D"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=0.6, height=np.nanmean(np.asarray(position_cells["NG_props_N"]), dtype=np.float64), width=0.1, edgecolor="black", color="gray", hatch="/")
    ax.errorbar(x=0.6, y=np.nanmean(np.asarray(position_cells["NG_props_N"]), dtype=np.float64), yerr=stats.sem(np.asarray(position_cells["NG_props_N"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)

    ax.bar(x=0.9, height=np.nanmean(np.asarray(distance_cells["NG_props_P"]), dtype=np.float64), width=0.1, edgecolor="black", color="turquoise", hatch="/")
    ax.errorbar(x=0.9, y=np.nanmean(np.asarray(distance_cells["NG_props_P"]), dtype=np.float64), yerr=stats.sem(np.asarray(distance_cells["NG_props_P"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=1.0, height=np.nanmean(np.asarray(distance_cells["NG_props_D"]), dtype=np.float64), width=0.1, edgecolor="black", color="orange", hatch="/")
    ax.errorbar(x=1.0, y=np.nanmean(np.asarray(distance_cells["NG_props_D"]), dtype=np.float64), yerr=stats.sem(np.asarray(distance_cells["NG_props_D"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=1.1, height=np.nanmean(np.asarray(distance_cells["NG_props_N"]), dtype=np.float64), width=0.1, edgecolor="black", color="gray", hatch="/")
    ax.errorbar(x=1.1, y=np.nanmean(np.asarray(distance_cells["NG_props_N"]), dtype=np.float64), yerr=stats.sem(np.asarray(distance_cells["NG_props_N"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)

    ax.bar(x=1.4, height=np.nanmean(np.asarray(null_cells["NG_props_P"]), dtype=np.float64), width=0.1, edgecolor="black", color="turquoise", hatch="/")
    ax.errorbar(x=1.4, y=np.nanmean(np.asarray(null_cells["NG_props_P"]), dtype=np.float64), yerr=stats.sem(np.asarray(null_cells["NG_props_P"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=1.5, height=np.nanmean(np.asarray(null_cells["NG_props_D"]), dtype=np.float64), width=0.1, edgecolor="black", color="orange", hatch="/")
    ax.errorbar(x=1.5, y=np.nanmean(np.asarray(null_cells["NG_props_D"]), dtype=np.float64), yerr=stats.sem(np.asarray(null_cells["NG_props_D"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)
    ax.bar(x=1.6, height=np.nanmean(np.asarray(null_cells["NG_props_N"]), dtype=np.float64), width=0.1, edgecolor="black", color="gray", hatch="/")
    ax.errorbar(x=1.6, y=np.nanmean(np.asarray(null_cells["NG_props_N"]), dtype=np.float64), yerr=stats.sem(np.asarray(null_cells["NG_props_N"], dtype=np.float64), nan_policy="omit"), color="black",elinewidth=errorbarwidth)

    ax.axhline(y=0.33, color="black", linestyle="dashed", linewidth=2)
    plt.ylabel('Probability', fontsize=20, labelpad = 10)
    plt.xlabel("Cell Class", fontsize=20, labelpad = 10)
    ax.set_xticks([0.5, 1, 1.5])
    ax.set_xticklabels(["P", "D", "N"])
    ax.set_yticks([0, 0.5,  1])
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=2)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/prediction_credence_NG.png', dpi=200)
    plt.close()

    return

def significance_bar(ax, start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 15,color = 'k'):
    # draw a line with downticks at the ends
    ax.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)
    # draw the text with a bounding box covering up the line
    ax.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize)



def plot_all_paired_vs_shuffle(df, output_path):
    for hmt in ["hit", "miss", "try"]:
        for tt in [0, 1]:
            subset_df = df[((df["hit_miss_try"]==hmt) & (df["trial_type"]==tt))]
            grid_pairs_vs_shuffle = np.array([])
            grid_non_grid_pairs_vs_shuffle = np.array([])
            non_grid_non_grid_pairs_vs_shuffle = np.array([])

            for index, session_row in subset_df.iterrows():
                session_row = session_row.to_frame().T.reset_index(drop=True)
                grid_pairs_vs_shuffle = np.concatenate((grid_pairs_vs_shuffle, session_row.iloc[0]["grid_pairs_vs_shuffle"]))
                grid_non_grid_pairs_vs_shuffle = np.concatenate((grid_non_grid_pairs_vs_shuffle, session_row.iloc[0]["grid_non_grid_pairs_vs_shuffle"]))
                non_grid_non_grid_pairs_vs_shuffle = np.concatenate((non_grid_non_grid_pairs_vs_shuffle, session_row.iloc[0]["non_grid_non_grid_pairs_vs_shuffle"]))

            grid_pairs_vs_shuffle = grid_pairs_vs_shuffle[~np.isnan(grid_pairs_vs_shuffle)]
            grid_non_grid_pairs_vs_shuffle = grid_non_grid_pairs_vs_shuffle[~np.isnan(grid_non_grid_pairs_vs_shuffle)]
            non_grid_non_grid_pairs_vs_shuffle = non_grid_non_grid_pairs_vs_shuffle[~np.isnan(non_grid_non_grid_pairs_vs_shuffle)]

            data = [grid_pairs_vs_shuffle, grid_non_grid_pairs_vs_shuffle, non_grid_non_grid_pairs_vs_shuffle];
            gg_p = stats.wilcoxon(grid_pairs_vs_shuffle)[1]; gg_p= get_p_text(gg_p, ns=False)
            gng_p = stats.wilcoxon(grid_non_grid_pairs_vs_shuffle)[1]; gng_p= get_p_text(gng_p, ns=False)
            ngng_p = stats.wilcoxon(non_grid_non_grid_pairs_vs_shuffle)[1]; ngng_p= get_p_text(ngng_p, ns=False)

            gg_ngng_p = stats.mannwhitneyu(grid_pairs_vs_shuffle, non_grid_non_grid_pairs_vs_shuffle)[1]
            gg_gng_p = stats.mannwhitneyu(grid_pairs_vs_shuffle, grid_non_grid_pairs_vs_shuffle)[1]
            gng_ngng_p = stats.mannwhitneyu(grid_non_grid_pairs_vs_shuffle, non_grid_non_grid_pairs_vs_shuffle)[1]

            fig, ax = plt.subplots(figsize=(6,6))
            vp = ax.violinplot(data, [2, 4, 6], widths=2,
                               showmeans=False, showmedians=True, showextrema=False)
            ax.axhline(y=0, linestyle="dashed", color="black")
            significance_bar(ax, start=2, end=4, height=0.5, displaystring=get_p_text(gg_gng_p))
            significance_bar(ax, start=4, end=6, height=0.55, displaystring=get_p_text(gng_ngng_p))
            significance_bar(ax, start=2, end=6, height=0.6, displaystring=get_p_text(gg_ngng_p))
            ax.set_xticks([2,4,6])
            ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
            ax.set_ylim([-0.4, 0.6])
            ax.set_xticklabels(["G-G", "G-NG", "NG-NG"])
            ax.set_ylabel("Change in spatial\ncorrelation vs shuffle", fontsize=20, labelpad=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_tick_params(length=0)
            ax.tick_params(axis='both', which='major', labelsize=25)
            fig.tight_layout()
            plt.savefig(output_path + '/joint_alignment_correlations_violin_shuffle_vs_real_'+hmt+'_'+str(tt)+'png', dpi=300)
            plt.close()


def plot_paired_vs_shuffle_by_tt(df, output_path):
    df = df[df["hit_miss_try"]=="hit"]
    b_df = df[df["trial_type"]==0]
    nb_df = df[df["trial_type"]==1]
    for collumn in ["grid_pairs_vs_shuffle", "grid_non_grid_pairs_vs_shuffle", "non_grid_non_grid_pairs_vs_shuffle"]:
        b_pairs_vs_shuffle = np.array([])
        nb_pairs_vs_shuffle = np.array([])

        for index, session_row in b_df.iterrows():
            session_row = session_row.to_frame().T.reset_index(drop=True)
            b_pairs_vs_shuffle = np.concatenate((b_pairs_vs_shuffle, session_row.iloc[0][collumn]))
        for index, session_row in nb_df.iterrows():
            session_row = session_row.to_frame().T.reset_index(drop=True)
            nb_pairs_vs_shuffle = np.concatenate((nb_pairs_vs_shuffle, session_row.iloc[0][collumn]))

        b_pairs_vs_shuffle = b_pairs_vs_shuffle[~np.isnan(b_pairs_vs_shuffle)]
        nb_pairs_vs_shuffle = nb_pairs_vs_shuffle[~np.isnan(nb_pairs_vs_shuffle)]

        data = [b_pairs_vs_shuffle, nb_pairs_vs_shuffle];
        b_p = stats.wilcoxon(b_pairs_vs_shuffle)[1]; b_p= get_p_text(b_p, ns=False)
        nb_p = stats.wilcoxon(nb_pairs_vs_shuffle)[1]; nb_p= get_p_text(nb_p, ns=False)

        bnb_p = stats.mannwhitneyu(b_pairs_vs_shuffle, nb_pairs_vs_shuffle)[1]

        fig, ax = plt.subplots(figsize=(6,6))
        vp = ax.violinplot(data, [2, 4], widths=2,
                           showmeans=False, showmedians=True, showextrema=False)
        ax.axhline(y=0, linestyle="dashed", color="black")
        significance_bar(ax, start=2, end=4, height=0.6, displaystring=get_p_text(bnb_p))
        ax.set_xticks([2,4])
        ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
        ax.set_ylim([-0.4, 0.6])
        ax.set_xticklabels(["Cued", "PI"])
        ax.set_ylabel("Change in spatial\ncorrelation vs shuffle", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='major', labelsize=25)
        fig.tight_layout()
        plt.savefig(output_path + '/joint_alignment_correlations_violin_shuffle_vs_real_'+collumn+'_by_tt.png', dpi=300)
        plt.close()

def plot_paired_vs_shuffle_by_hmt(df, output_path):
    df = df[df["trial_type"]==1]
    hit_df = df[df["hit_miss_try"]=="hit"]
    try_df = df[df["hit_miss_try"]=="try"]
    miss_df = df[df["hit_miss_try"]=="miss"]
    for collumn in ["grid_pairs_vs_shuffle", "grid_non_grid_pairs_vs_shuffle", "non_grid_non_grid_pairs_vs_shuffle"]:
        hit_pairs_vs_shuffle = np.array([])
        try_pairs_vs_shuffle = np.array([])
        miss_pairs_vs_shuffle = np.array([])

        for index, session_row in hit_df.iterrows():
            session_row = session_row.to_frame().T.reset_index(drop=True)
            hit_pairs_vs_shuffle = np.concatenate((hit_pairs_vs_shuffle, session_row.iloc[0][collumn]))
        for index, session_row in try_df.iterrows():
            session_row = session_row.to_frame().T.reset_index(drop=True)
            try_pairs_vs_shuffle = np.concatenate((try_pairs_vs_shuffle, session_row.iloc[0][collumn]))
        for index, session_row in miss_df.iterrows():
            session_row = session_row.to_frame().T.reset_index(drop=True)
            miss_pairs_vs_shuffle = np.concatenate((miss_pairs_vs_shuffle, session_row.iloc[0][collumn]))

        hit_pairs_vs_shuffle = hit_pairs_vs_shuffle[~np.isnan(hit_pairs_vs_shuffle)]
        try_pairs_vs_shuffle = try_pairs_vs_shuffle[~np.isnan(try_pairs_vs_shuffle)]
        miss_pairs_vs_shuffle = miss_pairs_vs_shuffle[~np.isnan(miss_pairs_vs_shuffle)]

        data = [hit_pairs_vs_shuffle, try_pairs_vs_shuffle, miss_pairs_vs_shuffle];
        h_p = stats.wilcoxon(hit_pairs_vs_shuffle)[1]; h_p= get_p_text(h_p, ns=False)
        t_p = stats.wilcoxon(try_pairs_vs_shuffle)[1]; t_p= get_p_text(t_p, ns=False)
        m_p = stats.wilcoxon(miss_pairs_vs_shuffle)[1]; m_p= get_p_text(m_p, ns=False)

        hm_p = stats.mannwhitneyu(hit_pairs_vs_shuffle, miss_pairs_vs_shuffle)[1]
        ht_p = stats.mannwhitneyu(hit_pairs_vs_shuffle, try_pairs_vs_shuffle)[1]
        mt_p = stats.mannwhitneyu(miss_pairs_vs_shuffle, try_pairs_vs_shuffle)[1]

        fig, ax = plt.subplots(figsize=(6,6))
        vp = ax.violinplot(data, [2, 4, 6], widths=2,
                           showmeans=False, showmedians=True, showextrema=False)
        ax.axhline(y=0, linestyle="dashed", color="black")
        significance_bar(ax, start=2, end=4, height=0.5, displaystring=get_p_text(ht_p))
        significance_bar(ax, start=4, end=6, height=0.55, displaystring=get_p_text(mt_p))
        significance_bar(ax, start=2, end=6, height=0.6, displaystring=get_p_text(hm_p))
        ax.set_xticks([2,4,6])
        ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
        ax.set_ylim([-0.4, 0.6])
        ax.set_xticklabels(["Hit", "Try", "Miss"])
        ax.set_ylabel("Change in spatial\ncorrelation vs shuffle", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='major', labelsize=25)
        fig.tight_layout()
        plt.savefig(output_path + '/joint_alignment_correlations_violin_shuffle_vs_real_'+collumn+'_by_hmt.png', dpi=300)
        plt.close()


def get_n_simultaneously_recorded_cells(grid_cells, group1, group2):
    #how many group2 cells have been simulatenously recorded with group1 cells
    n =0
    session_ids = np.unique(grid_cells["session_id"])
    for i in range(len(session_ids)):
        session_id = session_ids[i]
        single_session = grid_cells[grid_cells["session_id"]==session_id]
        n_group1 = len(single_session[single_session["Lomb_classifier_"] == group1])
        n_group2 = len(single_session[single_session["Lomb_classifier_"] == group2])

        if group1==group2:
            if n_group1>1:
               n+=n_group1
        else:
            if n_group1>0:
                n+=n_group2
    return n

def plot_n_cells_simulatenously_recorded(concantenated_dataframe,  save_path, normalised=False):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    grid_cells = add_lomb_classifier(grid_cells, suffix="")

    P_proportion = len(grid_cells[grid_cells["Lomb_classifier_"] == "Position"])/len(grid_cells)
    D_proportion = len(grid_cells[grid_cells["Lomb_classifier_"] == "Distance"])/len(grid_cells)
    N_proportion = len(grid_cells[grid_cells["Lomb_classifier_"] == "Null"])/len(grid_cells)
    if not normalised:
        P_proportion = 1
        D_proportion = 1
        N_proportion = 1

    objects = ["Position", "Distance", "Null"]
    x_pos = np.arange(len(objects))
    fig, axes = plt.subplots(3, 1, figsize=(6,6), sharex=True)
    for ax, group, color in zip(axes, objects, [Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]):
        n_p = get_n_simultaneously_recorded_cells(grid_cells, group1=group, group2="Position")/P_proportion
        n_d = get_n_simultaneously_recorded_cells(grid_cells, group1=group, group2="Distance")/D_proportion
        n_n = get_n_simultaneously_recorded_cells(grid_cells, group1=group, group2="Null")/N_proportion

        ax.bar(x_pos[0], n_p, color=Settings.allocentric_color, edgecolor="black")
        ax.bar(x_pos[1], n_d, color=Settings.egocentric_color, edgecolor="black")
        ax.bar(x_pos[2], n_n, color=Settings.null_color, edgecolor="black")

        Edmond.plot_utility2.style_vr_plot(ax)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.set_ylabel(group, fontsize=25)
        ax.set_yticks([0, int(np.round(ax.get_ylim()[1]))])
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["P", "D", "N"])
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    if normalised:
        plt.savefig(save_path + '/simulatenous_recorded_cell_number_normalised.png', dpi=300)
    else:
        plt.savefig(save_path + '/simulatenous_recorded_cell_numbers.png', dpi=300)
    plt.close()

    return


def reconstruct_signal_with_shifts(subset_processed_position_data, map_shifts_cluster_i, cluster_firing_maps):
    reconstructed_signal=[]
    for ti, tn in enumerate(subset_processed_position_data["trial_number"]):
        reconstructed_trial = get_reconstructed_trial_signal(map_shifts_cluster_i[ti], cluster_firing_maps[ti].flatten(),
                                                             min_shift=min(map_shifts_cluster_i), max_shift=max(map_shifts_cluster_i))
        reconstructed_signal.append(reconstructed_trial.tolist())
    reconstructed_signal = np.array(reconstructed_signal)
    return reconstructed_signal

def analyse_jitter_correlations(spike_data, of_spike_data, processed_position_data):
    matched_recording_df = pd.DataFrame()
    spike_data = pd.merge(spike_data, of_spike_data[["cluster_id", "grid_cell"]], on="cluster_id")

    for hmt, tt in zip(["all", "hit", "try", "miss", "hit"], ["all", 1, 1, 1, 0]):
        # subset a processed position dataframe
        if hmt == "all" and tt == "all":
            subset_processed_position_data = processed_position_data.copy()
        else:
            subset_processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] == hmt]
            subset_processed_position_data = subset_processed_position_data[subset_processed_position_data["trial_type"] == tt]

        # loop over cells using the maximised map shifts for a given cell
        for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
            cluster_i_df = spike_data[spike_data["cluster_id"]==cluster_id]

            # loop over cells which will have their ratemaps modified by cell i's map shifts
            for cluster_index_j, cluster_id_j in enumerate(spike_data.cluster_id):
                cluster_j_df = spike_data[spike_data["cluster_id"]==cluster_id_j]
                firing_times_cluster = cluster_j_df["firing_times"].iloc[0]

                # call the map shifts from cell i and shuffle if
                map_shifts_cluster_i = np.array(cluster_i_df['map_realignments'].iloc[0])

                # analyse if there is any spikes
                if (len(firing_times_cluster)>1):
                    cluster_firing_maps = np.array(cluster_j_df['fr_binned_in_space_smoothed'].iloc[0])
                    cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
                    cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
                    cluster_firing_maps = min_max_normalize(cluster_firing_maps)
                    avg_correlation = get_avg_correlation(cluster_firing_maps)

                    # reconstruct avg spatial correlation using the newly aligned trials
                    reconstructed_signal = reconstruct_signal_with_shifts(subset_processed_position_data, map_shifts_cluster_i, cluster_firing_maps)
                    reconstructed_signal_corr = get_avg_correlation(reconstructed_signal)

                    # now shuffle the map shifts and repeat the reconstruction
                    np.random.shuffle(map_shifts_cluster_i)
                    reconstructed_signal_shuffle = reconstruct_signal_with_shifts(subset_processed_position_data, map_shifts_cluster_i, cluster_firing_maps)
                    reconstructed_signal_corr_shuffle = get_avg_correlation(reconstructed_signal_shuffle)

                else:
                    reconstructed_signal_corr = np.nan
                    reconstructed_signal_corr_shuffle = np.nan

                # determine if the pair were on the same tetrode
                if cluster_i_df.tetrode.iloc[0] == cluster_j_df.tetrode.iloc[0]:
                    tetrode_level = "same"
                else:
                    tetrode_level = "different"

                # determine if the pair was a pair of grid cells or not
                if (cluster_i_df.grid_cell.iloc[0] == True) and (cluster_j_df.grid_cell.iloc[0] == True):
                    pair_type = "G-G"
                elif ((cluster_i_df.grid_cell.iloc[0] == True) or (cluster_j_df.grid_cell.iloc[0] == True)):
                    pair_type = "G-NG"
                else:
                    pair_type = "NG-NG"

                # collect all the reconstructed signals and the dependant variables
                cell_pair_df = pd.DataFrame()
                cell_pair_df["session_id_i"] = [cluster_i_df.session_id.iloc[0]]
                cell_pair_df["session_id_j"] = [cluster_j_df.session_id.iloc[0]]
                cell_pair_df["cluster_id_i"] = [cluster_i_df.cluster_id.iloc[0]]
                cell_pair_df["cluster_id_j"] = [cluster_j_df.cluster_id.iloc[0]]
                cell_pair_df["trial_type"] = [tt]
                cell_pair_df["hit_miss_try"] = [hmt]
                cell_pair_df["tetrode_level"] = [tetrode_level]
                cell_pair_df["pair_type"] = [pair_type]
                cell_pair_df["reconstructed_signal_corr"] = [reconstructed_signal_corr-avg_correlation]
                cell_pair_df["reconstructed_signal_corr_shuffle"] = [reconstructed_signal_corr_shuffle-avg_correlation]

                # only save when cells aren't the same exact cell in a pair
                if (cluster_id != cluster_id_j):
                    matched_recording_df = pd.concat([matched_recording_df, cell_pair_df], ignore_index=True)

    return matched_recording_df


def plot_jitter_correlations(matched_recording_df, output_path):
    print('plotting joint cell correlations...')
    save_path = output_path + '/Figures/joint_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for hmt, tt in zip(["all", "hit", "try", "miss", "hit"], ["all", 1, 1, 1, 0]):
        if hmt == "all" and tt == "all":
            subset_matched_recording_df = matched_recording_df[matched_recording_df["hit_miss_try"] == "all"]
            subset_matched_recording_df = subset_matched_recording_df[subset_matched_recording_df["trial_type"] == tt]
        else:
            subset_matched_recording_df = matched_recording_df[(matched_recording_df["hit_miss_try"] == hmt) &
                                                               (matched_recording_df["trial_type"] == tt)]

        fig, ax = plt.subplots(figsize=(6,6))
        ax.axhline(y=0, linestyle="dashed", color="gray")
        ax.axvline(x=0, linestyle="dashed", color="gray")
        ax.plot(np.arange(-2, 2), np.arange(-2, 2), linestyle="solid", color="black")

        for pair_type, pair_color in zip(["NG-NG", "G-NG", "G-G"], ["black", "blue", "red"]):
            pair_type_df = subset_matched_recording_df[subset_matched_recording_df["pair_type"] == pair_type]
            on_tetrode_pair_df = pair_type_df[pair_type_df["tetrode_level"] == "same"]
            off_tetrode_pair_df = pair_type_df[pair_type_df["tetrode_level"] == "different"]

            ax.scatter(off_tetrode_pair_df["reconstructed_signal_corr"], off_tetrode_pair_df["reconstructed_signal_corr_shuffle"], c=pair_color, marker="o")
            ax.scatter(on_tetrode_pair_df["reconstructed_signal_corr"], on_tetrode_pair_df["reconstructed_signal_corr_shuffle"], c=pair_color, marker="x")

        ax.set_ylabel("Change in R (Shuffle)", fontsize=20)
        ax.set_xlabel("Change in R (Real)", fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlim([-0.5, 0.5])
        ax.tick_params(axis='both', which='major', labelsize=15)
        fig.tight_layout()
        plt.savefig(save_path + '/' + matched_recording_df.session_id_i.iloc[0] + 'joint_alignment_correlations_shuffle_vs_real_'+hmt+'_'+str(tt)+'png', dpi=300)
        plt.close()


def plot_agreement_matrix(spike_data, of_spike_data, output_path, agreement_comparions_df):
    print('plotting agreement matrices...')
    save_path = output_path + '/Figures/agreement_matrices'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    spike_data = pd.merge(spike_data, of_spike_data[["cluster_id", "grid_cell"]], on="cluster_id")
    spike_data = spike_data[spike_data["grid_cell"] == True]
    cluster_ids = np.array(spike_data["cluster_id"])

    agreement_matrix = np.zeros((len(spike_data), len(spike_data)))
    agreement_matrix_shuffled = np.zeros((len(spike_data), len(spike_data)))

    for i in range(len(agreement_matrix)):
        for j in range(len(agreement_matrix[0])):
            cluster_j_df = spike_data[spike_data["cluster_id"] == cluster_ids[j]]
            cluster_i_df = spike_data[spike_data["cluster_id"] == cluster_ids[i]]
            rolling_classifier_i = np.array(cluster_i_df["rolling:rolling_lomb_classifiers"].iloc[0])
            rolling_classifier_j = np.array(cluster_j_df["rolling:rolling_lomb_classifiers"].iloc[0])
            rolling_classifier_i_shuffled = np.array(cluster_i_df["rolling:rolling_lomb_classifiers_shuffled_blocks"].iloc[0])
            rolling_classifier_j_shuffled = np.array(cluster_j_df["rolling:rolling_lomb_classifiers_shuffled_blocks"].iloc[0])

            encoding_p_j = cluster_j_df["rolling:proportion_encoding_position"].iloc[0]
            encoding_d_j = cluster_j_df["rolling:proportion_encoding_distance"].iloc[0]
            encoding_n_j = cluster_j_df["rolling:proportion_encoding_null"].iloc[0]
            encoding_p_i = cluster_i_df["rolling:proportion_encoding_position"].iloc[0]
            encoding_d_i = cluster_i_df["rolling:proportion_encoding_distance"].iloc[0]
            encoding_n_i = cluster_i_df["rolling:proportion_encoding_null"].iloc[0]

            agreement = np.sum(rolling_classifier_i==rolling_classifier_j)/len(rolling_classifier_i)
            agreement_shuffled_blocks = np.sum(rolling_classifier_i_shuffled==rolling_classifier_j_shuffled)/len(rolling_classifier_i_shuffled)

            agreement_matrix[i, j] = agreement
            agreement_matrix_shuffled[i, j] = agreement_shuffled_blocks

            if ((((encoding_p_j > 0.15) and  (encoding_p_j < 0.85)) and ((encoding_p_i > 0.15) and  (encoding_p_i < 0.85))) or
                (((encoding_d_j > 0.15) and  (encoding_d_j < 0.85)) and ((encoding_d_i > 0.15) and  (encoding_d_i < 0.85))) or
                (((encoding_n_j > 0.15) and  (encoding_n_j < 0.85)) and ((encoding_n_i > 0.15) and  (encoding_n_i < 0.85)))):



                # determine if the pair were on the same tetrode
                if cluster_i_df.tetrode.iloc[0] == cluster_j_df.tetrode.iloc[0]:
                    tetrode_level = "same"
                else:
                    tetrode_level = "different"
                cell_pair_df = pd.DataFrame()
                cell_pair_df["session_id_i"] = [cluster_i_df.session_id.iloc[0]]
                cell_pair_df["session_id_j"] = [cluster_j_df.session_id.iloc[0]]
                cell_pair_df["cluster_id_i"] = [cluster_i_df.cluster_id.iloc[0]]
                cell_pair_df["cluster_id_j"] = [cluster_j_df.cluster_id.iloc[0]]
                cell_pair_df["tetrode_level"] = [tetrode_level]
                cell_pair_df["agreement"] = [agreement]
                cell_pair_df["agreement_shuffled_blocks"] = [agreement_shuffled_blocks]
                agreement_comparions_df = pd.concat([agreement_comparions_df, cell_pair_df], ignore_index=True)

    fig, ax = plt.subplots()
    np.fill_diagonal(agreement_matrix, np.nan)
    im= ax.imshow(agreement_matrix, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cluster_ids)))
    ax.set_yticks(np.arange(len(cluster_ids)))
    ax.set_yticklabels(cluster_ids)
    ax.set_xticklabels(cluster_ids)
    ax.set_ylabel("Cluster ID", fontsize=20)
    ax.set_xlabel("Cluster ID", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    fig.colorbar(im, ax=ax)
    plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + 'agreement_matrix_rolling_classifiers.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    np.fill_diagonal(agreement_matrix_shuffled, np.nan)
    im= ax.imshow(agreement_matrix_shuffled, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cluster_ids)))
    ax.set_yticks(np.arange(len(cluster_ids)))
    ax.set_yticklabels(cluster_ids)
    ax.set_xticklabels(cluster_ids)
    ax.set_ylabel("Cluster ID", fontsize=20)
    ax.set_xlabel("Cluster ID", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    fig.colorbar(im, ax=ax)
    plt.savefig(save_path + '/' + spike_data.session_id.iloc[0] + 'agreement_matrix_rolling_classifiers_shuffled_blocks.png', dpi=300)
    plt.close()

    return agreement_comparions_df

def remove_same_cell_comparisons(agreement_comparions_df):
    new=pd.DataFrame()
    for index, cluster_row in agreement_comparions_df.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)
        cluster_id_i =cluster_row["cluster_id_i"].iloc[0]
        cluster_id_j =cluster_row["cluster_id_j"].iloc[0]

        if cluster_id_j != cluster_id_i:
            new = pd.concat([new, cluster_row], ignore_index=True)

    return new


def plot_agreement_vs_shuffled_blocks(agreement_comparions_df, save_path):
    agreement_comparions_df = remove_same_cell_comparisons(agreement_comparions_df)
    print(len(agreement_comparions_df))
    _, p = stats.ttest_rel(agreement_comparions_df["agreement"], agreement_comparions_df["agreement_shuffled_blocks"])
    on_tetrode = agreement_comparions_df[agreement_comparions_df["tetrode_level"] == "same"]
    off_tetrode = agreement_comparions_df[agreement_comparions_df["tetrode_level"] == "different"]
    fig, ax = plt.subplots(figsize=(4,6))
    for i in range(len(agreement_comparions_df)):
        ax.scatter([0,1], [agreement_comparions_df["agreement"].iloc[i], agreement_comparions_df["agreement_shuffled_blocks"].iloc[i]], marker="o", color="black")
        ax.plot([0,1], [agreement_comparions_df["agreement"].iloc[i], agreement_comparions_df["agreement_shuffled_blocks"].iloc[i]], color="black")
    ax.set_xticks([0,1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim([-0.5,1.5])
    plt.xticks(rotation = 30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels(["True", "Shuffled blocks"])
    ax.set_ylabel("Block agreement (frac. session)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    plt.savefig(save_path + '/' +'agreement_vs_shuffle.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    for i in range(len(on_tetrode)):
        ax.scatter([0,1], [on_tetrode["agreement"].iloc[i], on_tetrode["agreement_shuffled_blocks"].iloc[i]], marker="o", color="black")
        ax.plot([0,1], [on_tetrode["agreement"].iloc[i], on_tetrode["agreement_shuffled_blocks"].iloc[i]], color="black")
    for i in range(len(off_tetrode)):
        ax.scatter([3,4], [off_tetrode["agreement"].iloc[i], off_tetrode["agreement_shuffled_blocks"].iloc[i]], marker="o", color="black")
        ax.plot([3,4], [off_tetrode["agreement"].iloc[i], off_tetrode["agreement_shuffled_blocks"].iloc[i]], color="black")

    ax.set_xticks([0,1,3,4])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["TrueON", "Shuffled blocksON", "TrueOFF", "Shuffled blocksOFF"])
    ax.set_ylabel("Block agreement (frac. session)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    plt.savefig(save_path + '/' +'agreement_vs_shuffle_by_tetrode.png', dpi=300)
    plt.close()



def process_recordings(vr_recording_path_list, of_recording_path_list):
    concat_matched_recording_df = pd.DataFrame()

    agreement_comparions_df = pd.DataFrame()
    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            processed_position_data = add_avg_track_speed(processed_position_data, track_length=get_track_length(recording))
            processed_position_data, _ = add_hit_miss_try3(processed_position_data, track_length=get_track_length(recording))

            if paired_recording is not None:
                of_spike_data = pd.read_pickle(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl")
                #plot_firing_rate_maps_per_trial_aligned_other_neuron(spike_data=spike_data, of_spike_data=of_spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=get_track_length(recording), shuffled=False)
                #plot_firing_rate_maps_per_trial_aligned_other_neuron(spike_data=spike_data, of_spike_data=of_spike_data, processed_position_data=processed_position_data, output_path=output_path, track_length=get_track_length(recording), shuffled=True)

                # ANALYSIS OF ROLLING CLASSIFICATION
                agreement_comparions_df = plot_agreement_matrix(spike_data, of_spike_data, output_path, agreement_comparions_df)
                agreement_comparions_df.to_pickle("/mnt/datastore/Harry/VR_grid_cells/agreement_comparions_df.pkl")

                #spike_data = spike_data.head(10)
                #of_spike_data = of_spike_data.head(10)
                #spike_data = pd.merge(spike_data, of_spike_data[["cluster_id", "grid_cell"]], on="cluster_id")
                #spike_data = spike_data[spike_data["grid_cell"] == True]
                #of_spike_data = of_spike_data[of_spike_data["grid_cell"] == True]
                #matched_recording_df = analyse_jitter_correlations(spike_data, of_spike_data, processed_position_data)
                #plot_jitter_correlations(matched_recording_df, output_path)
                #concat_matched_recording_df = pd.concat([concat_matched_recording_df, matched_recording_df], ignore_index=True)
                #concat_matched_recording_df.to_pickle("/mnt/datastore/Harry/cohort8_may2021/matched_grid_recording_df.pkl")
                #plot_firing_rate_maps_per_trial_by_hmt_aligned_other_neuron(spike_data=spike_data, processed_position_data=processed_position_data, output_path=output_path, trial_types=[1])
                #matched_recording_df = plot_joint_jitter_correlations(spike_data, of_spike_data, processed_position_data, position_data, output_path, get_track_length(recording), matched_recording_df)
                #matched_recording_df.to_pickle("/mnt/datastore/Harry/cohort8_may2021/matched_grid_recording_df.pkl")
                #plot_joint_cell_cross_correlations(spike_data, output_path)


            #spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

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
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()]
    #vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()]

    of_path_list = []
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])

    # all of these recordings have at least 2 grid cells recorded
    vr_path_list = ["/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36",
                    "/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D11_2020-08-17_14-57-20",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D12_2021-05-25_09-49-23",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D15_2021-05-28_10-42-15",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D16_2021-05-31_10-21-05",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D18_2021-06-02_10-36-39",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D3_2021-05-12_09-37-41",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D34_2021-06-24_11-52-48",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D39_2021-07-01_11-47-10",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D40_2021-07-02_12-58-24",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D43_2021-07-07_11-51-08",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D44_2021-07-08_12-03-21",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D45_2021-07-09_11-39-02",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M14_D12_2021-05-25_11-03-39",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M14_D15_2021-05-28_12-29-15",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M14_D16_2021-05-31_12-01-35",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M14_D20_2021-06-04_12-20-57",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M14_D26_2021-06-14_12-22-50",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M14_D27_2021-06-15_12-21-58",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M14_D31_2021-06-21_12-07-01",
                    "/mnt/datastore/Harry/Cohort8_may2021/vr/M14_D35_2021-06-25_12-41-16",
                    "/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D23_2020-11-28_15-13-28",
                    "/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D18_2020-11-21_14-29-49",
                    "/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D22_2020-11-27_15-01-24",]
    #vr_path_list = ["/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36"]
    process_recordings(vr_path_list, of_path_list)

    #combined_df = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")
    #combined_df = add_lomb_classifier(combined_df,suffix="")
    #combined_df = add_percentage_for_lomb_classes(combined_df)

    # load df for plot_all_paired_vs_shuffle
    #plot_all_paired_vs_shuffle(pd.read_pickle("/mnt/datastore/Harry/cohort8_may2021/matched_grid_recording_df.pkl"), output_path= "/mnt/datastore/Harry/Vr_grid_cells/joint_activity")
    #plot_paired_vs_shuffle_by_hmt(pd.read_pickle("/mnt/datastore/Harry/cohort8_may2021/matched_grid_recording_df.pkl"), output_path= "/mnt/datastore/Harry/Vr_grid_cells/joint_activity")
    #plot_paired_vs_shuffle_by_tt(pd.read_pickle("/mnt/datastore/Harry/cohort8_may2021/matched_grid_recording_df.pkl"), output_path= "/mnt/datastore/Harry/Vr_grid_cells/joint_activity")

    #grid_cells = combined_df[combined_df["classifier"] == "G"]
    #grid_cells_from_same_recording = get_grid_cells_from_same_recording(grid_cells)
    #plot_class_prection_credence(grid_cells_from_same_recording, save_path="/mnt/datastore/Harry/Vr_grid_cells/joint_activity")
    #plot_n_cells_simulatenously_recorded(combined_df,  save_path="/mnt/datastore/Harry/Vr_grid_cells/joint_activity")
    #plot_n_cells_simulatenously_recorded(combined_df,  save_path="/mnt/datastore/Harry/Vr_grid_cells/joint_activity", normalised=True)


    agreement_comparions_df = pd.read_pickle("/mnt/datastore/Harry/VR_grid_cells/agreement_comparions_df.pkl")
    plot_agreement_vs_shuffled_blocks(agreement_comparions_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/joint_activity")
    print("look now")

if __name__ == '__main__':
    main()
