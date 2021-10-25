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
from scipy import signal
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel
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
warnings.filterwarnings('ignore')

def find_set(a,b):
    return set(a) & set(b)

def plot_shuffle_spatial_autocorrelogram(spike_data, position_data, output_path, track_length, suffix="", toggle_shuffle=True):
    print('plotting spike spatial autocorrelogram of the shuffled data...')
    save_path = output_path + '/Figures/spatial_autocorrelograms_shuffles'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    position_x_data= np.asarray(position_data["x_position_cm"])
    position_time_data = np.asarray(position_data["time_seconds"])
    position_tn_data = np.asarray(position_data["trial_number"])

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])
        trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])
        x_position_cluster = np.array(cluster_spike_data["x_position_cm"].iloc[0])
        recording_length_sampling_points = np.array(cluster_spike_data['recording_length_sampling_points'])[0]

        if len(firing_times_cluster)>1:
            for i in range(1):
                random_firing_additions = np.random.randint(low=int(20*settings.sampling_rate), high=int(580*settings.sampling_rate), size=len(firing_times_cluster))
                shuffled_firing_times = firing_times_cluster.copy()
                if toggle_shuffle: # use this boolean to turn on or off shuffling
                    shuffled_firing_times = firing_times_cluster + random_firing_additions
                    shuffled_firing_times[shuffled_firing_times >= recording_length_sampling_points] = shuffled_firing_times[shuffled_firing_times >= recording_length_sampling_points] - recording_length_sampling_points # wrap around the firing times that exceed the length of the recording
                shuffled_firing_times_seconds = shuffled_firing_times/(settings.sampling_rate) # convert from samples to seconds
                shuffled_firing_times_seconds = np.sort(shuffled_firing_times_seconds)

                x=[]
                tn=[]
                for j in range(len(shuffled_firing_times_seconds)):
                    closest_x = position_x_data[np.abs(position_time_data-shuffled_firing_times_seconds[j]).argmin()]
                    closest_tn = position_tn_data[np.abs(position_time_data-shuffled_firing_times_seconds[j]).argmin()]
                    x.append(closest_x)
                    tn.append(closest_tn)
                trial_numbers = np.array(tn)
                x_position_cluster = np.array(x)

                lap_distance_covered = (trial_numbers*track_length)-track_length #total elapsed distance
                x_position_cluster = x_position_cluster+lap_distance_covered
                x_position_cluster = x_position_cluster[~np.isnan(x_position_cluster)]
                x_position_cluster_bins = np.floor(x_position_cluster).astype(int)
                autocorr_window_size = 400
                lags = np.arange(0, autocorr_window_size, 1).astype(int) # were looking at 10 timesteps back and 10 forward

                autocorrelogram = np.array([])
                for lag in lags:
                    correlated = len(find_set(x_position_cluster_bins+lag, x_position_cluster_bins))
                    autocorrelogram = np.append(autocorrelogram, correlated)

                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
                ax.bar(lags[1:], autocorrelogram[1:], edgecolor="black", align="edge")
                plt.ylabel('Counts', fontsize=20, labelpad = 10)
                plt.xlabel('Lag (cm)', fontsize=20, labelpad = 10)
                plt.xlim(0,400)
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                Edmond.plot_utility2.style_vr_plot(ax, x_max=max(autocorrelogram[1:]))
                plt.locator_params(axis = 'x', nbins  = 8)
                tick_spacing = 50
                ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_autocorrelogram_shuffle_Cluster_' + str(cluster_id) + "_shuffle_" + str(i) + "_" + suffix + '.png', dpi=50)
                plt.close()

def get_track_length(recording_path):
    parameter_file_path = control_sorting_analysis.get_tags_parameter_file(recording_path)
    stop_threshold, track_length, cue_conditioned_goal = PostSorting.post_process_sorted_data_vr.process_running_parameter_tag(parameter_file_path)
    return track_length

def process_recordings(vr_recording_path_list):

    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            plot_shuffle_spatial_autocorrelogram(spike_data, position_data, output_path, track_length=get_track_length(recording), suffix="", toggle_shuffle=True)
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
    process_recordings(vr_path_list)

    print("look now`")


if __name__ == '__main__':
    main()
