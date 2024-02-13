import pandas as pd
import numpy as np
import Edmond.eLife_Grid_anchoring_2024.analysis_settings as Settings
from astropy.timeseries import LombScargle
from astropy.convolution import convolve, Gaussian1DKernel
import os
import traceback
import sys

def add_rolling_threshold(spike_data, shuffle_path):
    gauss_kernel_std=Settings.rate_map_gauss_kernel_std
    gauss_kernel = Gaussian1DKernel(stddev=gauss_kernel_std)
    n_window_size=Settings.rolling_window_size_for_lomb_classifier

    rolling_thresholds = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        power_threshold = cluster_spike_data["power_threshold"].iloc[0]
        cluster_shuffle_path = shuffle_path+str(cluster_id)+"_shuffle.pkl"
        if os.path.exists(cluster_shuffle_path):
            cluster_shuffle = pd.read_pickle(cluster_shuffle_path)

            peak_powers_from_a_single_rolling_window = []
            for i in range(len(cluster_shuffle)):
                shuffle_firing_rate_map = cluster_shuffle['shuffled_rate_map'].iloc[i]

                track_length = len(shuffle_firing_rate_map[0])
                n_trials = len(shuffle_firing_rate_map)
                elapsed_distance_bins = np.arange(0, (track_length*n_trials)+1, 1)
                elapsed_distance = 0.5*(elapsed_distance_bins[1:]+elapsed_distance_bins[:-1])/track_length
                frequency = Settings.frequency
                sliding_window_size=track_length*Settings.window_length_in_laps
                indices_to_test = np.arange(0, len(elapsed_distance)-sliding_window_size, 1, dtype=np.int64)[::Settings.power_estimate_step]
                indices_to_test = indices_to_test[:n_window_size] # for single window calculation

                # flatten and smooth shuffled rate map
                shuffle_firing_rate_map = shuffle_firing_rate_map.flatten()
                fr_smoothed = convolve(shuffle_firing_rate_map, gauss_kernel)

                powers = []
                for m in indices_to_test:
                    ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr_smoothed[m:m+sliding_window_size])
                    power = ls.power(frequency)
                    powers.append(power.tolist())
                powers = np.array(powers)
                avg_powers = np.nanmean(powers, axis=0)
                shuffle_peak_power = np.nanmax(avg_powers)

                peak_powers_from_a_single_rolling_window.append(shuffle_peak_power)

            percentile = np.nanpercentile(np.array(peak_powers_from_a_single_rolling_window), 99)
            rolling_thresholds.append(percentile)
        else:
            rolling_thresholds.append(np.nan)

    spike_data["rolling_threshold"] = rolling_thresholds
    return spike_data

def process_recordings(vr_recording_path_list):
    #vr_recording_path_list.sort()
    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            shuffle_path = recording+"/MountainSort/DataFrames/shuffles/"
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            spike_data = add_rolling_threshold(spike_data, shuffle_path=shuffle_path)
            spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # give a path for a directory of recordings or path of a single recording
    vr_recording_path_list = []
    vr_recording_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M13_D29_2021-06-17_11-50-37']
    vr_recording_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36']
    #, '/mnt/datastore/Harry/cohort8_may2021/vr/M11_D17_2021-06-01_10-36-53', '/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D44_2021-07-08_12-03-21']
    #vr_recording_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    #vr_recording_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    #vr_recording_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    process_recordings(vr_recording_path_list)

if __name__ == '__main__':
    main()