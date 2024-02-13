import pandas as pd
import os
import numpy as np
import time
from scipy.signal import find_peaks
import Edmond.eLife_Grid_anchoring_2024.analysis_settings as Settings
from astropy.timeseries import LombScargle
from astropy.convolution import convolve, Gaussian1DKernel


def fill_rate_map(firing_rate_map_by_trial, peaks, field_array, peak_fill_order):
    fr_original = firing_rate_map_by_trial.flatten()

    # replace nans in rate map if there is any
    fr_original[np.isnan(fr_original)] = 0

    # make empty rate map with nan placemarkers
    fr = np.zeros(len(fr_original)); fr[:] = np.nan

    # fill the rate map as in the style of a field shuffle
    bin_indices_used_in_fr = []
    bin_indices_used_in_fr_original = []
    for field_index in peak_fill_order:
        peak_i = peaks[field_index-1]
        field_left_i  = np.where((field_array==field_index)==True)[0][0]; field_size_left = peak_i-field_left_i
        field_right_i = np.where((field_array==field_index)==True)[0][-1]; field_size_right = field_right_i-peak_i

        # randomly place peak bin
        peak_bin_not_filled = True
        while peak_bin_not_filled:
            random_peak_bin_index = np.random.randint(low=0, high=len(fr))
            if np.isnan(fr[random_peak_bin_index]):
                fr[random_peak_bin_index] = fr_original[peak_i]
                bin_indices_used_in_fr.append(random_peak_bin_index)
                bin_indices_used_in_fr_original.append(peak_i)
                peak_bin_not_filled = False

        #walk left and assign field bins
        for field_j, j in enumerate(np.flip(np.arange(random_peak_bin_index-field_size_left, random_peak_bin_index))):
            field_bin_not_filled = True
            while field_bin_not_filled:
                if j < 0:
                    j += len(fr)
                if np.isnan(fr[j]):
                    fr[j] = fr_original[peak_i-field_j-1]
                    bin_indices_used_in_fr.append(j)
                    bin_indices_used_in_fr_original.append(peak_i-field_j-1)
                    field_bin_not_filled = False
                else:
                    j-=1

        # walk right and assign field bins
        for field_j, j in enumerate(np.arange(random_peak_bin_index+1, random_peak_bin_index+field_size_right+1)):
            field_bin_not_filled = True
            while field_bin_not_filled:
                if j >= len(fr):
                    j -= len(fr)
                if np.isnan(fr[j]):
                    fr[j] = fr_original[peak_i+field_j+1]
                    if j <0:
                        print("stop here")
                    bin_indices_used_in_fr.append(j)
                    bin_indices_used_in_fr_original.append(peak_i+field_j+1)
                    field_bin_not_filled = False
                else:
                    j+=1

    # assign the bins randomly not attributed to a field
    indices_not_assigned_fr = np.array(list((set(bin_indices_used_in_fr) | set(np.arange(0,len(fr)))) - (set(bin_indices_used_in_fr) & set(np.arange(0,len(fr))))))
    indices_not_assigned_fr_original = np.array(list((set(bin_indices_used_in_fr_original) | set(np.arange(0,len(fr)))) - (set(bin_indices_used_in_fr_original) & set(np.arange(0,len(fr))))))
    np.random.shuffle(indices_not_assigned_fr_original)
    for i, fr_i in enumerate(indices_not_assigned_fr):
        fr[fr_i] = fr_original[indices_not_assigned_fr_original[i]]

    return fr


def make_field_array(firing_rate_map_by_trial, peaks_indices):
    field_array = np.zeros(len(firing_rate_map_by_trial))
    for i in range(len(peaks_indices)):
        field_array[peaks_indices[i][0]:peaks_indices[i][1]] = i+1
    return field_array.astype(np.int64)


def get_peak_indices(firing_rate_map, peaks_i):
    peak_indices =[]
    for j in range(len(peaks_i)):
        peak_index_tuple = find_neighbouring_minima(firing_rate_map, peaks_i[j])
        peak_indices.append(peak_index_tuple)
    return peak_indices


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

    return (local_min_left, local_min_right)


def field_shuffle_and_get_false_alarm_rate(firing_rate_map_by_trial, gauss_kernel_std=Settings.rate_map_gauss_kernel_std,
                                           extra_smooth_gauss_kernel_std= Settings.rate_map_extra_smooth_gauss_kernel_std,
                                           peak_min_distance=Settings.minimum_peak_distance, compute_ls=True):

    firing_rate_map_by_trial_flattened = firing_rate_map_by_trial.flatten()
    gauss_kernel_extra = Gaussian1DKernel(stddev=extra_smooth_gauss_kernel_std)
    gauss_kernel = Gaussian1DKernel(stddev=gauss_kernel_std)
    firing_rate_map_by_trial_flattened_extra_smooth = convolve(firing_rate_map_by_trial_flattened, gauss_kernel_extra)

    track_length = len(firing_rate_map_by_trial[0])
    n_trials = len(firing_rate_map_by_trial)
    elapsed_distance_bins = np.arange(0, (track_length*n_trials)+1, 1)
    elapsed_distance = 0.5*(elapsed_distance_bins[1:]+elapsed_distance_bins[:-1])/track_length
    frequency = Settings.frequency
    sliding_window_size=track_length*Settings.window_length_in_laps
    indices_to_test = np.arange(0, len(elapsed_distance)-sliding_window_size, 1, dtype=np.int64)[::Settings.power_estimate_step]

    # find peaks and trough indices
    peaks_i = find_peaks(firing_rate_map_by_trial_flattened_extra_smooth, distance=peak_min_distance)[0]
    peaks_indices = get_peak_indices(firing_rate_map_by_trial_flattened_extra_smooth, peaks_i)
    field_array = make_field_array(firing_rate_map_by_trial_flattened, peaks_indices)

    peak_fill_order = np.arange(1, len(peaks_i)+1)
    np.random.shuffle(peak_fill_order) # randomise fill order
    fr = fill_rate_map(firing_rate_map_by_trial, peaks_i, field_array, peak_fill_order)
    fr_smoothed = convolve(fr, gauss_kernel)

    if compute_ls:
        powers = []
        rolling_peak_powers = []
        rolling_peak_sizes = []
        for m in indices_to_test:
            ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr_smoothed[m:m+sliding_window_size])
            power = ls.power(frequency)
            powers.append(power.tolist())

            # calculate rolling_peak powers and sizes
            if len(powers) % 100 == 0:
                rolling_powers = np.array(powers)
                avg_rolling_powers = np.nanmean(rolling_powers, axis=0)
                shuffle_peak_rolling_power = np.nanmax(avg_rolling_powers)
                rolling_peak_powers.append(shuffle_peak_rolling_power)
                rolling_peak_sizes.append(len(powers))
        rolling_peak_powers = np.array(rolling_peak_powers)
        rolling_peak_sizes = np.array(rolling_peak_sizes)

        powers = np.array(powers)
        avg_powers = np.nanmean(powers, axis=0)
        shuffle_peak_power = np.nanmax(avg_powers)
        return shuffle_peak_power, rolling_peak_powers, rolling_peak_sizes, np.reshape(fr, (n_trials, track_length)), np.reshape(fr_smoothed, (n_trials, track_length))
    else:
        return [], [], [], np.reshape(fr, (n_trials, track_length)), np.reshape(fr_smoothed, (n_trials, track_length))

def run_shuffle(cluster_spike_data):
    firing_rate_map = np.array(cluster_spike_data["fr_binned_in_space"].iloc[0])
    shuffle_peak_power, rolling_peak_powers, rolling_peak_sizes, shuffled_rate_map, _ = field_shuffle_and_get_false_alarm_rate(firing_rate_map)
    cluster_spike_data["peak_power"] = [shuffle_peak_power]
    cluster_spike_data["rolling_peak_powers"] = [rolling_peak_powers]
    cluster_spike_data["rolling_peak_sizes"] = [rolling_peak_sizes]
    cluster_spike_data["shuffled_rate_map"] = [shuffled_rate_map]
    single_shuffle = cluster_spike_data[["cluster_id", "peak_power", "rolling_peak_powers", "rolling_peak_sizes", "shuffled_rate_map"]]
    return single_shuffle


def one_job_shuffle_parallel(recording_path, cluster_id, n_shuffles):
    '''
    creates a single shuffle of each cell and saves it in recording/sorter/dataframes/shuffles/
    :param recording_path: path to a recording folder
    :param shuffle_id: integer id of a single shuffle
    '''
    time0 = time.time()
    checkpoint_interval = 30*60 # in seconds
    checkpoint_counter = 1

    spike_data_spatial = pd.read_pickle(recording_path+"/MountainSort/DataFrames/spatial_firing.pkl")
    cluster_spike_data = spike_data_spatial[(spike_data_spatial["cluster_id"] == cluster_id)]

    if os.path.isfile(recording_path + "/MountainSort/DataFrames/shuffles/"+str(int(cluster_id))+"_shuffle.pkl"):
        shuffle = pd.read_pickle(recording_path + "/MountainSort/DataFrames/shuffles/"+str(int(cluster_id))+"_shuffle.pkl")
        n_shuffles_pre_computed = len(shuffle)
    else:
        shuffle = pd.DataFrame()
        n_shuffles_pre_computed = 0

    shuffles_to_run = n_shuffles-n_shuffles_pre_computed

    if shuffles_to_run > 1:
        for i in range(shuffles_to_run):
            if len(cluster_spike_data["firing_times"]) == 0:
                shuffled_cluster_spike_data = pd.DataFrame(np.nan, index=[0], columns=["cluster_id", "peak_power"])
            else:
                shuffled_cluster_spike_data = run_shuffle(cluster_spike_data)

            shuffle = pd.concat([shuffle, shuffled_cluster_spike_data], ignore_index=True)
            print(i, " shuffle complete")

            time_elapsed = time.time()-time0

            if time_elapsed > (checkpoint_interval*checkpoint_counter):
                checkpoint_counter += 1
                checkpoint(shuffle, cluster_id, recording_path)

        checkpoint(shuffle, cluster_id, recording_path)
    print("shuffle analysis completed for ", recording_path)
    return

def checkpoint(shuffle, cluster_id, recording_path):
    if not os.path.exists(recording_path+"/MountainSort/DataFrames/shuffles"):
        os.mkdir(recording_path+"/MountainSort/DataFrames/shuffles")
    shuffle.to_pickle(recording_path + "/MountainSort/DataFrames/shuffles/"+str(int(cluster_id))+"_shuffle.pkl")
    print("checkpoint saved")


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #========================FOR RUNNING ON FROM TERMINAL=====================================#
    #=========================================================================================#
    #recording_path = os.environ['RECORDING_PATH']
    #n_shuffles = int(os.environ['SHUFFLE_NUMBER'])
    #cluster_id = int(os.environ["CLUSTER_ID"])
    #one_job_shuffle_parallel(recording_path, cluster_id, n_shuffles)
    #=========================================================================================#
    #=========================================================================================#


    #================FOR RUNNING ON ELEANOR (SINGLE RECORDING)================================#
    #=========================================================================================#

    recording_path = '/mnt/datastore/Harry/cohort8_may2021/vr/M10_D10_2021-05-21_09-05-27'
    spatial_firing = pd.read_pickle(recording_path+"/MountainSort/DataFrames/spatial_firing.pkl")
    for cluster_id in spatial_firing["cluster_id"]:
        one_job_shuffle_parallel(recording_path, cluster_id, n_shuffles=1000)

    #=========================================================================================#
    #=========================================================================================#
if __name__ == '__main__':
    main()