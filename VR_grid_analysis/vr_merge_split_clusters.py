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
from PostSorting.vr_spatial_firing import bin_fr_in_space, bin_fr_in_time, add_position_x
from scipy import stats
import Edmond.VR_grid_analysis.analysis_settings as Settings
from scipy import signal
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
import os
import traceback
from astropy.nddata import block_reduce
import warnings
import matplotlib.ticker as ticker
import sys
import Edmond.plot_utility2
import Edmond.VR_grid_analysis.hit_miss_try_firing_analysis
import settings
from scipy import stats
import matplotlib.pylab as plt
import matplotlib as mpl
import control_sorting_analysis
import PostSorting.post_process_sorted_data_vr
from astropy.timeseries import LombScargle
from Edmond.utility_functions.array_manipulations import *
from joblib import Parallel, delayed
import multiprocessing
import open_ephys_IO
warnings.filterwarnings('ignore')
from scipy.stats.stats import pearsonr
from scipy.stats import shapiro
plt.rc('axes', linewidth=3)

def merge_clusters(spike_data, matched_cluster_ids):
    for first_id, second_id in matched_cluster_ids:
        tetrode1 = spike_data[spike_data["cluster_id"] == first_id]["tetrode"].iloc[0]
        tetrode2 = spike_data[spike_data["cluster_id"] == second_id]["tetrode"].iloc[0]
        if tetrode1 != tetrode2:
            print("I am about to merge clusters from different tetrodes, are you sure about that?")

        firing_times_1 = spike_data[spike_data["cluster_id"] == first_id]["firing_times"].iloc[0]
        firing_times_2 = spike_data[spike_data["cluster_id"] == second_id]["firing_times"].iloc[0]

        concatenated_firing_times = np.sort(np.concatenate((firing_times_1, firing_times_2)))
        # reassign these firing times to the firing
        index = spike_data.loc[spike_data['cluster_id'] == first_id].index[0]
        spike_data.at[index,"firing_times"] = concatenated_firing_times
        spike_data = spike_data[spike_data["cluster_id"] != second_id]
    return spike_data

def find_paired_recording(recording_path, of_recording_path_list):
    mouse=recording_path.split("/")[-1].split("_")[0]
    training_day=recording_path.split("/")[-1].split("_")[1]

    for paired_recording in of_recording_path_list:
        paired_mouse=paired_recording.split("/")[-1].split("_")[0]
        paired_training_day=paired_recording.split("/")[-1].split("_")[1]

        if (mouse == paired_mouse) and (training_day == paired_training_day):
            return paired_recording, True
    return None, False

def process_recordings(vr_recording_path_list, of_recording_path_list, matched_cluster_ids_list):
    #vr_recording_path_list.sort()

    for recording, matched_cluster_ids in zip(vr_recording_path_list, matched_cluster_ids_list):
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            if paired_recording is not None:
                of_spike_data = pd.read_pickle(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl")
                spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
                of_spike_data = merge_clusters(of_spike_data, matched_cluster_ids=matched_cluster_ids)
                spike_data = merge_clusters(spike_data, matched_cluster_ids=matched_cluster_ids)

                of_spike_data.to_pickle(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl")
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
    vr_path_list = []
    of_path_list = []

    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    #of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort9_Junji/of") if f.is_dir()])

    #vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D17_2021-06-01_10-36-53',
    #                '/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D44_2021-07-08_12-03-21']
    #matched_cluster_ids_list = [[[9, 15]],
    #                            [[42, 40]]]

    #process_recordings(vr_path_list, of_path_list, matched_cluster_ids_list)


    print("look now")

if __name__ == '__main__':
    main()
