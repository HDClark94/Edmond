import pandas as pd
import numpy as np
import os
import sys
import traceback
import warnings
warnings.filterwarnings("ignore")


def add_shuffled_cutoffs(recordings_folder_to_process):

    recording_list = [f.path for f in os.scandir(recordings_folder_to_process) if f.is_dir()]

    for recording_path in recording_list:
        print("processing ", recording_path)

        shuffle=pd.DataFrame()
        if os.path.isdir(recording_path+r"/MountainSort/DataFrames/shuffles"):
            shuffle_list = [f.path for f in os.scandir(recording_path+r"/MountainSort/DataFrames/shuffles") if f.is_file()]

            for i in range(len(shuffle_list)):
                cluster_shuffle = pd.read_pickle(shuffle_list[i])
                cluster_shuffle = cluster_shuffle[["cluster_id", "peak_power"]]
                shuffle = pd.concat([shuffle, cluster_shuffle], ignore_index=False)
            print("I have found a shuffled dataframe")

            if os.path.isfile(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl"):
                spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

                if len(spatial_firing)>0:
                    print("cluster IDs in shuffle df: ", np.unique(shuffle.cluster_id))
                    print("cluster IDs in spatial df: ", np.unique(shuffle.cluster_id))

                    print("There are", len(shuffle)/len(spatial_firing), "shuffles per cell")

                    power_thresholds = []
                    power_n_nans_removed_from_shuffle = []

                    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
                        cluster_shuffle_df = shuffle[(shuffle.cluster_id == cluster_id)] # dataframe for that cluster
                        print("For cluster", cluster_id, " there are ", len(cluster_shuffle_df), " shuffles")

                        peak_powers = np.array(cluster_shuffle_df["peak_power"])
                        power_n_nans_removed_from_shuffle.append(len(cluster_shuffle_df)-np.count_nonzero(np.isnan(peak_powers)))

                        # print it out for people to see
                        print("There are this many non-nan values for the shuffle periodic power: ", power_n_nans_removed_from_shuffle[cluster_index])

                        #remove the nan values
                        peak_powers = peak_powers[~np.isnan(peak_powers)]

                        # calculate the 99th percentile threshold for individual clusters
                        adjusted_peak_threshold = np.nanpercentile(peak_powers, 99) # one tailed
                        power_thresholds.append(adjusted_peak_threshold)

                    spatial_firing["power_threshold"] = power_thresholds
                    spatial_firing["power_n_nans_removed_from_shuffle"] = power_n_nans_removed_from_shuffle


                    spatial_firing.to_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

                else:
                    print("There are no cells in this recordings")
            else:
                print("No spatial firing could be found")

#TODO use this snippet to classify the cells in this script
'''
def add_spatial_classifier_based_on_cutoffs(recordings_folder_to_process):
    recording_list = [f.path for f in os.scandir(recordings_folder_to_process) if f.is_dir()]

    for recording_path in recording_list:
        print("processing ", recording_path)
        if os.path.exists(recording_path+r"/MountainSort/DataFrames/shuffles/"):
            spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

            grid_cells = []
            for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
                cluster_spatial_firing = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

                if ((cluster_spatial_firing["grid_score"].iloc[0] > cluster_spatial_firing["grid_threshold"].iloc[0]) and
                    (cluster_spatial_firing["rate_map_correlation_first_vs_second_half"].iloc[0] > cluster_spatial_firing["half_session_threshold"].iloc[0])):
                    grid_cell = True
                else:
                    grid_cell = False
                grid_cells.append(grid_cell)

            spatial_firing["grid_cell"] = grid_cells
            spatial_firing.to_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")
'''

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('The shuffled analysis scripts (on Eddie) used python 3.8 to make the data frames. If you run this and '
          'get an error about pickle protocols, try to make a new python 3.8 virtual environment on Eleanor '
          '(conda create -n environmentname python=3.8) and use that. (The pipeline currently needs 3.6, so do not '
          'change that.')

    folders = []
    #folders.append("/mnt/datastore/Harry/Cohort9_Junji/vr")
    folders.append("/mnt/datastore/Harry/Cohort7_october2020/vr")
    folders.append("/mnt/datastore/Harry/Cohort6_july2020/vr")
    folders.append("/mnt/datastore/Harry/Cohort8_may2021/vr")

    for folder in folders:
        add_shuffled_cutoffs(folder)
        #add_spatial_classifier_based_on_cutoffs(folder)
    print("look now")

if __name__ == '__main__':
    main()