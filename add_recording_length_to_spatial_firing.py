import pandas as pd
import os
import open_ephys_IO
import PostSorting.load_firing_data
import numpy as np
import sys
sys.setrecursionlimit(100000)

def add_recording_length_to_spatial_firing(recording_to_process):
    path_to_spatial_firing = recording_to_process+"/MountainSort/DataFrames/spatial_firing.pkl"
    if os.path.exists(path_to_spatial_firing):
        spatial_firing = pd.read_pickle(path_to_spatial_firing)

        print(path_to_spatial_firing)
        print(len(spatial_firing))
        try:

            if not "recording_length_sampling_points" in list(spatial_firing):
                recording_length_sampling_points = len(open_ephys_IO.get_data_continuous(recording_to_process+"/"+PostSorting.load_firing_data.get_available_ephys_channels(recording_to_process)[0])) # needed for shuffling

                if len(spatial_firing)>0:
                    spatial_firing["recording_length_sampling_points"] = np.repeat(recording_length_sampling_points, len(spatial_firing)).tolist()
                    spatial_firing.to_pickle(recording_to_process+"/MountainSort/DataFrames/spatial_firing.pkl")
        except:
            print("stop here")

def process_recordings(recording_list):
    for recording in recording_list:
        add_recording_length_to_spatial_firing(recording)
    print("all recordings processed")

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # get list of all recordings in the recordings folder
    recording_list = []
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort8_may2021/of/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort8_may2021/vr/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/of/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort6_july2020/of/") if f.is_dir()])
    #recording_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort6_july2020/vr/") if f.is_dir()])

    #process_recordings(recording_list)
    print("look now")

if __name__ == '__main__':
    main()