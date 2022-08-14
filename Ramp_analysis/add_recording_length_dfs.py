import pandas as pd
import os
import OpenEphys
import numpy as np
import sys
sys.setrecursionlimit(100000)


def get_data_continuous(file_path):
    data = OpenEphys.load(file_path)
    signal = data['data']
    signal = np.asanyarray(signal)
    return signal


def get_available_ephys_channels(recording_to_process, prefix=f'_CH'):
    '''
    :param recording_to_process: absolute path of recroding to sort
    :return: list of named channels for ephys aquisition
    '''

    shared_ephys_channel_marker = prefix
    all_files_names = [f for f in os.listdir(recording_to_process) if os.path.isfile(os.path.join(recording_to_process, f))]
    all_ephys_file_names = [s for s in all_files_names if shared_ephys_channel_marker in s]

    return all_ephys_file_names


def add_recording_length_to_spatial_firing(recording_to_process):
    path_to_spatial_firing = recording_to_process+"/MountainSort/DataFrames/spatial_firing.pkl"
    if os.path.exists(path_to_spatial_firing):
        spatial_firing = pd.read_pickle(path_to_spatial_firing)

        print(path_to_spatial_firing)
        print(len(spatial_firing))
        try:

            if not "recording_length_sampling_points" in list(spatial_firing):
                recording_length_sampling_points = len(get_data_continuous(recording_to_process + "/" + get_available_ephys_channels(recording_to_process)[0])) # needed for shuffling

                if len(spatial_firing) > 0:
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