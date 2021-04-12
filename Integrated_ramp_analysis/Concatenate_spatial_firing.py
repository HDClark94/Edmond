import os
import glob
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

"""
This file creates a concatenated dataframe of all the recording directories passed to it and saves it where specified,
for collection of the processed position data for the vr side, there will be a link pointing to the original processed position
"""

def process_dir(recordings_path, concatenated_spike_data=None, save_path=None):

    # make an empty dataframe if concatenated frame given as none
    if concatenated_spike_data is None:
        concatenated_spike_data = pd.DataFrame()

    # get list of all recordings in the recordings folder
    recording_list = [f.path for f in os.scandir(recordings_path) if f.is_dir()]

    # loop over recordings and add spatial firing to the concatenated frame, add the paths to processed position
    for recording in recording_list:
        spatial_dataframe_path = recording + '/MountainSort/DataFrames/processed_position_data.pkl'
        spike_dataframe_path = recording + '/MountainSort/DataFrames/spatial_firing.pkl'

        if os.path.exists(spike_dataframe_path):
            spike_data = pd.read_pickle(spike_dataframe_path)
            if os.path.exists(spatial_dataframe_path):
                spike_data["processed_position_path"] = np.repeat(spatial_dataframe_path, len(spike_data))
                concatenated_spike_data = pd.concat([concatenated_spike_data, spike_data], ignore_index=True)
            else:
                print("couldn't find processed_position for ", recording)

    if save_path is not None:
        concatenated_spike_data.to_pickle(save_path+"concatenated_spike_data.pkl")
    return concatenated_spike_data


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    spike_data = process_dir(recordings_path= "/mnt/datastore/Harry/Cohort7_october2020/vr", concatenated_spike_data=None,
                             save_path= "/mnt/datastore/Harry/Ramp_cells_open_field_paper/")
    spike_data = process_dir(recordings_path= "/mnt/datastore/Harry/Cohort6_july2020/vr", concatenated_spike_data=spike_data,
                             save_path= "/mnt/datastore/Harry/Ramp_cells_open_field_paper/")

if __name__ == '__main__':
    main()





