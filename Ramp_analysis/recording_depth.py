import glob
import numpy as np
import os
import pandas as pd


def add_depth_to_data_frame(main_folder_on_server: str, data_frame: pd.DataFrame, depth_file_name='depth.txt'):
    """
    Recording depth (mm) is saved in the same folder as the raw ephys data and indicates how much the tetrodes were
    lowered relative to the brain surface (approx 1.6mm - 2.7mm).
    :param main_folder_on_server: path to the folder that contains recording folders with raw data
    :param data_frame: pandas data frame to add new column to where each row is a cell
    :param depth_file_name: name of metadata file that has the recording depth for the recording
    :return: pandas data frame with a new column that has the depth
    """
    recording_depths = []
    if 'recording_depth' not in data_frame:
        print('I will add the recording depth to the data frame.')
        for cell_index, cell in data_frame.iterrows():
            depth = np.nan
            depth_path = main_folder_on_server + cell.session_id + '/' + depth_file_name
            if os.path.isfile(depth_path):
                depth = float(np.loadtxt(depth_path))
            recording_depths.append(depth)
        data_frame['recording_depth'] = recording_depths
    return data_frame
