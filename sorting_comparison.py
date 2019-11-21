import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
'''

This script will compare the sorted results between single sorted recordings compared to recordings sorted together.

'''

def correlation(first_recording_spatial_firing_path, second_recording_spatial_firing_path):

    if os.path.exists(first_recording_spatial_firing_path):
        first_recording_spatial_firing = pd.read_pickle(first_recording_spatial_firing_path)
    if os.path.exists(second_recording_spatial_firing_path):
        second_recording_spatial_firing = pd.read_pickle(second_recording_spatial_firing_path)

    # for vr consider correlating "same cell" 'x_position_cm' firing
    # for of consider

    print("hello there")


def main():

    print('-------------------------------------------------------------')

    server_path = 'Z:\ActiveProjects\Harry\Recordings_waveform_matching'
    dataframe_subpath = "\MountainSort\DataFrames\spatial_firing.pkl"

    vr_sorted_together = server_path+"\M2_D3_2019-03-06_13-35-15"+dataframe_subpath
    of_sorted_together = server_path+"\M2_D3_2019-03-06_15-24-38"+dataframe_subpath

    vr_sorted_seperate = server_path+"\M2_D3_2019-03-06_13-35-15vrsingle"+dataframe_subpath
    of_sorted_seperate = server_path+"\M2_D3_2019-03-06_15-24-38ofsingle"+dataframe_subpath

    correlation(vr_sorted_together, vr_sorted_seperate)
    #correlation(of_sorted_together, of_sorted_seperate)


    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()