import pandas as pd
import numpy as np
import os
import PostSorting.open_field_spatial_firing
import PostSorting.speed
import PostSorting.open_field_firing_maps
import PostSorting.open_field_head_direction
import PostSorting.open_field_grid_cells
import os
import traceback
import PostSorting.parameters as pt
import warnings
import sys
import settings

prm = pt.Parameters()

def recompute_scores(spike_data, synced_spatial_data, prm):
    prm.set_sampling_rate(30000)
    prm.set_pixel_ratio(440)

    spike_data = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
    spike_data = PostSorting.speed.calculate_speed_score(synced_spatial_data, spike_data, settings.gauss_sd_for_speed_score, settings.sampling_rate)
    _, spike_data = PostSorting.open_field_head_direction.process_hd_data(spike_data, synced_spatial_data, prm)
    position_heatmap, spike_data = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data, prm)
    spike_data = PostSorting.open_field_grid_cells.process_grid_data(spike_data)
    spike_data = PostSorting.open_field_firing_maps.calculate_spatial_information(spike_data, position_heatmap)

    return spike_data

def process_dir(recording_folder_path):

    # get list of all recordings in the recordings folder
    recording_list = [f.path for f in os.scandir(recording_folder_path) if f.is_dir()]

    # loop over recordings and add spatial firing to the concatenated frame, add the paths to processed position
    for recording in recording_list:
        try:
            print("processeding ", recording.split("/")[-1])

            spike_data_spatial = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            synced_spatial_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position.pkl")

            spike_data_spatial = recompute_scores(spike_data_spatial, synced_spatial_data, prm)

            spike_data_spatial.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    process_dir(recording_folder_path= "/mnt/datastore/Harry/cohort7_october2020/of")
    print("were done for now ")

if __name__ == '__main__':
    main()
