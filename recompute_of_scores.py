import pandas as pd
import numpy as np
import os
import PostSorting.open_field_spatial_firing
import PostSorting.speed
import PostSorting.open_field_firing_maps
import PostSorting.open_field_head_direction
import PostSorting.open_field_grid_cells
import PostSorting.open_field_border_cells
import PostSorting.open_field_firing_fields
import PostSorting.compare_first_and_second_half
import os
import traceback
import PostSorting.parameters as pt
import warnings
import sys
import settings

prm = pt.Parameters()

def recompute_scores(spike_data, synced_spatial_data, recompute_speed_score=True, recompute_hd_score=True,
                     recompute_grid_score=True, recompute_spatial_score=True, recompute_border_score=True, recompute_stability_score=True):
    spike_data = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
    position_heatmap, spike_data = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data)
    if recompute_speed_score:
        spike_data = PostSorting.speed.calculate_speed_score(synced_spatial_data, spike_data, settings.gauss_sd_for_speed_score, settings.sampling_rate)
    if recompute_hd_score:
        _, spike_data = PostSorting.open_field_head_direction.process_hd_data(spike_data, synced_spatial_data)
    if recompute_grid_score:
        spike_data = PostSorting.open_field_grid_cells.process_grid_data(spike_data)
    if recompute_spatial_score:
        spike_data = PostSorting.open_field_firing_maps.calculate_spatial_information(spike_data, position_heatmap)
    if recompute_border_score:
        spike_data = PostSorting.open_field_border_cells.process_border_data(spike_data)
    if recompute_stability_score:
        spike_data, _, _, _, _, = PostSorting.compare_first_and_second_half.analyse_half_session_rate_maps(synced_spatial_data, spike_data)
    return spike_data

def process_dir(recording_folder_path, recompute_speed_score=True, recompute_hd_score=True, recompute_grid_score=True,
                recompute_spatial_score=True, recompute_border_score=True, recompute_stability_score=True):
    # get list of all recordings in the recordings folder
    recording_list = [f.path for f in os.scandir(recording_folder_path) if f.is_dir()]
    recording_list = ["/mnt/datastore/Harry/Cohort8_may2021/of/M11_D40_2021-07-02_12-11-42"]

    # loop over recordings and add spatial firing to the concatenated frame, add the paths to processed position
    for recording in recording_list:
        try:
            print("processeding ", recording.split("/")[-1])

            spike_data_spatial = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            synced_spatial_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position.pkl")
            spike_data_spatial = recompute_scores(spike_data_spatial, synced_spatial_data, recompute_speed_score=recompute_speed_score,
                                                  recompute_hd_score=recompute_hd_score, recompute_grid_score=recompute_grid_score,
                                                  recompute_spatial_score=recompute_spatial_score, recompute_border_score=recompute_border_score,
                                                  recompute_stability_score=recompute_stability_score)
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

    #process_dir(recording_folder_path= "/mnt/datastore/Harry/cohort7_october2020/of",
    #            recompute_speed_score=False, recompute_hd_score=True, recompute_grid_score=False,
    #            recompute_spatial_score=False, recompute_border_score=False, recompute_stability_score=False)
    process_dir(recording_folder_path= "/mnt/datastore/Harry/cohort8_may2021/of",
                recompute_speed_score=False, recompute_hd_score=True, recompute_grid_score=False,
                recompute_spatial_score=False, recompute_border_score=False, recompute_stability_score=False)
    print("were done for now ")

if __name__ == '__main__':
    main()
