import pandas as pd
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
import sys
import settings

prm = pt.Parameters()

def recompute_scores(spike_data, synced_spatial_data, recompute_speed_score=False, recompute_hd_score=False,
                     recompute_grid_score=False, recompute_spatial_score=False, recompute_border_score=False, recompute_stability_score=False):
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

def process_dir(recording_list, recompute_speed_score=True, recompute_hd_score=True, recompute_grid_score=True,
                recompute_spatial_score=True, recompute_border_score=True, recompute_stability_score=True):

    # loop over recordings and add spatial firing to the concatenated frame, add the paths to processed position
    for recording in recording_list:
        try:
            print("processeding ", recording.split("/")[-1])

            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            if "occupancy_maps" in list(spike_data):
                print("yesssssssssssssssssss")
            else:
                synced_spatial_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position.pkl")
                spike_data = recompute_scores(spike_data, synced_spatial_data, recompute_speed_score=recompute_speed_score,
                                              recompute_hd_score=recompute_hd_score, recompute_grid_score=recompute_grid_score,
                                              recompute_spatial_score=recompute_spatial_score, recompute_border_score=recompute_border_score,
                                              recompute_stability_score=recompute_stability_score)
                spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

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
    of_path_list=[]
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    of_path_list.sort()

    process_dir(recording_list=of_path_list,
                recompute_speed_score=False, recompute_hd_score=False, recompute_grid_score=False,
                recompute_spatial_score=False, recompute_border_score=False, recompute_stability_score=False)
    print("were done for now ")

if __name__ == '__main__':
    main()
