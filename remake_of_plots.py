import pandas as pd
import numpy as np
import os
import Edmond.recompute_of_scores
import PostSorting.compare_first_and_second_half
import PostSorting.curation
import PostSorting.lfp
import PostSorting.load_firing_data
import PostSorting.load_snippet_data
import PostSorting.make_opto_plots
import PostSorting.make_plots
import PostSorting.open_field_border_cells
import PostSorting.open_field_firing_fields
import PostSorting.open_field_firing_maps
import PostSorting.open_field_grid_cells
import PostSorting.open_field_head_direction
import PostSorting.open_field_light_data
import PostSorting.open_field_make_plots
import PostSorting.open_field_spatial_data
import PostSorting.open_field_spatial_firing
import PostSorting.open_field_sync_data
import PostSorting.parameters
import PostSorting.speed
import PostSorting.temporal_firing
import PostSorting.theta_modulation
import PostSorting.load_snippet_data_opto
# import PostSorting.waveforms_pca
import PreClustering.dead_channels
import os
import traceback
import warnings
import sys
import settings

prm = PostSorting.parameters.Parameters()


def remake_plots(position_data, spatial_firing, position_heat_map, hd_histogram, output_path, prm,
                 plot_waveform=False, plot_spike_histogram=False, plot_firing_rate_vs_speed=False, plot_speed_vs_firing_rate=False,
                 plot_autocorrelograms=False, plot_spikes_on_trajectory=False, plot_coverage=False, plot_firing_rate_maps=False,
                 plot_rate_map_autocorrelogram=False, plot_hd=False, plot_polar_head_direction_histogram=False, plot_hd_for_firing_fields=False,
                 plot_spikes_on_firing_fields=False, make_optogenetics_plots=False, make_combined_figure=False):
    if plot_waveform:
        PostSorting.make_plots.plot_waveforms(spatial_firing, output_path)
    if plot_spike_histogram:
        PostSorting.make_plots.plot_spike_histogram(spatial_firing, output_path)
    if plot_firing_rate_vs_speed:
        PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, prm)
    if plot_speed_vs_firing_rate:
        PostSorting.make_plots.plot_speed_vs_firing_rate(position_data, spatial_firing, prm.get_sampling_rate(), 250, prm)
    if plot_autocorrelograms:
        PostSorting.make_plots.plot_autocorrelograms(spatial_firing, output_path)
    if plot_spikes_on_trajectory:
        PostSorting.open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, prm)
    if plot_coverage:
        PostSorting.open_field_make_plots.plot_coverage(position_heat_map, prm)
    if plot_firing_rate_maps:
        PostSorting.open_field_make_plots.plot_firing_rate_maps(spatial_firing, prm)
    if plot_rate_map_autocorrelogram:
        PostSorting.open_field_make_plots.plot_rate_map_autocorrelogram(spatial_firing, prm)
    if plot_hd:
        PostSorting.open_field_make_plots.plot_hd(spatial_firing, position_data, prm)
    if plot_polar_head_direction_histogram:
        PostSorting.open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, prm)
    if plot_hd_for_firing_fields:
        PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position_data, prm)
    if plot_spikes_on_firing_fields:
        PostSorting.open_field_make_plots.plot_spikes_on_firing_fields(spatial_firing, prm)
    if make_optogenetics_plots:
        PostSorting.make_opto_plots.make_optogenetics_plots(spatial_firing, prm.get_output_path(), prm.get_sampling_rate())
    if make_combined_figure:
        PostSorting.open_field_make_plots.make_combined_figure(prm, spatial_firing)
    return

def process_recordings(of_recording_path_list):
    # loop over recordings and add spatial firing to the concatenated frame, add the paths to processed position
    for recording in of_recording_path_list:
        try:
            print("processeding ", recording.split("/")[-1])
            output_path = recording+'/'+settings.sorterName
            prm.set_output_path(recording + "/MountainSort")

            spatial_firing = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position.pkl")
            position_heat_map = np.load(recording+"/MountainSort/DataFrames/position_heat_map.npy")
            hd_histogram = np.load(recording+"/MountainSort/DataFrames/hd_histogram.npy")
            spatial_firing = Edmond.recompute_of_scores.recompute_scores(spatial_firing, position_data)
            remake_plots(position_data, spatial_firing, position_heat_map, hd_histogram, output_path, prm,
                         plot_firing_rate_maps=True)

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

    # give a path for a directory of recordings or path of a single recording
    of_recording_path_list = []
    of_recording_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    of_recording_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    of_recording_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    of_recording_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort9_Junji/of") if f.is_dir()])

    grid_cells = pd.read_csv("/mnt/datastore/Harry/Vr_grid_cells/grid_cells.csv")
    of_recording_path_list = np.unique(grid_cells["full_session_id_of"]).tolist()
    process_recordings(of_recording_path_list)

    print("were done for now ")

if __name__ == '__main__':
    main()
