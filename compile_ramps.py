import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import itertools
from scipy import stats
from Edmond.ramp_cells_of import *
import os
import shutil

def compile_top_scores(data, save_path, collumn_b, top_x_percent=5):
    # look at collumn a, remove nans, and remove those with too few spike,
    # then take top x percentage of these clusters based on collumn a, and copy the open field combined figure
    # and VR spike rate figure and put it in a directory
    data = filter_by_percent(data, collumn_b, top_x_percent)
    data = data[data["trial_type"] == "beaconed"]
    data.reset_index(inplace=True, drop=True)

    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        sorted_together_cluster_id = row["sorted_together_vr_cluster_ids"].iloc[0]
        sorted_seperately_cluster_id = row["sorted_seperately_vr_cluster_ids"].iloc[0]
        session_id = row["session_id"].iloc[0]

        spike_rate_fig_path = row["full_session_id"].iloc[0]+"/MountainSort/Figures/spike_rate"
        spike_traj_fig_path = row["full_session_id"].iloc[0]+"/MountainSort/Figures/spike_trajectories"
        OF_combined_fig_path = row["full_of_session_id"].iloc[0]+"/MountainSort/Figures/combined"

        try:
            spike_rate_fig_path = get_cluster_path(spike_rate_fig_path, sorted_seperately_cluster_id)
            spike_traj_fig_path = get_cluster_path(spike_traj_fig_path, sorted_seperately_cluster_id)
            OF_combined_fig_path = get_cluster_path(OF_combined_fig_path, sorted_together_cluster_id)

            # now copy these figures to the save_path
            save_figure(spike_rate_fig_path, save_path, "rank"+str(index+1)+"_"+session_id)
            save_figure(spike_traj_fig_path, save_path, "rank"+str(index+1)+"_"+session_id)
            save_figure(OF_combined_fig_path, save_path, "rank"+str(index+1)+"_"+session_id)
        except:
            print("Error occurred, file doesn't exist :( ")

    return

def save_figure(figure_path, new_save_path, session_id):
    figure_dir_path = new_save_path+"/"+session_id+"/"
    if os.path.exists(figure_dir_path) is False:
        os.makedirs(figure_dir_path)
    shutil.copy(figure_path, figure_dir_path)

def get_cluster_path(figure_path, cluster_id):
    figure_path_list = None
    figure_path_list = [f.path for f in os.scandir(figure_path)]
    if figure_path_list is not None:
        for i in range(len(figure_path_list)):
            if int(figure_path_list[i].split("_")[-1].split(".")[0]) == cluster_id:
                return figure_path_list[i]
            else:
                continue
    else:
        print("figure path wasn't found", figure_path)

def filter_by_percent(data, collumn_b, top_x_percent, n_spikes_min=1000):
    # remove those clusters with nans for spatial score
    data = data.dropna(subset=[collumn_b])
    # remove clusters that have very few spikes
    data = data[data["n_spikes_of"]>=n_spikes_min]
    # sort by collumn_b in descending order
    data = data.sort_values(by=[collumn_b], ascending=False)
    # take the top x%
    top_x_data = data.iloc[:int(len(data)*(top_x_percent/100))]
    return top_x_data

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    ramp_path =     "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_results_linearmodel.txt"
    ramp_scores_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ramp_score_export.csv"
    tetrode_location_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/tetrode_locations.csv"

    hd_save_path =  "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ranked_cells/head_direction_cells"
    b_save_path =   "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ranked_cells/border_cells"
    g_save_path =   "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ranked_cells/grid_cells"
    c_save_path =   "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ranked_cells/corner_cells"
    s_save_path =   "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ranked_cells/speed_cells"
    nss_save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ranked_cells/non_specific_spatial_cells"

    c5m1_path =     "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M1_sorting_stats.pkl"
    c5m2_path =     "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M2_sorting_stats.pkl"
    c4m2_path =     "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M2_sorting_stats.pkl"
    c4m3_path =     "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M3_sorting_stats.pkl"
    c3m1_path =     "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M1_sorting_stats.pkl"
    c3m6_path =     "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M6_sorting_stats.pkl"
    c2m245_path =   "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/245_sorting_stats.pkl"
    c2m1124_path =  "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/1124_sorting_stats.pkl"

    all_of_paths = [c5m1_path, c5m2_path, c4m2_path, c4m3_path, c3m1_path, c3m6_path, c2m245_path, c2m1124_path]
    all_of_paths = [c5m1_path, c5m2_path, c4m2_path, c4m3_path, c3m1_path, c3m6_path]
    data = concatenate_all(ramp_path, ramp_scores_path, tetrode_location_path, all_of_paths, include_unmatch=False)
    compile_top_scores(data, hd_save_path, collumn_b="hd_score", top_x_percent=20)
    compile_top_scores(data, s_save_path, collumn_b="speed_score", top_x_percent=10)
    compile_top_scores(data, b_save_path, collumn_b="border_score", top_x_percent=10)
    compile_top_scores(data, g_save_path, collumn_b="grid_score", top_x_percent=10)
    compile_top_scores(data, c_save_path, collumn_b="corner_score", top_x_percent=10)
    compile_top_scores(data, nss_save_path, collumn_b="rate_map_correlation_first_vs_second_half", top_x_percent=10)



if __name__ == '__main__':
    main()