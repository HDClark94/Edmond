import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_matched(path_to_match, path_list):
    recording1 = path_to_match.split("/")[-1]
    recording1_id = recording1.split("-")[0]

    for i in range(len(path_list)):
        recording2 = path_list[i].split("/")[-1]
        recording2_id = recording2.split("-")[0]
        if recording1_id == recording2_id:
            return path_list[i]

    print("couldn't find a matched recording for "+ path_to_match)
    return ""




def plot_combined_vr_of(vr_paths, of_paths, save_path):

    for i in range(len(vr_paths)):
        vr_path = vr_paths[i]
        of_path = get_matched(vr_path, of_paths)

        plt.close('all')
        figures_vr_path = vr_path+"/MountainSort/Figures/spike_trajectories/"
        figures_vr_fields_path = vr_path+"/MountainSort/Figures/field_trajectories/"
        figures_of_path = of_path+"/MountainSort/Figures/rate_maps/"
        figures_vr_rate_maps_trials_path =  vr_path+"/MountainSort/Figures/firing_rate_maps_trials/"

        vr_spatial_firing_path = vr_path+"/MountainSort/DataFrames/spatial_firing.pkl"
        of_spatial_firing_path = of_path+"/MountainSort/DataFrames/spatial_firing.pkl"

        if (of_path is not "") and os.path.exists(vr_spatial_firing_path) and os.path.exists(of_spatial_firing_path):

            vr_spatial_firing = pd.read_pickle(vr_spatial_firing_path)
            of_spatial_firing = pd.read_pickle(of_spatial_firing_path)

            if len(of_spatial_firing) == len(vr_spatial_firing):

                for cluster_index, cluster_id in enumerate(vr_spatial_firing.cluster_id):
                    vr_cluster_df = vr_spatial_firing[(vr_spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
                    of_cluster_df = of_spatial_firing[(of_spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

                    of_rate_map_path = figures_of_path + of_cluster_df['session_id'].iloc[0] + '_rate_map_' + str(cluster_id) + '.png'
                    vr_spike_traj_path = figures_vr_path + vr_cluster_df['session_id'].iloc[0] + '_track_firing_Cluster_' + str(cluster_id) + '.png'
                    vr_field_traj_path = figures_vr_fields_path + vr_cluster_df['session_id'].iloc[0] + '_track_fields_Cluster_' + str(cluster_id) + '.png'
                    vr_rate_maps_trials_path =  figures_vr_rate_maps_trials_path  + vr_cluster_df['session_id'].iloc[0] + '_firing_rate_map_trials_' + str(cluster_id) + '.png'

                    number_of_rows = 1
                    number_of_columns = 3

                    grid = plt.GridSpec(number_of_rows, number_of_columns, wspace=0.2, hspace=0.2)

                    #if os.path.exists(vr_spike_traj_path):
                    #    spike_traj = mpimg.imread(vr_spike_traj_path)
                    #    spike_traj_plot = plt.subplot(grid[0, 0])
                    #    spike_traj_plot.axis('off')
                    #    spike_traj_plot.imshow(spike_traj)
                    #    spike_traj_plot.set_title(vr_cluster_df['session_id'].iloc[0], fontsize=5)

                    if os.path.exists(vr_rate_maps_trials_path):
                        vr_rate_maps_trials = mpimg.imread(vr_rate_maps_trials_path)
                        vr_rate_maps_trials_plot = plt.subplot(grid[0, 0])
                        vr_rate_maps_trials_plot.axis('off')
                        vr_rate_maps_trials_plot.imshow(vr_rate_maps_trials)
                        vr_rate_maps_trials_plot.set_title(vr_cluster_df['session_id'].iloc[0], fontsize=5)

                    if os.path.exists(vr_field_traj_path):
                        vr_field_traj = mpimg.imread(vr_field_traj_path)
                        vr_field_traj_plot = plt.subplot(grid[0, 1])
                        vr_field_traj_plot.axis('off')
                        vr_field_traj_plot.imshow(vr_field_traj)
                        vr_field_traj_plot.set_title(vr_cluster_df['session_id'].iloc[0], fontsize=5)

                    if os.path.exists(of_rate_map_path):
                        of_rate_map = mpimg.imread(of_rate_map_path)
                        of_rate_map_plot = plt.subplot(grid[0, 2])
                        of_rate_map_plot.axis('off')
                        of_rate_map_plot.imshow(of_rate_map)
                        of_rate_map_plot.set_title(of_cluster_df['session_id'].iloc[0], fontsize=5)

                    plt.savefig(save_path + '/' + vr_cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '.png', dpi=300)
                    plt.close()







def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    save_path = "/mnt/datastore/Harry/Cohort7_october2020/combined"
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/of") if f.is_dir()]
    plot_combined_vr_of(vr_path_list, of_path_list, save_path)


    '''
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/combined"
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField") if f.is_dir()]
    plot_combined_vr_of(vr_path_list, of_path_list, save_path)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField") if f.is_dir()]
    plot_combined_vr_of(vr_path_list, of_path_list, save_path)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M2_sorted") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild") if f.is_dir()]
    plot_combined_vr_of(vr_path_list, of_path_list, save_path)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M3_sorted") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild") if f.is_dir()]
    plot_combined_vr_of(vr_path_list, of_path_list, save_path)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M1_sorted") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild/M1_sorted") if f.is_dir()]
    plot_combined_vr_of(vr_path_list, of_path_list, save_path)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M6_sorted") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild/M6_sorted") if f.is_dir()]
    plot_combined_vr_of(vr_path_list, of_path_list, save_path)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/245_sorted") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenField/245_sorted") if f.is_dir()]
    plot_combined_vr_of(vr_path_list, of_path_list, save_path)
    '''

if __name__ == '__main__':
    main()
