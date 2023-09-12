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


def plot_combined_vr_of(vr_paths, of_paths, save_path, condition, condition_above=None, condition_below=None):

    for i in range(len(vr_paths)):
        vr_path = vr_paths[i]
        print("processing", vr_path)
        of_path = get_matched(vr_path, of_paths)

        plt.close('all')
        figures_vr_path = vr_path+"/MountainSort/Figures/spike_trajectories/"
        figures_vr_fields_path = vr_path+"/MountainSort/Figures/spike_rate/"
        figures_of_path = of_path+"/MountainSort/Figures/rate_maps/"
        figures_vr_rate_maps_trials_path = vr_path+"/MountainSort/Figures/firing_rate_maps_trials/"
        figures_vr_spatial_autocorrelogram_path = vr_path+"/MountainSort/Figures/spatial_autocorrelograms/"
        figures_of_path_autocorrelogram_path = of_path+ "/MountainSort/Figures/rate_map_autocorrelogram/"

        vr_spatial_firing_path = vr_path+"/MountainSort/DataFrames/spatial_firing.pkl"
        of_spatial_firing_path = of_path+"/MountainSort/DataFrames/spatial_firing.pkl"

        if (of_path is not "") and os.path.exists(vr_spatial_firing_path) and os.path.exists(of_spatial_firing_path):

            vr_spatial_firing = pd.read_pickle(vr_spatial_firing_path)
            of_spatial_firing = pd.read_pickle(of_spatial_firing_path)

            if len(of_spatial_firing) == len(vr_spatial_firing):

                for cluster_index, cluster_id in enumerate(vr_spatial_firing.cluster_id):
                    vr_cluster_df = vr_spatial_firing[(vr_spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
                    of_cluster_df = of_spatial_firing[(of_spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

                    # filter by condition if one exists (only works for open field conditions)
                    if condition is not None:
                        if condition_above is not None:
                            of_cluster_df = of_cluster_df[of_cluster_df[condition] > condition_above]
                        elif condition_below is not None:
                            of_cluster_df = of_cluster_df[of_cluster_df[condition] < condition_below]

                    if len(of_cluster_df) == 1: # if it passed the condition filter or there isn't a filter being used
                        of_rate_map_path = figures_of_path + of_cluster_df['session_id'].iloc[0] + '_rate_map_' + str(cluster_id) + '.png'
                        vr_spike_traj_path = figures_vr_path + vr_cluster_df['session_id'].iloc[0] + '_track_firing_Cluster_' + str(cluster_id) + '.png'
                        vr_rate_map_path = figures_vr_fields_path + vr_cluster_df['session_id'].iloc[0] + '_rate_map_Cluster_' + str(cluster_id) + '.png'
                        vr_rate_maps_trials_path =  figures_vr_rate_maps_trials_path  + vr_cluster_df['session_id'].iloc[0] + '_firing_rate_map_trials_' + str(cluster_id) + '.png'
                        vr_spatial_autocorrelogram = figures_vr_spatial_autocorrelogram_path + vr_cluster_df['session_id'].iloc[0] + '_spatial_autocorrelogram_Cluster_' + str(cluster_id) + '.png'
                        of_rate_map_auto_path = figures_of_path_autocorrelogram_path + of_cluster_df['session_id'].iloc[0] + '_rate_map_autocorrelogram_' + str(cluster_id) + '.png'

                        number_of_rows = 1
                        number_of_columns = 6
                        grid = plt.GridSpec(number_of_rows, number_of_columns, width_ratios=[1, 1, 1, 1, 1, 1],
                                            wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)
                        #grid = plt.GridSpec(number_of_rows, number_of_columns, wspace=0.2, hspace=0.2)

                        if os.path.exists(vr_spike_traj_path):
                            spike_traj = mpimg.imread(vr_spike_traj_path)
                            #spike_traj_plot = plt.subplot(grid[0, 0])
                            spike_traj_plot = plt.subplot(1, 6, 1)
                            spike_traj_plot.axis('off')
                            spike_traj_plot.imshow(spike_traj)
                            spike_traj_plot.set_title(vr_cluster_df['session_id'].iloc[0], fontsize=5)

                        if os.path.exists(vr_rate_maps_trials_path):
                            vr_rate_maps_trials = mpimg.imread(vr_rate_maps_trials_path)
                            #vr_rate_maps_trials_plot =plt.subplot(grid[0, 0])
                            vr_rate_maps_trials_plot = plt.subplot(1, 6,2)
                            vr_rate_maps_trials_plot.axis('off')
                            vr_rate_maps_trials_plot.imshow(vr_rate_maps_trials)
                            #vr_rate_maps_trials_plot.set_title(vr_cluster_df['session_id'].iloc[0], fontsize=5)

                        if os.path.exists(vr_rate_map_path):
                            vr_rate_maps_trials = mpimg.imread(vr_rate_map_path)
                            #vr_rate_maps_trials_plot = plt.subplot(grid[0, 2])
                            vr_rate_maps_trials_plot = plt.subplot(1, 6,3)
                            vr_rate_maps_trials_plot.axis('off')
                            vr_rate_maps_trials_plot.imshow(vr_rate_maps_trials)
                            #vr_rate_maps_trials_plot.set_title(vr_cluster_df['session_id'].iloc[0], fontsize=5)

                        if os.path.exists(vr_spatial_autocorrelogram):
                            vr_field_traj = mpimg.imread(vr_spatial_autocorrelogram)
                            #vr_field_traj_plot = plt.subplot(grid[0, 3])
                            vr_field_traj_plot = plt.subplot(1, 6,4)
                            vr_field_traj_plot.axis('off')
                            vr_field_traj_plot.imshow(vr_field_traj)
                            #vr_field_traj_plot.set_title(vr_cluster_df['session_id'].iloc[0], fontsize=5)

                        if os.path.exists(of_rate_map_path):
                            of_rate_map = mpimg.imread(of_rate_map_path)
                            #of_rate_map_plot = plt.subplot(grid[0, 4])
                            of_rate_map_plot = plt.subplot(1, 6,5)
                            of_rate_map_plot.axis('off')
                            of_rate_map_plot.imshow(of_rate_map)
                            #of_rate_map_plot.set_title(of_cluster_df['session_id'].iloc[0], fontsize=5)

                        if os.path.exists(of_rate_map_auto_path):
                            of_rate_map = mpimg.imread(of_rate_map_auto_path)
                            #of_rate_map_plot = plt.subplot(grid[0, 5])
                            of_rate_map_plot = plt.subplot(1, 6,6)
                            of_rate_map_plot.axis('off')
                            of_rate_map_plot.imshow(of_rate_map)
                            #of_rate_map_plot.set_title(of_cluster_df['session_id'].iloc[0], fontsize=5)

                        #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
                        plt.tight_layout()
                        plt.savefig(save_path + '/' + vr_cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '.png', dpi=500)
                        plt.close()


def plot_snrs_by_tt(vr_paths, save_path, condition, combined_df):
    rc = {"axes.spines.left" : False,
          "axes.spines.right" : False,
          "axes.spines.bottom" : False,
          "axes.spines.top" : False,
          "xtick.bottom" : False,
          "xtick.labelbottom" : False,
          "ytick.labelleft" : False,
          "ytick.left" : False}
    plt.rcParams.update(rc)

    if condition=="G":
        combined_df = combined_df[combined_df["classifier"] == "G"]

    n_columns = 8
    n_rows = (len(combined_df)//n_columns)+1
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns*6, n_rows*6))
    fig = plt.figure(figsize=(n_columns*6, n_rows*6))

    i=0
    for index, cell in combined_df.iterrows():
        i += 1
        cell = cell.to_frame().T.reset_index(drop=True)
        session_id = cell["session_id"].iloc[0]
        cluster_id = cell["cluster_id"].iloc[0]
        vr_path = [s for s in vr_paths if session_id in s][0]
        figure_vr_path = vr_path+"/MountainSort/Figures/moving_lomb_power_by_hmt/"+"hit_tt_powers_"+str(int(cluster_id))+".png"
        im = mpimg.imread(figure_vr_path)
        fig.add_subplot(n_rows, n_columns, i)
        plt.title(session_id+"_"+str(int(cluster_id)))
        plt.imshow(im)

    plot_path = save_path + '/'+condition+'_all_power_by_tt.png'
    fig.savefig(plot_path, dpi=150)
    plt.close()

def plot_snrs_tests_by_hmt(vr_paths, save_path, condition, combined_df):
    rc = {"axes.spines.left" : False,
          "axes.spines.right" : False,
          "axes.spines.bottom" : False,
          "axes.spines.top" : False,
          "xtick.bottom" : False,
          "xtick.labelbottom" : False,
          "ytick.labelleft" : False,
          "ytick.left" : False}
    plt.rcParams.update(rc)

    if condition=="G":
        combined_df = combined_df[combined_df["classifier"] == "G"]

    n_columns = 8
    n_rows = (len(combined_df)//n_columns)+1
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns*6, n_rows*6))
    fig = plt.figure(figsize=(n_columns*6, n_rows*6))

    i=0
    for index, cell in combined_df.iterrows():
        i += 1
        cell = cell.to_frame().T.reset_index(drop=True)
        session_id = cell["session_id"].iloc[0]
        cluster_id = cell["cluster_id"].iloc[0]
        vr_path = [s for s in vr_paths if session_id in s][0]
        figure_vr_path = vr_path+"/MountainSort/Figures/moving_lomb_power_by_hmt/hmt_powers_test_"+str(int(cluster_id))+".png"
        if os.path.exists(figure_vr_path):
            im = mpimg.imread(figure_vr_path)
            fig.add_subplot(n_rows, n_columns, i)
            plt.title(session_id+"_"+str(int(cluster_id)))
            plt.imshow(im)

    plot_path = save_path + '/'+condition+'_all_snr_tests_by_hmt.png'
    fig.savefig(plot_path, dpi=150)
    plt.close()

def plot_hit_firing_rate_maps_by_tt(vr_paths, save_path, condition, combined_df):
    rc = {"axes.spines.left" : False,
          "axes.spines.right" : False,
          "axes.spines.bottom" : False,
          "axes.spines.top" : False,
          "xtick.bottom" : False,
          "xtick.labelbottom" : False,
          "ytick.labelleft" : False,
          "ytick.left" : False}
    plt.rcParams.update(rc)

    if condition=="G":
        combined_df = combined_df[combined_df["classifier"] == "G"]

    n_columns = 8
    n_rows = (len(combined_df)//n_columns)+1
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns*6, n_rows*6))
    fig = plt.figure(figsize=(n_columns*6, n_rows*6))

    i=0
    for index, cell in combined_df.iterrows():
        i += 1
        cell = cell.to_frame().T.reset_index(drop=True)
        session_id = cell["session_id"].iloc[0]
        cluster_id = cell["cluster_id"].iloc[0]
        vr_path = [s for s in vr_paths if session_id in s][0]
        figure_vr_path = vr_path+"/MountainSort/Figures/firing_rate_maps/"+session_id+"_firing_rate_map_hits_by_trial_type_"+str(int(cluster_id))+".png"
        if os.path.exists(figure_vr_path):
            im = mpimg.imread(figure_vr_path)
            fig.add_subplot(n_rows, n_columns, i)
            plt.title(session_id+"_"+str(int(cluster_id)))
            plt.imshow(im)

    plot_path = save_path + '/'+condition+'_all_hit_firing_rate_maps_by_tt.png'
    fig.savefig(plot_path, dpi=150)
    plt.close()


def plot_firing_rate_maps_by_hmt(vr_paths, save_path, condition, combined_df):
    rc = {"axes.spines.left" : False,
          "axes.spines.right" : False,
          "axes.spines.bottom" : False,
          "axes.spines.top" : False,
          "xtick.bottom" : False,
          "xtick.labelbottom" : False,
          "ytick.labelleft" : False,
          "ytick.left" : False}
    plt.rcParams.update(rc)

    if condition=="G":
        combined_df = combined_df[combined_df["classifier"] == "G"]

    n_columns = 8
    n_rows = (len(combined_df)//n_columns)+1
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns*6, n_rows*6))
    fig = plt.figure(figsize=(n_columns*6, n_rows*6))

    i=0
    for index, cell in combined_df.iterrows():
        i += 1
        cell = cell.to_frame().T.reset_index(drop=True)
        session_id = cell["session_id"].iloc[0]
        cluster_id = cell["cluster_id"].iloc[0]
        vr_path = [s for s in vr_paths if session_id in s][0]
        figure_vr_path = vr_path+"/MountainSort/Figures/firing_rate_maps/"+session_id+"_firing_rate_map_by_trial_outcome_"+str(int(cluster_id))+".png"
        if os.path.exists(figure_vr_path):
            im = mpimg.imread(figure_vr_path)
            fig.add_subplot(n_rows, n_columns, i)
            plt.title(session_id+"_"+str(int(cluster_id)))
            plt.imshow(im)

    plot_path = save_path + '/'+condition+'_all_firing_rate_maps_by_hmt.png'
    fig.savefig(plot_path, dpi=150)
    plt.close()

def plot_snrs_by_hmt(vr_paths, save_path, condition, combined_df):
    rc = {"axes.spines.left" : False,
          "axes.spines.right" : False,
          "axes.spines.bottom" : False,
          "axes.spines.top" : False,
          "xtick.bottom" : False,
          "xtick.labelbottom" : False,
          "ytick.labelleft" : False,
          "ytick.left" : False}
    plt.rcParams.update(rc)

    if condition=="G":
        combined_df = combined_df[combined_df["classifier"] == "G"]

    n_columns = 8
    n_rows = (len(combined_df)//n_columns)+1
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns*6, n_rows*6))
    fig = plt.figure(figsize=(n_columns*6, n_rows*6))

    i=0
    for index, cell in combined_df.iterrows():
        i += 1
        cell = cell.to_frame().T.reset_index(drop=True)
        session_id = cell["session_id"].iloc[0]
        cluster_id = cell["cluster_id"].iloc[0]
        vr_path = [s for s in vr_paths if session_id in s][0]
        figure_vr_path = vr_path+"/MountainSort/Figures/moving_lomb_power_by_hmt/"+"hmt_powers_"+str(int(cluster_id))+".png"
        im = mpimg.imread(figure_vr_path)
        fig.add_subplot(n_rows, n_columns, i)
        plt.title(session_id+"_"+str(int(cluster_id)))
        plt.imshow(im)

    plot_path = save_path + '/'+condition+'_all_power_by_hmt.png'
    fig.savefig(plot_path, dpi=150)
    plt.close()


def plot_fig4_combined(vr_paths, save_path, condition=None, combined_df=None):
    rc = {"axes.spines.left" : False,
          "axes.spines.right" : False,
          "axes.spines.bottom" : False,
          "axes.spines.top" : False,
          "xtick.bottom" : False,
          "xtick.labelbottom" : False,
          "ytick.labelleft" : False,
          "ytick.left" : False}
    plt.rcParams.update(rc)

    if condition is not None:
        combined_df = combined_df[combined_df["classifier"] == condition]

    for index, cell in combined_df.iterrows():
        cell = cell.to_frame().T.reset_index(drop=True)
        session_id = cell["session_id_vr"].iloc[0]
        cluster_id = cell["cluster_id"].iloc[0]
        track_length = cell["track_length"].iloc[0]

        if track_length == 200:
            vr_path = [s for s in vr_paths if session_id in s][0]
            figure_vr_path = vr_path+"/MountainSort/Figures/combined_fig4/stop_histograms_and_and_codes.png"\

            try:
                fig = plt.figure(figsize=(6, 6))
                im = mpimg.imread(figure_vr_path)
                plt.title(session_id+"_"+str(int(cluster_id)))
                plt.imshow(im)
                plot_path = save_path + '/'+condition+'_stop_histograms_and_codes_'+session_id+'_'+str(cluster_id)+'.png'
                fig.savefig(plot_path, dpi=300)
                plt.close()
            except Exception as ex:
                print('No file')
    return



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # get cells for paper
    combined_df = pd.DataFrame()
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort6.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort7.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")], ignore_index=True)
    combined_df = combined_df[combined_df["snippet_peak_to_trough"] < 500] # uV
    combined_df = combined_df[combined_df["track_length"] == 200]
    combined_df = combined_df[combined_df["n_trials"] >= 10]
    combined_df = combined_df[combined_df["mouse"] != "M2"]
    combined_df = combined_df[combined_df["mouse"] != "M4"]
    combined_df = combined_df[combined_df["mouse"] != "M15"]

    vr_path_list = []; of_path_list = []
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    of_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])

    plot_fig4_combined(vr_path_list, save_path ="/mnt/datastore/Harry/Vr_grid_cells/summary/fig4/", condition="G", combined_df=combined_df)


    combined_df = pd.read_pickle("/mnt/datastore/Harry/VR_grid_cells/combined_cohort8.pkl")
    #plot_snrs_by_hmt(vr_path_list, save_path ="/mnt/datastore/Harry/Cohort8_may2021/summary/combined_grid_cells_figures/", condition="G", combined_df=combined_df)
    #plot_snrs_by_tt(vr_path_list, save_path ="/mnt/datastore/Harry/Cohort8_may2021/summary/combined_grid_cells_figures/", condition="G", combined_df=combined_df)
    #plot_firing_rate_maps_by_hmt(vr_path_list, save_path ="/mnt/datastore/Harry/Cohort8_may2021/summary/combined_grid_cells_figures/", condition="G", combined_df=combined_df)
    plot_hit_firing_rate_maps_by_tt(vr_path_list, save_path ="/mnt/datastore/Harry/Cohort8_may2021/summary/combined_grid_cells_figures/", condition="G", combined_df=combined_df)
    plot_snrs_tests_by_hmt(vr_path_list, save_path ="/mnt/datastore/Harry/Cohort8_may2021/summary/combined_grid_cells_figures/", condition="G", combined_df=combined_df)
    #plot_combined_vr_of(vr_path_list, of_path_list,  save_path ="/mnt/datastore/Harry/Cohort8_may2021/summary/combined_grid_cells_figures", condition="grid_score", condition_above=0.2, condition_below=None)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/of") if f.is_dir()]
    #plot_combined_vr_of(vr_path_list, of_path_list,  save_path ="/mnt/datastore/Harry/Cohort7_october2020/summary/combined_grid_cells_figures", condition="grid_score", condition_above=0.2, condition_below=None)

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()]
    plot_combined_vr_of(vr_path_list, of_path_list,  save_path ="/mnt/datastore/Harry/Cohort6_july2020/summary/combined_grid_cells_figures", condition="grid_score", condition_above=0.2, condition_below=None)


if __name__ == '__main__':
    main()
