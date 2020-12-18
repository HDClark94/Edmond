import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
from scipy import stats
from scipy import signal
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import Edmond.Concatenate_from_server
from scipy import stats

import matplotlib.pyplot as plt

def check_structure_session_id(session_id):
    # looks at string of session id and returns the correct structure if extra bits are added
    # eg M1_D1_2020-08-03_16-11-14vr should be M1_D1_2020-08-03_16-11-14

    ending = session_id.split("-")[-1]
    corrected_ending = ''.join(filter(str.isdigit, ending))

    if corrected_ending == ending:
        return session_id
    else:
        return session_id.split(ending)[0]+corrected_ending

def add_session_identifiers(all_days_df):
    timestamp_list = []
    date_list = []
    mouse_list = []
    training_day_list = []

    for index, cluster_df in all_days_df.iterrows():
        session_id = cluster_df["session_id"]

        session_id = check_structure_session_id(session_id)
        timestamp_string = session_id.split("_")[-1][0:8]  # eg 14-49-23  time = 14:49, 23rd second
        date_string = session_id.split("_")[-2]
        mouse = session_id.split("_")[0]
        training_day = session_id.split("_")[1]

        timestamp_list.append(timestamp_string)
        date_list.append(date_string)
        mouse_list.append(mouse)
        training_day_list.append(training_day)

    all_days_df["timestamp"] = timestamp_list
    all_days_df["date"] = date_list
    all_days_df["mouse"] = mouse_list
    all_days_df["training_day"] = training_day_list

    return all_days_df

def next_trial_jitter_test(grid_cells, save_path):
    grid_cells = grid_cells.dropna(subset=['n_beaconed_fields_per_trial',
                                           'n_nonbeaconed_fields_per_trial'])
    fig, ax = plt.subplots(figsize=(6,6))
    x_pos = [0.4,1.1]
    trial_types = ["B-B", "B-nB"]
    same_all=[]
    not_same_all=[]
    n_nonbeaconed_fields_per_trial=[]

    colors = ["C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9"]

    for i in range(len(grid_cells)):
        cluster_df = grid_cells.iloc[i]

        fields_com = np.array(cluster_df["fields_com"])
        fields_around_rz = (fields_com>50) & (fields_com<150)

        minimum_distance_to_field_in_next_trial = np.array(cluster_df["minimum_distance_to_field_in_next_trial"])[fields_around_rz]
        minimum_distance_to_field_in_next_trial = minimum_distance_to_field_in_next_trial[~np.isnan(minimum_distance_to_field_in_next_trial)]
        fields_com_next_trial_type = np.array(cluster_df["fields_com_next_trial_type"])[fields_around_rz]
        fields_com_next_trial_type_tmp = fields_com_next_trial_type.copy()
        fields_com_next_trial_type = fields_com_next_trial_type[~np.isnan(fields_com_next_trial_type)]
        fields_com_trial_type = np.array(cluster_df["fields_com_trial_type"])[fields_around_rz]
        fields_com_trial_type = fields_com_trial_type[~np.isnan(fields_com_next_trial_type_tmp)]


        same_fields = minimum_distance_to_field_in_next_trial[fields_com_next_trial_type == fields_com_trial_type]
        not_same_fields = minimum_distance_to_field_in_next_trial[fields_com_next_trial_type != fields_com_trial_type]

        not_same = np.std(not_same_fields)
        same = np.std(same_fields)

        #if (len(same_fields)>10) & (len(not_same_fields)>10):
        #    same_all.append(same)
        #    not_same_all.append(not_same)
        #    ax.plot(x_pos, [same, not_same], marker="o", color='black', alpha=0.3)

        same_all.append(same)
        not_same_all.append(not_same)
        ax.plot(x_pos, [same, not_same], marker="o", color='black', alpha=0.3)

    ax.errorbar(x_pos,[np.mean(same_all),
                       np.mean(not_same_all)],
                yerr=[stats.sem(same_all),
                      stats.sem(not_same_all)], color="black", marker="o")

    p= stats.ttest_rel(same_all,not_same_all)[1]
    print("p="+str(p))

    plt.xticks(x_pos, trial_types, fontsize=20, rotation=0)
    plt.xlim([0,1.5])
    plt.gca().set_ylim(bottom=0)

    plt.ylabel("std distance to field\nin next Trial (cm)",  fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"/vr_grid_cells_jitter_test.png")

def grids_trial_type_paired_t_test(grid_cells, save_path):
    grid_cells = grid_cells.dropna(subset=['n_beaconed_fields_per_trial',
                                           'n_nonbeaconed_fields_per_trial'])
    fig, ax = plt.subplots(figsize=(6,6))
    x_pos = [0.4,1.1]
    trial_types = ["Beaconed", "Non-Beaconed"]
    n_beaconed_fields_per_trial=[]
    n_nonbeaconed_fields_per_trial=[]

    for i in range(len(grid_cells)):
        cluster_df = grid_cells.iloc[i]

        n_beaconed_fields_per_trial.append(cluster_df["n_beaconed_fields_per_trial"])
        n_nonbeaconed_fields_per_trial.append(cluster_df["n_nonbeaconed_fields_per_trial"])

        ax.plot(x_pos, [cluster_df["n_beaconed_fields_per_trial"],
                        cluster_df["n_nonbeaconed_fields_per_trial"]], marker="o", color="black", alpha=0.3)

    ax.errorbar(x_pos,[np.mean(n_beaconed_fields_per_trial),
                       np.mean(n_nonbeaconed_fields_per_trial)],
                yerr=[stats.sem(n_beaconed_fields_per_trial),
                      stats.sem(n_nonbeaconed_fields_per_trial)], color="black", marker="o")

    p= stats.ttest_rel(n_beaconed_fields_per_trial,n_nonbeaconed_fields_per_trial)[1]
    print("p="+str(p))

    plt.xticks(x_pos, trial_types, fontsize=20, rotation=0)
    plt.xlim([0,1.5])
    plt.gca().set_ylim(bottom=0)

    plt.ylabel("Fields / trial",  fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"/vr_grid_cells_non_beaconed_vs_beaconed.png")

def something(prm):
    '''
    cohort_vr_paths = Edmond.Concatenate_from_server.get_recording_paths([], "/mnt/datastore/Harry/Cohort6_july2020/vr")
    all_days_vr_df = Edmond.Concatenate_from_server.load_virtual_reality_spatial_firing(pd.DataFrame(), cohort_vr_paths, prm=prm)
    all_days_vr_df = add_session_identifiers(all_days_vr_df)

    cohort_of_paths = Edmond.Concatenate_from_server.get_recording_paths([], "/mnt/datastore/Harry/Cohort6_july2020/of")
    all_days_of_df = Edmond.Concatenate_from_server.load_open_field_spatial_firing(pd.DataFrame(), cohort_of_paths, prm=prm)
    all_days_of_df = add_session_identifiers(all_days_of_df)

    combined_df = Edmond.Concatenate_from_server.combine_of_vr_dataframes(all_days_vr_df, all_days_of_df)

    grid_cells = combined_df[(combined_df['rate_map_correlation_first_vs_second_half'] > 0) &
                             (combined_df['grid_score'] > 0.2)]
    grid_cells.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/grid_cells.pkl")
    '''

    grid_cells = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/grid_cells.pkl")
    next_trial_jitter_test(grid_cells, save_path="/mnt/datastore/Harry/Vr_grid_cells")
    grids_trial_type_paired_t_test(grid_cells, save_path="/mnt/datastore/Harry/Vr_grid_cells")


    print("do something with the grid cells ")


def main():
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.set_sampling_rate(30000)
    params.set_vr_grid_analysis_bin_size(20)
    params.set_pixel_ratio(440)

    something(params)

    print("look now`")


if __name__ == '__main__':
    main()