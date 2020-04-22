import pandas as pd
from Edmond import sorting_comparison
import os
from Edmond.plotting import *
import numpy as np
import sys
import traceback

def get_id(recording_path):
    id = recording_path.split("/")[-1]
    return id

def find_match(recording_path, recording_path_list):
    # return the matching recording_path in the list which contains the same mouse and day id
    id = get_id(recording_path)
    mouse_and_day = id.split("_20")[0].split("/")[-1] + "_20"
    matching_of_path = list(filter(lambda x: mouse_and_day in x, recording_path_list))[0]
    return matching_of_path

def add_open_field_stuff_to_agreement(agreement_dataframe, sorted_together_of_path):

    if os.path.exists(sorted_together_of_path):
        sorted_together_of = pd.read_pickle(sorted_together_of_path)

        of_session_id = sorted_together_of_path.split("/")[-4]

    new_agreement_dataframe = pd.DataFrame()
    for index, row in agreement_dataframe.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        sorted_together_cluster_id = row["sorted_together_cluster_ids"].iloc[0]
        cluster_of = sorted_together_of[(sorted_together_of["cluster_id"]) == sorted_together_cluster_id]

        cluster_of = cluster_of[['mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'peak_amp',
                                 'speed_score', 'speed_score_p_values', 'max_firing_rate_hd', 'preferred_HD',
                                 'hd_score', 'rayleigh_score', 'grid_spacing', 'field_size', 'grid_score',
                                 'border_score', 'corner_score', 'rate_map_correlation_first_vs_second_half',
                                 'percent_excluded_bins_rate_map_correlation_first_vs_second_half_p',
                                 'hd_correlation_first_vs_second_half', 'hd_correlation_first_vs_second_half_p']].reset_index(drop=True)

        cluster_of["of_session_id"] = of_session_id

        cluster_of = row.join(cluster_of)
        new_agreement_dataframe = pd.concat([new_agreement_dataframe,
                                             cluster_of], ignore_index=True)

    return new_agreement_dataframe

def run_curation_analysis(sorted_together_vr_dir_path, sorted_together_of_dir_path,
                          sorted_apart_vr_dir_path, sorted_apart_of_dir_path, save_path,
                          figs_path=None, add_of_and_save=False, cohort_mouse=None):

    # create list of recordings in sorted folders
    sorted_together_vr_list = [f.path for f in os.scandir(sorted_together_vr_dir_path) if f.is_dir()]
    sorted_apart_vr_list = [f.path for f in os.scandir(sorted_apart_vr_dir_path) if f.is_dir()]
    sorted_together_of_list = [f.path for f in os.scandir(sorted_together_of_dir_path) if f.is_dir()]
    sorted_apart_of_list = [f.path for f in os.scandir(sorted_apart_of_dir_path) if f.is_dir()]

    dataframe_subpath = "/MountainSort/DataFrames/spatial_firing.pkl"
    concat_together = pd.DataFrame()
    concat_apart = pd.DataFrame()
    concat_apart_of = pd.DataFrame()

    # loop over these lists
    for i in range(len(sorted_together_vr_list)):
        try:
            sorted_together_vr = sorted_together_vr_list[i]

            #look for matching recording
            sorted_together_of = find_match(sorted_together_vr, sorted_together_of_list)
            sorted_apart_vr = find_match(sorted_together_vr, sorted_apart_vr_list)
            sorted_seperately_of = find_match(sorted_together_vr, sorted_apart_of_list)

            print("Starting curation analysis for between ", get_id(sorted_together_vr), " and ", get_id(sorted_apart_vr))

            if os.path.exists(sorted_together_vr + dataframe_subpath):
                sorted_together = pd.read_pickle(sorted_together_vr + dataframe_subpath)
                if 'Curated' in list(sorted_together):
                    sorted_together = sorted_together[sorted_together["Curated"]==1]
                sorted_together = sorted_together[['cluster_id', 'mean_firing_rate', 'isolation',
                                                   'noise_overlap', 'peak_snr', 'peak_amp']]
                concat_together = pd.concat([concat_together, sorted_together], ignore_index=True)


            if os.path.exists(sorted_apart_vr + dataframe_subpath):
                sorted_seperately = pd.read_pickle(sorted_apart_vr + dataframe_subpath)
                sorted_seperately = sorted_seperately[['cluster_id', 'mean_firing_rate', 'isolation',
                                                       'noise_overlap', 'peak_snr', 'peak_amp']]
                concat_apart = pd.concat([concat_apart,sorted_seperately], ignore_index=True)

            if os.path.exists(sorted_seperately_of + dataframe_subpath):
                sorted_seperately_of = pd.read_pickle(sorted_seperately_of + dataframe_subpath)
                sorted_seperately_of = sorted_seperately_of[['cluster_id', 'mean_firing_rate', 'isolation',
                                                             'noise_overlap', 'peak_snr', 'peak_amp']]
                concat_apart_of = pd.concat([concat_apart_of, sorted_seperately_of], ignore_index=True)

        except Exception as ex:
            print("There was an issue. This is what Python says happened:")
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)

    plot_curation_stats(concat_apart, concat_together, concat_apart_of, figs_path, cohort_mouse)

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # take a list of paths of sorted vr_recordings from Sarah's server space
    # iterate over this list and pick out the spatial dataframes
    figs_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs"

    '''
    # cohort 5
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M1_sorted"
    sorted_apart_of_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/OpenField"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M1"
    run_curation_analysis(sorted_together_vr_dir_path, sorted_together_of_dir_path,
                          sorted_apart_vr_dir_path, sorted_apart_of_dir_path, save_path,
                  figs_path=figs_path, add_of_and_save=False, cohort_mouse="C5_M1")

    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M2_sorted"
    sorted_apart_of_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/OpenField"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M2"
    run_curation_analysis(sorted_together_vr_dir_path, sorted_together_of_dir_path,
                          sorted_apart_vr_dir_path, sorted_apart_of_dir_path, save_path,
                  figs_path=figs_path, add_of_and_save=False, cohort_mouse="C5_M2")
    '''

    # cohort 4
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M2_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild"
    sorted_apart_of_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/VirtualReality/M2_sorted"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M2"
    run_curation_analysis(sorted_together_vr_dir_path, sorted_together_of_dir_path,
                          sorted_apart_vr_dir_path, sorted_apart_of_dir_path, save_path,
                          figs_path=figs_path, add_of_and_save=False, cohort_mouse="C4_M2")

    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M3_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild"
    sorted_apart_of_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/VirtualReality/M3_sorted"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M3"
    run_curation_analysis(sorted_together_vr_dir_path, sorted_together_of_dir_path,
                          sorted_apart_vr_dir_path, sorted_apart_of_dir_path, save_path,
                          figs_path=figs_path, add_of_and_save=False, cohort_mouse="C4_M3")

    '''
    # cohort 3 # not done yet
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M1_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/VirtualReality/M1_sorted"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M1"
    run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, 
                  save_path, figs_path=figs_path, add_of_and_save=False, cohort_mouse="C6_M1")

    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M6_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/VirtualReality/M6_sorted"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M6"
    run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, save_path, 
                  figs_path=figs_path, add_of_and_save=False, cohort_mouse="C3_M6")
    print("look now")
    '''


if __name__ == '__main__':
    main()