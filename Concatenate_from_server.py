import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import Mouse_paths
import traceback
import warnings
import sys
import PostSorting.open_field_head_direction
import PostSorting.parameters
import PostSorting.open_field_grid_cells
import PostSorting.open_field_firing_maps
import PostSorting.theta_modulation

prm = PostSorting.parameters.Parameters()
prm.set_sampling_rate(30000)
prm.set_pixel_ratio(440)


def get_recording_paths(path_list, folder_path):
    list_of_recordings = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for recording_path in list_of_recordings:
        path_list.append(recording_path)
    return path_list

def add_full_session_id(spatial_firing, full_path):
    full_session_ids = []

    for index, spatial_firing_cluster in spatial_firing.iterrows():
        full_session_ids.append(full_path)
    spatial_firing["full_session_id"] = full_session_ids

    return spatial_firing


def load_open_field_spatial_firing(all_days_df, recording_paths, save_path, suffix=""):
    spatial_firing_path = "/MountainSort/DataFrames/spatial_firing.pkl"
    spatial_path = "/MountainSort/DataFrames/position.pkl"

    for path in recording_paths:
        data_frame_path = path+spatial_firing_path
        spatial_df_path = path+spatial_path
        print('Processing ' + data_frame_path)

        if os.path.exists(data_frame_path):
            try:
                print('I found a spatial data frame, processing ' + data_frame_path)
                spatial_firing = pd.read_pickle(data_frame_path)

                if "Curated" in list(spatial_firing):
                    spatial_firing = spatial_firing[spatial_firing["Curated"] == 1]

                spatial_firing = add_full_session_id(spatial_firing, path)


                if ("hd_score" not in list(spatial_firing)) and (len(spatial_firing) > 0):
                    spatial_data = pd.read_pickle(spatial_df_path)
                    _, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(spatial_firing, spatial_data, prm)

                if ("grid_score" not in list(spatial_firing)) and (len(spatial_firing) > 0):
                    spatial_data = pd.read_pickle(spatial_df_path)
                    position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(spatial_data, spatial_firing, prm)
                    spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)

                if ("Boccara_theta_class" not in list(spatial_firing)) and (len(spatial_firing) > 0):
                    prm.set_output_path(path+"/MountainSort")
                    spatial_firing = PostSorting.theta_modulation.calculate_theta_index(spatial_firing, prm)

                if len(spatial_firing) > 0:

                    spatial_firing=spatial_firing[["session_id",
                                                   "cluster_id",
                                                   "tetrode",
                                                   "primary_channel",
                                                   "hd_score",
                                                   "grid_score",
                                                   "full_session_id",
                                                   "ThetaIndex",
                                                   "Boccara_theta_class"]]

                    all_days_df = pd.concat([all_days_df, spatial_firing], ignore_index=True)
                    print('spatial firing data extracted from frame successfully')

                else:
                    print("There wasn't any cells to add")

            except Exception as ex:
                print('This is what Python says happened:')
                print(ex)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback)
                print('spatial firing data extracted frame unsuccessfully')

        else:
            print("I couldn't find a spatial firing dataframe")

    all_days_df.to_pickle(save_path+"/All_mice_of_"+suffix+".pkl")
    print("completed all in list")

def load_processed_position_all_days(recordings_folder_path, paths, mouse):
    processed_position_path = "\MountainSort\DataFrames\processed_position_data.pkl"

    all_days = pd.DataFrame()

    for recording in paths:
        data_frame_path = recordings_folder_path+recording+processed_position_path
        if os.path.exists(data_frame_path):
            print('I found a spatial data frame, processing ' + data_frame_path)
            session_id = recording

            processed_position = pd.read_pickle(data_frame_path)

            all_days = all_days.append({"session_id": session_id,
                                        'beaconed_total_trial_number': np.array(processed_position['beaconed_total_trial_number']),
                                        'nonbeaconed_total_trial_number': np.array(processed_position['nonbeaconed_total_trial_number']),
                                        'probe_total_trial_number': np.array(processed_position['probe_total_trial_number']),
                                        'goal_location': np.array(processed_position['goal_location']),
                                        'goal_location_trial_numbers': np.array(processed_position['goal_location_trial_numbers']),
                                        'goal_location_trial_types': np.array(processed_position['goal_location_trial_types']),
                                        'stop_location_cm': np.array(processed_position['stop_location_cm']),
                                        'stop_trial_number': np.array(processed_position['stop_trial_number']),
                                        'stop_trial_type': np.array(processed_position['stop_trial_type']),
                                        'first_series_location_cm': np.array(processed_position['first_series_location_cm']),
                                        'first_series_trial_number': np.array(processed_position['first_series_trial_number']),
                                        'first_series_trial_type': np.array(processed_position['first_series_trial_type']),
                                        'first_series_location_cm_postcue': np.array(processed_position['first_series_location_cm_postcue']),
                                        'first_series_trial_number_postcue': np.array(processed_position['first_series_trial_number_postcue']),
                                        'first_series_trial_type_postcue': np.array(processed_position['first_series_trial_type_postcue']),
                                        "cue_rewarded_positions": np.array(processed_position['cue_rewarded_positions']),
                                        "cue_rewarded_trial_number": np.array(processed_position['cue_rewarded_trial_number']),
                                        "cue_rewarded_trial_type": np.array(processed_position['cue_rewarded_trial_type'])}, ignore_index=True)


            print('Position data extracted from frame')

    all_days.to_pickle(recordings_folder_path + '/all_days_processed_position_' + mouse + '.pkl')

def main():

    print('-------------------------------------------------------------')

    server_path = "Z:\ActiveProjects\Harry\MouseVR\data\Cue_conditioned_cohort1_190902"
    #load_processed_position_all_days(server_path, Mouse_paths.M2_paths(), mouse="M2")
    #load_processed_position_all_days(server_path, Mouse_paths.M3_paths(), mouse="M3")
    #load_processed_position_all_days(server_path, Mouse_paths.M4_paths(), mouse="M4")
    #load_processed_position_all_days(server_path, Mouse_paths.M5_paths(), mouse="M5")

    server_path = ""

    # ------------------------------------ collect and or calculate all grid scores, hd scores and theta modulation across all recorded open field experiments -----------------------------
    c2 = get_recording_paths([], "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/OpenField")
    c3 = get_recording_paths([], "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild")
    c4 = get_recording_paths([], "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild")
    c5 = get_recording_paths([], "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField")
    c6 = get_recording_paths([], "/mnt/datastore/Junji/Data/2019cohort1/openfield")
    c7 = get_recording_paths([], "/mnt/datastore/Ian/Ephys/Openfield")
    c8 = get_recording_paths([], "/mnt/datastore/Harry/Cohort6_july2020/of")
    c9 = get_recording_paths([], "/mnt/datastore/Harry/MouseOF/data/Cue_conditioned_cohort1_190902")
    c10 = get_recording_paths([], "/mnt/datastore/Bri/sim1cre_invivo")
    c11 = get_recording_paths([], "/mnt/datastore/Klara/Open_field_opto_tagging_p038")

    all_days_df = pd.DataFrame()
    #load_open_field_spatial_firing(all_days_df, c2, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C2")
    #load_open_field_spatial_firing(all_days_df, c3, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C3")
    #load_open_field_spatial_firing(all_days_df, c4, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C4")
    #load_open_field_spatial_firing(all_days_df, c5, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C5")
    #load_open_field_spatial_firing(all_days_df, c6, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C6")
    #load_open_field_spatial_firing(all_days_df, c7, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C7")
    load_open_field_spatial_firing(all_days_df, c8, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C8")
    load_open_field_spatial_firing(all_days_df, c9, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C9")
    load_open_field_spatial_firing(all_days_df, c10, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C10")
    load_open_field_spatial_firing(all_days_df, c11, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper", suffix="C11")

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()