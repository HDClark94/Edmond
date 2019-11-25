import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import Mouse_paths

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
                                        'first_series_trial_type_postcue': np.array(processed_position['first_series_trial_type_postcue'])}, ignore_index=True)

            print('Position data extracted from frame')

    all_days.to_pickle(recordings_folder_path + '/all_days_processed_position_' + mouse + '.pkl')

def main():

    print('-------------------------------------------------------------')

    server_path = "Z:\ActiveProjects\Harry\MouseVR\data\Cue_conditioned_cohort1_190902"
    load_processed_position_all_days(server_path, Mouse_paths.M2_paths(), mouse="M2")
    #load_processed_position_all_days(server_path, Mouse_paths.M3_paths(), mouse="M3")
    #load_processed_position_all_days(server_path, Mouse_paths.M4_paths(), mouse="M4")
    #load_processed_position_all_days(server_path, Mouse_paths.M5_paths(), mouse="M5")


    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()