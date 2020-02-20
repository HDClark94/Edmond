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
                                        'stop_location_cm': np.array(processed_position['stop_location_cm']),
                                        'stop_trial_number': np.array(processed_position['stop_trial_number']),
                                        'stop_trial_type': np.array(processed_position['stop_trial_type']),
                                        'first_series_location_cm': np.array(processed_position['first_series_location_cm']),
                                        'first_series_trial_number': np.array(processed_position['first_series_trial_number']),
                                        'first_series_trial_type': np.array(processed_position['first_series_trial_type'])}, ignore_index=True)

            print('Position data extracted from frame')

    all_days.to_pickle(recordings_folder_path + '/all_days_processed_position_' + mouse + '.pkl')

def main():

    print('-------------------------------------------------------------')

    server_path = r"Z:\ActiveProjects\Harry\2019cohort1\vr"
    #load_processed_position_all_days(server_path, Mouse_paths.M1_junji_paths(), mouse="M1")
    load_processed_position_all_days(server_path, Mouse_paths.M2_junji_paths(), mouse="M2")


    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()