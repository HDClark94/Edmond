import numpy as np
import pandas as pd
import os
import traceback
import sys
import PostSorting.parameters
import file_utility
import settings
import PostSorting.open_field_firing_maps
from Edmond.utility_functions.array_manipulations import *
prm = PostSorting.parameters.Parameters()
prm.set_sampling_rate(30000)
prm.set_pixel_ratio(440)

def get_tags_parameter_file(recording_directory):
    tags = False
    parameters_path = recording_directory + '/parameters.txt'
    param_file_reader = open(parameters_path, 'r')
    parameters = param_file_reader.readlines()
    parameters = list([x.strip() for x in parameters])
    if len(parameters) > 2:
        tags = parameters[2]
    return tags

def remove_outlier_waveforms(all_waveforms, max_deviations=2):
    # remove snippets that have data points > 3 standard dev away from mean
    mean = all_waveforms.mean(axis=1)
    sd = all_waveforms.std(axis=1)
    distance_from_mean = all_waveforms.T - mean
    outliers = np.sum(distance_from_mean > max_deviations * sd, axis=1) > 0
    return all_waveforms[:, ~outliers]

def add_peaks_to_troughs(df):
    peak_to_troughs = []
    for index, row in df.iterrows():
        row = row.to_frame().T.reset_index(drop=True)
        primary_channel = row["primary_channel"].iloc[0]
        random_snippets = row["random_snippets"].iloc[0][primary_channel-1]
        random_snippets = remove_outlier_waveforms(random_snippets)
        troughs = np.min(random_snippets, axis=0)
        peaks = np.max(random_snippets, axis=0)
        peak_to_trough = max(peaks-troughs)
        peak_to_troughs.append(peak_to_trough)
    df["snippet_peak_to_trough"] = peak_to_troughs
    return df

def process_running_parameter_tag(running_parameter_tags):
    stop_threshold = 4.9  # defaults
    track_length = 200 # default assumptions
    cue_conditioned_goal = False

    if not running_parameter_tags:
        return stop_threshold, track_length, cue_conditioned_goal

    tags = [x.strip() for x in running_parameter_tags.split('*')]
    for tag in tags:
        if tag.startswith('stop_threshold'):
            stop_threshold = float(tag.split("=")[1])
        elif tag.startswith('track_length'):
            track_length = int(tag.split("=")[1])
        elif tag.startswith('cue_conditioned_goal'):
            cue_conditioned_goal = bool(tag.split('=')[1])
        else:
            print('Unexpected / incorrect tag in the third line of parameters file')
            unexpected_tag = True
    return stop_threshold, track_length, cue_conditioned_goal

def get_track_length(recording_path):
    parameter_file_path = get_tags_parameter_file(recording_path)
    stop_threshold, track_length, cue_conditioned_goal = process_running_parameter_tag(parameter_file_path)
    return track_length

def get_proportion_reward(processed_position, trial_type="all"):
    if trial_type == "beaconed":
        processed_position = processed_position[processed_position["trial_type"] == 0]
    if trial_type == "non_beaconed":
        processed_position = processed_position[processed_position["trial_type"] == 1]
    if trial_type == "probe":
        processed_position = processed_position[processed_position["trial_type"] == 2]
    elif trial_type == "all":
        processed_position = processed_position

    return np.sum(processed_position["rewarded"]/len(processed_position))

def get_performance(processed_position, trial_type="all"):
    if trial_type == "beaconed":
        processed_position = processed_position[processed_position["trial_type"] == 0]
    if trial_type == "non_beaconed":
        processed_position = processed_position[processed_position["trial_type"] == 1]
    if trial_type == "probe":
        processed_position = processed_position[processed_position["trial_type"] == 2]
    elif trial_type == "all":
        processed_position = processed_position

    rewarded_stops=0
    stops=0
    for index, row in processed_position.iterrows():
        trial_row = row.to_frame().T.reset_index(drop=True)

        n_stop_location_cm = len(trial_row["stop_location_cm"].iloc[0])
        n_reward_stop_location_cm = len(trial_row["reward_stop_location_cm"].iloc[0])

        rewarded_stops += n_reward_stop_location_cm
        stops += n_stop_location_cm

    return rewarded_stops/stops

def get_recording_paths(path_list, folder_path):
    list_of_recordings = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for recording_path in list_of_recordings:
        path_list.append(recording_path)
        print(recording_path.split("/datastore/")[-1])
    return path_list

def get_collumns_with_single_values(spatial_firing):

    collumn_names_to_keep = []
    for i in range(len(list(spatial_firing))):
        collumn_name = list(spatial_firing)[i]
        if (np.size(spatial_firing[collumn_name].iloc[0]) == 1):
            collumn_names_to_keep.append(collumn_name)
    return collumn_names_to_keep

def add_full_session_id(spatial_firing, full_path):
    full_session_ids = []

    for index, spatial_firing_cluster in spatial_firing.iterrows():
        full_session_ids.append(full_path)
    spatial_firing["full_session_id"] = full_session_ids

    return spatial_firing


def get_tags_parameter_file(recording_directory):
    tags = False
    parameters_path = recording_directory + '/parameters.txt'
    param_file_reader = open(parameters_path, 'r')
    parameters = param_file_reader.readlines()
    parameters = list([x.strip() for x in parameters])
    if len(parameters) > 2:
        tags = parameters[2]
    return tags

def check_for_tag_name(running_parameter_tags, tag_name):
    tag_in_file = None
    if running_parameter_tags is not False:
        tags = [x.strip() for x in running_parameter_tags.split('*')]
        for tag in tags:
            if tag.startswith(tag_name):
                tag_in_file = str(tag.split("=")[1]).split(',')[0]
    return tag_in_file

def add_processed_position_data_column_to_spatial_firing(spatial_firing, processed_position, columns=[]):
    for column in columns:#
        column_values = []
        for index, row in spatial_firing.iterrows():
            processed_position_values = pandas_collumn_to_2d_numpy_array(processed_position[column])
            column_values.append(processed_position_values)
        spatial_firing[column] = column_values
    return spatial_firing

def load_virtual_reality_spatial_firing(all_days_df, recording_paths, save_path=None, suffix=""):

    for path in recording_paths:
        tags = get_tags_parameter_file(path)
        sorter_name = check_for_tag_name(tags, "sorter_name")
        spatial_firing_path = "/"+sorter_name+"/DataFrames/spatial_firing.pkl"
        spatial_path = "/"+sorter_name+"/DataFrames/position_data.pkl"
        processed_path = "/"+sorter_name+"/DataFrames/processed_position_data.pkl"
        number_of_channels, _ = file_utility.count_files_that_match_in_folder(path, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')

        data_frame_path = path+spatial_firing_path
        spatial_df_path = path+spatial_path
        processed_position_path = path+processed_path

        print('Processing ' + data_frame_path)
        if os.path.exists(data_frame_path):
            try:
                print('I found a spatial data frame, processing ' + data_frame_path)
                spatial_firing = pd.read_pickle(data_frame_path)
                processed_position = pd.read_pickle(processed_position_path)
                spatial_firing = add_full_session_id(spatial_firing, path)
                spatial_firing = add_processed_position_data_column_to_spatial_firing(spatial_firing, processed_position, columns=['speeds_binned_in_space_smoothed'])
                spatial_firing = add_peaks_to_troughs(spatial_firing)
                spatial_firing["track_length"] = get_track_length(path)

                if len(spatial_firing) > 0:
                    collumn_names_to_keep = get_collumns_with_single_values(spatial_firing)
                    collumn_names_to_keep.append("firing_times")
                    collumn_names_to_keep.append("random_snippets")
                    if "firing_times" in list(spatial_firing):
                        collumn_names_to_keep.append("firing_times")
                    if "MOVING_LOMB_avg_power" in list(spatial_firing):
                        collumn_names_to_keep.append("MOVING_LOMB_avg_power")
                    if "MOVING_LOMB_all_powers" in list(spatial_firing):
                        collumn_names_to_keep.append("MOVING_LOMB_all_powers")
                    if "MOVING_LOMB_all_centre_trials" in list(spatial_firing):
                        collumn_names_to_keep.append("MOVING_LOMB_all_centre_trials")
                    if "stop_locations" in list(spatial_firing):
                        collumn_names_to_keep.append("stop_locations")
                    if "stop_trial_numbers" in list(spatial_firing):
                        collumn_names_to_keep.append("stop_trial_numbers")
                    if "behaviour_trial_numbers" in list(spatial_firing):
                        collumn_names_to_keep.append("behaviour_trial_numbers")
                    if "behaviour_hit_try_miss" in list(spatial_firing):
                        collumn_names_to_keep.append("behaviour_hit_try_miss")
                    if "behaviour_trial_types" in list(spatial_firing):
                        collumn_names_to_keep.append("behaviour_trial_types")
                    if "rolling:block_lengths_shuffled" in list(spatial_firing):
                        collumn_names_to_keep.append("rolling:block_lengths_shuffled")
                    if "rolling:block_lengths" in list(spatial_firing):
                        collumn_names_to_keep.append("rolling:block_lengths")
                    if "rolling:classifier_by_trial_number" in list(spatial_firing):
                        collumn_names_to_keep.append("rolling:classifier_by_trial_number")
                    if "rolling:rolling_centre_trials" in list(spatial_firing):
                        collumn_names_to_keep.append("rolling:rolling_centre_trials")
                    if "rolling:rolling_classifiers" in list(spatial_firing):
                        collumn_names_to_keep.append("rolling:rolling_classifiers")
                    if "rolling:position_correlation_by_trial_number_t2tmethod" in list(spatial_firing):
                        collumn_names_to_keep.append("rolling:position_correlation_by_trial_number_t2tmethod")
                    if "fr_binned_in_space_smoothed" in list(spatial_firing):
                        collumn_names_to_keep.append("fr_binned_in_space_smoothed")
                    if "avg_z_scored_firing_rates" in list(spatial_firing):
                        collumn_names_to_keep.append("avg_z_scored_firing_rates")
                    if "avg_z_scored_running_speeds" in list(spatial_firing):
                        collumn_names_to_keep.append("avg_z_scored_running_speeds")
                    if "rolling:grid_code_global" in list(spatial_firing):
                        collumn_names_to_keep.append("rolling:grid_code_global")
                    if "field_locations" in list(spatial_firing):
                        collumn_names_to_keep.append("field_locations")
                    if "field_trial_numbers" in list(spatial_firing):
                        collumn_names_to_keep.append("field_trial_numbers")
                    if "speeds_binned_in_space_smoothed" in list(spatial_firing):
                        collumn_names_to_keep.append("speeds_binned_in_space_smoothed")


                    spatial_firing=spatial_firing[collumn_names_to_keep]
                    # rename the mean_firing_rate_local collumn to be specific to vr or of
                    spatial_firing = spatial_firing.rename(columns={'mean_firing_rate': ('mean_firing_rate_vr')})
                    spatial_firing = spatial_firing.rename(columns={'spatial_information_score_Isec': ('spatial_information_score_Isec_vr')})
                    spatial_firing = spatial_firing.rename(columns={'spatial_information_score_Ispike': ('spatial_information_score_Ispike_vr')})
                    spatial_firing = spatial_firing.rename(columns={'firing_times': ('firing_times_vr')})
                    spatial_firing = spatial_firing.rename(columns={'random_snippets': ('random_snippets_vr')})
                    spatial_firing = spatial_firing.rename(columns={'full_session_id': ('full_session_id_vr')})
                    spatial_firing["session_id_vr"] = spatial_firing["session_id"]

                    # remove duplicated columns
                    spatial_firing = spatial_firing.loc[:,~spatial_firing.columns.duplicated()]

                    spatial_firing["n_trials"] = len(processed_position)
                    spatial_firing["number_of_channels"] = number_of_channels

                    all_days_df = pd.concat([all_days_df, spatial_firing], ignore_index=True)
                    print('spatial firing data extracted from frame successfully')

                else:
                    print("There wasn't any cells to add")
                    # make an empty row for the recording day to show no cells were found
                    #session_id = path.split("/")[-1]
                    #data = {'session_id': [session_id]}
                    #spatial_firing = pd.DataFrame(data)
                    #all_days_df = pd.concat([all_days_df, spatial_firing], ignore_index=True)

            except Exception as ex:
                print('This is what Python says happened:')
                print(ex)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback)
                print('something went wrong, the recording might be missing dataframes!')

        else:
            print("I couldn't find a spatial firing dataframe")
    if save_path is not None:

        all_days_df.to_pickle(save_path+"/All_mice_vr_"+suffix+".pkl")
    print("completed all in list")
    return all_days_df



def load_open_field_spatial_firing(all_days_df, recording_paths, save_path=None, suffix=""):
    for path in recording_paths:
        tags = get_tags_parameter_file(path)
        sorter_name = check_for_tag_name(tags, "sorter_name")
        spatial_firing_path = "/" + sorter_name + "/DataFrames/spatial_firing.pkl"
        spatial_path = "/" + sorter_name + "/DataFrames/position.pkl"
        data_frame_path = path + spatial_firing_path
        spatial_df_path = path + spatial_path

        print('Processing ' + data_frame_path)
        if os.path.exists(data_frame_path):
            try:
                print('I found a spatial data frame, processing ' + data_frame_path)
                spatial_firing = pd.read_pickle(data_frame_path)
                position_data = pd.read_pickle(spatial_df_path)

                if "occupancy_maps" in list(spatial_firing):
                    print("occupancy_maps found")
                else:
                    print("occupancy_maps not found")

                if "Curated" in list(spatial_firing):
                    spatial_firing = spatial_firing[spatial_firing["Curated"] == 1]

                spatial_firing = add_full_session_id(spatial_firing, path)

                if len(spatial_firing) > 0:
                    collumn_names_to_keep = get_collumns_with_single_values(spatial_firing)
                    collumn_names_to_keep.append("firing_times")
                    collumn_names_to_keep.append("random_snippets")
                    collumn_names_to_keep.append("firing_maps")
                    collumn_names_to_keep.append("occupancy_maps")
                    collumn_names_to_keep.append("rate_map_autocorrelogram")

                    spatial_firing=spatial_firing[collumn_names_to_keep]

                    # rename the mean_firing_rate_local collumn to be specific to vr or of
                    spatial_firing = spatial_firing.rename(columns={'mean_firing_rate': ('mean_firing_rate_of')})
                    spatial_firing = spatial_firing.rename(columns={'firing_times': ('firing_times_of')})
                    spatial_firing = spatial_firing.rename(columns={'random_snippets': ('random_snippets_of')})
                    spatial_firing = spatial_firing.rename(columns={'full_session_id': ('full_session_id_of')})
                    spatial_firing = spatial_firing.rename(columns={'firing_maps': ('firing_maps_of')})
                    spatial_firing = spatial_firing.rename(columns={'occupancy_maps': ('occupancy_maps_of')})
                    spatial_firing = spatial_firing.rename(columns={'rate_map_autocorrelogram': ('rate_map_autocorrelogram_of')})
                    spatial_firing["session_id_of"] = spatial_firing["session_id"]

                    # remove duplicated columns
                    spatial_firing = spatial_firing.loc[:,~spatial_firing.columns.duplicated()]

                    all_days_df = pd.concat([all_days_df, spatial_firing], ignore_index=True)
                    print('spatial firing data extracted from frame successfully')

                else:
                    print("There wasn't any cells to add")
                    # make an empty row for the recording day to show no cells were found
                    session_id = path.split("/")[-1]
                    data = {'session_id': [session_id]}
                    spatial_firing = pd.DataFrame(data)
                    all_days_df = pd.concat([all_days_df, spatial_firing], ignore_index=True)

            except Exception as ex:
                print('This is what Python says happened:')
                print(ex)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback)
                print('spatial firing data extracted frame unsuccessfully')

        else:
            print("I couldn't find a spatial firing dataframe")

    if save_path is not None:
        all_days_df.to_pickle(save_path+"/All_mice_of_"+suffix+".pkl")
    print("completed all in list")
    return all_days_df

def combine_of_vr_dataframes(vr_dataframe, of_dataframe):
    # combine and return of and vr matches with same session day, mouse id and cluster id
    combined_df = pd.DataFrame()
    print("==========================================")
    print(np.unique(vr_dataframe[vr_dataframe["mouse"]=="M16"]["recording_day"]))
    print("==========================================")

    for index, cluster_vr_series in vr_dataframe.iterrows():
        cluster_id = cluster_vr_series["cluster_id"]
        date=cluster_vr_series["date"]
        mouse=cluster_vr_series["mouse"]
        training_day=cluster_vr_series["recording_day"]

        cluster_of_df = of_dataframe[(of_dataframe.cluster_id == cluster_id) &
                                     (of_dataframe.date == date) &
                                     (of_dataframe.mouse == mouse)]

        cluster_vr_df = vr_dataframe[(vr_dataframe.cluster_id == cluster_id) &
                                     (vr_dataframe.date == date) &
                                     (vr_dataframe.mouse == mouse)]

        print("I am looking at cluster", str(cluster_id), " from ", str(date), ", mouse ", str(mouse), ", day", str(training_day))


        combined_cluster = cluster_of_df.copy()
        if ((len(cluster_vr_df) == 1) and (len(cluster_of_df) == 1)):
            collumns_to_add = np.setdiff1d(list(cluster_vr_df), list(cluster_of_df)) # finds collumns in 1 list that are not in the other
            for i in range(len(collumns_to_add)):
                combined_cluster[collumns_to_add[i]] = [cluster_vr_df[collumns_to_add[i]].iloc[0]]
            combined_df = pd.concat([combined_df, combined_cluster], ignore_index=True)
            print("Added combined cluster", str(cluster_id), " from ", str(date), ", mouse ", str(mouse), ", day",str(training_day))
        elif (len(cluster_vr_df) == 1):
            combined_df = pd.concat([combined_df, cluster_vr_df], ignore_index=True)
            print("Added vr cluster", str(cluster_id), " from ", str(date), ", mouse ", str(mouse), ", day", str(training_day))
        elif (len(cluster_of_df) == 1):
            combined_df = pd.concat([combined_df, cluster_of_df], ignore_index=True)
            print("Added of cluster", str(cluster_id), " from ", str(date), ", mouse ", str(mouse), ", day",str(training_day))
        else:
            print("Did not add cluster", str(cluster_id), " from ", str(date), ", mouse ", str(mouse), ", day", str(training_day))

    return combined_df

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
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()