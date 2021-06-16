import os
import glob
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

"""
This file creates a concatenated dataframe of all the recording directories passed to it and saves it where specified,
for collection of the processed position data for the vr side, there will be a link pointing to the original processed position
"""

def load_processed_position_data_collumns(spike_data, processed_position_data):
    for collumn in list(processed_position_data):
        collumn_data = processed_position_data[collumn].tolist()
        spike_data[collumn] = [collumn_data for x in range(len(spike_data))]
    return spike_data

def add_nested_time_binned_data(spike_data, processed_position_data):

    nested_lists = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]

        for trial_number in processed_position_data["trial_number"]:
            trial_proccessed_position_data = processed_position_data[(processed_position_data["trial_number"] == trial_number)]
            trial_type = trial_proccessed_position_data["trial_type"].iloc[0]

            speed_o = pd.Series(trial_proccessed_position_data['speeds_binned_in_time'].iloc[0])
            position_o = pd.Series(trial_proccessed_position_data['pos_binned_in_time'].iloc[0])
            #acceleration_o = pd.Series(trial_proccessed_position_data['acc_binned_in_time'].iloc[0])
            rates_o = pd.Series(cluster_spike_data['fr_time_binned'].iloc[0][trial_number-1])

            if len(speed_o)>0: # add a catch for nans?

                # remove outliers
                rates = rates_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))].to_numpy() # without outliers
                speed = speed_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))].to_numpy() # without outliers
                position = position_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))].to_numpy() # without outliers
                #acceleration = acceleration_o[speed_o.between(speed_o.quantile(.05), speed_o.quantile(.95))].to_numpy() # without outliers

                # make trial type, trial number and whether it was a rewarded trial into longform
                trial_numbers = np.repeat(trial_number, len(rates))
                trial_types = np.repeat(trial_type, len(rates))

                spikes_in_time = []
                spikes_in_time.append(rates)
                spikes_in_time.append(speed)
                spikes_in_time.append(position)
                spikes_in_time.append(trial_numbers)
                spikes_in_time.append(trial_types)

        nested_lists.append(spikes_in_time)

    spike_data["spike_rate_in_time"] = nested_lists

    return spike_data


def add_nested_space_binned_data(spike_data, processed_position_data):

    nested_lists = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]

        for trial_number in processed_position_data["trial_number"]:
            trial_proccessed_position_data = processed_position_data[(processed_position_data["trial_number"] == trial_number)]

            rates = cluster_spike_data['fr_binned_in_space'].iloc[0][trial_number-1]
            trial_numbers = np.repeat(trial_number, len(rates))
            trial_types = np.repeat(trial_proccessed_position_data["trial_type"].iloc[0], len(rates))

            spikes_in_space = []
            spikes_in_space.append(rates)
            spikes_in_space.append(trial_numbers)
            spikes_in_space.append(trial_types)

        nested_lists.append(spikes_in_space)

    spike_data["spike_rate_on_trials_smoothed"] = nested_lists

    return spike_data

def add_reward_variables(spike_data, processed_position_data):
    rewarded_locations_clusters = []
    rewarded_trials_clusters = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        rewarded_trials = []
        rewarded_locations = []

        for trial_number in processed_position_data["trial_number"]:
            trial_proccessed_position_data = processed_position_data[(processed_position_data["trial_number"] == trial_number)]

            rewarded = trial_proccessed_position_data["rewarded"].iloc[0]
            trial_rewarded_locations = trial_proccessed_position_data["reward_stop_location_cm"].iloc[0]
            if rewarded:
                rewarded_trials.append(trial_number)
                rewarded_locations.extend(trial_rewarded_locations)

        rewarded_locations_clusters.append(rewarded_locations)
        rewarded_trials_clusters.append(rewarded_trials)

    spike_data["rewarded_trials"] = rewarded_trials_clusters
    spike_data["rewarded_locations"] = rewarded_locations_clusters

    return spike_data

def remove_cluster_without_firing_events(spike_data):
    '''
    Removes rows where no firing times are found, this occurs when spikes are found in one session type and not the
    other when multiple sessions are spike sorted together
    '''

    spike_data_filtered = pd.DataFrame()
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]
        firing_times = cluster_spike_data["firing_times"].iloc[0]
        if len(firing_times)>0:
            spike_data_filtered = pd.concat([spike_data_filtered, cluster_spike_data])
        else:
            print("I am removing cluster ", cluster_id, " from this recording")
            print("because it has no firing events in this spatial firing dataframe")

    return spike_data_filtered


def process_dir(recordings_path, concatenated_spike_data=None, save_path=None):

    """
    Creates a dataset with spike data for ramp analysis and modelling

    :param recordings_path: path for a folder with all the recordings you want to process
    :param concatenated_spike_data: a pandas dataframe to append all processsed spike data to
    :param save_path: where we save the new processed spike data
    :return: processed spike data
    """

    # make an empty dataframe if concatenated frame given as none
    if concatenated_spike_data is None:
        concatenated_spike_data = pd.DataFrame()

    # get list of all recordings in the recordings folder
    recording_list = [f.path for f in os.scandir(recordings_path) if f.is_dir()]

    # loop over recordings and add spatial firing to the concatenated frame, add the paths to processed position
    for recording in recording_list:
        print("processeding ", recording.split("/")[-1])

        spatial_dataframe_path = recording + '/MountainSort/DataFrames/processed_position_data.pkl'
        spike_dataframe_path = recording + '/MountainSort/DataFrames/spatial_firing.pkl'
        mouse_id = recording.split("/")[-1].split("_")[0]

        if os.path.exists(spike_dataframe_path):
            spike_data = pd.read_pickle(spike_dataframe_path)
            spike_data = remove_cluster_without_firing_events(spike_data)
            if os.path.exists(spatial_dataframe_path):
                processed_position_data = pd.read_pickle(spatial_dataframe_path)

                # look for key collumns needs for ramp amalysis
                if ("fr_time_binned" in list(spike_data)) or ("fr_binned_in_space" in list(spike_data)):

                    spike_data = add_nested_time_binned_data(spike_data, processed_position_data)
                    spike_data = add_nested_space_binned_data(spike_data, processed_position_data)
                    spike_data = add_reward_variables(spike_data, processed_position_data)

                    columns_to_drop = ['all_snippets', 'random_snippets', 'beaconed_position_cm', 'beaconed_trial_number',
                                       'nonbeaconed_position_cm', 'nonbeaconed_trial_number', 'probe_position_cm',
                                       'probe_trial_number', 'beaconed_firing_rate_map', 'non_beaconed_firing_rate_map',
                                       'probe_firing_rate_map', 'beaconed_firing_rate_map_sem', 'non_beaconed_firing_rate_map_sem',
                                       'probe_firing_rate_map_sem']
                    for column in columns_to_drop:
                        if column in list(spike_data):
                            del spike_data[column]

                    concatenated_spike_data = pd.concat([concatenated_spike_data, spike_data], ignore_index=True)

                else:
                    if (len(spike_data)==0):
                        print("this recording has no units, ", recording.split("/")[-1])
                    else:
                        print("could not find correct binned collumn in recording ", recording.split("/")[-1])

            else:
                print("couldn't find processed_position for ", recording.split("/")[-1])

    if save_path is not None:
        concatenated_spike_data.to_pickle(save_path+"concatenated_spike_data.pkl")

    return concatenated_spike_data

#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    spike_data = process_dir(recordings_path= "/mnt/datastore/Harry/Cohort7_october2020/vr", concatenated_spike_data=None,
                             save_path= "/mnt/datastore/Harry/Ramp_cells_open_field_paper/")


    spike_data = pd.read_pickle("/mnt/datastore/Harry/Ramp_cells_open_field_paper/concatenated_spike_data.pkl")
    print("were done for now ")

if __name__ == '__main__':
    main()





