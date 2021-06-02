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

def make_longform(spike_data, processed_position_data, binning):
    long_form_spike_data = pd.DataFrame()

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]

        for trial_number in processed_position_data["trial_number"]:
            trial_proccessed_position_data = processed_position_data[(processed_position_data["trial_number"] == trial_number)]

            # space
            if binning == "space":
                speed = trial_proccessed_position_data['speeds_binned_in_space'].iloc[0]
                pos = trial_proccessed_position_data['pos_binned_in_space'].iloc[0]
                acc = trial_proccessed_position_data['acc_binned_in_space'].iloc[0]
                fr = np.array(cluster_spike_data['fr_binned_in_space'].iloc[0][trial_number-1])
                fr_pos_bin_centres = np.array(cluster_spike_data['fr_binned_in_space_bin_centres'].iloc[0][trial_number-1])

                if len(speed)>0: # add a catch for nans?
                    cluster_spike_data_long_form = pd.concat([cluster_spike_data]*len(fr), ignore_index=True)
                    del cluster_spike_data_long_form["fr_binned_in_space_bin_centres"]
                    del cluster_spike_data_long_form["fr_binned_in_space"]

                    trial_number_long_form = np.repeat(trial_number, len(fr))
                    trial_type_long_form = np.repeat(trial_proccessed_position_data["trial_type"].iloc[0], len(fr))
                    rewarded_long_form = np.repeat(trial_proccessed_position_data["rewarded"].iloc[0], len(fr))

                    cluster_spike_data_long_form["Trials"] = trial_number_long_form
                    cluster_spike_data_long_form["Trial_Type"] = trial_type_long_form
                    cluster_spike_data_long_form["Rewarded"] = rewarded_long_form
                    cluster_spike_data_long_form["Speeds"] = speed
                    cluster_spike_data_long_form["Position"] = pos
                    cluster_spike_data_long_form["Acceleration"] = acc
                    cluster_spike_data_long_form["Rates"] = fr

                    long_form_spike_data = pd.concat([long_form_spike_data, cluster_spike_data_long_form], ignore_index=True)

            # or from time
            elif binning == "time": # add a catch for nans?
                speed = trial_proccessed_position_data['speeds_binned_in_time'].iloc[0]
                pos = trial_proccessed_position_data['pos_binned_in_time'].iloc[0]
                acc = trial_proccessed_position_data['acc_binned_in_time'].iloc[0]
                fr = cluster_spike_data['fr_time_binned'].iloc[0][trial_number-1]

                if len(speed)>0: #only create longform if there is a time binned variable to add
                    cluster_spike_data_long_form = pd.concat([cluster_spike_data]*len(speed), ignore_index=True)
                    del cluster_spike_data_long_form["fr_time_binned"]

                    trial_number_long_form = np.repeat(trial_number, len(fr))
                    trial_type_long_form = np.repeat(trial_proccessed_position_data["trial_type"].iloc[0], len(fr))
                    rewarded_long_form = np.repeat(trial_proccessed_position_data["rewarded"].iloc[0], len(fr))

                    cluster_spike_data_long_form["Trials"] = trial_number_long_form
                    cluster_spike_data_long_form["Trial_Type"] = trial_type_long_form
                    cluster_spike_data_long_form["Rewarded"] = rewarded_long_form
                    cluster_spike_data_long_form["Speeds"] = speed
                    cluster_spike_data_long_form["Position"] = pos
                    cluster_spike_data_long_form["Acceleration"] = acc
                    cluster_spike_data_long_form["Rates"] = fr

                    long_form_spike_data = pd.concat([long_form_spike_data, cluster_spike_data_long_form], ignore_index=True)

    return long_form_spike_data


def delete_other_binning_collumn(spike_data, binning):
    if binning == "space":
        del spike_data['fr_time_binned']
    elif binning == "time":
        del spike_data['fr_binned_in_space']
        del spike_data['fr_binned_in_space_bin_centres']
    return spike_data

def process_dir(recordings_path, concatenated_spike_data=None, save_path=None, binning="space"):

    """
    Creates a long form dataset with spike data for ramp analysis and modelling

    :param recordings_path: path for a folder with all the recordings you want to process
    :param concatenated_spike_data: a pandas dataframe to append all processsed spike data to
    :param save_path: where we save the new processed spike data
    :param binning: "time" or "space": depends if you want the longform to include the space binned or time binned data
    :return: processed spike data in longform
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
            if os.path.exists(spatial_dataframe_path):
                processed_position_data = pd.read_pickle(spatial_dataframe_path)

                # look for key collumns needs for ramp amalysis
                if ("fr_time_binned" in list(spike_data)) or ("fr_binned_in_space" in list(spike_data)):

                    # load mouse id
                    spike_data["mouse_id"] = np.repeat(mouse_id, len(spike_data))

                    # drop any heavy collumns in the dataframe for housekeeping

                    columns_to_drop = ['firing_times', 'trial_number', 'trial_type',
                                       'all_snippets', 'random_snippets', 'speed_per200ms',
                                       'x_position_cm', 'beaconed_position_cm', 'beaconed_trial_number',
                                       'nonbeaconed_position_cm', 'nonbeaconed_trial_number', 'probe_position_cm',
                                       'probe_trial_number', 'beaconed_firing_rate_map', 'non_beaconed_firing_rate_map',
                                       'probe_firing_rate_map', 'beaconed_firing_rate_map_sem', 'non_beaconed_firing_rate_map_sem',
                                       'probe_firing_rate_map_sem', 'tetrode', 'primary_channel', 'isolation', 'noise_overlap',
                                       'peak_snr','peak_amp', 'number_of_spikes', 'mean_firing_rate', 'mean_firing_rate_local',
                                       'ThetaPower', 'ThetaIndex', 'Boccara_theta_class']

                    for column in columns_to_drop:
                        if column in list(spike_data):
                            del spike_data[column]
                    spike_data = delete_other_binning_collumn(spike_data, binning)

                    # create longform dataframe
                    spike_data_long_form = make_longform(spike_data, processed_position_data, binning=binning)
                    concatenated_spike_data = pd.concat([concatenated_spike_data, spike_data_long_form], ignore_index=True)
                else:
                    if (len(spike_data)==0):
                        print("this recording has no units, ", recording.split("/")[-1])
                    else:
                        print("could not find correct binned collumn in recording ", recording.split("/")[-1])

            else:
                print("couldn't find processed_position for ", recording.split("/")[-1])

    if save_path is not None:
        concatenated_spike_data.to_pickle(save_path+binning+"_binned_concatenated_spike_data.pkl")
        concatenated_spike_data.to_csv(save_path+binning+"_binned_concatenated_spike_data.csv")

    return concatenated_spike_data

#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    spike_data = process_dir(recordings_path= "/mnt/datastore/Harry/Cohort7_october2020/vr", concatenated_spike_data=None,
                             save_path= "/mnt/datastore/Harry/Ramp_cells_open_field_paper/", binning="time")

    print("wve done one")
    spike_data = process_dir(recordings_path= "/mnt/datastore/Harry/Cohort7_october2020/vr", concatenated_spike_data=None,
                             save_path= "/mnt/datastore/Harry/Ramp_cells_open_field_paper/", binning="space")


    spike_data = pd.read_pickle("/mnt/datastore/Harry/Ramp_cells_open_field_paper/time_binned_concatenated_spike_data.pkl")
    print("were done for now ")

if __name__ == '__main__':
    main()





