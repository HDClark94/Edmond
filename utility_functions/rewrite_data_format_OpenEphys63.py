import pandas as pd
import os
import traceback
import sys
import shutil

'''
This script enables quick writing of the parameter file for all recordings
provided the vr and open field recordings are in seperate folders within sub folders named either "vr" or "of"

a parameter helped csv file much be written with headings such as mouse_id, training_day and other optional parameters to be added 
such as track_length and stop_threshold
'''
def get_recording_paths(path_list, folder_path):
    list_of_recordings = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for recording_path in list_of_recordings:
        path_list.append(recording_path)
        print(recording_path.split("/datastore/")[-1])
    return path_list

def bring_out_of_record_node(recording_folder_path, record_node_dir_name):
    # in the case you record using open ephys 6.3
    for recording_path in recording_folder_path:
        print("I am looking at ", recording_path)
        paths_in_recording = [f.path for f in os.scandir(recording_path) if f.is_dir()]
        for path in paths_in_recording:
            if os.path.isdir(path) and record_node_dir_name in path:
                print("I will move all files in the Record Node up a level and delete the Record Node directory")
                paths_in_record_node = [f.path for f in os.scandir(path)]
                for path_in_record_node in paths_in_record_node:
                    new_path = path_in_record_node.replace("/"+record_node_dir_name, "")
                    shutil.move(path_in_record_node, new_path)
                print("I am now deleting the Recording Node if it is empty")
                if len([f.path for f in os.scandir(path)]) == 0:
                    shutil.rmtree(path)
                else:
                    raise Exception('It looks like the Record Node is not empty, something may not have copied correctly')
    return

def rename_channel_files(recording_folder_path, channel_file_name_in, channel_file_name_out):
    # this function looks within the given recording folder directory, loops over each recording, looks for .continuous files and renames
    # them according to the name conventions given in the arguments
    # for example 100_RhythmData_ADC1.continuous with channel_file_name_in=_RhythmData and channel_file_name_out="" will be renamed to 100_ADC1.continuous
    for recording_path in recording_folder_path:
        print("I am looking at ", recording_path)

        for path, subdirs, files in os.walk(recording_path):
            for name in files:
                file_path = os.path.join(path, name)
                if file_path.endswith(".continuous"):
                    if channel_file_name_in in file_path.split("/")[-1]:
                        new_name = file_path.replace(channel_file_name_in, channel_file_name_out)
                        os.rename(file_path, new_name)

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    base_path = "/mnt/datastore/Harry/Cohort10_october2023"
    vr_paths = get_recording_paths([], base_path+"/vr")
    of_paths = get_recording_paths([], base_path+"/of")

    bring_out_of_record_node(vr_paths, record_node_dir_name="Record Node 101")
    bring_out_of_record_node(of_paths, record_node_dir_name="Record Node 101")
    rename_channel_files(vr_paths, channel_file_name_in="_RhythmData", channel_file_name_out="")
    rename_channel_files(of_paths, channel_file_name_in="_RhythmData", channel_file_name_out="")

if __name__ == '__main__':
    main()