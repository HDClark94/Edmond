import pandas as pd
import os
import traceback
import sys
from Edmond.Concatenate_from_server import get_recording_paths

'''
This script enables quick writing of the parameter file for all recordings
provided the vr and open field recordings are in seperate folders within sub folders named either "vr" or "of"

a parameter helped csv file much be written with headings such as mouse_id, training_day and other optional parameters to be added 
such as track_length and stop_threshold
'''

def get_mouse(session_id):
    return session_id.split("_")[0]

def get_day(session_id):
    tmp =  session_id.split("_")[1]
    tmp = tmp.split("D")[1]
    tmp = ''.join(filter(str.isdigit, tmp))
    return int(tmp)

def get_recording_type(path):
    session_id = path.split("/")[-1]
    sub_folder = path.split("/")[-2]
    if sub_folder == "of" or sub_folder == "OpenFeild" or sub_folder == "OpenField":
        return "openfield"
    elif sub_folder == "vr" or sub_folder == "VirtualReality":
        return "vr"
    else:
        print("something went wrong, this function only works when sub folders are named of or vr")

def search_for_paired(path, paths_to_search_in):
    matched_paired = []

    session_id = path.split("/")[-1]
    mouse_i = get_mouse(session_id)
    day_i = get_day(session_id)

    for j in range(len(paths_to_search_in)):
        path_to_search_in = paths_to_search_in[j]
        paired_recording_type = get_recording_type(path_to_search_in)
        mouse_j = get_mouse(path_to_search_in.split("/")[-1])
        day_j = get_day(path_to_search_in.split("/")[-1])

        if (mouse_i == mouse_j) and (day_i == day_j):
            matched_paired.append(path_to_search_in)

    if len(matched_paired)==0:
        found = False
        return None, None, found
    elif len(matched_paired)==1:
        found = True
        return matched_paired[0], paired_recording_type, found
    else:
        found = True
        print("There appears to be multiple matches with recording, "+path)
        print("you might want to check this manually")
        return matched_paired[0], paired_recording_type, found


def write_param_file(primary_paths, paired_paths, recording_type, parameter_helper):

    for i in range(len(primary_paths)):
        primary_path = primary_paths[i]

        mouse_id = get_mouse(primary_path.split("/")[-1])
        training_day = get_day(primary_path.split("/")[-1])

        path_to_add = primary_path.split("datastore/")[-1]

        file = open(primary_path+"/parameters.txt", "w+")
        file.write(recording_type+"\n")
        file.write(path_to_add+"\n")

        paired_path, paired_recording_type, paired_is_found = search_for_paired(primary_path, paired_paths)

        parameter_helper["mouse_id"]= parameter_helper["mouse_id"].astype(str)
        parameter_helper_mouse_day = parameter_helper[(parameter_helper.mouse_id == mouse_id) &
                                                      (parameter_helper.training_day == training_day)]

        parameters_to_add = []
        if paired_is_found:
            paired_path_to_add = "paired="+paired_path.split("datastore/")[-1]
            parameters_to_add.append(paired_path_to_add)
            parameters_to_add.append("session_type_paired="+paired_recording_type)

        for collumn in list(parameter_helper_mouse_day):
            if (collumn not in ["mouse_id", "training_day"]):
                if recording_type=="vr":
                    if collumn == "track_length":
                        parameters_to_add.append(collumn+"="+str(int(parameter_helper_mouse_day[collumn].iloc[0])))
                    else:
                        parameters_to_add.append(collumn+"="+str(parameter_helper_mouse_day[collumn].iloc[0]))
                else:
                    parameters_to_add.append(collumn+"="+str(parameter_helper_mouse_day[collumn].iloc[0]))

        parameters_to_add = "*".join(parameters_to_add)
        file.write(parameters_to_add+"\n")
        file.close()

def write_dead_channel_file(primary_paths, dead_channel_helper):

    for i in range(len(primary_paths)):
        primary_path = primary_paths[i]
        file = open(primary_path+"/dead_channels.txt", "w+")

        mouse_id = get_mouse(primary_path.split("/")[-1])

        dead_channel_helper["mouse_id"]= dead_channel_helper["mouse_id"].astype(str)
        mouse_dead_channels_str = str(dead_channel_helper[(dead_channel_helper.mouse_id == mouse_id)]["dead_channels"].iloc[0])

        if not mouse_dead_channels_str == "nan":
            mouse_dead_channels_list = mouse_dead_channels_str.split(",")

            for j in range(len(mouse_dead_channels_list)):
                if not mouse_dead_channels_list[j] == "nan":
                    file.write(str(int(float(mouse_dead_channels_list[j])))+"\n")
        file.close()

def write_recording_list_tmp_file(primary_paths, save_path_full, save_path_tmp):
    full_list_txt_file = open(save_path_full, "r")
    path_already_in_list = [line.rstrip('\n') for line in full_list_txt_file]

    file = open(save_path_tmp, "w+")
    for i in range(len(primary_paths)):
        primary_path = primary_paths[i]
        path_to_add = primary_path.split("datastore/")[-1]
        if path_to_add not in path_already_in_list:
            file.write(path_to_add+"\n")

    file.close()


def write_recording_list_file(primary_paths, save_path):

    file = open(save_path, "w+")
    for i in range(len(primary_paths)):
        primary_path = primary_paths[i]
        path_to_add = primary_path.split("datastore/")[-1]
        file.write(path_to_add+"\n")
    file.close()

def print_track_lengths_from_param(vr_paths):

    for path in sorted(vr_paths):
        if os.path.exists(path+"/parameters.txt"):
            a_file = open(path+"/parameters.txt")

            lines = a_file.readlines()
            for line in lines:
                print(line)
    return


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


    # silicon probe mice
    base_path = "/mnt/datastore/Harry/Cohort9_february2023"
    vr_paths = get_recording_paths([], base_path+"/vr")
    of_paths = get_recording_paths([], base_path+"/of")
    #print_track_lengths_from_param(vr_paths)
    parameter_helper = pd.read_csv(base_path+"/parameter_helper.csv")
    dead_channel_helper = pd.read_csv(base_path+"/dead_channel_helper.csv")
    write_param_file(vr_paths, of_paths, recording_type="vr", parameter_helper=parameter_helper)
    write_param_file(of_paths, vr_paths, recording_type="openfield", parameter_helper=parameter_helper)
    write_dead_channel_file(vr_paths, dead_channel_helper=dead_channel_helper)
    write_dead_channel_file(of_paths,dead_channel_helper=dead_channel_helper)
    write_recording_list_file(vr_paths, save_path=base_path+"/vr/full_list.txt")
    write_recording_list_file(of_paths, save_path=base_path+"/of/full_list.txt")


    # junjis mice
    base_path = "/mnt/datastore/Harry/Cohort9_Junji"
    vr_paths = get_recording_paths([], base_path+"/vr")
    of_paths = get_recording_paths([], base_path+"/of")
    #print_track_lengths_from_param(vr_paths)
    parameter_helper = pd.read_csv(base_path+"/parameter_helper.csv")
    dead_channel_helper = pd.read_csv(base_path+"/dead_channel_helper.csv")
    write_param_file(vr_paths, of_paths, recording_type="vr", parameter_helper=parameter_helper)
    write_param_file(of_paths, vr_paths, recording_type="openfield", parameter_helper=parameter_helper)
    write_dead_channel_file(vr_paths, dead_channel_helper=dead_channel_helper)
    write_dead_channel_file(of_paths,dead_channel_helper=dead_channel_helper)
    write_recording_list_file(vr_paths, save_path=base_path+"/vr/full_list.txt")


    base_path = "/mnt/datastore/Harry/Cohort6_july2020"
    vr_paths = get_recording_paths([], base_path+"/vr")
    of_paths = get_recording_paths([], base_path+"/of")
    parameter_helper = pd.read_csv(base_path+"/parameter_helper.csv")
    dead_channel_helper = pd.read_csv(base_path+"/dead_channel_helper.csv")
    write_param_file(vr_paths, of_paths, recording_type="vr", parameter_helper=parameter_helper)
    write_param_file(of_paths, vr_paths, recording_type="openfield", parameter_helper=parameter_helper)
    write_dead_channel_file(vr_paths, dead_channel_helper=dead_channel_helper)
    write_dead_channel_file(of_paths,dead_channel_helper=dead_channel_helper)
    write_recording_list_file(vr_paths, save_path=base_path+"/vr/full_list.txt")

    # october2020 experiment
    base_path = "/mnt/datastore/Harry/Cohort7_october2020"
    vr_paths = get_recording_paths([], base_path+"/vr")
    of_paths = get_recording_paths([], base_path+"/of")
    parameter_helper = pd.read_csv(base_path+"/parameter_helper.csv")
    dead_channel_helper = pd.read_csv(base_path+"/dead_channel_helper.csv")
    write_param_file(vr_paths, of_paths, recording_type="vr", parameter_helper=parameter_helper)
    write_param_file(of_paths, vr_paths, recording_type="openfield", parameter_helper=parameter_helper)
    write_dead_channel_file(vr_paths, dead_channel_helper=dead_channel_helper)
    write_dead_channel_file(of_paths,dead_channel_helper=dead_channel_helper)
    write_recording_list_file(vr_paths, save_path=base_path+"/vr/full_list.txt")

    # may2021 experiment
    base_path = "/mnt/datastore/Harry/Cohort8_may2021"
    vr_paths = get_recording_paths([], base_path+"/vr")
    of_paths = get_recording_paths([], base_path+"/of")
    parameter_helper = pd.read_csv(base_path+"/parameter_helper.csv")
    dead_channel_helper = pd.read_csv(base_path+"/dead_channel_helper.csv")
    write_param_file(vr_paths, of_paths, recording_type="vr", parameter_helper=parameter_helper)
    write_param_file(of_paths, vr_paths, recording_type="openfield", parameter_helper=parameter_helper)
    write_dead_channel_file(vr_paths, dead_channel_helper=dead_channel_helper)
    write_dead_channel_file(of_paths,dead_channel_helper=dead_channel_helper)
    write_recording_list_file(vr_paths, save_path=base_path+"/vr/full_list.txt")


if __name__ == '__main__':
    main()