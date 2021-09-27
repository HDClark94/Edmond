import pandas as pd
import os

def print_folder_paths(foldeer_path):
    folder_list = [f.path for f in os.scandir(foldeer_path) if f.is_dir()]
    for i in range(len(folder_list)):
        print(folder_list[i])

def copy_dir_structure(path_to_folder_to_copy, path_to_copy_to):
    # eg. copy_dir_structure(path_to_folder_to_copy = "/mnt/datastore/Junji/",
    # path_to_copy_to = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/theta_index_figs/Junji")

    inputpath = path_to_folder_to_copy
    outputpath = path_to_copy_to

    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exist!")

    print("I hope this worked")
    # this function actually works

def remove_all_snippets_collumn(recordings_path):
    of_path_list = [f.path for f in os.scandir(recordings_path) if f.is_dir()]
    for recording in of_path_list:

        if recording == "/mnt/datastore/Harry/cohort7_october2020/of/M8_D15_2020-11-16_17-27-04":
            print("stop here")
        try:
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")



            if "all_snippets" in list(spike_data):
                del spike_data['all_snippets']
            spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

        except:
            print("couldn't load spike data")

    print("done")


def delete_files_with_extension(path_to_folder, extension):
    # eg. delete_files_with_extension(path_to_folder = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/", extension = ".spikes")
    inputpath = path_to_folder

    for dirpath, dirnames, filenames in os.walk(inputpath):
        for name in filenames:
            if os.path.join(dirpath, name).endswith(extension):
                print("I want to delete this file:, " +os.path.join(dirpath, name))
                os.remove(os.path.join(dirpath, name))
    print("I hope this worked")
    # this function actually works

def rename_channel_paths(path):
    of_path_list = [f.path for f in os.scandir(path) if f.is_dir()]

    for dir in of_path_list:
        filenames = [f.path for f in os.scandir(dir) if f.is_file()]
        for filename in filenames:
            if filename.split("/")[-1].split("_")[0] == "101":
                print("yep")
                new_filename = filename.split("101")[0]+"100"+filename.split("101")[-1]
                os.rename(filename, new_filename)

def find_paired_recording(recording_path, of_recording_path_list):
    mouse=recording_path.split("/")[-1].split("_")[0]
    training_day=recording_path.split("/")[-1].split("_")[1]

    for paired_recording in of_recording_path_list:
        paired_mouse=paired_recording.split("/")[-1].split("_")[0]
        paired_training_day=paired_recording.split("/")[-1].split("_")[1]

        if (mouse == paired_mouse) and (training_day == paired_training_day):
            return paired_recording, True
    return None, False

def print_paired_spatial_firing_clusters(vr_recording_path_list, of_recording_path_list):
    for recording in vr_recording_path_list:
        print("processing ", recording)
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        spike_data_vr = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
        if found_paired_recording:
            print("found paired, ", paired_recording)
            if os.path.isfile(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl"):
                spike_data_of =  pd.read_pickle(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl")
                print(spike_data_vr["cluster_id"].tolist())
                print(spike_data_of["cluster_id"].tolist())
    return

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    path_to_folder = ""
    #delete_files_with_extension(path_to_folder, extension=".spikes")

    a = pd.read_pickle("/mnt/datastore/Harry/Cohort8_may2021/of/M13_D5_2021-05-14_11-34-47/MountainSort/DataFrames/spatial_firing.pkl")
    print("hi")

if __name__ == '__main__':
    main()