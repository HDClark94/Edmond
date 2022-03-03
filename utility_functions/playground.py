import pandas as pd
import os
import shutil

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


def delete_folders_by_name(path_to_folder, name):
    # eg. delete_files_with_extension(path_to_folder = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/", name = "MountainSort")
    inputpath = path_to_folder
    for dirpath, dirnames, filenames in os.walk(inputpath):
        if dirpath.endswith(name):
            print("I want to delete this folder:, " +dirpath)
            shutil.rmtree(dirpath)
    print("I hope this worked")


def delete_files_with_extension(path_to_folder, extension):
    # eg. delete_files_with_extension(path_to_folder = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/", extension = ".spikes")
    inputpath = path_to_folder

    for dirpath, dirnames, filenames in os.walk(inputpath):
        for name in filenames:
            if os.path.join(dirpath, name).endswith(extension):
                print("I want to delete this file:, " +os.path.join(dirpath, name))
                os.remove(os.path.join(dirpath, name))
    #print("I hope this worked")
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

def rename_files(path_list, sub_out, sub_in):

    for path in path_list:

        file_list = [f.path for f in os.scandir(path) if f.is_file()]
        for file in file_list:
            file_name_old = file
            file_name_new = file_name_old.replace(sub_out, sub_in)

            if file_name_old != file_name_new:
                os.rename(file_name_old, file_name_new)

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


    #delete_files_with_extension(path_to_folder="", extension="_shuffle.pkl")

    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_Junji/vr") if f.is_dir()]
    J1_list = [k for k in vr_path_list if 'J1' in k]
    J2_list = [k for k in vr_path_list if 'J2' in k]
    J3_list = [k for k in vr_path_list if 'J3' in k]
    J4_list = [k for k in vr_path_list if 'J4' in k]
    J5_list = [k for k in vr_path_list if 'J5' in k]
    vr_path_list = []
    #vr_path_list.extend(J1_list)
    #vr_path_list.extend(J2_list)
    #vr_path_list.extend(J3_list)
    #vr_path_list.extend(J4_list)
    #vr_path_list.extend(J5_list)

    #for path in vr_path_list:
        #delete_files_with_extension(path_to_folder=path, extension='100_ADC1.continuous') # of sync pulse
        #delete_files_with_extension(path_to_folder=path, extension='100_ADC2.continuous') # vr movement
        #delete_files_with_extension(path_to_folder=path, extension='100_ADC3.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_ADC4.continuous') # vr trial pin1
        #delete_files_with_extension(path_to_folder=path, extension='100_ADC5.continuous') # vr trial pin2
        #delete_files_with_extension(path_to_folder=path, extension='100_ADC6.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_ADC7.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_ADC8.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_AUX1.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_AUX2.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_AUX3.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH1.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH2.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH3.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH4.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH5.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH6.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH7.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH8.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH9.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH10.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH11.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH12.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH13.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH14.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH15.continuous')
        #delete_files_with_extension(path_to_folder=path, extension='100_CH16.continuous')

    #rename_files(vr_path_list, sub_out="101", sub_in="100")


    #for folder in ["/mnt/datastore/Harry/Cohort7_october2020/of",
    #               "/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort5/OpenField",
    #               "/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort4/OpenFeild",
    #               "/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort3/OpenFeild",
    #               "/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort2/OpenField"]:

    #    of_path_list = [f.path for f in os.scandir(folder) if f.is_dir()]

    #    for path in of_path_list:
    #        delete_files_with_extension(path_to_folder=path, extension="/shuffle.pkl")


    print("hi")

if __name__ == '__main__':
    main()