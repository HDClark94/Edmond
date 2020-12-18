import pandas as pd
import os
from Edmond.track_theta import *

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

def delete_files_with_extension(path_to_folder, extension1, extension2):
    # eg. delete_files_with_extension(path_to_folder = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/",
    # path_to_copy_to = ".continuous"
    inputpath = path_to_folder

    for dirpath, dirnames, filenames in os.walk(inputpath):
        for name in filenames:
            if os.path.join(dirpath, name).endswith(extension1):
                print("I want to delete this file:, " +os.path.join(dirpath, name))
                os.remove(os.path.join(dirpath, name))
            elif os.path.join(dirpath, name).endswith(extension2):
                print("I want to delete this file:, " +os.path.join(dirpath, name))
                os.remove(os.path.join(dirpath, name))

    print("I hope this worked")
    # this function actually works

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    delete_files_with_extension("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/", extension1=".continuous", extension2=".spikes")
    '''
    tmp2 = pd.read_pickle("/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D9_2020-11-08_14-37-47/MountainSort/DataFrames/processed_position_data.pkl")

    tmp = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_other_users/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild/M6_D15_2018-10-25_12-48-39_of/MountainSort/DataFrames/lfp_data.pkl")

    processed_PATH = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted/M1_D8_2019-06-26_13-31-11/processed/sorted_df.pkl"
    processed_PATH = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted/M1_D8_2019-06-26_13-31-11/MountainSort/DataFrames/spatial_firing.pkl"
    processed_PATH = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M3_sorted/M3_D11_2019-03-18_12-28-47/MountainSort/DataFrames/spatial_firing.pkl"
    processed_PATH = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M1_sorting_stats.pkl"
    processed_PATH = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M3_agreement_stats_AT20_WS4.pkl"
    processed_PATH = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted/M2_D9_2019-06-27_13-58-07/MountainSort/DataFrames/spatial_firing.pkl"

    processed = pd.read_pickle(processed_PATH)

    ramp_path_lm = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_results_linearmodel.txt"
    ramp_scores_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ramp_score_export.csv"
    tetrode_location_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/tetrode_locations.csv"
    ramp_scores = pd.read_csv(ramp_scores_path)
    ramp_lm = pd.read_csv(ramp_path_lm, sep = "\t")
    tetrode_locations = pd.read_csv(tetrode_location_path)
    theta_df_VR = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta/theta_df_VR.pkl")
    data = add_ramp_scores(theta_df_VR, ramp_lm, ramp_scores, tetrode_locations)

    sorted_b = data.sort_values(by=["max_ramp_score"])
    g = data[(data.session_id=="M1_D30_2018-10-29_12-38-27")]

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    path1 = r"/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M6_sorted/M6_D8_2018-10-21_13-25-59/MountainSort/DataFrames/spatial_firing.pkl"
    path2 = r"/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/245_sorted/245_D18_2018-11-10_11-07-50/MountainSort/DataFrames/spatial_firing.pkl"

    #spatial_firing1 = pd.read_pickle(path1)
    #spatial_firing2 = pd.read_pickle(path2)
    #print_folder_paths(r"/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M3_sorted/")
    '''
    print("look now")


if __name__ == '__main__':
    main()