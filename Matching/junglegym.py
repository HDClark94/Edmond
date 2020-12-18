import pandas as pd
from Edmond import sorting_comparison
import os
from Edmond.plotting import *
import numpy as np
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def get_id(recording_path):
    id = recording_path.split("/")[-1]
    return id

def find_match(recording_path, recording_path_list):
    # return the matching recording_path in the list which contains the same mouse and day id
    id = get_id(recording_path)
    mouse_and_day = id.split("_20")[0].split("/")[-1] + "_20"
    found = None
    try:
        matching_of_path = list(filter(lambda x: mouse_and_day in x, recording_path_list))[0]
        return matching_of_path
    except Exception as ex:
        print("couldn't find match for ", id)
        matching_of_path = None
        return matching_of_path


def add_open_field_stuff_to_agreement(agreement_dataframe, sorted_together_of_path):

    if os.path.exists(sorted_together_of_path):
        sorted_together_of = pd.read_pickle(sorted_together_of_path)

        of_session_id = sorted_together_of_path.split("/")[-4]
        full_of_session_id = "/".join(sorted_together_of_path.split("/")[0:-3])

    new_agreement_dataframe = pd.DataFrame()
    for index, row in agreement_dataframe.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        sorted_together_cluster_id = row["sorted_together_vr_cluster_ids"].iloc[0]
        cluster_of = sorted_together_of[(sorted_together_of["cluster_id"]) == sorted_together_cluster_id]

        cluster_of = cluster_of[['mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'peak_amp',
                        'speed_score', 'speed_score_p_values', 'max_firing_rate_hd', 'preferred_HD',
                        'hd_score', 'rayleigh_score', 'grid_spacing', 'field_size', 'grid_score',
                        'border_score', 'corner_score', 'rate_map_correlation_first_vs_second_half',
                        'percent_excluded_bins_rate_map_correlation_first_vs_second_half_p',
                        'hd_correlation_first_vs_second_half', 'hd_correlation_first_vs_second_half_p', "ThetaIndex", "ThetaPower"]].reset_index(drop=True)

        cluster_of["of_session_id"] = of_session_id
        cluster_of["full_of_session_id"] = full_of_session_id

        cluster_of = row.join(cluster_of)
        new_agreement_dataframe = pd.concat([new_agreement_dataframe,
                                             cluster_of], ignore_index=True)

    return new_agreement_dataframe

def run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, sorted_apart_of_dir_path,
                  save_path, figs_path=None, add_of_and_save=False, cohort_mouse=None, agreement_threshold=90, autocorr_windowsize=4):

    # create list of recordings in sorted folders
    sorted_together_vr_list = [f.path for f in os.scandir(sorted_together_vr_dir_path) if f.is_dir()]
    sorted_apart_vr_list = [f.path for f in os.scandir(sorted_apart_vr_dir_path) if f.is_dir()]
    sorted_together_of_list = [f.path for f in os.scandir(sorted_together_of_dir_path) if f.is_dir()]
    sorted_apart_of_list = [f.path for f in os.scandir(sorted_apart_of_dir_path) if f.is_dir()]

    dataframe_subpath = "/MountainSort/DataFrames/spatial_firing.pkl"
    dataframe_subpath_alt = "/MountainSort/DataFrames/spatial_firing_all.pkl"
    concat_agreement = pd.DataFrame()
    concat_agreement_stats = pd.DataFrame()

    # loop over these lists
    for i in range(len(sorted_together_vr_list)):
        try:
            sorted_together_vr = sorted_together_vr_list[i]
            print("Starting sorting comparison between ", get_id(sorted_together_vr))

            #look for matching recording
            sorted_together_of = find_match(sorted_together_vr, sorted_together_of_list)
            sorted_apart_of = find_match(sorted_together_vr, sorted_apart_of_list)
            sorted_apart_vr = find_match(sorted_together_vr, sorted_apart_vr_list)

            if (sorted_together_of is not None) and (sorted_apart_of is not None) and (sorted_apart_vr is not None):

                agreement, agreement_stats = sorting_comparison.correlation(sorted_together_vr_path=sorted_together_vr + dataframe_subpath,
                                                                            sorted_seperately_vr_path=sorted_apart_vr + dataframe_subpath_alt,
                                                                            sorted_together_of_path=sorted_together_of + dataframe_subpath,
                                                                            sorted_seperately_of_path=sorted_apart_of + dataframe_subpath,
                                                                            return_agreement=True,
                                                                            plot=True,
                                                                            figs_path=figs_path,
                                                                            agreement_threshold=agreement_threshold,
                                                                            autocorr_windowsize=autocorr_windowsize)

                if add_of_and_save:
                    agreement = add_open_field_stuff_to_agreement(agreement, sorted_together_of + dataframe_subpath)

                concat_agreement_stats = pd.concat([concat_agreement_stats, agreement_stats], ignore_index=True)
                concat_agreement = pd.concat([concat_agreement, agreement], ignore_index=True)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)

    if add_of_and_save:
        concat_agreement.to_pickle(save_path+"_sorting_stats_AT"+str(agreement_threshold)+"_WS"+str(autocorr_windowsize)+".pkl")
        concat_agreement_stats.to_pickle(save_path+"_agreement_stats_AT"+str(agreement_threshold)+"_WS"+str(autocorr_windowsize)+".pkl")

    print("I have finished finding agreements")
    return concat_agreement_stats

def plot_summary_stats(agreement_statistics, figs_path, cohort_mouse):

    fig, ax = plt.subplots()
    x_pos = [1,2,3,4]
    totals = [np.sum(agreement_statistics["n_putative_splits_together_vr"]),
              np.sum(agreement_statistics["n_putative_splits_seperately_vr"]),
              np.sum(agreement_statistics["n_putative_splits_together_of"]),
              np.sum(agreement_statistics["n_putative_splits_seperately_of"])]

    ax.bar(x_pos, totals, align='center',color="navy")
    ax.set_xlabel('Sorting Method', fontsize=20)
    ax.set_ylabel('Total Putative Splits', fontsize=20)
    ax.set_title('Sorting Totals', fontsize=23)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([r"VR+OF$_{VR}$", r"VR$_{VR+OF}$", r"VR+OF$_{OF}$", r"OF$_{VR+OF}$"])
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"stats_total.png")
    plt.show()



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # take a list of paths of sorted vr_recordings from Sarah's server space
    # iterate over this list and pick out the spatial dataframes
    figs_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/auco_matches"
    agreement_thres=20
    window_allo=4

    # cohort 5
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField"
    sorted_apart_vr_dir_path = r"/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M1_sorted"
    sorted_apart_of_dir_path = r"/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/OpenField"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M1"
    C5M1 = run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, sorted_apart_of_dir_path, save_path,
                  figs_path=figs_path, add_of_and_save=True, cohort_mouse="C5_M1", agreement_threshold=agreement_thres, autocorr_windowsize=window_allo)


    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M2_sorted"
    sorted_apart_of_dir_path = r"/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/OpenField"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M2"
    C5M2 = run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, sorted_apart_of_dir_path, save_path,
                  figs_path=figs_path, add_of_and_save=True, cohort_mouse="C5_M2", agreement_threshold=agreement_thres, autocorr_windowsize=window_allo)
    
    # cohort 4
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M2_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/VirtualReality/M2_sorted"
    sorted_apart_of_dir_path = r"/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/OpenFeild"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M2"
    C4M2 = run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, sorted_apart_of_dir_path, save_path,
                  figs_path=figs_path, add_of_and_save=True, cohort_mouse="C4_M2", agreement_threshold=agreement_thres, autocorr_windowsize=window_allo)
    
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M3_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/VirtualReality/M3_sorted"
    sorted_apart_of_dir_path = r"/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/OpenFeild"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M3"
    C4M3 = run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, sorted_apart_of_dir_path, save_path,
                  figs_path=figs_path, add_of_and_save=True, cohort_mouse="C4_M3", agreement_threshold=agreement_thres, autocorr_windowsize=window_allo)

    # cohort 3
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M1_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/VirtualReality/M1_sorted"
    sorted_apart_of_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/OpenFeild"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M1"
    C3M1 = run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, sorted_apart_of_dir_path, 
                  save_path, figs_path=figs_path, add_of_and_save=True, cohort_mouse="C3_M1", agreement_threshold=agreement_thres, autocorr_windowsize=window_allo)

    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M6_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/VirtualReality/M6_sorted"
    sorted_apart_of_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/OpenFeild"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M6"
    C3M6 = run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, sorted_apart_of_dir_path, 
                  save_path, figs_path=figs_path, add_of_and_save=True, cohort_mouse="C3_M6", agreement_threshold=agreement_thres, autocorr_windowsize=window_allo)
    
    # cohort 2
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/245_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/OpenField"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort2/VirtualReality/245_sorted"
    sorted_apart_of_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort2/OpenField"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/245"
    C2_254 = run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, sorted_apart_of_dir_path,
                  save_path, figs_path=figs_path, add_of_and_save=True, cohort_mouse="C2_254", agreement_threshold=agreement_thres, autocorr_windowsize=window_allo)
    
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/1124_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/OpenField"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort2/VirtualReality/1124_sorted"
    sorted_apart_of_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort2/OpenField"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/1124"
    C2_1124 = run_agreement(sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, sorted_apart_of_dir_path,
                  save_path, figs_path=figs_path, add_of_and_save=True, cohort_mouse="C2_1124", agreement_threshold=agreement_thres, autocorr_windowsize=window_allo)


    C5_M1 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M1_agreement_stats_AT20_WS4.pkl")
    C5_M2 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M2_agreement_stats_AT20_WS4.pkl")

    C4_M2 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M2_agreement_stats_AT20_WS4.pkl")
    C4_M3 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M3_agreement_stats_AT20_WS4.pkl")

    C3_M1 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M1_agreement_stats_AT20_WS4.pkl")
    C3_M6 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M6_agreement_stats_AT20_WS4.pkl")

    C2_245 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/245_agreement_stats_AT20_WS4.pkl")
    C2_1124 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/1124_agreement_stats_AT20_WS4.pkl")

    all_mice = pd.DataFrame()
    all_mice = pd.concat([all_mice, C5_M1], ignore_index=True)
    all_mice = pd.concat([all_mice, C5_M2], ignore_index=True)
    all_mice = pd.concat([all_mice, C4_M2], ignore_index=True)
    all_mice = pd.concat([all_mice, C4_M3], ignore_index=True)
    all_mice = pd.concat([all_mice, C3_M1], ignore_index=True)
    all_mice = pd.concat([all_mice, C3_M6], ignore_index=True)
    all_mice = pd.concat([all_mice, C2_245], ignore_index=True)
    all_mice = pd.concat([all_mice, C2_1124], ignore_index=True)

    plot_agreement_stats(all_mice,
                         figs_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper",
                         agreement_threshold=agreement_thres)

    print("look now")
if __name__ == '__main__':
    main()