import pandas as pd
from Edmond import sorting_comparison
import os
from Edmond.plotting import *
import numpy as np
import sys
import traceback

def find_match(recording_path, recording_path_list):
    # return the matching recording_path in the list which contains the same mouse and day id
    id = recording_path.split("/")[-1]
    mouse_and_day = id.split("_20")[0].split("/")[-1] + "_20"
    matching_of_path = list(filter(lambda x: mouse_and_day in x, recording_path_list))[0]
    return matching_of_path

def run_agreement_assay(agreement_thresholds, window_sizes, sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path, save_path, figs_path=None):

    # create list of recordings in sorted folders
    sorted_together_vr_list = [f.path for f in os.scandir(sorted_together_vr_dir_path) if f.is_dir()]
    sorted_apart_vr_list = [f.path for f in os.scandir(sorted_apart_vr_dir_path) if f.is_dir()]
    sorted_together_of_list = [f.path for f in os.scandir(sorted_together_of_dir_path) if f.is_dir()]
    dataframe_subpath = "/MountainSort/DataFrames/spatial_firing.pkl"

    totals = np.zeros((len(agreement_thresholds), len(window_sizes)))

    for m in range(len(agreement_thresholds)):
        for n in range(len(window_sizes)):

            concat_agreeement = pd.DataFrame()
            concat_agreement_stats = pd.DataFrame()
            # loop over these lists
            for i in range(len(sorted_together_vr_list)):
                try:
                    sorted_together_vr = sorted_together_vr_list[i]

                    #look for matching recording
                    sorted_together_of = find_match(sorted_together_vr, sorted_together_of_list)
                    sorted_apart_vr = find_match(sorted_together_vr, sorted_apart_vr_list)

                    print("Starting sorting comparison between ", sorted_together_vr, " and ", sorted_apart_vr)

                    # agreement is a dataframe with cluster ids and agreement percentages between putative matches
                    agreement, agreement_stats = sorting_comparison.correlation(sorted_together_path=sorted_together_vr + dataframe_subpath,
                                                                                sorted_seperately_path=sorted_apart_vr + dataframe_subpath,
                                                                                title_tag="VR",
                                                                                return_agreement=True,
                                                                                plot=True,
                                                                                figs_path=figs_path,
                                                                                agreement_threshold=agreement_thresholds[m],
                                                                                autocorr_windowsize=window_sizes[n])

                    concat_agreement_stats = pd.concat([concat_agreement_stats, agreement_stats], ignore_index=True)
                    concat_agreeement = pd.concat([concat_agreeement, agreement], ignore_index=True)

                except Exception as ex:
                    print("There was an issue with VR", sorted_together_vr, " and OF ", sorted_together_of)
                    print('This is what Python says happened:')
                    print(ex)
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_tb(exc_traceback)

            totals[m, n] = np.sum(np.sum(concat_agreement_stats["n_agreements"]))

    return totals

def plot_agreement_assay(agreement_thresholds, window_sizes, list_of_totals, figs_path):

    fig, ax = plt.subplots()
    for i in range(len(list_of_totals)):
        for j in range(len(window_sizes)):
            ax.plot(agreement_thresholds, list_of_totals[i][:, j], label="WS = "+ str(np.round(window_sizes[j]*(1/30), decimals=2))+ "ms")

        ax.set_ylim(0, (np.max(list_of_totals[0])+5))
        ax.set_xlim(0, 100)
        ax.set_ylabel("Matched clusters found", fontsize=20)
        ax.set_xlabel("Agreement threshold (% firing matches)", fontsize=20)
        ax.set_title("Agreement threshold assay", fontsize=23)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        fig.tight_layout()
        plt.legend()
        plt.savefig(figs_path+"/agreement_threshold_assay"+str(i+2)+".png")
        plt.show()

        print("figure saved at, ", figs_path)


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # take a list of paths of sorted vr_recordings from Sarah's server space
    # iterate over this list and pick out the spatial dataframes
    figs_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs"
    agreement_thresholds = [0, 0.01, 0.5, 1, 2, 5, 10, 20, 50, 70, 95, 97, 99.5, 100]
    window_sizes = [1, 2, 4, 8, 16, 32]
    '''
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField"
    sorted_apart_vr_dir_path = r"/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M1_sorted"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted"
    totalsM1 = run_agreement_assay(agreement_thresholds, window_sizes,
                                   sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path,
                                   save_path, figs_path=figs_path)

    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M2_sorted"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted"
    totalsM2 = run_agreement_assay(agreement_thresholds, window_sizes,
                                   sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path,
                                   save_path, figs_path=figs_path)
    '''

    # cohort 4
    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M2_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/VirtualReality/M2_sorted"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M2"
    totalsM2 = run_agreement_assay(agreement_thresholds, window_sizes,
                                   sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path,
                                   save_path, figs_path=figs_path)

    sorted_together_vr_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M3_sorted"
    sorted_together_of_dir_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/OpenFeild"
    sorted_apart_vr_dir_path =    "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/VirtualReality/M3_sorted"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M3"
    totalsM3 = run_agreement_assay(agreement_thresholds, window_sizes,
                                   sorted_together_vr_dir_path, sorted_together_of_dir_path, sorted_apart_vr_dir_path,
                                   save_path, figs_path=figs_path)

    plot_agreement_assay(agreement_thresholds, window_sizes, [totalsM2, totalsM3], figs_path)
    #plot_agreement_assay(agreement_thresholds, [totalsM1], figs_path)

    print("look now")

if __name__ == '__main__':
    main()