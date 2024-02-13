import pandas as pd
import os
import warnings
import matplotlib.pylab as plt
from Edmond.utility_functions.array_manipulations import *
warnings.filterwarnings("ignore")


def plot_field_shuffle_false_assay(recordings_folder_to_process):

    recording_list = [f.path for f in os.scandir(recordings_folder_to_process) if f.is_dir()]
    recording_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36']

    for recording_path in recording_list:
        print("processing ", recording_path)

        shuffle=pd.DataFrame()
        if os.path.isdir(recording_path+r"/MountainSort/DataFrames/shuffles"):
            shuffle_list = [f.path for f in os.scandir(recording_path+r"/MountainSort/DataFrames/shuffles") if f.is_file()]

            for i in range(len(shuffle_list)):
                cluster_shuffle = pd.read_pickle(shuffle_list[i])
                cluster_shuffle = cluster_shuffle[["cluster_id", "peak_power", "rolling_peak_powers", "rolling_peak_sizes"]]
                shuffle = pd.concat([shuffle, cluster_shuffle], ignore_index=False)
            print("I have found a shuffled dataframe")

            if os.path.isfile(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl"):
                spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

                if len(spatial_firing)>0:
                    fig, ax = plt.subplots(figsize=(6,4))

                    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
                        cluster_shuffle_df = shuffle[(shuffle.cluster_id == cluster_id)] # dataframe for that cluster
                        print("For cluster", cluster_id, " there are ", len(cluster_shuffle_df), " shuffles")

                        peak_powers = np.array(cluster_shuffle_df["peak_power"])
                        rolling_powers = pandas_collumn_to_2d_numpy_array(cluster_shuffle_df["rolling_peak_powers"])
                        rolling_peak_sizes = pandas_collumn_to_2d_numpy_array(cluster_shuffle_df["rolling_peak_sizes"])

                        save_path = recording_path + '/MountainSort/Figures/rolling_shuffle_assay'
                        if os.path.exists(save_path) is False:
                            os.makedirs(save_path)

                        #fig, ax = plt.subplots(figsize=(6,4))
                        #ax.fill_between(rolling_peak_sizes[0], np.nanmean(rolling_powers, axis=0)-stats.sem(rolling_powers, axis=0, nan_policy="omit"), np.nanmean(rolling_powers, axis=0)+stats.sem(rolling_powers, axis=0, nan_policy="omit"), color="red",alpha=0.3)
                        ax.plot(rolling_peak_sizes[0], np.nanmean(rolling_powers, axis=0), "-", color="red", alpha=0.5)
                    ax.tick_params(axis='both', which='major', labelsize=20)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    #ax.spines['bottom'].set_visible(False)
                    ax.set_ylim(bottom=0, top=0.1)
                    #ax.set_xlim(left=0.5, right=3.5)
                    #ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
                    #ax.set_yticks([0, 0.5, 1, 1.5])
                    #ax.set_xticklabels(["G", "NG"])
                    fig.tight_layout()
                    plt.subplots_adjust(left=0.25, bottom=0.2)
                    ax.set_xlabel("Rolling window sample size", fontsize=20)
                    ax.set_ylabel("False alarm value", fontsize=20)
                    plt.savefig(save_path + '/false_alarm_vs_rolling_window_size.png', dpi=300)
                    plt.close()

                else:
                    print("There are no cells in this recordings")
            else:
                print("No spatial firing could be found")

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('The shuffled analysis scripts (on Eddie) used python 3.8 to make the data frames. If you run this and '
          'get an error about pickle protocols, try to make a new python 3.8 virtual environment on Eleanor '
          '(conda create -n environmentname python=3.8) and use that. (The pipeline currently needs 3.6, so do not '
          'change that.')

    folders = []
    folders.append("/mnt/datastore/Harry/Cohort7_october2020/vr")
    folders.append("/mnt/datastore/Harry/Cohort6_july2020/vr")
    folders.append("/mnt/datastore/Harry/Cohort8_may2021/vr")

    for folder in folders:
        plot_field_shuffle_false_assay(folder)
    print("look now")

if __name__ == '__main__':
    main()