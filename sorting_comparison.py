import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
'''

This script will compare the sorted results between single sorted recordings compared to recordings sorted together.

'''


def time_to_pacman_level(level_x):
    '''
    The world record, according to Twin Galaxies,
    is currently held by David Race, with the fastest
    completion time of 3 hours, 28 minutes, and 49 seconds
    '''
    hours=3
    minutes=28
    seconds=49
    levels_completed = 256

    total_seconds = (hours*60*60) + (minutes*60) + seconds
    avg_seconds_per_level = total_seconds/levels_completed

    time_til_level_x = avg_seconds_per_level*level_x
    print("It would take ", time_til_level_x, " seconds to get to level ", level_x)
    time_mins = time_til_level_x/60/60
    time_hrs = time_mins/60
    time_days = time_hrs/24
    time_years = time_days/365.25
    print("It would take ", int(time_years), " years to get to level ", level_x)
    time_million_years = time_years/1000000
    print("It would take ", int(time_million_years), " million years to get to level ", level_x)
    time_trillion_years = time_million_years/1000
    print("It would take ", int(time_trillion_years), " trillion years to get to level ", level_x)

def find_set(a,b):
    return set(a) & set(b)

def reconstruct_firing(firing_times, array_size):
    firing_times = np.array(firing_times, dtype=np.int64)
    reconstructed_firing_array = np.zeros(array_size, dtype=np.int8)
    reconstructed_firing_array[firing_times] = 1
    return np.array(reconstructed_firing_array, dtype=np.int8)

def autocorrelogram(match, sorted_together, sorted_seperately, title_tag="", plot=True):

    cluster_id_i = float(match[0])
    cluster_id_j = float(match[1])
    firing_i = sorted_together["firing_times"][sorted_together["cluster_id"]==cluster_id_i].iloc[0]
    firing_j = sorted_seperately["firing_times"][sorted_seperately["cluster_id"]==cluster_id_j].iloc[0]
    max_len = max(len(firing_i), len(firing_j))

    timesteps_lags = np.arange(-10, 10, 1) # were looking at 10 timesteps back and 10 forward
    autocorrelogram = np.array([])

    for lag in timesteps_lags:
        correlated = len(find_set(firing_i+lag, firing_j))/max_len
        autocorrelogram = np.append(autocorrelogram, correlated)

    if plot:
        fig, ax = plt.subplots()
        im = ax.bar(timesteps_lags, autocorrelogram, align='center', width=1, color='black')
        ax.set_xlabel("Lag (n Timesteps)")
        ax.set_ylabel("Proportion Firing Time Matches")
        ax.set_title(title_tag+", Autocorrelogram, ST Cluster ID: "+str(cluster_id_i)+ \
                     ", SS Cluster ID: "+str(cluster_id_j))
        ax.set_ylim(0,1)
        fig.tight_layout()
        plt.show()

    return np.sum(autocorrelogram)

def correlation(sorted_together_path, sorted_seperately_path, title_tag=None):

    if os.path.exists(sorted_together_path):
        sorted_together = pd.read_pickle(sorted_together_path)
        sorted_together = sorted_together[["firing_times", "cluster_id"]]

    if os.path.exists(sorted_seperately_path):
        sorted_seperately = pd.read_pickle(sorted_seperately_path)
        sorted_seperately = sorted_seperately[["firing_times", "cluster_id"]]

    putative_matches = []

    n_clusters_i = len(sorted_together["cluster_id"])
    n_clusters_j = len(sorted_seperately["cluster_id"])

    correlation_matrix = np.zeros((n_clusters_i, n_clusters_j))
    correlation_coef_matrix = np.zeros((n_clusters_i, n_clusters_j))
    firing_event_ratio_matrix = np.zeros((n_clusters_i, n_clusters_j))

    for i in range(n_clusters_i):
        for j in range(n_clusters_j):
            cluster_id_i = sorted_together["cluster_id"].iloc[i]
            cluster_id_j = sorted_seperately["cluster_id"].iloc[j]
            len1 = len(sorted_together["firing_times"].iloc[i])
            len2 = len(sorted_seperately["firing_times"].iloc[j])
            firing_event_ratio = len1/len2
            max_len = max(len1, len2)
            max_firing_ind = int(max(max(sorted_together["firing_times"].iloc[i]),
                                 max(sorted_seperately["firing_times"].iloc[j])))+1

            firing_sorted_together = reconstruct_firing(sorted_together["firing_times"].iloc[i], max_firing_ind)
            firing_sorted_separately = reconstruct_firing(sorted_seperately["firing_times"].iloc[j], max_firing_ind)
            #correlation_coef_matrix[i,j] = np.corrcoef(firing_sorted_together, firing_sorted_separately)[1,0]
            firing_event_ratio_matrix[i,j] = firing_event_ratio
            matched = set(sorted_together["firing_times"].iloc[i]) & set(sorted_seperately["firing_times"].iloc[j])

            if len(matched)>0:
                match_percent = (len(matched)/max_len)*100
            else:
                match_percent = 0
            #correlation_matrix[i,j] = match_percent

            if match_percent>2:
                putative_matches.append([cluster_id_i, cluster_id_j])


            #---------------------------------------------------------------------------------------------------#
            # trying this
            sum_of_lags = autocorrelogram([cluster_id_i, cluster_id_j], sorted_together, sorted_seperately, title_tag, plot=False)
            correlation_matrix[i,j] = sum_of_lags*100
            #----------------------------------------------------------------------------------------------------#

    # ------------------------------------------------------------------------------ #
    fig, ax = plt.subplots()
    im= ax.imshow(correlation_matrix)
    ax.set_xticks(np.arange(n_clusters_j))
    ax.set_yticks(np.arange(n_clusters_i))
    ax.set_yticklabels(sorted_together["cluster_id"].values)
    ax.set_xticklabels(sorted_seperately["cluster_id"].values)
    ax.set_ylabel("Sorted together cluster IDs")
    ax.set_xlabel("Sorted Separately cluster IDs")
    #ax.set_xlim(-10,10)

    for i in range(n_clusters_i):
        for j in range(n_clusters_j):
            text = ax.text(j, i, np.round(correlation_matrix[i, j], decimals=1),
                           ha="center", va="center", color="w")

    if title_tag is not None:
        ax.set_title(title_tag + ", Percentage Matched Firing Times")
    else:
        ax.set_title("Percentage Matched Firing Times")
    fig.tight_layout()
    ax.set_ylim(n_clusters_i-0.5, -0.5)
    plt.show()

    for match in putative_matches:
        _ = autocorrelogram(match, sorted_together, sorted_seperately, title_tag, plot=True)
    # ------------------------------------------------------------------------------ #

def main():

    print('-------------------------------------------------------------')

    server_path = 'Z:\ActiveProjects\Harry\Recordings_waveform_matching'
    dataframe_subpath = "\MountainSort\DataFrames\spatial_firing.pkl"

    vr_sorted_together = server_path+"\M2_D3_2019-03-06_13-35-15"+dataframe_subpath
    of_sorted_together = server_path+"\M2_D3_2019-03-06_15-24-38"+dataframe_subpath

    vr_sorted_seperate = server_path+"\M2_D3_2019-03-06_13-35-15vrsingle"+dataframe_subpath
    of_sorted_seperate = server_path+"\M2_D3_2019-03-06_15-24-38ofsingle"+dataframe_subpath

    correlation(vr_sorted_together, vr_sorted_seperate, title_tag="VR")
    #correlation(of_sorted_together, of_sorted_seperate, title_tag="OF")


    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()