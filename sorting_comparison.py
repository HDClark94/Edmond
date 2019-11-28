import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
'''

This script will compare the sorted results between single sorted recordings compared to recordings sorted together.

'''

def correlation(sorted_together_path, sorted_seperately_path, title_tag=None):

    if os.path.exists(sorted_together_path):
        sorted_together = pd.read_pickle(sorted_together_path)
        sorted_together = sorted_together[["firing_times", "cluster_id"]]

    if os.path.exists(sorted_seperately_path):
        sorted_seperately = pd.read_pickle(sorted_seperately_path)
        sorted_seperately = sorted_seperately[["firing_times", "cluster_id"]]

    predicted_matches = []

    n_clusters_i = len(sorted_together["cluster_id"])
    n_clusters_j = len(sorted_seperately["cluster_id"])

    correlation_matrix = np.zeros((n_clusters_i, n_clusters_j))

    for i in range(n_clusters_i):
        for j in range(n_clusters_j):
            cluster_id_i = sorted_together["cluster_id"].iloc[i]
            cluster_id_j = sorted_seperately["cluster_id"].iloc[j]

            matched = set(sorted_together["firing_times"].iloc[i]) & set(sorted_seperately["firing_times"].iloc[j])
            len1 = len(sorted_together["firing_times"].iloc[i])
            len2 = len(sorted_seperately["firing_times"].iloc[j])
            max_len = max(len1, len2)
            if len(matched)>0:
                match_percent = (len(matched)/max_len)*100
            else:
                match_percent = 0
            print("Match of ", match_percent, "% found between Cluster, ",
                  cluster_id_i, "in Sorted together and cluster ", cluster_id_j, " in sorted apart")

            if match_percent>5:
                predicted_matches.append([cluster_id_i,cluster_id_j, match_percent])

            correlation_matrix[i,j] = match_percent


    fig, ax = plt.subplots()
    im= ax.imshow(correlation_matrix)
    # We want to show all ticks...
    ax.set_xticks(np.arange(n_clusters_j))
    ax.set_yticks(np.arange(n_clusters_i))
    # ... and label them with the respective list entries
    ax.set_yticklabels(sorted_together["cluster_id"].values)
    ax.set_xticklabels(sorted_seperately["cluster_id"].values)
    ax.set_ylabel("Sorted together cluster IDs")
    ax.set_xlabel("Sorted Separately cluster IDs")

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_yticklabels(), rotation=45, rotation_mode="anchor")
    #plt.setp(ax.get_xticklabels(), rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
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

    for match in predicted_matches:
        print("I think cluster ", match[0], "from sorted together and cluster ", match[1],
              " from sorted apart are the same cluster, they have ", match[2], "% match")

def main():

    print('-------------------------------------------------------------')

    server_path = 'Z:\ActiveProjects\Harry\Recordings_waveform_matching'
    dataframe_subpath = "\MountainSort\DataFrames\spatial_firing.pkl"

    vr_sorted_together = server_path+"\M2_D3_2019-03-06_13-35-15"+dataframe_subpath
    of_sorted_together = server_path+"\M2_D3_2019-03-06_15-24-38"+dataframe_subpath

    vr_sorted_seperate = server_path+"\M2_D3_2019-03-06_13-35-15vrsingle"+dataframe_subpath
    of_sorted_seperate = server_path+"\M2_D3_2019-03-06_15-24-38ofsingle"+dataframe_subpath

    correlation(vr_sorted_together, vr_sorted_seperate, title_tag="VR")
    correlation(of_sorted_together, of_sorted_seperate, title_tag="OF")


    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()