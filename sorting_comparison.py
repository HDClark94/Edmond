import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

'''
This script will compare the sorted results between single sorted recordings compared to recordings sorted together.
'''

def find_set(a,b):
    return set(a) & set(b)

def reconstruct_firing(firing_times, array_size):
    firing_times = np.array(firing_times, dtype=np.int64)
    reconstructed_firing_array = np.zeros(array_size, dtype=np.int8)
    reconstructed_firing_array[firing_times] = 1
    return np.array(reconstructed_firing_array, dtype=np.int8)

def autocorrelogram(match, sorted_together, sorted_seperately, title_tag="",
                    plot=True, figs_path=None, session_id=None, autocorr_window_size=30):

    cluster_id_i = float(match[0])
    cluster_id_j = float(match[1])
    firing_i = sorted_together["firing_times"][sorted_together["cluster_id"]==cluster_id_i].iloc[0]
    firing_j = sorted_seperately["firing_times"][sorted_seperately["cluster_id"]==cluster_id_j].iloc[0]
    max_len = max(len(firing_i), len(firing_j))
    max_len = len(firing_j)

    if autocorr_window_size>1:
        timesteps_lags = np.arange(-autocorr_window_size/2, autocorr_window_size/2, 1).astype(int) # were looking at 10 timesteps back and 10 forward
    else:
        timesteps_lags = [0]

    autocorrelogram = np.array([])
    for lag in timesteps_lags:
        correlated = len(find_set(firing_i+lag, firing_j))/max_len
        autocorrelogram = np.append(autocorrelogram, correlated)

    if plot:
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(1,1,1)
        im = ax.bar(timesteps_lags, autocorrelogram, align='center', width=1, color='black')
        ax.set_xlabel("Lag (ms)", fontsize=20)
        ax.set_ylabel("Proportion Firing Time Matches", fontsize=20)
        ax.set_title(title_tag + " Cluster ID: "+ str(int(cluster_id_j))+
                     ", VR + OF Cluster ID: " + str(int(cluster_id_i)), fontsize=15)
        ax.set_ylim(0,1)
        ax.set_xlim(-15, 15)
        ax.set_xticks([-15, 0, 15])
        ax.set_xticklabels(["-0.5", "0", "0.5"])
        fig.tight_layout()
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.savefig(figs_path+"/autocorrelogram_"+session_id+".png")
        plt.show()

    return np.sum(autocorrelogram)

def correlation(sorted_together_path, sorted_seperately_path, title_tag=None,
                return_agreement=False, plot=False, figs_path=None, agreement_threshold=90,
                autocorr_windowsize=20):

    if os.path.exists(sorted_together_path):
        sorted_together = pd.read_pickle(sorted_together_path)
        sorted_together = sorted_together[["firing_times", "cluster_id"]]

    if os.path.exists(sorted_seperately_path):
        sorted_seperately = pd.read_pickle(sorted_seperately_path)
        sorted_seperately = sorted_seperately[["firing_times", "cluster_id"]]

    putative_matches = []
    session_ids = []

    session_id = sorted_seperately_path.split("/")[-4]

    n_clusters_i = len(sorted_together["cluster_id"])
    n_clusters_j = len(sorted_seperately["cluster_id"])

    correlation_matrix = np.zeros((n_clusters_i, n_clusters_j))

    for i in range(n_clusters_i):
        for j in range(n_clusters_j):
            cluster_id_i = sorted_together["cluster_id"].iloc[i]
            cluster_id_j = sorted_seperately["cluster_id"].iloc[j]
            len1 = len(sorted_together["firing_times"].iloc[i])
            len2 = len(sorted_seperately["firing_times"].iloc[j])
            max_len = max(len1, len2)

            matched = set(sorted_together["firing_times"].iloc[i]) & set(sorted_seperately["firing_times"].iloc[j])

            #---------------------------------------------------------------------------------------------------#
            # trying this
            sum_of_lags = autocorrelogram([cluster_id_i, cluster_id_j], sorted_together, sorted_seperately, title_tag,
                                          plot=False, figs_path=figs_path, session_id=session_id, autocorr_window_size=autocorr_windowsize)
            correlation_matrix[i,j] = sum_of_lags*100

            if sum_of_lags*100 > agreement_threshold:
                putative_matches.append([cluster_id_i, cluster_id_j])
                session_ids.append([session_id])
            #----------------------------------------------------------------------------------------------------#

    # ------------------------------------------------------------------------------ #
    if plot:
        fig, ax = plt.subplots()
        im= ax.imshow(correlation_matrix, vmin=0, vmax=100)
        ax.set_xticks(np.arange(n_clusters_j))
        ax.set_yticks(np.arange(n_clusters_i))
        ax.set_yticklabels(sorted_together["cluster_id"].values)
        ax.set_xticklabels(sorted_seperately["cluster_id"].values)
        ax.set_ylabel("VR + OF Cluster IDs", fontsize=20)
        ax.set_xlabel(title_tag+" Cluster IDs", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.set_xlim(-10,10)

        for i in range(n_clusters_i):
            for j in range(n_clusters_j):
                text = ax.text(j, i, np.round(correlation_matrix[i, j], decimals=1),
                               ha="center", va="center", color="w", fontsize=10)

        if title_tag is not None:
            ax.set_title("% Matched Firing Times", fontsize=21)
        else:
            ax.set_title("% Matched Firing Times", fontsize=21)
        fig.tight_layout()
        fig.colorbar(im, ax=ax)
        ax.set_ylim(n_clusters_i-0.5, -0.5)
        plt.savefig(figs_path+"/Matches_"+session_id+".png")
        plt.show()


    agreement = pd.DataFrame()
    agreement_statistics = pd.DataFrame()
    agreements = []

    if len(putative_matches)>0:

        agreement['session_id'] = np.array(session_ids)[:,0]
        agreement['sorted_together_cluster_ids'] = np.array(putative_matches)[:,0]
        agreement['sorted_seperately_cluster_ids'] = np.array(putative_matches)[:,1]
        for match in putative_matches:
            agreements.append(autocorrelogram(match, sorted_together, sorted_seperately, title_tag,
                                              plot=True, figs_path=figs_path, session_id=session_id,
                                              autocorr_window_size=autocorr_windowsize))

        agreement["agreement"] = agreements

    agreement_statistics["session_id"] = [session_id]
    agreement_statistics["n_clusters_together"] = [n_clusters_i]
    agreement_statistics["n_clusters_seperately"] = [n_clusters_j]
    agreement_statistics["n_agreements"] = [len(putative_matches)]
    print("sorted_together n = ", n_clusters_i, " and sorted_apart n = ", n_clusters_j)

    if return_agreement:
        return agreement, agreement_statistics

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