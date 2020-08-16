import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

'''
This script will compare the sorted results between single sorted recordings compared to recordings sorted together.
'''
def get_n_spikes(cluster_id, spatial_firing):
    cluster_spatial_firing = spatial_firing[spatial_firing["cluster_id"] == cluster_id]
    return len(cluster_spatial_firing["firing_times"].iloc[0])

def get_theta(cluster_id, spatial_firing, type):
    cluster_spatial_firing = spatial_firing[spatial_firing["cluster_id"] == cluster_id]
    if type == "power":
        return cluster_spatial_firing["ThetaPower"].iloc[0]
    elif type == "index":
        return cluster_spatial_firing["ThetaIndex"].iloc[0]

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
        plt.savefig(figs_path+"/autocorrelogram_"+session_id+"_match_"+str(match[0])+"_"+str(match[1])+".png")
        plt.show()

    return np.sum(autocorrelogram)

def correlation(sorted_together_vr_path, sorted_seperately_vr_path,
                sorted_together_of_path, sorted_seperately_of_path,
                return_agreement=False, plot=False, figs_path=None, agreement_threshold=90,
                autocorr_windowsize=20, ignore_non_curated=True):

    agreement = pd.DataFrame()
    agreement_statistics = pd.DataFrame()

    if os.path.exists(sorted_together_vr_path):
        sorted_together_vr_u = pd.read_pickle(sorted_together_vr_path)
        if 'Curated' in list(sorted_together_vr_u) and ignore_non_curated:
            sorted_together_vr = sorted_together_vr_u[sorted_together_vr_u["Curated"]==1]
        else:
            sorted_together_vr = sorted_together_vr_u
    else:
        print("sorted together vr dataframe does not exist, it may not have sorted properly")
        return agreement, agreement_statistics

    if os.path.exists(sorted_seperately_vr_path):
        sorted_seperately_vr = pd.read_pickle(sorted_seperately_vr_path)
    else:
        print("this recording wasn't originally sorted successfully (individually sorted)")
        return agreement, agreement_statistics

    if os.path.exists(sorted_together_of_path):
        sorted_together_of_u = pd.read_pickle(sorted_together_of_path)
        if 'Curated' in list(sorted_together_of_u) and ignore_non_curated:
            sorted_together_of = sorted_together_of_u[sorted_together_of_u["Curated"]==1]
        else:
            sorted_together_of = sorted_together_of_u
    else:
        print("sorted together of dataframe does not exist, it may not have sorted properly")
        return agreement, agreement_statistics

    if os.path.exists(sorted_seperately_of_path):
        sorted_seperately_of = pd.read_pickle(sorted_seperately_of_path)
        sorted_seperately_of = sorted_seperately_of[["firing_times", "cluster_id", "tetrode", "primary_channel"]]
    else:
        print("Sorting seperately of doesnt exist")
        return agreement, agreement_statistics

    if not np.array_equal(np.unique(sorted_together_vr_u["cluster_id"]),
                          np.unique(sorted_together_of_u["cluster_id"])):
        print("this recording wasn't dual sorted successfully")
        return agreement, agreement_statistics


    putative_matches_vr = []
    putative_matches_of = []
    session_ids_vr = []
    full_session_ids_vr = []
    session_ids_of = []

    session_id_vr = sorted_seperately_vr_path.split("/")[-4]
    full_session_id_vr = "/".join(sorted_seperately_vr_path.split("/")[0:-3])
    session_id_of = sorted_seperately_of_path.split("/")[-4]

    n_clusters_i = len(sorted_together_vr["cluster_id"])
    n_clusters_j = len(sorted_seperately_vr["cluster_id"])
    n_clusters_m = len(sorted_together_of["cluster_id"])
    n_clusters_n = len(sorted_seperately_of["cluster_id"])

    correlation_matrix_of = np.zeros((n_clusters_m, n_clusters_n))
    correlation_matrix_vr = np.zeros((n_clusters_i, n_clusters_j))

    # vr
    for i in range(n_clusters_i):
        for j in range(n_clusters_j):
            cluster_id_i = sorted_together_vr["cluster_id"].iloc[i]
            cluster_id_j = sorted_seperately_vr["cluster_id"].iloc[j]
            cluster_tetrode_i = sorted_together_vr["tetrode"].iloc[i]
            cluster_tetrode_j = sorted_seperately_vr["tetrode"].iloc[j]
            cluster_primchan_i = sorted_together_vr["primary_channel"].iloc[i]
            cluster_primchan_j = sorted_seperately_vr["primary_channel"].iloc[j]

            if (cluster_primchan_i == cluster_primchan_j):
                sum_of_lags = autocorrelogram([cluster_id_i, cluster_id_j], sorted_together_vr, sorted_seperately_vr, title_tag="VR",
                                            plot=False, figs_path=figs_path, session_id=session_id_vr, autocorr_window_size=autocorr_windowsize)
            else:
                sum_of_lags = 0 # no firing time matches if cluster found on a different cluster

            correlation_matrix_vr[i,j] = sum_of_lags*100

            if correlation_matrix_vr[i,j] > agreement_threshold:
                putative_matches_vr.append([cluster_id_i, cluster_id_j])
                session_ids_vr.append([session_id_vr])
                full_session_ids_vr.append([full_session_id_vr])

    # of
    for m in range(n_clusters_m):
        for n in range(n_clusters_n):
            cluster_id_m = sorted_together_of["cluster_id"].iloc[m]
            cluster_id_n = sorted_seperately_of["cluster_id"].iloc[n]
            cluster_tetrode_m = sorted_together_of["tetrode"].iloc[m]
            cluster_tetrode_n = sorted_seperately_of["tetrode"].iloc[n]
            cluster_primchan_m = sorted_together_of["primary_channel"].iloc[m]
            cluster_primchan_n = sorted_seperately_of["primary_channel"].iloc[n]

            if (cluster_primchan_m == cluster_primchan_n):
                sum_of_lags = autocorrelogram([cluster_id_m, cluster_id_n], sorted_together_of, sorted_seperately_of, title_tag="OF",
                                              plot=False, figs_path=figs_path, session_id=session_id_of, autocorr_window_size=autocorr_windowsize)
            else:
                sum_of_lags = 0 # no firing time matches if cluster found on a different cluster

            correlation_matrix_of[m,n] = sum_of_lags*100

            if correlation_matrix_of[m,n] > agreement_threshold:
                putative_matches_of.append([cluster_id_m, cluster_id_n])
                session_ids_of.append([session_id_of])


    # ------------------------------------------------------------------------------ #
    if plot:
        # vr
        fig, ax = plt.subplots()
        im= ax.imshow(correlation_matrix_vr, vmin=0, vmax=100)
        ax.set_xticks(np.arange(n_clusters_j))
        ax.set_yticks(np.arange(n_clusters_i))
        ax.set_yticklabels(sorted_together_vr["cluster_id"].values)
        ax.set_xticklabels(sorted_seperately_vr["cluster_id"].values)
        ax.set_ylabel("VR + OF Cluster IDs", fontsize=20)
        ax.set_xlabel("VR Cluster IDs", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10)
        for i in range(n_clusters_i):
            for j in range(n_clusters_j):
                text = ax.text(j, i, np.round(correlation_matrix_vr[i, j], decimals=1),
                               ha="center", va="center", color="w", fontsize=7)
        ax.set_title("% Matched Firing Times", fontsize=21)
        fig.tight_layout()
        fig.colorbar(im, ax=ax)
        ax.set_ylim(n_clusters_i-0.5, -0.5)
        plt.savefig(figs_path+"/Matches_"+session_id_vr+".png")
        plt.show()

        # of
        if os.path.exists(sorted_seperately_of_path):
            fig, ax = plt.subplots()
            im= ax.imshow(correlation_matrix_of, vmin=0, vmax=100)
            ax.set_xticks(np.arange(n_clusters_n))
            ax.set_yticks(np.arange(n_clusters_m))
            ax.set_yticklabels(sorted_together_of["cluster_id"].values)
            ax.set_xticklabels(sorted_seperately_of["cluster_id"].values)
            ax.set_ylabel("VR + OF Cluster IDs", fontsize=20)
            ax.set_xlabel("OF Cluster IDs", fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=15)
            for m in range(n_clusters_m):
                for n in range(n_clusters_n):
                    text = ax.text(n, m, np.round(correlation_matrix_of[m, n], decimals=1),
                                   ha="center", va="center", color="w", fontsize=7)
            ax.set_title("% Matched Firing Times", fontsize=21)
            fig.tight_layout()
            fig.colorbar(im, ax=ax)
            ax.set_ylim(n_clusters_i-0.5, -0.5)
            plt.savefig(figs_path+"/Matches_"+session_id_of+".png")
            plt.show()

    agreements = []
    n_spikes_vr = []
    n_spikes_of = []
    n_spikes_vr_original = []
    split_cluster = []
    theta_vr = []
    thetaP_vr = []
    if len(putative_matches_vr)>0:

        agreement['session_id'] = np.array(session_ids_vr)[:,0]
        agreement['full_session_id'] = np.array(full_session_ids_vr)[:,0]
        agreement['sorted_together_vr_cluster_ids'] = np.array(putative_matches_vr)[:,0]
        agreement['sorted_seperately_vr_cluster_ids'] = np.array(putative_matches_vr)[:,1]
        for match in putative_matches_vr:
            n_spikes_vr_original.append(get_n_spikes(match[1], sorted_seperately_vr))
            n_spikes_of.append(get_n_spikes(match[0], sorted_together_of))
            n_spikes_vr.append(get_n_spikes(match[0], sorted_together_vr))
            theta_vr.append(get_theta(match[0], sorted_together_vr, type="index"))
            thetaP_vr.append(get_theta(match[0], sorted_together_vr, type="power"))
            agreements.append(autocorrelogram(match, sorted_together_vr, sorted_seperately_vr, title_tag="VR",
                                              plot=True, figs_path=figs_path, session_id=session_id_vr,
                                              autocorr_window_size=autocorr_windowsize))
            split_cluster.append(check_for_split_match(match, np.array(putative_matches_vr)))
        agreement["n_spikes_vr"] = n_spikes_vr
        agreement["n_spikes_of"] = n_spikes_of
        agreement["n_spikes_vr_original"] = n_spikes_vr_original
        agreement["ThetaIndex_vr"] = theta_vr
        agreement["ThetaPower_vr"] = thetaP_vr
        agreement["agreement_vr"] = agreements
        agreement["split_cluster"] = split_cluster

    agreement_statistics["session_id_vr"] = [session_id_vr]
    agreement_statistics["session_id_of"] = [session_id_of]
    agreement_statistics["n_clusters_together_vr"] = [n_clusters_i]
    agreement_statistics["n_clusters_seperately_vr"] = [n_clusters_j]
    agreement_statistics["n_clusters_together_of"] = [n_clusters_m]
    if os.path.exists(sorted_seperately_of_path):
        agreement_statistics["n_clusters_seperately_of"] = [n_clusters_n]
    agreement_statistics["n_agreements_vr"] = [len(putative_matches_vr)]
    agreement_statistics["n_agreements_of"] = [len(putative_matches_of)]
    agreement_statistics["n_putative_splits_together_vr"] = [count_split(np.array(putative_matches_vr), together=True)]
    agreement_statistics["n_putative_splits_together_of"] = [count_split(np.array(putative_matches_of), together=True)]
    agreement_statistics["n_putative_splits_seperately_vr"] = [count_split(np.array(putative_matches_vr), together=False)]
    agreement_statistics["n_putative_splits_seperately_of"] = [count_split(np.array(putative_matches_of), together=False)]

    if return_agreement:
        return agreement, agreement_statistics

    # ------------------------------------------------------------------------------ #

def check_for_split_match(match_tuple, putative_matches_list):
    for i in range(len(putative_matches_list)):
        if putative_matches_list[:,0].tolist().count(match_tuple[0]) > 1:
            return True
        elif putative_matches_list[:,1].tolist().count(match_tuple[1]) > 1:
            return True
        else:
            return False

def count_split(cluster_ids_with_match, together=True):
    if (together==True) and len(cluster_ids_with_match>0):
        count = len(cluster_ids_with_match[:,1]) - len(np.unique(cluster_ids_with_match[:,1]))
    elif (together==False) and len(cluster_ids_with_match>0):
        count = len(cluster_ids_with_match[:,0]) - len(np.unique(cluster_ids_with_match[:,0]))
    else:
        count = 0
    return count

def main():

    print('-------------------------------------------------------------')

    dataframe_subpath = "/MountainSort/DataFrames/spatial_firing.pkl"
    agreement_thres=20
    window_allo=4
    figs_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/junelabmeeting"

    vr_sorted_together =  "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted/M2_D11_2019-07-01_13-50-47" + dataframe_subpath
    of_sorted_together = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/OpenField/M2_D11_2019-07-01_14-32-41" + dataframe_subpath

    vr_sorted_seperate = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_D11_2019-07-01_13-50-47" + dataframe_subpath
    of_sorted_seperate = "/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/OpenField/M2_D11_2019-07-01_14-32-41" + dataframe_subpath

    correlation(sorted_together_vr_path = vr_sorted_together,
                sorted_seperately_vr_path = vr_sorted_seperate,
                sorted_together_of_path = of_sorted_together,
                sorted_seperately_of_path = of_sorted_seperate,
                return_agreement=False,
                plot=True,
                figs_path=figs_path,
                agreement_threshold=agreement_thres,
                autocorr_windowsize=window_allo,
                ignore_non_curated=True)

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()