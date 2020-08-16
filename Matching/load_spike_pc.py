import pandas as pd
import OpenEphys as oe
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

n_comp = 10
whiten=True
seed = 42

def tetrode2str(tetrode_int):
    if tetrode_int == 1:
        return "A"
    elif tetrode_int == 2:
        return "B"
    elif tetrode_int == 3:
        return "C"
    elif tetrode_int == 4:
        return "D"


def plot_snippet_pc(spatial_firing_1, figs_path, max_spikes_pca=1000000):
    spatial_firing_1 = spatial_firing_1.reset_index()
    fig_combos = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]

    for tetrode in [1,2,3,4]:
        print("looking at tetrode ", tetrode)
        all_waveforms = None

        fig , axs = plt.subplots(2, 3)
        m = 0
        for channel_combo in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
            fig_combo = fig_combos[m]

            pca = PCA(n_components=n_comp, whiten=whiten, random_state=seed)
            indexes=[]
            for i, cluster_row in spatial_firing_1.iterrows():
                cluster_tetrode = cluster_row["tetrode"]

                cluster_snippets_a = np.asarray(cluster_row["all_snippets"])[channel_combo[0], :, :]
                cluster_snippets_b = np.asarray(cluster_row["all_snippets"])[channel_combo[1], :, :]

                cluster_snippets_ab = np.dstack((cluster_snippets_a, cluster_snippets_b))
                cluster_snippets_ab = cluster_snippets_ab.reshape(cluster_snippets_ab.shape[2],
                                                                  cluster_snippets_ab.shape[0],
                                                                  cluster_snippets_ab.shape[1])

                if cluster_tetrode == tetrode:
                    if len(cluster_snippets_ab[0,0]) > max_spikes_pca:
                        cluster_snippets_ab = cluster_snippets_ab[:, :, :max_spikes_pca]
                        indexes.append(int(max_spikes_pca))
                    else:
                        indexes.append(int(len(cluster_snippets_ab[0,0])))

                    if len(indexes)==1:
                        all_waveforms = cluster_snippets_ab
                    else:
                        all_waveforms = np.concatenate((all_waveforms, cluster_snippets_ab), axis=2)

            if all_waveforms is not None:
                indexes = np.cumsum(indexes)
                all_waveforms = all_waveforms.reshape(-1, all_waveforms.shape[-1])
                #all_waveforms = StandardScaler().fit_transform(all_waveforms)
                pca.fit(all_waveforms)
                print(pca.explained_variance_ratio_, "= explained_variance_ratio")

                # now plot
                j=0; n=0
                for _, cluster_row in spatial_firing_1.iterrows():
                    if cluster_row["tetrode"] == tetrode:
                        axs[fig_combo[0], fig_combo[1]].scatter(pca.components_[0, j:indexes[n]],pca.components_[1, j:indexes[n]],
                                                                marker="x", label="ID: "+str(cluster_row["cluster_id"]))
                        j=indexes[n]
                        n+=1

                axs[fig_combo[0], fig_combo[1]].text(0.2, 0.85, tetrode2str(tetrode)+str(channel_combo[0]+1),
                                                       verticalalignment='bottom', horizontalalignment='right',
                                                       transform=axs[fig_combo[0], fig_combo[1]].transAxes)
                axs[fig_combo[0], fig_combo[1]].text(0.95, 0.05, tetrode2str(tetrode)+str(channel_combo[1]+1),
                                                     verticalalignment='bottom', horizontalalignment='right',
                                                     transform=axs[fig_combo[0], fig_combo[1]].transAxes)

                axs[fig_combo[0], fig_combo[1]].set_yticklabels([])
                axs[fig_combo[0], fig_combo[1]].set_xticklabels([])
                if m==5:
                    axs[fig_combo[0], fig_combo[1]].legend(loc=(1.05,0.95))



            m+=1

        axs[0,0].set(ylabel='PC 2')
        axs[1,0].set(xlabel='PC 1', ylabel='PC 2')
        axs[1,1].set(xlabel='PC 1')
        axs[1,2].set(xlabel='PC 1')
        fig.subplots_adjust(right=0.8)
        plt.savefig(figs_path+"/snippets_T"+str(tetrode)+".png")
        plt.show()
        print("hello there!")



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    recording_1 = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M1_sorted/M1_D5_2018-09-10_11-43-57/"
    figs_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs"
    spatial_firing_1 = pd.read_pickle(recording_1+"MountainSort/DataFrames/spatial_firing.pkl")

    plot_snippet_pc(spatial_firing_1, figs_path)

    print("look now")

if __name__ == '__main__':
    main()