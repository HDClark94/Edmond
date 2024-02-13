import matplotlib.pyplot as plt
import imageio
import scipy.stats as stats
import Edmond.eLife_Grid_anchoring_2024.analysis_settings as Settings
from Edmond.Concatenate_from_server import *
from Edmond.eLife_Grid_anchoring_2024.vr_grid_cells import add_lomb_classifier
plt.rc('axes', linewidth=3)

def summarise_grid_cells(combined_df, save_path, save=True):
    stats = pd.DataFrame()

    n_cells_total = 0; n_sessions_total = 0; n_grids_total = 0
    for mouse in np.unique(combined_df["mouse"]):
        mouse_stats = pd.DataFrame()
        mouse_df = combined_df[combined_df["mouse"] == mouse]
        n_cells = len(mouse_df)
        n_sessions = len(np.unique(mouse_df["session_id"]))
        n_grids = len(mouse_df[mouse_df["classifier"] == "G"])
        n_cells_total += n_cells
        n_sessions_total += n_sessions
        n_grids_total += n_grids

        mouse_stats["mouse"] = [mouse]
        mouse_stats["n_cells"] = [n_cells]
        mouse_stats["n_sessions"] = [n_sessions]
        mouse_stats["n_grids"] = [n_grids]
        mouse_stats["percentage_grids"] = [(n_grids/n_cells)*100]
        stats = pd.concat([stats, mouse_stats], ignore_index=True)

    total_stats = pd.DataFrame()
    total_stats["mouse"] = ["total"]
    total_stats["n_cells"] = [n_cells_total]
    total_stats["n_sessions"] = [n_sessions_total]
    total_stats["n_grids"] = [n_grids_total]
    total_stats["percentage_grids"] = [(n_grids_total/n_cells_total)*100]
    stats = pd.concat([stats, total_stats], ignore_index=True)
    if save:
        stats.to_csv(save_path+"grid_stats.csv")

    combined_df = combined_df[["session_id_vr", "session_id_of", "full_session_id_of", "full_session_id_vr", "cluster_id", "mouse", "n_trials", "grid_spacing", "field_size", "classifier", "Lomb_classifier_", "ML_Freqs", "grid_score", "grid_spacing",
                               "hd_score", "border_score", "ThetaIndex", "mean_firing_rate_of", "rate_map_correlation_first_vs_second_half", "spatial_information_score", "spatial_information_score_Isec_vr",
                               "snippet_peak_to_trough", "rolling:proportion_encoding_position", "rolling:proportion_encoding_distance",  "rolling:proportion_encoding_null"]]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    if save:
        combined_df.to_csv(save_path+"cells.csv")
        grid_cells.to_csv(save_path+"grid_cells.csv")
    return stats, grid_cells

def plot_of_rate_maps_PDN(grid_cells, save_path):
    #grid_cells = grid_cells[grid_cells["grid_score"] > 0]
    #grid_cells = grid_cells[grid_cells["rate_map_correlation_first_vs_second_half"] > 0]
    for coding_scheme in ["Position", "Null", "Distance"]:
        PDN_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == coding_scheme]
        PDN_grid_cells = PDN_grid_cells.sort_values(by='grid_score', ascending=False)
        #PDN_grid_cells = PDN_grid_cells.sort_values(by='mean_firing_rate_of', ascending=False)

        imgs=[]
        for i in range(len(PDN_grid_cells)):
            cluster_id = str(int(PDN_grid_cells.cluster_id.iloc[i]))
            session_id = PDN_grid_cells.session_id_of.iloc[i]
            full_of_path = PDN_grid_cells.full_session_id_of.iloc[i]
            im = imageio.imread(full_of_path+"/MountainSort/Figures/rate_maps/"+session_id+"_rate_map_"+cluster_id+".png")
            im = im[253:253+975, 169:169+975, :]

            # consider clipping the image so we only see the rate map and not the colorbar and text
            imgs.append(im)

        n_columns = 8
        n_rows = int(len(PDN_grid_cells)//8)+1
        fig, axs = plt.subplots(8, n_columns, figsize=(8, n_columns))
        axs = axs.flatten()

        for ax in axs:
            ax.grid(False)
            ax.axis("off")
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
        plt.subplots_adjust(wspace=0.1, hspace=0.35)
        plt.savefig(save_path+"grid_cells_rate_maps_"+coding_scheme+".png", dpi=1000)
        plt.close()


def get_p_text(p, ns=False):

    if p is not None:

        if p<0.0001:
            return "****"
        elif p<0.001:
            return "***"
        elif p<0.01:
            return "**"
        elif p<0.05:
            return "*"
        elif ns:
            return "ns"
        else:
            return " "
    else:
        return " "

def plot_of_metrics_PDN(grid_cells, save_path):
    grid_cells = grid_cells[grid_cells["grid_score"] > 0]
    grid_cells = grid_cells[grid_cells["rate_map_correlation_first_vs_second_half"] > 0]

    P_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Position"]
    D_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]
    N_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == "Null"]

    for column, column_tidy_name in zip(["grid_score", "grid_spacing", "mean_firing_rate_of", "ThetaIndex",
                                         "rate_map_correlation_first_vs_second_half", "spatial_information_score", "hd_score"],
                                        ["Grid score", "Grid spacing", "FR (OF)", "Theta index",
                                         "Session stability", "Spatial information", "HD score"]):

        P = np.asarray(P_grid_cells[column]); P=P[~np.isnan(P)]
        D = np.asarray(D_grid_cells[column]); D=D[~np.isnan(D)]
        N = np.asarray(N_grid_cells[column]); N=N[~np.isnan(N)]

        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlabel(column_tidy_name, fontsize=25, labelpad=10)
        ax.set_ylabel("Cumulative Density", fontsize=25, labelpad=10)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.locator_params(axis='y', nbins=6)
        plt.locator_params(axis='x', nbins=4)
        _, _, patchesP = ax.hist(P, bins=500, color=Settings.allocentric_color, histtype="step", density=True, cumulative=True, linewidth=2); patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
        _, _, patchesD = ax.hist(D, bins=500, color=Settings.egocentric_color, histtype="step", density=True, cumulative=True, linewidth=2); patchesD[0].set_xy(patchesD[0].get_xy()[:-1])
        _, _, patchesN = ax.hist(N, bins=500, color=Settings.null_color, histtype="step", density=True, cumulative=True, linewidth=2); patchesN[0].set_xy(patchesN[0].get_xy()[:-1])

        p = stats.ks_2samp(P, D)[1]
        print("p =",p, "for "+column +", "+get_p_text(p))

        ax.set_ylim([0,1])
        ax.set_yticks([0, 0.5, 1])
        #ax.set_xlim([])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path+"of_metrics_"+column+".png", dpi=300)
        plt.close()

def main():
    print('-------------------------------------------------------------')
    combined_df = pd.DataFrame()
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort6.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort7.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")], ignore_index=True)
    #combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort9.pkl")], ignore_index=True)

    combined_df = combined_df[combined_df["snippet_peak_to_trough"] < 500] # uV
    combined_df = combined_df[combined_df["mean_firing_rate_of"] > 0.2] # Hz
    combined_df = combined_df[combined_df["track_length"] == 200]
    combined_df = combined_df[combined_df["n_trials"] >= 10]
    combined_df = add_lomb_classifier(combined_df,suffix="")
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]

    # remove mice without any grid cells
    combined_df = combined_df[combined_df["mouse"] != "M2"]
    combined_df = combined_df[combined_df["mouse"] != "M4"]
    combined_df = combined_df[combined_df["mouse"] != "M15"]

    grid_cell_stats, grid_cells = summarise_grid_cells(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/", save=True)

    print("number of sessions = ", len(np.unique(combined_df["session_id_vr"])))
    print("number of cells  = ", len(combined_df))
    print("avg trial number  = ", np.nanmean(combined_df["n_trials"]))
    print("std trial number  = ", np.nanstd(combined_df["n_trials"]))

    print("number of sessions with grid cells = ", len(np.unique(grid_cells["session_id_vr"])))
    print("number of grid cells cells  = ", len(grid_cells))
    print("avg trial number for sessions with grid cells = ", np.nanmean(grid_cells["n_trials"]))
    print("std trial number for sessions with grid cells = ", np.nanstd(grid_cells["n_trials"]))

    for mouse in np.unique(combined_df["mouse"]):
        print("for mouse ", str(mouse))
        mouse_df = combined_df[combined_df["mouse"] == mouse]
        print("number of sessions = ", len(np.unique(mouse_df["session_id_vr"])))
        print("number of cells  = ", len(mouse_df))
        print("avg trial number  = ", np.nanmean(mouse_df["n_trials"]))
        print("std trial number  = ", np.nanstd(mouse_df["n_trials"]))

    #plot_of_rate_maps_PDN(grid_cells, save_path="/mnt/datastore/Harry/Vr_grid_cells/open_field_comparison/")
    #plot_of_metrics_PDN(grid_cells, save_path="/mnt/datastore/Harry/Vr_grid_cells/open_field_comparison/")

    print("look now")

if __name__ == '__main__':
    main()
