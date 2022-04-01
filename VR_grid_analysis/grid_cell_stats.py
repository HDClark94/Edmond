import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from Edmond.Concatenate_from_server import *
plt.rc('axes', linewidth=3)

def summarise_grid_cells(combined_df, save_path, save=True):
    stats = pd.DataFrame()

    n_cells_total = 0
    n_sessions_total = 0
    n_grids_total = 0
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

    grid_cells = combined_df[combined_df["classifier"] == "G"]
    grid_cells = grid_cells[["session_id", "full_session_id", "cluster_id", "mouse", "classifier", "Lomb_classifier_", "grid_score", "hd_score"]]
    if save:
        grid_cells.to_csv(save_path+"grid_cells.csv")
    return stats, grid_cells

def plot_of_rate_maps_PDN(grid_cells, save_path):
    for coding_scheme in ["Position", "Distance", "Null"]:
        PDN_grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == coding_scheme]
        PDN_grid_cells = PDN_grid_cells.sort_values(by='grid_score', ascending=False)

        imgs=[]
        for i in range(len(PDN_grid_cells)):
            cluster_id = str(PDN_grid_cells.cluster_id.iloc[0])
            session_id = PDN_grid_cells.session_id.iloc[0]
            full_of_path = PDN_grid_cells.full_session_id_of.iloc[0]
            imgs.append(imageio.imread(full_of_path+"/MountainSort/Figures/rate_maps/"+session_id+"_rate_map_"+cluster_id+".png"))

        _, axs = plt.subplots(8, 8, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
        plt.savefig(save_path+"grid_cells_rate_maps_"+coding_scheme+".png", dpi=1000)
        plt.close()








def main():
    print('-------------------------------------------------------------')
    combined_df = pd.DataFrame()
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort6.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort7.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort9.pkl")], ignore_index=True)

    combined_df = combined_df[combined_df["snippet_peak_to_trough"] < 500] # uV remove lick artefacts
    combined_df = combined_df[combined_df["track_length"] == 200] # only look at default task

    grid_cell_stats, grid_cells = summarise_grid_cells(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")

    plot_of_rate_maps_PDN(grid_cells, save_path="/mnt/datastore/Harry/Vr_grid_cells/")

    print("look now")

if __name__ == '__main__':
    main()
