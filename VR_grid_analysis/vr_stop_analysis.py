import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from numpy import inf
import matplotlib.colors as colors
from scipy.ndimage import uniform_filter1d
import PostSorting.parameters
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from Edmond.VR_grid_analysis.remake_position_data import syncronise_position_data
from PostSorting.vr_spatial_firing import bin_fr_in_space, bin_fr_in_time, add_position_x, add_trial_number, add_trial_type
from scipy import stats
import Edmond.VR_grid_analysis.analysis_settings as Settings
from scipy import signal
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
import os
import traceback
from astropy.nddata import block_reduce
import warnings
import matplotlib.ticker as ticker
import sys
import Edmond.plot_utility2
import Edmond.VR_grid_analysis.hit_miss_try_firing_analysis
import settings
from scipy import stats
import matplotlib.pylab as plt
import matplotlib as mpl
import control_sorting_analysis
import PostSorting.post_process_sorted_data_vr
from astropy.timeseries import LombScargle
from Edmond.utility_functions.array_manipulations import *
from joblib import Parallel, delayed
import multiprocessing
import open_ephys_IO
warnings.filterwarnings('ignore')
from scipy.stats.stats import pearsonr
from scipy.stats import shapiro
plt.rc('axes', linewidth=3)
from scipy.signal import find_peaks
from Edmond.VR_grid_analysis.vr_grid_cells import *
import umap

def plot_umap(spike_data, output_path):
    print('plotting trial firing rate maps similarity matrix...')
    save_path = output_path + '/Figures/firing_rate_similarity_matrices'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    firing_rates = []
    positions_all_clusters = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_times_cluster = cluster_df.firing_times.iloc[0]
        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(cluster_df['fr_binned_in_space_smoothed'].iloc[0])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            percentile_90th = np.nanpercentile(cluster_firing_maps, 90); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_90th)
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0

            positions = np.tile(np.arange(0.5, 200, 1), len(cluster_firing_maps))

            firing_rates.append(cluster_firing_maps.flatten())
            positions_all_clusters.append(positions.flatten())

    firing_rates = np.array(firing_rates).T
    positions_all_clusters = np.array(positions_all_clusters).T

    mapper = umap.UMAP(random_state=42, metric="cosine").fit_transform(firing_rates)
    fig, ax = plt.subplots()

    # ax.set_title("ID: "+str(cluster_id), fontsize= 25)
    # ax.set_ylabel("Trial", fontsize=20)
    # ax.set_xlabel("Trial", fontsize=20)
    ax.scatter(mapper[:,0], mapper[:,1], marker="o", c=positions, cmap='twilight',s=0.1)
    #ax.tick_params(axis='both', which='major', labelsize=30)
    #tick_spacing = 100
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax.tick_params(width=1)
    #ax.invert_yaxis()
    #fig.tight_layout()
    plt.savefig(save_path + '/umap.png',dpi=300)
    plt.close()
    return

def process_recordings(vr_recording_path_list):
    vr_recording_path_list.sort()
    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            tags = control_sorting_analysis.get_tags_parameter_file(recording)
            sorter_name = control_sorting_analysis.check_for_tag_name(tags, "sorter_name")
            output_path = recording+'/'+sorter_name
            position_data = pd.read_pickle(recording+"/"+sorter_name+"/DataFrames/position_data.pkl")
            spike_data = pd.read_pickle(recording+"/"+sorter_name+"/DataFrames/spatial_firing.pkl")

            if len(spike_data) != 0:
                plot_umap(spike_data, output_path)

                print("successfully processed and saved vr_grid analysis on "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)


def main():
    print('-------------------------------------------------------------')
    vr_path_list = []
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    vr_path_list = ['/mnt/datastore/Harry/cohort8_may2021/vr/M11_D36_2021-06-28_12-04-36']
    vr_path_list = ["/mnt/datastore/Harry/Cohort8_may2021/vr/M11_D19_2021-06-03_10-50-41"]
    process_recordings(vr_path_list)

    print("look now")

if __name__ == '__main__':
    main()
