import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from scipy import stats
from scipy import signal
from scipy.stats import gaussian_kde
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import os
import traceback
import warnings
import matplotlib.ticker as ticker
import sys
import math
import EdmondHC.plot_utility2
import settings
import matplotlib.pylab as plt
import matplotlib as mpl
import control_sorting_analysis
import PostSorting.post_process_sorted_data_vr
warnings.filterwarnings('ignore')
from scipy.stats.stats import pearsonr

def min_max_normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def find_set(a, b):
    return set(a) & set(b)

def plot_egocentric_spike_spatial_autocorrelogram(spike_data, output_path, return_spike_data=False):
    print('plotting egocentric spike spatial autocorrelogram...')
    save_path = output_path + '/Figures/egocentric_spatial_autocorrelograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    spatial_auto_peak = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data["firing_times"].iloc[0].astype(np.int32)

        if len(firing_times_cluster)>1:
            x_position_cluster = np.round(np.array(cluster_spike_data["position_x"].iloc[0])).astype(np.int8)
            y_position_cluster = np.round(np.array(cluster_spike_data["position_y"].iloc[0])).astype(np.int8)
            hd_cluster = np.radians(np.array(cluster_spike_data["hd"].iloc[0])).astype(np.float16)

            max_spikes = 50000
            if len(firing_times_cluster)>max_spikes:
                x_position_cluster = x_position_cluster[:max_spikes]
                y_position_cluster = y_position_cluster[:max_spikes]
                hd_cluster = hd_cluster[:max_spikes]

            x_spike_stack = np.vstack([x_position_cluster]*len(x_position_cluster))
            y_spike_stack = np.vstack([y_position_cluster]*len(y_position_cluster))

            x_spike_stack = x_spike_stack - x_position_cluster[:,None]
            y_spike_stack = y_spike_stack - y_position_cluster[:,None]

            tmp1 = (np.cos(hd_cluster)[:,None]*x_spike_stack).astype(np.int8)
            tmp2 = (np.sin(hd_cluster)[:,None]*y_spike_stack).astype(np.int8)
            rotated_pos_from_spikes_x = tmp1 - tmp2

            tmp1 = (np.sin(hd_cluster)[:,None]*x_spike_stack).astype(np.int8)
            tmp2 = (np.cos(hd_cluster)[:,None]*y_spike_stack).astype(np.int8)
            rotated_pos_from_spikes_y = tmp1 + tmp2

            rotated_pos_from_spikes_x = rotated_pos_from_spikes_x.astype(np.int16).flatten()
            rotated_pos_from_spikes_y = rotated_pos_from_spikes_y.astype(np.int16).flatten()


            autocorrelogram, xedges, yedges = np.histogram2d(rotated_pos_from_spikes_x, rotated_pos_from_spikes_y,
                                                             bins=[200,200], range=[[-100, 100], [-100, 100]], density=False)

            euclidean_distances = np.sqrt((np.square(rotated_pos_from_spikes_x)+np.square(rotated_pos_from_spikes_y)))

            autocorrelogram_1d, bin_edges = np.histogram(euclidean_distances, bins=150, range=[0, 150], density=False)
            peaks = signal.argrelextrema(autocorrelogram_1d, np.greater, order=20)[0]
            if len(peaks)>0:
                spatial_auto_peak.append(peaks.tolist())
            else:
                spatial_auto_peak.append([])

            autocorrelogram[100,100] = autocorrelogram[100,99] # negate the middle point
            autocorrelogram = min_max_normalize(autocorrelogram)

            # make the figure
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            cmap = plt.cm.get_cmap("inferno")
            c = ax.imshow(autocorrelogram, interpolation='none', cmap=cmap, vmin=0, vmax=1, origin='lower')
            clb = fig.colorbar(c, ax=ax, shrink=0.5)
            clb.mappable.set_clim(0, 1)
            plt.ylabel('Y lag (cm)', fontsize=20, labelpad = 10)
            plt.xlabel('X lag (cm)', fontsize=20, labelpad = 10)
            plt.yticks(ticks=[50, 100, 150], labels=["-50", "0", "50"], fontsize=12)
            plt.xticks(ticks=[50, 100, 150], labels=["-50", "0", "50"], fontsize=12)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_egocentric_spatial_autocorrelogram_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

            # make an enhanced contrast figure
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            cmap = plt.cm.get_cmap("inferno")
            c = ax.imshow(autocorrelogram, interpolation='none', cmap=cmap, vmin=0, vmax=0.1, origin='lower')
            clb = fig.colorbar(c, ax=ax, shrink=0.5)
            clb.mappable.set_clim(0, 0.1)
            plt.ylabel('Y lag (cm)', fontsize=20, labelpad = 10)
            plt.xlabel('X lag (cm)', fontsize=20, labelpad = 10)
            plt.yticks(ticks=[50, 100, 150], labels=["-50", "0", "50"], fontsize=12)
            plt.xticks(ticks=[50, 100, 150], labels=["-50", "0", "50"], fontsize=12)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_egocentric_spatial_autocorrelogram_Cluster_' + str(cluster_id) + '_enhanced_constrast.png', dpi=200)
            plt.close()

            # make a 1D figure
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            ax.bar(bin_centres, autocorrelogram_1d, color="black", edgecolor="black", align="edge")
            ax.scatter(bin_centres[peaks], autocorrelogram_1d[peaks], marker="x", color="r")
            plt.ylabel('Counts', fontsize=20, labelpad = 10)
            plt.xlabel('Distance (cm)', fontsize=20, labelpad = 10)
            plt.xlim(0,150)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            EdmondHC.plot_utility2.style_vr_plot(ax, x_max=max(autocorrelogram_1d[1:]))
            plt.locator_params(axis = 'x', nbins  = 8)
            tick_spacing = 50
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_egocentric_spatial_autocorrelogram_Cluster_' + str(cluster_id) + '_1D.png', dpi=200)
            plt.close()

        else:
            spatial_auto_peak.append([])


    spike_data["spatial_autocorr_peak_cm"] = spatial_auto_peak
    if return_spike_data:
        return spike_data
    else:
        return


def process_recordings(of_recording_path_list):

    for recording in of_recording_path_list:
        print("processing ", recording)
        try:
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position.pkl")
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            spike_data = plot_egocentric_spike_spatial_autocorrelogram(spike_data, output_path, return_spike_data=True)
            spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            print("successfully processed and saved vr_grid analysis on "+recording)
        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)

def plot_peak_histogram(of_path_list, save_path):
    concat = pd.DataFrame()
    for recording in of_path_list:
        print("processing ", recording)
        try:
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            spike_data = spike_data[["session_id", "classifier", "spatial_autocorr_peak_cm", "grid_score", "cluster_id"]]
            concat = pd.concat([concat, spike_data], ignore_index=True)


        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)

    grid_cells = concat[concat["classifier"] == "G"]
    non_grid_cells = concat[concat["classifier"] != "G"]

    grid_cells_peaks = EdmondHC.plot_utility2.pandas_collumn_to_numpy_array(grid_cells["spatial_autocorr_peak_cm"])
    non_grid_cells_peaks = EdmondHC.plot_utility2.pandas_collumn_to_numpy_array(non_grid_cells["spatial_autocorr_peak_cm"])

    grid_cell_hist, bin_edges = np.histogram(grid_cells_peaks, bins=15, range=[0, 150], density=True)
    non_grid_cell_hist, bin_edges = np.histogram(non_grid_cells_peaks, bins=15, range=[0, 150], density=True)

    # make a 1D figure
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    ax.bar(bin_edges[:-1], non_grid_cell_hist/np.sum(non_grid_cell_hist), width=np.diff(bin_edges), edgecolor="black", align="edge", alpha=0.5, color="black")
    ax.bar(bin_edges[:-1], grid_cell_hist/np.sum(grid_cell_hist), width=np.diff(bin_edges), edgecolor="black", align="edge", alpha=0.5, color="red")
    plt.ylabel('Density', fontsize=17, labelpad = 10)
    plt.xlabel('Autocorrelogram peak (cm)', fontsize=17, labelpad = 10)
    plt.xlim(0,150)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.locator_params(axis = 'x', nbins  = 8)
    tick_spacing = 50
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    #plot kernel density estimation
    density = gaussian_kde(grid_cells_peaks)
    x_vals = np.linspace(0,150,1500) # Specifying the limits of our data
    density.covariance_factor = lambda : 0.1 #Smoothing parameter
    density._compute_covariance()
    ax.plot(x_vals,density(x_vals)*10, color="red")
    density = gaussian_kde(non_grid_cells_peaks)
    x_vals = np.linspace(0,150,1500) # Specifying the limits of our data
    density.covariance_factor = lambda : 0.1 #Smoothing parameter
    density._compute_covariance()
    ax.plot(x_vals,density(x_vals)*10, color="black")

    plt.savefig(save_path + 'grid_cell_autocorrelogram_peaks_comparions_1D.png', dpi=200)
    plt.close()
    print("hello")

    # make a scatter of figure
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    for i in range(len(grid_cells)):
        peaks = grid_cells["spatial_autocorr_peak_cm"].iloc[i]
        #ax.plot(peaks, np.repeat(grid_cells["grid_score"].iloc[i], len(peaks)), alpha=0.2, color="red")
        ax.scatter(peaks, np.repeat(grid_cells["grid_score"].iloc[i], len(peaks)), alpha=0.3, color="red", marker="x")
    for i in range(len(non_grid_cells)):
        peaks = non_grid_cells["spatial_autocorr_peak_cm"].iloc[i]
        #ax.plot(peaks, np.repeat(non_grid_cells["grid_score"].iloc[i], len(peaks)), alpha=0.2, color="black")
        ax.scatter(peaks, np.repeat(non_grid_cells["grid_score"].iloc[i], len(peaks)), alpha=0.3, color="black", marker="x")
    plt.ylabel('Grid score', fontsize=17, labelpad = 10)
    plt.xlabel('Autocorrelogram peak (cm)', fontsize=17, labelpad = 10)
    plt.xlim(0,150)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.locator_params(axis = 'x', nbins  = 8)
    tick_spacing = 50
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(save_path + 'grid_cell_autocorrelogram_peaks_scatter.png', dpi=200)
    plt.close()







def main():
    print('-------------------------------------------------------------')


    # give a path for a directory of recordings or path of a single recording
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()]
    #process_recordings(of_path_list)

    plot_peak_histogram(of_path_list, save_path="/mnt/datastore/Harry/Vr_grid_cells/")


    print("look now")

if __name__ == '__main__':
    main()