import numpy as np
from matplotlib.ticker import MaxNLocator
from Edmond.Toy_grid_cell.plot_example_periodic_cells import *
import pandas as pd
import scipy
from Edmond.VR_grid_analysis.vr_grid_population_level_plots import add_max_block_lengths, add_mean_block_lengths, add_median_block_lengths
from Edmond.utility_functions.array_manipulations import *
from Edmond.VR_grid_analysis.vr_grid_cells import get_max_SNR, distance_from_integer, get_lomb_classifier, get_rolling_lomb_classifier_for_centre_trial, \
    compress_rolling_stats, get_rolling_lomb_classifier_for_centre_trial2, get_rolling_lomb_classifier_for_centre_trial_frequentist, get_block_lengths_any_code

def plot_peak_frequencies(sim_data, real_data, save_path):

    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    d_0_simdata = sim_data[(sim_data["true_classification"] == "D") & (sim_data["field_noise_sigma"] == 0)]
    d_5_simdata = sim_data[(sim_data["true_classification"] == "D") & (sim_data["field_noise_sigma"] == 5)]
    d_10_simdata = sim_data[(sim_data["true_classification"] == "D") & (sim_data["field_noise_sigma"] == 10)]
    p_0_simdata = sim_data[(sim_data["true_classification"] == "P") & (sim_data["field_noise_sigma"] == 0)]
    p_5_simdata = sim_data[(sim_data["true_classification"] == "P") & (sim_data["field_noise_sigma"] == 5)]
    p_10_simdata = sim_data[(sim_data["true_classification"] == "P") & (sim_data["field_noise_sigma"] == 10)]

    _, bin_edges = np.histogram(d_0_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

    ax.plot(bin_centres, np.histogram(d_0_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.egocentric_color, linestyle="solid", linewidth=3, label="D,fstd=0")
    ax.plot(bin_centres, np.histogram(d_5_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.egocentric_color, linestyle="dashed", linewidth=3, label="D,fstd=5")
    ax.plot(bin_centres, np.histogram(d_10_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.egocentric_color, linestyle="dotted", linewidth=3, label="D,fstd=10")
    ax.plot(bin_centres, np.histogram(p_0_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.allocentric_color, linestyle="solid", linewidth=3, label="P,fstd=0")
    ax.plot(bin_centres, np.histogram(p_5_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.allocentric_color, linestyle="dashed", linewidth=3, label="P,fstd=5")
    ax.plot(bin_centres, np.histogram(p_10_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.allocentric_color, linestyle="dotted", linewidth=3, label="P,fstd=10")

    #ax.set_title("min="+str(np.round(total_accuracies.min(), decimals=2))+"@Hz="+str(frequency_thresholds[np.where(total_accuracies==total_accuracies.min())[0][0]]), fontsize=20)
    ax.set_ylabel("Number of cells", fontsize=30, labelpad = 10)
    ax.set_xlabel('Spatial Frequency', fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 0.5])
    #ax.set_ylim([0, 1])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    ax.legend(loc='best')
    plt.savefig(save_path + '/peak_frequency_distribution.png', dpi=300)
    plt.close()
    return

def compute_peak_statistics(sim_data):
    peak_frequencies = []
    peak_frequencies_delta_int = []
    peak_powers = []
    for index, row in sim_data.iterrows():
        spatial_periodogram = row["spatial_periodogram"]
        avg_power = np.nanmean(spatial_periodogram, axis=0)
        max_SNR, max_SNR_freq = get_max_SNR(Settings.frequency, avg_power)
        max_SNR_freq_delta_int = distance_from_integer(max_SNR_freq)

        peak_frequencies.append(max_SNR_freq)
        peak_frequencies_delta_int.append(max_SNR_freq_delta_int)
        peak_powers.append(max_SNR)

    sim_data["peak_frequency"] = peak_frequencies
    sim_data["peak_frequency_delta_int"] = peak_frequencies_delta_int
    sim_data["peak_power"] = peak_powers
    return sim_data


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


    real_data = pd.DataFrame()
    real_data = pd.concat([real_data, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort6.pkl")], ignore_index=True)
    real_data = pd.concat([real_data, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort7.pkl")], ignore_index=True)
    real_data = pd.concat([real_data, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")], ignore_index=True)
    real_data = real_data[real_data["snippet_peak_to_trough"] < 500] # uV
    real_data = real_data[real_data["track_length"] == 200]
    real_data = real_data[real_data["n_trials"] >= 10]

    data_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/grid_data/simulated_grid_cells.pkl"
    save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/"
    sim_data = pd.read_pickle(data_path)
    sim_data = compute_peak_statistics(sim_data)

    plot_peak_frequencies(sim_data, real_data, save_path=save_path)

if __name__ == '__main__':
    main()
