import numpy as np
from matplotlib.ticker import MaxNLocator
from Edmond.Toy_grid_cell.plot_example_periodic_cells import *
import pandas as pd
import scipy
from Edmond.VR_grid_analysis.vr_grid_population_level_plots import add_max_block_lengths, add_mean_block_lengths, add_median_block_lengths
from Edmond.utility_functions.array_manipulations import *
from Edmond.VR_grid_analysis.vr_grid_cells import get_max_SNR, distance_from_integer, get_lomb_classifier, get_rolling_lomb_classifier_for_centre_trial, \
    compress_rolling_stats, get_rolling_lomb_classifier_for_centre_trial2, get_rolling_lomb_classifier_for_centre_trial_frequentist, get_block_lengths_any_code
'''
This script looks to answer the question:
what overlap is there between the peak frequency of a population of distance and position encoding grid cells
 
- a population of 1000 grid cells are simulated with grid spacings 40-600cm
- anchoring is either switched on or off (deterimining if the cell is position or distance encoding)
- spatial periodograms are computed for all cells and the frequency at which the peak occurs is collected
- distributions of both peak frequencies are drawn up
- The overlap of these frequency distributions (position and distance populations) will indicate the confidence level of assigning 
a classifation based on any given peak distribution
- This is then repeated under different parametised conditions such as with varying levels of noise 

'''

def compute_peak(firing_rate_map_by_trial):
    fr=firing_rate_map_by_trial.flatten()
    track_length = len(firing_rate_map_by_trial[0])
    n_trials = len(firing_rate_map_by_trial)
    elapsed_distance_bins = np.arange(0, (track_length*n_trials)+1, 1)
    elapsed_distance = 0.5*(elapsed_distance_bins[1:]+elapsed_distance_bins[:-1])/track_length
    # construct the lomb-scargle periodogram
    frequency = Settings.frequency
    sliding_window_size=track_length*Settings.window_length_in_laps
    powers = []
    centre_distances = []
    indices_to_test = np.arange(0, len(fr)-sliding_window_size, 1, dtype=np.int64)[::10]
    for m in indices_to_test:
        ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr[m:m+sliding_window_size])
        power = ls.power(frequency)
        powers.append(power.tolist())
        centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
    powers = np.array(powers)
    avg_power = np.nanmean(powers, axis=0)
    max_peak_power, max_peak_freq = get_max_SNR(frequency, avg_power)
    return max_peak_power, max_peak_freq

def generate_spatial_periodogram(firing_rate_map_by_trial):
    fr=firing_rate_map_by_trial.flatten()
    track_length = len(firing_rate_map_by_trial[0])
    n_trials = len(firing_rate_map_by_trial)
    elapsed_distance_bins = np.arange(0, (track_length*n_trials)+1, 1)
    elapsed_distance = 0.5*(elapsed_distance_bins[1:]+elapsed_distance_bins[:-1])/track_length
    # construct the lomb-scargle periodogram
    frequency = Settings.frequency
    sliding_window_size=track_length*Settings.window_length_in_laps
    powers = []
    centre_distances = []
    indices_to_test = np.arange(0, len(fr)-sliding_window_size, 1, dtype=np.int64)[::10]
    for m in indices_to_test:
        ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr[m:m+sliding_window_size])
        power = ls.power(frequency)
        powers.append(power.tolist())
        centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
    powers = np.array(powers)
    centre_trials = np.round(np.array(centre_distances)).astype(np.int64)
    return powers, centre_trials, track_length

def switch_grid_cells(switch_coding_mode, grid_stability, grid_spacings, n_cells, trial_switch_probability, field_noise_std, return_shuffled=False):
    # generate n switch grid cells
    # generated n positional grid cells
    powers_all_cells = []
    powers_all_cells_shuffled = []
    true_classifications_all_cells = []
    rate_maps_all_cells = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        _, _, _, firing_rate_map_by_trial_smoothed, true_classifications = get_switch_cluster_firing(switch_coding_mode=switch_coding_mode,
                                                                                                     grid_stability=grid_stability,
                                                                                                     field_spacing=grid_spacing,
                                                                                                     trial_switch_probability=trial_switch_probability,
                                                                                                     field_noise_std=field_noise_std)

        powers, centre_trials, track_length = generate_spatial_periodogram(firing_rate_map_by_trial_smoothed)
        powers_all_cells.append(powers)
        rate_maps_all_cells.append(firing_rate_map_by_trial_smoothed)

        if return_shuffled:
            np.random.shuffle(firing_rate_map_by_trial_smoothed)
            powers, centre_trials, track_length = generate_spatial_periodogram(firing_rate_map_by_trial_smoothed)
            powers_all_cells_shuffled.append(powers)

        true_classifications_all_cells.append(true_classifications)

    return powers_all_cells, powers_all_cells_shuffled, centre_trials, track_length, true_classifications_all_cells, rate_maps_all_cells

def generate_stable_grid_cells(field_noise, save_path, grid_spacings, n_cells):
    # this function creates a dataframe of cells of a certain parametisation and creates the spatial periodograms and rate maps
    simulated_data_set = pd.DataFrame()
    for i in range(len(field_noise)):
        # calculate the stats for the most random simulated cell
        grid_stability="imperfect"
        switch_coding = "block"
        # generate_cells
        powers_all_cells, _, centre_trials, track_length, true_classifications_all_cells, rate_maps = \
            switch_grid_cells(switch_coding, grid_stability, grid_spacings, n_cells, trial_switch_probability=0, field_noise_std=field_noise[i], return_shuffled=False)

        for n in range(n_cells):
            avg_power = np.nanmean(powers_all_cells[n], axis=0)
            max_SNR, max_SNR_freq = get_max_SNR(Settings.frequency, avg_power)
            lomb_classifier = get_lomb_classifier(max_SNR, max_SNR_freq, 0, 0.05, numeric=False)

            cell_row = pd.DataFrame()
            cell_row["rate_maps_smoothened"] = [rate_maps[n]]
            cell_row["spatial_periodogram"] = [powers_all_cells[n]]
            cell_row["centre_trials"] = [centre_trials[n]]
            cell_row["track_length"] = [track_length]
            cell_row["field_noise_sigma"] = [field_noise[i]]
            cell_row["grid_spacing"] = [grid_spacings[n]]
            cell_row["lomb_classification"] = [lomb_classifier]
            cell_row["true_classification"] = [true_classifications_all_cells[n][0]]
            simulated_data_set = pd.concat([simulated_data_set, cell_row], ignore_index=True)
            print("added_cell_to_dataframe")

    simulated_data_set.to_pickle(save_path+"simulated_grid_cells.pkl")
    return

def generate_switch_grid_cells(field_noise, trial_switch_probability, save_path, grid_spacings, n_cells):
    # this function creates a dataframe of cells of a certain parametisation and creates the spatial periodograms and rate maps

    simulated_data_set = pd.DataFrame()
    for i in range(len(field_noise)):
        # calculate the stats for the most random simulated cell
        grid_stability="imperfect"
        switch_coding = "by_trial"

        # generate_cells
        powers_all_cells, _, centre_trials, track_length, _, rate_maps = \
            switch_grid_cells(switch_coding, grid_stability, grid_spacings, n_cells, trial_switch_probability[0], field_noise_std=field_noise[i], return_shuffled=False)

        for n in range(n_cells):
            rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
                get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_all_cells[n], power_threshold=Settings.lomb_rolling_threshold,
                                                             power_step=Settings.power_estimate_step, track_length=track_length,
                                                             n_window_size=Settings.rolling_window_size_for_lomb_classifier, lomb_frequency_threshold=Settings.lomb_frequency_threshold)

            cell_row = pd.DataFrame()
            cell_row["rate_maps_smoothened"] = [rate_maps[n]]
            cell_row["spatial_periodogram"] = [powers_all_cells[n]]
            cell_row["centre_trials"] = [centre_trials[n]]
            cell_row["track_length"] = [track_length]
            cell_row["rolling_lomb_classifier"] = [rolling_lomb_classifier]
            cell_row["rolling_window_size_for_lomb_classifier"] = [Settings.rolling_window_size_for_lomb_classifier]
            cell_row["lomb_frequency_threshold"] = [Settings.lomb_frequency_threshold]
            cell_row["lomb_power_threshold"] = [Settings.lomb_rolling_threshold]
            cell_row["field_noise_sigma"] = [field_noise[i]]
            cell_row["switch_coding"] = [switch_coding]
            cell_row["grid_spacing"] = [grid_spacings[i]]
            simulated_data_set = pd.concat([simulated_data_set, cell_row], ignore_index=True)
            print("added_cell_to_dataframe")

        switch_coding = "block"
        # generate_cells
        for j in range(len(trial_switch_probability)):
            powers_all_cells, _, centre_trials, track_length, _, rate_maps = \
                switch_grid_cells(switch_coding, grid_stability, grid_spacings, n_cells, trial_switch_probability[j], field_noise_std=field_noise[i], return_shuffled=False)

            # classify trials
            for n in range(n_cells):
                rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
                    get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_all_cells[n], power_threshold=Settings.lomb_rolling_threshold,
                                                                 power_step=Settings.power_estimate_step, track_length=track_length,
                                                                 n_window_size=Settings.rolling_window_size_for_lomb_classifier, lomb_frequency_threshold=Settings.lomb_frequency_threshold)
                cell_row = pd.DataFrame()
                cell_row["rate_maps_smoothened"] = [rate_maps[n]]
                cell_row["spatial_periodogram"] = [powers_all_cells[n]]
                cell_row["centre_trials"] = [centre_trials[n]]
                cell_row["track_length"] = [track_length]
                cell_row["rolling_lomb_classifier"] = [rolling_lomb_classifier]
                cell_row["rolling_window_size_for_lomb_classifier"] = [Settings.rolling_window_size_for_lomb_classifier]
                cell_row["lomb_frequency_threshold"] = [Settings.lomb_frequency_threshold]
                cell_row["lomb_power_threshold"] = [Settings.lomb_rolling_threshold]
                cell_row["field_noise_sigma"] = [field_noise[i]]
                cell_row["switch_coding"] = [switch_coding]
                cell_row["grid_spacing"] = [grid_spacings[i]]
                simulated_data_set = pd.concat([simulated_data_set, cell_row], ignore_index=True)
                print("added_cell_to_dataframe")

    simulated_data_set.to_pickle(save_path+"simulated_grid_cells.pkl")
    return


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


    np.random.seed(0)
    save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/grid_data/"
    n_cells = 1000
    grid_spacing_low = 40
    grid_spacing_high = 400
    grid_spacings = np.random.uniform(low=grid_spacing_low, high=grid_spacing_high, size=n_cells);
    #generate_switch_grid_cells(field_noise=[0,5,10], trial_switch_probability=[0.5, 0.1, 0.05, 0.01],
    #                           save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells)

    generate_stable_grid_cells(field_noise=[0,5,10],save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells)

if __name__ == '__main__':
    main()
