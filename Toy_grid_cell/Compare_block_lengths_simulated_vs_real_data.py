import numpy as np
from matplotlib.ticker import MaxNLocator
from Edmond.Toy_grid_cell.plot_example_periodic_cells import *
from matplotlib.markers import TICKDOWN

import pandas as pd
import scipy
import settings
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
    spike_locations_all_cells = []
    spike_trial_numbers_all_cells = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        spikes_locations, spike_trial_numbers, _, firing_rate_map_by_trial_smoothed, true_classifications = get_switch_cluster_firing(switch_coding_mode=switch_coding_mode,
                                                                                                     grid_stability=grid_stability,
                                                                                                     field_spacing=grid_spacing,
                                                                                                     trial_switch_probability=trial_switch_probability,
                                                                                                     field_noise_std=field_noise_std)

        powers, centre_trials, track_length = generate_spatial_periodogram(firing_rate_map_by_trial_smoothed)
        powers_all_cells.append(powers)
        spike_locations_all_cells.append(spikes_locations)
        spike_trial_numbers_all_cells.append(spike_trial_numbers)

        if return_shuffled:
            np.random.shuffle(firing_rate_map_by_trial_smoothed)
            powers, centre_trials, track_length = generate_spatial_periodogram(firing_rate_map_by_trial_smoothed)
            powers_all_cells_shuffled.append(powers)

        true_classifications_all_cells.append(true_classifications)

    return powers_all_cells, powers_all_cells_shuffled, centre_trials, track_length, true_classifications_all_cells, spike_locations_all_cells, spike_trial_numbers_all_cells


def perfect_grid_cells(grid_spacings, n_cells, field_noise_std):

    # generated n positional grid cells
    position_peak_frequencies = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed = get_cluster_firing(cell_type_str="stable_allocentric_grid_cell", field_spacing=grid_spacing, field_noise_std=field_noise_std)
        max_peak_power, max_peak_freq = compute_peak(firing_rate_map_by_trial_smoothed)
        position_peak_frequencies.append(max_peak_freq)

        # generated n distance grid cells
    distance_peak_frequencies = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed = get_cluster_firing(cell_type_str="stable_egocentric_grid_cell", field_spacing=grid_spacing, field_noise_std=field_noise_std)
        max_peak_power, max_peak_freq = compute_peak(firing_rate_map_by_trial_smoothed)
        distance_peak_frequencies.append(max_peak_freq)

    return np.array(position_peak_frequencies), np.array(distance_peak_frequencies)


def imperfect_grid_cells(grid_spacings, n_cells, field_noise_std):

    # generated n positional grid cells
    position_peak_frequencies = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed = get_cluster_firing(cell_type_str="unstable_allocentric_grid_cell", field_spacing=grid_spacing, field_noise_std=field_noise_std)
        max_peak_power, max_peak_freq = compute_peak(firing_rate_map_by_trial_smoothed)
        position_peak_frequencies.append(max_peak_freq)

        # generated n distance grid cells
    distance_peak_frequencies = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed = get_cluster_firing(cell_type_str="unstable_egocentric_grid_cell", field_spacing=grid_spacing, field_noise_std=field_noise_std)
        max_peak_power, max_peak_freq = compute_peak(firing_rate_map_by_trial_smoothed)
        distance_peak_frequencies.append(max_peak_freq)

    return np.array(position_peak_frequencies), np.array(distance_peak_frequencies)



def ignore_end_trials_in_block(cell_true_classifications):
    new_cell_true_classifications = cell_true_classifications.copy()
    for i in range(len(cell_true_classifications)):
        if (i == 0) or (i == len(cell_true_classifications)-1):
            ignore = True
        elif (cell_true_classifications[i] != cell_true_classifications[i-1]) or \
                (cell_true_classifications[i] != cell_true_classifications[i+1]):
            ignore = True
        else:
            ignore = False

        if ignore:
            new_cell_true_classifications[i] = ""
    return new_cell_true_classifications



def compare_block_length_assay(real_data, field_noise, trial_switch_probability, save_path,
                               grid_spacings, n_cells):

    real_data = real_data[real_data["Lomb_classifier_"] != "Unclassified"]
    real_data = real_data[real_data["Lomb_classifier_"] != "Null"]
    grid_cells = real_data[real_data["classifier"] == "G"]
    grid_cells = add_max_block_lengths(grid_cells)
    grid_cells = add_mean_block_lengths(grid_cells)
    grid_cells = add_median_block_lengths(grid_cells)

    for i in range(len(field_noise)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1, 1]})

        # calculate the stats for the most random simulated cell
        grid_stability="imperfect"
        switch_coding = "by_trial"
        # generate_cells
        powers_all_cells, powers_all_cells_shuffled, centre_trials, track_length, _, _, _ = \
            switch_grid_cells(switch_coding, grid_stability, grid_spacings, n_cells, trial_switch_probability[0], field_noise_std=field_noise[i], return_shuffled=True)

        max_block_lengths_diff_all_cells = []
        mean_block_lengths_diff_all_cells = []
        median_block_lengths_diff_all_cells = []
        # classify trials
        for n in range(n_cells):
            rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
                get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_all_cells[n], power_threshold=Settings.lomb_rolling_threshold,
                                                             power_step=Settings.power_estimate_step, track_length=track_length,
                                                             n_window_size=Settings.rolling_window_size_for_lomb_classifier, lomb_frequency_threshold=Settings.lomb_frequency_threshold)

            rolling_lomb_classifier_shuffled, _, _, rolling_frequencies, rolling_points = \
                get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_all_cells_shuffled[n], power_threshold=Settings.lomb_rolling_threshold,
                                                             power_step=Settings.power_estimate_step, track_length=track_length,
                                                             n_window_size=Settings.rolling_window_size_for_lomb_classifier, lomb_frequency_threshold=Settings.lomb_frequency_threshold)

            max_block_lengths_diff_all_cells.append(np.nanmax(get_block_lengths_any_code(rolling_lomb_classifier)-np.nanmax(get_block_lengths_any_code(rolling_lomb_classifier_shuffled))))
            mean_block_lengths_diff_all_cells.append(np.nanmean(get_block_lengths_any_code(rolling_lomb_classifier)-np.nanmean(get_block_lengths_any_code(rolling_lomb_classifier_shuffled))))
            median_block_lengths_diff_all_cells.append(np.nanmedian(get_block_lengths_any_code(rolling_lomb_classifier)-np.nanmedian(get_block_lengths_any_code(rolling_lomb_classifier_shuffled))))

        max_counts, bin_edges = np.histogram(max_block_lengths_diff_all_cells, bins= 10000, range=(-1,1)); bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        ax1.plot(bin_centres, np.cumsum(max_counts/np.sum(max_counts)), color="black" , linewidth=3)
        mean_counts, bin_edges = np.histogram(mean_block_lengths_diff_all_cells, bins= 10000, range=(-1,1)); bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        ax2.plot(bin_centres, np.cumsum(mean_counts/np.sum(mean_counts)), color="black" , linewidth=3)
        median_counts, bin_edges = np.histogram(median_block_lengths_diff_all_cells, bins= 10000, range=(-1,1)); bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        ax3.plot(bin_centres, np.cumsum(median_counts/np.sum(median_counts)), color="black" , linewidth=3, label="BT,s:"+str(field_noise[i]))


        switch_coding = "block"
        # generate_cells
        for j, color in zip(range(len(trial_switch_probability)), ["dimgrey", "darkgrey", "silver", "gainsboro"]):
            powers_all_cells, powers_all_cells_shuffled, centre_trials, track_length, _, _, _ = \
                switch_grid_cells(switch_coding, grid_stability, grid_spacings, n_cells, trial_switch_probability[j], field_noise_std=field_noise[i], return_shuffled=True)

            max_block_lengths_diff_all_cells = []
            mean_block_lengths_diff_all_cells = []
            median_block_lengths_diff_all_cells = []
            # classify trials
            for n in range(n_cells):
                rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
                    get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_all_cells[n], power_threshold=Settings.lomb_rolling_threshold,
                                                                 power_step=Settings.power_estimate_step, track_length=track_length,
                                                                 n_window_size=Settings.rolling_window_size_for_lomb_classifier, lomb_frequency_threshold=Settings.lomb_frequency_threshold)

                rolling_lomb_classifier_shuffled, _, _, rolling_frequencies, rolling_points = \
                    get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_all_cells_shuffled[n], power_threshold=Settings.lomb_rolling_threshold,
                                                                 power_step=Settings.power_estimate_step, track_length=track_length,
                                                                 n_window_size=Settings.rolling_window_size_for_lomb_classifier, lomb_frequency_threshold=Settings.lomb_frequency_threshold)

                max_block_lengths_diff_all_cells.append(np.nanmax(get_block_lengths_any_code(rolling_lomb_classifier)-np.nanmax(get_block_lengths_any_code(rolling_lomb_classifier_shuffled))))
                mean_block_lengths_diff_all_cells.append(np.nanmean(get_block_lengths_any_code(rolling_lomb_classifier)-np.nanmean(get_block_lengths_any_code(rolling_lomb_classifier_shuffled))))
                median_block_lengths_diff_all_cells.append(np.nanmedian(get_block_lengths_any_code(rolling_lomb_classifier)-np.nanmedian(get_block_lengths_any_code(rolling_lomb_classifier_shuffled))))

            max_counts, bin_edges = np.histogram(max_block_lengths_diff_all_cells, bins= 10000, range=(-1,1)); bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            ax1.plot(bin_centres, np.cumsum(max_counts/np.sum(max_counts)), color=color, linewidth=3)
            mean_counts, bin_edges = np.histogram(mean_block_lengths_diff_all_cells, bins= 10000, range=(-1,1)); bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            ax2.plot(bin_centres, np.cumsum(mean_counts/np.sum(mean_counts)), color=color, linewidth=3)
            median_counts, bin_edges = np.histogram(median_block_lengths_diff_all_cells, bins= 10000, range=(-1,1)); bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            ax3.plot(bin_centres, np.cumsum(median_counts/np.sum(median_counts)), color=color, linewidth=3, label="B,s:"+str(field_noise[i])+",tsp:"+str(trial_switch_probability[j]))

        # calculate stats for the real data
        block_lengths = pandas_collumn_to_numpy_array(grid_cells["rolling:block_lengths"])
        block_lengths_shuffled = pandas_collumn_to_numpy_array(grid_cells["rolling:block_lengths_shuffled"])
        max_block_lengths = pandas_collumn_to_numpy_array(grid_cells["rolling:max_block_lengths"])
        max_block_lengths_shuffled = pandas_collumn_to_numpy_array(grid_cells["rolling:max_block_lengths_shuffled"])
        mean_block_lengths = pandas_collumn_to_numpy_array(grid_cells["rolling:mean_block_lengths"])
        mean_block_lengths_shuffled = pandas_collumn_to_numpy_array(grid_cells["rolling:mean_block_lengths_shuffled"])
        median_block_lengths = pandas_collumn_to_numpy_array(grid_cells["rolling:median_block_lengths"])
        median_block_lengths_shuffled = pandas_collumn_to_numpy_array(grid_cells["rolling:median_block_lengths_shuffled"])

        max_length_diff = max_block_lengths-max_block_lengths_shuffled
        mean_length_diff = mean_block_lengths-mean_block_lengths_shuffled
        median_length_diff = median_block_lengths-median_block_lengths_shuffled

        max_counts, bin_edges = np.histogram(max_length_diff, bins= 10000, range=(-1,1)); bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        ax1.plot(bin_centres, np.cumsum(max_counts/np.sum(max_counts)), color="red" , linewidth=3)
        mean_counts, bin_edges = np.histogram(mean_length_diff, bins= 10000, range=(-1,1)); bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        ax2.plot(bin_centres, np.cumsum(mean_counts/np.sum(mean_counts)), color="red" , linewidth=3)
        median_counts, bin_edges = np.histogram(median_length_diff, bins= 10000, range=(-1,1)); bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        ax3.plot(bin_centres, np.cumsum(median_counts/np.sum(median_counts)), color="red" , linewidth=3, label="Real data")

        ax1.spines['top'].set_visible(False)
        ax2.set_yticks([])
        ax3.set_yticks([])
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax1.tick_params(axis='both', which='both', labelsize=30)
        ax2.tick_params(axis='both', which='both', labelsize=30)
        ax3.tick_params(axis='both', which='both', labelsize=30)
        ax3.legend(loc='best')
        #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/comparison_field_noise='+str(field_noise[i])+'.png', dpi=300)
        plt.close()

        print("plotted noise:"+str(field_noise[i]))
    return


def get_spatial_information_score_for_trials_simulated_data(track_length, spike_locations, spike_trial_numbers, trial_ids):
    number_of_trials = len(trial_ids)
    spikes_locations = spike_locations[np.isin(spike_trial_numbers, trial_ids)]

    if number_of_trials>0:
        number_of_spikes = len(spikes_locations)

        position_heatmap = np.ones(track_length)*number_of_trials
        occupancy_probability_map = position_heatmap/np.sum(position_heatmap) # Pj

        vr_bin_size_cm = settings.vr_bin_size_cm
        gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/vr_bin_size_cm)

        mean_firing_rate = number_of_spikes/(number_of_trials*track_length) # λ   # the time to complete every lap is 20 seconds using 10cm/s
        spikes, _ = np.histogram(spikes_locations, bins=track_length, range=(0,track_length))
        rates = spikes/position_heatmap
        #rates = convolve(rates, gauss_kernel)
        mrate = mean_firing_rate
        index = rates>0
        Ispike = np.sum(occupancy_probability_map[index] * (rates[index]/mrate) * np.log2(rates[index]/mrate))
        if Ispike < 0:
            print("hello")
    else:
        Ispike = np.nan
    return Ispike

def run_spatial_information_condition(switch_coding, grid_stability, grid_spacings, n_cells, trial_switch_probability, field_noise_std, return_shuffled, save_path):
    # generate_cells
    powers_all_cells, _, centre_trials, track_length, true_classifications_all_cells, spike_locations_all_cells, spike_trial_numbers_all_cells = \
        switch_grid_cells(switch_coding, grid_stability, grid_spacings, n_cells, trial_switch_probability, field_noise_std=field_noise_std, return_shuffled=return_shuffled)

    position_spatial_information_scores = []; n_p_trials = []
    distance_spatial_information_scores = []; n_d_trials = []

    for n in range(n_cells):
        true_classifications = true_classifications_all_cells[n]
        rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
            get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_all_cells[n], power_threshold=Settings.lomb_rolling_threshold,
                                                         power_step=Settings.power_estimate_step, track_length=track_length,
                                                         n_window_size=Settings.rolling_window_size_for_lomb_classifier, lomb_frequency_threshold=Settings.lomb_frequency_threshold)

        rolling_centre_trials, rolling_classifiers, _ = compress_rolling_stats(rolling_points, rolling_lomb_classifier)
        P_trial_numbers = rolling_centre_trials[rolling_classifiers=="P"]
        D_trial_numbers = rolling_centre_trials[rolling_classifiers=="D"]

        position_spatial_information_score = get_spatial_information_score_for_trials_simulated_data(track_length, spike_locations_all_cells[n], spike_trial_numbers_all_cells[n], P_trial_numbers)
        distance_spatial_information_score = get_spatial_information_score_for_trials_simulated_data(track_length, spike_locations_all_cells[n], spike_trial_numbers_all_cells[n], D_trial_numbers)
        position_spatial_information_scores.append(position_spatial_information_score)
        distance_spatial_information_scores.append(distance_spatial_information_score)
        n_p_trials.append(len(true_classifications[true_classifications=="P"])/len(true_classifications))
        n_d_trials.append(len(true_classifications[true_classifications=="D"])/len(true_classifications))

    position_spatial_information_scores = np.array(position_spatial_information_scores)
    distance_spatial_information_scores = np.array(distance_spatial_information_scores)
    n_p_trials = np.array(n_p_trials)
    n_d_trials = np.array(n_d_trials)
    plot_spatial_information_p_vs_d(position_spatial_information_scores, distance_spatial_information_scores, n_p_trials, n_d_trials, switch_coding=switch_coding, field_noise=field_noise_std, trial_switch_probability=trial_switch_probability, save_path=save_path)
    return

def get_p_text(p, ns=True):

    if p is not None:
        if np.isnan(p):
            return " "
        if p<0.00001:
            return 'p < 1e'+('{:.1e}'.format(p)).split("e")[-1]
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

def significance_bar(start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 15,color = 'k'):
    # draw a line with downticks at the ends
    plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)
    # draw the text with a bounding box covering up the line
    plt.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize)

def get_color_to_denort_p_d_trial_number_bias(n_p_trials, n_d_trials):
    import matplotlib
    cmap = matplotlib.cm.get_cmap('RdYlGn')
    color = cmap(n_p_trials)
    alpha = 0.5
    return color, alpha




def plot_spatial_information_p_vs_d(position_spatial_information_scores, distance_spatial_information_scores,  n_p_trials, n_d_trials, switch_coding, field_noise,trial_switch_probability, save_path):

    nan_mask = ~np.isnan(position_spatial_information_scores) & ~np.isnan(distance_spatial_information_scores)
    position_spatial_information_scores = position_spatial_information_scores[nan_mask]
    distance_spatial_information_scores = distance_spatial_information_scores[nan_mask]

    print("comping samples for agreement plot,"+
          " p = ", str(stats.wilcoxon(x=position_spatial_information_scores, y=distance_spatial_information_scores)[1]),
          " t = ", str(stats.wilcoxon(x=position_spatial_information_scores, y=distance_spatial_information_scores)[0]),
          ", df = ",str(len(position_spatial_information_scores)-1))
    _, p = stats.wilcoxon(position_spatial_information_scores, distance_spatial_information_scores)

    fig, ax = plt.subplots(figsize=(4,6))
    for i in range(len(position_spatial_information_scores)):
        c,a = get_color_to_denort_p_d_trial_number_bias(n_p_trials[i], n_d_trials[i])
        ax.scatter([0,1], [position_spatial_information_scores[i], distance_spatial_information_scores[i]], marker="o", color=c)
        ax.plot([0,1],  [position_spatial_information_scores[i], distance_spatial_information_scores[i]], color=c, alpha=a)
    ax.set_xticks([0,1])
    #ax.set_yticks([0, 0.5, 1])
    ax.set_xlim([-0.5,1.5])
    plt.xticks(rotation = 30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels(["Position", "Distance"])
    ax.set_ylabel("Spat info (bits/Hz)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    significance_bar(start=0, end=1, height=ax.get_ylim()[1], displaystring=get_p_text(stats.wilcoxon(position_spatial_information_scores, distance_spatial_information_scores)[1]))
    plt.savefig(save_path + '/' +'position_vs_distance_trials_spatial_scores_switching_coding='+str(switch_coding)+'_field_noise'+str(field_noise)+'_switch_prob'+str(trial_switch_probability)+'.png', dpi=300)
    plt.close()
    return


def spatial_information_across_classifications(field_noise, trial_switch_probability,
                                               save_path, grid_spacings, n_cells):
    for i in range(len(field_noise)):
        grid_stability="imperfect"
        switch_coding = "by_trial"
        run_spatial_information_condition(switch_coding, grid_stability, grid_spacings, n_cells, trial_switch_probability="na", field_noise_std=field_noise[i], return_shuffled=False, save_path=save_path)

        switch_coding = "block"
        for j in range(len(trial_switch_probability)):
            run_spatial_information_condition(switch_coding, grid_stability, grid_spacings, n_cells, trial_switch_probability=trial_switch_probability[j], field_noise_std=field_noise[i], return_shuffled=False, save_path=save_path)
    return

def plot_spatial_infomation_vs_number_of_trials(cell_type_str, grid_spacings, save_path):
    fig, ax = plt.subplots(figsize=(6,6))
    n_cells = len(grid_spacings)
    for j in range(n_cells):
        track_length=200
        n_trials = 100
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, \
        firing_rate_map_by_trial_smoothed, true_classifications = get_cluster_firing(cell_type_str=cell_type_str,
                                                                                     n_trials=n_trials,track_length=track_length,gauss_kernel_std=2,
                                                                                     field_spacing=grid_spacings[j], field_noise_std=5,switch_code_prob=None)
        spat_infos = []
        for i in np.arange(1, n_trials+1):
            spatial_information_score = get_spatial_information_score_for_trials_simulated_data(track_length, spikes_locations, spike_trial_numbers, np.arange(1,n_trials+1)[0:i])
            spat_infos.append(spatial_information_score)
        spat_infos = np.array(spat_infos)

        #ax.scatter(np.arange(1, n_trials+1), spat_infos, marker="o", color="black")
        ax.plot(np.arange(1, n_trials+1), spat_infos, color="black", alpha=0.3)
    #ax.set_xticks([0,1])
    #ax.set_yticks([0, 0.5, 1])
    #ax.set_xlim([-0.5,1.5])
    #plt.xticks(rotation = 30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_xticklabels(["Position", "Distance"])
    #ax.set_ylabel("Spat info (bits/Hz)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    plt.savefig(save_path + '/'+cell_type_str+'_spatial_information_vs_number_of_trials.png', dpi=300)
    plt.close()


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    combined_df = pd.DataFrame()
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort6.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort7.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")], ignore_index=True)
    combined_df = combined_df[combined_df["snippet_peak_to_trough"] < 500] # uV
    combined_df = combined_df[combined_df["track_length"] == 200]
    combined_df = combined_df[combined_df["n_trials"] >= 10]

    save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/position_vs_distance/"
    n_cells = 10
    grid_spacing_low = 40
    grid_spacing_high = 200
    grid_spacings = np.random.uniform(low=grid_spacing_low, high=grid_spacing_high, size=n_cells);
    #compare_block_length_assay(real_data=combined_df, field_noise=[0,5,10], trial_switch_probability=[0.5, 0.1, 0.05, 0.01], save_path=save_path,
    #                           grid_spacings=grid_spacings, n_cells=n_cells)
    np.random.seed(0)

    #spatial_information_across_classifications(field_noise=[0,5,10], trial_switch_probability=[0.01, 0.05, 0.1, 0.1],
    #                                           save_path=save_path,grid_spacings=grid_spacings, n_cells=n_cells)


    plot_spatial_infomation_vs_number_of_trials(cell_type_str="unstable_allocentric_grid_cell", grid_spacings=grid_spacings, save_path=save_path)
    plot_spatial_infomation_vs_number_of_trials(cell_type_str="unstable_egocentric_grid_cell", grid_spacings=grid_spacings, save_path=save_path)



if __name__ == '__main__':
    main()
