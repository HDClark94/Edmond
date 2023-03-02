import numpy as np
from matplotlib.ticker import MaxNLocator
from Edmond.Toy_grid_cell.plot_example_periodic_cells import *
from Edmond.VR_grid_analysis.vr_grid_cells import get_max_SNR, distance_from_integer, get_lomb_classifier, get_rolling_lomb_classifier_for_centre_trial, compress_rolling_stats, get_rolling_lomb_classifier_for_centre_trial2
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

def switch_grid_cells(switch_coding_mode, grid_stability, grid_spacings, n_cells, trial_switch_probability, field_noise_std):
    # generate n switch grid cells
    # generated n positional grid cells
    powers_all_cells = []
    true_classifications_all_cells = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        _, _, _, firing_rate_map_by_trial_smoothed, true_classifications = get_switch_cluster_firing(switch_coding_mode=switch_coding_mode,
                                                                                                     grid_stability=grid_stability,
                                                                                                     field_spacing=grid_spacing,
                                                                                                     trial_switch_probability=trial_switch_probability,
                                                                                                     field_noise_std=field_noise_std)

        powers, centre_trials, track_length = generate_spatial_periodogram(firing_rate_map_by_trial_smoothed)
        powers_all_cells.append(powers)
        true_classifications_all_cells.append(true_classifications)

    return powers_all_cells, centre_trials, track_length, true_classifications_all_cells

def perfect_grid_cells(grid_spacings, n_cells, field_noise_std):

    # generated n positional grid cells
    position_peak_frequencies = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed,_ = get_cluster_firing(cell_type_str="stable_allocentric_grid_cell", field_spacing=grid_spacing, field_noise_std=field_noise_std)
        max_peak_power, max_peak_freq = compute_peak(firing_rate_map_by_trial_smoothed)
        position_peak_frequencies.append(max_peak_freq)

        # generated n distance grid cells
    distance_peak_frequencies = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed,_ = get_cluster_firing(cell_type_str="stable_egocentric_grid_cell", field_spacing=grid_spacing, field_noise_std=field_noise_std)
        max_peak_power, max_peak_freq = compute_peak(firing_rate_map_by_trial_smoothed)
        distance_peak_frequencies.append(max_peak_freq)

    return np.array(position_peak_frequencies), np.array(distance_peak_frequencies)


def imperfect_grid_cells(grid_spacings, n_cells, field_noise_std):

    # generated n positional grid cells
    position_peak_frequencies = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed,_ = get_cluster_firing(cell_type_str="unstable_allocentric_grid_cell", field_spacing=grid_spacing, field_noise_std=field_noise_std)
        max_peak_power, max_peak_freq = compute_peak(firing_rate_map_by_trial_smoothed)
        position_peak_frequencies.append(max_peak_freq)

        # generated n distance grid cells
    distance_peak_frequencies = []
    for i in range(n_cells):
        grid_spacing = grid_spacings[i]
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed,_ = get_cluster_firing(cell_type_str="unstable_egocentric_grid_cell", field_spacing=grid_spacing, field_noise_std=field_noise_std)
        max_peak_power, max_peak_freq = compute_peak(firing_rate_map_by_trial_smoothed)
        distance_peak_frequencies.append(max_peak_freq)

    return np.array(position_peak_frequencies), np.array(distance_peak_frequencies)



def run_whole_cell_analysis(grid_stability, save_path, grid_spacings, n_cells, field_noise_std, plot_suffix=""):

    # first lets collect the frequency peaks from the position and distance encoding grid cells
    if grid_stability == "perfect":
        position_peak_frequencies, distance_peak_frequencies = perfect_grid_cells(grid_spacings, n_cells, field_noise_std)
    elif grid_stability == "imperfect":
        position_peak_frequencies, distance_peak_frequencies = imperfect_grid_cells(grid_spacings, n_cells, field_noise_std)

    # Next lets plot the distibution of the position and distance encoding across the spatial frequencies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4), gridspec_kw={'width_ratios': [1, 1]})

    pos_counts, bin_edges = np.histogram(position_peak_frequencies, bins=25, range=[0, 5])
    dis_counts, bin_edges = np.histogram(distance_peak_frequencies, bins=25, range=[0, 5])
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    ax1.bar(bin_centres, pos_counts, color=Settings.allocentric_color, alpha=0.5, width=(bin_edges[0]-bin_edges[1]))
    ax1.bar(bin_centres, dis_counts, color=Settings.egocentric_color, alpha=0.5, width=(bin_edges[0]-bin_edges[1]))

    position_peak_frequencies_from_int = distance_from_integer(position_peak_frequencies)
    distance_peak_frequencies_from_int = distance_from_integer(distance_peak_frequencies)
    pos_counts, bin_edges = np.histogram(position_peak_frequencies_from_int, bins=25, range=[0, 0.5])
    dis_counts, bin_edges = np.histogram(distance_peak_frequencies_from_int, bins=25, range=[0, 0.5])
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    ax2.bar(bin_centres, pos_counts, color=Settings.allocentric_color, alpha=0.5, width=(bin_edges[0]-bin_edges[1]))
    ax2.bar(bin_centres, dis_counts, color=Settings.egocentric_color, alpha=0.5, width=(bin_edges[0]-bin_edges[1]))

    ax1.set_ylabel("Number of cells",color="black",fontsize=25, labelpad=10)
    ax1.set_xlabel("Spatial frequency", color="black", fontsize=25, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax2.set_xticks([0, 0.5])
    ax2.set_xticklabels(["0", "0.5"])
    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax2.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    for f in range(1,6):
        ax1.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.set_xlim([-0.05,5.05])
    ax2.set_xlim([-0.05,0.55])
    ax2.set_xlabel(r'$\Delta$ from Int', color="black", fontsize=25, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path + '/'+grid_stability+''+plot_suffix+'.png', dpi=200)
    plt.close()

    # Finally lets use the frequency peaks to determine an optimal threshold
    frequency_thresholds = np.arange(0.01, 0.5, 0.01)
    pos_accuracies = []
    dis_accuracies = []
    total_accuracies = []
    for freq_threshold in frequency_thresholds:
        pos_classification_accuracies = []
        dis_classification_accuracies = []

        # compute accuracy for true position cells
        for i in range(n_cells):
            pos_cell_peak_freq = position_peak_frequencies[i]
            dis_cell_peak_freq = distance_peak_frequencies[i]

            if get_lomb_classifier(1, pos_cell_peak_freq, 0, freq_threshold) == "Position":
                pos_classification_accuracies.append(1)
            else:
                pos_classification_accuracies.append(0)
            if get_lomb_classifier(1, dis_cell_peak_freq, 0, freq_threshold) == "Distance":
                dis_classification_accuracies.append(1)
            else:
                dis_classification_accuracies.append(0)

        pos_classification_accuracies = np.array(pos_classification_accuracies)
        dis_classification_accuracies = np.array(dis_classification_accuracies)

        pos_accuracy = np.sum(pos_classification_accuracies!=1)/len(position_peak_frequencies)
        dis_accuracy = np.sum(dis_classification_accuracies!=1)/len(distance_peak_frequencies)
        total_accuracy = (np.sum(dis_classification_accuracies!=1)+np.sum(pos_classification_accuracies!=1))/(len(distance_peak_frequencies)+len(position_peak_frequencies))

        pos_accuracies.append(pos_accuracy)
        dis_accuracies.append(dis_accuracy)
        total_accuracies.append(total_accuracy)

    pos_accuracies = np.array(pos_accuracies)
    dis_accuracies = np.array(dis_accuracies)
    total_accuracies = np.array(total_accuracies)

    # now lets plot the accuracies to determine the ideal place to threshold
    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.plot(frequency_thresholds, pos_accuracies, linewidth=3, color=Settings.allocentric_color)
    ax.plot(frequency_thresholds, dis_accuracies, linewidth=3, color=Settings.egocentric_color)
    ax.plot(frequency_thresholds, total_accuracies, linewidth=3, color="black")
    ax.set_title("min="+str(np.round(total_accuracies.min(), decimals=2))+"@Hz="+str(frequency_thresholds[np.where(total_accuracies==total_accuracies.min())[0][0]]), fontsize=20)
    ax.set_ylabel('False positive rate', fontsize=30, labelpad = 10)
    ax.set_xlabel('Frequency threshold', fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 1])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/threshold_accuracies_'+grid_stability+'_cell'+plot_suffix+'.png', dpi=300)
    plt.close()
    return

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

def run_switch_coding_analysis(switch_coding_mode, grid_stability, save_path, grid_spacings, n_cells, trial_switch_probability, field_noise_std=5, plot_suffix=""):
    """
    this analysis will simulate n cells for 100 trials, either perfect or imperfect grid cell
    each cell will switch between allcentric and egocentric coding. To simulate blocks of allocentric/egocentric coding,
    for each trial there is a probability set for the code to switch

    Once all simulations are complete,
    this analysis will assay the frequency threshold for classifying position and distance encoding trials as well as
    the rolling window size for coding epochs
    """

    # first lets generate n cells
    powers_all_cells, centre_trials, track_length, true_classifications = \
        switch_grid_cells(switch_coding_mode, grid_stability, grid_spacings, n_cells, trial_switch_probability, field_noise_std)

    # next lets assay the spatial frequency threshold and the rolling window size
    frequency_thresholds = np.arange(0.01, 0.5, 0.02)
    rolling_window_sizes = np.arange(1, 201, 20)

    total_coding_accuracy = np.zeros((len(frequency_thresholds), len(rolling_window_sizes)))
    position_coding_accuracy = np.zeros((len(frequency_thresholds), len(rolling_window_sizes)))
    distance_coding_accuracy = np.zeros((len(frequency_thresholds), len(rolling_window_sizes)))

    for m in range(len(frequency_thresholds)):
        for n in range(len(rolling_window_sizes)):

            total_accuracies = []
            pos_accuracies = []
            dis_accuracies = []
            for i in range(n_cells):
                rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
                    get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_all_cells[i], power_threshold=0,
                                                                 power_step=Settings.power_estimate_step, track_length=track_length,
                                                                 n_window_size=rolling_window_sizes[n], lomb_frequency_threshold=frequency_thresholds[m])
                rolling_centre_trials, rolling_classifiers, _ = compress_rolling_stats(centre_trials, rolling_lomb_classifier)

                # compare the true classifications to the estimated classifications
                cell_true_classifications = true_classifications[i]
                cell_true_classifications = ignore_end_trials_in_block(cell_true_classifications)
                cell_true_classifications = cell_true_classifications[1:len(rolling_classifiers)+1]

                cell_pos_FPR = np.sum(rolling_classifiers[cell_true_classifications == "P"] != cell_true_classifications[cell_true_classifications == "P"])/np.sum(cell_true_classifications == "P")
                cell_dis_FPR = np.sum(rolling_classifiers[cell_true_classifications == "D"] != cell_true_classifications[cell_true_classifications == "D"])/np.sum(cell_true_classifications == "D")
                cell_FPR = (np.sum(rolling_classifiers[cell_true_classifications == "P"] != cell_true_classifications[cell_true_classifications == "P"])+
                            np.sum(rolling_classifiers[cell_true_classifications == "D"] != cell_true_classifications[cell_true_classifications == "D"]))/\
                           (np.sum(cell_true_classifications == "P")+np.sum(cell_true_classifications == "D"))
                total_accuracies.append(cell_FPR)
                pos_accuracies.append(cell_pos_FPR)
                dis_accuracies.append(cell_dis_FPR)

            position_coding_accuracy[m,n] = np.nanmean(np.array(pos_accuracies))
            distance_coding_accuracy[m,n] = np.nanmean(np.array(dis_accuracies))
            total_coding_accuracy[m,n] = np.nanmean(np.array(total_accuracies))

    # Next lets plot the distibution of the position and distance encoding across the spatial frequencies
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 1, 1]})

    Y, X = np.meshgrid(frequency_thresholds, rolling_window_sizes)
    cmap = plt.cm.get_cmap("cividis")
    ax1.pcolormesh(X, Y, total_coding_accuracy.T, cmap=cmap, shading="flat", vmin=0, vmax=1)
    ax2.pcolormesh(X, Y, position_coding_accuracy.T, cmap=cmap, shading="flat", vmin=0, vmax=1)
    ax3.pcolormesh(X, Y, distance_coding_accuracy.T, cmap=cmap, shading="flat", vmin=0, vmax=1)

    ax1.set_ylabel("Spatial frequency",color="black",fontsize=20, labelpad=10)
    ax1.set_xlabel("Rolling window sizes", color="black", fontsize=20, labelpad=10)
    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    plt.setp(ax2.get_xticklabels(), fontsize=20)
    plt.setp(ax2.get_yticklabels(), fontsize=20)
    plt.setp(ax3.get_xticklabels(), fontsize=20)
    plt.setp(ax3.get_yticklabels(), fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path + '/switch_assay_'+grid_stability+'_cell'+plot_suffix+'.png', dpi=300)
    plt.close()

    # Finally lets use the frequency peaks to determine an optimal threshold
    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_title("min="+str(np.round(total_coding_accuracy.min(), decimals=2))+"@ws="+str(np.round(rolling_window_sizes[np.argwhere(total_coding_accuracy==total_coding_accuracy.min())[0][0]]))+"@Hz="
                 +str(frequency_thresholds[np.where(total_coding_accuracy==total_coding_accuracy.min())[1][0]]), fontsize=20)
    ax.plot(frequency_thresholds, position_coding_accuracy[:, np.argwhere(total_coding_accuracy==total_coding_accuracy.min())[0][0]], linewidth=3, color=Settings.allocentric_color)
    ax.plot(frequency_thresholds, distance_coding_accuracy[:, np.argwhere(total_coding_accuracy==total_coding_accuracy.min())[0][0]], linewidth=3, color=Settings.egocentric_color)
    ax.plot(frequency_thresholds, total_coding_accuracy[:, np.argwhere(total_coding_accuracy==total_coding_accuracy.min())[0][0]], linewidth=3, color="black")
    ax.set_ylabel('False positive rate', fontsize=20, labelpad = 10)
    ax.set_xlabel('Frequency threshold', fontsize=20, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 1])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/switch_threshold_accuracies_'+grid_stability+'_cell'+plot_suffix+'.png', dpi=300)
    plt.close()
    return


def run_switch_coding_analysis2(switch_coding_mode, grid_stability, save_path, grid_spacings, n_cells, trial_switch_probability, field_noise_std=5, plot_suffix=""):
    """
    this analysis will simulate n cells for 100 trials, either perfect or imperfect grid cell
    each cell will switch between allcentric and egocentric coding. To simulate blocks of allocentric/egocentric coding,
    for each trial there is a probability set for the code to switch

    Once all simulations are complete,
    this analysis will assay the frequency threshold for classifying position and distance encoding trials as well as
    the rolling window size for coding epochs
    """

    # first lets generate n cells
    powers_all_cells, centre_trials, track_length, true_classifications = \
        switch_grid_cells(switch_coding_mode, grid_stability, grid_spacings, n_cells, trial_switch_probability, field_noise_std)

    # next lets assay the spatial frequency threshold and the rolling window size
    frequency_thresholds = np.arange(0, 0.5, 0.02)

    total_coding_accuracy = np.zeros(len(frequency_thresholds))
    position_coding_accuracy = np.zeros(len(frequency_thresholds))
    distance_coding_accuracy = np.zeros(len(frequency_thresholds))

    for m in range(len(frequency_thresholds)):

        total_accuracies = []
        pos_accuracies = []
        dis_accuracies = []
        for i in range(n_cells):
            rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
                get_rolling_lomb_classifier_for_centre_trial2(centre_trials=centre_trials, powers=powers_all_cells[i], power_threshold=0,
                                                             power_step=Settings.power_estimate_step, track_length=track_length, lomb_frequency_threshold=frequency_thresholds[m])

            # compare the true classifications to the estimated classifications
            cell_true_classifications = true_classifications[i]
            cell_true_classifications = ignore_end_trials_in_block(cell_true_classifications)
            cell_true_classifications = cell_true_classifications[1:len(rolling_lomb_classifier)+1]

            cell_pos_FPR = np.sum(rolling_lomb_classifier[cell_true_classifications == "P"] != cell_true_classifications[cell_true_classifications == "P"])/np.sum(cell_true_classifications == "P")
            cell_dis_FPR = np.sum(rolling_lomb_classifier[cell_true_classifications == "D"] != cell_true_classifications[cell_true_classifications == "D"])/np.sum(cell_true_classifications == "D")
            cell_FPR = (np.sum(rolling_lomb_classifier[cell_true_classifications == "P"] != cell_true_classifications[cell_true_classifications == "P"])+
                        np.sum(rolling_lomb_classifier[cell_true_classifications == "D"] != cell_true_classifications[cell_true_classifications == "D"]))/ \
                       (np.sum(cell_true_classifications == "P")+np.sum(cell_true_classifications == "D"))
            total_accuracies.append(cell_FPR)
            pos_accuracies.append(cell_pos_FPR)
            dis_accuracies.append(cell_dis_FPR)

        position_coding_accuracy[m] = np.nanmean(np.array(pos_accuracies))
        distance_coding_accuracy[m] = np.nanmean(np.array(dis_accuracies))
        total_coding_accuracy[m] = np.nanmean(np.array(total_accuracies))

    # Finally lets use the frequency peaks to determine an optimal threshold
    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_title("min="+str(np.round(total_coding_accuracy.min(), decimals=2))+"@Hz="+str(np.round(frequency_thresholds[np.argmin(total_coding_accuracy)], decimals=2)), fontsize=20)
    ax.plot(frequency_thresholds, position_coding_accuracy, linewidth=3, color=Settings.allocentric_color)
    ax.plot(frequency_thresholds, distance_coding_accuracy, linewidth=3, color=Settings.egocentric_color)
    ax.plot(frequency_thresholds, total_coding_accuracy, linewidth=3, color="black")
    ax.set_ylabel('False positive rate', fontsize=20, labelpad = 10)
    ax.set_xlabel('Spat. Freq. threshold', fontsize=20, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 1])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/switch_threshold_accuracies_'+grid_stability+'_cell'+plot_suffix+'.png', dpi=300)
    plt.close()
    return



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    np.random.seed(0)

    save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/position_vs_distance/"
    n_cells = 50
    grid_spacing_low = 40
    grid_spacing_high = 200
    field_noise_std = 10
    swtich_coding = "block" # enter "block" or "by_trial"
    grid_spacings = np.random.uniform(low=grid_spacing_low, high=grid_spacing_high, size=n_cells);
    plot_suffix="_grid_spacings-"+str(grid_spacing_low)+"-"+str(grid_spacing_high)+"cm_field_noise-"+str(field_noise_std)+"sigma_switch_coding="+swtich_coding
    #run_switch_coding_analysis2(switch_coding_mode=swtich_coding, grid_stability="imperfect", save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells, trial_switch_probability=0.1, field_noise_std=field_noise_std, plot_suffix=plot_suffix)
    run_whole_cell_analysis(grid_stability="imperfect", save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells,field_noise_std=field_noise_std, plot_suffix=plot_suffix)

    swtich_coding = "by_trial" # enter "block" or "by_trial"
    plot_suffix="_grid_spacings-"+str(grid_spacing_low)+"-"+str(grid_spacing_high)+"cm_field_noise-"+str(field_noise_std)+"sigma_switch_coding="+swtich_coding
    #run_switch_coding_analysis2(switch_coding_mode=swtich_coding, grid_stability="imperfect", save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells, trial_switch_probability=0.1, field_noise_std=field_noise_std, plot_suffix=plot_suffix)
    run_whole_cell_analysis(grid_stability="imperfect", save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells,field_noise_std=field_noise_std, plot_suffix=plot_suffix)

    # rerun for larger grid cell periods
    grid_spacing_low = 200
    grid_spacing_high = 600
    grid_spacings = np.random.uniform(low=grid_spacing_low, high=grid_spacing_high, size=n_cells);
    swtich_coding = "block" # enter "block" or "by_trial"
    plot_suffix="_grid_spacings-"+str(grid_spacing_low)+"-"+str(grid_spacing_high)+"cm_field_noise-"+str(field_noise_std)+"sigma_switch_coding="+swtich_coding
    #run_switch_coding_analysis2(switch_coding_mode=swtich_coding, grid_stability="imperfect", save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells, trial_switch_probability=0.1, field_noise_std=field_noise_std, plot_suffix=plot_suffix)
    run_whole_cell_analysis(grid_stability="imperfect", save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells,field_noise_std=field_noise_std, plot_suffix=plot_suffix)

    swtich_coding = "by_trial" # enter "block" or "by_trial"
    plot_suffix="_grid_spacings-"+str(grid_spacing_low)+"-"+str(grid_spacing_high)+"cm_field_noise-"+str(field_noise_std)+"sigma_switch_coding="+swtich_coding
    #run_switch_coding_analysis2(switch_coding_mode=swtich_coding, grid_stability="imperfect", save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells, trial_switch_probability=0.1, field_noise_std=field_noise_std, plot_suffix=plot_suffix)
    run_whole_cell_analysis(grid_stability="imperfect", save_path=save_path, grid_spacings=grid_spacings, n_cells=n_cells,field_noise_std=field_noise_std, plot_suffix=plot_suffix)
    print("look now")

if __name__ == '__main__':
    main()
