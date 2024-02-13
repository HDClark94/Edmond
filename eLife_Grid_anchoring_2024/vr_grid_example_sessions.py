import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from Edmond.eLife_Grid_anchoring_2024.vr_grid_cells import *
from Edmond.eLife_Grid_anchoring_2024.Helpers.array_manipulations import *
import Edmond.eLife_Grid_anchoring_2024.Helpers.plot_utility as plot_utility

warnings.filterwarnings('ignore')

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

def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array


def get_vmin_vmax(cluster_firing_maps, bin_cm=8):
    cluster_firing_maps_reduced = []
    for i in range(len(cluster_firing_maps)):
        cluster_firing_maps_reduced.append(block_reduce(cluster_firing_maps[i], bin_cm, func=np.mean))
    cluster_firing_maps_reduced = np.array(cluster_firing_maps_reduced)
    vmin= 0
    vmax= np.max(cluster_firing_maps_reduced)
    return vmin, vmax

def get_engaged(hit_trial_numbers, processed_position_data, tt):
    hit_trial_numbers = np.array(hit_trial_numbers)
    tt_trial_numbers = np.array(processed_position_data[processed_position_data["trial_type"] == tt]["trial_number"])
    engaged = []
    for tn in processed_position_data["trial_number"]:
        if (tn in hit_trial_numbers) and (tn in tt_trial_numbers):
            engagement = 1
        else:
            engagement = 0
        engaged.append(engagement)
    return np.array(engaged)


def plot_spatial_periodogram(spike_data, save_path, track_length, plot_rolling_marker, n_window_size_for_rolling_window=Settings.rolling_window_size_for_lomb_classifier):

    power_step = Settings.power_estimate_step
    step = Settings.frequency_step
    frequency = Settings.frequency

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times_vr"].iloc[0])

        if len(firing_times_cluster)>1:
            rolling_power_threshold =  cluster_spike_data["rolling_threshold"].iloc[0]
            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)
            modal_class = cluster_spike_data['Lomb_classifier_'].iloc[0]

            fig = plt.figure()
            fig.set_size_inches(5, 5, forward=True)
            ax = fig.add_subplot(1, 1, 1)
            n_y_ticks = int(max(centre_trials)//50)+1
            y_tick_locs= np.linspace(np.ceil(min(centre_trials)), max(centre_trials), n_y_ticks, dtype=np.int64)
            powers[np.isnan(powers)] = 0
            Y, X = np.meshgrid(centre_trials, frequency)
            cmap = plt.cm.get_cmap("inferno")
            ax.pcolormesh(X, Y, powers.T, cmap=cmap, shading="flat")
            for f in range(1,5):
                ax.axvline(x=f, color="white", linewidth=2,linestyle="dotted")
            x_pos = 4.8
            legend_freq = np.linspace(x_pos, x_pos+0.2, 5)
            rolling_lomb_classifier, rolling_lomb_classifier_numeric, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers, power_threshold=rolling_power_threshold, power_step=power_step, track_length=track_length, n_window_size=n_window_size_for_rolling_window)

            rolling_lomb_classifier_tiled = np.tile(rolling_lomb_classifier_numeric,(len(legend_freq),1))
            cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
            boundaries = [0, 1, 2, 3, 4]
            norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
            Y, X = np.meshgrid(centre_trials, legend_freq)
            if plot_rolling_marker:
                ax.pcolormesh(X, Y, rolling_lomb_classifier_tiled, cmap=cmap, norm=norm, shading="flat")
            #ax.set_ylabel('Centre Trial', fontsize=30, labelpad = 10)
            #ax.set_xlabel('Track frequency', fontsize=30, labelpad = 10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.set_yticks(y_tick_locs.tolist())
            ax.set_xlim([0.1,5])
            ax.set_ylim([min(centre_trials), max(centre_trials)])
            ax.yaxis.set_tick_params(labelsize=20)
            ax.xaxis.set_tick_params(labelsize=20)
            #fig.tight_layout(pad=2.0)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
            if plot_rolling_marker:
                plt.savefig(save_path + '/spatial_moving_lomb_scargle_periodogram_with_rolling_marker_' + spike_data.session_id.iloc[cluster_index] + '_' + str(int(cluster_id)) +'.png', dpi=300)
            else:
                plt.savefig(save_path + '/spatial_moving_lomb_scargle_periodogram_' + spike_data.session_id.iloc[cluster_index] + '_' + str(int(cluster_id)) +'.png', dpi=300)
            plt.close()
    return


def plot_coding_schemes_vs_hits(spike_data, processed_position_data, save_path, track_length, n_window_size_for_rolling_window=Settings.rolling_window_size_for_lomb_classifier, plot_spatial_information=False):
    spike_data = spike_data.sort_values(by=["grid_cell", "agreement_between_cell_and_grid_global"], ascending=False)

    cued_hits_trial_numbers = processed_position_data[(processed_position_data["hit_miss_try"] == "hit") &
                                                      (processed_position_data["trial_type"] == 0)]["trial_number"]
    PI_hits_trial_numbers = processed_position_data[(processed_position_data["hit_miss_try"] == "hit") &
                                                    (processed_position_data["trial_type"] == 1)]["trial_number"]
    engaged_cued = get_engaged(cued_hits_trial_numbers, processed_position_data, tt=0)
    engaged_pi = get_engaged(PI_hits_trial_numbers, processed_position_data, tt=1)
    trial_numbers = np.array(processed_position_data["trial_number"])
    trial_types = np.array(processed_position_data["trial_type"])

    rolling_classifiers_all_cells = []
    rolling_classifiers_grid_cells = []
    rolling_classifiers_non_grid_cells = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times_vr"].iloc[0])

        if len(firing_times_cluster)>1:
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)
            rolling_classifiers_numeric = code2_numeric(cluster_spike_data["rolling:classifier_by_trial_number"].iloc[0])

            rolling_classifiers_all_cells.append(rolling_classifiers_numeric.tolist())
            if cluster_spike_data["grid_cell"].iloc[0] == True:
                rolling_classifiers_grid_cells.append(rolling_classifiers_numeric.tolist())
            else:
                rolling_classifiers_non_grid_cells.append(rolling_classifiers_numeric.tolist())

    rolling_classifiers_grid_cells = np.array(rolling_classifiers_grid_cells)
    rolling_classifiers_non_grid_cells = np.array(rolling_classifiers_non_grid_cells)

    if not len(rolling_classifiers_grid_cells)>0:
        return

    # plot the cell coding
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(5.5, 5), gridspec_kw={'width_ratios': [0.7, 2]})
    cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
    boundaries = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    legend_freq = np.arange(0, 0.95, 0.01)
    cmap = colors.ListedColormap(["tab:blue", "tab:red", "tab:red"])
    boundaries = [-1, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    trial_types_tiled = np.tile(trial_types, (len(legend_freq), 1))
    Y, X = np.meshgrid(trial_numbers, legend_freq)
    ax[0].pcolormesh(X, Y, trial_types_tiled, cmap=cmap, norm=norm, shading="flat")

    cmap = colors.ListedColormap(["white", "green"])
    boundaries = [0, 0.9, 2]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    legend_freq += 1
    engaged_cued_tiled = np.tile(engaged_cued, (len(legend_freq), 1))
    Y, X = np.meshgrid(trial_numbers, legend_freq)
    ax[0].pcolormesh(X, Y, engaged_cued_tiled, cmap=cmap, norm=norm, shading="flat")

    legend_freq += 1
    engaged_pi_tiled = np.tile(engaged_pi, (len(legend_freq), 1))
    Y, X = np.meshgrid(trial_numbers, legend_freq)
    ax[0].pcolormesh(X, Y, engaged_pi_tiled, cmap=cmap, norm=norm, shading="flat")

    cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
    boundaries = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    for i in range(len(rolling_classifiers_grid_cells)):
        legend_freq +=1
        Y, X = np.meshgrid(trial_numbers, legend_freq)
        grid_cell_classifier_tiled = np.tile(rolling_classifiers_grid_cells[i], (len(legend_freq), 1))
        ax[1].pcolormesh(X, Y, grid_cell_classifier_tiled, cmap=cmap, norm=norm, shading="flat")
    for i in range(len(rolling_classifiers_non_grid_cells)):
        legend_freq +=1
        Y, X = np.meshgrid(trial_numbers, legend_freq)
        nongrid_cell_classifier_tiled = np.tile(rolling_classifiers_non_grid_cells[i], (len(legend_freq), 1))
        ax[1].pcolormesh(X, Y, nongrid_cell_classifier_tiled, cmap=cmap, norm=norm, shading="flat")

    for axis in ax:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.set_ylim([min(centre_trials), max(centre_trials)])
        axis.xaxis.set_tick_params(labelbottom=False)
        axis.yaxis.set_tick_params(labelleft=False)
        axis.set_xticks([])
        axis.set_yticks([])

    plt.subplots_adjust(hspace=None, wspace=.25, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/cell_modes_and_hits_'+spike_data["session_id_vr"].iloc[0]+'.png', dpi=300)
    plt.close()
    return


def plot_all_firing_rates(spike_data, processed_position_data, save_path, track_length, n_window_size_for_rolling_window=Settings.rolling_window_size_for_lomb_classifier):
    spike_data = spike_data.sort_values(by=["grid_cell", "agreement_between_cell_and_global"], ascending=False)

    power_step = Settings.power_estimate_step
    nrows = 4
    ncols = int(np.ceil(len(spike_data)/nrows))

    i=0
    j=0
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 8), squeeze=False)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times_vr"].iloc[0])

        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            percentile_99th_display = np.nanpercentile(cluster_firing_maps, 99);
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99);
            cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)
            locations = np.arange(0, len(cluster_firing_maps[0]))
            ordered = np.arange(0, len(processed_position_data), 1)
            X, Y = np.meshgrid(locations, ordered)
            cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
            ax[j,i].pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=0, vmax=vmax)

            # plot the classifier
            rolling_power_threshold = cluster_spike_data["rolling_threshold"].iloc[0]
            powers = np.array(cluster_spike_data["MOVING_LOMB_all_powers"].iloc[0])
            centre_trials = np.array(cluster_spike_data["MOVING_LOMB_all_centre_trials"].iloc[0])
            centre_trials = np.round(centre_trials).astype(np.int64)
            powers[np.isnan(powers)] = 0
            x_pos = track_length + 10
            legend_freq = np.linspace(x_pos, x_pos + 16, 5)
            rolling_lomb_classifier, rolling_lomb_classifier_numeric, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(
                centre_trials=centre_trials, powers=powers, power_threshold=rolling_power_threshold, power_step=power_step, track_length=track_length, n_window_size=n_window_size_for_rolling_window)
            rolling_lomb_classifier_tiled = np.tile(rolling_lomb_classifier_numeric, (len(legend_freq), 1))
            cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
            boundaries = [0, 1, 2, 3, 4]
            norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
            Y, X = np.meshgrid(centre_trials, legend_freq)
            ax[j, i].pcolormesh(X, Y, rolling_lomb_classifier_tiled, cmap=cmap, norm=norm, shading="flat")
            ax[j, i].tick_params(axis='both', which='both', labelsize=20)
            ax[j, i].set_xlim([0, track_length+21])
            ax[j, i].set_ylim([0, len(processed_position_data) - 1])
            i+=1
            if i==ncols:
                i=0
                j+=1

    for i in range(ncols):
        for j in range(nrows):
            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            ax[j, i].spines['bottom'].set_visible(False)
            ax[j, i].spines['left'].set_visible(False)
            ax[j, i].tick_params(axis='both', labelsize=10)
            tick_spacing = 100
            ax[j, i].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax[j, i].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            if j != nrows-1:
                ax[j, i].set_xticks([])
                ax[j, i].xaxis.set_tick_params(labelbottom=False)
            if i != 0:
                ax[j, i].set_yticks([])
                ax[j, i].yaxis.set_tick_params(labelleft=False)

    plt.subplots_adjust(hspace=.1, wspace=.1, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/all_firing_rates_'+spike_data["session_id_vr"].iloc[0]+'.png', dpi=300)
    plt.close()
    return

def plot_stops_and_histograms(spike_data, processed_position_data, save_path, track_length):
    axis_thickness = 2

    # make an cell dataframe representative of all cells in the session using the global grid code
    cell = spike_data.head(1)
    cell["rolling:classifier_by_trial_number"] = cell["rolling:grid_code_global"]

    fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(5, 8), gridspec_kw={'height_ratios': [0.25, 0.25, 0.5]})

    # plot histograms
    ymax_for_both = 0
    ymin_for_both = 0
    style_track_plot(ax[0], 200)
    ax[0].axhline(y=0, color="black", linestyle="dashed", linewidth=2)
    stops_counts_p, stops_counts_p_sem, bin_centres, n_trials_P, _, _ = get_stop_histogram(cell, tt=0, coding_scheme="P",shuffle=False,track_length=track_length, trial_classification_column_name="rolling:classifier_by_trial_number")
    stops_counts_d, stops_counts_d_sem, bin_centres, n_trials_D, _, _ = get_stop_histogram(cell, tt=0, coding_scheme="D",shuffle=False,track_length=track_length, trial_classification_column_name="rolling:classifier_by_trial_number")
    stops_counts_n, stops_counts_n_sem, bin_centres, n_trials_N, _, _ = get_stop_histogram(cell, tt=0, coding_scheme="N",shuffle=False,track_length=track_length, trial_classification_column_name="rolling:classifier_by_trial_number")
    ax[0].plot(bin_centres, stops_counts_d[0], color=Settings.egocentric_color, linewidth=2, clip_on=False)
    ax[0].plot(bin_centres, stops_counts_p[0], color=Settings.allocentric_color, linewidth=2, clip_on=False)
    if np.nanmax([np.nanmax(stops_counts_p), np.nanmax(stops_counts_d)]) > ymax_for_both:
        ymax_for_both = np.nanmax([np.nanmax(stops_counts_p), np.nanmax(stops_counts_d)]) + 0.05
    if np.nanmin([np.nanmin(stops_counts_p), np.nanmin(stops_counts_d)]) < ymin_for_both:
        ymin_for_both = np.nanmin([np.nanmin(stops_counts_p), np.nanmin(stops_counts_d)]) - 0.05
    ax[0].set_ylim(bottom=0, top=np.nanmax([np.nanmax(stops_counts_p), np.nanmax(stops_counts_d)]) + 0.05)
    ax[0].text(x=0.7, y=0.8, s="P:" + str(n_trials_P[0]), horizontalalignment='center', verticalalignment='center',transform=ax[0].transAxes)
    ax[0].text(x=0.7, y=1, s="D:" + str(n_trials_D[0]), horizontalalignment='center', verticalalignment='center',transform=ax[0].transAxes)
    ax[0].text(x=0.7, y=0.9, s="N:" + str(n_trials_N[0]), horizontalalignment='center', verticalalignment='center',transform=ax[0].transAxes)
    ax[0].yaxis.set_tick_params(labelsize=25)
    ax[0].set_xlim(0, 200)
    #ax[0].set_ylim(bottom=0, top=0.1)
    #ax[0].set_yticks([0, 0.1])
    ax[0].set_xticks([0, 100, 200])
    ax[0].set_xticklabels(["", "", ""])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[0].spines[axis].set_linewidth(axis_thickness)
    plot_utility.style_vr_plot(ax[0])

    style_track_plot_no_RZ(ax[1], 200)
    stops_counts_p, stops_counts_p_sem, bin_centres, n_trials_P, _, _ = get_stop_histogram(cell, tt=1,coding_scheme="P",shuffle=False,track_length=track_length, trial_classification_column_name="rolling:classifier_by_trial_number")
    stops_counts_d, stops_counts_d_sem, bin_centres, n_trials_D, _, _ = get_stop_histogram(cell, tt=1,coding_scheme="D",shuffle=False,track_length=track_length, trial_classification_column_name="rolling:classifier_by_trial_number")
    stops_counts_n, stops_counts_n_sem, bin_centres, n_trials_N, _, _ = get_stop_histogram(cell, tt=1,coding_scheme="N",shuffle=False,track_length=track_length, trial_classification_column_name="rolling:classifier_by_trial_number")
    ax[1].plot(bin_centres, stops_counts_d[0], color=Settings.egocentric_color, linewidth=2, clip_on=False)
    ax[1].plot(bin_centres, stops_counts_p[0], color=Settings.allocentric_color, linewidth=2, clip_on=False)
    ax[1].text(x=0.7, y=0.8, s="P:" + str(n_trials_P[0]), horizontalalignment='center', verticalalignment='center',transform=ax[1].transAxes)
    ax[1].text(x=0.7, y=1, s="D:" + str(n_trials_D[0]), horizontalalignment='center', verticalalignment='center',transform=ax[1].transAxes)
    ax[1].text(x=0.7, y=0.9, s="N:" + str(n_trials_N[0]), horizontalalignment='center', verticalalignment='center',transform=ax[1].transAxes)
    ax[1].set_xlim(0, 200)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[1].spines[axis].set_linewidth(axis_thickness)
    plot_utility.style_vr_plot(ax[1])
    #ax[1].set_yticks([0,0.1])
    ax[1].set_xticks([0,100,200])
    ax[1].set_xticklabels(["", "", ""])
    if not np.isnan(np.nanmax([np.nanmax(stops_counts_p), np.nanmax(stops_counts_d)])):
        ax[1].set_ylim(bottom=0, top=np.nanmax([np.nanmax(stops_counts_p), np.nanmax(stops_counts_d)]) + 0.05)
    #ax[1].set_ylim(bottom=0, top=0.1)
    ax[1].yaxis.set_tick_params(labelsize=25)

    spike_data = curate_stops_spike_data(spike_data, track_length)
    stop_locations = np.array(spike_data["stop_locations"].iloc[0])
    stop_trial_numbers = np.array(spike_data["stop_trial_numbers"].iloc[0])
    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_number = trial_row["trial_number"].iloc[0]
        trial_stop_color = get_trial_color(trial_type)#
        if trial_stop_color == "lightcoral":
            alpha=0
        else:
            alpha=1
        ax[2].plot(stop_locations[stop_trial_numbers == trial_number],
                   trial_number*np.ones(len(stop_locations[stop_trial_numbers == trial_number])), 'o', color=trial_stop_color, markersize=3, alpha=alpha)

    ax[2].set_xlim(0,track_length)
    tick_spacing = 100
    ax[2].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax[2].yaxis.set_ticks_position('left')
    ax[2].xaxis.set_ticks_position('bottom')
    ax[2].xaxis.set_tick_params(labelsize=25)
    ax[2].yaxis.set_tick_params(labelsize=25)
    plot_utility.style_track_plot(ax[2], track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[2].spines[axis].set_linewidth(axis_thickness)
    plot_utility.style_vr_plot(ax[2], x_max)
    plt.subplots_adjust(hspace =0.2, wspace =None,  bottom=None, left =0.3, right = None, top = None)
    plt.savefig(save_path + '/stops_and_histograms_'+spike_data["session_id_vr"].iloc[0]+'.png', dpi=200)
    plt.close()
    return


def plot_all_firing_rates_compact(spike_data, processed_position_data, save_path, track_length, n_window_size_for_rolling_window=Settings.rolling_window_size_for_lomb_classifier):
    spike_data = spike_data.sort_values(by=["grid_cell", "agreement_between_cell_and_grid_global"], ascending=False)

    nrows = 2
    ncols = 9

    i=0
    j=0
    n=0
    n_max = nrows * ncols

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 4), squeeze=False)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        n+=1
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times_vr"].iloc[0])

        if len(firing_times_cluster)>1:
            cluster_firing_maps = np.array(spike_data['fr_binned_in_space_smoothed'].iloc[cluster_index])
            cluster_firing_maps[np.isnan(cluster_firing_maps)] = 0
            cluster_firing_maps[np.isinf(cluster_firing_maps)] = 0
            cluster_firing_maps = min_max_normalize(cluster_firing_maps)
            percentile_99th = np.nanpercentile(cluster_firing_maps, 99);
            cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
            vmin, vmax = get_vmin_vmax(cluster_firing_maps)
            locations = np.arange(0, len(cluster_firing_maps[0]))
            ordered = np.arange(0, len(processed_position_data), 1)
            X, Y = np.meshgrid(locations, ordered)
            cmap = plt.cm.get_cmap(Settings.rate_map_cmap)

            if n<=n_max:
                ax[j, i].pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=0, vmax=vmax)
                ax[j, i].set_xlim([0, track_length])
                ax[j, i].set_ylim([0, len(processed_position_data)])
            i+=1
            if i==ncols:
                i=0
                j+=1

    for i in range(ncols):
        for j in range(nrows):
            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            ax[j, i].spines['bottom'].set_visible(False)
            ax[j, i].spines['left'].set_visible(False)
            ax[j, i].tick_params(axis='both', labelsize=12)
            tick_spacing = 100
            ax[j, i].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax[j, i].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            if j != nrows-1:
                ax[j, i].set_xticks([])
                ax[j, i].xaxis.set_tick_params(labelbottom=False)
            if i != 0:
                ax[j, i].set_yticks([])
                ax[j, i].yaxis.set_tick_params(labelleft=False)

    plt.subplots_adjust(hspace=.1, wspace=.1, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/all_firing_rates_compact_'+spike_data["session_id_vr"].iloc[0]+'.png', dpi=400)
    plt.close()
    return


def process_recordings(all_firing_df, all_behaviour_df, example_session_ids, save_path):
    for session_id in example_session_ids:
        try:
            print("processing ", session_id)
            firing_df = all_firing_df[all_firing_df["session_id_vr"] == session_id]
            behaviour_df = all_behaviour_df[all_behaviour_df["session_id_vr"] == session_id]

            # session wide plots
            plot_all_firing_rates(spike_data=firing_df, processed_position_data=behaviour_df, save_path=save_path, track_length=200)
            plot_all_firing_rates_compact(spike_data=firing_df, processed_position_data=behaviour_df, save_path=save_path, track_length=200)
            plot_coding_schemes_vs_hits(spike_data=firing_df, processed_position_data=behaviour_df, save_path=save_path, track_length=200)
            plot_stops_and_histograms(spike_data=firing_df, processed_position_data=behaviour_df, save_path=save_path, track_length=200)

            # individual cells with sessions plots
            plot_of_rate_map(spike_data=firing_df, save_path=save_path)
            plot_of_autocorrelogram(spike_data=firing_df, save_path=save_path)
            plot_firing_rate_maps_short(spike_data=firing_df, processed_position_data=behaviour_df, save_path=save_path, track_length=200)
            plot_firing_rate_maps_per_trial(spike_data=firing_df, processed_position_data=behaviour_df, save_path=save_path, track_length=200)
            plot_spatial_autocorrelogram_fr(spike_data=firing_df, save_path=save_path, track_length=200)
            plot_spatial_periodogram(spike_data=firing_df, save_path=save_path, track_length=200, plot_rolling_marker=True)
            plot_firing_rate_maps_short_with_rolling_classifications(spike_data=firing_df, save_path=save_path, track_length=200)
            plot_avg_spatial_periodograms_with_rolling_classifications(spike_data=firing_df, processed_position_data=behaviour_df, save_path=save_path, track_length =200)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("")


def main():
    print("------------------------------------------------------------------------------------------------")
    print("-----------------------------------------Hello there--------------------------------------------")
    print("------------------------------------------------------------------------------------------------")
    print("---- This analysis was run on a ubuntu 16.04 image on a 96GB ram virtual machine (with 30GB ----")
    print("---- extra swap space) using the conda environment provided in grid_behaviour.yml           ----")
    print("------------------------------------------------------------------------------------------------")
    print("---- This script (1) loads spatial firing dataframes                                        ----")
    print("----             (2) filters out cells based on an exclusion criteria                       ----")
    print("----             (3) creates example cell plots as seen in Clark and Nolan 2023             ----")
    print("------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------")

    save_path = "/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/plots/example_sessions"

    # load dataframe, each row represents one cell

    # load dataframe, each row represents one cell
    combined_df = pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/combined_cohorts.pkl")
    #combined_df = pd.DataFrame()
    #combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/combined_cohort6.pkl")], ignore_index=True)
    #combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/combined_cohort7.pkl")], ignore_index=True)
    #combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/combined_cohort8.pkl")], ignore_index=True)

    # load behaviour dataframe
    all_behaviour = pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/all_behaviour_200cm.pkl") # load behaviour-only dataframe

    # remove artefacts, low firing rates in the open field, sessions with non 200cm track lengths and sessions with less than 10 trials
    combined_df = combined_df[combined_df["snippet_peak_to_trough"] < 500] # uV
    combined_df = combined_df[combined_df["mean_firing_rate_of"] > 0.2] # Hz
    combined_df = combined_df[combined_df["track_length"] == 200]
    combined_df = combined_df[combined_df["n_trials"] >= 10]
    combined_df = add_lomb_classifier(combined_df,suffix="")
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"] # removes cells without firing in the virtual reality track

    # remove mice without any grid cells
    combined_df = combined_df[combined_df["mouse"] != "M2"]
    combined_df = combined_df[combined_df["mouse"] != "M4"]
    combined_df = combined_df[combined_df["mouse"] != "M15"]

    # Example sessions listed in order of appearance
    example_session_ids =    ["M11_D19_2021-06-03_10-50-41", # Fig 2A and Fig5C
                              "M1_D8_2020-08-12_15-06-01", # Fig 2A
                              "M11_D43_2021-07-07_11-51-08", # Fig 2A
                              "M14_D14_2021-05-27_11-46-30", # Fig 2B
                              "M14_D34_2021-06-24_12-48-57", # Fig 2B  and Fig5D
                              "M13_D24_2021-06-10_12-01-54", # Fig 2B
                              "M13_D29_2021-06-17_11-50-37", # Fig 2C
                              "M11_D21_2021-06-07_10-26-21", # Fig 3B
                              "M14_D31_2021-06-21_12-07-01", # Fig 3C and Fig 5-S3F
                              "M1_D11_2020-08-17_14-57-20", # Fig 3D and Fig 5-S3A
                              "M11_D36_2021-06-28_12-04-36", # Fig 3E and Fig 4
                              "M11_D45_2021-07-09_11-39-02", # Fig 5E
                              "M14_D12_2021-05-25_11-03-39", # Fig 5F
                              "M3_D18_2020-11-21_14-29-49", # Fig 5-S3B
                              "M6_D23_2020-11-28_17-01-43", # Fig 5-S3C
                              "M7_D25_2020-11-30_16-24-49", # Fig 5-S3D
                              "M11_D39_2021-07-01_11-47-10"] # Fig 5-S3E

    process_recordings(combined_df, all_behaviour, example_session_ids, save_path=save_path)

    print("------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------")



if __name__ == '__main__':
    main()