import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel
from Edmond.VR_grid_analysis.vr_grid_stability_plots import get_allocentric_peak, get_egocentric_peak, get_rolling_lomb_classifier_for_centre_trial
import matplotlib.ticker as ticker
import Edmond.VR_grid_analysis.analysis_settings as Settings

from Edmond.VR_grid_analysis.vr_grid_cells import get_max_int_SNR, get_max_SNR, reduce_digits, get_first_peak
plt.rc('axes', linewidth=3)

def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def plot_power_hypotheses_mfr(output_path):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xs = np.linspace(-np.pi, 4*np.pi, 10000)
    ax.plot(xs, np.sin(2*xs), label="Trial 1", linewidth=2, color="black")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=-1, top=1.05)
    ax.set_xlim(left=0, right=10)
    fig.tight_layout()
    ax.set_ylabel('FR (Hz)', fontsize=25, labelpad=8)
    ax.set_xlabel('Position', fontsize=25, labelpad=8)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(output_path + '/grid_cell_jitter_mfr.png', dpi=200)
    plt.close()

def plot_power_hypotheses(jitter, output_path):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    colors = cm.rainbow(np.linspace(0, 1, 8))
    xs = np.linspace(-np.pi, 4*np.pi, 10000)
    ax.plot(xs, np.sin(2*xs), label="Trial 1", linewidth=2, color=colors[0])
    ax.plot(xs+jitter, np.sin(2*xs), label="Trial 2", linewidth=2, color=colors[1])
    ax.plot(xs-jitter, np.sin(2*xs), label="Trial 3", linewidth=2, color=colors[2])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=-1, top=1.05)
    ax.set_xlim(left=0, right=10)
    fig.tight_layout()
    ax.set_ylabel('FR (Hz)', fontsize=25, labelpad=8)
    ax.set_xlabel('Position', fontsize=25, labelpad=8)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(output_path + '/grid_cell_jitter'+str(jitter)+'.png', dpi=200)
    plt.close()

def plot_zero_inflation_hypotheses(jitter, output_path):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    colors = cm.rainbow(np.linspace(0, 1, 8))
    xs = np.linspace(-np.pi, 4*np.pi, 10000)
    y1 = np.sin(2*xs)
    y2 = np.sin(2*xs); y2[3500:5550] = -1
    y3 = np.sin(2*xs); y3[5550:7500] = -1
    ax.plot(xs+jitter, y1, label="Trial 1", linewidth=2, color=colors[0])
    ax.plot(xs, y2, label="Trial 2", linewidth=2, color=colors[1])
    ax.plot(xs-jitter, y3, label="Trial 3", linewidth=2, color=colors[2])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=-1, top=1.05)
    ax.set_xlim(left=0, right=10)
    fig.tight_layout()
    ax.set_ylabel('FR (Hz)', fontsize=25, labelpad=8)
    ax.set_xlabel('Position', fontsize=25, labelpad=8)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(output_path + '/grid_cell_skipped_fields.png', dpi=200)
    plt.close()

def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array

def plot_allo_ego_grid_cell(output_path):
    n_trials=100
    bin_size_cm=1
    sampling_rate=100
    avg_speed_cmps=10
    p_scalar=1
    track_length = 200
    grid_spacing = 90
    gauss_kernel = Gaussian1DKernel(stddev=1)

    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    trial_numbers = (locations//track_length)+1

    locations_first_half = locations[:int(len(locations)/2)]
    locations_second_half = locations[int(len(locations)/2):]

    trial_numbers_first_half = (locations_first_half//track_length)+1
    trial_numbers_second_half = (locations_second_half//track_length)+1

    #================= FIRST HALF _ ALLOCENTRIC GRID CELL ==========================#
    spikes_at_locations_first_half = []
    for trial_number in np.unique(trial_numbers_first_half):
        trial_locations = (locations_first_half%track_length)[trial_numbers_first_half==trial_number]
        firing_p = np.sin(2*np.pi*(1/grid_spacing)*trial_locations)
        firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
        firing_p = min_max_normlise(firing_p, 0, 1)
        firing_p = firing_p*p_scalar
        spikes_at_locations_trial = np.zeros(len(trial_locations))
        for i in range(len(spikes_at_locations_trial)):
            spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
        spikes_at_locations_first_half.extend(spikes_at_locations_trial.tolist())
    spikes_at_locations_first_half = np.array(spikes_at_locations_first_half)
    spike_locations = locations_first_half[spikes_at_locations_first_half==1]
    trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length
    rates = []
    for trial_number in np.unique(trial_numbers_first_half):
        trial_spike_locations = spike_locations[trial_numbers == trial_number]
        trial_rates, bin_edges = np.histogram(trial_spike_locations, bins=int(track_length/bin_size_cm), range=(0, track_length))
        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        rates.append(trial_rates.tolist())
    rates_first_half = np.array(rates)
    #================= FIRST HALF _ ALLOCENTRIC GRID CELL ==========================#


    #================= SECOND HALF _ EGOCENTRIC GRID CELL ==========================#
    firing_p = np.sin(2*np.pi*(1/grid_spacing)*locations_second_half)
    firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
    firing_p = min_max_normlise(firing_p, 0, 1)
    firing_p = firing_p*p_scalar
    spikes_at_locations_second_half = np.zeros(len(locations_second_half))
    for i in range(len(locations_second_half)):
        spikes_at_locations_second_half[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    spike_locations = locations_second_half[spikes_at_locations_second_half==1]
    trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length
    rates = []
    for trial_number in np.unique(trial_numbers_second_half):
        trial_spike_locations = spike_locations[trial_numbers == trial_number]
        trial_rates, bin_edges = np.histogram(trial_spike_locations, bins=int(track_length/bin_size_cm), range=(0, track_length))
        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        rates.append(trial_rates.tolist())
    rates_second_half = np.array(rates)
    #================= SECOND HALF _ EGOCENTRIC GRID CELL ==========================#


    # Concatenate first and second half
    spikes_at_locations = np.append(spikes_at_locations_first_half, spikes_at_locations_second_half)
    rates = np.vstack((rates_first_half, rates_second_half))

    locations_ = np.arange(0, len(rates[0]))
    trial_numbers_ = np.arange(0, len(rates))+1
    X, Y = np.meshgrid(locations_, trial_numbers_)
    cmap = plt.cm.get_cmap(Settings.rate_map_cmap)

    rates_flattened = np.ravel(rates)
    rates_flattened = convolve(rates_flattened, gauss_kernel)
    rates_smoothened = rates_flattened.reshape((len(rates), len(rates[0])))

    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    c = ax.pcolormesh(X, Y, rates_smoothened, cmap=cmap, shading="auto", vmin=np.min(rates), vmax=np.max(rates))
    plt.ylabel('Trial number', fontsize=30, labelpad = 10)
    plt.xlabel("Location (cm)", fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xticks([0,100,200])
    ax.set_yticks([1,50,100])

    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0)
    cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
    cbar.set_ticks([0,np.max(rates_smoothened)])
    cbar.set_ticklabels(["0", "Max"])
    cbar.ax.tick_params(size=0)
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plot_path = output_path + '/allo_and_egocentric_grid_cell_rates.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Calculate the Lomb - Scargle periodogram
    numerator, bin_edges = np.histogram(locations, bins=2*int(track_length/1)*n_trials, range=(0, track_length*n_trials), weights=spikes_at_locations)
    denominator, _ = np.histogram(locations, bins=2*int(track_length/1)*n_trials, range=(0, track_length*n_trials))
    set_fr = numerator/denominator
    set_elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
    gauss_kernel = Gaussian1DKernel(stddev=1)
    set_fr = convolve(set_fr, gauss_kernel)
    set_fr = moving_sum(set_fr, window=2)/2
    set_fr = np.append(set_fr, np.zeros(len(set_elapsed_distance)-len(set_fr)))
    step = Settings.frequency_step
    frequency = Settings.frequency
    sliding_window_size=track_length*15
    powers = []
    centre_distances = []
    indices_to_test = np.arange(0, len(set_fr)-sliding_window_size, 1, dtype=np.int64)[::10]
    for j in indices_to_test:
        ls = LombScargle(set_elapsed_distance[j:j+sliding_window_size], set_fr[j:j+sliding_window_size])
        power = ls.power(frequency)
        powers.append(power.tolist())
        centre_distances.append(np.nanmean(set_elapsed_distance[j:j+sliding_window_size]))
    centre_trials = np.round(np.array(centre_distances)).astype(np.int64)
    powers = np.array(powers)
    avg_power = np.nanmean(powers, axis=0)

    first_half_powers = powers[int(len(powers)/2):, :]
    second_half_powers = powers[:int(len(powers)/2), :]

    avg_power_first_half = np.nanmean(first_half_powers, axis=0)
    avg_power_second_half = np.nanmean(second_half_powers, axis=0)

    # plot the avg periodgram
    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    for f in range(1,6):
        ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    ax.plot(frequency, avg_power, color="black", linewidth=3)
    ax.axhline(y=Settings.measured_far, color="red", linewidth=3, linestyle="dashed")
    allocentric_peak_freq, allocentric_peak_power, allo_i = get_allocentric_peak(frequency, avg_power, tolerance=0.05)
    egocentric_peak_freq, egocentric_peak_power, ego_i = get_egocentric_peak(frequency, avg_power, tolerance=0.05)
    ax.scatter(allocentric_peak_freq, allocentric_peak_power, color=Settings.allocentric_color, marker="v", s=200, zorder=10)
    ax.scatter(egocentric_peak_freq, egocentric_peak_power, color=Settings.egocentric_color, marker="v", s=200, zorder=10)
    ax.set_xlim(0,max(frequency))
    ax.set_ylim(0,0.5)
    plt.ylabel('Periodic Power', fontsize=30, labelpad = 10)
    plt.xlabel("Track Frequency", fontsize=30, labelpad = 10)
    plt.xlim(0,5.05)
    ax.set_xticks([0,5])
    ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #ax.set_yticks([0, 10, 20, 30])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #far = ls.false_alarm_level(1-(1.e-10))
    #ax.axhline(y=far, xmin=0, xmax=max(frequency), linestyle="dashed", color="red") # change method to "bootstrap" when you have time
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plot_path =  output_path + '/allo_and_egocentric_grid_cell_avg_lomb.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # plot the periodgram over time
    n_y_ticks = int(max(centre_trials)//50)+1
    y_tick_locs= np.linspace(np.ceil(min(centre_trials)), max(centre_trials), n_y_ticks, dtype=np.int64)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    powers[np.isnan(powers)] = 0
    Y, X = np.meshgrid(centre_trials, frequency)
    #powers = np.flip(powers, axis=0)
    cmap = plt.cm.get_cmap("inferno")
    c = ax.pcolormesh(X, Y, powers.T, cmap=cmap, shading="flat")
    for f in range(1,5):
        ax.axvline(x=f, color="white", linewidth=2,linestyle="dotted")

    x_pos = 4.8
    legend_freq = np.linspace(x_pos, x_pos+0.2, 5)
    rolling_lomb_classifier, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(centre_trials, powers)
    rolling_lomb_classifier_tiled = np.tile(rolling_lomb_classifier,(len(legend_freq),1))
    cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
    boundaries = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    Y, X = np.meshgrid(centre_trials, legend_freq)
    ax.pcolormesh(X, Y, rolling_lomb_classifier_tiled, cmap=cmap, norm=norm, shading="flat")



    plt.xlabel('Track Frequency', fontsize=30, labelpad = 10)
    plt.ylabel('Centre Trial', fontsize=30, labelpad = 10)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(y_tick_locs.tolist())
    ax.set_xlim([0.1,5])
    ax.set_ylim([min(centre_trials), max(centre_trials)])

    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0)
    cbar.set_label('Periodic Power', rotation=270, fontsize=20)
    cbar.set_ticks([np.min(powers), np.max(powers)])
    cbar.set_ticklabels(["0   ", "1   "])
    cbar.ax.tick_params(size=0)
    cbar.ax.tick_params(labelsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plot_path =  output_path + '/allo_and_egocentric_grid_cell_lomb_periodogram.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return

def main():
    print('-------------------------------------------------------------')
    output_path = "/mnt/datastore/Harry/VR_grid_cells/paper_figures/grid_schematics"
    plot_power_hypotheses_mfr(output_path=output_path)
    plot_power_hypotheses(jitter=0.1, output_path=output_path)
    plot_power_hypotheses(jitter=0.4, output_path=output_path)
    plot_zero_inflation_hypotheses(jitter=0.1, output_path=output_path)
    plot_allo_ego_grid_cell(output_path=output_path)
    print("look now")


if __name__ == '__main__':
    main()
