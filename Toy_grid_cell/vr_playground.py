import matplotlib.pyplot as plt
import numpy as np


def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array


def plot_linear_grid_cell_rates(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(track_lengths), len(offsets), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))

            firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
            firing_p = min_max_normlise(firing_p, 0, 1)
            firing_p = firing_p*p_scalar

            spikes_at_locations = np.zeros(len(locations))

            for i in range(len(locations)):
                spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]

            spike_locations = locations[spikes_at_locations==1]
            trial_numbers = (spike_locations//track_length)+1
            spike_locations = spike_locations%track_length

            rates = []
            for trial_number in np.arange(1, n_trials+1):
                trial_spike_locations = spike_locations[trial_numbers == trial_number]
                trial_rates, bin_edges = np.histogram(trial_spike_locations, bins=int(track_length/bin_size_cm), range=(0, track_length))
                bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

                rates.append(trial_rates.tolist())
            rates = np.array(rates)
            #rates = min_max_normalise_2d(rates, 0, 1)

            axes[m,n].imshow(rates, aspect="auto", cmap="cividis")

            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "P= "+str(offset)
                axes[m, n].set_ylabel(y_title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_non_anchored_rates_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'

    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_linear_grid_cell_rates_anchored(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(track_lengths), len(offsets), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
            trial_numbers = (locations//track_length)+1
            spikes_at_locations = []

            for trial_number in np.unique(trial_numbers):
                trial_locations = (locations%track_length)[trial_numbers==trial_number]
                firing_p = np.sin((2*np.pi*(1/grid_spacing)*trial_locations)+offset)
                firing_p = min_max_normlise(firing_p, 0, 1)
                firing_p = firing_p*p_scalar

                spikes_at_locations_trial = np.zeros(len(trial_locations))
                for i in range(len(spikes_at_locations_trial)):
                    spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
                spikes_at_locations.extend(spikes_at_locations_trial.tolist())

            spikes_at_locations = np.array(spikes_at_locations)

            spike_locations = locations[spikes_at_locations==1]
            trial_numbers = (spike_locations//track_length)+1
            spike_locations = spike_locations%track_length

            rates = []
            for trial_number in np.arange(1, n_trials+1):
                trial_spike_locations = spike_locations[trial_numbers == trial_number]
                trial_rates, bin_edges = np.histogram(trial_spike_locations, bins=int(track_length/bin_size_cm), range=(0, track_length))
                bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

                rates.append(trial_rates.tolist())
            rates = np.array(rates)
            #rates = min_max_normalise_2d(rates, 0, 1)

            axes[m,n].imshow(rates, aspect="auto", cmap="cividis")

            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "P= "+str(offset)
                axes[m, n].set_ylabel(y_title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_anchored_rates_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'

    plt.savefig(plot_path, dpi=300)
    plt.close()

def find_set(a,b):
    return set(a) & set(b)

def plot_linear_grid_cells_spatial_autocorreologram(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(track_lengths), len(offsets), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))

            firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
            firing_p = min_max_normlise(firing_p, 0, 1)
            firing_p = firing_p*p_scalar

            spikes_at_locations = np.zeros(len(locations))

            for i in range(len(locations)):
                spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]

            spike_locations_abs = locations[spikes_at_locations==1]
            trial_numbers = (spike_locations_abs//track_length)+1
            x_position_cluster = spike_locations_abs%track_length
            lap_distance_covered = (trial_numbers*track_length)-track_length #total elapsed distance
            x_position_cluster = x_position_cluster+lap_distance_covered
            x_position_cluster = x_position_cluster[~np.isnan(x_position_cluster)]
            x_position_cluster_bins = np.floor(x_position_cluster).astype(int)

            autocorr_window_size = int(2*track_length)
            lags = np.arange(0, autocorr_window_size, 1).astype(int) # were looking at 10 timesteps back and 10 forward

            autocorrelogram = np.array([])
            for lag in lags:
                correlated = len(find_set(x_position_cluster_bins+lag, x_position_cluster_bins))
                autocorrelogram = np.append(autocorrelogram, correlated)

            axes[m, n].bar(lags[1:], autocorrelogram[1:], color="black", edgecolor="black", align="edge")
            axes[m, n].set_xlim(1,autocorr_window_size)
            axes[m, n].set_ylim(min(autocorrelogram[1:]), max(autocorrelogram[1:]))

            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "P= "+str(offset)
                axes[m, n].set_ylabel(y_title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_non_anchored_spatial_autocorrelograms_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_linear_grid_cells_spatial_autocorreologram_anchored(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(track_lengths), len(offsets), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
            trial_numbers = (locations//track_length)+1
            spikes_at_locations = []

            for trial_number in np.unique(trial_numbers):
                trial_locations = (locations%track_length)[trial_numbers==trial_number]
                firing_p = np.sin((2*np.pi*(1/grid_spacing)*trial_locations)+offset)
                firing_p = min_max_normlise(firing_p, 0, 1)
                firing_p = firing_p*p_scalar

                spikes_at_locations_trial = np.zeros(len(trial_locations))
                for i in range(len(spikes_at_locations_trial)):
                    spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
                spikes_at_locations.extend(spikes_at_locations_trial.tolist())

            spikes_at_locations = np.array(spikes_at_locations)

            spike_locations_abs = locations[spikes_at_locations==1]
            trial_numbers = (spike_locations_abs//track_length)+1
            x_position_cluster = spike_locations_abs%track_length
            lap_distance_covered = (trial_numbers*track_length)-track_length #total elapsed distance
            x_position_cluster = x_position_cluster+lap_distance_covered
            x_position_cluster = x_position_cluster[~np.isnan(x_position_cluster)]
            x_position_cluster_bins = np.floor(x_position_cluster).astype(int)

            autocorr_window_size = int(2*track_length)
            lags = np.arange(0, autocorr_window_size, 1).astype(int) # were looking at 10 timesteps back and 10 forward

            autocorrelogram = np.array([])
            for lag in lags:
                correlated = len(find_set(x_position_cluster_bins+lag, x_position_cluster_bins))
                autocorrelogram = np.append(autocorrelogram, correlated)

            axes[m, n].bar(lags[1:], autocorrelogram[1:], color="black", edgecolor="black", align="edge")
            axes[m, n].set_xlim(1,autocorr_window_size)
            axes[m, n].set_ylim(min(autocorrelogram[1:]), max(autocorrelogram[1:]))

            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "P= "+str(offset)
                axes[m, n].set_ylabel(y_title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_anchored_spatial_autocorrelograms_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_linear_grid_cell(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(track_lengths), len(offsets), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))

            firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
            firing_p = min_max_normlise(firing_p, 0, 1)
            firing_p = firing_p*p_scalar

            spikes_at_locations = np.zeros(len(locations))

            for i in range(len(locations)):
                spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]

            spike_locations = locations[spikes_at_locations==1]
            trial_numbers = (spike_locations//track_length)+1
            spike_locations = spike_locations%track_length

            axes[m, n].scatter(spike_locations, trial_numbers, marker="|", color="black", alpha=0.025+((1-p_scalar)*0.025))
            axes[m, n].set_xlim(0,track_length)
            axes[m, n].set_ylim(0, n_trials)

            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "P= "+str(offset)
                axes[m, n].set_ylabel(y_title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_non_anchored_spikes_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'

    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_linear_grid_cell_anchored(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(track_lengths), len(offsets), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
            trial_numbers = (locations//track_length)+1
            spikes_at_locations = []

            for trial_number in np.unique(trial_numbers):
                trial_locations = (locations%track_length)[trial_numbers==trial_number]
                firing_p = np.sin((2*np.pi*(1/grid_spacing)*trial_locations)+offset)
                firing_p = min_max_normlise(firing_p, 0, 1)
                firing_p = firing_p*p_scalar

                spikes_at_locations_trial = np.zeros(len(trial_locations))
                for i in range(len(spikes_at_locations_trial)):
                    spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
                spikes_at_locations.extend(spikes_at_locations_trial.tolist())

            spikes_at_locations = np.array(spikes_at_locations)

            spike_locations = locations[spikes_at_locations==1]
            trial_numbers = (spike_locations//track_length)+1
            spike_locations = spike_locations%track_length

            axes[m, n].scatter(spike_locations, trial_numbers, marker="|", color="black", alpha=0.025+((1-p_scalar)*0.025))
            axes[m, n].set_xlim(0,track_length)
            axes[m, n].set_ylim(0, n_trials)

            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "P= "+str(offset)
                axes[m, n].set_ylabel(y_title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_anchored_spikes_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'

    plt.savefig(plot_path, dpi=300)
    plt.close()

def float2str(tmp):
    return "-".join(str(tmp).split("."))


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    save_path = "/mnt/datastore/Harry/Vr_grid_cells"


    for p_scalar in [1.0]:
        for n_trials in [80]:
            plot_linear_grid_cell_rates(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            plot_linear_grid_cells_spatial_autocorreologram(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            plot_linear_grid_cell(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)

    for p_scalar in [1.0]:
        for n_trials in [80]:
            plot_linear_grid_cell_anchored(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            plot_linear_grid_cell_rates_anchored(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            plot_linear_grid_cells_spatial_autocorreologram_anchored(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)

    print("look now")

if __name__ == '__main__':
    main()
