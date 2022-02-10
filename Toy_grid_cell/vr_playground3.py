import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Gaussian1DKernel
from scipy import stats
import Edmond.VR_grid_analysis.analysis_settings as Settings
from Edmond.VR_grid_analysis.vr_grid_cells import get_max_int_SNR, get_max_SNR, reduce_digits, get_first_peak
plt.rc('axes', linewidth=2)

def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

def plot_linear_grid_cell_rates_anchored(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
            trial_numbers = (locations//track_length)+1

            spikes_at_locations = []
            for trial_number in np.unique(trial_numbers):
                trial_locations = (locations%track_length)[trial_numbers==trial_number]
                firing_p = np.sin((2*np.pi*(1/grid_spacing)*trial_locations)+offset)
                firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
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

def plot_linear_grid_cell_rates_null(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length
            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
            firing_p = np.ones(len(locations))*0.5
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
    plot_path = save_path + '/toy_grid_assay_null_rates_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'

    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_linear_grid_cell_rates(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length
            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
            firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
            firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
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


def plot_linear_grid_cell_lomb_anchored(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):
    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):

            distance_covered = n_trials*track_length
            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps)+1)
            trial_numbers = (locations//track_length)+1
            spikes_at_locations = []
            for trial_number in np.unique(trial_numbers):
                trial_locations = (locations%track_length)[trial_numbers==trial_number]
                firing_p = np.sin((2*np.pi*(1/grid_spacing)*trial_locations)+offset)
                firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
                firing_p = min_max_normlise(firing_p, 0, 1)
                firing_p = firing_p*p_scalar
                spikes_at_locations_trial = np.zeros(len(trial_locations))
                for i in range(len(spikes_at_locations_trial)):
                    spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
                spikes_at_locations.extend(spikes_at_locations_trial.tolist())
            spikes_at_locations = np.array(spikes_at_locations)
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
            sliding_window_size=track_length*Settings.window_length_in_laps

            powers = []
            indices_to_test = np.arange(0, len(set_fr)-sliding_window_size, 1, dtype=np.int64)[::10]
            for j in indices_to_test:
                ls = LombScargle(set_elapsed_distance[j:j+sliding_window_size], set_fr[j:j+sliding_window_size])
                power = ls.power(frequency)
                powers.append(power.tolist())
            powers = np.array(powers)
            avg_power = np.nanmean(powers, axis=0)

            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
            max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
            max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))
            axes[m,n].plot(frequency, avg_power, color="blue")
            axes[m, n].set_xlim(0, max(frequency))
            axes[m, n].set_ylim(0,1)
            axes[m, n].text(0.9, 0.9, max_SNR_text, ha='right', va='center', transform=axes[m, n].transAxes, fontsize=4)
            axes[m, n].text(0.9, 0.8, max_SNR_freq_test, ha='right', va='center', transform=axes[m, n].transAxes, fontsize=4)
            far = ls.false_alarm_level(1-(1.e-10))
            axes[m, n].axhline(y=far, xmin=0, xmax=max(frequency), linestyle="dashed", color="red") # change method to "bootstrap" when you have time
            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "P= "+str(offset)
                axes[m, n].set_ylabel(y_title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_anchored_lomb_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_linear_grid_cell_lomb_null(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):
    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):

            distance_covered = n_trials*track_length
            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps)+1)
            firing_p = np.ones(len(locations))*0.5
            firing_p = firing_p*p_scalar
            spikes_at_locations = np.zeros(len(locations))
            for i in range(len(locations)):
                spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
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
            sliding_window_size=track_length*Settings.window_length_in_laps

            powers = []
            indices_to_test = np.arange(0, len(set_fr)-sliding_window_size, 1, dtype=np.int64)[::10]
            for j in indices_to_test:
                ls = LombScargle(set_elapsed_distance[j:j+sliding_window_size], set_fr[j:j+sliding_window_size])
                power = ls.power(frequency)
                powers.append(power.tolist())
            powers = np.array(powers)
            avg_power = np.nanmean(powers, axis=0)

            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
            max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
            max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))
            axes[m,n].plot(frequency, avg_power, color="blue")
            axes[m, n].set_xlim(0,max(frequency))
            axes[m, n].set_ylim(0,1)
            axes[m, n].text(0.9, 0.9, max_SNR_text, ha='right', va='center', transform=axes[m, n].transAxes, fontsize=4)
            axes[m, n].text(0.9, 0.8, max_SNR_freq_test, ha='right', va='center', transform=axes[m, n].transAxes, fontsize=4)
            far = ls.false_alarm_level(1-(1.e-10))
            axes[m, n].axhline(y=far, xmin=0, xmax=max(frequency), linestyle="dashed", color="red") # change method to "bootstrap" when you have time
            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "P= "+str(offset)
                axes[m, n].set_ylabel(y_title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_null_lomb_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_linear_grid_cell_lomb(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):
    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):

            distance_covered = n_trials*track_length
            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps)+1)
            firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
            firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
            firing_p = min_max_normlise(firing_p, 0, 1)
            firing_p = firing_p*p_scalar
            spikes_at_locations = np.zeros(len(locations))
            for i in range(len(locations)):
                spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
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
            sliding_window_size=track_length*Settings.window_length_in_laps

            powers = []
            indices_to_test = np.arange(0, len(set_fr)-sliding_window_size, 1, dtype=np.int64)[::10]
            for j in indices_to_test:
                ls = LombScargle(set_elapsed_distance[j:j+sliding_window_size], set_fr[j:j+sliding_window_size])
                power = ls.power(frequency)
                powers.append(power.tolist())
            powers = np.array(powers)
            avg_power = np.nanmean(powers, axis=0)

            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
            max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
            max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))
            axes[m,n].plot(frequency, avg_power, color="blue")
            axes[m, n].set_xlim(0,max(frequency))
            axes[m, n].set_ylim(0,1)
            axes[m, n].text(0.9, 0.9, max_SNR_text, ha='right', va='center', transform=axes[m, n].transAxes, fontsize=4)
            axes[m, n].text(0.9, 0.8, max_SNR_freq_test, ha='right', va='center', transform=axes[m, n].transAxes, fontsize=4)
            far = ls.false_alarm_level(1-(1.e-10))
            axes[m, n].axhline(y=far, xmin=0, xmax=max(frequency), linestyle="dashed", color="red") # change method to "bootstrap" when you have time
            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "P= "+str(offset)
                axes[m, n].set_ylabel(y_title, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_non_anchored_lomb_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

def get_sig_distances(power, distances, far):
    return distances[power>far]
    states = [power>far]
    true_indexes = np.where(states)[1]

    tmp_distances = [np.nan, np.nan]
    sig_distances = []

    index_changes = np.append(np.diff(true_indexes), 0)
    counter=0
    for i in true_indexes:
        if index_changes[counter] > 30:
            sig_distances.append(np.nanmean(tmp_distances))
            tmp_distances = [np.nan, np.nan]
        else:
            tmp_distances.append(distances[i])
        counter+=1

    return np.array(sig_distances)

def find_set(a,b):
    return set(a) & set(b)

def plot_linear_grid_cells_spatial_autocorreologram(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    #track_lengths = [100,110]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))

            firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
            firing_p = np.clip(firing_p, a_min=0.8, a_max=1)
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

            numerator, bin_edges = np.histogram(x_position_cluster, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))
            fr = numerator
            elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
            trial_numbers_by_bin=((0.5*(bin_edges[1:]+bin_edges[:-1])//track_length)+1).astype(np.int32)
            gauss_kernel = Gaussian1DKernel(stddev=1)

            # remove nan values that coincide with start and end of the track before convolution
            fr = convolve(fr, gauss_kernel)
            fr = moving_sum(fr, window=2)/2
            fr = np.append(fr, np.zeros(len(elapsed_distance)-len(fr)))
            normalised_elapsed_distance = elapsed_distance/track_length

            autocorr_window_size = track_length*2
            lags = np.arange(0, autocorr_window_size, 1) # were looking at 10 timesteps back and 10 forward
            autocorrelogram = []
            for i in range(len(lags)):
                fr_lagged = fr[i:]
                corr = stats.pearsonr(fr_lagged, fr[:len(fr_lagged)])[0]
                autocorrelogram.append(corr)
            autocorrelogram= np.array(autocorrelogram)

            for f in range(1,6):
                axes[m, n].axvline(x=track_length*f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
            axes[m, n].axhline(y=0, color="black", linewidth=2,linestyle="dashed")
            axes[m, n].plot(lags, autocorrelogram, color="blue")
            axes[m, n].set_xlim(1,autocorr_window_size)
            axes[m, n].set_ylim([-1,1])
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

def plot_linear_grid_cells_spatial_autocorreologram_null(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    first_peaks = np.zeros((len(offsets), len(track_lengths)))
    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))

            firing_p = np.ones(len(locations))*0.5
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

            numerator, bin_edges = np.histogram(x_position_cluster, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))
            fr = numerator
            elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
            trial_numbers_by_bin=((0.5*(bin_edges[1:]+bin_edges[:-1])//track_length)+1).astype(np.int32)
            gauss_kernel = Gaussian1DKernel(stddev=1)
            # remove nan values that coincide with start and end of the track before convolution
            fr = convolve(fr, gauss_kernel)
            fr = moving_sum(fr, window=2)/2
            fr = np.append(fr, np.zeros(len(elapsed_distance)-len(fr)))
            normalised_elapsed_distance = elapsed_distance/track_length
            autocorr_window_size = track_length*2
            lags = np.arange(0, autocorr_window_size, 1) # were looking at 10 timesteps back and 10 forward
            autocorrelogram = []
            for i in range(len(lags)):
                fr_lagged = fr[i:]
                corr = stats.pearsonr(fr_lagged, fr[:len(fr_lagged)])[0]
                autocorrelogram.append(corr)
            autocorrelogram= np.array(autocorrelogram)

            for f in range(1,6):
                axes[m, n].axvline(x=track_length*f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
            axes[m, n].axhline(y=0, color="black", linewidth=2,linestyle="dashed")
            axes[m, n].plot(lags, autocorrelogram, color="blue")
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
    plot_path = save_path + '/toy_grid_assay_null_spatial_autocorrelograms_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_linear_grid_cells_spatial_autocorreologram_anchored(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
            trial_numbers = (locations//track_length)+1

            spikes_at_locations = []
            for trial_number in np.unique(trial_numbers):
                trial_locations = (locations%track_length)[trial_numbers==trial_number]
                firing_p = np.sin((2*np.pi*(1/grid_spacing)*trial_locations)+offset)
                firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
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

            numerator, bin_edges = np.histogram(x_position_cluster, bins=int(track_length/1)*n_trials, range=(0, track_length*n_trials))
            fr = numerator
            elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
            trial_numbers_by_bin=((0.5*(bin_edges[1:]+bin_edges[:-1])//track_length)+1).astype(np.int32)
            gauss_kernel = Gaussian1DKernel(stddev=1)

            # remove nan values that coincide with start and end of the track before convolution
            fr = convolve(fr, gauss_kernel)
            fr = moving_sum(fr, window=2)/2
            fr = np.append(fr, np.zeros(len(elapsed_distance)-len(fr)))
            normalised_elapsed_distance = elapsed_distance/track_length

            autocorr_window_size = track_length*2
            lags = np.arange(0, autocorr_window_size, 1) # were looking at 10 timesteps back and 10 forward
            autocorrelogram = []
            for i in range(len(lags)):
                fr_lagged = fr[i:]
                corr = stats.pearsonr(fr_lagged, fr[:len(fr_lagged)])[0]
                autocorrelogram.append(corr)
            autocorrelogram= np.array(autocorrelogram)

            for f in range(1,6):
                axes[m, n].axvline(x=track_length*f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
            axes[m, n].axhline(y=0, color="black", linewidth=2,linestyle="dashed")
            axes[m, n].plot(lags, autocorrelogram, color="blue")
            axes[m, n].set_xlim(1,autocorr_window_size)
            axes[m, n].set_ylim([-1, 1])
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

    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))

            firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
            firing_p = np.clip(firing_p, a_min=0.8, a_max=1)
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


def plot_linear_grid_cell_null(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))

            firing_p = np.ones(len(locations))*0.5
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
    plot_path = save_path + '/toy_grid_assay_null_spikes_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'

    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_linear_grid_cell_anchored(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):

    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offsets = [10, 20, 30, 40, 50, 60]

    fig, axes = plt.subplots(len(offsets), len(track_lengths), figsize=(8, 6))

    for n, track_length in enumerate(track_lengths):
        for m, offset in enumerate(offsets):
            distance_covered = n_trials*track_length

            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
            trial_numbers = (locations//track_length)+1
            spikes_at_locations = []

            for trial_number in np.unique(trial_numbers):
                trial_locations = (locations%track_length)[trial_numbers==trial_number]
                firing_p = np.sin((2*np.pi*(1/grid_spacing)*trial_locations)+offset)
                firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
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
            axes[m, n].set_xlim(0, track_length)
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


def plot_lomb_power_vs_p_scalar_by_coding_scheme(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):
    track_length = 80
    grid_spacing = 30
    offset=0
    p_scalars = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    step = Settings.frequency_step
    frequency = Settings.frequency
    sliding_window_size=track_length*Settings.window_length_in_laps

    distance_max_peaks = []
    position_max_peaks = []
    null_max_peaks = []
    for m, p_scalar in enumerate(p_scalars):
        distance_covered = n_trials*track_length
        locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps)+1)

        # DISTANCE MODEL
        firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
        firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
        firing_p = min_max_normlise(firing_p, 0, 1)
        firing_p = firing_p*p_scalar
        spikes_at_locations = np.zeros(len(locations))
        for i in range(len(locations)):
            spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
        numerator, bin_edges = np.histogram(locations, bins=2*int(track_length/1)*n_trials, range=(0, track_length*n_trials), weights=spikes_at_locations)
        denominator, _ = np.histogram(locations, bins=2*int(track_length/1)*n_trials, range=(0, track_length*n_trials))
        set_fr = numerator/denominator
        set_elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
        gauss_kernel = Gaussian1DKernel(stddev=1)
        set_fr = convolve(set_fr, gauss_kernel)
        set_fr = moving_sum(set_fr, window=2)/2
        set_fr = np.append(set_fr, np.zeros(len(set_elapsed_distance)-len(set_fr)))
        powers = []
        indices_to_test = np.arange(0, len(set_fr)-sliding_window_size, 1, dtype=np.int64)[::10]
        for j in indices_to_test:
            ls = LombScargle(set_elapsed_distance[j:j+sliding_window_size], set_fr[j:j+sliding_window_size])
            power = ls.power(frequency)
            powers.append(power.tolist())
        powers = np.array(powers)
        avg_power = np.nanmean(powers, axis=0)
        max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
        distance_max_peaks.append(max_SNR)

        # POSITION MODEL
        trial_numbers = (locations//track_length)+1
        spikes_at_locations = []
        for trial_number in np.unique(trial_numbers):
            trial_locations = (locations%track_length)[trial_numbers==trial_number]
            firing_p = np.sin((2*np.pi*(1/grid_spacing)*trial_locations)+offset)
            firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
            firing_p = min_max_normlise(firing_p, 0, 1)
            firing_p = firing_p*p_scalar
            spikes_at_locations_trial = np.zeros(len(trial_locations))
            for i in range(len(spikes_at_locations_trial)):
                spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
            spikes_at_locations.extend(spikes_at_locations_trial.tolist())
        spikes_at_locations = np.array(spikes_at_locations)
        numerator, bin_edges = np.histogram(locations, bins=2*int(track_length/1)*n_trials, range=(0, track_length*n_trials), weights=spikes_at_locations)
        denominator, _ = np.histogram(locations, bins=2*int(track_length/1)*n_trials, range=(0, track_length*n_trials))
        set_fr = numerator/denominator
        set_elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
        gauss_kernel = Gaussian1DKernel(stddev=1)
        set_fr = convolve(set_fr, gauss_kernel)
        set_fr = moving_sum(set_fr, window=2)/2
        set_fr = np.append(set_fr, np.zeros(len(set_elapsed_distance)-len(set_fr)))
        powers = []
        indices_to_test = np.arange(0, len(set_fr)-sliding_window_size, 1, dtype=np.int64)[::10]
        for j in indices_to_test:
            ls = LombScargle(set_elapsed_distance[j:j+sliding_window_size], set_fr[j:j+sliding_window_size])
            power = ls.power(frequency)
            powers.append(power.tolist())
        powers = np.array(powers)
        avg_power = np.nanmean(powers, axis=0)
        max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
        position_max_peaks.append(max_SNR)

        # NULL MODEL
        firing_p = np.ones(len(locations))*0.5
        firing_p = firing_p*p_scalar
        spikes_at_locations = np.zeros(len(locations))
        for i in range(len(locations)):
            spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
        numerator, bin_edges = np.histogram(locations, bins=2*int(track_length/1)*n_trials, range=(0, track_length*n_trials), weights=spikes_at_locations)
        denominator, _ = np.histogram(locations, bins=2*int(track_length/1)*n_trials, range=(0, track_length*n_trials))
        set_fr = numerator/denominator
        set_elapsed_distance = 0.5*(bin_edges[1:]+bin_edges[:-1])/track_length
        gauss_kernel = Gaussian1DKernel(stddev=1)
        set_fr = convolve(set_fr, gauss_kernel)
        set_fr = moving_sum(set_fr, window=2)/2
        set_fr = np.append(set_fr, np.zeros(len(set_elapsed_distance)-len(set_fr)))
        powers = []
        indices_to_test = np.arange(0, len(set_fr)-sliding_window_size, 1, dtype=np.int64)[::10]
        for j in indices_to_test:
            ls = LombScargle(set_elapsed_distance[j:j+sliding_window_size], set_fr[j:j+sliding_window_size])
            power = ls.power(frequency)
            powers.append(power.tolist())
        powers = np.array(powers)
        avg_power = np.nanmean(powers, axis=0)
        max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
        null_max_peaks.append(max_SNR)


    fig, ax = plt.subplots(figsize=(8,8))
    distance_max_peaks = np.array(distance_max_peaks)
    position_max_peaks = np.array(position_max_peaks)
    null_max_peaks = np.array(null_max_peaks)

    ax.plot(p_scalars, null_max_peaks, "o-", color="gray", linewidth=2)
    ax.plot(p_scalars, distance_max_peaks, "o-", color="orange", linewidth=2)
    ax.plot(p_scalars, position_max_peaks, "o-", color="turquoise", linewidth=2)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_xlabel("Spike Probability Factor", fontsize=25, labelpad=10)
    ax.set_ylabel("Peak Power", fontsize=25, labelpad=10)
    ax.set_xlim([min(p_scalars), max(p_scalars)])
    ax.set_xscale('log')
    plt.tight_layout()
    plot_path = save_path+'/toy_grid_assay_p_scalar_assay.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return


def plot_rates_vs_p_scalar_by_coding_scheme(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1, PDN=""):
    track_length = 80
    grid_spacing = 30
    offset=0
    p_scalars = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    fig, axes = plt.subplots(1, len(p_scalars), figsize=(8, 1.75))
    for m, p_scalar in enumerate(p_scalars):
        distance_covered = n_trials*track_length
        locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps)+1)

        if PDN == "Distance":
            # DISTANCE MODEL
            firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
            firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
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
                rates.append(trial_rates.tolist())
            rates = np.array(rates)

        elif PDN == "Position":
            # POSITION MODEL
            trial_numbers = (locations//track_length)+1
            spikes_at_locations = []
            for trial_number in np.unique(trial_numbers):
                trial_locations = (locations%track_length)[trial_numbers==trial_number]
                firing_p = np.sin((2*np.pi*(1/grid_spacing)*trial_locations)+offset)
                firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
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
                rates.append(trial_rates.tolist())
            rates = np.array(rates)

        elif PDN == "Null":
            # NULL MODEL
            firing_p = np.ones(len(locations))*0.5
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
                rates.append(trial_rates.tolist())
            rates = np.array(rates)

        axes[m].imshow(rates, aspect="auto", cmap="cividis")
        x_title = "SPF="+str(reduce_digits(np.round(p_scalar, decimals=2), n_digits=6))
        axes[m].set_title(x_title, fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_p_scalar_assay_PDN-' + PDN + '_ntrials-' +str(n_trials) + '.png'

    plt.savefig(plot_path, dpi=300)
    plt.close()


def plot_lomb_power_vs_p_scalar(n_trials, save_path, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10, p_scalar=1):
    track_lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    grid_spacing = 30
    offset=0
    p_scalars = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    fig, axes = plt.subplots(len(p_scalars), len(track_lengths), figsize=(8, 8))

    for n, track_length in enumerate(track_lengths):
        for m, p_scalar in enumerate(p_scalars):

            distance_covered = n_trials*track_length
            locations = np.linspace(0, distance_covered-0.000001, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps)+1)
            firing_p = np.sin((2*np.pi*(1/grid_spacing)*locations)+offset)
            firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
            firing_p = min_max_normlise(firing_p, 0, 1)
            firing_p = firing_p*p_scalar
            spikes_at_locations = np.zeros(len(locations))
            for i in range(len(locations)):
                spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
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
            sliding_window_size=track_length*Settings.window_length_in_laps

            powers = []
            indices_to_test = np.arange(0, len(set_fr)-sliding_window_size, 1, dtype=np.int64)[::10]
            for j in indices_to_test:
                ls = LombScargle(set_elapsed_distance[j:j+sliding_window_size], set_fr[j:j+sliding_window_size])
                power = ls.power(frequency)
                powers.append(power.tolist())
            powers = np.array(powers)
            avg_power = np.nanmean(powers, axis=0)

            max_SNR, max_SNR_freq = get_max_SNR(frequency, avg_power)
            max_SNR_text = "SNR: " + reduce_digits(np.round(max_SNR, decimals=2), n_digits=6)
            max_SNR_freq_test = "Freq: " + str(np.round(max_SNR_freq, decimals=1))
            axes[m,n].plot(frequency, avg_power, color="blue")
            axes[m, n].set_xlim(0,max(frequency))
            axes[m, n].set_ylim(0,1)
            axes[m, n].text(0.9, 0.9, max_SNR_text, ha='right', va='center', transform=axes[m, n].transAxes, fontsize=4)
            axes[m, n].text(0.9, 0.8, max_SNR_freq_test, ha='right', va='center', transform=axes[m, n].transAxes, fontsize=4)
            far = ls.false_alarm_level(1-(1.e-10))
            axes[m, n].axhline(y=far, xmin=0, xmax=max(frequency), linestyle="dashed", color="red") # change method to "bootstrap" when you have time
            if m == 0:
                x_title = "L="+str(track_length)
                axes[m, n].set_title(x_title, fontsize=10)
            if n == 0:
                y_title = "PDF= "+str(reduce_digits(np.round(p_scalar, decimals=2), n_digits=6))
                axes[m, n].set_ylabel(y_title, fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plot_path = save_path + '/toy_grid_assay_non_anchored_p_scalar_assay_lomb_p_scalar-' + str(float2str(p_scalar)) + '_ntrials-' +str(n_trials) + '.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data"

    for p_scalar in [1.0]:
        for n_trials in [500]:
            plot_linear_grid_cell_rates(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            #plot_linear_grid_cells_spatial_autocorreologram(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            #plot_linear_grid_cell(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            plot_linear_grid_cell_lomb(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            print("")

    for p_scalar in [1.0]:
        for n_trials in [500]:
            plot_linear_grid_cell_anchored(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            #plot_linear_grid_cell_rates_anchored(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            plot_linear_grid_cells_spatial_autocorreologram_anchored(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            #plot_linear_grid_cell_lomb_anchored(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            print("")

    for p_scalar in [1.0]:
        for n_trials in [500]:
            #plot_linear_grid_cell_null(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            #plot_linear_grid_cell_rates_null(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            #plot_linear_grid_cells_spatial_autocorreologram_null(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            #plot_linear_grid_cell_lomb_null(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)
            print("")

    plot_lomb_power_vs_p_scalar(n_trials=n_trials, save_path=save_path)
    plot_lomb_power_vs_p_scalar_by_coding_scheme(n_trials=10, save_path=save_path)
    plot_rates_vs_p_scalar_by_coding_scheme(n_trials=10, save_path=save_path, PDN="Position")
    plot_rates_vs_p_scalar_by_coding_scheme(n_trials=10, save_path=save_path, PDN="Distance")
    plot_rates_vs_p_scalar_by_coding_scheme(n_trials=10, save_path=save_path, PDN="Null")
    print("look now")

if __name__ == '__main__':
    main()
