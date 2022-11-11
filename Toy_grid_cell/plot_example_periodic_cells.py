import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from astropy.timeseries import LombScargle
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.signal import find_peaks
from Edmond.VR_grid_analysis.FieldShuffleAnalysis.shuffle_analysis import fill_rate_map, make_field_array, \
    get_peak_indices, find_neighbouring_minima
from scipy import stats
import matplotlib.ticker as ticker
import Edmond.VR_grid_analysis.analysis_settings as Settings
import Edmond.plot_utility2
import scipy.interpolate as interp
plt.rc('axes', linewidth=3)
import warnings
warnings.filterwarnings('ignore')

def make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm):
    rates = []
    for trial_number in np.arange(1, n_trials+1):
        trial_spike_locations = spike_locations[spike_trial_numbers == trial_number]
        trial_rates, bin_edges = np.histogram(trial_spike_locations, bins=int(track_length/bin_size_cm), range=(0, track_length))
        rates.append(trial_rates.tolist())
    firing_rate_map_by_trial = np.array(rates)
    return firing_rate_map_by_trial

def getStableAllocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                                 p_scalar, track_length, field_spacing, step):

    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    trial_numbers = (locations//track_length)+1
    spikes_at_locations = []

    for trial_number in np.unique(trial_numbers):
        trial_locations = (locations%track_length)[trial_numbers==trial_number]
        firing_p = np.sin((2*np.pi*(1/field_spacing)*trial_locations))
        firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
        firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
        firing_p = firing_p*p_scalar
        spikes_at_locations_trial = np.zeros(len(trial_locations))
        for i in range(len(spikes_at_locations_trial)):
            spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
        spikes_at_locations.extend(spikes_at_locations_trial.tolist())
    spikes_at_locations = np.array(spikes_at_locations)
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial

def getUnstableAllocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                                   p_scalar, track_length, field_spacing, step):

    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    trial_numbers = (locations//track_length)+1
    spikes_at_locations = []

    for trial_number in np.unique(trial_numbers):
        trial_locations = (locations%track_length)[trial_numbers==trial_number]

        # add an offset for all trials
        offset = 0
        if trial_number%1 == 0:
            offset = np.random.normal(0, 1)
            #offset = np.random.randint(low=-field_spacing/4, high=field_spacing/4)

        firing_p = np.sin((2*np.pi*(1/field_spacing)*trial_locations)+offset)
        firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
        firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
        firing_p = firing_p*p_scalar
        spikes_at_locations_trial = np.zeros(len(trial_locations))
        for i in range(len(spikes_at_locations_trial)):
            spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
        spikes_at_locations.extend(spikes_at_locations_trial.tolist())
    spikes_at_locations = np.array(spikes_at_locations)
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial


def getStableEgocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                                p_scalar, track_length, field_spacing, step):
    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    firing_p = np.sin((2*np.pi*(1/field_spacing)*locations))
    firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
    firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
    firing_p = firing_p*p_scalar
    spikes_at_locations = np.zeros(len(locations))
    for i in range(len(locations)):
        spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial

def getUnstableEgocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                                  p_scalar, track_length, field_spacing, step):
    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    firing_p = np.sin((2*np.pi*(1/field_spacing)*locations))

    # make the sin waves happen at irregular times for half of the trials
    original_length_of_firing_p = len(firing_p)
    indices_in_trial = int(len(locations)/n_trials)
    # insert 20cm n times (n will be the number of trials/2)
    for i in range(int(n_trials/2)):
        random_i = np.random.randint(low=0, high=len(firing_p))
        firing_p = np.insert(firing_p, random_i, np.zeros(int(indices_in_trial/4)))
    firing_p = firing_p[:original_length_of_firing_p]

    firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
    firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
    firing_p = firing_p*p_scalar
    spikes_at_locations = np.zeros(len(locations))
    for i in range(len(locations)):
        spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial

def getPlaceCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                 p_scalar, track_length, step):
    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    trial_numbers = (locations//track_length)+1
    spikes_at_locations = []

    for trial_number in np.unique(trial_numbers):
        trial_locations = (locations%track_length)[trial_numbers==trial_number]
        firing_p = signal.gaussian(len(trial_locations), std=80)
        firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
        firing_p = firing_p*p_scalar
        spikes_at_locations_trial = np.zeros(len(trial_locations))
        for i in range(len(spikes_at_locations_trial)):
            spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
        spikes_at_locations.extend(spikes_at_locations_trial.tolist())
    spikes_at_locations = np.array(spikes_at_locations)
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial

def getRampCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step):
    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    trial_numbers = (locations//track_length)+1
    spikes_at_locations = []

    for trial_number in np.unique(trial_numbers):
        trial_locations = (locations%track_length)[trial_numbers==trial_number]
        firing_p = np.linspace(0, 1, len(trial_locations))
        firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
        firing_p = firing_p*p_scalar
        spikes_at_locations_trial = np.zeros(len(trial_locations))
        for i in range(len(spikes_at_locations_trial)):
            spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
        spikes_at_locations.extend(spikes_at_locations_trial.tolist())
    spikes_at_locations = np.array(spikes_at_locations)
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial

def getNoisyCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step):
    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    firing_p = 0.5*np.ones(len(locations))
    firing_p = firing_p*p_scalar
    spikes_at_locations = np.zeros(len(locations))
    for i in range(len(locations)):
        spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial

def getNoisyFieldCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step):
    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    firing_p = np.zeros(len(locations))
    n_fields = int((track_length/field_spacing)*n_trials)
    for i in range(n_fields):
        i = np.random.randint(low=0, high=len(locations)-track_length*10)
        firing_p[i: i+track_length*10] = signal.gaussian(track_length*10, std=80)
    firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
    firing_p = firing_p*p_scalar
    spikes_at_locations = np.zeros(len(locations))
    for i in range(len(locations)):
        spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial


def getShuffledPlaceCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step):
    _, _, firing_rate_map_by_trial = getPlaceCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step)
    _, field_shuffled_rate_map_smoothed, field_shuffled_rate_map = field_shuffle_and_get_false_alarm_rate(firing_rate_map_by_trial, p_threshold=0.99, n_shuffles=1)
    field_shuffled_rate_map_smoothed = field_shuffled_rate_map_smoothed[0]
    field_shuffled_rate_map = field_shuffled_rate_map[0]

    field_shuffled_rate_map_smoothed_flattened = field_shuffled_rate_map_smoothed.flatten()

    # remake firing from shuffled place cell rate map
    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))

    arr_interp = interp.interp1d(np.arange(field_shuffled_rate_map_smoothed_flattened.size),field_shuffled_rate_map_smoothed_flattened)
    firing_p = arr_interp(np.linspace(0,field_shuffled_rate_map_smoothed_flattened.size-1,locations.size))

    firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
    firing_p = firing_p*p_scalar
    spikes_at_locations = np.zeros(len(locations))
    for i in range(len(locations)):
        spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial

def plot_cell_spikes(cell_type, save_path, spikes_locations, spike_trial_numbers, firing_rate_map_by_trial):
    n_trials = len(firing_rate_map_by_trial)
    track_length = len(firing_rate_map_by_trial[0])

    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(1, 1, 1)
    ax.scatter(spikes_locations, spike_trial_numbers, marker="|", color="black", alpha=0.15)
    plt.ylabel('Trial number', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=20)
    plt.xlim(0,track_length)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    Edmond.plot_utility2.style_vr_plot(ax, n_trials)
    plt.tight_layout()
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/' + cell_type + 'spike_trajectory.png', dpi=200)
    plt.close()

def plot_cell_rates(cell_type, save_path, firing_rate_map_by_trial):
    n_trials = len(firing_rate_map_by_trial)
    track_length = len(firing_rate_map_by_trial[0])

    cluster_firing_maps = firing_rate_map_by_trial
    where_are_NaNs = np.isnan(cluster_firing_maps)
    cluster_firing_maps[where_are_NaNs] = 0
    cluster_firing_maps = Edmond.plot_utility2.min_max_normalize(cluster_firing_maps)
    percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
    vmin, vmax = Edmond.plot_utility2.get_vmin_vmax(cluster_firing_maps)

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(6, 6, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    locations = np.arange(0, len(cluster_firing_maps[0]))
    ordered = np.arange(1, n_trials+1, 1)
    X, Y = np.meshgrid(locations, ordered)
    cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
    c = ax.pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    plt.ylabel('Trial Number', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0, track_length)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_xlim([0, track_length])
    ax.set_ylim([0, n_trials-1])
    ax.set_yticks([1, 50, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    #cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    #cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
    #cbar.set_ticks([0,vmax])
    #cbar.set_ticklabels(["0", "Max"])
    #cbar.outline.set_visible(False)
    #cbar.ax.tick_params(labelsize=20)
    plt.savefig(save_path + '/'+cell_type+'_rate_map.png', dpi=300)
    plt.close()


    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(6, 2, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    locations = np.arange(0, len(cluster_firing_maps[0]))
    ax.fill_between(locations, np.nanmean(cluster_firing_maps, axis=0)-stats.sem(cluster_firing_maps, axis=0), np.nanmean(cluster_firing_maps, axis=0)+stats.sem(cluster_firing_maps, axis=0), color="black", alpha=0.3)
    ax.plot(locations, np.nanmean(cluster_firing_maps, axis=0), color="black", linewidth=3)
    plt.ylabel('FR (Hz)', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0, track_length)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_xlim([0, track_length])
    max_fr = max(np.nanmean(cluster_firing_maps, axis=0)+stats.sem(cluster_firing_maps, axis=0))
    max_fr = max_fr+(0.1*(max_fr))
    #ax.set_ylim([0, max_fr])

    ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
    ax.set_yticks([0, 1])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    #cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    #cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
    #cbar.set_ticks([0,vmax])
    #cbar.set_ticklabels(["0", "Max"])
    #cbar.outline.set_visible(False)
    #cbar.ax.tick_params(labelsize=20)
    plt.savefig(save_path + '/'+cell_type+'_avg_rate_map.png', dpi=300)
    plt.close()


def plot_field_shuffled_rate_map(cell_type, field_shuffled_rate_map, shuffled_save_path, plot_n_shuffles=10):
    for i in np.arange(plot_n_shuffles):
        n_trials = len(field_shuffled_rate_map[0])
        track_length = len(field_shuffled_rate_map[0][0])

        cluster_firing_maps = field_shuffled_rate_map[i]
        where_are_NaNs = np.isnan(cluster_firing_maps)
        cluster_firing_maps[where_are_NaNs] = 0
        cluster_firing_maps = Edmond.plot_utility2.min_max_normalize(cluster_firing_maps)
        percentile_99th = np.nanpercentile(cluster_firing_maps, 99); cluster_firing_maps = np.clip(cluster_firing_maps, a_min=0, a_max=percentile_99th)
        vmin, vmax = Edmond.plot_utility2.get_vmin_vmax(cluster_firing_maps)

        spikes_on_track = plt.figure()
        spikes_on_track.set_size_inches(6, 6, forward=True)
        ax = spikes_on_track.add_subplot(1, 1, 1)
        locations = np.arange(0, len(cluster_firing_maps[0]))
        ordered = np.arange(1, n_trials+1, 1)
        X, Y = np.meshgrid(locations, ordered)
        cmap = plt.cm.get_cmap(Settings.rate_map_cmap)
        c = ax.pcolormesh(X, Y, cluster_firing_maps, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        plt.ylabel('Trial Number', fontsize=25, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
        plt.xlim(0, track_length)
        ax.tick_params(axis='both', which='both', labelsize=20)
        ax.set_xlim([0, track_length])
        ax.set_ylim([0, n_trials-1])
        ax.set_yticks([1, 50, 100])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
        #cbar = spikes_on_track.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        #cbar.set_label('Firing Rate (Hz)', rotation=270, fontsize=20)
        #cbar.set_ticks([0,np.max(cluster_firing_maps)])
        #cbar.set_ticklabels(["0", "Max"])
        #cbar.ax.tick_params(labelsize=20)
        plt.savefig(shuffled_save_path + '/field_shuffled_'+cell_type+'_rate_map_'+str(i+1)+'.png', dpi=300)
        plt.close()


def plot_cell_spatial_autocorrelogram(cell_type, save_path, firing_rate_map_by_trial):
    fr=firing_rate_map_by_trial.flatten()
    track_length = len(firing_rate_map_by_trial[0])
    autocorr_window_size = track_length*4
    lags = np.arange(0, autocorr_window_size, 1)
    autocorrelogram = []
    for i in range(len(lags)):
        fr_lagged = fr[i:]
        corr = stats.pearsonr(fr_lagged, fr[:len(fr_lagged)])[0]
        autocorrelogram.append(corr)
    autocorrelogram= np.array(autocorrelogram)

    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(1, 1, 1)
    for f in range(1,6):
        ax.axvline(x=track_length*f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    ax.axhline(y=0, color="black", linewidth=2,linestyle="dashed")
    ax.plot(lags, autocorrelogram, color="black", linewidth=3)
    plt.ylabel('Spatial Autocorrelation', fontsize=25, labelpad = 10)
    plt.xlabel('Lag (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,(track_length*2)+3)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylim([np.floor(min(autocorrelogram[5:])*10)/10,np.ceil(max(autocorrelogram[5:])*10)/10])
    if np.floor(min(autocorrelogram[5:])*10)/10 < 0:
        ax.set_yticks([np.floor(min(autocorrelogram[5:])*10)/10, 0, np.ceil(max(autocorrelogram[5:])*10)/10])
    else:
        ax.set_yticks([-0.1, 0, np.ceil(max(autocorrelogram[5:])*10)/10])
    tick_spacing = track_length
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/' + cell_type + '_spatial_autocorrelogram.png', dpi=200)
    plt.close()

def plot_cell_avg_spatial_periodogram(cell_type, save_path, firing_rate_map_by_trial, far):
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

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    for f in range(1,6):
        ax.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    avg_subset_powers = np.nanmean(powers, axis=0)
    sem_subset_powers = stats.sem(powers, axis=0, nan_policy="omit")
    ax.fill_between(frequency, avg_subset_powers-sem_subset_powers, avg_subset_powers+sem_subset_powers, color="black", alpha=0.3)
    ax.plot(frequency, avg_subset_powers, color="black", linestyle="solid", linewidth=3)
    ax.axhline(y=far, color="red", linewidth=3, linestyle="dashed")
    plt.ylabel('Periodic Power', fontsize=25, labelpad = 10)
    plt.xlabel("Track Frequency", fontsize=25, labelpad = 10)
    plt.xlim(0,5.05)
    ax.set_xticks([0,5])
    ax.set_yticks([0, 1])
    ax.set_ylim(bottom=0, top=1)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/'+cell_type+'_avg_spatial_periodogram.png', dpi=300)
    plt.close()


def plot_cell_spatial_periodogram(cell_type, save_path, firing_rate_map_by_trial):
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

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    n_y_ticks = int(max(centre_trials)//50)+1
    y_tick_locs= np.linspace(np.ceil(min(centre_trials)), max(centre_trials), n_y_ticks, dtype=np.int64)
    powers[np.isnan(powers)] = 0
    Y, X = np.meshgrid(centre_trials, frequency)
    cmap = plt.cm.get_cmap("inferno")
    c = ax.pcolormesh(X, Y, powers.T, cmap=cmap, shading="flat")
    for f in range(1,6):
        ax.axvline(x=f, color="white", linewidth=2,linestyle="dotted")
    plt.xlabel('Track Frequency', fontsize=25, labelpad = 10)
    plt.ylabel('Centre Trial', fontsize=25, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks(y_tick_locs.tolist())
    ax.set_xlim([0.1,5])
    ax.set_ylim([min(centre_trials), max(centre_trials)])
    #cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    #cbar.set_label('Power', rotation=270, fontsize=20)
    #cbar.set_ticks([0,np.max(powers)])
    #cbar.set_ticklabels(["0", "Max"])
    #cbar.ax.tick_params(labelsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/'+cell_type+'_spatial_periodogram.png', dpi=300)
    plt.close()

def smoothen_rate_map(firing_rate_map_by_trial, n_trials, track_length, gauss_kernel_std):
    # smoothen and reshape
    gauss_kernel = Gaussian1DKernel(stddev=gauss_kernel_std)
    firing_rate_map_by_trial_flat = firing_rate_map_by_trial.flatten()
    firing_rate_map_by_trial_flat_smoothened = convolve(firing_rate_map_by_trial_flat, gauss_kernel)
    firing_rate_map_by_trial_smoothened = np.reshape(firing_rate_map_by_trial_flat_smoothened, (n_trials, track_length))
    return firing_rate_map_by_trial_smoothened


def field_shuffle_and_get_false_alarm_rate(firing_rate_map_by_trial, p_threshold, n_shuffles=1000,
                                           gauss_kernel_std=Settings.rate_map_gauss_kernel_std,
                                           extra_smooth_gauss_kernel_std=Settings.rate_map_extra_smooth_gauss_kernel_std,
                                           peak_min_distance=Settings.minimum_peak_distance):

    firing_rate_map_by_trial_flattened = firing_rate_map_by_trial.flatten()
    gauss_kernel_extra = Gaussian1DKernel(stddev=extra_smooth_gauss_kernel_std)
    gauss_kernel = Gaussian1DKernel(stddev=gauss_kernel_std)
    firing_rate_map_by_trial_flattened_extra_smooth = convolve(firing_rate_map_by_trial_flattened, gauss_kernel_extra)

    track_length = len(firing_rate_map_by_trial[0])
    n_trials = len(firing_rate_map_by_trial)
    elapsed_distance_bins = np.arange(0, (track_length*n_trials)+1, 1)
    elapsed_distance = 0.5*(elapsed_distance_bins[1:]+elapsed_distance_bins[:-1])/track_length
    frequency = Settings.frequency
    sliding_window_size=track_length*Settings.window_length_in_laps
    indices_to_test = np.arange(0, len(elapsed_distance)-sliding_window_size, 1, dtype=np.int64)[::Settings.power_estimate_step]

    # find peaks and trough indices
    peaks_i = find_peaks(firing_rate_map_by_trial_flattened_extra_smooth, distance=peak_min_distance)[0]
    peaks_indices = get_peak_indices(firing_rate_map_by_trial_flattened_extra_smooth, peaks_i)
    field_array = make_field_array(firing_rate_map_by_trial_flattened, peaks_indices)

    shuffle_peaks = []
    shuffle_rate_maps = []
    shuffle_rate_maps_smoothed = []
    for i in np.arange(n_shuffles):
        peak_fill_order = np.arange(1, len(peaks_i)+1)
        np.random.shuffle(peak_fill_order) # randomise fill order

        fr = fill_rate_map(firing_rate_map_by_trial, peaks_i, field_array, peak_fill_order)
        fr_smoothed = convolve(fr, gauss_kernel)

        powers = []
        for m in indices_to_test:
            ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr_smoothed[m:m+sliding_window_size])
            power = ls.power(frequency)
            powers.append(power.tolist())
        powers = np.array(powers)

        avg_powers = np.nanmean(powers, axis=0)
        shuffle_peak = np.nanmax(avg_powers)
        shuffle_peaks.append(shuffle_peak)
        shuffle_rate_maps.append(np.reshape(fr, (n_trials, track_length)))
        shuffle_rate_maps_smoothed.append(np.reshape(fr_smoothed, (n_trials, track_length)))

    shuffle_peaks = np.array(shuffle_peaks)
    return np.nanpercentile(shuffle_peaks, p_threshold*100), shuffle_rate_maps_smoothed, shuffle_rate_maps



def get_cluster_firing(cell_type_str, n_trials=100, bin_size_cm=1, sampling_rate=100, avg_speed_cmps=10,
                       p_scalar=1, track_length=200, field_spacing=90, gauss_kernel_std=2, step=0.000001):

    if cell_type_str == "stable_allocentric_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial = getStableAllocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "unstable_allocentric_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial = getUnstableAllocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "stable_egocentric_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial = getStableEgocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "unstable_egocentric_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial = getUnstableEgocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "noisy_field_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial = getNoisyFieldCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "shuffled_place_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial = getShuffledPlaceCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "place_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial = getPlaceCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step)
    elif cell_type_str == "ramp_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial = getRampCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step)
    elif cell_type_str == "noisy_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial = getNoisyCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step)

    firing_rate_map_by_trial_smoothed = smoothen_rate_map(firing_rate_map_by_trial, n_trials, track_length, gauss_kernel_std)

    return spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed


def plot_cell(cell_type, save_path, shuffled_save_path, n_trials=100, track_length=200):
    spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed = get_cluster_firing(cell_type_str=cell_type, n_trials=n_trials, track_length=track_length, gauss_kernel_std=2)

    # default plots
    plot_cell_spikes(cell_type, save_path, spikes_locations, spike_trial_numbers, firing_rate_map_by_trial_smoothed)
    plot_cell_rates(cell_type, save_path, firing_rate_map_by_trial_smoothed)
    plot_cell_spatial_autocorrelogram(cell_type, save_path, firing_rate_map_by_trial_smoothed)
    plot_cell_spatial_periodogram(cell_type, save_path, firing_rate_map_by_trial_smoothed)

    # plots require a field shuffle
    far, field_shuffled_rate_map_smoothed, field_shuffled_rate_map = field_shuffle_and_get_false_alarm_rate(firing_rate_map_by_trial, p_threshold=0.99, n_shuffles=1000)
    plot_cell_avg_spatial_periodogram(cell_type, save_path, firing_rate_map_by_trial_smoothed, far)
    plot_field_shuffled_rate_map(cell_type, field_shuffled_rate_map_smoothed, shuffled_save_path, plot_n_shuffles=10)
    print("plotted ", cell_type)

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    np.random.seed(0)

    save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data"
    shuffled_save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/shuffled"

    plot_cell(cell_type="shuffled_place_cell", save_path=save_path, shuffled_save_path=shuffled_save_path)
    plot_cell(cell_type="noisy_cell", save_path=save_path, shuffled_save_path=shuffled_save_path)
    plot_cell(cell_type="unstable_egocentric_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path)
    plot_cell(cell_type="stable_allocentric_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path)
    plot_cell(cell_type="unstable_allocentric_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path)
    plot_cell(cell_type="stable_egocentric_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path)
    plot_cell(cell_type="place_cell", save_path=save_path, shuffled_save_path=shuffled_save_path)
    plot_cell(cell_type="ramp_cell", save_path=save_path, shuffled_save_path=shuffled_save_path)
    plot_cell(cell_type="noisy_field_cell", save_path=save_path, shuffled_save_path=shuffled_save_path)

    print("look now")

if __name__ == '__main__':
    main()
