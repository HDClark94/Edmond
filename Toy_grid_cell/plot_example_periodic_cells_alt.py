import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from astropy.timeseries import LombScargle
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.signal import find_peaks
from Edmond.VR_grid_analysis.FieldShuffleAnalysis.shuffle_analysis import fill_rate_map, make_field_array, \
    get_peak_indices, find_neighbouring_minima
from scipy import stats
from matplotlib import colors
from Edmond.VR_grid_analysis.vr_grid_cells import get_rolling_lomb_classifier_for_centre_trial
import matplotlib.ticker as ticker
import Edmond.VR_grid_analysis.analysis_settings as Settings
import Edmond.plot_utility2
import scipy.interpolate as interp
from scipy.stats import norm

plt.rc('axes', linewidth=3)
import warnings
warnings.filterwarnings('ignore')

def get_avg_firing_rates(number_of_spikes, time_interval):
    return number_of_spikes/time_interval

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm):
    rates = []
    for trial_number in np.arange(1, n_trials+1):
        trial_spike_locations = spike_locations[spike_trial_numbers == trial_number]
        trial_rates, bin_edges = np.histogram(trial_spike_locations, bins=int(track_length/bin_size_cm), range=(0, track_length))
        rates.append(trial_rates.tolist())
    firing_rate_map_by_trial = np.array(rates)
    return firing_rate_map_by_trial

def switch_code(code):
    if code == "D":
        return "P"
    elif code == "P":
        return "D"

def getSwitchGridCellType2(grid_stability, n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                           p_scalar, track_length, field_spacing, step, field_noise_std=5):
    # this variant of the switch grid cell doesn't assume blocks of distance and position_encoding trials, instead
    # a random subset of trials are position and distance encoding

    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    trial_numbers = (locations//track_length)+1
    spikes_at_locations = []
    true_classifications = []

    modes = np.random.choice(["P", "D"], p=[0.5, 0.5], size=len(np.unique(trial_numbers)))
    # add spike locations for Position code
    for ti, trial_number in enumerate(np.unique(trial_numbers)):

        # choose allocentric or an egocentric mode for the trial
        mode = modes[ti]

        trial_locations = (locations%track_length)[trial_numbers==trial_number]
        # add an offset for all trials

        fields_to_insert = int((np.max(trial_locations)/field_spacing)+1)
        firing_p = np.zeros(len(trial_locations))
        offset = 0
        for i in range(fields_to_insert):
            if grid_stability == "imperfect":
                offset = np.random.normal(0, field_noise_std)
            firing_p += gaussian(x=trial_locations, mu=offset+(field_spacing*i), sig=field_spacing/10)
        #firing_p = np.sin((2*np.pi*(1/field_spacing)*trial_locations)+offset)
        #firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
        firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
        firing_p = firing_p*p_scalar
        spikes_at_locations_trial = np.zeros(len(trial_locations))
        for i in range(len(spikes_at_locations_trial)):
            spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
        spikes_at_locations.extend(spikes_at_locations_trial.tolist())
        true_classifications.append(mode)
    true_classifications = np.array(true_classifications)
    spikes_at_locations = np.array(spikes_at_locations)

    # now we add in the Distance coding trials
    fields_to_insert = int((np.max(locations)/field_spacing)+1)
    firing_p = np.zeros(len(locations))
    offset = 0
    for i in range(fields_to_insert):
        if grid_stability == "imperfect":
            previous_offset = offset
            offset = np.random.normal(0, field_noise_std)
        firing_p += gaussian(x=locations, mu=previous_offset+offset+(field_spacing*i), sig=field_spacing/10)
    firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
    firing_p = firing_p*p_scalar
    spikes_at_locations_D = np.zeros(len(locations))
    true_classifications_long = []
    for i in range(len(locations)):
        tn = trial_numbers[i]
        true_classifications_long.append(true_classifications[int(tn)-1])
        spikes_at_locations_D[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    true_classifications_long = np.array(true_classifications_long)

    # merge spikes_at_locations and spikes_at_locations_D
    spikes_at_locations[true_classifications_long=="D"] = spikes_at_locations_D[true_classifications_long=="D"]
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications


def getSwitchGridCell(grid_stability, n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                                   p_scalar, track_length, field_spacing, step, trial_switch_probability, field_noise_std=5):

    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    trial_numbers = (locations//track_length)+1
    spikes_at_locations = []
    true_classifications = []

    # choose allocentric or an egocentric mode to start with
    mode = np.random.choice(["P", "D"], p=[0.5, 0.5])
    modes = []
    for ti, trial_number in enumerate(np.unique(trial_numbers)):
        if np.random.random()<trial_switch_probability:
            mode = switch_code(mode)
        modes.append(mode)

    # add spike locations for Position code
    for ti, trial_number in enumerate(np.unique(trial_numbers)):
        mode = modes[ti]

        trial_locations = (locations%track_length)[trial_numbers==trial_number]
        # add an offset for all trials

        fields_to_insert = int((np.max(trial_locations)/field_spacing)+1)
        firing_p = np.zeros(len(trial_locations))
        offset = 0
        for i in range(fields_to_insert):
            if grid_stability == "imperfect":
                offset = np.random.normal(0, field_noise_std)
            firing_p += gaussian(x=trial_locations, mu=offset+(field_spacing*i), sig=field_spacing/10)
        #firing_p = np.sin((2*np.pi*(1/field_spacing)*trial_locations)+offset)
        #firing_p = np.clip(firing_p, a_min=-0.8, a_max=None)
        firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
        firing_p = firing_p*p_scalar
        spikes_at_locations_trial = np.zeros(len(trial_locations))
        for i in range(len(spikes_at_locations_trial)):
            spikes_at_locations_trial[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
        spikes_at_locations.extend(spikes_at_locations_trial.tolist())
        true_classifications.append(mode)
    true_classifications = np.array(true_classifications)
    spikes_at_locations = np.array(spikes_at_locations)

    # now we add in the Distance coding trials
    fields_to_insert = int((np.max(locations)/field_spacing)+1)
    firing_p = np.zeros(len(locations))
    offset = 0
    previous_offset=0
    for i in range(fields_to_insert):
        if grid_stability == "imperfect":
            previous_offset = offset
            offset = np.random.normal(0, field_noise_std)
        firing_p += gaussian(x=locations, mu=previous_offset+offset+(field_spacing*i), sig=field_spacing/10)
    firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
    firing_p = firing_p*p_scalar
    spikes_at_locations_D = np.zeros(len(locations))
    true_classifications_long = []
    for i in range(len(locations)):
        tn = trial_numbers[i]
        true_classifications_long.append(true_classifications[int(tn)-1])
        spikes_at_locations_D[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    true_classifications_long = np.array(true_classifications_long)

    # merge spikes_at_locations and spikes_at_locations_D
    spikes_at_locations[true_classifications_long=="D"] = spikes_at_locations_D[true_classifications_long=="D"]

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications


def getStableAllocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                                 p_scalar, track_length, field_spacing, step):

    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    trial_numbers = (locations//track_length)+1
    spikes_at_locations = []

    for trial_number in np.unique(trial_numbers):
        trial_locations = (locations%track_length)[trial_numbers==trial_number]

        fields_to_insert = int((np.max(trial_locations)/field_spacing)+1)
        firing_p = np.zeros(len(trial_locations))
        offset = 0
        for i in range(fields_to_insert):
            firing_p += gaussian(x=trial_locations, mu=offset+(field_spacing*i), sig=field_spacing/10)
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

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    true_classification=["P"]
    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classification

def getUnstableAllocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                                   p_scalar, track_length, field_spacing, step, field_noise_std):

    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    trial_numbers = (locations//track_length)+1
    spikes_at_locations = []

    for trial_number in np.unique(trial_numbers):
        trial_locations = (locations%track_length)[trial_numbers==trial_number]
        fields_to_insert = int((np.max(trial_locations)/field_spacing)+1)
        firing_p = np.zeros(len(trial_locations))
        offset = 0
        for i in range(fields_to_insert):
            offset = np.random.normal(0, field_noise_std)
            firing_p += gaussian(x=trial_locations, mu=offset+(field_spacing*i), sig=field_spacing/10)
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

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    true_classification=["P"]
    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classification


def getStableEgocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                                p_scalar, track_length, field_spacing, step):
    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    fields_to_insert = int((np.max(locations)/field_spacing)+1)
    firing_p = np.zeros(len(locations))
    offset = 0
    for i in range(fields_to_insert):
        firing_p += gaussian(x=locations, mu=offset+(field_spacing*i), sig=field_spacing/10)
    firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
    firing_p = firing_p*p_scalar
    spikes_at_locations = np.zeros(len(locations))
    for i in range(len(locations)):
        spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    true_classification=["D"]
    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classification

def getUnstableEgocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps,
                                  p_scalar, track_length, field_spacing, step, field_noise_std):
    distance_covered = n_trials*track_length
    locations = np.linspace(0, distance_covered-step, int(sampling_rate*(distance_covered/bin_size_cm)/avg_speed_cmps))
    fields_to_insert = int((np.max(locations)/field_spacing)+1)
    firing_p = np.zeros(len(locations))
    offset = 0
    for i in range(fields_to_insert):
        previous_offset=offset
        offset = np.random.normal(0, field_noise_std)
        firing_p += gaussian(x=locations, mu=previous_offset+offset+(field_spacing*i), sig=field_spacing/10)
    firing_p = Edmond.plot_utility2.min_max_normlise(firing_p, 0, 1)
    firing_p = firing_p*p_scalar
    spikes_at_locations = np.zeros(len(locations))
    for i in range(len(locations)):
        spikes_at_locations[i] = np.random.choice([1, 0], 1, p=[firing_p[i], 1-firing_p[i]])[0]
    spike_locations = locations[spikes_at_locations==1]
    spike_trial_numbers = (spike_locations//track_length)+1
    spike_locations = spike_locations%track_length

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    true_classification=["D"]
    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classification

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

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    true_classification=["P"]
    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classification

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

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    true_classification=["P"]
    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classification

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

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    true_classification=["N"]
    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classification

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

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    true_classification=["N"]
    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classification


def getShuffledPlaceCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step):
    _, _, firing_rate_map_by_trial, _ = getPlaceCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step)
    _, _,_,_, field_shuffled_rate_map_smoothed, field_shuffled_rate_map = field_shuffle_and_get_false_alarm_rate(firing_rate_map_by_trial, gamma=None, p_threshold=0.99, n_shuffles=1)
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

    avg_fr = get_avg_firing_rates(number_of_spikes=np.sum(spikes_at_locations), time_interval=track_length*n_trials/avg_speed_cmps)
    print(avg_fr)

    firing_rate_map_by_trial = make_rate_map(spike_locations, spike_trial_numbers, n_trials, track_length, bin_size_cm)

    true_classification=["N"]
    return spike_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classification

def plot_cell_spikes(cell_type, save_path, spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, plot_suffix):
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
    plt.savefig(save_path + '/' + cell_type + 'spike_trajectory'+plot_suffix+'.png', dpi=200)
    plt.close()

def plot_cell_rates(cell_type, save_path, firing_rate_map_by_trial, plot_suffix):
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
    plt.savefig(save_path + '/'+cell_type+'_rate_map'+plot_suffix+'.png', dpi=300)
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
    plt.savefig(save_path + '/'+cell_type+'_avg_rate_map'+plot_suffix+'.png', dpi=300)
    plt.close()


def plot_field_shuffled_rate_map(cell_type, field_shuffled_rate_map, shuffled_save_path, plot_n_shuffles, plot_suffix):
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
        plt.savefig(shuffled_save_path + '/field_shuffled_'+cell_type+'_rate_map_'+str(i+1)+''+plot_suffix+'.png', dpi=300)
        plt.close()


def plot_cell_spatial_autocorrelogram(cell_type, save_path, firing_rate_map_by_trial, plot_suffix):
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

    peaks_i = find_peaks(autocorrelogram, distance=5)[0]
    gamma=lags[peaks_i[0]]

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
    plt.savefig(save_path + '/' + cell_type + '_spatial_autocorrelogram'+plot_suffix+'.png', dpi=200)
    plt.close()
    return gamma

def plot_cell_avg_spatial_periodogram(cell_type, save_path, firing_rate_map_by_trial, far, plot_suffix):
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

    fig = plt.figure(figsize=(6,2))
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
    plt.savefig(save_path + '/'+cell_type+'_avg_spatial_periodogram'+plot_suffix+'.png', dpi=300)
    plt.close()

def get_numeric_classifation(classifications):
    numeric_classifications = []
    for i in range(len(classifications)):
        if classifications[i]== "P":
            numeric_classifications.append(0.5)
        elif classifications[i]== "D":
            numeric_classifications.append(1.5)
        elif classifications[i]== "N":
            numeric_classifications.append(2.5)
    return np.array(numeric_classifications)

def get_classifications_using_alt_method(firing_rate_map_by_trial, window_size_in_trials, rolling_far_pos, rolling_far_dis, gamma):
    track_length = len(firing_rate_map_by_trial[0])
    n_trials = len(firing_rate_map_by_trial)

    classifications = []
    classifications_numeric = []
    for tn in range(len(firing_rate_map_by_trial)):
        if tn<int(window_size_in_trials/2):
            fr_window = firing_rate_map_by_trial[:tn+int(window_size_in_trials/2),:]
        elif tn+int(window_size_in_trials/2)>len(firing_rate_map_by_trial):
            fr_window = firing_rate_map_by_trial[tn-int(window_size_in_trials/2):, :]
        else:
            fr_window = firing_rate_map_by_trial[tn-int(window_size_in_trials/2):tn+int(window_size_in_trials/2), :]

        fr=fr_window.flatten()
        gauss_kernel_extra = Gaussian1DKernel(stddev=Settings.rate_map_extra_smooth_gauss_kernel_std)
        gauss_kernel = Gaussian1DKernel(stddev=Settings.rate_map_gauss_kernel_std)
        fr_extra_smoothed = convolve(fr, gauss_kernel_extra)
        fr_smoothed = convolve(fr, gauss_kernel)

        peaks_i = find_peaks(fr_extra_smoothed, distance=Settings.minimum_peak_distance)[0]

        pos_scores = []
        dis_scores = []
        for i in peaks_i:
            if((i+track_length+gamma < len(fr_smoothed)-1) and (i+gamma+gamma < len(fr_smoothed)-1)):
                pos_scores.append(stats.pearsonr(x=fr_smoothed[i:i+gamma], y=fr_smoothed[i+track_length:i+track_length+gamma])[0])
                dis_scores.append(stats.pearsonr(x=fr_smoothed[i:i+gamma], y=fr_smoothed[i+gamma:i+gamma+gamma])[0])
        pos_score = np.nanmean(pos_scores)
        dis_score = np.nanmean(dis_scores)

        if pos_score > dis_score:
            if pos_score > rolling_far_pos:
                trial_classification = "P"
                trial_classification_numeric = 0.5
            else:
                trial_classification = "N"
                trial_classification_numeric = 2.5
        elif dis_score > pos_score:
            if dis_score > rolling_far_dis:
                trial_classification = "D"
                trial_classification_numeric = 1.5
            else:
                trial_classification = "N"
                trial_classification_numeric = 2.5
        else:
            trial_classification = "U"
            trial_classification_numeric = 3.5

        classifications.append(trial_classification)
        classifications_numeric.append(trial_classification_numeric)

    return classifications, classifications_numeric

def plot_rolling_classification_vs_true_classification(cell_type, save_path, firing_rate_map_by_trial, gamma, true_classifications, rolling_far_pos=None, rolling_far_dis=None, plot_suffix=""):
    window_size_in_trials = 10
    classifications, classifications_numeric = get_classifications_using_alt_method(firing_rate_map_by_trial=firing_rate_map_by_trial, window_size_in_trials=window_size_in_trials,
                                                           rolling_far_pos=rolling_far_pos, rolling_far_dis=rolling_far_dis, gamma=gamma)

    fig = plt.figure(figsize=(2,6))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    x_pos = 0.3
    legend_freq = np.linspace(x_pos, x_pos+0.2, 5)
    rolling_classifications_tiled = np.tile(classifications_numeric,(len(legend_freq),1))
    cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
    boundaries = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    Y, X = np.meshgrid(np.arange(1,len(firing_rate_map_by_trial)+1), legend_freq)
    ax.pcolormesh(X, Y, rolling_classifications_tiled, cmap=cmap, norm=norm, shading="flat", zorder=2)

    x_pos = 0
    legend_freq = np.linspace(x_pos, x_pos+0.2, 5)
    if len(true_classifications)==1:
        true_classifications = np.repeat(true_classifications[0], len(firing_rate_map_by_trial))
    true_classifications_numeric = get_numeric_classifation(true_classifications)
    true_classifications_tiled = np.tile(true_classifications_numeric, (len(legend_freq),1))
    cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
    boundaries = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    Y, X = np.meshgrid(np.arange(1,len(firing_rate_map_by_trial)+1), legend_freq)
    ax.pcolormesh(X, Y, true_classifications_tiled, cmap=cmap, norm=norm, shading="flat", zorder=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([0.1, 0.4])
    ax.set_xticklabels(["T", "P"])
    ax.set_yticks([])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/'+cell_type+'_rolling_classification_vs_true_classsificiation_'+plot_suffix+'.png', dpi=300)
    plt.close()


def plot_cell_spatial_periodogram(cell_type, save_path, firing_rate_map_by_trial, rolling_far=None, rolling_window_size_for_lomb_classifier=200, plot_suffix=""):
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

    if rolling_far is not None:
        x_pos = 4.8
        legend_freq = np.linspace(x_pos, x_pos+0.2, 5)
        rolling_lomb_classifier, rolling_lomb_classifier_numeric, rolling_lomb_classifier_colors, rolling_frequencies, rolling_points = get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers, power_threshold=rolling_far, power_step=Settings.power_estimate_step, track_length=track_length, n_window_size=rolling_window_size_for_lomb_classifier)
        rolling_lomb_classifier_tiled = np.tile(rolling_lomb_classifier_numeric,(len(legend_freq),1))
        cmap = colors.ListedColormap([Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, 'black'])
        boundaries = [0, 1, 2, 3, 4]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        Y, X = np.meshgrid(centre_trials, legend_freq)
        ax.pcolormesh(X, Y, rolling_lomb_classifier_tiled, cmap=cmap, norm=norm, shading="flat", zorder=2)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/'+cell_type+'_spatial_periodogram'+plot_suffix+'.png', dpi=300)
    plt.close()

def smoothen_rate_map(firing_rate_map_by_trial, n_trials, track_length, gauss_kernel_std):
    # smoothen and reshape
    gauss_kernel = Gaussian1DKernel(stddev=gauss_kernel_std)
    firing_rate_map_by_trial_flat = firing_rate_map_by_trial.flatten()
    firing_rate_map_by_trial_flat_smoothened = convolve(firing_rate_map_by_trial_flat, gauss_kernel)
    firing_rate_map_by_trial_smoothened = np.reshape(firing_rate_map_by_trial_flat_smoothened, (n_trials, track_length))
    return firing_rate_map_by_trial_smoothened


def field_shuffle_and_get_false_alarm_rate(firing_rate_map_by_trial, gamma, p_threshold, n_shuffles=1000,
                                           gauss_kernel_std=Settings.rate_map_gauss_kernel_std,
                                           extra_smooth_gauss_kernel_std=Settings.rate_map_extra_smooth_gauss_kernel_std,
                                           peak_min_distance=Settings.minimum_peak_distance,
                                           rolling_window_size=Settings.rolling_window_size_for_lomb_classifier):

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

    shuffle_pos_scores = []
    shuffle_dis_scores = []
    shuffle_pos_scores_first_window = []
    shuffle_dis_scores_first_window = []
    shuffle_rate_maps = []
    shuffle_rate_maps_smoothed = []
    for j in np.arange(n_shuffles):
        peak_fill_order = np.arange(1, len(peaks_i)+1)
        np.random.shuffle(peak_fill_order) # randomise fill order

        fr = fill_rate_map(firing_rate_map_by_trial, peaks_i, field_array, peak_fill_order)
        fr_smoothed = convolve(fr, gauss_kernel)

        if gamma is not None:
            pos_scores = []
            dis_scores = []
            pos_scores_first_window = []
            dis_scores_first_window = []
            for i in peaks_i:
                if((i+track_length+gamma < len(fr_smoothed)-1) and (i+gamma+gamma < len(fr_smoothed)-1)):
                    pos_scores.append(stats.pearsonr(x=fr_smoothed[i:i+gamma], y=fr_smoothed[i+track_length:i+track_length+gamma])[0])
                    dis_scores.append(stats.pearsonr(x=fr_smoothed[i:i+gamma], y=fr_smoothed[i+gamma:i+gamma+gamma])[0])

                    if ((i+track_length+gamma < ((track_length*10)-1)) or (i+gamma+gamma < ((track_length*10)-1))):
                        pos_scores_first_window.append(pos_scores)
                        dis_scores_first_window.append(dis_scores)

            shuffle_pos_scores.append(np.nanmean(pos_scores))
            shuffle_dis_scores.append(np.nanmean(dis_scores))
            shuffle_pos_scores_first_window.append(np.nanmean(pos_scores_first_window))
            shuffle_dis_scores_first_window.append(np.nanmean(dis_scores_first_window))

        shuffle_rate_maps.append(np.reshape(fr, (n_trials, track_length)))
        shuffle_rate_maps_smoothed.append(np.reshape(fr_smoothed, (n_trials, track_length)))

    shuffle_pos_scores = np.array(shuffle_pos_scores)
    shuffle_dis_scores = np.array(shuffle_dis_scores)
    shuffle_pos_scores_first_window = np.array(shuffle_pos_scores_first_window)
    shuffle_dis_scores_first_window = np.array(shuffle_dis_scores_first_window)

    return np.nanpercentile(shuffle_pos_scores, p_threshold*100), np.nanpercentile(shuffle_dis_scores, p_threshold*100), \
           np.nanpercentile(shuffle_pos_scores_first_window, p_threshold*100), np.nanpercentile(shuffle_dis_scores_first_window, p_threshold*100),\
           shuffle_rate_maps_smoothed, shuffle_rate_maps

def get_switch_cluster_firing(switch_coding_mode, grid_stability, n_trials=Settings.sim_n_trials, bin_size_cm=Settings.sim_bin_size_cm, sampling_rate=Settings.sim_sampling_rate, avg_speed_cmps=Settings.sim_avg_speed_cmps,
                              p_scalar=Settings.sim_p_scalar, track_length=Settings.sim_track_length, field_spacing=Settings.sim_field_spacing, gauss_kernel_std=Settings.sim_gauss_kernel_std, step=Settings.sim_step, trial_switch_probability=None, field_noise_std=Settings.sim_field_noise_std):

    if switch_coding_mode == "block":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getSwitchGridCell(grid_stability, n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step, trial_switch_probability, field_noise_std)
    elif switch_coding_mode == "by_trial":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getSwitchGridCellType2(grid_stability, n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step, field_noise_std)

    firing_rate_map_by_trial_smoothed = smoothen_rate_map(firing_rate_map_by_trial, n_trials, track_length, gauss_kernel_std)

    return spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed, true_classifications


def get_cluster_firing(cell_type_str, n_trials=Settings.sim_n_trials, bin_size_cm=Settings.sim_bin_size_cm, sampling_rate=Settings.sim_sampling_rate, avg_speed_cmps=Settings.sim_avg_speed_cmps,
                       p_scalar=Settings.sim_p_scalar, track_length=Settings.sim_track_length, field_spacing=Settings.sim_field_spacing, gauss_kernel_std=Settings.sim_gauss_kernel_std, step=Settings.sim_step, field_noise_std=Settings.sim_field_noise_std, switch_code_prob=0.05):

    if cell_type_str == "stable_allocentric_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getStableAllocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "unstable_allocentric_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getUnstableAllocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step, field_noise_std)
    elif cell_type_str == "stable_egocentric_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getStableEgocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "unstable_egocentric_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getUnstableEgocentricGridCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step, field_noise_std)
    elif cell_type_str == "noisy_field_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getNoisyFieldCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "shuffled_place_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getShuffledPlaceCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step)
    elif cell_type_str == "place_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getPlaceCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step)
    elif cell_type_str == "ramp_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getRampCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step)
    elif cell_type_str == "noisy_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getNoisyCell(n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, step)
    elif cell_type_str == "stable_switch_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getSwitchGridCell("perfect", n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step, switch_code_prob, field_noise_std)
    elif cell_type_str == "unstable_switch_grid_cell":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getSwitchGridCell("imperfect", n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step, switch_code_prob, field_noise_std)
    elif cell_type_str == "unstable_switch_grid_cell_type2":
        spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, true_classifications = getSwitchGridCellType2("imperfect", n_trials, bin_size_cm, sampling_rate, avg_speed_cmps, p_scalar, track_length, field_spacing, step, field_noise_std)

    firing_rate_map_by_trial_smoothed = smoothen_rate_map(firing_rate_map_by_trial, n_trials, track_length, gauss_kernel_std)

    return spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed, true_classifications


def plot_cell(cell_type, save_path, shuffled_save_path, n_trials=100, track_length=200, rolling_far=None, field_spacing=90, field_noise_std=5, rolling_window_size_for_lomb_classifier=Settings.rolling_window_size_for_lomb_classifier, switch_code_prob=0.05, plot_suffix=""):
    spikes_locations, spike_trial_numbers, firing_rate_map_by_trial, firing_rate_map_by_trial_smoothed, true_classifications = get_cluster_firing(cell_type_str=cell_type, n_trials=n_trials,track_length=track_length,
                                                                                                                                                  gauss_kernel_std=2,field_spacing=field_spacing, field_noise_std=field_noise_std,switch_code_prob=switch_code_prob)

    # default plots
    plot_cell_spikes(cell_type, save_path, spikes_locations, spike_trial_numbers, firing_rate_map_by_trial_smoothed, plot_suffix=plot_suffix)
    plot_cell_rates(cell_type, save_path, firing_rate_map_by_trial_smoothed, plot_suffix=plot_suffix)
    gamma = plot_cell_spatial_autocorrelogram(cell_type, save_path, firing_rate_map_by_trial_smoothed, plot_suffix=plot_suffix)
    plot_cell_spatial_periodogram(cell_type, save_path, firing_rate_map_by_trial_smoothed, rolling_far=None,
                                  rolling_window_size_for_lomb_classifier=rolling_window_size_for_lomb_classifier, plot_suffix=plot_suffix) # requires field shuffle IF the rolling classification is wanted

    # plots require a field shuffle
    far_pos, far_dis, rolling_far_pos, rolling_far_dis, field_shuffled_rate_map_smoothed, field_shuffled_rate_map = field_shuffle_and_get_false_alarm_rate(firing_rate_map_by_trial, gamma, p_threshold=0.99, n_shuffles=100)
    #plot_cell_avg_spatial_periodogram(cell_type, save_path, firing_rate_map_by_trial_smoothed, far, plot_suffix=plot_suffix)
    #plot_field_shuffled_rate_map(cell_type, field_shuffled_rate_map_smoothed, shuffled_save_path, plot_n_shuffles=10, plot_suffix=plot_suffix)

    plot_rolling_classification_vs_true_classification(cell_type, save_path, firing_rate_map_by_trial, gamma, true_classifications=true_classifications,
                                                       rolling_far_pos=rolling_far_pos, rolling_far_dis=rolling_far_dis, plot_suffix=plot_suffix) # requires field shuffle IF the rolling classification is wanted
    print("plotted ", cell_type)

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    np.random.seed(0)
    lomb_window_size_rolling = 200

    save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/alt"
    shuffled_save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/alt/shuffled"

    # supp
    np.random.seed(0)
    plot_cell(cell_type="unstable_egocentric_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, field_noise_std=10, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling, plot_suffix="_field_noise=10");np.random.seed(0)
    plot_cell(cell_type="unstable_egocentric_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, field_noise_std=0, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling, plot_suffix="_field_noise=0");np.random.seed(0)
    plot_cell(cell_type="unstable_allocentric_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, field_noise_std=10, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling, plot_suffix="_field_noise=10");np.random.seed(0)
    plot_cell(cell_type="unstable_allocentric_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, field_noise_std=0, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling, plot_suffix="_field_noise=0");np.random.seed(0)
    plot_cell(cell_type="shuffled_place_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling)
    plot_cell(cell_type="noisy_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling)
    plot_cell(cell_type="place_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling)
    plot_cell(cell_type="ramp_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling)
    #plot_cell(cell_type="noisy_field_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling)

    # another supp
    plot_cell(cell_type="unstable_switch_grid_cell_type2", save_path=save_path, shuffled_save_path=shuffled_save_path, field_noise_std=10, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling, plot_suffix="_field_noise=10");np.random.seed(0)
    plot_cell(cell_type="unstable_switch_grid_cell_type2", save_path=save_path, shuffled_save_path=shuffled_save_path, field_noise_std=0, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling, plot_suffix="_field_noise=0");np.random.seed(0)
    plot_cell(cell_type="unstable_switch_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, field_noise_std=10, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling, plot_suffix="_field_noise=10");np.random.seed(0)
    plot_cell(cell_type="unstable_switch_grid_cell", save_path=save_path, shuffled_save_path=shuffled_save_path, field_noise_std=0, rolling_window_size_for_lomb_classifier=lomb_window_size_rolling, plot_suffix="_field_noise=0");np.random.seed(0)
    print("look now")

if __name__ == '__main__':
    main()
