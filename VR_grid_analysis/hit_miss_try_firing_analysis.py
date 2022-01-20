import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import settings
from numpy import inf
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from Edmond.Concatenate_from_server import *
from scipy import stats
import seaborn as sns
from matplotlib.markers import TICKDOWN
import matplotlib as mpl
plt.rc('axes', linewidth=3)

def get_p_text(p, ns=True):

    if p is not None:
        if np.isnan(p):
            return " "
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

def distance_from_integer(frequencies):
    distance_from_zero = np.asarray(frequencies)%1
    distance_from_one = 1-(np.asarray(frequencies)%1)
    tmp = np.vstack((distance_from_zero, distance_from_one))
    return np.min(tmp, axis=0)

def add_lomb_classifier(spatial_firing, suffix=""):
    """
    :param spatial_firing:
    :param suffix: specific set string for subsets of results
    :return: spatial_firing with classifier collumn of type ["Lomb_classifer_"+suffix] with either "Distance", "Position" or "Null"
    """
    lomb_classifiers = []
    for index, row in spatial_firing.iterrows():
        if "ML_SNRs"+suffix in list(spatial_firing):
            lomb_SNR = row["ML_SNRs"+suffix]
            lomb_freq = row["ML_Freqs"+suffix]
            lomb_distance_from_int = distance_from_integer(lomb_freq)[0]

            if lomb_SNR>0.05:
                if lomb_distance_from_int<0.025:
                    lomb_classifier = "Position"
                else:
                    lomb_classifier = "Distance"
            elif lomb_distance_from_int<0.025:
                lomb_classifier = "Position"
            else:
                lomb_classifier = "Null"
        else:
            lomb_classifier = "Unclassifed"

        lomb_classifiers.append(lomb_classifier)

    spatial_firing["Lomb_classifier_"+suffix] = lomb_classifiers
    return spatial_firing

def get_spatial_information_score(cluster_spike_data, position_data, processed_position_data,
                                  hit_miss_try, trial_type, track_length):

    # filter the processed_position_data based on hit miss try and the trial type
    if hit_miss_try == "all":
        processed_position_data = processed_position_data
    else:
        processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hit_miss_try)]

    if trial_type == "all":
        processed_position_data = processed_position_data
    else:
        processed_position_data = processed_position_data[(processed_position_data["trial_type"] == trial_type)]

    # compute the spatial information score if any trials qualify
    if len(processed_position_data) == 0:
        spatial_information_score = np.nan
        return spatial_information_score
    else:
        # filter data by the trial numbers now in processed_position_data
        trial_numbers = np.unique(processed_position_data["trial_number"]).tolist()
        position_data = position_data.loc[position_data["trial_number"].isin(trial_numbers)]
        time_seconds = np.array(position_data['time_in_bin_seconds'].to_numpy())
        x_position_cm = np.array(position_data['x_position_cm'].to_numpy())
        spike_trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])
        spike_x_positions = np.array(cluster_spike_data["x_position_cm"].iloc[0])
        trial_mask = np.isin(spike_trial_numbers, trial_numbers)
        spike_x_positions = spike_x_positions[trial_mask]

        # compute the firing rate map and occupancy map
        spike_heatmap, bin_edges = np.histogram(spike_x_positions, bins=int(track_length/settings.vr_grid_analysis_bin_size), range=(0, track_length))
        position_heatmap, bin_edges = np.histogram(x_position_cm, bins=int(track_length/settings.vr_grid_analysis_bin_size), range=(0, track_length), weights=time_seconds)
        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
        firing_rate_map = spike_heatmap/position_heatmap
        occupancy_probability_map = position_heatmap/np.sum(position_heatmap)

        # compute the mean firing rate
        n_spikes = len(spike_x_positions)
        time_elapsed = np.sum(time_seconds)
        mean_firing_rate = n_spikes/time_elapsed

        # compute the spatial information score
        log_term = np.log2(firing_rate_map/mean_firing_rate)
        log_term[log_term == -inf] = 0
        Isec = np.sum(occupancy_probability_map*firing_rate_map*log_term)
        spatial_information_score = Isec/mean_firing_rate
        return spatial_information_score

def process_spatial_information(spike_data, position_data, processed_position_data, track_length):
    print('adding spatial information scores...')

    hit_beaconed = []
    hit_non_beaconed = []
    hit_probe = []
    hit_all = []

    try_beaconed = []
    try_non_beaconed = []
    try_probe = []
    try_all = []

    miss_beaconed = []
    miss_non_beaconed = []
    miss_probe = []
    miss_all =[]

    all_beaconed = []
    all_non_beaconed = []
    all_probe = []
    all__all =[]

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        cluster_firing_maps = np.array(cluster_spike_data["firing_rate_maps"].iloc[0])
        where_are_NaNs = np.isnan(cluster_firing_maps)
        cluster_firing_maps[where_are_NaNs] = 0
        if len(cluster_firing_maps) == 0:
            print("stop here")

        hit_beaconed.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="hit", trial_type=0, track_length=track_length))
        hit_non_beaconed.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="hit", trial_type=1, track_length=track_length))
        hit_probe.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="hit", trial_type=2, track_length=track_length))
        hit_all.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="hit", trial_type="all", track_length=track_length))
        try_beaconed.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="try", trial_type=0, track_length=track_length))
        try_non_beaconed.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="try", trial_type=1, track_length=track_length))
        try_probe.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="try", trial_type=2, track_length=track_length))
        try_all.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="try", trial_type="all", track_length=track_length))
        miss_beaconed.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="miss", trial_type=0, track_length=track_length))
        miss_non_beaconed.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="miss", trial_type=1, track_length=track_length))
        miss_probe.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="miss", trial_type=2, track_length=track_length))
        miss_all.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="miss", trial_type="all", track_length=track_length))
        all_beaconed.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="all", trial_type=0, track_length=track_length))
        all_non_beaconed.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="all", trial_type=1, track_length=track_length))
        all_probe.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="all", trial_type=2, track_length=track_length))
        all__all.append(get_spatial_information_score(cluster_spike_data, position_data, processed_position_data, hit_miss_try="all", trial_type="all", track_length=track_length))

    spike_data["hmt_hit_tt_beaconed"] = hit_beaconed
    spike_data["hmt_hit_tt_non_beaconed"] = hit_non_beaconed
    spike_data["hmt_hit_tt_probe"] = hit_probe
    spike_data["hmt_hit_tt_all"] = hit_all

    spike_data["hmt_try_tt_beaconed"] = try_beaconed
    spike_data["hmt_try_tt_non_beaconed"] = try_non_beaconed
    spike_data["hmt_try_tt_probe"] = try_probe
    spike_data["hmt_try_tt_all"] = try_all

    spike_data["hmt_miss_tt_beaconed"] = miss_beaconed
    spike_data["hmt_miss_tt_non_beaconed"] = miss_non_beaconed
    spike_data["hmt_miss_tt_probe"] = miss_probe
    spike_data["hmt_miss_tt_all"] = miss_all

    spike_data["hmt_all_tt_beaconed"] = all_beaconed
    spike_data["hmt_all_tt_non_beaconed"] = all_non_beaconed
    spike_data["hmt_all_tt_probe"] = all_probe
    spike_data["hmt_all_tt_all"] = all__all

    return spike_data


def process_pairwise_pearson_correlations(spike_data, processed_position_data):
    print('adding pairwise pearson correlations scores...')
    hits =[]
    misses =[]
    tries =[]
    all = []

    hits_beaconed = []
    hits_probe = []
    hits_non_beaconed = []

    misses_beaconed = []
    misses_non_beaconed = []
    misses_probe = []

    try_beaconed = []
    try_non_beaconed = []
    try_probe = []

    all_beaconed = []
    all_non_beaconed = []
    all_probe = []

    all_trials = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        cluster_firing_maps = np.array(cluster_spike_data["firing_rate_maps"].iloc[0])
        where_are_NaNs = np.isnan(cluster_firing_maps)
        cluster_firing_maps[where_are_NaNs] = 0

        hits_beaconed.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="hit", trial_type=0)[0])
        hits_non_beaconed.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="hit", trial_type=1)[0])
        hits_probe.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="hit", trial_type=2)[0])

        misses_beaconed.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="miss", trial_type=0)[0])
        misses_non_beaconed.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="miss", trial_type=1)[0])
        misses_probe.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="miss", trial_type=2)[0])

        try_beaconed.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="try", trial_type=0)[0])
        try_non_beaconed.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="try", trial_type=1)[0])
        try_probe.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="try", trial_type=2)[0])

        all_beaconed.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="all", trial_type=0)[0])
        all_non_beaconed.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="all", trial_type=1)[0])
        all_probe.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="all", trial_type=2)[0])

        hits.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="hit", trial_type="all")[0])
        misses.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="miss", trial_type="all")[0])
        tries.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="try", trial_type="all")[0])
        all.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="all", trial_type="all")[0])

        all_trials.append(get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try="all", trial_type="all")[1])


    spike_data["avg_pairwise_trial_pearson_r"] = all
    spike_data["avg_pairwise_trial_pearson_r_hit"] = hits
    spike_data["avg_pairwise_trial_pearson_r_miss"] = misses
    spike_data["avg_pairwise_trial_pearson_r_try"] = tries
    spike_data["avg_pairwise_trial_pearson_r_hit_b"] = hits_beaconed
    spike_data["avg_pairwise_trial_pearson_r_hit_nb"] = hits_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_hit_p"] = hits_probe
    spike_data["avg_pairwise_trial_pearson_r_miss_b"] = misses_beaconed
    spike_data["avg_pairwise_trial_pearson_r_miss_nb"] = misses_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_miss_p"] = misses_probe
    spike_data["avg_pairwise_trial_pearson_r_try_b"] = try_beaconed
    spike_data["avg_pairwise_trial_pearson_r_try_nb"] = try_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_try_p"] = try_probe
    spike_data["avg_pairwise_trial_pearson_r_all_b"] = all_beaconed
    spike_data["avg_pairwise_trial_pearson_r_all_nb"] = all_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_all_p"] = all_probe
    spike_data["pairwise_trial_pearson_r"] = all_trials
    return spike_data

def add_pairwise_classifier(spike_data):
    print('adding pairwise pearson correlations shuffle_scores...')
    hits =[]
    misses =[]
    tries =[]
    all = []

    hits_beaconed = []
    hits_probe = []
    hits_non_beaconed = []

    misses_beaconed = []
    misses_non_beaconed = []
    misses_probe = []

    try_beaconed = []
    try_non_beaconed = []
    try_probe = []

    all_beaconed = []
    all_non_beaconed = []
    all_probe = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]

        if cluster_spike_data["avg_pairwise_trial_pearson_r"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_bin_shuffle_threshold"].iloc[0]:
            all.append(True)
        else:
            all.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_hit"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_hit_bin_shuffle_threshold"].iloc[0]:
            hits.append(True)
        else:
            hits.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_miss"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_miss_bin_shuffle_threshold"].iloc[0]:
            misses.append(True)
        else:
            misses.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_try"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_try_bin_shuffle_threshold"].iloc[0]:
            tries.append(True)
        else:
            tries.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_hit_b"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_hit_b_bin_shuffle_threshold"].iloc[0]:
            hits_beaconed.append(True)
        else:
            hits_beaconed.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_hit_nb"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_hit_nb_bin_shuffle_threshold"].iloc[0]:
            hits_non_beaconed.append(True)
        else:
            hits_non_beaconed.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_hit_p"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_hit_p_bin_shuffle_threshold"].iloc[0]:
            hits_probe.append(True)
        else:
            hits_probe.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_miss_b"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_miss_b_bin_shuffle_threshold"].iloc[0]:
            misses_beaconed.append(True)
        else:
            misses_beaconed.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_miss_nb"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_miss_nb_bin_shuffle_threshold"].iloc[0]:
            misses_non_beaconed.append(True)
        else:
            misses_non_beaconed.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_miss_p"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_miss_p_bin_shuffle_threshold"].iloc[0]:
            misses_probe.append(True)
        else:
            misses_probe.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_try_b"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_try_b_bin_shuffle_threshold"].iloc[0]:
            try_beaconed.append(True)
        else:
            try_beaconed.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_try_nb"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_try_nb_bin_shuffle_threshold"].iloc[0]:
            try_non_beaconed.append(True)
        else:
            try_non_beaconed.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_try_p"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_try_p_bin_shuffle_threshold"].iloc[0]:
            try_probe.append(True)
        else:
            try_probe.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_all_b"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_all_b_bin_shuffle_threshold"].iloc[0]:
            all_beaconed.append(True)
        else:
            all_beaconed.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_all_nb"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_all_nb_bin_shuffle_threshold"].iloc[0]:
            all_non_beaconed.append(True)
        else:
            all_non_beaconed.append(False)

        if cluster_spike_data["avg_pairwise_trial_pearson_r_all_p"].iloc[0] > cluster_spike_data["avg_pairwise_trial_pearson_r_all_p_bin_shuffle_threshold"].iloc[0]:
            all_probe.append(True)
        else:
            all_probe.append(False)

    spike_data["avg_pairwise_trial_pearson_r_stable"] = all
    spike_data["avg_pairwise_trial_pearson_r_stable_hits"] = hits
    spike_data["avg_pairwise_trial_pearson_r_stable_misses"] = misses
    spike_data["avg_pairwise_trial_pearson_r_stable_tries"] = tries
    spike_data["avg_pairwise_trial_pearson_r_stable_b_hits"] = hits_beaconed
    spike_data["avg_pairwise_trial_pearson_r_stable_nb_hits"] = hits_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_stable_p_hits"] = hits_probe
    spike_data["avg_pairwise_trial_pearson_r_stable_b_misses"] = misses_beaconed
    spike_data["avg_pairwise_trial_pearson_r_stable_nb_misses"] = misses_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_stable_p_misses"] = misses_probe
    spike_data["avg_pairwise_trial_pearson_r_stable_b_tries"] = try_beaconed
    spike_data["avg_pairwise_trial_pearson_r_stable_nb_tries"] = try_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_stable_p_tries"] = try_probe
    spike_data["avg_pairwise_trial_pearson_r_stable_b"] = all_beaconed
    spike_data["avg_pairwise_trial_pearson_r_stable_nb"] = all_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_stable_p"] = all_probe
    return spike_data


def process_pairwise_pearson_correlations_shuffle(spike_data, processed_position_data):
    print('adding pairwise pearson correlations shuffle_scores...')
    hits =[]
    misses =[]
    tries =[]
    all = []

    hits_beaconed = []
    hits_probe = []
    hits_non_beaconed = []

    misses_beaconed = []
    misses_non_beaconed = []
    misses_probe = []

    try_beaconed = []
    try_non_beaconed = []
    try_probe = []

    all_beaconed = []
    all_non_beaconed = []
    all_probe = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        cluster_firing_maps = np.array(cluster_spike_data["firing_rate_maps"].iloc[0])
        where_are_NaNs = np.isnan(cluster_firing_maps)
        cluster_firing_maps[where_are_NaNs] = 0

        hits.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="hit", trial_type="all"))
        misses.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="miss", trial_type="all"))
        tries.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="try", trial_type="all"))
        all.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="all", trial_type="all"))

        hits_beaconed.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="hit", trial_type=0))
        hits_non_beaconed.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="hit", trial_type=1))
        hits_probe.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="hit", trial_type=2))

        misses_beaconed.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="miss", trial_type=0))
        misses_non_beaconed.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="miss", trial_type=1))
        misses_probe.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="miss", trial_type=2))

        try_beaconed.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="try", trial_type=0))
        try_non_beaconed.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="try", trial_type=1))
        try_probe.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="try", trial_type=2))

        all_beaconed.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="all", trial_type=0))
        all_non_beaconed.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="all", trial_type=1))
        all_probe.append(get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try="all", trial_type=2))

    spike_data["avg_pairwise_trial_pearson_r_bin_shuffle_threshold"] = all
    spike_data["avg_pairwise_trial_pearson_r_hit_bin_shuffle_threshold"] = hits
    spike_data["avg_pairwise_trial_pearson_r_miss_bin_shuffle_threshold"] = misses
    spike_data["avg_pairwise_trial_pearson_r_try_bin_shuffle_threshold"] = tries
    spike_data["avg_pairwise_trial_pearson_r_hit_b_bin_shuffle_threshold"] = hits_beaconed
    spike_data["avg_pairwise_trial_pearson_r_hit_nb_bin_shuffle_threshold"] = hits_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_hit_p_bin_shuffle_threshold"] = hits_probe
    spike_data["avg_pairwise_trial_pearson_r_miss_b_bin_shuffle_threshold"] = misses_beaconed
    spike_data["avg_pairwise_trial_pearson_r_miss_nb_bin_shuffle_threshold"] = misses_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_miss_p_bin_shuffle_threshold"] = misses_probe
    spike_data["avg_pairwise_trial_pearson_r_try_b_bin_shuffle_threshold"] = try_beaconed
    spike_data["avg_pairwise_trial_pearson_r_try_nb_bin_shuffle_threshold"] = try_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_try_p_bin_shuffle_threshold"] = try_probe
    spike_data["avg_pairwise_trial_pearson_r_all_b_bin_shuffle_threshold"] = all_beaconed
    spike_data["avg_pairwise_trial_pearson_r_all_nb_bin_shuffle_threshold"] = all_non_beaconed
    spike_data["avg_pairwise_trial_pearson_r_all_p_bin_shuffle_threshold"] = all_probe
    return spike_data

def get_pairwise_score(cluster_firing_maps, processed_position_data, hit_miss_try, trial_type):
    # filter the processed_position_data based on hit miss try and the trial type
    if hit_miss_try == "all":
        processed_position_data = processed_position_data
    else:
        processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hit_miss_try)]

    if trial_type == "all":
        processed_position_data = processed_position_data
    else:
        processed_position_data = processed_position_data[(processed_position_data["trial_type"] == trial_type)]

    trial_correlations = []
    # compute the spatial information score if any trials qualify
    if len(processed_position_data) == 0:
        pairwise_score = np.nan
        return pairwise_score, trial_correlations
    else:
        trial_numbers = np.unique(processed_position_data["trial_number"]).tolist()
        trial_numbers_indices = np.array(trial_numbers)-1
        cluster_firing_maps = cluster_firing_maps[trial_numbers_indices, :]

        for i in range(len(cluster_firing_maps)-1):
            pearson_r = stats.pearsonr(cluster_firing_maps[i].flatten(),cluster_firing_maps[i+1].flatten())
            trial_correlations.append(pearson_r[0])
        pairwise_score = np.nanmean(np.array(trial_correlations))
        return pairwise_score, trial_correlations

def get_pairwise_shuffle_score(cluster_firing_maps, processed_position_data, hit_miss_try, trial_type):
    # filter the processed_position_data based on hit miss try and the trial type
    if hit_miss_try == "all":
        processed_position_data = processed_position_data
    else:
        processed_position_data = processed_position_data[(processed_position_data["hit_miss_try"] == hit_miss_try)]

    if trial_type == "all":
        processed_position_data = processed_position_data
    else:
        processed_position_data = processed_position_data[(processed_position_data["trial_type"] == trial_type)]

    # compute the spatial information score if any trials qualify
    if len(processed_position_data) == 0:
        adjusted_pairwise_threshold = np.nan
        return adjusted_pairwise_threshold
    else:
        trial_numbers = np.unique(processed_position_data["trial_number"]).tolist()
        trial_numbers_indices = np.array(trial_numbers)-1
        cluster_firing_maps = cluster_firing_maps[trial_numbers_indices, :]

        N_SHUFFLES = 1000
        shuffled_pairwise_scores = []
        for i in range(N_SHUFFLES):
            # shuffle the bin locations in cluster_firing_maps n times
            shuffled_firing_maps = np.array([list(np.random.permutation(x)) for x in cluster_firing_maps])
            trial_correlations = []
            for i in range(len(cluster_firing_maps)-1):
                pearson_r = stats.pearsonr(shuffled_firing_maps[i].flatten(),shuffled_firing_maps[i+1].flatten())
                trial_correlations.append(pearson_r[0])
            pairwise_score = np.nanmean(np.array(trial_correlations))
            shuffled_pairwise_scores.append(pairwise_score)

        shuffled_pairwise_scores_og = np.array(shuffled_pairwise_scores)
        shuffled_pairwise_scores = shuffled_pairwise_scores_og[~np.isnan(shuffled_pairwise_scores_og)]
        print("There was this many nan values for the shuffled rate maps:", len(shuffled_pairwise_scores_og)-len(shuffled_pairwise_scores))
        adjusted_pairwise_threshold = np.nanmean(shuffled_pairwise_scores) + (np.nanstd(shuffled_pairwise_scores)*2.326) # one tailed 99th percentile

        return adjusted_pairwise_threshold

def scramble(a, axis=-1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """
    b = a.swapaxes(axis, -1)
    n = a.shape[axis]
    idx = np.random.choice(n, n, replace=False)
    b = b[..., idx]
    return b.swapaxes(axis, -1)

def plot_cumhist_hmt(combined_df, save_path):
    combined_df = combined_df[combined_df["grid_cell"] == 1]

    hmt_hit_tt_beaconed = np.asarray(combined_df["hmt_hit_tt_beaconed"])
    hmt_hit_tt_non_beaconed = np.asarray(combined_df["hmt_hit_tt_non_beaconed"])
    hmt_hit_tt_probe = np.asarray(combined_df["hmt_hit_tt_probe"])
    hmt_hit_tt_all = np.asarray(combined_df["hmt_hit_tt_all"])
    hmt_try_tt_beaconed = np.asarray(combined_df["hmt_try_tt_beaconed"])
    hmt_try_tt_non_beaconed = np.asarray(combined_df["hmt_try_tt_non_beaconed"])
    hmt_try_tt_probe = np.asarray(combined_df["hmt_try_tt_probe"])
    hmt_try_tt_all = np.asarray(combined_df["hmt_try_tt_all"])
    hmt_miss_tt_beaconed = np.asarray(combined_df["hmt_miss_tt_beaconed"])
    hmt_miss_tt_non_beaconed = np.asarray(combined_df["hmt_miss_tt_non_beaconed"])
    hmt_miss_tt_probe = np.asarray(combined_df["hmt_miss_tt_probe"])
    hmt_miss_tt_all = np.asarray(combined_df["hmt_miss_tt_all"])
    hmt_all_tt_beaconed = np.asarray(combined_df["hmt_all_tt_beaconed"])
    hmt_all_tt_non_beaconed = np.asarray(combined_df["hmt_all_tt_non_beaconed"])
    hmt_all_tt_probe = np.asarray(combined_df["hmt_all_tt_probe"])
    hmt_all_tt_all = np.asarray(combined_df["hmt_all_tt_all"])

    non_beaconed = np.vstack((hmt_hit_tt_non_beaconed, hmt_try_tt_non_beaconed, hmt_miss_tt_non_beaconed))
    fig, ax = plt.subplots(figsize=(4,4))
    objects = ["Hit", "Try", "Miss"]
    x_pos = np.arange(len(objects))
    for i in range(len(non_beaconed[0])):
        ax.plot(x_pos, non_beaconed[:, i], 'o-', color="red", alpha=0.3)
    plt.xticks(x_pos, objects, fontsize=8)
    plt.xticks(rotation=-45)
    plt.locator_params(axis='y', nbins=4)
    plt.ylabel("Spatial Information",  fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    plt.ylim((0, 10))
    #plt.gca().spines['top'].set_visible(False)
    #plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"/nb_bar.png", dpi=300)
    plt.show()

def plot_spatial_info_hist(combined_df, save_path):
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.2, right=0.9, bottom=0.2, top=0.9,
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlabel("Grid Score", fontsize=20, labelpad=10)
    ax.set_ylabel("Avg Trial-pair Pearson R", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    border_cells = combined_df[combined_df["classifier"] == "B"]
    hd_cells = combined_df[combined_df["classifier"] == "HD"]
    spatial_cells = combined_df[combined_df["classifier"] == "NG"]
    non_spatial_cells = combined_df[combined_df["classifier"] == "NS"]

    y = np.asarray(spatial_cells["avg_pairwise_trial_pearson_r"])
    x = np.asarray(spatial_cells["grid_score"])
    scatter_hist(x, y, ax, bin_width_x=0.2, bin_width_y=0.2, lim_x1=-0.5, lim_x2=1.5, lim_y1=-0.5, lim_y2=1, color="mediumblue", alpha=1)

    y = np.asarray(grid_cells["avg_pairwise_trial_pearson_r"])
    x = np.asarray(grid_cells["grid_score"])
    scatter_hist(x, y, ax, bin_width_x=0.2, bin_width_y=0.2, lim_x1=-0.5, lim_x2=1.5, lim_y1=-0.5, lim_y2=1, color="turquoise", alpha=1)

    y = np.asarray(hd_cells["avg_pairwise_trial_pearson_r"])
    x = np.asarray(hd_cells["grid_score"])
    scatter_hist(x, y, ax, bin_width_x=0.2, bin_width_y=0.2, lim_x1=-0.5, lim_x2=1.5, lim_y1=-0.5, lim_y2=1, color="darkorange", alpha=1)

    y = np.asarray(border_cells["avg_pairwise_trial_pearson_r"])
    x = np.asarray(border_cells["grid_score"])
    scatter_hist(x, y, ax, bin_width_x=0.2, bin_width_y=0.2, lim_x1=-0.5, lim_x2=1.5, lim_y1=-0.5, lim_y2=1, color="indianred", alpha=1)

    y = np.asarray(non_spatial_cells["avg_pairwise_trial_pearson_r"])
    x = np.asarray(non_spatial_cells["grid_score"])
    scatter_hist(x, y, ax, bin_width_x=0.2, bin_width_y=0.2, lim_x1=-0.5, lim_x2=1.5, lim_y1=-0.5, lim_y2=1, color="blueviolet", alpha=1)

    plt.tight_layout()
    plt.ylim((-0.5, 1))
    plt.savefig(save_path+"/scatterhist_grid_score.png", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(8,3))
    ax.set_ylabel("Avg Trial-pair Pearson R", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    objects = ["Border", "Non-spatial", "Grid", "HD", "Spatial"]
    x_pos = np.arange(len(objects))

    y = np.asarray(border_cells["avg_pairwise_trial_pearson_r"])
    ax.scatter(x_pos[0]*np.ones(len(y))+np.random.uniform(-0.2, 0.2, len(y)), np.asarray(y), edgecolor="indianred", marker="o", facecolors='none', alpha=0.3)
    ax.errorbar(x_pos[0], np.nanmean(y), yerr=stats.sem(y, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")

    y = np.asarray(non_spatial_cells["avg_pairwise_trial_pearson_r"])
    ax.scatter(x_pos[1]*np.ones(len(y))+np.random.uniform(-0.2, 0.2, len(y)), np.asarray(y), edgecolor="blueviolet", marker="o", facecolors='none', alpha=0.3)
    ax.errorbar(x_pos[1], np.nanmean(y), yerr=stats.sem(y, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")

    y = np.asarray(grid_cells["avg_pairwise_trial_pearson_r"])
    ax.scatter(x_pos[2]*np.ones(len(y))+np.random.uniform(-0.2, 0.2, len(y)), np.asarray(y), edgecolor="turquoise", marker="o", facecolors='none', alpha=0.3)
    ax.errorbar(x_pos[2], np.nanmean(y), yerr=stats.sem(y, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")

    y = np.asarray(hd_cells["avg_pairwise_trial_pearson_r"])
    ax.scatter(x_pos[3]*np.ones(len(y))+np.random.uniform(-0.2, 0.2, len(y)), np.asarray(y), edgecolor="darkorange", marker="o", facecolors='none', alpha=0.3)
    ax.errorbar(x_pos[3], np.nanmean(y), yerr=stats.sem(y, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")

    y = np.asarray(spatial_cells["avg_pairwise_trial_pearson_r"])
    ax.scatter(x_pos[4]*np.ones(len(y))+np.random.uniform(-0.2, 0.2, len(y)), np.asarray(y), edgecolor="mediumblue", marker="o", facecolors='none', alpha=0.3)
    ax.errorbar(x_pos[4], np.nanmean(y), yerr=stats.sem(y, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")

    plt.xticks(x_pos, objects, fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_bar_all_celltypes.png", dpi=300)
    plt.close()

def plot_spatial_info_spatial_info_hist(combined_df, save_path):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlabel("Spatial Information (OF)", fontsize=20, labelpad=10)
    ax.set_ylabel("Spatial Information (VR)", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]
    border_cells = combined_df[combined_df["classifier"] == "B"]
    hd_cells = combined_df[combined_df["classifier"] == "HD"]
    spatial_cells = combined_df[combined_df["classifier"] == "NG"]
    non_spatial_cells = combined_df[combined_df["classifier"] == "NS"]

    y = np.asarray(non_grid_cells["hmt_all_tt_all"])
    x = np.asarray(non_grid_cells["spatial_information_score"])
    scatter_hist(x, y, ax, ax_histx, ax_histy, bin_width_x=0.2, bin_width_y=0.2, lim_x1=0, lim_x2=5, lim_y1=0, lim_y2=5, color="black", alpha=0.5)
    y = np.asarray(grid_cells["hmt_all_tt_all"])
    x = np.asarray(grid_cells["spatial_information_score"])
    scatter_hist(x, y, ax, ax_histx, ax_histy, bin_width_x=0.2, bin_width_y=0.2, lim_x1=0, lim_x2=5, lim_y1=0, lim_y2=5, color="red", alpha=0.5)
    plt.tight_layout()
    plt.ylim((0,5))
    plt.savefig(save_path+"/scatterhist2.png", dpi=300)
    plt.show()

def plot_spatial_info_cum_hist(combined_df, save_path):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlabel("Spatial Information", fontsize=20, labelpad=10)
    ax.set_ylabel("Cumulative Density", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    non_grid_spatial_information_vr = np.asarray(non_grid_cells["hmt_all_tt_all"])
    non_grid_spatial_information_of = np.asarray(non_grid_cells["spatial_information_score"])
    grid_spatial_information_vr = np.asarray(grid_cells["hmt_all_tt_all"])
    grid_spatial_information_of = np.asarray(grid_cells["spatial_information_score"])

    _, _, patchesP = ax.hist(grid_spatial_information_vr, bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=1, linestyle="--")
    _, _, patchesA = ax.hist(grid_spatial_information_of, bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=1)
    #_, _, patchesS = ax.hist(non_grid_spatial_information_vr, bins=500, color="black", histtype="step", density=True, cumulative=True, linewidth=1, linestyle="--")
    #_, _, patchesJ = ax.hist(non_grid_spatial_information_of, bins=500, color="black", histtype="step", density=True, cumulative=True, linewidth=1)
    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    #patchesS[0].set_xy(patchesS[0].get_xy()[:-1])
    #patchesJ[0].set_xy(patchesJ[0].get_xy()[:-1])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"spatial_cumhist.png", dpi=300)
    plt.show()



def plot_pairwise_comparison(combined_df, save_path, CT="", PDN=""):
    combined_df = add_lomb_classifier(combined_df)
    if CT=="G":
        grid_cells = combined_df[combined_df["classifier"] == "G"]
    elif CT=="NG":
        grid_cells = combined_df[combined_df["classifier"] != "G"]
    if PDN != "":
        grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == PDN]
    hits = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_hit"])
    misses = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_miss"])
    tries = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_try"])

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel("Avg Trial-pair Pearson R", fontsize=20, labelpad=10)
    ax.set_ylabel("Cumulative Density", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    _, _, patchesP = ax.hist(hits[~np.isnan(hits)], bins=500, color="green", histtype="step", density=True, cumulative=True, linewidth=1)
    _, _, patchesA = ax.hist(misses[~np.isnan(misses)], bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=1)
    _, _, patchesS = ax.hist(tries[~np.isnan(tries)], bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=1)
    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    patchesS[0].set_xy(patchesS[0].get_xy()[:-1])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_cumhist_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_ylabel("Avg Trial-pair Pearson R", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    objects = ["Hit", "Try", "Miss"]
    x_pos = np.arange(len(objects))
    for i in range(len(hits)):
        ax.plot(x_pos, [hits[i], tries[i], misses[i]], color="black", alpha=0.1)

    ax.errorbar(x_pos[0], np.nanmean(hits), yerr=stats.sem(hits, nan_policy='omit'), ecolor='green', capsize=20, fmt="o", color="green")
    ax.scatter(x_pos[0]*np.ones(len(hits)), np.asarray(hits), edgecolor="green", marker="o", facecolors='none')

    ax.errorbar(x_pos[1], np.nanmean(tries), yerr=stats.sem(tries, nan_policy='omit'), ecolor='orange', capsize=20, fmt="o", color="orange")
    ax.scatter(x_pos[1]*np.ones(len(tries)), np.asarray(tries), edgecolor="orange", marker="o", facecolors='none')

    ax.errorbar(x_pos[2], np.nanmean(misses), yerr=stats.sem(misses, nan_policy='omit'), ecolor='red', capsize=20, fmt="o", color="red")
    ax.scatter(x_pos[2]*np.ones(len(misses)), np.asarray(misses), edgecolor="red", marker="o", facecolors='none')

    bad_hm = ~np.logical_or(np.isnan(hits), np.isnan(misses))
    bad_ht = ~np.logical_or(np.isnan(hits), np.isnan(tries))
    bad_tm = ~np.logical_or(np.isnan(tries), np.isnan(misses))
    hit_miss_p = stats.wilcoxon(np.compress(bad_hm, hits), np.compress(bad_hm, misses))[1]
    hit_try_p = stats.wilcoxon(np.compress(bad_ht, hits), np.compress(bad_ht, tries))[1]
    try_miss_p = stats.wilcoxon(np.compress(bad_tm, tries), np.compress(bad_tm, misses))[1]

    all_behaviour = []; all_behaviour.extend(hits.tolist()); all_behaviour.extend(misses.tolist()); all_behaviour.extend(tries.tolist())
    significance_bar(start=x_pos[0], end=x_pos[1], height=np.nanmax(all_behaviour)+0, displaystring=get_p_text(hit_try_p))
    significance_bar(start=x_pos[1], end=x_pos[2], height=np.nanmax(all_behaviour)+0.1, displaystring=get_p_text(try_miss_p))
    significance_bar(start=x_pos[0], end=x_pos[2], height=np.nanmax(all_behaviour)+0.2, displaystring=get_p_text(hit_miss_p))

    plt.xticks(x_pos, objects, fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_bar_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()


def hmt2collumn(hmt, tt):
    if tt == "all" and hmt == "all":
        return 'ML_SNRs'
    elif tt == "all" and hmt == "hit":
        return 'ML_SNRs_all_hits'
    elif tt == "all" and hmt == "try":
        return 'ML_SNRs_all_tries'
    elif tt == "all" and hmt == "miss":
        return 'ML_SNRs_all_misses'
    elif tt == "beaconed" and hmt == "all":
        return 'ML_SNRs_all_beaconed'
    elif tt == "beaconed" and hmt == "hit":
        return 'ML_SNRs_beaconed_hits'
    elif tt == "beaconed" and hmt == "try":
        return 'ML_SNRs_beaconed_tries'
    elif tt == "beaconed" and hmt == "miss":
        return 'ML_SNRs_beaconed_misses'
    elif tt == "non_beaconed" and hmt == "all":
        return 'ML_SNRs_all_nonbeaconed'
    elif tt == "non_beaconed" and hmt == "hit":
        return 'ML_SNRs_nonbeaconed_hits'
    elif tt == "non_beaconed" and hmt == "try":
        return 'ML_SNRs_nonbeaconed_tries'
    elif tt == "non_beaconed" and hmt == "miss":
        return 'ML_SNRs_nonbeaconed_misses'
    elif tt == "probe" and hmt == "all":
        return 'ML_SNRs_all_probe'
    elif tt == "probe" and hmt == "hit":
        return 'ML_SNRs_probe_hits'
    elif tt == "probe" and hmt == "try":
        return 'ML_SNRs_probe_tries'
    elif tt == "probe" and hmt == "miss":
        return 'ML_SNRs_probe_misses'


def plot_SNR_comparison_tt(combined_df, save_path, CT="", PDN="", hmt="", get_lomb_classifier=True):
    if get_lomb_classifier:
        combined_df = add_lomb_classifier(combined_df)
    if CT=="G":
        grid_cells = combined_df[combined_df["classifier"] == "G"]
    elif CT=="NG":
        grid_cells = combined_df[combined_df["classifier"] != "G"]
    if PDN != "":
        grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == PDN]

    a = np.asarray(grid_cells[hmt2collumn(hmt, tt="beaconed")], dtype=np.float64)
    b = np.asarray(grid_cells[hmt2collumn(hmt, tt="non_beaconed")], dtype=np.float64)
    c = np.asarray(grid_cells[hmt2collumn(hmt, tt="probe")], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel("Periodic Power", fontsize=40, labelpad=10)
    ax.set_ylabel("Cumulative Density", fontsize=40, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=4)
    _, _, patchesP = ax.hist(a[~np.isnan(a)], bins=500, color="black", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesA = ax.hist(b[~np.isnan(b)], bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesS = ax.hist(c[~np.isnan(c)], bins=500, color="blue", histtype="step", density=True, cumulative=True, linewidth=2)
    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    patchesS[0].set_xy(patchesS[0].get_xy()[:-1])
    ax.set_ylim([0,1])
    ax.set_xlim([0,0.3])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_cumhist_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()


    x1 = 0 * np.ones(len(a[~np.isnan(a)]))
    x2 = 1 * np.ones(len(b[~np.isnan(b)]))
    x3 = 2 * np.ones(len(c[~np.isnan(c)]))
    y1 = a[~np.isnan(a)]
    y2 = b[~np.isnan(b)]
    y3 = c[~np.isnan(c)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)
    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)


    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_ylabel("Periodic Power", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ["B", "NB", "P"]
    objects = ["Cued", "PI"]
    x_pos = np.arange(len(objects))
    for i in range(len(a)):
        ax.plot(x_pos, [a[i], b[i]], color="black", alpha=0.1)

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0)
    ax.errorbar(x_pos[0], np.nanmean(a), yerr=stats.sem(a, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")
    ax.bar(x_pos[0], np.nanmean(a), edgecolor="black", color="None", facecolor="None", linewidth=3, width=0.5)
    #ax.scatter(x_pos[0]*np.ones(len(a)), np.asarray(a), edgecolor="black", marker="o", facecolors='none')

    ax.errorbar(x_pos[1], np.nanmean(b), yerr=stats.sem(b, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")
    ax.bar(x_pos[1], np.nanmean(b), edgecolor="blue", color="None", facecolor="None", linewidth=3, width=0.5)
    #ax.scatter(x_pos[1]*np.ones(len(b)), np.asarray(b), edgecolor="red", marker="o", facecolors='none')

    #ax.errorbar(x_pos[2], np.nanmean(c), yerr=stats.sem(c, nan_policy='omit'), ecolor='blue', capsize=20, fmt="o", color="blue")
    #ax.bar(x_pos[2], np.nanmean(c), edgecolor="blue", color="None", facecolor="None", linewidth=3, width=0.5)
    #ax.scatter(x_pos[2]*np.ones(len(c)), np.asarray(c), edgecolor="blue", marker="o", facecolors='none')

    ax.plot(x_pos, [np.nanmean(a), np.nanmean(b)], color="black", linestyle="solid", linewidth=2)

    bad_ac = ~np.logical_or(np.isnan(a), np.isnan(c))
    bad_ab = ~np.logical_or(np.isnan(a), np.isnan(b))
    bad_bc = ~np.logical_or(np.isnan(b), np.isnan(c))
    ac_p = stats.wilcoxon(np.compress(bad_ac, a), np.compress(bad_ac, c))[1]
    ab_p = stats.wilcoxon(np.compress(bad_ab, a), np.compress(bad_ab, b))[1]
    bc_p = stats.wilcoxon(np.compress(bad_bc, b), np.compress(bad_bc, c))[1]

    all_behaviour = []; all_behaviour.extend(a.tolist()); all_behaviour.extend(c.tolist()); all_behaviour.extend(b.tolist())
    significance_bar(start=x_pos[0], end=x_pos[1], height=0.3, displaystring=get_p_text(ab_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=np.nanmax(all_behaviour)+0.1, displaystring=get_p_text(bc_p))
    #significance_bar(start=x_pos[0], end=x_pos[2], height=np.nanmax(all_behaviour)+0.2, displaystring=get_p_text(ac_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=0, top=0.3)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_bar_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()


def hmt2spatial_information_collumn(hmt, tt):
    if tt == "all" and hmt == "all":
        return 'hmt_all_tt_all'
    elif tt == "all" and hmt == "hit":
        return 'hmt_hit_tt_all'
    elif tt == "all" and hmt == "try":
        return 'hmt_try_tt_all'
    elif tt == "all" and hmt == "miss":
        return 'hmt_miss_tt_all'
    elif tt == "beaconed" and hmt == "all":
        return 'hmt_all_tt_beaconed'
    elif tt == "beaconed" and hmt == "hit":
        return 'hmt_hit_tt_beaconed'
    elif tt == "beaconed" and hmt == "try":
        return 'hmt_try_tt_beaconed'
    elif tt == "beaconed" and hmt == "miss":
        return 'hmt_miss_tt_beaconed'
    elif tt == "non_beaconed" and hmt == "all":
        return 'hmt_all_tt_non_beaconed'
    elif tt == "non_beaconed" and hmt == "hit":
        return 'hmt_hit_tt_non_beaconed'
    elif tt == "non_beaconed" and hmt == "try":
        return 'hmt_try_tt_non_beaconed'
    elif tt == "non_beaconed" and hmt == "miss":
        return 'hmt_miss_tt_non_beaconed'
    elif tt == "probe" and hmt == "all":
        return 'hmt_all_tt_probe'
    elif tt == "probe" and hmt == "hit":
        return 'hmt_hit_tt_probe'
    elif tt == "probe" and hmt == "try":
        return 'hmt_try_tt_probe'
    elif tt == "probe" and hmt == "miss":
        return 'hmt_miss_tt_probe'


def plot_spatial_info_comparison(combined_df, save_path, CT="", PDN="", tt="", get_lomb_classifier=True):

    if get_lomb_classifier:
        combined_df = add_lomb_classifier(combined_df)
    if CT=="G":
        grid_cells = combined_df[combined_df["classifier"] == "G"]
    elif CT=="NG":
        grid_cells = combined_df[combined_df["classifier"] != "G"]

    if PDN == "PD":
        grid_cells = grid_cells[(grid_cells["Lomb_classifier_"] == "Position") |
                                (grid_cells["Lomb_classifier_"] == "Distance")]
    elif PDN != "":
        grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == PDN]

    hits = np.asarray(grid_cells[hmt2spatial_information_collumn(hmt="hit", tt=tt)])
    misses = np.asarray(grid_cells[hmt2spatial_information_collumn(hmt="miss", tt=tt)])
    tries = np.asarray(grid_cells[hmt2spatial_information_collumn(hmt="try", tt=tt)])

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel("Spatial Information", fontsize=40, labelpad=10)
    ax.set_ylabel("Cumulative Density", fontsize=40, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=4)
    _, _, patchesP = ax.hist(hits[~np.isnan(hits)], bins=500, color="green", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesA = ax.hist(misses[~np.isnan(misses)], bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesS = ax.hist(tries[~np.isnan(tries)], bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=2)
    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    patchesS[0].set_xy(patchesS[0].get_xy()[:-1])
    ax.set_ylim([0,1])
    #ax.set_xlim([0,0.3])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_cumhist_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()

    diff = hits - misses
    diff = diff[~np.isnan(diff)]

    x1 = 0 * np.ones(len(hits[~np.isnan(hits)]))
    x2 = 1 * np.ones(len(misses[~np.isnan(misses)]))
    x3 = 2 * np.ones(len(diff[~np.isnan(diff)]))
    y1 = hits[~np.isnan(hits)]
    y2 = misses[~np.isnan(misses)]
    y3 = diff[~np.isnan(diff)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)

    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel("Spatial Information", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ['Hit', 'Miss', r'$\Delta$']
    x_pos = np.arange(len(objects))
    ax.axhline(y=0, linewidth=3, color="black")

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0)
    ax.errorbar(x_pos[0], np.nanmean(hits), yerr=stats.sem(hits, nan_policy='omit'), ecolor='green', capsize=10, fmt="o", color="green", linewidth=3)
    ax.bar(x_pos[0], np.nanmean(hits), edgecolor="green", color="None", facecolor="None", linewidth=3)
    #ax.scatter(x_pos[0]*np.ones(len(hits)), np.asarray(hits), edgecolor="green", marker="o", facecolors='none')
    #ax.errorbar(x_pos[1], np.nanmean(tries), yerr=stats.sem(tries, nan_policy='omit'), ecolor='orange', capsize=20, fmt="o", color="orange")
    #ax.scatter(x_pos[1]*np.ones(len(tries)), np.asarray(tries), edgecolor="orange", marker="o", facecolors='none')
    ax.errorbar(x_pos[1], np.nanmean(misses), yerr=stats.sem(misses, nan_policy='omit'), ecolor='red', capsize=10, fmt="o", color="red", linewidth=3)
    ax.bar(x_pos[1], np.nanmean(misses), edgecolor="red", color="None", facecolor="None", linewidth=3)

    ax.errorbar(x_pos[2], np.nanmean(diff), yerr=stats.sem(diff, nan_policy='omit'), ecolor='black', capsize=10, fmt="o", color="black", linewidth=3)
    ax.bar(x_pos[2], np.nanmean(diff), edgecolor="black", color="None", facecolor="None", linewidth=3)
    #ax.scatter(x_pos[2]*np.ones(len(misses)), np.asarray(misses), edgecolor="red", marker="o", facecolors='none')

    bad_hm = ~np.logical_or(np.isnan(hits), np.isnan(misses))
    #bad_ht = ~np.logical_or(np.isnan(hits), np.isnan(tries))
    #bad_tm = ~np.logical_or(np.isnan(tries), np.isnan(misses))
    hit_miss_p = stats.wilcoxon(np.compress(bad_hm, hits), np.compress(bad_hm, misses))[1]
    #hit_try_p = stats.wilcoxon(np.compress(bad_ht, hits), np.compress(bad_ht, tries))[1]
    #try_miss_p = stats.wilcoxon(np.compress(bad_tm, tries), np.compress(bad_tm, misses))[1]

    all_behaviour = []; all_behaviour.extend(hits.tolist()); all_behaviour.extend(misses.tolist())
    #significance_bar(start=x_pos[0], end=x_pos[1], height=np.nanmax(all_behaviour)+0, displaystring=get_p_text(hit_try_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=np.nanmax(all_behaviour)+0.1, displaystring=get_p_text(try_miss_p))
    significance_bar(start=x_pos[0], end=x_pos[1], height=0.3, displaystring=get_p_text(hit_miss_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    #ax.set_ylim(bottom=-0.1, top=0.31)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_bar_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()

def get_indices(hmt, tt):
    i = tt
    if hmt=="hit":
        j = 0
    elif hmt=="miss":
        j = 1
    elif hmt=="try":
        j = 2
    return i, j

def get_avg_correlations(spike_data, hmt, tt):
    i, j = get_indices(hmt, tt) # i is for tt and j is for hmt
    avg_correlations = []
    for index, cluster_data in spike_data.iterrows():
        cluster_data = cluster_data.to_frame().T.reset_index(drop=True)
        avg_correlation = cluster_data["avg_correlations_hmt_by_trial_type"].iloc[0][i][j]
        avg_correlations.append(avg_correlation)
    return np.array(avg_correlations)

def get_avg_map_shifts(spike_data, hmt, tt):
    i, j = get_indices(hmt, tt) # i is for tt and j is for hmt
    avg_map_shifts = []
    for index, cluster_data in spike_data.iterrows():
        cluster_data = cluster_data.to_frame().T.reset_index(drop=True)
        map_shifts = cluster_data["field_realignments_hmt_by_trial_type"].iloc[0][i][j]
        avg_shift = np.nanmean(np.abs(map_shifts))
        avg_map_shifts.append(avg_shift)
    return np.array(avg_map_shifts)

def plot_avg_correlation_comparison_tt(combined_df, save_path, CT="", PDN="", hmt="", get_lomb_classifier=True):
    if get_lomb_classifier:
        combined_df = add_lomb_classifier(combined_df)
    if CT=="G":
        grid_cells = combined_df[combined_df["classifier"] == "G"]
    elif CT=="NG":
        grid_cells = combined_df[combined_df["classifier"] != "G"]
    if PDN != "":
        grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == PDN]

    a = get_avg_correlations(grid_cells, hmt=hmt, tt=0)
    b = get_avg_correlations(grid_cells, hmt=hmt, tt=1)
    c = get_avg_correlations(grid_cells, hmt=hmt, tt=2)

    x1 = 0 * np.ones(len(a[~np.isnan(a)]))
    x2 = 1 * np.ones(len(b[~np.isnan(b)]))
    x3 = 2 * np.ones(len(c[~np.isnan(c)]))
    y1 = a[~np.isnan(a)]
    y2 = b[~np.isnan(b)]
    y3 = c[~np.isnan(c)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)
    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_ylabel("Avg R", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ["Cued", "PI"]
    x_pos = np.arange(len(objects))
    for i in range(len(a)):
        ax.plot(x_pos, [a[i], b[i]], color="black", alpha=0.1)

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0)
    ax.errorbar(x_pos[0], np.nanmean(a), yerr=stats.sem(a, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")
    ax.bar(x_pos[0], np.nanmean(a), edgecolor="black", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[1], np.nanmean(b), yerr=stats.sem(b, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")
    ax.bar(x_pos[1], np.nanmean(b), edgecolor="blue", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.plot(x_pos, [np.nanmean(a), np.nanmean(b)], color="black", linestyle="solid", linewidth=2)

    bad_ac = ~np.logical_or(np.isnan(a), np.isnan(c))
    bad_ab = ~np.logical_or(np.isnan(a), np.isnan(b))
    bad_bc = ~np.logical_or(np.isnan(b), np.isnan(c))
    ac_p = stats.wilcoxon(np.compress(bad_ac, a), np.compress(bad_ac, c))[1]
    ab_p = stats.wilcoxon(np.compress(bad_ab, a), np.compress(bad_ab, b))[1]
    bc_p = stats.wilcoxon(np.compress(bad_bc, b), np.compress(bad_bc, c))[1]

    all_behaviour = []; all_behaviour.extend(a.tolist()); all_behaviour.extend(c.tolist()); all_behaviour.extend(b.tolist())
    significance_bar(start=x_pos[0], end=x_pos[1], height=0.45, displaystring=get_p_text(ab_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=np.nanmax(all_behaviour)+0.1, displaystring=get_p_text(bc_p))
    #significance_bar(start=x_pos[0], end=x_pos[2], height=np.nanmax(all_behaviour)+0.2, displaystring=get_p_text(ac_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.axhline(y=0, color="black", linewidth=3)
    ax.set_ylim(bottom=-0.25, top=0.5)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_bar_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()


def plot_map_shifts_comparison_tt(combined_df, save_path, CT="", PDN="", hmt="", get_lomb_classifier=True):
    if get_lomb_classifier:
        combined_df = add_lomb_classifier(combined_df)
    if CT=="G":
        grid_cells = combined_df[combined_df["classifier"] == "G"]
    elif CT=="NG":
        grid_cells = combined_df[combined_df["classifier"] != "G"]
    if PDN != "":
        grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == PDN]

    a = get_avg_map_shifts(grid_cells, hmt=hmt, tt=0)
    b = get_avg_map_shifts(grid_cells, hmt=hmt, tt=1)
    c = get_avg_map_shifts(grid_cells, hmt=hmt, tt=2)

    x1 = 0 * np.ones(len(a[~np.isnan(a)]))
    x2 = 1 * np.ones(len(b[~np.isnan(b)]))
    x3 = 2 * np.ones(len(c[~np.isnan(c)]))
    y1 = a[~np.isnan(a)]
    y2 = b[~np.isnan(b)]
    y3 = c[~np.isnan(c)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)
    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_ylabel("Map Shift (cm)", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ["Cued", "PI"]
    x_pos = np.arange(len(objects))
    for i in range(len(a)):
        ax.plot(x_pos, [a[i], b[i]], color="black", alpha=0.1)

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0)
    ax.errorbar(x_pos[0], np.nanmean(a), yerr=stats.sem(a, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")
    ax.bar(x_pos[0], np.nanmean(a), edgecolor="black", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[1], np.nanmean(b), yerr=stats.sem(b, nan_policy='omit'), ecolor='black', capsize=20, fmt="o", color="black")
    ax.bar(x_pos[1], np.nanmean(b), edgecolor="blue", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.plot(x_pos, [np.nanmean(a), np.nanmean(b)], color="black", linestyle="solid", linewidth=2)

    bad_ac = ~np.logical_or(np.isnan(a), np.isnan(c))
    bad_ab = ~np.logical_or(np.isnan(a), np.isnan(b))
    bad_bc = ~np.logical_or(np.isnan(b), np.isnan(c))
    ac_p = stats.wilcoxon(np.compress(bad_ac, a), np.compress(bad_ac, c))[1]
    ab_p = stats.wilcoxon(np.compress(bad_ab, a), np.compress(bad_ab, b))[1]
    bc_p = stats.wilcoxon(np.compress(bad_bc, b), np.compress(bad_bc, c))[1]

    all_behaviour = []; all_behaviour.extend(a.tolist()); all_behaviour.extend(c.tolist()); all_behaviour.extend(b.tolist())
    significance_bar(start=x_pos[0], end=x_pos[1], height=45, displaystring=get_p_text(ab_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=np.nanmax(all_behaviour)+0.1, displaystring=get_p_text(bc_p))
    #significance_bar(start=x_pos[0], end=x_pos[2], height=np.nanmax(all_behaviour)+0.2, displaystring=get_p_text(ac_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.axhline(y=0, color="black", linewidth=3)
    ax.set_ylim(bottom=0, top=50)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_bar_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()


def plot_map_shifts_comparison(combined_df, save_path, CT="", PDN="", tt="", get_lomb_classifier=True):

    if get_lomb_classifier:
        combined_df = add_lomb_classifier(combined_df)
    if CT=="G":
        grid_cells = combined_df[combined_df["classifier"] == "G"]
    elif CT=="NG":
        grid_cells = combined_df[combined_df["classifier"] != "G"]

    if PDN == "PD":
        grid_cells = grid_cells[(grid_cells["Lomb_classifier_"] == "Position") |
                                (grid_cells["Lomb_classifier_"] == "Distance")]
    elif PDN != "":
        grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == PDN]

    hits = get_avg_map_shifts(grid_cells, hmt="hit", tt=tt)
    misses = get_avg_map_shifts(grid_cells, hmt="miss", tt=tt)
    tries = get_avg_map_shifts(grid_cells, hmt="try", tt=tt)

    x1 = 0 * np.ones(len(hits[~np.isnan(hits)]))
    x2 = 1 * np.ones(len(tries[~np.isnan(tries)]))
    x3 = 2 * np.ones(len(misses[~np.isnan(misses)]))
    y1 = hits[~np.isnan(hits)]
    y2 = tries[~np.isnan(tries)]
    y3 = misses[~np.isnan(misses)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)

    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel("Map Shift (cm)", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ['Hit', 'Try', 'Run']
    x_pos = np.arange(len(objects))
    ax.axhline(y=0, linewidth=3, color="black")

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0)
    ax.errorbar(x_pos[0], np.nanmean(hits), yerr=stats.sem(hits, nan_policy='omit'), ecolor='green', capsize=10, fmt="o", color="green", linewidth=3)
    ax.bar(x_pos[0], np.nanmean(hits), edgecolor="green", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[1], np.nanmean(tries), yerr=stats.sem(tries, nan_policy='omit'), ecolor='orange', capsize=20, fmt="o", color="orange")
    ax.bar(x_pos[1], np.nanmean(tries), edgecolor="orange", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[2], np.nanmean(misses), yerr=stats.sem(misses, nan_policy='omit'), ecolor='red', capsize=10, fmt="o", color="red", linewidth=3)
    ax.bar(x_pos[2], np.nanmean(misses), edgecolor="red", color="None", facecolor="None", linewidth=3, width=0.5)
    #ax.errorbar(x_pos[2], np.nanmean(diff), yerr=stats.sem(diff, nan_policy='omit'), ecolor='black', capsize=10, fmt="o", color="black", linewidth=3)
    #ax.bar(x_pos[2], np.nanmean(diff), edgecolor="black", color="None", facecolor="None", linewidth=3)
    ax.plot(x_pos, [np.nanmean(hits), np.nanmean(tries), np.nanmean(misses)], color="black", linestyle="solid", linewidth=2)

    bad_hm = ~np.logical_or(np.isnan(hits), np.isnan(misses))
    bad_ht = ~np.logical_or(np.isnan(hits), np.isnan(tries))
    bad_tm = ~np.logical_or(np.isnan(tries), np.isnan(misses))
    hit_miss_p = stats.wilcoxon(np.compress(bad_hm, hits), np.compress(bad_hm, misses))[1]
    hit_try_p = stats.wilcoxon(np.compress(bad_ht, hits), np.compress(bad_ht, tries))[1]
    try_miss_p = stats.wilcoxon(np.compress(bad_tm, tries), np.compress(bad_tm, misses))[1]

    all_behaviour = []; all_behaviour.extend(hits.tolist()); all_behaviour.extend(misses.tolist())
    #significance_bar(start=x_pos[0], end=x_pos[1], height=0.28, displaystring=get_p_text(hit_try_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=0.26, displaystring=get_p_text(try_miss_p))
    significance_bar(start=x_pos[0], end=x_pos[2], height=45, displaystring=get_p_text(hit_miss_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=0, top=50)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_bar_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()


def plot_avg_correlation_comparison(combined_df, save_path, CT="", PDN="", tt="", get_lomb_classifier=True):

    if get_lomb_classifier:
        combined_df = add_lomb_classifier(combined_df)
    if CT=="G":
        grid_cells = combined_df[combined_df["classifier"] == "G"]
    elif CT=="NG":
        grid_cells = combined_df[combined_df["classifier"] != "G"]

    if PDN == "PD":
        grid_cells = grid_cells[(grid_cells["Lomb_classifier_"] == "Position") |
                                (grid_cells["Lomb_classifier_"] == "Distance")]
    elif PDN != "":
        grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == PDN]

    hits = get_avg_correlations(grid_cells, hmt="hit", tt=tt)
    misses = get_avg_correlations(grid_cells, hmt="miss", tt=tt)
    tries = get_avg_correlations(grid_cells, hmt="try", tt=tt)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlabel("Avg R", fontsize=30, labelpad=10)
    ax.set_ylabel("Cumulative Density", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=4)
    _, _, patchesP = ax.hist(hits[~np.isnan(hits)], bins=500, color="green", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesA = ax.hist(misses[~np.isnan(misses)], bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesS = ax.hist(tries[~np.isnan(tries)], bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=2)
    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    patchesS[0].set_xy(patchesS[0].get_xy()[:-1])
    ax.set_ylim([0,1])
    #ax.set_xlim([0,0.3])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_cumhist_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()

    x1 = 0 * np.ones(len(hits[~np.isnan(hits)]))
    x2 = 1 * np.ones(len(tries[~np.isnan(tries)]))
    x3 = 2 * np.ones(len(misses[~np.isnan(misses)]))
    y1 = hits[~np.isnan(hits)]
    y2 = tries[~np.isnan(tries)]
    y3 = misses[~np.isnan(misses)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)

    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel("Avg R", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ['Hit', 'Try', 'Run']
    x_pos = np.arange(len(objects))
    ax.axhline(y=0, linewidth=3, color="black")

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0)
    ax.errorbar(x_pos[0], np.nanmean(hits), yerr=stats.sem(hits, nan_policy='omit'), ecolor='green', capsize=10, fmt="o", color="green", linewidth=3)
    ax.bar(x_pos[0], np.nanmean(hits), edgecolor="green", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[1], np.nanmean(tries), yerr=stats.sem(tries, nan_policy='omit'), ecolor='orange', capsize=20, fmt="o", color="orange")
    ax.bar(x_pos[1], np.nanmean(tries), edgecolor="orange", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[2], np.nanmean(misses), yerr=stats.sem(misses, nan_policy='omit'), ecolor='red', capsize=10, fmt="o", color="red", linewidth=3)
    ax.bar(x_pos[2], np.nanmean(misses), edgecolor="red", color="None", facecolor="None", linewidth=3, width=0.5)
    #ax.errorbar(x_pos[2], np.nanmean(diff), yerr=stats.sem(diff, nan_policy='omit'), ecolor='black', capsize=10, fmt="o", color="black", linewidth=3)
    #ax.bar(x_pos[2], np.nanmean(diff), edgecolor="black", color="None", facecolor="None", linewidth=3)
    ax.plot(x_pos, [np.nanmean(hits), np.nanmean(tries), np.nanmean(misses)], color="black", linestyle="solid", linewidth=2)

    bad_hm = ~np.logical_or(np.isnan(hits), np.isnan(misses))
    bad_ht = ~np.logical_or(np.isnan(hits), np.isnan(tries))
    bad_tm = ~np.logical_or(np.isnan(tries), np.isnan(misses))
    hit_miss_p = stats.wilcoxon(np.compress(bad_hm, hits), np.compress(bad_hm, misses))[1]
    hit_try_p = stats.wilcoxon(np.compress(bad_ht, hits), np.compress(bad_ht, tries))[1]
    try_miss_p = stats.wilcoxon(np.compress(bad_tm, tries), np.compress(bad_tm, misses))[1]

    all_behaviour = []; all_behaviour.extend(hits.tolist()); all_behaviour.extend(misses.tolist())
    #significance_bar(start=x_pos[0], end=x_pos[1], height=0.28, displaystring=get_p_text(hit_try_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=0.26, displaystring=get_p_text(try_miss_p))
    significance_bar(start=x_pos[0], end=x_pos[2], height=0.45, displaystring=get_p_text(hit_miss_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=-0.25, top=0.5)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_bar_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()


def plot_SNR_comparison(combined_df, save_path, CT="", PDN="", tt="", get_lomb_classifier=True):

    if get_lomb_classifier:
        combined_df = add_lomb_classifier(combined_df)
    if CT=="G":
        grid_cells = combined_df[combined_df["classifier"] == "G"]
    elif CT=="NG":
        grid_cells = combined_df[combined_df["classifier"] != "G"]

    if PDN == "PD":
        grid_cells = grid_cells[(grid_cells["Lomb_classifier_"] == "Position") |
                                (grid_cells["Lomb_classifier_"] == "Distance")]
    elif PDN != "":
        grid_cells = grid_cells[grid_cells["Lomb_classifier_"] == PDN]

    hits = np.asarray(grid_cells[hmt2collumn(hmt="hit", tt=tt)], dtype=np.float64)
    misses = np.asarray(grid_cells[hmt2collumn(hmt="miss", tt=tt)], dtype=np.float64)
    tries = np.asarray(grid_cells[hmt2collumn(hmt="try", tt=tt)], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlabel("Periodic Power", fontsize=30, labelpad=10)
    ax.set_ylabel("Cumulative Density", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.locator_params(axis='y', nbins=6)
    plt.locator_params(axis='x', nbins=4)
    _, _, patchesP = ax.hist(hits[~np.isnan(hits)], bins=500, color="green", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesA = ax.hist(misses[~np.isnan(misses)], bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesS = ax.hist(tries[~np.isnan(tries)], bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=2)
    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    patchesS[0].set_xy(patchesS[0].get_xy()[:-1])
    ax.set_ylim([0,1])
    ax.set_xlim([0,0.3])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_cumhist_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()

    x1 = 0 * np.ones(len(hits[~np.isnan(hits)]))
    x2 = 1 * np.ones(len(tries[~np.isnan(tries)]))
    x3 = 2 * np.ones(len(misses[~np.isnan(misses)]))
    y1 = hits[~np.isnan(hits)]
    y2 = tries[~np.isnan(tries)]
    y3 = misses[~np.isnan(misses)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)

    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel("Periodic Power", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ['Hit', 'Try', 'Run']
    x_pos = np.arange(len(objects))
    #ax.axhline(y=0, linewidth=3, color="black")

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=0)
    ax.errorbar(x_pos[0], np.nanmean(hits), yerr=stats.sem(hits, nan_policy='omit'), ecolor='green', capsize=10, fmt="o", color="green", linewidth=3)
    ax.bar(x_pos[0], np.nanmean(hits), edgecolor="green", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[1], np.nanmean(tries), yerr=stats.sem(tries, nan_policy='omit'), ecolor='orange', capsize=20, fmt="o", color="orange")
    ax.bar(x_pos[1], np.nanmean(tries), edgecolor="orange", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[2], np.nanmean(misses), yerr=stats.sem(misses, nan_policy='omit'), ecolor='red', capsize=10, fmt="o", color="red", linewidth=3)
    ax.bar(x_pos[2], np.nanmean(misses), edgecolor="red", color="None", facecolor="None", linewidth=3, width=0.5)
    #ax.errorbar(x_pos[2], np.nanmean(diff), yerr=stats.sem(diff, nan_policy='omit'), ecolor='black', capsize=10, fmt="o", color="black", linewidth=3)
    #ax.bar(x_pos[2], np.nanmean(diff), edgecolor="black", color="None", facecolor="None", linewidth=3)
    ax.plot(x_pos, [np.nanmean(hits), np.nanmean(tries), np.nanmean(misses)], color="black", linestyle="solid", linewidth=2)

    bad_hm = ~np.logical_or(np.isnan(hits), np.isnan(misses))
    bad_ht = ~np.logical_or(np.isnan(hits), np.isnan(tries))
    bad_tm = ~np.logical_or(np.isnan(tries), np.isnan(misses))
    hit_miss_p = stats.wilcoxon(np.compress(bad_hm, hits), np.compress(bad_hm, misses))[1]
    hit_try_p = stats.wilcoxon(np.compress(bad_ht, hits), np.compress(bad_ht, tries))[1]
    try_miss_p = stats.wilcoxon(np.compress(bad_tm, tries), np.compress(bad_tm, misses))[1]

    all_behaviour = []; all_behaviour.extend(hits.tolist()); all_behaviour.extend(misses.tolist())
    #significance_bar(start=x_pos[0], end=x_pos[1], height=0.28, displaystring=get_p_text(hit_try_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=0.26, displaystring=get_p_text(try_miss_p))
    significance_bar(start=x_pos[0], end=x_pos[2], height=0.3, displaystring=get_p_text(hit_miss_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=0, top=0.31)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"MOVING_LOMB_bar_"+CT+"_"+PDN+".png", dpi=300)
    plt.close()


def significance_bar(start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 15,color = 'k'):
    # draw a line with downticks at the ends
    plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)
    # draw the text with a bounding box covering up the line
    plt.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize)


def scatter_hist(x, y, ax, bin_width_x, bin_width_y,
                 lim_x1, lim_x2, lim_y1, lim_y2, color="black", alpha=1):

    # the scatter plot:
    ax.scatter(x, y, color=color, alpha=alpha)
    # now determine nice limits by hand:
    bins_x = np.arange(lim_x1, lim_x2 + bin_width_x, bin_width_x)
    bins_y = np.arange(lim_y1, lim_y2 + bin_width_y, bin_width_y)
    #ax_histx.hist(x, bins=bins_x, color=color, alpha=alpha, density=True)
    #ax_histy.hist(y, bins=bins_y, orientation='horizontal', color=color, alpha=alpha, density=True)

def add_celltype_classifier(df_shuffle, spike_data):
    spike_data = add_lomb_classifier(spike_data)
    classifiers=[]
    lomb_classifiers=[]
    for index, row in df_shuffle.iterrows():
        cluster_id = row["cluster_id"]
        session_id = row["session_id"]
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id) &
                                        (spike_data["session_id"] == session_id)]
        if len(cluster_spike_data) == 1:
            classifier = cluster_spike_data["classifier"].iloc[0]
            lomb_classifier = cluster_spike_data["Lomb_classifier_"].iloc[0]
            classifiers.append(classifier)
            lomb_classifiers.append(lomb_classifier)
        else:
            classifiers.append(np.nan)
            lomb_classifiers.append(np.nan)
            print("stop here")

    df_shuffle["classifier"] = classifiers
    df_shuffle["Lomb_classifier_"] = lomb_classifiers
    return df_shuffle


def main():
    print('-------------------------------------------------------------')
    combined_df = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")
    combined_df_shuffle = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8_lomb_shuffle.pkl")

    #combined_df = combined_df[combined_df["track_length"] == 200]
    combined_df_shuffle = combined_df_shuffle[combined_df_shuffle["track_length"] == 200]

    add_celltype_classifier(combined_df_shuffle, combined_df)
    #plot_spatial_info_hist(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    #plot_spatial_info_spatial_info_hist(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    #plot_spatial_info_cum_hist(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    #plot_cumhist_hmt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    combined_df_M11 = combined_df[combined_df["mouse"] == "M11"]
    combined_df_M14 = combined_df[combined_df["mouse"] == "M14"]

    # compare hit miss try according to lomb classifications using the pairwise trial correlations
    plot_spatial_info_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/si/G/", CT="G", PDN="", tt="all")
    plot_spatial_info_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/si/NG/", CT="NG", PDN="", tt="all")

    # compare hit miss try according to lomb classifications using the pairwise trial correlations
    plot_pairwise_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/pairwise/G/Position/", CT="G", PDN="Position")
    plot_pairwise_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/pairwise/G/Distance/", CT="G", PDN="Distance")
    plot_pairwise_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/pairwise/G/Null/", CT="G", PDN="Null")
    plot_pairwise_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/pairwise/NG/Position/", CT="NG", PDN="Position")
    plot_pairwise_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/pairwise/NG/Null/", CT="NG", PDN="Null")
    plot_pairwise_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/pairwise/NG/Distance/", CT="NG", PDN="Distance")

    plot_avg_correlation_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/avg_correlation/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1)
    plot_avg_correlation_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/avg_correlation/G/Distance/nonbeaconed/", CT="G", PDN="Distance", tt=1)
    plot_avg_correlation_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/avg_correlation/G/Position/hit/", CT="G", PDN="Position", hmt="hit")
    plot_avg_correlation_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/avg_correlation/G/Distance/hit/", CT="G", PDN="Distance", hmt="hit")

    plot_map_shifts_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/map_shifts/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1)
    plot_map_shifts_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/map_shifts/G/Distance/nonbeaconed/", CT="G", PDN="Distance", tt=1)
    plot_map_shifts_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/map_shifts/G/Position/hit/", CT="G", PDN="Position", hmt="hit")
    plot_map_shifts_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/map_shifts/G/Distance/hit/", CT="G", PDN="Distance", hmt="hit")

# compare hit miss try according to lomb classifications using the pairwise trial correlations
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/", CT="G", PDN="", tt="all")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/", CT="NG", PDN="", tt="all")

    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Position/", CT="G", PDN="Position", tt="all")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Position/beaconed/", CT="G", PDN="Position", tt="beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Position/nonbeaconed/", CT="G", PDN="Position", tt="non_beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Position/probe/", CT="G", PDN="Position", tt="probe")

    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Distance/", CT="G", PDN="Distance", tt="all")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Distance/beaconed/", CT="G", PDN="Distance", tt="beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Distance/nonbeaconed/", CT="G", PDN="Distance", tt="non_beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Distance/probe/", CT="G", PDN="Distance", tt="probe")

    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Null/", CT="G", PDN="Null", tt="all")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Null/beaconed/", CT="G", PDN="Null", tt="beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Null/nonbeaconed/", CT="G", PDN="Null", tt="non_beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/G/Null/probe/", CT="G", PDN="Null", tt="probe")

    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Position/", CT="NG", PDN="Position", tt="all")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Position/beaconed/", CT="NG", PDN="Position", tt="beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Position/nonbeaconed/", CT="NG", PDN="Position", tt="non_beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Position/probe/", CT="NG", PDN="Position", tt="probe")

    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Null/", CT="NG", PDN="Null", tt="all")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Null/beaconed/", CT="NG", PDN="Null", tt="beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Null/nonbeaconed/", CT="NG", PDN="Null", tt="non_beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Null/probe/", CT="NG", PDN="Null", tt="probe")

    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Distance/", CT="NG", PDN="Distance", tt="all")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Distance/beaconed/", CT="NG", PDN="Distance", tt="beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Distance/nonbeaconed/", CT="NG", PDN="Distance", tt="non_beaconed")
    plot_SNR_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/snr/NG/Distance/probe/", CT="NG", PDN="Distance", tt="probe")


    # compare hit miss try according to shuffle lomb classifications using the power
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/", CT="G", PDN="", tt="all")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/", CT="NG", PDN="", tt="all")


    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Position/", CT="G", PDN="Position", get_lomb_classifier=False, tt="all")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Position/beaconed/", CT="G", PDN="Position", get_lomb_classifier=False, tt="beaconed")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Position/nonbeaconed/", CT="G", PDN="Position", get_lomb_classifier=False, tt="non_beaconed")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Position/probe/", CT="G", PDN="Position", get_lomb_classifier=False, tt="probe")

    #plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Distance/", CT="G", PDN="Distance", get_lomb_classifier=False, tt="all")
    #plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Distance/beaconed/", CT="G", PDN="Distance", get_lomb_classifier=False, tt="beaconed")
    #plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Distance/nonbeaconed/", CT="G", PDN="Distance", get_lomb_classifier=False, tt="non_beaconed")
    #plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Distance/probe/", CT="G", PDN="Distance", get_lomb_classifier=False, tt="probe")

    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Null/", CT="G", PDN="Null", get_lomb_classifier=False, tt="all")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Null/beaconed/", CT="G", PDN="Null", get_lomb_classifier=False, tt="beaconed")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Null/nonbeaconed/", CT="G", PDN="Null", get_lomb_classifier=False, tt="non_beaconed")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/G/Null/probe/", CT="G", PDN="Null", get_lomb_classifier=False, tt="probe")

    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Position/", CT="NG", PDN="Position", get_lomb_classifier=False, tt="all")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Position/beaconed/", CT="NG", PDN="Position", get_lomb_classifier=False, tt="beaconed")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Position/nonbeaconed/", CT="NG", PDN="Position", get_lomb_classifier=False, tt="non_beaconed")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Position/probe/", CT="NG", PDN="Position", get_lomb_classifier=False, tt="probe")

    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Null/", CT="NG", PDN="Null", get_lomb_classifier=False, tt="all")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Null/beaconed/", CT="NG", PDN="Null", get_lomb_classifier=False, tt="beaconed")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Null/nonbeaconed/", CT="NG", PDN="Null", get_lomb_classifier=False, tt="non_beaconed")
    plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Null/probe/", CT="NG", PDN="Null", get_lomb_classifier=False, tt="probe")

    #plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Distance/", CT="NG", PDN="Distance", get_lomb_classifier=False, tt="all")
    #plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Distance/beaconed/", CT="NG", PDN="Distance", get_lomb_classifier=False, tt="beaconed")
    #plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Distance/nonbeaconed/", CT="NG", PDN="Distance", get_lomb_classifier=False, tt="non_beaconed")
    #plot_SNR_comparison(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/shuffles/snr/NG/Distance/probe/", CT="NG", PDN="Distance", get_lomb_classifier=False, tt="probe")

    # now by trial_type specificially
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/", CT="G", PDN="", hmt="hit")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/", CT="G", PDN="", hmt="hit")

    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Position/", CT="G", PDN="Position", hmt="all")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Position/hit/", CT="G", PDN="Position", hmt="hit")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Position/try/", CT="G", PDN="Position", hmt="try")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Position/miss/", CT="G", PDN="Position", hmt="miss")

    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Distance/", CT="G", PDN="Distance", hmt="all")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Distance/hit/", CT="G", PDN="Distance", hmt="hit")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Distance/try/", CT="G", PDN="Distance", hmt="try")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Distance/miss/", CT="G", PDN="Distance", hmt="miss")

    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Null/", CT="G", PDN="Null", hmt="all")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Null/hit/", CT="G", PDN="Null", hmt="hit")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Null/try/", CT="G", PDN="Null", hmt="try")
    #plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/G/Null/miss/", CT="G", PDN="Null", hmt="miss")

    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Position/", CT="NG", PDN="Position", hmt="all")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Position/hit/", CT="NG", PDN="Position", hmt="hit")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Position/try/", CT="NG", PDN="Position", hmt="try")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Position/miss/", CT="NG", PDN="Position", hmt="miss")

    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Null/", CT="NG", PDN="Null", hmt="all")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Null/hit/", CT="NG", PDN="Null", hmt="hit")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Null/try/", CT="NG", PDN="Null", hmt="try")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Null/miss/", CT="NG", PDN="Null", hmt="miss")

    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Distance/", CT="NG", PDN="Distance", hmt="all")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Distance/hit/", CT="NG", PDN="Distance", hmt="hit")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Distance/try/", CT="NG", PDN="Distance", hmt="try")
    plot_SNR_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/snr/NG/Distance/miss/", CT="NG", PDN="Distance", hmt="miss")


    # compare hit miss try according to shuffle lomb classifications using the power
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/", CT="G", PDN="", hmt="hit")
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/", CT="NG", PDN="", hmt="hit")

    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Position/", CT="G", PDN="Position", hmt="all", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Position/hit/", CT="G", PDN="Position", hmt="hit", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Position/try/", CT="G", PDN="Position", hmt="try", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Position/miss/", CT="G", PDN="Position", hmt="miss", get_lomb_classifier=False)

    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Distance/", CT="G", PDN="Distance", hmt="all", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Distance/hit/", CT="G", PDN="Distance", hmt="hit", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Distance/try/", CT="G", PDN="Distance", hmt="try", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Distance/miss/", CT="G", PDN="Distance", hmt="miss", get_lomb_classifier=False)

    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Null/", CT="G", PDN="Null", hmt="all", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Null/hit/", CT="G", PDN="Null", hmt="hit", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Null/try/", CT="G", PDN="Null", hmt="try", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/G/Null/miss/", CT="G", PDN="Null", hmt="miss", get_lomb_classifier=False)

    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Position/", CT="NG", PDN="Position", hmt="all", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Position/hit/", CT="NG", PDN="Position", hmt="hit", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Position/try/", CT="NG", PDN="Position", hmt="try", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Position/miss/", CT="NG", PDN="Position", hmt="miss", get_lomb_classifier=False)

    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Null/", CT="NG", PDN="Null", hmt="all", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Null/hit/", CT="NG", PDN="Null", hmt="hit",  get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Null/try/", CT="NG", PDN="Null", hmt="try", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Null/miss/", CT="NG", PDN="Null", hmt="miss", get_lomb_classifier=False)

    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Distance/", CT="NG", PDN="Distance", hmt="all", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Distance/hit/", CT="NG", PDN="Distance", hmt="hit", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Distance/try/", CT="NG", PDN="Distance", hmt="try", get_lomb_classifier=False)
    plot_SNR_comparison_tt(combined_df_shuffle, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/shuffles/snr/NG/Distance/miss/", CT="NG", PDN="Distance", hmt="miss", get_lomb_classifier=False)

    print("look now")

if __name__ == '__main__':
    main()
