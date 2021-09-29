import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import settings
from numpy import inf
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from Edmond.Concatenate_from_server import *
from scipy import stats

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

def plot_pairwise_comparison_b(combined_df, save_path):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    hits = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_hit_b"])
    misses = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_miss_b"])
    tries = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_try_b"])

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel("Avg Trial-pair Pearson R", fontsize=20, labelpad=10)
    ax.set_ylabel("Cumulative Density", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    _, _, patchesP = ax.hist(hits, bins=500, color="green", histtype="step", density=True, cumulative=True, linewidth=1)
    _, _, patchesA = ax.hist(misses, bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=1)
    _, _, patchesS = ax.hist(tries, bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=1)
    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    patchesS[0].set_xy(patchesS[0].get_xy()[:-1])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title("Beaconed", fontsize=25)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_cumhist_b.png", dpi=300)
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

    plt.xticks(x_pos, objects, fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_bar_b.png", dpi=300)
    plt.close()

def plot_pairwise_comparison_p(combined_df, save_path):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    hits = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_hit_p"])
    misses = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_miss_p"])
    tries = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_try_p"])

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel("Avg Trial-pair Pearson R", fontsize=20, labelpad=10)
    ax.set_ylabel("Cumulative Density", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    _, _, patchesP = ax.hist(hits, bins=500, color="green", histtype="step", density=True, cumulative=True, linewidth=1)
    _, _, patchesA = ax.hist(misses, bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=1)
    _, _, patchesS = ax.hist(tries, bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=1)
    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    patchesS[0].set_xy(patchesS[0].get_xy()[:-1])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title("Probe", fontsize=25)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_cumhist_p.png", dpi=300)
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

    plt.xticks(x_pos, objects, fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_bar_p.png", dpi=300)
    plt.close()

def plot_pairwise_comparison_nb(combined_df, save_path):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    hits = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_hit_nb"])
    misses = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_miss_nb"])
    tries = np.asarray(grid_cells["avg_pairwise_trial_pearson_r_try_nb"])

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlabel("Avg Trial-pair Pearson R", fontsize=20, labelpad=10)
    ax.set_ylabel("Cumulative Density", fontsize=20, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    _, _, patchesP = ax.hist(hits, bins=500, color="green", histtype="step", density=True, cumulative=True, linewidth=1)
    _, _, patchesA = ax.hist(misses, bins=500, color="red", histtype="step", density=True, cumulative=True, linewidth=1)
    _, _, patchesS = ax.hist(tries, bins=500, color="orange", histtype="step", density=True, cumulative=True, linewidth=1)
    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    patchesS[0].set_xy(patchesS[0].get_xy()[:-1])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title("Non Beaconed", fontsize=25)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_cumhist_nb.png", dpi=300)
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

    plt.xticks(x_pos, objects, fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_bar_nb.png", dpi=300)
    plt.close()

def plot_pairwise_comparison(combined_df, save_path):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
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
    plt.savefig(save_path+"pairwise_cumhist.png", dpi=300)
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

    plt.xticks(x_pos, objects, fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"pairwise_bar.png", dpi=300)
    plt.close()


def scatter_hist(x, y, ax, bin_width_x, bin_width_y,
                 lim_x1, lim_x2, lim_y1, lim_y2, color="black", alpha=1):

    # the scatter plot:
    ax.scatter(x, y, color=color, alpha=alpha)
    # now determine nice limits by hand:
    bins_x = np.arange(lim_x1, lim_x2 + bin_width_x, bin_width_x)
    bins_y = np.arange(lim_y1, lim_y2 + bin_width_y, bin_width_y)
    #ax_histx.hist(x, bins=bins_x, color=color, alpha=alpha, density=True)
    #ax_histy.hist(y, bins=bins_y, orientation='horizontal', color=color, alpha=alpha, density=True)

def main():
    print('-------------------------------------------------------------')
    #vr_data = pd.read_pickle("/mnt/datastore/Harry/Cohort8_may2021/summary/All_mice_vr.pkl")
    #of_data = pd.read_pickle("/mnt/datastore/Harry/Cohort8_may2021/summary/All_mice_of.pkl")
    #combined_df = combine_of_vr_dataframes(vr_data, of_data)
    #combined_df.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")

    combined_df = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")
    plot_spatial_info_hist(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    #plot_spatial_info_spatial_info_hist(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    #plot_spatial_info_cum_hist(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    #plot_cumhist_hmt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")

    plot_pairwise_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    plot_pairwise_comparison_b(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    plot_pairwise_comparison_nb(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    plot_pairwise_comparison_p(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/")
    print("look now")

if __name__ == '__main__':
    main()
