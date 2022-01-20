import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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

def get_n_fields_indices(hmt, tt):
    i = tt
    if hmt=="hit":
        j = 0
    elif hmt=="miss":
        j = 1
    elif hmt=="try":
        j = 2
    return i, j


def get_field_jitters(spike_data, hmt, tt, pre_post_rz=""):
    i, j = get_n_fields_indices(hmt, tt) # i is for tt and j is for hmt

    field_jitter_all_data = []
    for index, cluster_data in spike_data.iterrows():
        cluster_data = cluster_data.to_frame().T.reset_index(drop=True)
        field_jitter_collumn = "fields_jitter_hmt_by_trial_type"+pre_post_rz
        field_jitter = cluster_data[field_jitter_collumn].iloc[0][i][j]
        field_jitter_all_data.append(field_jitter)

    return np.array(field_jitter_all_data)

def get_field_sizes(spike_data, hmt, tt, pre_post_rz=""):
    i, j = get_n_fields_indices(hmt, tt) # i is for tt and j is for hmt

    field_sizes_all_data = []
    for index, cluster_data in spike_data.iterrows():
        cluster_data = cluster_data.to_frame().T.reset_index(drop=True)
        field_size_collumn = "fields_sizes_hmt_by_trial_type"+pre_post_rz
        field_size = cluster_data[field_size_collumn].iloc[0][i][j]
        field_sizes_all_data.append(field_size)

    return np.array(field_sizes_all_data)

def get_n_fields(spike_data, hmt, tt, pre_post_rz=""):
    i, j = get_n_fields_indices(hmt, tt) # i is for tt and j is for hmt

    n_fields_all_data = []
    for index, cluster_data in spike_data.iterrows():
        cluster_data = cluster_data.to_frame().T.reset_index(drop=True)
        n_fields_collumn = "fields_per_trial_hmt_by_trial_type"+pre_post_rz
        n_fields = cluster_data[n_fields_collumn].iloc[0][i][j]
        n_fields_all_data.append(n_fields)

    return np.array(n_fields_all_data)

def plot_field_size_comparison_tt(combined_df, save_path, CT="", PDN="", hmt="", pre_post_rz="", get_lomb_classifier=True):

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

    b = get_field_sizes(grid_cells, hmt="hit", tt=0, pre_post_rz=pre_post_rz)
    nb = get_field_sizes(grid_cells, hmt="hit", tt=1, pre_post_rz=pre_post_rz)
    bad_bnb = ~np.logical_or(np.isnan(b), np.isnan(nb))
    b = np.compress(bad_bnb, b)
    nb = np.compress(bad_bnb, nb)

    x1 = 0 * np.ones(len(b[~np.isnan(b)]))
    x2 = 1 * np.ones(len(nb[~np.isnan(nb)]))
    y1 = b[~np.isnan(b)]
    y2 = nb[~np.isnan(nb)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel("Field Size (cm)", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ['Cued', 'PI']
    x_pos = np.arange(len(objects))
    for i in range(len(b)):
        ax.plot(x_pos, [b[i], nb[i]], color="black", alpha=0.1)

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=1)
    ax.errorbar(x_pos[0], np.nanmean(b), yerr=stats.sem(b, nan_policy='omit'), ecolor='black', capsize=10, fmt="o", color="black", linewidth=3)
    ax.bar(x_pos[0], np.nanmean(b), edgecolor="black", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[1], np.nanmean(nb), yerr=stats.sem(nb, nan_policy='omit'), ecolor='blue', capsize=20, fmt="o", color="blue")
    ax.bar(x_pos[1], np.nanmean(nb), edgecolor="blue", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.plot(x_pos, [np.nanmean(b), np.nanmean(nb)], color="black", linestyle="solid", linewidth=2)

    bnb_p = stats.wilcoxon(b,nb)[1]

    significance_bar(start=x_pos[0], end=x_pos[1], height=60, displaystring=get_p_text(bnb_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=0, top=65)
    plt.locator_params(axis='y', nbins=5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"field_sizes_bar_"+CT+"_"+PDN+pre_post_rz+".png", dpi=300)
    plt.close()


def plot_nfields_comparison_tt(combined_df, save_path, CT="", PDN="", hmt="", pre_post_rz="", get_lomb_classifier=True):

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

    b = get_n_fields(grid_cells, hmt="hit", tt=0, pre_post_rz=pre_post_rz)
    nb = get_n_fields(grid_cells, hmt="hit", tt=1, pre_post_rz=pre_post_rz)
    bad_bnb = ~np.logical_or(np.isnan(b), np.isnan(nb))
    b = np.compress(bad_bnb, b)
    nb = np.compress(bad_bnb, nb)

    x1 = 0 * np.ones(len(b[~np.isnan(b)]))
    x2 = 1 * np.ones(len(nb[~np.isnan(nb)]))
    y1 = b[~np.isnan(b)]
    y2 = nb[~np.isnan(nb)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel("AVG Fields/Trial", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ['Cued', 'PI']
    x_pos = np.arange(len(objects))
    #ax.axhline(y=0, linewidth=3, color="black")

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=1)
    ax.errorbar(x_pos[0], np.nanmean(b), yerr=stats.sem(b, nan_policy='omit'), ecolor='black', capsize=10, fmt="o", color="black", linewidth=3)
    ax.bar(x_pos[0], np.nanmean(b), edgecolor="black", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[1], np.nanmean(nb), yerr=stats.sem(nb, nan_policy='omit'), ecolor='blue', capsize=20, fmt="o", color="blue")
    ax.bar(x_pos[1], np.nanmean(nb), edgecolor="blue", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.plot(x_pos, [np.nanmean(b), np.nanmean(nb)], color="black", linestyle="solid", linewidth=2)

    for i in range(len(b)):
        ax.plot(x_pos, [b[i], nb[i]], color="black", alpha=0.1)

    bnb_p = stats.wilcoxon(b, nb)[1]

    significance_bar(start=x_pos[0], end=x_pos[1], height=5, displaystring=get_p_text(bnb_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=0, top=5)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"fields_per_trial_bar_"+CT+"_"+PDN+pre_post_rz+".png", dpi=300)
    plt.close()


def plot_field_jitter_comparison_tt(combined_df, save_path, CT="", PDN="", hmt="", pre_post_rz="", get_lomb_classifier=True):

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

    b = get_field_jitters(grid_cells, hmt="hit", tt=0, pre_post_rz=pre_post_rz)
    nb = get_field_jitters(grid_cells, hmt="hit", tt=1, pre_post_rz=pre_post_rz)
    bad_bnb = ~np.logical_or(np.isnan(b), np.isnan(nb))
    b = np.compress(bad_bnb, b)
    nb = np.compress(bad_bnb, nb)

    x1 = 0 * np.ones(len(b[~np.isnan(b)]))
    x2 = 1 * np.ones(len(nb[~np.isnan(nb)]))
    y1 = b[~np.isnan(b)]
    y2 = nb[~np.isnan(nb)]
    #Combine the sampled data together
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]
    open_circle = mpl.path.Path(vert)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel("Jitter (cm)", fontsize=30, labelpad=10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    objects = ['Cued', 'PI']
    x_pos = np.arange(len(objects))

    bnb_p = stats.wilcoxon(b, nb)[1]

    sns.stripplot(x, y, ax=ax, color="black", marker=open_circle, linewidth=.001, zorder=1)
    ax.errorbar(x_pos[0], np.nanmean(b), yerr=stats.sem(b, nan_policy='omit'), ecolor='black', capsize=10, fmt="o", color="black", linewidth=3)
    ax.bar(x_pos[0], np.nanmean(b), edgecolor="black", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.errorbar(x_pos[1], np.nanmean(nb), yerr=stats.sem(nb, nan_policy='omit'), ecolor='blue', capsize=20, fmt="o", color="blue")
    ax.bar(x_pos[1], np.nanmean(nb), edgecolor="blue", color="None", facecolor="None", linewidth=3, width=0.5)

    ax.plot(x_pos, [np.nanmean(b), np.nanmean(nb)], color="black", linestyle="solid", linewidth=2)

    for i in range(len(b)):
        ax.plot(x_pos, [b[i], nb[i]], color="black", alpha=0.1)

    offset = 60
    significance_bar(start=x_pos[0], end=x_pos[1], height=0+offset, displaystring=get_p_text(bnb_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=0, top=offset)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"field_jitter_bar_"+CT+"_"+PDN+pre_post_rz+".png", dpi=300)
    plt.close()

def convert_trial_type_marker(str_or_int):
    if isinstance(str_or_int, str):
        if str_or_int == "beaconed":
            return 0
        elif str_or_int == "non_beaconed":
            return 1
        elif str_or_int == "probe":
            return 2
    elif isinstance(str_or_int, int):
        if str_or_int == 0:
            return "beaconed"
        elif str_or_int == 1:
            return "non_beaconed"
        elif str_or_int == 2:
            return "probe"

def plot_correlation_of_hit_miss_diff_with_jitter(combined_df, save_path, CT="", PDN="", pre_post_rz="", tt="", get_lomb_classifier=True):
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

    hits = get_field_jitters(grid_cells, hmt="hit", tt=tt)
    misses = get_field_jitters(grid_cells, hmt="miss", tt=tt)
    jitter_difference = hits-misses
    hits = np.asarray(grid_cells[hmt2collumn(hmt="hit", tt=convert_trial_type_marker(tt))], dtype=np.float64)
    misses = np.asarray(grid_cells[hmt2collumn(hmt="miss", tt=convert_trial_type_marker(tt))], dtype=np.float64)
    power_difference = hits-misses

    bad = ~np.logical_or(np.isnan(power_difference), np.isnan(jitter_difference))
    pearson_r = stats.pearsonr(np.compress(bad, power_difference).flatten(),np.compress(bad, jitter_difference).flatten())

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(np.compress(bad, power_difference).flatten().reshape(-1, 1),np.compress(bad, jitter_difference).flatten().reshape(-1, 1))  # pe

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel("Hit-Miss Jitter", fontsize=30, labelpad=10)
    ax.set_xlabel("Hit-Miss Power", fontsize=30, labelpad=10)
    ax.scatter(np.compress(bad, power_difference).flatten(),np.compress(bad, jitter_difference).flatten(), marker="o", color="black")
    test_x = np.linspace(np.nanmin(power_difference), np.nanmax(power_difference), 100)
    y = linear_regressor.predict(test_x.reshape(-1, 1))
    ax.plot(test_x, y, linestyle="solid", color="red")
    #ax.text(x=1, y=12, s="R = "+str(np.round(pearson_r[0], decimals=2)), color="black", fontsize=20)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #plt.xticks(x_pos, objects, fontsize=30)
    #plt.xlim((-0.5, len(objects)-0.5))
    #ax.set_ylim(bottom=0, top=0.31)
    #plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"jitter_vs_power_diff"+CT+"_"+PDN+pre_post_rz+".png", dpi=300)
    plt.close()



def plot_field_jitter_comparison(combined_df, save_path, CT="", PDN="", tt="", pre_post_rz="", get_lomb_classifier=True):

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

    hits = get_field_jitters(grid_cells, hmt="hit", tt=tt, pre_post_rz=pre_post_rz)
    misses = get_field_jitters(grid_cells, hmt="miss", tt=tt, pre_post_rz=pre_post_rz)
    tries = get_field_jitters(grid_cells, hmt="try", tt=tt, pre_post_rz=pre_post_rz)

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
    ax.set_ylabel("Jitter (cm)", fontsize=30, labelpad=10)
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
    ax.plot(x_pos, [np.nanmean(hits), np.nanmean(tries), np.nanmean(misses)], color="black", linestyle="solid", linewidth=2)

    bad_hm = ~np.logical_or(np.isnan(hits), np.isnan(misses))
    bad_ht = ~np.logical_or(np.isnan(hits), np.isnan(tries))
    bad_tm = ~np.logical_or(np.isnan(tries), np.isnan(misses))
    hit_miss_p = stats.wilcoxon(np.compress(bad_hm, hits), np.compress(bad_hm, misses))[1]
    hit_try_p = stats.wilcoxon(np.compress(bad_ht, hits), np.compress(bad_ht, tries))[1]
    try_miss_p = stats.wilcoxon(np.compress(bad_tm, tries), np.compress(bad_tm, misses))[1]
    offset=60
    all_behaviour = []; all_behaviour.extend(hits.tolist()); all_behaviour.extend(misses.tolist())
    significance_bar(start=x_pos[0], end=x_pos[1], height=offset-5, displaystring=get_p_text(hit_try_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=0.26, displaystring=get_p_text(try_miss_p))
    significance_bar(start=x_pos[0], end=x_pos[2], height=offset, displaystring=get_p_text(hit_miss_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=0, top=offset)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"field_jitters_bar_"+CT+"_"+PDN+pre_post_rz+".png", dpi=300)
    plt.close()

def plot_field_size_comparison(combined_df, save_path, CT="", PDN="", tt="", pre_post_rz="", get_lomb_classifier=True):

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

    hits = get_field_sizes(grid_cells, hmt="hit", tt=tt, pre_post_rz=pre_post_rz)
    misses = get_field_sizes(grid_cells, hmt="miss", tt=tt, pre_post_rz=pre_post_rz)
    tries = get_field_sizes(grid_cells, hmt="try", tt=tt, pre_post_rz=pre_post_rz)

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
    ax.set_ylabel("Field Size (cm)", fontsize=30, labelpad=10)
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
    ax.plot(x_pos, [np.nanmean(hits), np.nanmean(tries), np.nanmean(misses)], color="black", linestyle="solid", linewidth=2)

    bad_hm = ~np.logical_or(np.isnan(hits), np.isnan(misses))
    bad_ht = ~np.logical_or(np.isnan(hits), np.isnan(tries))
    bad_tm = ~np.logical_or(np.isnan(tries), np.isnan(misses))
    hit_miss_p = stats.wilcoxon(np.compress(bad_hm, hits), np.compress(bad_hm, misses))[1]
    hit_try_p = stats.wilcoxon(np.compress(bad_ht, hits), np.compress(bad_ht, tries))[1]
    try_miss_p = stats.wilcoxon(np.compress(bad_tm, tries), np.compress(bad_tm, misses))[1]

    all_behaviour = []; all_behaviour.extend(hits.tolist()); all_behaviour.extend(misses.tolist())
    significance_bar(start=x_pos[0], end=x_pos[1], height=55, displaystring=get_p_text(hit_try_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=0.26, displaystring=get_p_text(try_miss_p))
    significance_bar(start=x_pos[0], end=x_pos[2], height=60, displaystring=get_p_text(hit_miss_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=0, top=65)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"field_sizes_bar_"+CT+"_"+PDN+pre_post_rz+".png", dpi=300)
    plt.close()

def plot_nfields_comparison(combined_df, save_path, CT="", PDN="", tt="", pre_post_rz="", get_lomb_classifier=True):

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

    hits = get_n_fields(grid_cells, hmt="hit", tt=tt, pre_post_rz=pre_post_rz)
    misses = get_n_fields(grid_cells, hmt="miss", tt=tt, pre_post_rz=pre_post_rz)
    tries = get_n_fields(grid_cells, hmt="try", tt=tt, pre_post_rz=pre_post_rz)

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
    ax.set_ylabel("AVG Fields/Trial", fontsize=30, labelpad=10)
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
    significance_bar(start=x_pos[0], end=x_pos[1], height=4.5, displaystring=get_p_text(hit_try_p))
    #significance_bar(start=x_pos[1], end=x_pos[2], height=0.26, displaystring=get_p_text(try_miss_p))
    significance_bar(start=x_pos[0], end=x_pos[2], height=5, displaystring=get_p_text(hit_miss_p))

    plt.xticks(x_pos, objects, fontsize=30)
    plt.xlim((-0.5, len(objects)-0.5))
    ax.set_ylim(bottom=0, top=6)
    plt.locator_params(axis='y', nbins=5)
    #plt.xticks(rotation=-45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"fields_per_trial_bar_"+CT+"_"+PDN+pre_post_rz+".png", dpi=300)
    plt.close()


def significance_bar(start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 15,color = 'k'):
    # draw a line with downticks at the ends
    plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)
    # draw the text with a bounding box covering up the line
    plt.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize)

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
    add_celltype_classifier(combined_df_shuffle, combined_df)

    combined_df = combined_df[combined_df["rate_map_correlation_first_vs_second_half"]>0]
    #combined_df = combined_df[combined_df["track_length"] == 200]
    #combined_df = combined_df[combined_df["power_test_hit_miss_p"] < 0.05]
    #combined_df_shuffle = combined_df_shuffle[combined_df_shuffle["track_length"] == 200]


    #plot_correlation_of_hit_miss_diff_with_jitter(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", tt=1, pre_post_rz="")


    plot_nfields_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/fields/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1, pre_post_rz="")
    plot_field_size_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/fields/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1, pre_post_rz="")
    plot_field_jitter_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/fields/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1, pre_post_rz="")

    plot_nfields_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/fields/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1, pre_post_rz="_pre_rz")
    plot_field_size_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/fields/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1, pre_post_rz="_pre_rz")
    plot_field_jitter_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/fields/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1, pre_post_rz="_pre_rz")

    plot_nfields_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/fields/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1, pre_post_rz="_post_rz")
    plot_field_size_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/fields/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1, pre_post_rz="_post_rz")
    plot_field_jitter_comparison(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/hmt/fields/G/Position/nonbeaconed/", CT="G", PDN="Position", tt=1, pre_post_rz="_post_rz")


    #compare hits across trial types
    plot_nfields_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", hmt="hit", pre_post_rz="")
    plot_field_size_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", hmt="hit", pre_post_rz="")
    plot_field_jitter_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", hmt="hit", pre_post_rz="")

    plot_nfields_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", hmt="hit", pre_post_rz="_pre_rz")
    plot_field_size_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", hmt="hit", pre_post_rz="_pre_rz")
    plot_field_jitter_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", hmt="hit", pre_post_rz="_pre_rz")

    plot_nfields_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", hmt="hit", pre_post_rz="_post_rz")
    plot_field_size_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", hmt="hit", pre_post_rz="_post_rz")
    plot_field_jitter_comparison_tt(combined_df, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/tt/fields/G/Position/", CT="G", PDN="Position", hmt="hit", pre_post_rz="_post_rz")
    print("look now")

if __name__ == '__main__':
    main()
