import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import stats
import matplotlib.pylab as plt

def get_clusters_per_day(df):
    cluster_per_day = []
    for session_id in np.unique(df["session_id"]):
        session_df = df[df["session_id"] == session_id]
        cluster_per_day.append(len(session_df))
    return np.array(cluster_per_day)

def get_clusters_per_day_per_shank(df):
    cluster_per_day = []
    for session_id in np.unique(df["session_id"]):
        session_df = df[df["session_id"] == session_id]
        number_of_channels = session_df["number_of_channels"].iloc[0]
        cluster_per_day.append(len(session_df)/(number_of_channels/16))
    return np.array(cluster_per_day)

def plot_clusters_over_days(combined_df, output_path=""):
    tetrodes = combined_df[combined_df["electrode"] == "tetrode"]
    probes = combined_df[combined_df["electrode"] == "silicon_probe"]

    fig, ax = plt.subplots(1,1, figsize=(8,6))

    for mouse in np.unique(tetrodes["mouse"]):
        mouse_df = tetrodes[tetrodes["mouse"] == mouse]
        days = np.arange(1,max(mouse_df["recording_day"]))

        n_clusters = []
        for day in days:
            day_df = mouse_df[mouse_df["recording_day"] == day]
            n_clusters.append(len(day_df))
        ax.plot(days, n_clusters, color="grey", alpha=0.5)

    for mouse in np.unique(probes["mouse"]):
        mouse_df = probes[probes["mouse"] == mouse]
        days = np.arange(1, max(mouse_df["recording_day"]))

        n_clusters = []
        for day in days:
            day_df = mouse_df[mouse_df["recording_day"] == day]
            n_clusters.append(len(day_df))

        ax.plot(days, n_clusters, color="blue", alpha=0.5)

    ax.tick_params(axis='both', which='major', labelsize=20)
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([0, 100])
    #ax.set_xlabel("Encoding grid cells", fontsize=20)
    #ax.set_ylabel("Percentage hits", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/clusters_across_days.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(8,6))

    for mouse in np.unique(tetrodes["mouse"]):
        mouse_df = tetrodes[tetrodes["mouse"] == mouse]
        n_channels = mouse_df["number_of_channels"].iloc[0]
        days = np.arange(1,max(mouse_df["recording_day"]))

        n_clusters = []
        for day in days:
            day_df = mouse_df[mouse_df["recording_day"] == day]
            n_clusters.append(len(day_df)/(n_channels/16))
        ax.plot(days, n_clusters, color="grey", alpha=0.5)

    for mouse in np.unique(probes["mouse"]):
        mouse_df = probes[probes["mouse"] == mouse]
        n_channels= mouse_df["number_of_channels"].iloc[0]
        days = np.arange(1, max(mouse_df["recording_day"]))

        n_clusters = []
        for day in days:
            day_df = mouse_df[mouse_df["recording_day"] == day]
            n_clusters.append(len(day_df)/(n_channels/16))
        ax.plot(days, n_clusters, color="blue", alpha=0.5)

    ax.tick_params(axis='both', which='major', labelsize=20)
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([0, 100])
    #ax.set_xlabel("Encoding grid cells", fontsize=20)
    #ax.set_ylabel("Percentage hits", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/clusters_across_days_per_shank.png', dpi=300)
    plt.close()


def plot_spatial_metrics(combined_df, output_path=""):
    tetrodes = combined_df[combined_df["electrode"] == "tetrode"]
    probes = combined_df[combined_df["electrode"] == "silicon_probe"]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')

    #theta index
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes['ThetaIndex'].dropna(), probes['ThetaIndex'].dropna()]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/theta_index.png', dpi=300)
    plt.close()


    #hd_score
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes['hd_score'].dropna(), probes['hd_score'].dropna()]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/hd_score.png', dpi=300)
    plt.close()


    #grid_score
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes['grid_score'].dropna(), probes['grid_score'].dropna()]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/grid_score.png', dpi=300)
    plt.close()


    #spatial_information_score
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes['spatial_information_score'].dropna(), probes['spatial_information_score'].dropna()]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/spatial_information_score.png', dpi=300)
    plt.close()


    #border_score
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes['border_score'].dropna(), probes['border_score'].dropna()]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/border_score.png', dpi=300)
    plt.close()


    #mean_firing_rate
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes['mean_firing_rate_vr'].dropna(), probes['mean_firing_rate_vr'].dropna()]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/mean_firing_rate_vr.png', dpi=300)
    plt.close()


    #mean_firing_rate_of
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes['mean_firing_rate_of'].dropna(), probes['mean_firing_rate_of'].dropna()]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/mean_firing_rate_of.png', dpi=300)
    plt.close()

    #speed_score
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes['speed_score'].dropna(), probes['speed_score'].dropna()]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/speed_score.png', dpi=300)
    plt.close()

    #rate_map_correlation_first_vs_second_half
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes['rate_map_correlation_first_vs_second_half'].dropna(), probes['rate_map_correlation_first_vs_second_half'].dropna()]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/rate_map_correlation_first_vs_second_half.png', dpi=300)
    plt.close()
    return

def plot_quality_metrics(combined_df, output_path=""):
    tetrodes = combined_df[combined_df["electrode"] == "tetrode"]
    probes = combined_df[combined_df["electrode"] == "silicon_probe"]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')

    #isolation
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes["isolation"], probes["nn_isolation"]]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/isolation.png', dpi=300)
    plt.close()

    #noise_overlap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes["noise_overlap"], probes["nn_noise_overlap"]]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/noise_overlap.png', dpi=300)
    plt.close()

    #snr
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes["peak_snr"], probes["snr"]]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/snr.png', dpi=300)
    plt.close()

    #snr
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["grey", "blue"]
    data = [tetrodes["snippet_peak_to_trough"], probes["snippet_peak_to_trough"]]
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    ax.set_xlim([0, 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    plt.savefig(output_path + '/snippet_peak_to_trough.png', dpi=300)
    plt.close()

    return



def plot_power_spectral_peaks(combined_df, output_path=""):
    tetrodes = combined_df[combined_df["electrode"] == "tetrode"]
    probes = combined_df[combined_df["electrode"] == "silicon_probe"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.hist(tetrodes["peak_power_from_classic_power_spectra"], bins=250, range=(0,500), color='grey', alpha=0.5, density=True)
    ax.hist(probes["peak_power_from_classic_power_spectra"], bins=250, range=(0,500), color='blue', alpha=0.5, density=True)
    #plt.xlim(([0, 500]))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Density of peaks')
    plt.savefig(output_path + '/spectral_peaks.png', dpi=300)
    plt.close()
    return

def plots_n_clusters_per_day(combined_df, output_path=""):
    tetrodes = combined_df[combined_df["electrode"] == "tetrode"]
    probes = combined_df[combined_df["electrode"] == "silicon_probe"]

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    colors = ["grey", "blue"]
    data = [get_clusters_per_day(tetrodes), get_clusters_per_day(probes)]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1,2]):
        ax.scatter((np.ones(len(data[i]))*x)+np.random.uniform(low=-0.1, high=+0.1,size=len(data[i])), data[i], color="black", alpha=0.3,zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,2])
    ax.set_xticklabels(["T", "P"])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([0, 100])
    ax.set_xlim([0, 3])
    #ax.set_xlabel("Encoding grid cells", fontsize=20)
    #ax.set_ylabel("Percentage hits", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    #plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/n_clusters_per_day.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    colors = ["grey", "blue"]
    data = [get_clusters_per_day_per_shank(tetrodes), get_clusters_per_day_per_shank(probes)]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1, 2], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for i, x in enumerate([1, 2]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color="black", alpha=0.3, zorder=1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["T", "P"])
    # ax.set_yticks([-1, 0, 1])
    # ax.set_ylim([0, 100])
    ax.set_xlim([0, 3])
    # ax.set_xlabel("Encoding grid cells", fontsize=20)
    # ax.set_ylabel("Percentage hits", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    # plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/n_clusters_per_day_per_shank.png', dpi=300)
    plt.close()
    return

def plot_total_clusters_per_channel(combined_df, output_path=""):
    return


def main():
    print('-------------------------------------------------------------')
    print("hello")

    tetrode_recordings = pd.DataFrame()
    tetrode_recordings = pd.concat([tetrode_recordings, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort6.pkl")],ignore_index=True)
    tetrode_recordings = pd.concat([tetrode_recordings, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort7.pkl")],ignore_index=True)
    tetrode_recordings = pd.concat([tetrode_recordings, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")],ignore_index=True)
    tetrode_recordings["electrode"] = "tetrode"
    tetrode_recordings["number_of_channels"] = 16
    silicon_probe_recordings = pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort9.pkl")
    silicon_probe_recordings["electrode"] = "silicon_probe"

    silicon_probe_df = silicon_probe_recordings[["session_id_vr", "session_id_of", "full_session_id_of", "full_session_id_vr", "cluster_id", "mouse", "n_trials", "grid_spacing", "field_size", "grid_score", "grid_spacing",
                               "hd_score", "border_score", "ThetaIndex", "mean_firing_rate_of", "rate_map_correlation_first_vs_second_half", "spatial_information_score", "snippet_peak_to_trough"]]
    silicon_probe_df.to_csv("/mnt/datastore/Harry/Vr_grid_cells/silicon_probes.csv")

    combined = pd.concat([tetrode_recordings, silicon_probe_recordings], ignore_index=True)
    plot_power_spectral_peaks(combined, output_path="/mnt/datastore/Harry/Vr_grid_cells/electrode_comparison")

    # remove artefacts
    combined = combined[combined["snippet_peak_to_trough"] < 500]
    #combined = combined[combined["peak_power_from_classic_power_spectra"]%120 < 1e3] # unknown noise picked up @120Hz in VR
    plot_clusters_over_days(combined, output_path="/mnt/datastore/Harry/Vr_grid_cells/electrode_comparison")
    plots_n_clusters_per_day(combined, output_path="/mnt/datastore/Harry/Vr_grid_cells/electrode_comparison")

    plot_quality_metrics(combined, output_path="/mnt/datastore/Harry/Vr_grid_cells/electrode_comparison")
    plot_spatial_metrics(combined, output_path="/mnt/datastore/Harry/Vr_grid_cells/electrode_comparison")
    print("hello")

if __name__ == '__main__':
    main()