import plot_utility
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

isolation_threshold = 0.9
noise_overlap_threshold = 0.05
peak_snr_threshold = 1
firing_rate_threshold = 0.5

def plot_agreement_stats(concat_agreement_stats_dataframe, figs_path=None, cohort_mouse="", agreement_threshold=90):

    fig, ax = plt.subplots()

    for i in range(len(concat_agreement_stats_dataframe)):
        ax.plot([1,2,3,4,5], [concat_agreement_stats_dataframe["n_clusters_together_vr"][i],
                                concat_agreement_stats_dataframe["n_clusters_seperately_vr"][i],
                                concat_agreement_stats_dataframe["n_agreements_vr"][i],
                                concat_agreement_stats_dataframe["n_clusters_seperately_of"][i],
                                concat_agreement_stats_dataframe["n_agreements_of"][i]], alpha=0.3)

        ax.errorbar([1,2,3,4,5], [np.mean(concat_agreement_stats_dataframe["n_clusters_together_vr"]),
                                    np.mean(concat_agreement_stats_dataframe["n_clusters_seperately_vr"]),
                                    np.mean(concat_agreement_stats_dataframe["n_agreements_vr"]),
                                    np.mean(concat_agreement_stats_dataframe["n_clusters_seperately_of"]),
                                    np.mean(concat_agreement_stats_dataframe["n_agreements_of"])],
                            yerr=[stats.sem(concat_agreement_stats_dataframe["n_clusters_together_vr"]),
                                  stats.sem(concat_agreement_stats_dataframe["n_clusters_seperately_vr"]),
                                  stats.sem(concat_agreement_stats_dataframe["n_agreements_vr"]),
                                  stats.sem(concat_agreement_stats_dataframe["n_clusters_seperately_of"]),
                                  stats.sem(concat_agreement_stats_dataframe["n_agreements_of"])], color="navy", capsize=0.5)

    ax.set_ylim(0, max(max(concat_agreement_stats_dataframe["n_clusters_seperately_of"]),
                       max(max(concat_agreement_stats_dataframe["n_clusters_together_vr"]), max(concat_agreement_stats_dataframe["n_clusters_seperately_vr"])))+1)

    ax.set_xlim(0.75, 6.25)
    ax.set_ylabel("Curated clusters found", fontsize=20)
    ax.set_xlabel("Sorting method", fontsize=20)
    #ax.set_title("Sorting comparison", fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["VR+OF", "VR", r"VR$_{Inter}$", "OF", r"OF$_{Inter}$"])
    fig.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"stats_compare_"+str(agreement_threshold)+"at.png")
    plt.show()

    fig, ax = plt.subplots()
    y_pos = [1,2,3,4,5]
    totals = [np.sum(concat_agreement_stats_dataframe["n_clusters_together_vr"]),
              np.sum(concat_agreement_stats_dataframe["n_clusters_seperately_vr"]),
              np.sum(concat_agreement_stats_dataframe["n_agreements_vr"]),
              np.sum(concat_agreement_stats_dataframe["n_clusters_seperately_of"]),
              np.sum(concat_agreement_stats_dataframe["n_agreements_of"])]

    print(np.sum(concat_agreement_stats_dataframe["n_clusters_together_vr"]), "n together vr")
    print(np.sum(concat_agreement_stats_dataframe["n_clusters_together_of"]), "n together of")

    print(np.sum(concat_agreement_stats_dataframe["n_agreements_vr"]), "n_agreements_vr")
    print(np.sum(concat_agreement_stats_dataframe["n_agreements_of"]), " n_agreements_of")

    ax.bar(y_pos, totals, align='center',color="navy")
    ax.set_xlabel('Sorting Method', fontsize=20)
    ax.set_ylabel('Total Curated Clusters', fontsize=20)
    #ax.set_title('Sorting Totals', fontsize=23)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["VR+OF", "VR", r"VR$_{Inter}$", "OF", r"OF$_{Inter}$"])
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"stats_total_"+str(agreement_threshold)+"at.png")
    plt.show()

def plot_curation_stats(concat_apart, concat_together, concat_apart_of, figs_path, cohort_mouse):

    mean_apart = concat_apart.mean(axis=0)
    sem_apart = concat_apart.sem(axis=0)
    mean_together = concat_together.mean(axis=0)
    sem_together = concat_together.sem(axis=0)
    mean_apart_of = concat_apart_of.mean(axis=0)
    sem_apart_of = concat_apart_of.sem(axis=0)

    # Mean Firing Rate
    fig, ax = plt.subplots()
    y_pos = [1,2,3]
    means = [mean_together["mean_firing_rate"],
              mean_apart["mean_firing_rate"],
             mean_apart_of["mean_firing_rate"]]
    sems = [sem_together["mean_firing_rate"],
            sem_apart["mean_firing_rate"],
            sem_apart_of["mean_firing_rate"]]

    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="purple")
    for i in range(len(concat_together)):
        ax.scatter(1, concat_together["mean_firing_rate"][i], marker="x", color="purple")
    for i in range(len(concat_apart)):
        ax.scatter(2, concat_apart["mean_firing_rate"][i], marker="x", color="purple")
    for i in range(len(concat_apart_of)):
        ax.scatter(3, concat_apart_of["mean_firing_rate"][i], marker="x", color="purple")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Mean Firing Rate (Hz)', fontsize=25)
    ax.set_title('Mean Firing Rate', fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xlim([0,4])
    ax.set_ylim([0,20])
    ax.set_xticklabels(["VR + OF", "VR", "OF"])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.hlines(y=firing_rate_threshold, xmin=0, xmax=4, linestyles='--', color='k')
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"mean_firing_rate.png")
    plt.show()

    # Isolation
    fig, ax = plt.subplots()
    y_pos = [1,2,3]
    means = [mean_together["isolation"],
             mean_apart["isolation"],
             mean_apart_of["isolation"]]
    sems = [sem_together["isolation"],
            sem_apart["isolation"],
            sem_apart_of["isolation"]]
    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="navy")
    for i in range(len(concat_together)):
        ax.scatter(1, concat_together["isolation"][i], marker="x", color="navy")
    for i in range(len(concat_apart)):
        ax.scatter(2, concat_apart["isolation"][i], marker="x", color="navy")
    for i in range(len(concat_apart_of)):
        ax.scatter(3, concat_apart_of["isolation"][i], marker="x", color="navy")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Isolation', fontsize=25)
    ax.set_title('Isolation', fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xlim([0,4])
    ax.set_ylim([0.8, 1])
    ax.set_xticklabels(["VR + OF", "VR", "OF"])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.hlines(y=isolation_threshold, xmin=0, xmax=4, linestyles='--', color='k')
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"isolation.png")
    plt.show()

    # noise_overlap
    fig, ax = plt.subplots()
    y_pos = [1,2, 3]
    means = [mean_together["noise_overlap"],
             mean_apart["noise_overlap"],
             mean_apart_of["noise_overlap"]]
    sems = [sem_together["noise_overlap"],
            sem_apart["noise_overlap"],
            sem_apart_of["noise_overlap"]]
    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="red")
    for i in range(len(concat_together)):
        ax.scatter(1, concat_together["noise_overlap"][i], marker="x", color="red")
    for i in range(len(concat_apart)):
        ax.scatter(2, concat_apart["noise_overlap"][i], marker="x", color="red")
    for i in range(len(concat_apart_of)):
        ax.scatter(3, concat_apart_of["noise_overlap"][i], marker="x", color="red")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Noise Overlap', fontsize=25)
    ax.set_title('Noise Overlap', fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xlim([0,4])
    ax.set_ylim([0,0.1])
    ax.set_xticklabels(["VR + OF", "VR", "OF"])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.hlines(y=noise_overlap_threshold, xmin=0, xmax=4, linestyles='--', color='k')
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"noise_overlap.png")
    plt.show()

    # peak snr
    fig, ax = plt.subplots()
    y_pos = [1,2,3]
    means = [mean_together["peak_snr"],
             mean_apart["peak_snr"],
             mean_apart_of["peak_snr"]]
    sems = [sem_together["peak_snr"],
            sem_apart["peak_snr"],
            sem_apart_of["peak_snr"]]
    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="orange")
    for i in range(len(concat_together)):
        ax.scatter(1, concat_together["peak_snr"][i], marker="x", color="orange")
    for i in range(len(concat_apart)):
        ax.scatter(2, concat_apart["peak_snr"][i], marker="x", color="orange")
    for i in range(len(concat_apart_of)):
        ax.scatter(3, concat_apart_of["peak_snr"][i], marker="x", color="orange")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Peak SNR', fontsize=25)
    ax.set_title('Peak SNR', fontsize=25)
    ax.set_xticks([1, 2, 3])
    ax.set_xlim([0,4])
    ax.set_ylim([0,5])
    ax.set_xticklabels(["VR + OF", "VR", "OF"])
    ax.hlines(y=peak_snr_threshold, xmin=0, xmax=4, linestyles='--', color='k')
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"peak_snr.png")
    plt.show()

    # peak amp
    fig, ax = plt.subplots()
    means = [mean_together["peak_amp"],
             mean_apart["peak_amp"],
             mean_apart_of["peak_amp"]]
    sems = [sem_together["peak_amp"],
            sem_apart["peak_amp"],
            sem_apart_of["peak_amp"]]
    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="green")
    for i in range(len(concat_together)):
        ax.scatter(1, concat_together["peak_amp"][i], marker="x", color="green")
    for i in range(len(concat_apart)):
        ax.scatter(2, concat_apart["peak_amp"][i], marker="x", color="green")
    for i in range(len(concat_apart_of)):
        ax.scatter(3, concat_apart_of["peak_amp"][i], marker="x", color="green")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Peak Amplitude', fontsize=25)
    ax.set_title('Peak Amplitude', fontsize=25)
    ax.set_xticks([1, 2,3])
    ax.set_xlim([0,4])
    ax.set_ylim([0,15])
    ax.set_xticklabels(["VR + OF", "VR", "OF"])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"peak_amp.png")
    plt.show()

    print("hello there")