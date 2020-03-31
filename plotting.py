import plot_utility
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

isolation_threshold = 0.9
noise_overlap_threshold = 0.05
peak_snr_threshold = 1
firing_rate_threshold = 0.5

def plot_agreement_stats(concat_agreement_stats_dataframe, figs_path=None, cohort_mouse=""):

    fig, ax = plt.subplots()

    for i in range(len(concat_agreement_stats_dataframe)):
        ax.plot([1,2,3], [concat_agreement_stats_dataframe["n_clusters_together"][i], concat_agreement_stats_dataframe["n_clusters_seperately"][i], concat_agreement_stats_dataframe["n_agreements"][i]], alpha=0.3)
        ax.errorbar([1,2,3], [np.mean(concat_agreement_stats_dataframe["n_clusters_together"]),np.mean(concat_agreement_stats_dataframe["n_clusters_seperately"]), np.mean(concat_agreement_stats_dataframe["n_agreements"])],
                            yerr=[stats.sem(concat_agreement_stats_dataframe["n_clusters_together"]),stats.sem(concat_agreement_stats_dataframe["n_clusters_seperately"]), stats.sem(concat_agreement_stats_dataframe["n_agreements"])],
                    color="navy", capsize=0.5)

    ax.set_ylim(0, max(max(concat_agreement_stats_dataframe["n_clusters_together"]), max(concat_agreement_stats_dataframe["n_clusters_seperately"]))+1)
    ax.set_xlim(0.75, 3.25)
    ax.set_ylabel("Curated clusters found", fontsize=20)
    ax.set_xlabel("Sorting method", fontsize=20)
    ax.set_title("Sorting comparison", fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["VR + OF", "VR", "Intersect"])
    fig.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"stats_compare.png")
    plt.show()

    fig, ax = plt.subplots()
    tick_labels = ("VR + OF", "VR", "Intersect")
    y_pos = [1,2,3]
    totals = [np.sum(concat_agreement_stats_dataframe["n_clusters_together"]),
                   np.sum(concat_agreement_stats_dataframe["n_clusters_seperately"]),
                   np.sum(concat_agreement_stats_dataframe["n_agreements"])]

    ax.bar(y_pos, totals, align='center',color="navy")
    ax.set_xlabel('Sorting Method', fontsize=20)
    ax.set_ylabel('Total Curated Clusters', fontsize=20)
    ax.set_title('Sorting Totals', fontsize=23)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["VR + OF", "VR", "Intersect"])
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"stats_total.png")
    plt.show()




def plot_curation_stats(concat_apart, concat_together, figs_path, cohort_mouse):

    mean_apart = concat_apart.mean(axis=0)
    sem_apart = concat_apart.sem(axis=0)
    mean_together = concat_together.mean(axis=0)
    sem_together = concat_together.sem(axis=0)

    # Mean Firing Rate
    fig, ax = plt.subplots()
    y_pos = [1,2]
    means = [mean_together["mean_firing_rate"],
              mean_apart["mean_firing_rate"]]
    sems = [sem_together["mean_firing_rate"],
            sem_apart["mean_firing_rate"]]

    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="purple")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Mean Firing Rate (Hz)', fontsize=25)
    ax.set_title('Mean Firing Rate', fontsize=25)
    ax.set_xticks([1, 2])
    ax.set_xlim([0,3])
    ax.set_ylim([0,20])
    ax.set_xticklabels(["VR + OF", "VR"])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.hlines(y=firing_rate_threshold, xmin=0, xmax=3, linestyles='--', color='k')
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"mean_firing_rate.png")
    plt.show()

    # Isolation
    fig, ax = plt.subplots()
    y_pos = [1,2]
    means = [mean_together["isolation"],
             mean_apart["isolation"]]
    sems = [sem_together["isolation"],
            sem_apart["isolation"]]

    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="navy")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Isolation', fontsize=25)
    ax.set_title('Isolation', fontsize=25)
    ax.set_xticks([1, 2])
    ax.set_xlim([0,3])
    ax.set_ylim([0.8, 1])
    ax.set_xticklabels(["VR + OF", "VR"])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.hlines(y=isolation_threshold, xmin=0, xmax=3, linestyles='--', color='k')
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"isolation.png")
    plt.show()

    # noise_overlap
    fig, ax = plt.subplots()
    y_pos = [1,2]
    means = [mean_together["noise_overlap"],
             mean_apart["noise_overlap"]]
    sems = [sem_together["noise_overlap"],
            sem_apart["noise_overlap"]]

    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="red")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Noise Overlap', fontsize=25)
    ax.set_title('Noise Overlap', fontsize=25)
    ax.set_xticks([1, 2])
    ax.set_xlim([0,3])
    ax.set_ylim([0,0.1])
    ax.set_xticklabels(["VR + OF", "VR"])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.hlines(y=noise_overlap_threshold, xmin=0, xmax=3, linestyles='--', color='k')
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"noise_overlap.png")
    plt.show()

    # peak snr
    fig, ax = plt.subplots()
    y_pos = [1,2]
    means = [mean_together["peak_snr"],
             mean_apart["peak_snr"]]
    sems = [sem_together["peak_snr"],
            sem_apart["peak_snr"]]

    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="orange")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Peak SNR', fontsize=25)
    ax.set_title('Peak SNR', fontsize=25)
    ax.set_xticks([1, 2])
    ax.set_xlim([0,3])
    ax.set_ylim([0,5])
    ax.set_xticklabels(["VR + OF", "VR"])
    ax.hlines(y=peak_snr_threshold, xmin=0, xmax=3, linestyles='--', color='k')
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"peak_snr.png")
    plt.show()

    # peak amp
    fig, ax = plt.subplots()
    y_pos = [1,2]
    means = [mean_together["peak_amp"],
             mean_apart["peak_amp"]]
    sems = [sem_together["peak_amp"],
            sem_apart["peak_amp"]]

    ax.errorbar(y_pos, means, yerr=sems, fmt='o', capsize=40,
                elinewidth=2, markeredgewidth=2, color="green")
    ax.set_xlabel('Sorting Method', fontsize=25)
    ax.set_ylabel('Peak Amplitude', fontsize=25)
    ax.set_title('Peak Amplitude', fontsize=25)
    ax.set_xticks([1, 2])
    ax.set_xlim([0,3])
    ax.set_ylim([0,15])
    ax.set_xticklabels(["VR + OF", "VR"])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figs_path+"/"+cohort_mouse+"peak_amp.png")
    plt.show()

    print("hello there")