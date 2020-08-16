import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import itertools
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
from Edmond.ramp_cells_of import *
warnings.filterwarnings('ignore')

def tetrode_depth_correlation(data, collumn_a, label_collumn, trial_type, save_path, of_n_spike_thres=1000):
    data = data[(data["trial_type"]==trial_type)]
    data = data.dropna(subset=["tetrode_location"])
    data = data.dropna(subset=[collumn_a])
    # remove clusters that have very few spikes in of to calculate spatial scores on
    if "n_spikes_of" in list(data):
        data = data[data["n_spikes_of"]>=of_n_spike_thres]
    color = label_collumn2color(data, label_collumn)

    fig, ax = plt.subplots()
    ax.set_title("label="+label_collumn+", tt="+get_tidy_title(trial_type), fontsize=15)
    ax.scatter(data["tetrode_location"], data[collumn_a], edgecolor=color, marker="o", facecolors='none')
    plot_regression(ax, data["tetrode_location"], data[collumn_a])
    plt.ylabel(get_tidy_title(collumn_a), fontsize=20, labelpad=10)
    plt.xlabel("Tetrode Depth (mm)", fontsize=20, labelpad=10)
    #plt.xlim(0, 2)
    #plt.ylim(-150, 150)
    plt.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(save_path+"/tetrode_depth_correlations_"+collumn_a+"_"+label_collumn+"_"+trial_type+".png")
    print("plotted depth correlation")

def tetrode_depth2(data, trial_type, save_path, of_n_spike_thres=1000, bound="outbound"):
    data = data[(data["trial_type"]==trial_type)]
    data = data.dropna(subset=["tetrode_location"])
    # remove clusters that have very few spikes in of to calculate spatial scores on
    if "n_spikes_of" in list(data):
        data = data[data["n_spikes_of"]>=of_n_spike_thres]
    if (trial_type == "beaconed") and (bound=="outbound"):
        collumn_lm = "lm_result_b_outbound"
    elif (trial_type == "non-beaconed") and (bound=="outbound"):
        collumn_lm = "lm_result_nb_outbound"
    elif (trial_type == "probe") and (bound=="outbound"):
        collumn_lm = "lm_result_p_outbound"
    elif (trial_type == "beaconed") and (bound=="homebound"):
        collumn_lm = "lm_result_b_homebound"
    elif (trial_type == "non-beaconed") and (bound=="homebound"):
        collumn_lm = "lm_result_nb_homebound"
    elif (trial_type == "probe") and (bound=="homebound"):
        collumn_lm = "lm_result_p_homebound"

    non_ramp = data[(data[collumn_lm] == "None")]
    neg_ramp =  data[(data[collumn_lm] == "Negative")]
    pos_ramp =  data[(data[collumn_lm] == "Positive")]

    fig, ax = plt.subplots()
    ax.set_title("tt="+get_tidy_title(trial_type)+", b="+bound, fontsize=15)

    all_hist = np.histogram(data["tetrode_location"], bins=5, range=(1.5, 3))
    pos_ramps_hist = np.histogram(pos_ramp["tetrode_location"], bins=5, range=(1.5, 3))
    neg_ramps_hist = np.histogram(neg_ramp["tetrode_location"], bins=5, range=(1.5, 3))
    non_ramps_hist = np.histogram(non_ramp["tetrode_location"], bins=5, range=(1.5, 3))
    bin_centers = (pos_ramps_hist[1][:-1] + pos_ramps_hist[1][1:]) / 2

    # plot
    barWidth = 0.3
    # Create green Bars
    ax.bar(bin_centers, non_ramps_hist[0]/all_hist[0], color='grey', edgecolor='white', width=barWidth)
    ax.bar(bin_centers, neg_ramps_hist[0]/all_hist[0], bottom=non_ramps_hist[0]/all_hist[0], color="red", edgecolor='white', width=barWidth)
    ax.bar(bin_centers, pos_ramps_hist[0]/all_hist[0], bottom=[i+j for i,j in zip(non_ramps_hist[0]/all_hist[0],
                                                                                  neg_ramps_hist[0]/all_hist[0])], color='blue', edgecolor='white', width=barWidth)
    plt.ylabel("Proportion of neurons", fontsize=20, labelpad=10)
    plt.xlabel("Tetrode Depth (mm)", fontsize=20, labelpad=10)
    plt.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if bound == "outbound":
        plt.savefig(save_path+"/outbound_tetrode_depth_histo2_"+trial_type+".png")
    else:
        plt.savefig(save_path+"/homebound_tetrode_depth_histo2_"+trial_type+".png")
        print("plotted depth correlation")

def tetrode_depth5(data, trial_type, save_path, of_n_spike_thres=1000, cohort_mouse=None):
    data = data[(data["trial_type"]==trial_type)]
    data = data.dropna(subset=["tetrode_location"])
    if cohort_mouse is not None:
        data=data[(data["cohort_mouse"] == cohort_mouse)]
        if len(data)==0:
            return
    # remove clusters that have very few spikes in of to calculate spatial scores on
    if "n_spikes_of" in list(data):
        data = data[data["n_spikes_of"]>=of_n_spike_thres]
    collumn_lm = "ramp_driver"

    non_ramp = data[(data[collumn_lm] == "None")]
    cue_ramp =  data[(data[collumn_lm] == "Cue")]
    pi_ramp =  data[(data[collumn_lm] == "PI")]

    fig, ax = plt.subplots()
    ax.set_title("tt="+get_tidy_title(trial_type), fontsize=15)

    all_hist = np.histogram(data["tetrode_location"], bins=5, range=(1.5, 3))
    pi_ramps_hist = np.histogram(pi_ramp["tetrode_location"], bins=5, range=(1.5, 3))
    cue_ramps_hist = np.histogram(cue_ramp["tetrode_location"], bins=5, range=(1.5, 3))
    non_ramps_hist = np.histogram(non_ramp["tetrode_location"], bins=5, range=(1.5, 3))
    bin_centers = (pi_ramps_hist[1][:-1] + pi_ramps_hist[1][1:]) / 2

    # plot
    barWidth = 0.3
    # Create green Bars
    ax.bar(bin_centers, non_ramps_hist[0]/all_hist[0], color='grey', edgecolor='white', width=barWidth)
    ax.bar(bin_centers, cue_ramps_hist[0]/all_hist[0], bottom=non_ramps_hist[0]/all_hist[0], color="green", edgecolor='white', width=barWidth)
    ax.bar(bin_centers, pi_ramps_hist[0]/all_hist[0], bottom=[i+j for i,j in zip(non_ramps_hist[0]/all_hist[0],
                                                                                  cue_ramps_hist[0]/all_hist[0])], color='yellow', edgecolor='white', width=barWidth)
    plt.ylabel("Proportion of neurons", fontsize=20, labelpad=10)
    plt.xlabel("Tetrode Depth (mm)", fontsize=20, labelpad=10)
    plt.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if cohort_mouse is not None:
        plt.savefig(save_path+"/"+cohort_mouse+"_tetrode_depth5_histo_"+trial_type+".png")
    else:
        plt.savefig(save_path+"/tetrode_depth5_histo_"+trial_type+".png")
        print("plotted depth correlation")

def tetrode_depth_avg_by_lm(data, trial_type, collumn_a, save_path, of_n_spike_thres=1000, bound="outbound"):
    data = data[(data["trial_type"]==trial_type)]
    data = data.dropna(subset=["tetrode_location"])
    # remove clusters that have very few spikes in of to calculate spatial scores on
    if "n_spikes_of" in list(data):
        data = data[data["n_spikes_of"]>=of_n_spike_thres]

    if (trial_type == "beaconed") and (bound=="outbound"):
        collumn_lm = "lm_result_b_outbound"
    elif (trial_type == "non-beaconed") and (bound=="outbound"):
        collumn_lm = "lm_result_nb_outbound"
    elif (trial_type == "probe") and (bound=="outbound"):
        collumn_lm = "lm_result_p_outbound"
    elif (trial_type == "beaconed") and (bound=="homebound"):
        collumn_lm = "lm_result_b_homebound"
    elif (trial_type == "non-beaconed") and (bound=="homebound"):
        collumn_lm = "lm_result_nb_homebound"
    elif (trial_type == "probe") and (bound=="homebound"):
        collumn_lm = "lm_result_p_homebound"

    top25 = data.sort_values(by=collumn_a, ascending=False)[:int(len(data)/4)]
    bottom25 = data.sort_values(by=collumn_a, ascending=True)[:int(len(data)/4)]

    non_ramp = data[(data[collumn_lm] == "None")]
    neg_ramp =  data[(data[collumn_lm] == "Negative")]
    pos_ramp =  data[(data[collumn_lm] == "Positive")]

    non_ramp_top25 =  top25[(top25[collumn_lm] == "None")]
    neg_ramp_top25 =  top25[(top25[collumn_lm] == "Negative")]
    pos_ramp_top25 =  top25[(top25[collumn_lm] == "Positive")]
    non_ramp_bottom25 =  bottom25[(bottom25[collumn_lm] == "None")]
    neg_ramp_bottom25 =  bottom25[(bottom25[collumn_lm] == "Negative")]
    pos_ramp_bottom25 =  bottom25[(bottom25[collumn_lm] == "Positive")]

    bin_edges = [(1.5, 1.8), (1.8, 2.1), (2.1, 2.4), (2.4, 2.7), (2.7, 3.0)]
    fig, ax = plt.subplots()
    ax.set_title("tt="+get_tidy_title(trial_type)+", b= "+bound, fontsize=15)
    non_ramp_top25_bins = []
    neg_ramp_top25_bins = []
    pos_ramp_top25_bins = []
    non_ramp_bottom25_bins = []
    neg_ramp_bottom25_bins = []
    pos_ramp_bottom25_bins = []
    non_ramp_bins = []
    neg_ramp_bins = []
    pos_ramp_bins = []

    data_ramps_bins=[]


    for i in range(len(bin_edges)):

        data_bin = data[((data["tetrode_location"] > bin_edges[i][0]) & (data["tetrode_location"] <= bin_edges[i][1]))]
        non_ramp_bin = non_ramp[((non_ramp["tetrode_location"] > bin_edges[i][0]) & (non_ramp["tetrode_location"] <= bin_edges[i][1]))]
        neg_ramp_bin = neg_ramp[((neg_ramp["tetrode_location"] > bin_edges[i][0]) & (neg_ramp["tetrode_location"] <= bin_edges[i][1]))]
        pos_ramp_bin = pos_ramp[((pos_ramp["tetrode_location"] > bin_edges[i][0]) & (pos_ramp["tetrode_location"] <= bin_edges[i][1]))]

        non_ramp_top25_bin = non_ramp_top25[((non_ramp_top25["tetrode_location"] > bin_edges[i][0]) & (non_ramp_top25["tetrode_location"] <= bin_edges[i][1]))]
        neg_ramp_top25_bin = neg_ramp_top25[((neg_ramp_top25["tetrode_location"] > bin_edges[i][0]) & (neg_ramp_top25["tetrode_location"] <= bin_edges[i][1]))]
        pos_ramp_top25_bin = pos_ramp_top25[((pos_ramp_top25["tetrode_location"] > bin_edges[i][0]) & (pos_ramp_top25["tetrode_location"] <= bin_edges[i][1]))]
        non_ramp_bottom25_bin = non_ramp_bottom25[((non_ramp_bottom25["tetrode_location"] > bin_edges[i][0]) & (non_ramp_bottom25["tetrode_location"] <= bin_edges[i][1]))]
        neg_ramp_bottom25_bin = neg_ramp_bottom25[((neg_ramp_bottom25["tetrode_location"] > bin_edges[i][0]) & (neg_ramp_bottom25["tetrode_location"] <= bin_edges[i][1]))]
        pos_ramp_bottom25_bin = pos_ramp_bottom25[((pos_ramp_bottom25["tetrode_location"] > bin_edges[i][0]) & (pos_ramp_bottom25["tetrode_location"] <= bin_edges[i][1]))]

        non_ramp_bins.append(non_ramp_bin[collumn_a].mean())
        neg_ramp_bins.append(neg_ramp_bin[collumn_a].mean())
        pos_ramp_bins.append(pos_ramp_bin[collumn_a].mean())
        non_ramp_top25_bins.append(non_ramp_top25_bin[collumn_a].mean())
        neg_ramp_top25_bins.append(neg_ramp_top25_bin[collumn_a].mean())
        pos_ramp_top25_bins.append(pos_ramp_top25_bin[collumn_a].mean())
        non_ramp_bottom25_bins.append(non_ramp_bottom25_bin[collumn_a].mean())
        neg_ramp_bottom25_bins.append(neg_ramp_bottom25_bin[collumn_a].mean())
        pos_ramp_bottom25_bins.append(pos_ramp_bottom25_bin[collumn_a].mean())
        data_ramps_bins.append(data_bin[collumn_a].mean())

        ax.errorbar(x=np.mean(bin_edges[i]), y=data_bin[collumn_a].mean(), yerr=data_bin[collumn_a].sem(), color="grey", capsize=10)

        #ax.errorbar(x=np.mean(bin_edges[i]), y=non_ramp_bin[collumn_a].mean(), yerr=non_ramp_bin[collumn_a].sem(), color="grey", capsize=10)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=neg_ramp_bin[collumn_a].mean(), yerr=neg_ramp_bin[collumn_a].sem(), color="red", capsize=10)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=pos_ramp_bin[collumn_a].mean(), yerr=pos_ramp_bin[collumn_a].sem(), color="blue", capsize=10)

        #ax.errorbar(x=np.mean(bin_edges[i]), y=non_ramp_top25_bin[collumn_a].mean(), yerr=non_ramp_top25_bin[collumn_a].sem(), color="grey", capsize=5, alpha=0.5)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=neg_ramp_top25_bin[collumn_a].mean(), yerr=neg_ramp_top25_bin[collumn_a].sem(), color="red", capsize=5, alpha=0.5)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=pos_ramp_top25_bin[collumn_a].mean(), yerr=pos_ramp_top25_bin[collumn_a].sem(), color="blue", capsize=5, alpha=0.5)

        #ax.errorbar(x=np.mean(bin_edges[i]), y=non_ramp_bottom25_bin[collumn_a].mean(), yerr=non_ramp_bottom25_bin[collumn_a].sem(), color="grey", capsize=5, alpha=0.5)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=neg_ramp_bottom25_bin[collumn_a].mean(), yerr=neg_ramp_bottom25_bin[collumn_a].sem(), color="red", capsize=5, alpha=0.5)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=pos_ramp_bottom25_bin[collumn_a].mean(), yerr=pos_ramp_bottom25_bin[collumn_a].sem(), color="blue", capsize=5, alpha=0.5)

    bin_centres = [1.65, 1.95, 2.25, 2.55, 2.85]
    ax.plot(bin_centres, data_ramps_bins, color="grey")
    #ax.plot(bin_centres, non_ramp_bins, color="grey")
    #ax.plot(bin_centres, neg_ramp_bins, color="red")
    #ax.plot(bin_centres, pos_ramp_bins, color="blue")
    #ax.plot(bin_centres, non_ramp_top25_bins, color="grey", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, neg_ramp_top25_bins, color="red", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, pos_ramp_top25_bins, color="blue", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, non_ramp_bottom25_bins, color="grey", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, neg_ramp_bottom25_bins, color="red", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, pos_ramp_bottom25_bins, color="blue", linestyle="dashed", alpha=0.5)

    plt.ylabel(get_tidy_title(collumn_a), fontsize=20, labelpad=10)
    plt.xlabel("Binned Tetrode Depth (mm)", fontsize=20, labelpad=10)
    plt.xticks([1.5, 1.8, 2.1, 2.4, 2.7, 3])
    plt.tick_params(labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.savefig(save_path+"/tetrode_depth_avg_by_lm_"+collumn_a+"_"+trial_type+"_"+bound+".png")
    print("plotted depth correlation")

def tetrode_depth4(data, trial_type, collumn_a, save_path, of_n_spike_thres=1000):
    data = data[(data["trial_type"]==trial_type)]
    data = data.dropna(subset=["tetrode_location"])
    # remove clusters that have very few spikes in of to calculate spatial scores on
    if "n_spikes_of" in list(data):
        data = data[data["n_spikes_of"]>=of_n_spike_thres]
    collumn_lm = "ramp_driver"

    top25 = data.sort_values(by=collumn_a, ascending=False)[:int(len(data)/4)]
    bottom25 = data.sort_values(by=collumn_a, ascending=True)[:int(len(data)/4)]

    non_ramp =  data[(data[collumn_lm] == "None")]
    cue_ramp =  data[(data[collumn_lm] == "Cue")]
    pi_ramp =   data[(data[collumn_lm] == "PI")]

    non_ramp_top25 =  top25[(top25[collumn_lm] == "None")]
    cue_ramp_top25 =  top25[(top25[collumn_lm] == "Cue")]
    pi_ramp_top25 =   top25[(top25[collumn_lm] == "PI")]
    non_ramp_bottom25 =  bottom25[(bottom25[collumn_lm] == "None")]
    cue_ramp_bottom25 =  bottom25[(bottom25[collumn_lm] == "Cue")]
    pi_ramp_bottom25 =   bottom25[(bottom25[collumn_lm] == "PI")]

    bin_edges = [(1.5, 1.8), (1.8, 2.1), (2.1, 2.4), (2.4, 2.7), (2.7, 3.0)]
    fig, ax = plt.subplots()
    ax.set_title("tt="+get_tidy_title(trial_type), fontsize=15)
    non_ramp_top25_bins = []
    cue_ramp_top25_bins = []
    pi_ramp_top25_bins = []
    non_ramp_bottom25_bins = []
    cue_ramp_bottom25_bins = []
    pi_ramp_bottom25_bins = []
    non_ramp_bins = []
    cue_ramp_bins = []
    pi_ramp_bins = []


    for i in range(len(bin_edges)):
        non_ramp_bin = non_ramp[((non_ramp["tetrode_location"] > bin_edges[i][0]) & (non_ramp["tetrode_location"] <= bin_edges[i][1]))]
        cue_ramp_bin = cue_ramp[((cue_ramp["tetrode_location"] > bin_edges[i][0]) & (cue_ramp["tetrode_location"] <= bin_edges[i][1]))]
        pi_ramp_bin = pi_ramp[((pi_ramp["tetrode_location"] > bin_edges[i][0]) & (pi_ramp["tetrode_location"] <= bin_edges[i][1]))]

        non_ramp_top25_bin = non_ramp_top25[((non_ramp_top25["tetrode_location"] > bin_edges[i][0]) & (non_ramp_top25["tetrode_location"] <= bin_edges[i][1]))]
        cue_ramp_top25_bin = cue_ramp_top25[((cue_ramp_top25["tetrode_location"] > bin_edges[i][0]) & (cue_ramp_top25["tetrode_location"] <= bin_edges[i][1]))]
        pi_ramp_top25_bin = pi_ramp_top25[((pi_ramp_top25["tetrode_location"] > bin_edges[i][0]) & (pi_ramp_top25["tetrode_location"] <= bin_edges[i][1]))]
        non_ramp_bottom25_bin = non_ramp_bottom25[((non_ramp_bottom25["tetrode_location"] > bin_edges[i][0]) & (non_ramp_bottom25["tetrode_location"] <= bin_edges[i][1]))]
        cue_ramp_bottom25_bin = cue_ramp_bottom25[((cue_ramp_bottom25["tetrode_location"] > bin_edges[i][0]) & (cue_ramp_bottom25["tetrode_location"] <= bin_edges[i][1]))]
        pi_ramp_bottom25_bin = pi_ramp_bottom25[((pi_ramp_bottom25["tetrode_location"] > bin_edges[i][0]) & (pi_ramp_bottom25["tetrode_location"] <= bin_edges[i][1]))]

        non_ramp_bins.append(non_ramp_bin[collumn_a].mean())
        cue_ramp_bins.append(cue_ramp_bin[collumn_a].mean())
        pi_ramp_bins.append(pi_ramp_bin[collumn_a].mean())
        non_ramp_top25_bins.append(non_ramp_top25_bin[collumn_a].mean())
        cue_ramp_top25_bins.append(cue_ramp_top25_bin[collumn_a].mean())
        pi_ramp_top25_bins.append(pi_ramp_top25_bin[collumn_a].mean())
        non_ramp_bottom25_bins.append(non_ramp_bottom25_bin[collumn_a].mean())
        cue_ramp_bottom25_bins.append(cue_ramp_bottom25_bin[collumn_a].mean())
        pi_ramp_bottom25_bins.append(pi_ramp_bottom25_bin[collumn_a].mean())

        ax.errorbar(x=np.mean(bin_edges[i]), y=non_ramp_bin[collumn_a].mean(), yerr=non_ramp_bin[collumn_a].sem(), color="grey", capsize=10)
        ax.errorbar(x=np.mean(bin_edges[i]), y=cue_ramp_bin[collumn_a].mean(), yerr=cue_ramp_bin[collumn_a].sem(), color="green", capsize=10)
        ax.errorbar(x=np.mean(bin_edges[i]), y=pi_ramp_bin[collumn_a].mean(), yerr=pi_ramp_bin[collumn_a].sem(), color="yellow", capsize=10)

        #ax.errorbar(x=np.mean(bin_edges[i]), y=non_ramp_top25_bin[collumn_a].mean(), yerr=non_ramp_top25_bin[collumn_a].sem(), color="grey", capsize=5, alpha=0.5)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=cue_ramp_top25_bin[collumn_a].mean(), yerr=cue_ramp_top25_bin[collumn_a].sem(), color="green", capsize=5, alpha=0.5)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=pi_ramp_top25_bin[collumn_a].mean(), yerr=pi_ramp_top25_bin[collumn_a].sem(), color="yellow", capsize=5, alpha=0.5)

        #ax.errorbar(x=np.mean(bin_edges[i]), y=non_ramp_bottom25_bin[collumn_a].mean(), yerr=non_ramp_bottom25_bin[collumn_a].sem(), color="grey", capsize=5, alpha=0.5)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=cue_ramp_bottom25_bin[collumn_a].mean(), yerr=cue_ramp_bottom25_bin[collumn_a].sem(), color="green", capsize=5, alpha=0.5)
        #ax.errorbar(x=np.mean(bin_edges[i]), y=pi_ramp_bottom25_bin[collumn_a].mean(), yerr=pi_ramp_bottom25_bin[collumn_a].sem(), color="yellow", capsize=5, alpha=0.5)

    bin_centres = [1.65, 1.95, 2.25, 2.55, 2.85]
    ax.plot(bin_centres, non_ramp_bins, color="grey")
    ax.plot(bin_centres, cue_ramp_bins, color="green")
    ax.plot(bin_centres, pi_ramp_bins, color="yellow")
    #ax.plot(bin_centres, non_ramp_top25_bins, color="grey", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, cue_ramp_top25_bins, color="green", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, pi_ramp_top25_bins, color="yellow", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, non_ramp_bottom25_bins, color="grey", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, cue_ramp_bottom25_bins, color="green", linestyle="dashed", alpha=0.5)
    #ax.plot(bin_centres, pi_ramp_bottom25_bins, color="yellow", linestyle="dashed", alpha=0.5)

    plt.ylabel(get_tidy_title(collumn_a), fontsize=20, labelpad=10)
    plt.xlabel("Binned Tetrode Depth (mm)", fontsize=20, labelpad=10)
    plt.xticks([1.5, 1.8, 2.1, 2.4, 2.7, 3])
    plt.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.savefig(save_path+"/tetrode_depth_histo4_"+collumn_a+"_"+trial_type+".png")
    print("plotted depth correlation")

    data = data[(data["trial_type"]==trial_type)]
    data = data.dropna(subset=["tetrode_location"])
    # remove clusters that have very few spikes in of to calculate spatial scores on
    if "n_spikes_of" in list(data):
        data = data[data["n_spikes_of"]>=of_n_spike_thres]
    collumn_lm = "ramp_driver"

    non_ramp = data[(data[collumn_lm] == "None")]
    cue_ramp =  data[(data[collumn_lm] == "Cue")]
    pi_ramp =  data[(data[collumn_lm] == "PI")]

    fig, ax = plt.subplots()
    ax.set_title("tt="+get_tidy_title(trial_type), fontsize=15)

    all_hist = np.histogram(data["tetrode_location"], bins=5, range=(1.5, 3))
    pi_ramps_hist = np.histogram(pi_ramp["tetrode_location"], bins=5, range=(1.5, 3))
    cue_ramps_hist = np.histogram(cue_ramp["tetrode_location"], bins=5, range=(1.5, 3))
    non_ramps_hist = np.histogram(non_ramp["tetrode_location"], bins=5, range=(1.5, 3))
    bin_centers = (pi_ramps_hist[1][:-1] + pi_ramps_hist[1][1:]) / 2
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # plot
    barWidth = 0.3
    # Create green Bars
    ax.bar(bin_centers, non_ramps_hist[0]/all_hist[0], color='grey', edgecolor='white', width=barWidth)
    ax.bar(bin_centers, cue_ramps_hist[0]/all_hist[0], bottom=non_ramps_hist[0]/all_hist[0], color="green", edgecolor='white', width=barWidth)
    ax.bar(bin_centers, pi_ramps_hist[0]/all_hist[0], bottom=[i+j for i,j in zip(non_ramps_hist[0]/all_hist[0],
                                                                                 cue_ramps_hist[0]/all_hist[0])], color='yellow', edgecolor='white', width=barWidth)
    plt.ylabel("Proportion of neurons", fontsize=20, labelpad=10)
    plt.xlabel("Tetrode Depth (mm)", fontsize=20, labelpad=10)
    plt.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.savefig(save_path+"/tetrode_depth5_histo_"+trial_type+".png")
    print("plotted depth correlation")

def tetrode_depth6(data, trial_type, save_path, of_n_spike_thres=1000, bound="outbound"):
    data = data[(data["trial_type"]==trial_type)]
    data = data.dropna(subset=["tetrode_location"])
    # remove clusters that have very few spikes in of to calculate spatial scores on
    if "n_spikes_of" in list(data):
        data = data[data["n_spikes_of"]>=of_n_spike_thres]
    collumn_lm = "ramp_driver"

    # currently trial type is only provided for beaconed
    if bound == "outbound":
        collumn_lm = "lmer_result_outbound"
    elif bound == "homebound":
        collumn_lm = "lmer_result_homebound"

    none_cells = data[(data[collumn_lm] == "None")]
    p_cells = data[(data[collumn_lm] == "P")]
    s_cells = data[(data[collumn_lm] == "S")]
    a_cells = data[(data[collumn_lm] == "A")]
    ps_cells = data[(data[collumn_lm] == "PS")]
    psa_cells = data[(data[collumn_lm] == "PSA")]
    sa_cells = data[(data[collumn_lm] == "SA")]
    pa_cells = data[(data[collumn_lm] == "PA")]

    fig, ax = plt.subplots()
    ax.set_title("tt="+get_tidy_title(trial_type)+", b="+bound, fontsize=15)

    all_hist = np.histogram(data["tetrode_location"], bins=5, range=(1.5, 3))
    none_cells_hist = np.histogram(none_cells["tetrode_location"], bins=5, range=(1.5, 3))
    p_cells_hist = np.histogram(p_cells["tetrode_location"], bins=5, range=(1.5, 3))
    s_cells_hist = np.histogram(s_cells["tetrode_location"], bins=5, range=(1.5, 3))
    a_cells_hist = np.histogram(a_cells["tetrode_location"], bins=5, range=(1.5, 3))
    ps_cells_hist = np.histogram(ps_cells["tetrode_location"], bins=5, range=(1.5, 3))
    psa_cells_hist = np.histogram(psa_cells["tetrode_location"], bins=5, range=(1.5, 3))
    sa_cells_hist = np.histogram(sa_cells["tetrode_location"], bins=5, range=(1.5, 3))
    pa_cells_hist = np.histogram(pa_cells["tetrode_location"], bins=5, range=(1.5, 3))
    bin_centers = (none_cells_hist[1][:-1] + none_cells_hist[1][1:]) / 2

    # plot
    barWidth = 0.3
    # Create green Bars
    ax.bar(bin_centers, none_cells_hist[0]/all_hist[0], color='grey', edgecolor='white', width=barWidth)
    ax.bar(bin_centers, p_cells_hist[0]/all_hist[0], bottom=none_cells_hist[0]/all_hist[0], color="deeppink", edgecolor='white', width=barWidth)
    ax.bar(bin_centers, s_cells_hist[0]/all_hist[0], bottom=[i+j for i,j in zip(none_cells_hist[0]/all_hist[0],
                                                                                p_cells_hist[0]/all_hist[0])], color='salmon', edgecolor='white', width=barWidth)
    ax.bar(bin_centers, a_cells_hist[0]/all_hist[0], bottom=[i+j+l for i,j,l in zip(none_cells_hist[0]/all_hist[0],
                                                                                p_cells_hist[0]/all_hist[0],
                                                                                s_cells_hist[0]/all_hist[0])], color='lightskyblue', edgecolor='white', width=barWidth)
    ax.bar(bin_centers, ps_cells_hist[0]/all_hist[0], bottom=[i+j+l+m for i,j,l,m in zip(none_cells_hist[0]/all_hist[0],
                                                                                    p_cells_hist[0]/all_hist[0],
                                                                                    s_cells_hist[0]/all_hist[0],
                                                                                    a_cells_hist[0]/all_hist[0])], color='indianred', edgecolor='white', width=barWidth)
    ax.bar(bin_centers, psa_cells_hist[0]/all_hist[0], bottom=[i+j+l+m+n for i,j,l,m,n in zip(none_cells_hist[0]/all_hist[0],
                                                                                            p_cells_hist[0]/all_hist[0],
                                                                                            s_cells_hist[0]/all_hist[0],
                                                                                            a_cells_hist[0]/all_hist[0],
                                                                                            ps_cells_hist[0]/all_hist[0])], color='forestgreen', edgecolor='white', width=barWidth)
    ax.bar(bin_centers, sa_cells_hist[0]/all_hist[0], bottom=[i+j+l+m+n+o for i,j,l,m,n,o in zip(none_cells_hist[0]/all_hist[0],
                                                                                              p_cells_hist[0]/all_hist[0],
                                                                                              s_cells_hist[0]/all_hist[0],
                                                                                              a_cells_hist[0]/all_hist[0],
                                                                                              ps_cells_hist[0]/all_hist[0],
                                                                                              psa_cells_hist[0]/all_hist[0])], color='mediumaquamarine', edgecolor='white', width=barWidth)
    ax.bar(bin_centers, pa_cells_hist[0]/all_hist[0], bottom=[i+j+l+m+n+o+p for i,j,l,m,n,o,p in zip(none_cells_hist[0]/all_hist[0],
                                                                                                 p_cells_hist[0]/all_hist[0],
                                                                                                 s_cells_hist[0]/all_hist[0],
                                                                                                 a_cells_hist[0]/all_hist[0],
                                                                                                 ps_cells_hist[0]/all_hist[0],
                                                                                                 psa_cells_hist[0]/all_hist[0],
                                                                                                 sa_cells_hist[0]/all_hist[0])], color='orchid', edgecolor='white', width=barWidth)

    plt.ylabel("Proportion of neurons", fontsize=20, labelpad=10)
    plt.xlabel("Tetrode Depth (mm)", fontsize=20, labelpad=10)
    plt.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if bound == "outbound":
        plt.savefig(save_path+"/outbound_tetrode_depth6_histo_"+trial_type+".png")
    else:
        plt.savefig(save_path+"/homebound_tetrode_depth6histo_"+trial_type+".png")
        print("plotted depth correlation")

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    ramp_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_results_linearmodel.txt"
    ramp_scores_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ramp_score_export.csv"
    tetrode_location_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/tetrode_locations.csv"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/tetrode_depth_figs"
    c5m1_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M1_sorting_stats.pkl"
    c5m2_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M2_sorting_stats.pkl"
    c4m2_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M2_sorting_stats.pkl"
    c4m3_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M3_sorting_stats.pkl"
    c3m1_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M1_sorting_stats.pkl"
    c3m6_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M6_sorting_stats.pkl"
    c2m245_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/245_sorting_stats.pkl"
    c2m1124_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/1124_sorting_stats.pkl"

    all_of_paths = [c5m1_path, c5m2_path, c4m2_path, c4m3_path, c3m1_path, c3m6_path, c2m245_path, c2m1124_path]
    data = concatenate_all(ramp_path, ramp_scores_path, tetrode_location_path, all_of_paths, include_unmatch=False, ignore_of=False)

    '''
    data = concatenate_all(ramp_path, ramp_scores_path, tetrode_location_path, all_of_paths, include_unmatch=False, ignore_of=False)
    print(len(data)/3)
    data = data[(data["spike_ratio"]>0.5)]
    print(len(data)/3)
    
    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        for collumn_a in ["ramp_score_out", "ramp_score_home", "ramp_score", "max_ramp_score", "hd_score", "speed_score", "grid_score", "border_score", "rayleigh_score", "rate_map_correlation_first_vs_second_half"]:
            for label_collumn in ["lm_result_b_homebound", "lm_result_b_outbound", "lmer_result_homebound", "lmer_result_outbound", "ramp_driver", "max_ramp_score_label"]:
                tetrode_depth_correlation(data, collumn_a, label_collumn, trial_type, save_path, of_n_spike_thres=1000, cohort_mouse=None)

    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        for bound in ["homebound", "outbound"]:
            for collumn_a in ["ramp_score_out", "ramp_score_home", "ramp_score", "max_ramp_score", "hd_score", "speed_score", "grid_score", "border_score", "rayleigh_score", "rate_map_correlation_first_vs_second_half"]:
                tetrode_depth_avg_by_lm(data, trial_type, collumn_a, save_path, of_n_spike_thres=1000, bound=bound)

    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        for collumn_a in ["ramp_score_out", "ramp_score_home", "ramp_score", "max_ramp_score", "hd_score", "speed_score", "grid_score", "border_score", "rayleigh_score", "rate_map_correlation_first_vs_second_half"]:
            tetrode_depth4(data, trial_type, collumn_a, save_path, of_n_spike_thres=1000)
    '''

    data_no_of = concatenate_all(ramp_path, ramp_scores_path, tetrode_location_path, all_of_paths, include_unmatch=False, ignore_of=True)

    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        for collumn_a in ["ramp_score_out", "ramp_score_home", "ramp_score", "max_ramp_score"]:
            for label_collumn in ["lm_result_b_homebound", "lm_result_b_outbound", "lmer_result_homebound", "lmer_result_outbound", "ramp_driver", "max_ramp_score_label"]:
                tetrode_depth_correlation(data_no_of, collumn_a, label_collumn, trial_type, save_path, of_n_spike_thres=1000)

    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        for bound in ["homebound", "outbound"]:
            for collumn_a in ["ramp_score_out", "ramp_score_home", "ramp_score", "max_ramp_score"]:
                tetrode_depth_avg_by_lm(data_no_of, trial_type, collumn_a, save_path, of_n_spike_thres=1000, bound=bound)

    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        for bound in ["homebound", "outbound"]:
            tetrode_depth6(data_no_of, trial_type, save_path, of_n_spike_thres=1000, bound=bound)

    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        for bound in ["homebound", "outbound"]:
            tetrode_depth2(data_no_of, trial_type, save_path, of_n_spike_thres=1000, bound=bound)

    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        tetrode_depth5(data_no_of, trial_type, save_path, of_n_spike_thres=1000, cohort_mouse=None)



if __name__ == '__main__':
    main()