import pandas as pd
import pickle5 as pkl5
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
import itertools
import scipy
from scipy import stats
from scipy.stats import kde
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

def split_data_by_recording_day(data):
    #split them into early and late sessions
    sorted_days = np.sort(np.asarray(data.recording_day))
    mid_point_day = sorted_days[int(len(sorted_days)/2)]

    early_data = data[(data["recording_day"] < mid_point_day)]
    late_data =  data[(data["recording_day"] >= mid_point_day)]
    return early_data, late_data

def get_tidy_title(collumn):
    if collumn == "speed_score":
        return "Speed Score"
    if collumn == "abs_speed_score":
        return "Abs Speed Score"
    elif collumn == "grid_score":
        return "Grid Score"
    elif collumn == "border_score":
        return "Border Score"
    elif collumn == "corner_score":
        return "Corner Score"
    elif collumn == "hd_score":
        return "HD Score"
    elif collumn == "ramp_score_out":
        return "Ramp Score Outbound"
    elif collumn == "ramp_score_home":
        return "Ramp Score Homebound"
    elif collumn == "ramp_score":
        return "Ramp Score"
    elif collumn == "abs_ramp_score":
        return "Abs Ramp Score"
    elif collumn == "max_ramp_score":
        return "Max Ramp Score"
    elif collumn == 'rayleigh_score':
        return 'Rayleigh Score'
    elif collumn == "rate_map_correlation_first_vs_second_half":
        return "Spatial Stability"
    elif collumn == "lm_result_b_outbound":
        return "LM Outbound fit"
    elif collumn == "lm_result_b_homebound":
        return "LM Homebound fit"
    elif collumn == "lmer_result_b_outbound":
        return "LMER Outbound fit"
    elif collumn == "lmer_result_b_homebound":
        return "LMER Homebound fit"
    elif collumn == "beaconed":
        return "Beaconed"
    elif collumn == "non-beaconed":
        return "Non Beaconed"
    elif collumn == "probe":
        return "Probe"
    elif collumn == "all":
        return "All Trial Types"
    elif collumn == "spike_ratio":
        return "Spike Ratio"
    elif collumn == "_cohort5":
        return "C5"
    elif collumn == "_cohort4":
        return "C4"
    elif collumn == "_cohort3":
        return "C3"
    elif collumn == "_cohort2":
        return "C2"
    elif collumn == "ThetaIndex_vr":
        return "Theta Index VR"
    elif collumn == "ThetaPower_vr":
        return "Theta Power VR"
    elif collumn == "ThetaIndex":
        return "Theta Index"
    elif collumn == "ThetaPower":
        return "Theta Power"
    elif collumn == 'best_theta_idx_vr':
        return "Max Theta Index VR"
    elif collumn == 'best_theta_idx_of':
        return "Max Theta Index OF"
    elif collumn == 'best_theta_idx_combined':
        return "Max Theta Index VR+OF"
    elif collumn == 'best_theta_pwr_vr':
        return "Max Theta Power VR"
    elif collumn == 'best_theta_pwr_of':
        return "Max Theta Power OF"
    elif collumn == 'best_theta_pwr_combined':
        return "Max Theta Power VR+OF"
    elif collumn == 'mean_firing_rate_vr':
        return "Mean FR - VR"
    elif collumn == 'mean_firing_rate_of':
        return "Mean FR - OF"
    else:
        print("collumn title not found!")
        return collumn

def get_score_threshold(collumn):
    if collumn == "speed_score":
        return 0.18
    elif collumn == "grid_score":
        return 0.4
    elif collumn == "border_score":
        return 0.5
    elif collumn == "corner_score":
        return 0.5
    elif collumn == "hd_score":
        return 0.5
    elif collumn == "rate_map_correlation_first_vs_second_half":
        return None

def absolute_ramp_score(data):
    absolute_ramp_scores = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        ramp_score = row["ramp_score"].iloc[0]
        absolute_ramp_scores.append(np.abs(ramp_score))
    data["abs_ramp_score"] = absolute_ramp_scores
    return data

def analyse_ramp_driver(data, trialtypes_linear_model):

    ramp_driver=[]
    for index, row in data.iterrows():
        label= "None"
        row = row.to_frame().T.reset_index(drop=True)
        session_id = row.session_id.iloc[0]
        cluster_id = row.cluster_id.iloc[0]

        lm_cluster = trialtypes_linear_model[(trialtypes_linear_model.cluster_id == cluster_id) &
                                             (trialtypes_linear_model.session_id == session_id)]

        if len(lm_cluster) > 0:

            if (lm_cluster.lm_result_b_outbound.iloc[0] == "Negative") or (lm_cluster.lm_result_b_outbound.iloc[0] == "Positive"): # significant on beaconed
                if (lm_cluster.lm_result_nb_outbound.iloc[0] == "Negative") or (lm_cluster.lm_result_nb_outbound.iloc[0] == "Positive"): # significant on non_beaconed
                    label="PI"
                elif (lm_cluster.lm_result_nb_outbound.iloc[0] == "None")  : # not significant on non beaconed
                    label="Cue"
                else:
                    print("if this prints then something is wrong")
            else:
                label="None"
        else:
            label=np.nan

        ramp_driver.append(label)

    data["ramp_driver"] = ramp_driver
    return data

def get_p_text(p, ns=False):

    if p is not None:

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

def lmer_result_color(lmer_result):
    if lmer_result=="PA":
        return ((211.0/255,118.0/255,255.0/255))
    elif lmer_result=="PS":
        return ((255.0/255,176.0/255,100.0/255))
    elif lmer_result=="A":
        return ((111.0/255, 172.0/255, 243.0/255))
    elif lmer_result=="S":
        return ((255.0/255,226.0/255,101.0/255))
    elif lmer_result=="P":
        return ((255.0/255,115.0/255,121.0/255))
    elif lmer_result=="PSA":
        return ((120.0/255,138.0/255,138.0/255))
    elif lmer_result=="SA":
        return ((153.0/255,220.0/255,97.0/255))
    elif lmer_result=="None":
        return ((216.0/255,216.0/255,216.0/255))

def lm_result_color(lm_result):
    if lm_result=="None":
        return "grey"
    elif lm_result=="Negative":
        return "red"
    elif lm_result=="Positive":
        return "blue"

def ramp_driver_color(ramp_driver):
    if ramp_driver == "PI":
        return "yellow"
    elif ramp_driver == "Cue":
        return "green"
    elif ramp_driver == "None":
        return "grey"

def max_ramp_score_label_color(max_ramp_score_label):
    if max_ramp_score_label == "outbound":
        return "cyan"
    elif max_ramp_score_label == "homebound":
        return "blue"
    elif max_ramp_score_label == "full_track":
        return "black"
    else:
        print("label color not found")

def cohort_mouse_label_color(cohort_mouse_label):
    if cohort_mouse_label == "C2_1124":
        return "C0"
    elif cohort_mouse_label == "C2_245":
        return "C1"
    elif cohort_mouse_label == "C3_M1":
        return "C2"
    elif cohort_mouse_label == "C3_M6":
        return "C3"
    elif cohort_mouse_label == "C4_M2":
        return "C4"
    elif cohort_mouse_label == "C4_M3":
        return "C5"
    elif cohort_mouse_label == "C5_M1":
        return "C6"
    elif cohort_mouse_label == "C5_M2":
        return "C7"

def label_collumn2color(data, label_collumn):
    colors=[]
    if (label_collumn == "lmer_result_homebound") or (label_collumn == "lmer_result_outbound"):
        for i in range(len(data[label_collumn])):
            colors.append(lmer_result_color(data[label_collumn].iloc[i]))
    elif (label_collumn == "lm_result_b_homebound") or (label_collumn == "lm_result_b_outbound") or \
            (label_collumn == "lm_result_p_homebound") or (label_collumn == "lm_result_p_outbound") or \
            (label_collumn == "lm_result_nb_homebound") or (label_collumn == "lm_result_nb_outbound"):
        for i in range(len(data[label_collumn])):
            colors.append(lm_result_color(data[label_collumn].iloc[i]))
    elif (label_collumn == "ramp_driver"):
        for i in range(len(data[label_collumn])):
            colors.append(ramp_driver_color(data[label_collumn].iloc[i]))
    elif (label_collumn == "max_ramp_score_label"):
        for i in range(len(data[label_collumn])):
            colors.append(max_ramp_score_label_color(data[label_collumn].iloc[i]))
    elif (label_collumn == "cohort_mouse"):
        for i in range(len(data[label_collumn])):
            colors.append(cohort_mouse_label_color(data[label_collumn].iloc[i]))
    return colors

def simple_histogram(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, filter_by_slope=False):
    fig, ax = plt.subplots(figsize=(6,6))

    PS = data[data.tetrode_location == "PS"]
    MEC = data[data.tetrode_location == "MEC"]
    UN = data[data.tetrode_location == "UN"]

    PS_neg = PS[PS.ramp_score < 0]
    MEC_neg = MEC[MEC.ramp_score < 0]
    PS_pos = PS[PS.ramp_score > 0]
    MEC_pos = MEC[MEC.ramp_score > 0]

    if trial_type == "all" and filter_by_slope==True:
        p = stats.ks_2samp(np.asarray(PS_neg[collumn]), np.asarray(MEC_neg[collumn]))[1]
        print("p =",p, "for negative slopes, ", get_p_text(p))
        p = stats.ks_2samp(np.asarray(PS_pos[collumn]), np.asarray(MEC_pos[collumn]))[1]
        print("p =",p, "for positive slopes, ", get_p_text(p))

    p = stats.ks_2samp(np.asarray(PS[collumn]), np.asarray(MEC[collumn]))[1]
    p_str = get_p_text(p, ns=True)
    #print("p=", p)

    #density_PS = kde.gaussian_kde(np.asarray(PS[collumn]).astype(float)); x = np.linspace(-1,1,300); y=density_PS(x); ax.plot(x,y, color="b");
    #density_MEC = kde.gaussian_kde(np.asarray(MEC[collumn]).astype(float)); x = np.linspace(-1,1,300); y=density_MEC(x); ax.plot(x,y, color="r");

    #PS_bw = density_PS.covariance_factor()*np.asarray(PS[collumn]).astype(float).std()
    #MEC_bw = density_MEC.covariance_factor()*np.asarray(MEC[collumn]).astype(float).std()
    #print("bandwidth for PS = ", PS_bw)
    #print("bandwidth for MEC = ", MEC_bw)

    ax.hist(np.asarray(PS_neg[collumn]), range=(-1, 1), bins=25, alpha=0.3, color="b", label="MEC", histtype="bar", density=True, cumulative=False, linewidth=4)
    density_PS_neg = kde.gaussian_kde(np.asarray(PS_neg[collumn]).astype(float)); x = np.linspace(-1,1,300); y=density_PS_neg(x); ax.plot(x,y, color="b");

    ax.hist(np.asarray(MEC_neg[collumn]), range=(-1, 1), bins=25, alpha=0.3, color="r", label="MEC", histtype="bar", density=True, cumulative=False, linewidth=4)
    density_MEC_neg = kde.gaussian_kde(np.asarray(MEC_neg[collumn]).astype(float)); x = np.linspace(-1,1,300); y=density_MEC_neg(x); ax.plot(x,y, color="r");


    ax.hist(np.asarray(PS_pos[collumn]), range=(-1, 1), bins=25, alpha=0.3, color="b", label="PS", histtype="bar", density=True, cumulative=False, linewidth=4)
    density_PS_pos = kde.gaussian_kde(np.asarray(PS_pos[collumn]).astype(float)); x = np.linspace(-1,1,300); y=density_PS_pos(x); ax.plot(x,y, color="b");

    ax.hist(np.asarray(MEC_pos[collumn]), range=(-1, 1), bins=25, alpha=0.3, color="r", label="MEC", histtype="bar", density=True, cumulative=False, linewidth=4)
    density_MEC_pos = kde.gaussian_kde(np.asarray(MEC_pos[collumn]).astype(float)); x = np.linspace(-1,1,300); y=density_MEC_pos(x); ax.plot(x,y, color="r");



    ax.set_ylabel("Density", fontsize=25)
    ax.set_xlabel(get_tidy_title(collumn), fontsize=25)
    if collumn == "ramp_score":
        ax.set_xlim(left=-0.75, right=0.75)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlim(left=-1, right=1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2)
    #ax.legend(loc="upper right")
    ax.set_xlim(left=-1, right=1)
    #plt.subplots_adjust(top=0.8)

    ax.text(0.1, 0.9, p_str, ha='center', va='center', transform=ax.transAxes, fontsize=12)
    if save_path is not None:
        if filter_by_slope:
            plt.savefig(save_path+"/FBS_location_histo_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
        else:
            plt.savefig(save_path+"/location_histo_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)


    plt.show()
    plt.close()
    return

def simple_boxplot(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, filter_by_slope=False):
    fig, ax = plt.subplots(figsize=(6,3))


    PS = data[data.tetrode_location == "PS"]
    MEC = data[data.tetrode_location == "MEC"]
    UN = data[data.tetrode_location == "UN"]

    p = stats.ks_2samp(np.asarray(PS[collumn]), np.asarray(MEC[collumn]))[1]
    p_str = get_p_text(p, ns=True)
    #ax.text(0.1, 0.9, p_str, ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)
    ax.set_title(p_str, fontsize=20)
    objects = ("PS", "MEC", "UN")
    objects = ("PS", "MEC")
    y_pos = np.arange(len(objects))

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')

    bplot1 = ax.boxplot(np.asarray(PS[collumn]), positions = [0], widths=0.9,
                        boxprops=boxprops, medianprops=medianprops,
                        whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, vert=False)

    bplot2 = ax.boxplot(np.asarray(MEC[collumn]), positions = [1], widths=0.9,
                        boxprops=boxprops, medianprops=medianprops,
                        whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, vert=False)

    #bplot3 = ax.boxplot(np.asarray(UN[collumn]), positions = [2], widths=0.9,
    #                    boxprops=boxprops, medianprops=medianprops,
    #                    whiskerprops=whiskerprops, capprops=capprops, patch_artist=True)

    # fill with colors
    colors = ['b', 'r', 'grey']
    colors = ['b', 'r']
    i=0
    #for bplot in (bplot1, bplot2, bplot3):
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(colors[i])
        i+=1

    #ax.text(0.95, 1.25, "p= "+str(np.round(p, decimals=4)), ha='right', va='top', transform=ax.transAxes, fontsize=20)
    plt.yticks(y_pos, objects, fontsize=25)
    plt.xlabel(get_tidy_title(collumn),  fontsize=25)
    plt.ylim((-1,3))
    plt.ylim((-0.75,1.5))
    if collumn == "ramp_score":
        ax.set_xlim(left=-1, right=1)
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    #plt.title('Programming language usage')
    #ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        if filter_by_slope:
            plt.savefig(save_path+"/FBS_location_boxplot_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
        else:
            plt.savefig(save_path+"/location_boxplot_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
    plt.show()
    plt.close()

def simple_bar_mouse(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False, filter_by_slope=False):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)

    objects = np.unique(data["cohort_mouse"])
    x_pos = np.arange(len(objects))

    use_color_cycle=True
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(len(objects)):
        y = data[(data["cohort_mouse"] == objects[i])]

        if use_color_cycle:
            ax.errorbar(x_pos[i], np.mean(np.asarray(y[collumn])), yerr=stats.sem(np.asarray(y[collumn])), ecolor=cycle_colors[i], capsize=10, fmt="o", color=cycle_colors[i])
            ax.scatter(x_pos[i]*np.ones(len(np.asarray(y[collumn]))), np.asarray(y[collumn]), edgecolor=cycle_colors[i], marker="o", facecolors='none')

        else:
            ax.errorbar(x_pos[i], np.mean(np.asarray(y[collumn])), yerr=stats.sem(np.asarray(y[collumn])), ecolor='black', capsize=10, fmt="o", color="black")
            ax.scatter(x_pos[i]*np.ones(len(np.asarray(y[collumn]))), np.asarray(y[collumn]), edgecolor="black", marker="o", facecolors='none')
            #ax.bar(x_pos[i], np.mean(np.asarray(y[collumn])), yerr=stats.sem(np.asarray(y[collumn])), align='center', alpha=0.5, ecolor='black', capsize=10)

        #ax.bar(x_pos[i], np.mean(np.asarray(y[collumn])), yerr=stats.sem(np.asarray(y[collumn])), align='center', alpha=0.5, ecolor='black', capsize=10)

    #ax.text(0.95, 1, p_str, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    plt.xticks(x_pos, objects, fontsize=8)
    plt.xticks(rotation=-45)
    plt.locator_params(axis='y', nbins=4)
    plt.ylabel(get_tidy_title(collumn),  fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    #if collumn == "ramp_score":
    #    plt.ylim(-0.6, 0.6)
    #elif collumn == "abs_ramp_score":
    #    plt.ylim(0, 0.6)
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    #plt.title('Programming language usage')
    #ax.legend()

    if print_p:
        print(p)

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        if filter_by_slope:
            plt.savefig(save_path+"/FBS_mouse_bar_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
        else:
            plt.savefig(save_path+"/mouse_bar_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
    plt.show()
    plt.close()

def simple_bar_location(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False, filter_by_slope=False):
    fig, ax = plt.subplots(figsize=(3,6.5))
    p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)

    objects = np.unique(data["tetrode_location"])
    x_pos = np.arange(len(objects))

    for i in range(len(objects)):
        y = data[(data["tetrode_location"] == objects[i])]
        ax.errorbar(x_pos[i], np.mean(np.asarray(y[collumn])), yerr=stats.sem(np.asarray(y[collumn])), ecolor='black', capsize=10, fmt="o", color="black")
        ax.scatter(x_pos[i]*np.ones(len(np.asarray(y[collumn]))), np.asarray(y[collumn]), edgecolor="black", marker="o", facecolors='none')
        #ax.bar(x_pos[i], np.mean(np.asarray(y[collumn])), yerr=stats.sem(np.asarray(y[collumn])), align='center', alpha=0.5, ecolor='black', capsize=10)

    #ax.text(0.95, 1, p_str, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    plt.xticks(x_pos, objects, fontsize=8)
    plt.xticks(rotation=-45)
    plt.ylabel(get_tidy_title(collumn),  fontsize=20)
    plt.locator_params(axis='y', nbins=4)
    plt.xlim((-0.5, len(objects)-0.5))
    #if collumn == "ramp_score":
    #    plt.ylim(-0.6, 0.6)
    #elif collumn == "abs_ramp_score":
    #    plt.ylim(0, 0.6)
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    #plt.title('Programming language usage')
    #ax.legend()

    if print_p:
        print(p)

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        if filter_by_slope:
            plt.savefig(save_path+"/FBS_location_bar_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
        else:
            plt.savefig(save_path+"/location_bar_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
    plt.show()
    plt.close()

def simple_lm_stack_mouse(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False):
    fig, ax = plt.subplots(figsize=(3,6))
    #p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)

    aggregated = data.groupby([collumn, "cohort_mouse"]).count().reset_index()
    if (collumn == "lm_result_hb") or (collumn == "lm_result_ob"):
        colors_lm = [((238.0/255,58.0/255,140.0/255)), ((102.0/255,205.0/255,0.0/255)), "black", "grey"]
        groups = ["Negative", "Positive", "None", "NoSlope"]
    elif (collumn == "ramp_driver"):
        colors_lm = ["grey", "green", "yellow"]
        groups = [ "None", "PI", "Cue"]
    else:
        colors_lm = ["lightgrey", "lightslategray", "limegreen", "violet", "orange",
                     "cornflowerblue", "yellow", "lightcoral"]
        groups = ["P", "S", "A", "PS", "PA", "SA", "PSA", "None"]
        groups = ["None", "PSA", "SA", "PA", "PS", "A", "S", "P"]

    objects = np.unique(aggregated["cohort_mouse"])
    x_pos = np.arange(len(objects))

    for object, x in zip(objects, x_pos):
        tetrode_location = aggregated[aggregated["cohort_mouse"] == object]

        bottom=0
        for color, group in zip(colors_lm, groups):
            count = tetrode_location[(tetrode_location[collumn] == group)]["Unnamed: 0"]
            if len(count)==0:
                count = 0
            else:
                count = int(count)

            percent = (count/np.sum(tetrode_location["Unnamed: 0"]))*100
            ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
            bottom = bottom+percent

    #ax.text(0.95, 1, p_str, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    plt.xticks(x_pos, objects, fontsize=8)
    plt.xticks(rotation=-45)
    plt.ylabel("Percent of neurons",  fontsize=20)

    plt.xlim((-0.5, len(objects)-0.5))
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    #plt.title('Programming language usage')
    #ax.legend()

    if print_p:
        print(p)

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+"/mouse_slope_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
    plt.show()
    plt.close()


def simple_lm_stack(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False):
    fig, ax = plt.subplots(figsize=(3,6))
    data = data[(data["tetrode_location"] != "V1")]

    #p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)

    aggregated = data.groupby([collumn, "tetrode_location"]).count().reset_index()
    if (collumn == "lm_result_hb") or (collumn == "lm_result_ob"):
        colors_lm = [((238.0/255,58.0/255,140.0/255)), ((102.0/255,205.0/255,0.0/255)), "black"]
        groups = ["Negative", "Positive", "Unclassified"]
    elif (collumn == "ramp_driver"):
        colors_lm = ["grey", "green", "yellow"]
        groups = [ "None", "PI", "Cue"]
    else:
        colors_lm = ["lightgrey", "lightslategray", "limegreen", "violet", "orange",
                     "cornflowerblue", "yellow", "lightcoral"]
        groups = ["P", "S", "A", "PS", "PA", "SA", "PSA", "None"]
        groups = ["None", "PSA", "SA", "PA", "PS", "A", "S", "P"]
        colors_lm = [lmer_result_color(c) for c in groups]

    objects = np.unique(aggregated["tetrode_location"])
    x_pos = np.arange(len(objects))

    for object, x in zip(objects, x_pos):
        tetrode_location = aggregated[aggregated["tetrode_location"] == object]

        bottom=0
        for color, group in zip(colors_lm, groups):
            count = tetrode_location[(tetrode_location[collumn] == group)]["Unnamed: 0"]
            if len(count)==0:
                count = 0
            else:
                count = int(count)

            percent = (count/np.sum(tetrode_location["Unnamed: 0"]))*100
            ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
            bottom = bottom+percent

    #ax.text(0.95, 1, p_str, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    plt.xticks(x_pos, objects, fontsize=8)
    plt.xticks(rotation=-45)
    plt.ylabel("Percent of neurons",  fontsize=25)

    plt.xlim((-0.5, len(objects)-0.5))
    plt.ylim((0,100))
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    #plt.title('Programming language usage')
    #ax.legend()

    if print_p:
        print(p)

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+"/location_slope_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
    plt.show()
    plt.close()

def add_theta_modulated_marker(data, threshold=0.07):
    ThetaIndexLabel = []

    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        ThetaIndex = row["ThetaIndex"].iloc[0]

        if ThetaIndex> threshold:
            binary = "TR"
        else:
            binary = "NR"

        ThetaIndexLabel.append(binary)

    data["ThetaIndexLabel"] = ThetaIndexLabel
    return data

def simple_lm_stack_theta(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False):

    # were only interested in the postive and negative sloping neurons when looking at the proportions of lmer neurons
    if (collumn == "lmer_result_ob"):
        data = data[(data["lm_result_ob"] == "Positive") | (data["lm_result_ob"] == "Negative")]
    elif (collumn == "lmer_result_hb"):
        data = data[(data["lm_result_hb"] == "Positive") | (data["lm_result_hb"] == "Negative")]


    fig, ax = plt.subplots(figsize=(3,6))
    data = add_theta_modulated_marker(data)

    aggregated = data.groupby([collumn, "ThetaIndexLabel"]).count().reset_index()
    if (collumn == "lm_result_hb") or (collumn == "lm_result_ob"):
        colors_lm = [((238.0/255,58.0/255,140.0/255)), ((102.0/255,205.0/255,0.0/255)), "black"]
        groups = ["Negative", "Positive", "Unclassified"]
    elif (collumn == "ramp_driver"):
        colors_lm = ["grey", "green", "yellow"]
        groups = [ "None", "PI", "Cue"]
    else:
        groups = ["None", "PSA", "SA", "PA", "PS", "A", "S", "P"]
        colors_lm = [lmer_result_color(c) for c in groups]

    objects = np.unique(aggregated["ThetaIndexLabel"])
    x_pos = np.arange(len(objects))

    for object, x in zip(objects, x_pos):
        ThetaIndexLabel = aggregated[aggregated["ThetaIndexLabel"] == object]

        bottom=0
        for color, group in zip(colors_lm, groups):
            count = ThetaIndexLabel[(ThetaIndexLabel[collumn] == group)]["ThetaIndex"]
            if len(count)==0:
                count = 0
            else:
                count = int(count)

            print("stack_theta, ramp_region=", ramp_region, " , group=", group, ", theta=", object, "count=", count)

            percent = (count/np.sum(ThetaIndexLabel["ThetaIndex"]))*100
            ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
            bottom = bottom+percent

    #ax.text(0.95, 1, p_str, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    plt.xticks(x_pos, objects, fontsize=15)
    #plt.xticks(rotation=-45)
    plt.ylabel("Percent of neurons",  fontsize=25)
    plt.xlim((-0.5, len(objects)-0.5))
    plt.ylim((0,100))

    if print_p:
        print(p)

    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+"/ThetaIndexLabel_stack_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png", dpi=300)
    plt.show()
    plt.close()


def simple_lm_stack_negpos(combined_df, collumn, save_path=None, track_region=None, trial_type=None, p=None, print_p=False):

    df = combined_df[(combined_df["trial_type"] == trial_type) &
                     (combined_df["track_region"] == track_region)]

    df.dropna(subset = ["linear_model_class", "mixed_model_class"], inplace=True)

    fig, ax = plt.subplots(figsize=(4,6))
    groups = ["None", "PSA", "SA", "PA", "PS", "A", "S", "P"]
    colors_lm = [lmer_result_color(c) for c in groups]

    objects = ["Negative", "Positive", "Unclassified"]
    x_pos = np.arange(len(objects))

    for object, x in zip(objects, x_pos):
        ramp_class =  df[df["linear_model_class"] == object]

        bottom=0
        for color, group in zip(colors_lm, groups):
            count = len(ramp_class[(ramp_class["mixed_model_class"] == group)])
            percent = (count/len(ramp_class))*100
            ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
            ax.text(x,bottom, str(count), color="k", fontsize=10, ha="center")
            bottom = bottom+percent

    #ax.text(0.95, 1, p_str, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    plt.xticks(x_pos, objects, fontsize=15)
    plt.xticks(rotation=+45)
    plt.ylabel("Percent of neurons",  fontsize=25)
    plt.xlim((-0.5, len(objects)-0.5))
    plt.ylim((0,100))
    if print_p:
        print(p)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    trial_type_str = get_trial_type(trial_type)
    plt.savefig(save_path+"negpos_stack_tt_"+trial_type_str+"_rr_"+track_region+"_"+collumn+".png", dpi=300)
    plt.show()
    plt.close()


def add_locations(ramp_scores_df, tetrode_locations_df):

    data = pd.DataFrame()
    for index, row_ramp_score in ramp_scores_df.iterrows():
        row_ramp_score =  row_ramp_score.to_frame().T.reset_index(drop=True)
        session_id_short = row_ramp_score.session_id_short.iloc[0]

        print("processing "+session_id_short)

        session_tetrode_info = tetrode_locations_df[(tetrode_locations_df.session_id_short == session_id_short)]
        row_ramp_score["tetrode_location"] = session_tetrode_info.estimated_location.iloc[0]
        data = pd.concat([data, row_ramp_score], ignore_index=True)
    return data

def get_day(full_session_id):
    session_id = full_session_id.split("/")[-1]
    training_day = session_id.split("_")[1]
    training_day = training_day.split("D")[1]
    training_day = ''.join(filter(str.isdigit, training_day))
    return int(training_day)

def get_year(session_id):
    for i in range(11, 30):
        if "20"+str(i) in session_id:
            return "20"+str(i)

def get_suedo_day(full_session_id):
    session_id = full_session_id.split("/")[-1]
    year = get_year(session_id)
    tmp = session_id.split(year)
    month = tmp[1].split("-")[1]
    day = tmp[1].split("-")[2].split("_")[0]
    return(int(year+month+day)) # this ruturns a useful number in terms of the order of recordings

def get_cohort(full_session_id):
    if "Klara" in full_session_id:
        return "K"
    if "Bri" in full_session_id:
        return "B"
    if "Junji" in full_session_id:
        return "J"
    if "Ian" in full_session_id:
        return "I"
    if "Cohort6_july2020" in full_session_id:
        return "H2"
    if "Cohort7_october2020" in full_session_id:
        return "H3"
    if "Cue_conditioned_cohort1_190902" in full_session_id:
        return "H1"

    elements = full_session_id.split("/")
    for i in range(len(elements)):
        if "cohort" in elements[i]:
            return elements[i]

def get_recording_paths(path_list, folder_path):
    list_of_recordings = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for recording_path in list_of_recordings:
        path_list.append(recording_path)
    return path_list

def get_lmer_colours(lmer_results):
    tmp=[]
    for lmer_result in lmer_results:
        tmp.append(lmer_result_color(lmer_result))
    return tmp

def firing_rate_vr_vs_of(all_mice, save_path, ramp_region, trial_type):

    df_regionx = all_mice[(all_mice["ramp_region"] == ramp_region) &
                          (all_mice["trial_type"] == trial_type)]

    df_regionx["mean_firing_rate_vr"] = pd.to_numeric(df_regionx["mean_firing_rate_vr"])
    df_regionx["mean_firing_rate"] = pd.to_numeric(df_regionx["mean_firing_rate"])

    if ramp_region == "outbound":
        lmer_collumn = "lmer_result_ob"
    elif ramp_region == "homebound":
        lmer_collumn = "lmer_result_hb"

    mean_fr_vr = df_regionx.groupby([lmer_collumn])["mean_firing_rate_vr"].mean().reset_index()
    mean_fr_of = df_regionx.groupby([lmer_collumn])["mean_firing_rate"].mean().reset_index()
    sem_fr_vr = df_regionx.groupby([lmer_collumn])["mean_firing_rate_vr"].sem().reset_index()
    sem_fr_of = df_regionx.groupby([lmer_collumn])["mean_firing_rate"].sem().reset_index()

    fig, ax = plt.subplots()
    #plt.title("rs: "+str(np.round(ramp_score, decimals=2)), fontsize=20)
    plt.xlabel("Cell Types", fontsize=20, labelpad=10)
    plt.ylim([0, 30])
    plt.xticks(np.arange(0, len(mean_fr_of)), labels=mean_fr_of[lmer_collumn], fontsize=10)
    plt.ylabel("Mean Firing Rate", fontsize=20, labelpad=10)
    ax.bar(x=np.arange(0, len(mean_fr_of))-0.15, height=np.asarray(mean_fr_vr["mean_firing_rate_vr"]), color=get_lmer_colours(mean_fr_of[lmer_collumn]), width=0.3, yerr=np.asarray(sem_fr_vr["mean_firing_rate_vr"]), ecolor='black', capsize=10)
    ax.bar(x=np.arange(0, len(mean_fr_of))+0.15, height=np.asarray(mean_fr_of["mean_firing_rate"]), color=get_lmer_colours(mean_fr_of[lmer_collumn]), hatch="//",width=0.3,  yerr=np.asarray(sem_fr_of["mean_firing_rate"]), ecolor='black', capsize=10)
    #ax.errorbar(x=np.arange(0, len(mean_fr_of))-0.15, y=np.asarray(mean_fr_vr["mean_firing_rate_vr"]), yerr=np.asarray(sem_fr_vr["mean_firing_rate_vr"]), color="k", linewidth=0, capsize=0.1)
    #ax.errorbar(x=np.arange(0, len(mean_fr_of))+0.15, y=np.asarray(mean_fr_of["mean_firing_rate"]), yerr=np.asarray(sem_fr_of["mean_firing_rate"]), color="k", linewidth=0, capsize=0.1)
    plt.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path+"/fr_comparision_lmer"+ramp_region+".png", dpi=300)
    plt.close()

    print("finished with tis ghrng ")

def remove_mouse(data, cohort_mouse_list):

    for cohort_mouse in cohort_mouse_list:
        data = data[(data["cohort_mouse"] != cohort_mouse)]
    return data

def remove_location_classification(data, locations):

    for location in locations:
        data = data[(data["tetrode_location"]) != location]
    return data


def plot_histograms_for_mm(combined_df, trial_type, track_region, column, save_path):

    df = combined_df[(combined_df["trial_type"] == trial_type) &
                     (combined_df["track_region"] == track_region)]

    df.dropna(subset = ["linear_model_class", "mixed_model_class"], inplace=True)

    for mixed_type in np.unique(df["mixed_model_class"]):
        mixed_type_df = df[(df["mixed_model_class"] == mixed_type)]

        fig, ax = plt.subplots(figsize=(3,4))
        ax.hist(np.asarray(mixed_type_df[column]), bins=20, alpha=0.5, color=lmer_result_color(mixed_type))
        plt.xlabel(get_tidy_title(column),  fontsize=15)
        plt.title(mixed_type, fontsize=15)
        plt.xlim(get_limits(column))
        #plt.xticks([0, 0.5])
        plt.ylabel("Number of Cells",  fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        trial_type_str = get_trial_type(trial_type)
        plt.tight_layout()
        plt.savefig(save_path+"/"+column+"_"+trial_type_str+"_"+track_region+"_"+mixed_type+"_mm_histo.png", dpi=300)
        plt.show()

def plot_histograms_for_mm_by_encoder(combined_df, trial_type, track_region, column, save_path):

    df = combined_df[(combined_df["trial_type"] == trial_type) &
                     (combined_df["track_region"] == track_region)]

    df.dropna(subset = ["linear_model_class", "mixed_model_class"], inplace=True)

    for encoder in ["P", "S", "A"]:

        # select all neurons with any one of the encoders
        encoder_df = pd.DataFrame()
        for index, row in df.iterrows():
            row = row.to_frame().T.reset_index(drop=True)
            mixed_model_class = row["mixed_model_class"].iloc[0]
            all_classes = list(mixed_model_class)
            if encoder in all_classes:
                encoder_df = pd.concat([encoder_df, row], ignore_index=True)

        fig, ax = plt.subplots(figsize=(3,4))
        ax.hist(np.asarray(encoder_df[column]), bins=20, alpha=0.5, color=lmer_result_color(encoder))
        plt.xlabel(get_tidy_title(column),  fontsize=15)
        plt.title(encoder, fontsize=15)
        #plt.xticks([0, 0.5])
        plt.xlim(get_limits(column))
        plt.ylabel("Number of Cells",  fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        trial_type_str = get_trial_type(trial_type)
        plt.tight_layout()
        plt.savefig(save_path+"/"+column+"_"+trial_type_str+"_"+track_region+"_"+encoder+"_mm_histo.png", dpi=300)
        plt.show()

def plot_bar_chart_for_mm_by_encoder(combined_df, trial_type, track_region, column, save_path):

    df = combined_df[(combined_df["trial_type"] == trial_type) &
                     (combined_df["track_region"] == track_region)]

    df.dropna(subset = ["linear_model_class", "mixed_model_class"], inplace=True)

    fig, ax = plt.subplots(figsize=(3,2))
    top = get_limits(column)[-1]
    P = df[df["mixed_model_class"] == "P"]
    A = df[df["mixed_model_class"] == "A"]
    S = df[df["mixed_model_class"] == "S"]

    ax.scatter(np.repeat(0.5, len(P)), np.asarray(P[column]), color=lmer_result_color("P"), alpha=0.1)
    ax.errorbar(0.5, np.nanmean(np.asarray(P[column])), yerr= scipy.stats.sem(np.asarray(P[column]), nan_policy="omit"), color=lmer_result_color("P"), capsize=12)

    ax.scatter(np.repeat(1.5, len(A)), np.asarray(A[column]), color=lmer_result_color("A"), alpha=0.1)
    ax.errorbar(1.5, np.nanmean(np.asarray(A[column])), yerr= scipy.stats.sem(np.asarray(A[column]), nan_policy="omit"), color=lmer_result_color("A"), capsize=12)

    ax.scatter(np.repeat(2.5, len(S)), np.asarray(S[column]), color=lmer_result_color("S"), alpha=0.1)
    ax.errorbar(2.5, np.nanmean(np.asarray(S[column])), yerr= scipy.stats.sem(np.asarray(S[column]), nan_policy="omit"), color=lmer_result_color("S"), capsize=12)

    PA_p = stats.ks_2samp(np.asarray(P[column]), np.asarray(A[column]))[1]
    PA_p_str = get_p_text(PA_p, ns=True)
    ax.text(1, top-0.125, PA_p_str, color="k", fontsize=10, ha="center")
    ax.plot([0.5,1.5], [top-0.15, top-0.15],"-k")

    PS_p = stats.ks_2samp(np.asarray(P[column]), np.asarray(S[column]))[1]
    PS_p_str = get_p_text(PS_p, ns=True)
    ax.text(1.5, top-0.025, PS_p_str, color="k", fontsize=10, ha="center")
    ax.plot([0.5,2.5], [top-0.05, top-0.05],"-k")

    AS_p = stats.ks_2samp(np.asarray(A[column]), np.asarray(S[column]))[1]
    AS_p_str = get_p_text(AS_p, ns=True)
    ax.text(2, top-0.225, AS_p_str, color="k", fontsize=10, ha="center")
    ax.plot([1.5,2.5], [top-0.25, top-0.25],"-k")

    plt.ylim(get_limits(column))
    plt.xlim((0,3))
    plt.xticks([0.5, 1.5, 2.5], ["P", "A", "S"], fontsize=15)
    plt.ylabel(get_tidy_title(column),  fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    trial_type_str = get_trial_type(trial_type)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path+"/"+column+"_"+trial_type_str+"_"+track_region+"_mm_bar.png", dpi=300)
    plt.show()

def plot_cumhist_for_mm_by_encoder(combined_df, trial_type, track_region, column, save_path):

    df = combined_df[(combined_df["trial_type"] == trial_type) &
                     (combined_df["track_region"] == track_region)]

    df.dropna(subset = ["linear_model_class", "mixed_model_class"], inplace=True)

    fig, ax = plt.subplots(figsize=(3,4))

    P = df[df["mixed_model_class"] == "P"]
    A = df[df["mixed_model_class"] == "A"]
    S = df[df["mixed_model_class"] == "S"]

    _, _, patchesP = ax.hist(np.asarray(P[column]), bins=500, color=lmer_result_color("P"), histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesA = ax.hist(np.asarray(A[column]), bins=500, color=lmer_result_color("A"), histtype="step", density=True, cumulative=True, linewidth=2)
    _, _, patchesS = ax.hist(np.asarray(S[column]), bins=500, color=lmer_result_color("S"), histtype="step", density=True, cumulative=True, linewidth=2)

    patchesP[0].set_xy(patchesP[0].get_xy()[:-1])
    patchesA[0].set_xy(patchesA[0].get_xy()[:-1])
    patchesS[0].set_xy(patchesS[0].get_xy()[:-1])

    plt.xlim(get_limits(column))
    plt.ylabel("Density",  fontsize=20)
    plt.xlabel(get_tidy_title(column),  fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    trial_type_str = get_trial_type(trial_type)
    plt.tight_layout()

    plt.savefig(save_path+"/"+column+"_"+trial_type_str+"_"+track_region+"_mm_cumhist.png", dpi=300)
    plt.show()

def get_trial_type(tt):
    if tt == 0:
        return "beaconed"
    elif tt == 1:
        return "non_beaconed"
    elif tt == 2:
        return "probe"

def get_limits(column):
    if column == "border_score":
        return (-0.7, 0.9)
    if column == "ThetaIndex":
        return (0, 0.7)
    if column == "speed_score":
        return (-1, 1)
    if column == "grid_score":
        return (-0.6, 0.6)
    if column == "hd_score":
        return (0, 1)
    if column == "mean_firing_rate_of":
        return (0, 55)
    if column == "mean_firing_rate_vr":
        return (0, 55)
    if column == "rate_map_correlation_first_vs_second_half":
        return (-1, 1)
    if column == "abs_speed_score":
        return (0,1)

def add_abs_speed_score(combined_df):
    abs_speed_score = []
    for index, row in combined_df.iterrows():
        row = row.to_frame().T.reset_index(drop=True)
        speed_score = row["speed_score"].iloc[0]
        abs_speed_score.append(np.abs(speed_score))
    combined_df["abs_speed_score"] = abs_speed_score
    return combined_df

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    firing_rate_threshold = 1

    combined_df = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/summary/All_mice_vr_and_of_metrics.pkl")
    combined_df = combined_df[(combined_df["mean_firing_rate_of"] > firing_rate_threshold)]
    combined_df = add_abs_speed_score(combined_df)

    # this recreates the proportions on mixed model types within the linear model types
    simple_lm_stack_negpos(combined_df, collumn="linear_model_class", save_path="/mnt/datastore/Harry/Ramp_cells_open_field_paper/", track_region="outbound", trial_type=0, p=None)
    simple_lm_stack_negpos(combined_df, collumn="linear_model_class", save_path="/mnt/datastore/Harry/Ramp_cells_open_field_paper/", track_region="homebound", trial_type=0, p=None)


    for trial_type in [0]:
        for track_region in ["outbound"]:
            for column in ["border_score", "ThetaIndex", "grid_score", "speed_score", "abs_speed_score", "hd_score", "mean_firing_rate_of", "mean_firing_rate_vr", "rate_map_correlation_first_vs_second_half"]:
                #plot_histograms_for_mm(combined_df, trial_type, track_region, column, save_path="/mnt/datastore/Harry/Cohort7_october2020/summary/histograms_for_mm")
                #plot_histograms_for_mm_by_encoder(combined_df, trial_type, track_region, column, save_path="/mnt/datastore/Harry/Cohort7_october2020/summary/histograms_for_mm_encoder")
                #plot_bar_chart_for_mm_by_encoder(combined_df, trial_type, track_region, column, save_path="/mnt/datastore/Harry/Cohort7_october2020/summary/bar_chart_for_mm_encoder")
                plot_cumhist_for_mm_by_encoder(combined_df, trial_type, track_region, column, save_path="/mnt/datastore/Harry/Cohort7_october2020/summary/cumulative_histograms_for_encoders")
    print("look now ")

if __name__ == '__main__':
    main()




