import pandas as pd
import pickle5 as pkl5
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
import itertools
import scipy
from scipy import stats
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
        print("p =",p, "for negative slopes")
        p = stats.ks_2samp(np.asarray(PS_pos[collumn]), np.asarray(MEC_pos[collumn]))[1]
        print("p =",p, "for positive slopes")

    p = stats.ks_2samp(np.asarray(PS[collumn]), np.asarray(MEC[collumn]))[1]
    p_str = get_p_text(p, ns=True)
    #print("p=", p)

    #ax.hist(np.asarray(UN[collumn]), bins=50, alpha=0.2, color="k", label="Unclassified", histtype="step", density=True)
    ax.hist(np.asarray(MEC[collumn]), bins=50, alpha=0.5, color="r", label="MEC", histtype="bar", density=False, cumulative=False, linewidth=4)
    ax.hist(np.asarray(PS[collumn]), bins=50, alpha=0.5, color="b", label="PS", histtype="bar", density=False, cumulative=False, linewidth=4)

    ax.set_ylabel("Counts", fontsize=25)
    ax.set_xlabel(get_tidy_title(collumn), fontsize=25)
    if collumn == "ramp_score":
        ax.set_xlim(left=-0.75, right=0.75)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlim(left=-1, right=1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    plt.locator_params(axis='x', nbins=5)
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
    plt.ylabel("Percent of neurons",  fontsize=20)

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
    data = data[(data["tetrode_location"] != "V1")]
    data = add_theta_modulated_marker(data)

    aggregated = data.groupby([collumn, "ThetaIndexLabel"]).count().reset_index()
    if (collumn == "lm_result_hb") or (collumn == "lm_result_ob"):
        colors_lm = [((238.0/255,58.0/255,140.0/255)), ((102.0/255,205.0/255,0.0/255)), "black", "grey"]
        groups = ["Negative", "Positive", "None", "NoSlope"]
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
            count = ThetaIndexLabel[(ThetaIndexLabel[collumn] == group)]["Unnamed: 0"]
            if len(count)==0:
                count = 0
            else:
                count = int(count)

            percent = (count/np.sum(ThetaIndexLabel["Unnamed: 0"]))*100
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

def add_short_session_id(df):

    data = pd.DataFrame()
    for index, row in df.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        session_id = row.session_id.iloc[0]
        session_id_short = "_".join(session_id.split("_")[0:3])
        row["session_id_short"] = session_id_short
        data = pd.concat([data, row], ignore_index=True)
    return data

def add_theta(data, theta_df):
    data_new = pd.DataFrame()
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        session_id = row.session_id.iloc[0]
        cluster_id = row.cluster_id.iloc[0]
        thetaIdx = theta_df[(theta_df.session_id == session_id) & (theta_df.cluster_id == cluster_id)].iloc[0].ThetaIndex
        thetaPwr = theta_df[(theta_df.session_id == session_id) & (theta_df.cluster_id == cluster_id)].iloc[0].ThetaPower
        Boccara_theta_class = theta_df[(theta_df.session_id == session_id) & (theta_df.cluster_id == cluster_id)].iloc[0].Boccara_theta_class
        row["ThetaIndex"] = thetaIdx
        row["ThetaPower"] = thetaPwr
        row["Boccara_theta_class"] = Boccara_theta_class
        data_new = pd.concat([data_new, row], ignore_index=True)
    return data_new

def add_lm(data, linear_model_df):
    data_new = pd.DataFrame()
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        session_id = row.session_id.iloc[0]
        if "cluster_id" in list(row):
            cluster_id = row.cluster_id.iloc[0]
        elif "sorted_seperately_vr_cluster_ids" in list(row):
            cluster_id = row.sorted_seperately_vr_cluster_ids.iloc[0]

        if len(linear_model_df[(linear_model_df.session_id == session_id) & (linear_model_df.cluster_id == cluster_id)])>0:
            lm_result_hb = linear_model_df[(linear_model_df.session_id == session_id) & (linear_model_df.cluster_id == cluster_id)].iloc[0].lm_result_homebound
            lm_result_ob = linear_model_df[(linear_model_df.session_id == session_id) & (linear_model_df.cluster_id == cluster_id)].iloc[0].lm_result_outbound
            lmer_result_ob = linear_model_df[(linear_model_df.session_id == session_id) & (linear_model_df.cluster_id == cluster_id)].iloc[0].lmer_result_outbound
            lmer_result_hb = linear_model_df[(linear_model_df.session_id == session_id) & (linear_model_df.cluster_id == cluster_id)].iloc[0].lmer_result_homebound
        else:
            lm_result_hb = np.nan
            lm_result_ob = np.nan
            lmer_result_ob = np.nan
            lmer_result_hb = np.nan

        row["lm_result_hb"] = lm_result_hb
        row["lm_result_ob"] = lm_result_ob
        row["lmer_result_ob"] = lmer_result_ob
        row["lmer_result_hb"] = lmer_result_hb
        data_new = pd.concat([data_new, row], ignore_index=True)
    return data_new


def mouse_ramp(data, collumn, save_path, ramp_region="outbound", trial_type="beaconed", print_p=False, filter_by_slope=False):
    if filter_by_slope:
        if ramp_region == "outbound":
            data = data[(data.lm_result_ob == "Positive") | (data.lm_result_ob == "Negative")]
        elif ramp_region == "homebound":
            data = data[(data.lm_result_hb == "Positive") | (data.lm_result_hb == "Negative")]
        elif ramp_region == "all":
            data = data[(data.lm_result_ob == "Positive") | (data.lm_result_ob == "Negative") |
                        (data.lm_result_hb == "Positive") | (data.lm_result_hb == "Negative")]

    # only look at beacoend and outbound
    data = data[(data.trial_type == trial_type) &
                (data.ramp_region == ramp_region)]
    simple_histogram(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, filter_by_slope=filter_by_slope)
    #simple_boxplot(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, filter_by_slope=filter_by_slope)
    simple_bar_mouse(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, print_p=print_p, filter_by_slope=filter_by_slope)
    return

def location_ramp(data, collumn, save_path, ramp_region="outbound", trial_type="beaconed", print_p=False, filter_by_slope=False):
    print("running location_ramp")

    if filter_by_slope:
        if ramp_region == "outbound":
            data = data[(data.lm_result_ob == "Positive") | (data.lm_result_ob == "Negative")]
        elif ramp_region == "homebound":
            data = data[(data.lm_result_hb == "Positive") | (data.lm_result_hb == "Negative")]
        elif ramp_region == "all":
            data = data[(data.lm_result_ob == "Positive") | (data.lm_result_ob == "Negative") |
                        (data.lm_result_hb == "Positive") | (data.lm_result_hb == "Negative")]

    # only look at beacoend and outbound
    data = data[(data.trial_type == trial_type) &
                (data.ramp_region == ramp_region)]

    simple_histogram(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, filter_by_slope=filter_by_slope)
    simple_boxplot(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, filter_by_slope=filter_by_slope)
    simple_bar_location(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, print_p=print_p, filter_by_slope=filter_by_slope)
    return

def mouse_slope(data, collumn, save_path, ramp_region="outbound", trial_type="beaconed", print_p=False):
    data = data[(data[collumn] != np.nan)]

    # only look at beacoend and outbound
    data = data[(data.trial_type == trial_type) &
                (data.ramp_region == ramp_region)]

    simple_lm_stack_mouse(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None)


def location_slope(data, collumn, save_path, ramp_region="outbound", trial_type="beaconed", print_p=False):
    data = data[(data[collumn] != np.nan)]

    # were only interested in the postive and negative sloping neurons when looking at the proportions of lmer neurons
    if (collumn == "lmer_result_ob"):
        data = data[(data["lm_result_ob"] == "Positive") | (data["lm_result_ob"] == "Negative")]
    elif (collumn == "lmer_result_hb"):
        data = data[(data["lm_result_hb"] == "Positive") | (data["lm_result_hb"] == "Negative")]

    # only look at beacoend and outbound
    data = data[(data.trial_type == trial_type) &
                (data.ramp_region == ramp_region)]

    simple_lm_stack(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None)

def cue_theta_location_hist(data, save_path):
    data = data[(data.trial_type == "all") & (data.ramp_region == "outbound")]
    PS_cue_d = data[(data.tetrode_location == "PS") & (data.ramp_driver == "Cue")]
    PS_cue_i = data[(data.tetrode_location == "PS") & (data.ramp_driver == "PI")]
    MEC_d = data[(data.tetrode_location == "MEC") & (data.ramp_driver == "Cue")]
    MEC_i = data[(data.tetrode_location == "MEC") & (data.ramp_driver == "PI")]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(np.asarray(PS_cue_d["ThetaIndex"]), bins=1000, alpha=0.5, color="y", label="MEC", histtype="step", density=True, cumulative=True, linewidth=2)
    ax.hist(np.asarray(PS_cue_i["ThetaIndex"]), bins=1000, alpha=0.5, color="g", label="PS", histtype="step", density=True, cumulative=True, linewidth=2)
    ax.set_ylabel("Cumulative Density", fontsize=15)
    ax.set_xlabel("Theta Index", fontsize=15)
    ax.set_xlim(left=-0.1, right=0.4)
    #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    #plt.subplots_adjust(left=0.2, right=0.6, top=0.8, bottom=0.2)
    #ax.legend(loc="upper right")
    plt.savefig(save_path+"/cue_theta_location_hist_PS.png", dpi=300)
    plt.show()
    plt.close()
    print("PS p= ", stats.ks_2samp(np.asarray(PS_cue_d["ThetaIndex"]), np.asarray(PS_cue_i["ThetaIndex"]))[1])

    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(np.asarray(MEC_d["ThetaIndex"]), bins=1000, alpha=0.5, color="y", label="MEC", histtype="step", density=True, cumulative=True, linewidth=2)
    ax.hist(np.asarray(MEC_i["ThetaIndex"]), bins=1000, alpha=0.5, color="g", label="PS", histtype="step", density=True, cumulative=True, linewidth=2)
    ax.set_ylabel("Cumulative Density", fontsize=15)
    ax.set_xlabel("Theta Index", fontsize=15)
    ax.set_xlim(left=-0.1, right=0.4)
    #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    #plt.subplots_adjust(left=0.2, right=0.6, top=0.8, bottom=0.2)
    #ax.legend(loc="upper right")
    plt.savefig(save_path+"/cue_theta_location_hist_MEC.png", dpi=300)
    plt.show()
    plt.close()
    print("MEC, p= ", stats.ks_2samp(np.asarray(MEC_d["ThetaIndex"]), np.asarray(MEC_i["ThetaIndex"]))[1])
    return


def cue_theta_location_bar(data, save_path):
    fig, ax = plt.subplots(figsize=(6,6))

    PS_cue_d = data[(data.tetrode_location == "PS") & (data.ramp_driver == "Cue")]
    PS_cue_i = data[(data.tetrode_location == "PS") & (data.ramp_driver == "PI")]
    MEC_d = data[(data.tetrode_location == "MEC") & (data.ramp_driver == "Cue")]
    MEC_i = data[(data.tetrode_location == "MEC") & (data.ramp_driver == "PI")]

    objects = ("PS|CD", "PS|CI", "MEC|CD", "MEC|CI")
    x_pos = [0, 2, 4, 6]

    ax.bar(x_pos, [np.mean(np.asarray(PS_cue_d["ThetaIndex"])),
                   np.mean(np.asarray(PS_cue_i["ThetaIndex"])),
                   np.mean(np.asarray(MEC_d["ThetaIndex"])),
                   np.mean(np.asarray(MEC_i["ThetaIndex"]))],
           yerr=  [stats.sem(np.asarray(PS_cue_d["ThetaIndex"])),
                   stats.sem(np.asarray(PS_cue_i["ThetaIndex"])),
                   stats.sem(np.asarray(MEC_d["ThetaIndex"])),
                   stats.sem(np.asarray(MEC_i["ThetaIndex"]))],

           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10,
           color =['b', 'r', 'b', 'r'])

    print("p = ", stats.ttest_ind(np.asarray(MEC_d["ThetaIndex"]),
                                  np.asarray(MEC_i["ThetaIndex"]))[1])

    plt.xticks(x_pos, objects, fontsize=15)
    plt.ylabel("Theta Index",  fontsize=20)
    plt.xlim((-1,7))
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"/theta_cue_location.png", dpi=300)
    plt.show()
    plt.close()

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

def get_mouse(session_id):
    return session_id.split("_")[0]

def get_cohort_mouse(full_session_id):
    session_id = full_session_id.split("/")[-1]
    mouse = get_mouse(session_id)
    cohort_tmp = get_cohort(full_session_id)
    cohort = get_tidy_title(cohort_tmp)
    return cohort+"_"+mouse

def add_cohort_mouse_label(data):
    cohort_mouse = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        cohort_mouse_str = get_cohort_mouse(row["full_session_id"].iloc[0])
        cohort_mouse.append(cohort_mouse_str)
    data["cohort_mouse"] = cohort_mouse
    return data

def add_recording_day(data):
    recording_days = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        recording_days.append(get_suedo_day(row["session_id"].iloc[0]))
    data["recording_day"] = recording_days
    return data

def get_recording_paths(path_list, folder_path):
    list_of_recordings = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for recording_path in list_of_recordings:
        path_list.append(recording_path)
    return path_list

def add_full_session_id(data, all_recording_paths):
    full_session_ids = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        session_id = row["session_id"].iloc[0]
        full_session_id = [s for s in all_recording_paths if session_id in s]
        full_session_ids.append(full_session_id[0])
    data["full_session_id"] = full_session_ids
    return data

def ramp_histogram_by_mouse(data, save_path):

    for cohort_mouse in np.unique(data["cohort_mouse"]):
        for ramp_region in ["outbound", "homebound", "all"]:
            for trial_type in ["all"]:

                cohort_mouse_data = data[(data["cohort_mouse"] == cohort_mouse) &
                                         (data["ramp_region"] == ramp_region) &
                                         (data["trial_type"] == trial_type)]

                fig, ax = plt.subplots(figsize=(6,6))
                ax.set_title("rr="+ramp_region+",tt="+trial_type+",m="+cohort_mouse)
                ax.hist(np.asarray(cohort_mouse_data["ramp_score"]), bins=50, alpha=0.7, color="r", density=False, cumulative=False, linewidth=4)
                ax.set_ylabel("Count", fontsize=15)
                ax.set_xlabel("Ramp Score", fontsize=15)
                ax.set_xlim(left=-0.75, right=0.75)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.tick_params(axis='both', which='major', labelsize=20)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.tight_layout()

                if ramp_region=="outbound":
                    collumn = "lm_result_ob"
                elif ramp_region=="homebound":
                    collumn = "lm_result_hb"


                if ramp_region == "outbound" or ramp_region=="homebound":
                    cohort_mouse_data = cohort_mouse_data[(cohort_mouse_data[collumn] == "None") |
                                                          (cohort_mouse_data[collumn] == "NoSlope") |
                                                          (cohort_mouse_data[collumn] == "Negative") |
                                                          (cohort_mouse_data[collumn] == "Positive")]
                    n_total= len(cohort_mouse_data)
                    ax.text(0.75, 0.95, "None, n="+    str(len(cohort_mouse_data[cohort_mouse_data[collumn] == "None"]))+", "    +str(np.round((len(cohort_mouse_data[cohort_mouse_data[collumn] == "None"])/n_total)*100, decimals=0))+"%", transform=ax.transAxes, fontsize=8, verticalalignment='top')
                    ax.text(0.75, 0.90, "No Slope, n="+str(len(cohort_mouse_data[cohort_mouse_data[collumn] == "NoSlope"]))+", " +str(np.round((len(cohort_mouse_data[cohort_mouse_data[collumn]== "NoSlope"])/n_total)*100, decimals=0))+"%", transform=ax.transAxes, fontsize=8, verticalalignment='top')
                    ax.text(0.75, 0.85, "Negative, n="+str(len(cohort_mouse_data[cohort_mouse_data[collumn] == "Negative"]))+", "+str(np.round((len(cohort_mouse_data[cohort_mouse_data[collumn] == "Negative"])/n_total)*100, decimals=0))+"%", transform=ax.transAxes, fontsize=8, verticalalignment='top')
                    ax.text(0.75, 0.80, "Positive, n="+str(len(cohort_mouse_data[cohort_mouse_data[collumn] == "Positive"]))+", "+str(np.round((len(cohort_mouse_data[cohort_mouse_data[collumn] == "Positive"])/n_total)*100, decimals=0))+"%", transform=ax.transAxes, fontsize=8, verticalalignment='top')

                if save_path is not None:
                    plt.savefig(save_path+"/bymouse_ramp_histo_m_"+cohort_mouse+"_tt_"+trial_type+"_rr_"+ramp_region+".png", dpi=300)

                plt.show()
                plt.close()

    return


def percentage_boccara(data,save_path, split="None", suffix=""):

    fig, ax = plt.subplots(figsize=(12,6))
    data = data[(data["cohort_mouse"] != "C3_M3")]

    cohort_mice = np.unique(data["cohort_mouse"])
    x_pos = np.arange(len(cohort_mice))

    if "ramp_region" in list(data):
        data = data[(data["ramp_region"] == "all") &
                    (data["trial_type"] == "all")]

    colors = ["C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9"]

    for counter, cohort_mouse in enumerate(cohort_mice):
        cohort_mouse_data = data[(data["cohort_mouse"] == cohort_mouse)]
        boccara_t = len(cohort_mouse_data[(cohort_mouse_data["Boccara_theta_class"] == 1)])
        n_total = len(cohort_mouse_data)
        label_str= str(boccara_t)+"/"+str(n_total)

        if split == "two":
            early_data, late_data = split_data_by_recording_day(data)

            n_early_t = len(early_data[(early_data["Boccara_theta_class"] == 1)])
            n_late_t = len(late_data[(late_data["Boccara_theta_class"] == 1)])
            n_early =len(early_data)
            n_late = len(late_data)

            if n_early==0:
                n_early_t=0
                n_early=1
            if n_late == 0:
                n_late_t=0
                n_late=1

            ax.bar([x_pos[counter]-0.2, x_pos[counter]+0.2], [n_early_t/n_early, n_late_t/n_late], align='center', alpha=0.5, width=0.4, color=colors[counter])
            ax.text(x=x_pos[counter], y =max([n_early_t/n_early, n_late_t/n_late]), s=label_str,  horizontalalignment='center', fontsize=12)

        elif split == "daily":
            n_days = len(np.unique(cohort_mouse_data["recording_day"]))
            max_y=0

            for day_counter, day in enumerate(np.unique(cohort_mouse_data["recording_day"])):
                cohort_mouse_data_day = cohort_mouse_data[(cohort_mouse_data["recording_day"] == day)]
                n_t = len(cohort_mouse_data_day[(cohort_mouse_data_day["Boccara_theta_class"] == 1)])
                n_ =  len(cohort_mouse_data_day)

                if n_==0:
                    n_ = 1
                    n_t = 0

                if n_t/n_ > max_y:
                    max_y = n_t/n_

                ax.bar(x_pos[counter]- 0.4+(day_counter*0.8/n_days), n_t/n_,
                       align='center', alpha=0.5, width=0.8/n_days, color=colors[counter])
            ax.text(x=x_pos[counter], y =max_y, s=label_str,  horizontalalignment='center', fontsize=12)

        elif split == "None":
            ax.bar(x_pos[counter], boccara_t/n_total, align='center', alpha=0.5, color=colors[counter])
            ax.text(x=x_pos[counter], y =boccara_t/n_total, s=label_str,  horizontalalignment='center', fontsize=12)


    plt.xticks(x_pos, cohort_mice, fontsize=5, rotation=-90)
    plt.ylim([0,1])
    plt.ylabel("Prop Theta Modulation",  fontsize=20)
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if split == "two":
        plt.savefig(save_path+"/Boccara_theta_cohort_mouse_earlylate"+suffix+".png", dpi=300)
    elif split == "None":
        plt.savefig(save_path+"/Boccara_theta_cohort_mous"+suffix+".png", dpi=300)
    elif split == "daily":
        plt.savefig(save_path+"/Boccara_theta_cohort_mouse_daily"+suffix+".png", dpi=300)

    plt.show()
    plt.close()

def percentage_gc(data,save_path, split="None", suffix="", threshold=0.4):

    data = data[(data["cohort_mouse"] != "C3_M3")]

    hd_thres = threshold
    fig, ax = plt.subplots(figsize=(12,6))

    cohort_mice = np.unique(data["cohort_mouse"])
    x_pos = np.arange(len(cohort_mice))

    colors = ["C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9"]

    for counter, cohort_mouse in enumerate(cohort_mice):
        cohort_mouse_data = data[(data["cohort_mouse"] == cohort_mouse)]

        grid_cells = cohort_mouse_data[(cohort_mouse_data["grid_score"]> hd_thres)]
        for cluster_index, cluster_id in enumerate(grid_cells.cluster_id):
            cluster_df = grid_cells[(grid_cells.cluster_id == cluster_id)] # dataframe for that cluster
            print(str(cluster_df["full_session_id"].iloc[0])+", cluster_id= "+str(cluster_df["cluster_id"].iloc[0]))

        hd_score_t = len(cohort_mouse_data[(cohort_mouse_data["grid_score"]> hd_thres)])
        n_total = len(cohort_mouse_data)

        label_str = str(hd_score_t)+"/"+str(n_total)

        if split == "two":
            early_data, late_data = split_data_by_recording_day(data)

            n_early_t = len(early_data[(early_data["grid_score"]> hd_thres)])
            n_late_t = len(late_data[(late_data["grid_score"]> hd_thres)])
            n_early =len(early_data)
            n_late = len(late_data)

            if n_early==0:
                n_early_t=0
                n_early=1
            if n_late == 0:
                n_late_t=0
                n_late=1

            ax.bar([x_pos[counter]-0.2, x_pos[counter]+0.2], [n_early_t/n_early, n_late_t/n_late], align='center', alpha=0.5, width=0.4, color=colors[counter])
            ax.text(x=x_pos[counter], y =max([n_early_t/n_early, n_late_t/n_late]), s=label_str,  horizontalalignment='center', fontsize=12)

        elif split == "daily":
            n_days = len(np.unique(cohort_mouse_data["recording_day"]))
            max_y = 0

            for day_counter, day in enumerate(np.unique(cohort_mouse_data["recording_day"])):
                cohort_mouse_data_day = cohort_mouse_data[(cohort_mouse_data["recording_day"] == day)]
                n_t = len(cohort_mouse_data_day[(cohort_mouse_data_day["grid_score"]> hd_thres)])
                n_ =  len(cohort_mouse_data_day)
                if n_t/n_ > max_y:
                    max_y = n_t/n_

                ax.bar(x_pos[counter]- 0.4+(day_counter*0.8/n_days), n_t/n_,
                       align='center', alpha=0.5, width=0.8/n_days, color=colors[counter])

            ax.text(x=x_pos[counter], y =max_y, s=label_str,  horizontalalignment='center', fontsize=12)

        elif split == "None":
            ax.bar(x_pos[counter], hd_score_t/n_total, align='center', alpha=0.5, color=colors[counter])
            ax.text(x=x_pos[counter], y =hd_score_t/n_total, s=label_str,  horizontalalignment='center', fontsize=12)

    plt.xticks(x_pos, cohort_mice, fontsize=5, rotation=-90)
    plt.ylim([0,1])
    plt.ylabel("Prop Grid Cells",  fontsize=20)
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if split == "two":
        plt.savefig(save_path+"/grid_score_cohort_mouse_earlylate"+suffix+".png", dpi=300)
    elif split == "None":
        plt.savefig(save_path+"/grid_score_cohort_mous"+suffix+".png", dpi=300)
    elif split == "daily":
        plt.savefig(save_path+"/grid_score_cohort_mouse_daily"+suffix+".png", dpi=300)

    plt.show()
    plt.close()

def percentage_hd(data,save_path, split="None", suffix="", threshold=0.4):

    data = data[(data["cohort_mouse"] != "C3_M3")]

    hd_thres = threshold
    fig, ax = plt.subplots(figsize=(12,6))

    cohort_mice = np.unique(data["cohort_mouse"])
    x_pos = np.arange(len(cohort_mice))

    colors = ["C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9"]

    for counter, cohort_mouse in enumerate(cohort_mice):
        cohort_mouse_data = data[(data["cohort_mouse"] == cohort_mouse)]
        hd_score_t = len(cohort_mouse_data[(cohort_mouse_data["hd_score"]> hd_thres)])
        n_total = len(cohort_mouse_data)

        label_str = str(hd_score_t)+"/"+str(n_total)

        if split == "two":
            early_data, late_data = split_data_by_recording_day(data)

            n_early_t = len(early_data[(early_data["hd_score"]> hd_thres)])
            n_late_t = len(late_data[(late_data["hd_score"]> hd_thres)])
            n_early =len(early_data)
            n_late = len(late_data)

            if n_early==0:
                n_early_t=0
                n_early=1
            if n_late == 0:
                n_late_t=0
                n_late=1

            ax.bar([x_pos[counter]-0.2, x_pos[counter]+0.2], [n_early_t/n_early, n_late_t/n_late], align='center', alpha=0.5, width=0.4, color=colors[counter])
            ax.text(x=x_pos[counter], y =max([n_early_t/n_early, n_late_t/n_late]), s=label_str,  horizontalalignment='center', fontsize=12)

        elif split == "daily":
            n_days = len(np.unique(cohort_mouse_data["recording_day"]))
            max_y = 0

            for day_counter, day in enumerate(np.unique(cohort_mouse_data["recording_day"])):
                cohort_mouse_data_day = cohort_mouse_data[(cohort_mouse_data["recording_day"] == day)]
                n_t = len(cohort_mouse_data_day[(cohort_mouse_data_day["hd_score"]> hd_thres)])
                n_ =  len(cohort_mouse_data_day)
                if n_t/n_ > max_y:
                    max_y = n_t/n_

                ax.bar(x_pos[counter]- 0.4+(day_counter*0.8/n_days), n_t/n_,
                       align='center', alpha=0.5, width=0.8/n_days, color=colors[counter])

            ax.text(x=x_pos[counter], y =max_y, s=label_str,  horizontalalignment='center', fontsize=12)

        elif split == "None":
            ax.bar(x_pos[counter], hd_score_t/n_total, align='center', alpha=0.5, color=colors[counter])
            ax.text(x=x_pos[counter], y =hd_score_t/n_total, s=label_str,  horizontalalignment='center', fontsize=12)

    plt.xticks(x_pos, cohort_mice, fontsize=5, rotation=-90)
    plt.ylim([0,1])
    plt.ylabel("Prop Head Direction Cells",  fontsize=20)
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if split == "two":
        plt.savefig(save_path+"/hd_score_cohort_mouse_earlylate"+suffix+".png", dpi=300)
    elif split == "None":
        plt.savefig(save_path+"/hd_score_cohort_mous"+suffix+".png", dpi=300)
    elif split == "daily":
        plt.savefig(save_path+"/hd_score_cohort_mouse_daily"+suffix+".png", dpi=300)

    plt.show()
    plt.close()

def percentage_encoding(data, lmer_result, ramp_region, save_path, split="None", suffix=""):
    suffix=ramp_region

    fig, ax = plt.subplots(figsize=(12,6))

    cohort_mice = np.unique(data["cohort_mouse"])
    x_pos = np.arange(len(cohort_mice))
    data = data[(data["ramp_region"] == ramp_region) & (data["trial_type"] == "all")]

    if ramp_region=="outbound":
        collumn ="lmer_result_ob"
    elif ramp_region=="homebound":
        collumn ="lmer_result_hb"

    colors = ["C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9"]

    for counter, cohort_mouse in enumerate(cohort_mice):
        cohort_mouse_data = data[(data["cohort_mouse"] == cohort_mouse)]

        n_encoding = len(cohort_mouse_data[(cohort_mouse_data[collumn] == lmer_result)])
        n_total = len(cohort_mouse_data)
        label_str = str(n_encoding)+"/"+str(n_total)

        if split == "two":
            early_data, late_data = split_data_by_recording_day(cohort_mouse_data)

            n_early_encoding =  len(early_data[(early_data[collumn] == lmer_result)])
            n_late_encoding =  len(late_data[(late_data[collumn] == lmer_result)])
            n_early =len(early_data)
            n_late = len(late_data)

            ax.bar([x_pos[counter]-0.2, x_pos[counter]+0.2], [n_early_encoding/n_early, n_late_encoding/n_late], align='center', alpha=0.5, width=0.4, color=colors[counter])
            ax.text(x=x_pos[counter], y =max([n_early_encoding/n_early, n_late_encoding/n_late]), s=label_str,  horizontalalignment='center', fontsize=12)

        elif split == "None":
            ax.bar(x_pos[counter], n_encoding/n_total, align='center', alpha=0.5, color=colors[counter])
            ax.text(x=x_pos[counter], y =n_encoding/n_total, s=label_str,  horizontalalignment='center', fontsize=12)

    plt.xticks(x_pos, cohort_mice, fontsize=15, rotation=-45)
    plt.ylim([0,1])
    plt.ylabel("Prop "+lmer_result+" Cells",  fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if split == "two":
        plt.savefig(save_path+"/prop_"+lmer_result+"_cohort_mouse_earlylate"+suffix+".png", dpi=300)
    elif split == "None":
        plt.savefig(save_path+"/prop_"+lmer_result+"_cohort_mous"+suffix+".png", dpi=300)

    plt.show()
    plt.close()

def percentage_ramp(data, ramp_region, save_path, split="None", suffix=""):
    suffix=ramp_region

    fig, ax = plt.subplots(figsize=(12,6))

    cohort_mice = np.unique(data["cohort_mouse"])
    x_pos = np.arange(len(cohort_mice))

    data = data[(data["ramp_region"] == ramp_region) & (data["trial_type"] == "all")]
    if ramp_region=="outbound":
        collumn ="lm_result_ob"
    elif ramp_region=="homebound":
        collumn ="lm_result_hb"

    colors = ["C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9"]

    for counter, cohort_mouse in enumerate(cohort_mice):
        cohort_mouse_data = data[(data["cohort_mouse"] == cohort_mouse)]
        #n_ramps = len(cohort_mouse_data[(cohort_mouse_data[collumn] == "Negative") |
        #                                (cohort_mouse_data[collumn] == "Positive") ])
        #n_total = len(cohort_mouse_data)

        #split them into early and late sessions
        #max_recording_day = max(cohort_mouse_data.recording_day)
        #min_recording_day = min(cohort_mouse_data.recording_day)

        if split == "two":
            #split them into early and late sessions
            max_recording_day = max(cohort_mouse_data.recording_day)
            min_recording_day = min(cohort_mouse_data.recording_day)

            early_data = cohort_mouse_data[(cohort_mouse_data["recording_day"] < (max_recording_day+min_recording_day)/2)]
            late_data =  cohort_mouse_data[(cohort_mouse_data["recording_day"] >= (max_recording_day+min_recording_day)/2)]

            n_early_t =  len(early_data[(early_data[collumn] == "Negative") |
                                        (early_data[collumn] == "Positive") ])
            n_late_t =  len(late_data[(late_data[collumn] == "Negative") |
                                      (late_data[collumn] == "Positive") ])
            n_early =len(early_data)
            n_late = len(late_data)

            ax.bar([x_pos[counter]-0.2, x_pos[counter]+0.2], [n_early_t/n_early, n_late_t/n_late], align='center', alpha=0.5, width=0.4, color=colors[counter])

        elif split == "daily":
            n_days = len(np.unique(cohort_mouse_data["recording_day"]))

            for day_counter, day in enumerate(np.unique(cohort_mouse_data["recording_day"])):
                cohort_mouse_data_day = cohort_mouse_data[(cohort_mouse_data["recording_day"] == day)]
                n_t = len(cohort_mouse_data_day[(cohort_mouse_data_day[collumn] == "Negative") |
                                                (cohort_mouse_data_day[collumn] == "Positive") ])
                n_ =  len(cohort_mouse_data_day)

                ax.bar(x_pos[counter]- 0.4+(day_counter*0.8/n_days), n_t/n_,
                       align='center', alpha=0.5, width=0.8/n_days, color=colors[counter])

        elif split == "None":
            n_ramps = len(cohort_mouse_data[(cohort_mouse_data[collumn] == "Negative") |
                                            (cohort_mouse_data[collumn] == "Positive") ])
            n_total = len(cohort_mouse_data)

            ax.bar(x_pos[counter], n_ramps/n_total, align='center', alpha=0.5, color=colors[counter])

    plt.xticks(x_pos, cohort_mice, fontsize=15, rotation=-45)
    plt.ylim([0,1])
    plt.ylabel("Prop Ramp Cells",  fontsize=20)
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if split == "two":
        plt.savefig(save_path+"/prop_ramp_cohort_mouse_earlylate"+suffix+".png", dpi=300)
    elif split == "None":
        plt.savefig(save_path+"/prop_ramp_cohort_mous"+suffix+".png", dpi=300)
    elif split == "daily":
        plt.savefig(save_path+"/prop_ramp_cohort_mouse_daily"+suffix+".png", dpi=300)

    plt.show()
    plt.close()

def percentage_ramp_rs(data, ramp_region, save_path, split="None", suffix=""):
    suffix=ramp_region

    fig, ax = plt.subplots(figsize=(12,6))

    cohort_mice = np.unique(data["cohort_mouse"])
    x_pos = np.arange(len(cohort_mice))

    data = data[(data["ramp_region"] == ramp_region) & (data["trial_type"] == "all")]

    colors = ["C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9",
              "C0","C1","C2","C3", "C4","C5","C6", "C7", "C8", "C9"]

    for counter, cohort_mouse in enumerate(cohort_mice):
        cohort_mouse_data = data[(data["cohort_mouse"] == cohort_mouse)]

        if split == "two":
            #split them into early and late sessions
            max_recording_day = max(cohort_mouse_data.recording_day)
            min_recording_day = min(cohort_mouse_data.recording_day)

            early_data = cohort_mouse_data[(cohort_mouse_data["recording_day"] < (max_recording_day+min_recording_day)/2)]
            late_data =  cohort_mouse_data[(cohort_mouse_data["recording_day"] >= (max_recording_day+min_recording_day)/2)]

            n_early_t =  len(early_data[(early_data["abs_ramp_score"] > 0.5)])
            n_late_t =   len(late_data[(late_data["abs_ramp_score"] > 0.5)])
            n_early =len(early_data)
            n_late = len(late_data)

            ax.bar([x_pos[counter]-0.2, x_pos[counter]+0.2], [n_early_t/n_early, n_late_t/n_late], align='center', alpha=0.5, width=0.4, color=colors[counter])

        elif split == "daily":
            n_days = len(np.unique(cohort_mouse_data["recording_day"]))

            for day_counter, day in enumerate(np.unique(cohort_mouse_data["recording_day"])):
                cohort_mouse_data_day = cohort_mouse_data[(cohort_mouse_data["recording_day"] == day)]
                n_t = len(cohort_mouse_data_day[(cohort_mouse_data_day["abs_ramp_score"] > 0.5)])
                n_ =  len(cohort_mouse_data_day)

                ax.bar(x_pos[counter]- 0.4+(day_counter*0.8/n_days), n_t/n_,
                       align='center', alpha=0.5, width=0.8/n_days, color=colors[counter])

        elif split == "None":
            n_ramps = len(cohort_mouse_data[(cohort_mouse_data["abs_ramp_score"] > 0.5)])
            n_total = len(cohort_mouse_data)

            ax.bar(x_pos[counter], n_ramps/n_total, align='center', alpha=0.5, color=colors[counter])

    plt.xticks(x_pos, cohort_mice, fontsize=15, rotation=-45)
    plt.ylim([0,1])
    plt.ylabel("Prop Ramp Cells",  fontsize=20)
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if split == "two":
        plt.savefig(save_path+"/prop_ramp_rs_cohort_mouse_earlylate"+suffix+".png", dpi=300)
    elif split == "None":
        plt.savefig(save_path+"/prop_ramp_rs_cohort_mous"+suffix+".png", dpi=300)
    elif split == "daily":
        plt.savefig(save_path+"/prop_ramp_rs_cohort_mouse_daily"+suffix+".png", dpi=300)

    plt.show()
    plt.close()

def hd_per_mouse(new, save_path, threshold=0.4):
    percentage_hd(new, save_path, split="daily", threshold=threshold)
    percentage_hd(new, save_path, split="two", threshold=threshold)
    percentage_hd(new, save_path, split="None", threshold=threshold)

def gc_per_mouse(new, save_path, threshold=0.4):
    percentage_gc(new, save_path, split="daily", threshold=threshold)
    percentage_gc(new, save_path, split="two", threshold=threshold)
    percentage_gc(new, save_path, split="None", threshold=threshold)

def add_ramp_scores_to_matched_cluster(mice_df, ramp_scores_df):
    new=pd.DataFrame()

    for index_j, row_ramp_score in ramp_scores_df.iterrows():
        row_ramp_score =  row_ramp_score.to_frame().T.reset_index(drop=True)
        session_id = row_ramp_score["session_id"].iloc[0]
        cluster_id = row_ramp_score["cluster_id"].iloc[0]

        paired_cluster = mice_df[((mice_df["session_id"] == session_id) & (mice_df["sorted_seperately_vr_cluster_ids"] == cluster_id))]

        if len(paired_cluster)==1:
            paired_cluster["trial_type"] = [row_ramp_score['trial_type'].iloc[0]]
            paired_cluster['ramp_score'] = [row_ramp_score['ramp_score'].iloc[0]]
            paired_cluster['ramp_region'] = [row_ramp_score['ramp_region'].iloc[0]]
            paired_cluster['fr_range'] = [row_ramp_score['fr_range'].iloc[0]]
            paired_cluster['breakpoint'] = [row_ramp_score['breakpoint'].iloc[0]]
            paired_cluster['fr_smooth'] = [row_ramp_score['fr_smooth'].iloc[0]]
            paired_cluster['pos_bin'] = [row_ramp_score['pos_bin'].iloc[0]]
            paired_cluster['ramp_span'] = [row_ramp_score['ramp_span'].iloc[0]]
            paired_cluster['is_shuffled'] = [row_ramp_score['is_shuffled'].iloc[0]]

            if row_ramp_score['is_shuffled'].iloc[0] == 0:
                new = pd.concat([new, paired_cluster], ignore_index=True)

    #new = correct_datatypes(new, ignore_of=ignore_of)
    #new = analyse_ramp_driver(new)
    #new = get_best_ramp_score(new)
    return new

def correlate_vr_vs_of(mice_df, save_path, ramp_region, trial_type, collumn=""):

    dffdf_tmp = mice_df[(mice_df["trial_type"] == "all") &
                        ((mice_df["ramp_region"] == "outbound") |
                         (mice_df["ramp_region"] == "homebound"))]

    dffdf_tmp = dffdf_tmp[(dffdf_tmp["n_spikes_of"] > 50)]

    fig, ax = plt.subplots()
    ax.set_title("tt="+get_tidy_title(trial_type)+", rr="+ramp_region, fontsize=15)

    ramp_region_tt_mice = mice_df[(mice_df["ramp_region"] == ramp_region) & (mice_df["trial_type"] == trial_type)]
    if ramp_region == "outbound":
        lm_collumn = "lm_result_ob"
        lmer_collumn = "lmer_result_ob"
    elif ramp_region == "homebound":
        lm_collumn = "lm_result_hb"
        lmer_collumn = "lmer_result_hb"

    else:
        ax.scatter(ramp_region_tt_mice["ramp_score"], ramp_region_tt_mice[collumn], edgecolor="black", marker="o", facecolors='none')

    if ramp_region=="outbound" or ramp_region=="homebound":
        #None_group = ramp_region_tt_mice[(ramp_region_tt_mice[lm_collumn] == "NoSlope")]
        #Unclass = ramp_region_tt_mice[(ramp_region_tt_mice[lm_collumn] == "None")]
        #Neg_group = ramp_region_tt_mice[(ramp_region_tt_mice[lm_collumn] == "Negative")]
        #Pos_group = ramp_region_tt_mice[(ramp_region_tt_mice[lm_collumn] == "Positive")]
        #ax.scatter(None_group["ramp_score"], None_group[collumn], edgecolor="black", marker="o", facecolors='none')
        #ax.scatter(Unclass["ramp_score"], Unclass[collumn], edgecolor="grey", marker="o", facecolors='none')
        #ax.scatter(Neg_group["ramp_score"], Neg_group[collumn], edgecolor="blue", marker="o", facecolors='none')
        #ax.scatter(Pos_group["ramp_score"], Pos_group[collumn], edgecolor="red", marker="o", facecolors='none')

        for lmer_result in np.unique(ramp_region_tt_mice[lmer_collumn]):
            lmer_result_stats = ramp_region_tt_mice[(ramp_region_tt_mice[lmer_collumn] == lmer_result)]
            c = lmer_result_color(lmer_result)
            ax.scatter(lmer_result_stats["ramp_score"], lmer_result_stats[collumn],  facecolor=c, edgecolor=c, marker="o", facecolors='none')

        lmer_result_P = ramp_region_tt_mice[((ramp_region_tt_mice[lmer_collumn] == "P") |
                                             (ramp_region_tt_mice[lmer_collumn] == "PS") |
                                             (ramp_region_tt_mice[lmer_collumn] == "PA") |
                                             (ramp_region_tt_mice[lmer_collumn] == "PSA"))]
        #lmer_result_P = ramp_region_tt_mice[(ramp_region_tt_mice[lmer_collumn] == "P")]

        lmer_result_A = ramp_region_tt_mice[((ramp_region_tt_mice[lmer_collumn] == "A") |
                                             (ramp_region_tt_mice[lmer_collumn] == "SA") |
                                             (ramp_region_tt_mice[lmer_collumn] == "PA") |
                                             (ramp_region_tt_mice[lmer_collumn] == "PSA"))]
        #lmer_result_A = ramp_region_tt_mice[(ramp_region_tt_mice[lmer_collumn] == "A")]

        lmer_result_S = ramp_region_tt_mice[((ramp_region_tt_mice[lmer_collumn] == "S") |
                                             (ramp_region_tt_mice[lmer_collumn] == "PS") |
                                             (ramp_region_tt_mice[lmer_collumn] == "SA") |
                                             (ramp_region_tt_mice[lmer_collumn] == "PSA"))]
        #lmer_result_S = ramp_region_tt_mice[(ramp_region_tt_mice[lmer_collumn] == "S")]

        ##ax.scatter(np.mean(lmer_result_P["ramp_score"]), np.mean(lmer_result_P[collumn]), facecolor="r", marker="x")
        #ax.scatter(np.mean(lmer_result_A["ramp_score"]), np.mean(lmer_result_A[collumn]), facecolor="b", marker="x")
        #ax.scatter(np.mean(lmer_result_S["ramp_score"]), np.mean(lmer_result_S[collumn]), facecolor="y", marker="x")

    plt.xlabel("Ramp Score", fontsize=20, labelpad=10)
    plt.xlim([-1,1])
    plt.ylabel(get_tidy_title(collumn), fontsize=20, labelpad=10)
    plt.tick_params(labelsize=20)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
    plt.savefig(save_path+"/ramp_score_vs_"+collumn+"_"+ramp_region+"_tt_"+trial_type+".png", dpi=300)
    print("plotted depth correlation")

def lfp_vs_trial_number(all_mice_lfp_data, save_path):

    for cohort_mouse in np.unique(all_mice_lfp_data.cohort_mouse):
        fig, ax = plt.subplots()
        cohort_mouse_lfp_data = all_mice_lfp_data[(all_mice_lfp_data["cohort_mouse"] == cohort_mouse)]

        tetrode1 = cohort_mouse_lfp_data[(cohort_mouse_lfp_data["tetrode"] == 1)]
        tetrode2 = cohort_mouse_lfp_data[(cohort_mouse_lfp_data["tetrode"] == 2)]
        tetrode3 = cohort_mouse_lfp_data[(cohort_mouse_lfp_data["tetrode"] == 3)]
        tetrode4 = cohort_mouse_lfp_data[(cohort_mouse_lfp_data["tetrode"] == 4)]

        ax.scatter(tetrode1["trial_numbers"], tetrode1["theta_lfp_pwr"], marker="o", label=cohort_mouse, color="k")
        ax.scatter(tetrode2["trial_numbers"], tetrode2["theta_lfp_pwr"], marker="o", label=cohort_mouse, color="r")
        ax.scatter(tetrode3["trial_numbers"], tetrode3["theta_lfp_pwr"], marker="o", label=cohort_mouse, color="b")
        ax.scatter(tetrode4["trial_numbers"], tetrode4["theta_lfp_pwr"], marker="o", label=cohort_mouse, color="g")

        #ax.scatter(cohort_mouse_lfp_data["trial_numbers"], cohort_mouse_lfp_data["theta_lfp_pwr"], marker="o", label=cohort_mouse, color="k")
        #ax.set_title("tt=, rr=", fontsize=15)
        plt.xlabel("Total trial per Session", fontsize=20, labelpad=10)
        plt.ylabel("Theta LFP Power", fontsize=20, labelpad=10)
        plt.ylim(0, 2500)
        plt.tick_params(labelsize=20)
        #plt.legend()
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig(save_path+"/lfp_theta_vs_trials_"+cohort_mouse+".png", dpi=300)
        plt.close()
        print("plotted depth correlation")

def add_lfp_theta_power(all_mice_lfp_data):
    theta_lfp_power = []

    for i, row in all_mice_lfp_data.iterrows():
        row = row.to_frame().T.reset_index(drop=True)
        freqs = row["freqs"].iloc[0]
        pwrs = row["pwr_specs"].iloc[0]
        theta_pwr = np.mean(pwrs[(freqs>=6) & (freqs<=10)])
        theta_lfp_power.append(theta_pwr)

    all_mice_lfp_data["theta_lfp_pwr"] = theta_lfp_power
    return all_mice_lfp_data


def plot_all_ramps(all_mice, save_path, trial_types):

    for i, row in all_mice.iterrows():
        fig, ax = plt.subplots()
        row = row.to_frame().T.reset_index(drop=True)
        pos_bin = row["pos_bin"].iloc[0]
        fr_smooth = row["fr_smooth"].iloc[0]
        breakpoints = row["breakpoint"].iloc[0]
        trial_type = row["trial_type"].iloc[0]

        if trial_type in trial_types:
            ramp_region = row["ramp_region"].iloc[0]
            session_id = row["session_id"].iloc[0]
            cluster_id_st = row["sorted_together_vr_cluster_ids"].iloc[0]
            cluster_id_ss = row["sorted_seperately_vr_cluster_ids"].iloc[0]
            ramp_score = row["ramp_score"].iloc[0]
            ax.plot(pos_bin, fr_smooth, color="r")
            ax.scatter(breakpoints[0], fr_smooth[pos_bin==breakpoints[0]], marker="o", color="k")
            ax.scatter(breakpoints[1], fr_smooth[pos_bin==breakpoints[1]], marker="o", color="k")
            ax.plot(breakpoints, [max(fr_smooth), max(fr_smooth)], color="k")

            plt.title("rs: "+str(np.round(ramp_score, decimals=2)), fontsize=20)
            plt.xlabel("cm", fontsize=20, labelpad=10)
            plt.xlim([min(pos_bin),max(pos_bin)])
            plt.yticks([np.round(min(fr_smooth), decimals=1),np.round(max(fr_smooth), decimals=1)])
            plt.xticks([min(pos_bin), max(pos_bin)])
            plt.ylabel("Hz", fontsize=20, labelpad=10)
            plt.tick_params(labelsize=20)
            plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.32, right = 0.87, top = 0.92)
            plt.savefig(save_path+"/"+session_id+"ssID"+str(cluster_id_ss)+"_stID"+str(cluster_id_st)+
                        "_rr_"+ramp_region+"_tt_"+trial_type+".png", dpi=300)
        plt.close()

    print("done pltting")



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

def plot_theta_histogram(data, save_path):
    trial_type_theta_df = data[(data.trial_type == "beaconed")]
    trial_type_theta_df = trial_type_theta_df[(trial_type_theta_df.ramp_region == "outbound")]

    rythmic = trial_type_theta_df[(trial_type_theta_df["ThetaIndex"] > 0.07)]
    no_rythmic = trial_type_theta_df[(trial_type_theta_df["ThetaIndex"] < 0.07)]

    fig, ax = plt.subplots(figsize=(3,4))
    ax.hist(np.asarray(rythmic["ThetaIndex"]), bins=20, alpha=0.5, color="r")
    ax.hist(np.asarray(no_rythmic["ThetaIndex"]), bins=20, alpha=0.5, color="k")
    plt.xlabel("Theta Index",  fontsize=15)
    plt.ylabel("Number of Cells",  fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"/theta_histo.png", dpi=300)
    plt.show()

def plot_lm_proportions(theta_df_VR, ramp_region, save_path):

    theta_df_VR = theta_df_VR[(theta_df_VR["trial_type"] == "all")]
    theta_df_VR = theta_df_VR[(theta_df_VR["ramp_region"] == ramp_region)]
    if ramp_region == "outbound":
        collumn = "lm_result_ob"
    elif ramp_region == "homebound":
        collumn = "lm_result_hb"

    rythmic = theta_df_VR[(theta_df_VR.ThetaIndex > 0.07)]
    no_rythmic = theta_df_VR[(theta_df_VR.ThetaIndex < 0.07)]

    fig, ax = plt.subplots(figsize=(3,6))

    pos_rythimc = len(rythmic[rythmic[collumn] == "Positive"])*100/len(rythmic)
    neg_rythmic = len(rythmic[rythmic[collumn] == "Negative"])*100/len(rythmic)
    non_rythmic = len(rythmic[rythmic[collumn] == "None"])*100/len(rythmic)

    pos_norythimc = len(no_rythmic[no_rythmic[collumn] == "Positive"])*100/len(no_rythmic)
    neg_norythmic = len(no_rythmic[no_rythmic[collumn] == "Negative"])*100/len(no_rythmic)
    non_norythmic = len(no_rythmic[no_rythmic[collumn] == "None"])*100/len(no_rythmic)

    ax.bar(x=1, height=neg_rythmic, bottom=0, color=((238.0/255,58.0/255,140.0/255)))
    ax.bar(x=1, height=pos_rythimc, bottom=neg_rythmic, color=((102.0/255,205.0/255,0.0/255)))
    ax.bar(x=1, height=non_rythmic, bottom=pos_rythimc+ neg_rythmic, color="black")

    ax.bar(x=0, height=neg_norythmic, bottom=0, color=((238.0/255,58.0/255,140.0/255)))
    ax.bar(x=0, height=pos_norythimc, bottom=neg_norythmic, color=((102.0/255,205.0/255,0.0/255)))
    ax.bar(x=0, height=non_norythmic, bottom=pos_norythimc+ neg_norythmic, color="black")

    #ax.text(x=1 , y=0+0.05, s=str(len(rythmic[rythmic[collumn] == "Negative"])), color="white", fontsize=12, horizontalalignment='center')
    #ax.text(x=1 , y=neg_rythmic+0.05, s=str(len(rythmic[rythmic[collumn] == "None"])), color="white", fontsize=12, horizontalalignment='center')
    #ax.text(x=1 , y=non_rythmic+ neg_rythmic+0.05, s=str(len(rythmic[rythmic[collumn] == "Positive"])), color="white", fontsize=12, horizontalalignment='center')

    #ax.text(x=0 , y=0+0.05, s=str(len(no_rythmic[no_rythmic[collumn] == "Negative"])), color="white", fontsize=12, horizontalalignment='center')
    #ax.text(x=0 , y=neg_norythmic+0.05, s=str(len(no_rythmic[no_rythmic[collumn] == "None"])), color="white", fontsize=12, horizontalalignment='center')
    #ax.text(x=0 , y=non_norythmic+ neg_norythmic+0.05, s=str(len(no_rythmic[no_rythmic[collumn] == "Positive"])), color="white", fontsize=12, horizontalalignment='center')

    #ax.text(x=1 , y=103.05, s=str(np.round((len(rythmic)/(len(rythmic)+len(no_rythmic)))*100, decimals=0))+"%", color="black", fontsize=12, horizontalalignment='center')
    #ax.text(x=0 , y=103.05, s=str(np.round((len(no_rythmic)/(len(rythmic)+len(no_rythmic)))*100, decimals=0))+"%", color="black", fontsize=12, horizontalalignment='center')

    objects = ('NR', 'TR')
    x_pos = np.arange(len(objects))
    plt.xticks(x_pos, objects, fontsize=15)
    plt.ylabel("Percent of neurons",  fontsize=25)
    plt.xlim((-0.5, len(objects)-0.5))
    plt.ylim((0,100))
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"/"+collumn+"_theta_proportion.png", dpi=300)
    plt.show()

def plot_theta(theta_df, save_path):

    fig, ax = plt.subplots(figsize=(9,6))

    for mouse in np.unique(theta_df["cohort_mouse"]):
        mouse_df = theta_df[theta_df["cohort_mouse"] == mouse]
        mouse_df = mouse_df.dropna()
        mean = mouse_df.groupby('recording_day')["ThetaIndex"].mean().reset_index()
        sem = mouse_df.groupby('recording_day')["ThetaIndex"].sem().reset_index()
        max_by_day = mouse_df.groupby('recording_day')["ThetaIndex"].max().reset_index()
        mean = mean.fillna(0)
        sem = sem.fillna(0)
        max_by_day = max_by_day.fillna(0)

        gauss_kernel = Gaussian1DKernel(2) # 2 days
        smoothed_max_theta = convolve(max_by_day["ThetaIndex"], gauss_kernel)

        plt.plot(max_by_day["recording_day"], smoothed_max_theta, label=mouse)
        #plt.plot(max_by_day["recording_day"], max_by_day["ThetaIndex"], label=mouse)
        #plt.fill_between(mean["recording_day"], mean["ThetaIndex"]-sem["ThetaIndex"], mean["ThetaIndex"]+sem["ThetaIndex"], alpha=0.5)

    ax.hlines(y=0.07, xmin=0, xmax=max(theta_df["recording_day"]), linestyles="--") # threshold specified for rythmic cells Kornienko et al.
    ax.set_xlabel("Training day", fontsize=15)
    ax.set_ylabel("Maximum Theta Index", fontsize=15)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.tight_layout()
    plt.savefig(save_path+"/tracked_theta_index.png", dpi=300)
    plt.show()


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #all_mice_lfp_data = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_mice_lfp_data.pkl")
    #all_mice_lfp_data = add_lfp_theta_power(all_mice_lfp_data)
    #lfp_vs_trial_number(all_mice_lfp_data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/")

    ramp_scores_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ramp_score_coeff_export.csv"
    ramp_score_path2 = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ramp_score_coeff_export.pkl"
    tetrode_location_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/tetrode_locations.csv"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/vr_vs_of"
    theta_df_VR = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta/theta_df_VR.pkl")
    linear_model_path = pd.read_csv("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_results_linearmodel.txt", sep="\t")
    trialtypes_linear_model = pd.read_csv("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_results_linearmodel_trialtypes.txt", sep="\t")
    ramp_scores = pd.read_csv(ramp_scores_path)
    tetrode_locations = pd.read_csv(tetrode_location_path)
    ramp_scores_2 = pd.read_pickle(ramp_score_path2)


    '''
    #------------------------------------------------------------------------------------------------#
    # for looking a head directional proportion of cells in each mouse

    new = pd.DataFrame()
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C2.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C3.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C4.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C5.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C6.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C7.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C8.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C9.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C10.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C11.pkl")], ignore_index=True)
    new = pd.concat([new, pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/All_mice_of_C12.pkl")], ignore_index=True)
    new = add_cohort_mouse_label(new)
    new = add_recording_day(new)
    hd_per_mouse(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", threshold=0.4)
    gc_per_mouse(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", threshold=0.4)
    percentage_boccara(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", split="daily", suffix="")
    percentage_boccara(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", split="two", suffix="")
    percentage_boccara(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", split="None", suffix="")

    #=================================================================================================#
    '''

    C5_M1 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M1_sorting_stats_AT20_WS4.pkl")
    C5_M2 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M2_sorting_stats_AT20_WS4.pkl")
    C4_M2 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M2_sorting_stats_AT20_WS4.pkl")
    C4_M3 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M3_sorting_stats_AT20_WS4.pkl")
    C3_M1 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M1_sorting_stats_AT20_WS4.pkl")
    C3_M6 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M6_sorting_stats_AT20_WS4.pkl")
    C2_245 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/245_sorting_stats_AT20_WS4.pkl")
    C2_1124 = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/1124_sorting_stats_AT20_WS4.pkl")

    all_mice = pd.DataFrame()
    all_mice = pd.concat([all_mice, C5_M1], ignore_index=True)
    all_mice = pd.concat([all_mice, C5_M2], ignore_index=True)
    all_mice = pd.concat([all_mice, C4_M2], ignore_index=True)
    all_mice = pd.concat([all_mice, C4_M3], ignore_index=True)
    all_mice = pd.concat([all_mice, C3_M1], ignore_index=True)
    all_mice = pd.concat([all_mice, C3_M6], ignore_index=True)
    all_mice = pd.concat([all_mice, C2_245], ignore_index=True)
    #all_mice = pd.concat([all_mice, C2_1124], ignore_index=True)
    #all_mice = add_ramp_scores_to_matched_cluster(all_mice, ramp_scores)
    all_mice = add_ramp_scores_to_matched_cluster(all_mice, ramp_scores_2)
    all_mice = absolute_ramp_score(all_mice)
    all_mice = add_lm(all_mice, linear_model_path)

    '''
    plot_all_ramps(all_mice, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/all_ramps", trial_types=["all"])

    correlate_vr_vs_of(all_mice, save_path, ramp_region="outbound", trial_type="all", collumn="hd_score")
    correlate_vr_vs_of(all_mice, save_path, ramp_region="homebound", trial_type="all", collumn="hd_score")

    correlate_vr_vs_of(all_mice, save_path, ramp_region="outbound", trial_type="all", collumn="speed_score")
    correlate_vr_vs_of(all_mice, save_path, ramp_region="homebound", trial_type="all", collumn="speed_score")

    correlate_vr_vs_of(all_mice, save_path, ramp_region="outbound", trial_type="all", collumn="rate_map_correlation_first_vs_second_half")
    correlate_vr_vs_of(all_mice, save_path, ramp_region="homebound", trial_type="all", collumn="rate_map_correlation_first_vs_second_half")

    correlate_vr_vs_of(all_mice, save_path, ramp_region="outbound", trial_type="all", collumn="border_score")
    correlate_vr_vs_of(all_mice, save_path, ramp_region="homebound", trial_type="all", collumn="border_score")

    correlate_vr_vs_of(all_mice, save_path, ramp_region="outbound", trial_type="all", collumn="grid_score")
    correlate_vr_vs_of(all_mice, save_path, ramp_region="homebound", trial_type="all", collumn="grid_score")

    firing_rate_vr_vs_of(all_mice, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/vr_vs_of", ramp_region="outbound", trial_type="all")
    firing_rate_vr_vs_of(all_mice, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/vr_vs_of", ramp_region="homebound", trial_type="all")
    '''

    all_paths = []
    all_paths = get_recording_paths(all_paths, "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/245_sorted")
    all_paths = get_recording_paths(all_paths, "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/1124_sorted")
    all_paths = get_recording_paths(all_paths, "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M1_sorted")
    all_paths = get_recording_paths(all_paths, "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M6_sorted")
    all_paths = get_recording_paths(all_paths, "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M2_sorted")
    all_paths = get_recording_paths(all_paths, "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M3_sorted")
    all_paths = get_recording_paths(all_paths, "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted")
    all_paths = get_recording_paths(all_paths, "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted")

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    ramp_scores_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ramp_score_coeff_export.csv"
    tetrode_location_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/tetrode_locations.csv"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs"
    theta_df_VR = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta/theta_df_VR.pkl")
    linear_model_path = pd.read_csv("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_results_linearmodel.txt", sep="\t")
    linear_model_path['lm_result_homebound'] = linear_model_path['lm_result_homebound'].replace(["NoSlope"],"None")
    linear_model_path['lm_result_outbound'] = linear_model_path['lm_result_outbound'].replace(["NoSlope"],"None")

    trialtypes_linear_model = pd.read_csv("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_results_linearmodel_trialtypes.txt", sep="\t")
    ramp_scores = pd.read_csv(ramp_scores_path)
    tetrode_locations = pd.read_csv(tetrode_location_path)

    ramp_scores = add_short_session_id(ramp_scores)
    tetrode_locations = add_short_session_id(tetrode_locations)
    data = add_locations(ramp_scores, tetrode_locations)
    data = absolute_ramp_score(data)
    data = add_theta(data, theta_df_VR)
    data = add_lm(data, linear_model_path)
    data = analyse_ramp_driver(data, trialtypes_linear_model)
    data = add_full_session_id(data, all_paths)
    data = add_cohort_mouse_label(data)
    data = add_recording_day(data)
    data = remove_mouse(data, cohort_mouse_list=["C2_1124"])
    data = remove_location_classification(data, locations=["UN"])


    percentage_encoding(data, lmer_result="P", ramp_region="outbound", save_path=save_path, split="two")
    percentage_encoding(data, lmer_result="P", ramp_region="outbound", save_path=save_path, split="None")

    percentage_encoding(data, lmer_result="P", ramp_region="homebound", save_path=save_path, split="two")
    percentage_encoding(data, lmer_result="P", ramp_region="homebound", save_path=save_path, split="None")

    percentage_ramp(data, "outbound", save_path, split="daily")
    percentage_ramp(data, "outbound", save_path, split="two")
    percentage_ramp(data, "outbound", save_path, split="None")
    percentage_ramp(data, "homebound", save_path, split="daily")
    percentage_ramp(data, "homebound", save_path, split="two")
    percentage_ramp(data, "homebound", save_path, split="None")

    percentage_ramp_rs(data, "outbound", save_path, split="daily")
    percentage_ramp_rs(data, "outbound", save_path, split="two")
    percentage_ramp_rs(data, "outbound", save_path, split="None")
    percentage_ramp_rs(data, "homebound", save_path, split="daily")
    percentage_ramp_rs(data, "homebound", save_path, split="two")
    percentage_ramp_rs(data, "homebound", save_path, split="None")


    ramp_histogram_by_mouse(data, save_path)
    # to analyse theta modulate between cue dependant and independant neurons

    #cue_theta_location_bar(data, save_path)
    #cue_theta_location_hist(data, save_path)

    # chi squared slope by region
    for trial_type in ["beaconed", "all", "non_beaconed"]:
        for ramp_region in ["outbound", "homebound", "all"]:
            location_slope(data, collumn ="lm_result_ob", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)
            location_slope(data, collumn ="lm_result_hb", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)
            location_slope(data, collumn ="lmer_result_ob", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)
            location_slope(data, collumn ="lmer_result_hb", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)

            mouse_slope(data, collumn ="lm_result_ob", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)
            mouse_slope(data, collumn ="lm_result_hb", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)
            mouse_slope(data, collumn ="lmer_result_ob", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)
            mouse_slope(data, collumn ="lmer_result_hb", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)

    location_slope(data, collumn="ramp_driver", save_path=save_path, ramp_region="outbound", trial_type="all")
    mouse_slope(data, collumn="ramp_driver", save_path=save_path, ramp_region="outbound", trial_type="all")


    for filter_by_slope in [True, False]:
        print("filter_by_slope=", filter_by_slope)
        for trial_type in ["beaconed", "all"]:
            print("trial_type=", trial_type)
            for ramp_region in ["outbound", "homebound", "all"]:
                print("ramp_region=", ramp_region)
                location_ramp(data, collumn="ramp_score", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs/cumhists", ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)
                location_ramp(data, collumn="abs_ramp_score", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs/cumhists", ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)

                mouse_ramp(data, collumn="ramp_score", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)
                mouse_ramp(data, collumn="abs_ramp_score", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)

    # theta comparison by region
    location_ramp(data, collumn="ThetaIndex", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", ramp_region=ramp_region, trial_type=trial_type, print_p=True)
    location_ramp(data, collumn="ThetaPower", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", ramp_region=ramp_region, trial_type=trial_type, print_p=True)
    mouse_ramp(data, collumn="ThetaIndex", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", ramp_region=ramp_region, trial_type=trial_type, print_p=True)
    mouse_ramp(data, collumn="ThetaPower", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", ramp_region=ramp_region, trial_type=trial_type, print_p=True)

    theta_df_VR = remove_mouse(theta_df_VR, cohort_mouse_list=["C2_1124", "C6_M1", "C6_M2", "C7_M1", "C7_M2",
                                                               "C8_M1", "C8_M2", "C9_M1", "C9_M2", "C9_M3",
                                                               "C9_M4", "C9_M5"])
    plot_theta(theta_df_VR, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    plot_theta_histogram(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    plot_lm_proportions(data, ramp_region="outbound", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    plot_lm_proportions(data, ramp_region="homebound", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    simple_lm_stack_theta(data, collumn="lmer_result_ob", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", ramp_region="outbound", trial_type="all", p=None)
    simple_lm_stack_theta(data, collumn="lmer_result_hb", save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", ramp_region="homebound", trial_type="all", p=None)
    print("look now")

if __name__ == '__main__':
    main()





