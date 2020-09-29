import pandas as pd
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
        return "orchid"
    elif lmer_result=="PS":
        return "indianred"
    elif lmer_result=="A":
        return "lightskyblue"
    elif lmer_result=="S":
        return "salmon"
    elif lmer_result=="P":
        return "deeppink"
    elif lmer_result=="PSA":
        return "forestgreen"
    elif lmer_result=="SA":
        return "mediumaquamarine"
    elif lmer_result=="None":
        return "grey"

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
    p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)
    ax.set_title(p_str, fontsize=12)

    PS = data[data.tetrode_location == "PS"]
    MEC = data[data.tetrode_location == "MEC"]
    UN = data[data.tetrode_location == "UN"]

    #ax.hist(np.asarray(UN[collumn]), bins=50, alpha=0.2, color="k", label="Unclassified", histtype="step", density=True)
    ax.hist(np.asarray(MEC[collumn]), bins=50, alpha=0.5, color="r", label="MEC", histtype="step", density=True, cumulative=True, linewidth=4)
    ax.hist(np.asarray(PS[collumn]), bins=50, alpha=0.5, color="b", label="PS", histtype="step", density=True, cumulative=True, linewidth=4)

    ax.set_ylabel("Cumulative Density", fontsize=15)
    ax.set_xlabel(get_tidy_title(collumn), fontsize=15)
    if collumn == "ramp_score":
        ax.set_xlim(left=-0.75, right=0.75)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    #plt.subplots_adjust(left=0.2, right=0.6, top=0.8, bottom=0.2)
    #ax.legend(loc="upper right")
    #ax.set_xlim(left=0)
    if save_path is not None:
        if filter_by_slope:
            plt.savefig(save_path+"/FBS_location_histo_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")
        else:
            plt.savefig(save_path+"/location_histo_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")

    plt.show()
    plt.close()
    return

def simple_boxplot(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, filter_by_slope=False):
    ig, ax = plt.subplots(figsize=(6,3))
    p_str = get_p_text(p, ns=True)
    ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)
    ax.set_title(p_str, fontsize=20)

    PS = data[data.tetrode_location == "PS"]
    MEC = data[data.tetrode_location == "MEC"]
    UN = data[data.tetrode_location == "UN"]

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
    plt.yticks(y_pos, objects, fontsize=20)
    plt.xlabel(get_tidy_title(collumn),  fontsize=20)
    plt.ylim((-1,3))
    plt.ylim((-0.75,1.5))
    if collumn == "ramp_score":
        ax.set_xlim(left=-0.75, right=0.75)
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    #plt.title('Programming language usage')
    #ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        if filter_by_slope:
            plt.savefig(save_path+"/FBS_location_boxplot_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")
        else:
            plt.savefig(save_path+"/location_boxplot_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")
    plt.show()
    plt.close()

def simple_bar_mouse(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False, filter_by_slope=False):
    fig, ax = plt.subplots(figsize=(3,6))
    p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)

    objects = np.unique(data["cohort_mouse"])
    x_pos = np.arange(len(objects))

    for i in range(len(objects)):
        y = data[(data["cohort_mouse"] == objects[i])]
        ax.bar(x_pos[i], np.mean(np.asarray(y[collumn])), yerr=stats.sem(np.asarray(y[collumn])), align='center', alpha=0.5, ecolor='black', capsize=10)

    #ax.text(0.95, 1, p_str, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    plt.xticks(x_pos, objects, fontsize=8)
    plt.xticks(rotation=-45)
    plt.ylabel(get_tidy_title(collumn),  fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    if collumn == "ramp_score":
        plt.ylim(-0.6, 0.6)
    elif collumn == "abs_ramp_score":
        plt.ylim(0, 0.6)
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
            plt.savefig(save_path+"/FBS_mouse_bar_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")
        else:
            plt.savefig(save_path+"/mouse_bar_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")
    plt.show()
    plt.close()

def simple_bar_location(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False, filter_by_slope=False):
    fig, ax = plt.subplots(figsize=(3,6))
    p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)

    objects = np.unique(data["tetrode_location"])
    x_pos = np.arange(len(objects))

    for i in range(len(objects)):
        y = data[(data["tetrode_location"] == objects[i])]
        ax.bar(x_pos[i], np.mean(np.asarray(y[collumn])), yerr=stats.sem(np.asarray(y[collumn])), align='center', alpha=0.5, ecolor='black', capsize=10)

    #ax.text(0.95, 1, p_str, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    plt.xticks(x_pos, objects, fontsize=8)
    plt.xticks(rotation=-45)
    plt.ylabel(get_tidy_title(collumn),  fontsize=20)
    plt.xlim((-0.5, len(objects)-0.5))
    if collumn == "ramp_score":
        plt.ylim(-0.6, 0.6)
    elif collumn == "abs_ramp_score":
        plt.ylim(0, 0.6)
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
            plt.savefig(save_path+"/FBS_location_bar_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")
        else:
            plt.savefig(save_path+"/location_bar_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")
    plt.show()
    plt.close()

def simple_lm_stack_mouse(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False):
    fig, ax = plt.subplots(figsize=(3,6))
    #p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)

    aggregated = data.groupby([collumn, "cohort_mouse"]).count().reset_index()
    if (collumn == "lm_result_hb") or (collumn == "lm_result_ob"):
        colors_lm = ["pink", "black", "grey", "green"]
        groups = ["Negative", "None", "NoSlope", "Positive"]
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
        plt.savefig(save_path+"/mouse_slope_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")
    plt.show()
    plt.close()


def simple_lm_stack(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False):
    fig, ax = plt.subplots(figsize=(3,6))
    #p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)

    aggregated = data.groupby([collumn, "tetrode_location"]).count().reset_index()
    if (collumn == "lm_result_hb") or (collumn == "lm_result_ob"):
        colors_lm = ["pink", "black", "grey", "green"]
        groups = ["Negative", "None", "NoSlope", "Positive"]
    elif (collumn == "ramp_driver"):
        colors_lm = ["grey", "green", "yellow"]
        groups = [ "None", "PI", "Cue"]
    else:
        colors_lm = ["lightgrey", "lightslategray", "limegreen", "violet", "orange",
                     "cornflowerblue", "yellow", "lightcoral"]
        groups = ["P", "S", "A", "PS", "PA", "SA", "PSA", "None"]
        groups = ["None", "PSA", "SA", "PA", "PS", "A", "S", "P"]

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
        plt.savefig(save_path+"/location_slope_"+"_tt_"+trial_type+"_rr_"+ramp_region+"_"+collumn+".png")
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
        cluster_id = row.cluster_id.iloc[0]

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
    #simple_histogram(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, filter_by_slope=filter_by_slope)
    #simple_boxplot(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, filter_by_slope=filter_by_slope)
    simple_bar_mouse(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, print_p=print_p, filter_by_slope=filter_by_slope)
    return

def location_ramp(data, collumn, save_path, ramp_region="outbound", trial_type="beaconed", print_p=False, filter_by_slope=False):

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

    #simple_histogram(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, filter_by_slope=filter_by_slope)
    #simple_boxplot(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=None, filter_by_slope=filter_by_slope)
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
    plt.savefig(save_path+"/cue_theta_location_hist_PS.png")
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
    plt.savefig(save_path+"/cue_theta_location_hist_MEC.png")
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
    plt.savefig(save_path+"/theta_cue_location.png")
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
                    plt.savefig(save_path+"/bymouse_ramp_histo_m_"+cohort_mouse+"_tt_"+trial_type+"_rr_"+ramp_region+".png")

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
        plt.savefig(save_path+"/Boccara_theta_cohort_mouse_earlylate"+suffix+".png")
    elif split == "None":
        plt.savefig(save_path+"/Boccara_theta_cohort_mous"+suffix+".png")
    elif split == "daily":
        plt.savefig(save_path+"/Boccara_theta_cohort_mouse_daily"+suffix+".png")

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
        plt.savefig(save_path+"/grid_score_cohort_mouse_earlylate"+suffix+".png")
    elif split == "None":
        plt.savefig(save_path+"/grid_score_cohort_mous"+suffix+".png")
    elif split == "daily":
        plt.savefig(save_path+"/grid_score_cohort_mouse_daily"+suffix+".png")

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
        plt.savefig(save_path+"/hd_score_cohort_mouse_earlylate"+suffix+".png")
    elif split == "None":
        plt.savefig(save_path+"/hd_score_cohort_mous"+suffix+".png")
    elif split == "daily":
        plt.savefig(save_path+"/hd_score_cohort_mouse_daily"+suffix+".png")

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
        plt.savefig(save_path+"/prop_"+lmer_result+"_cohort_mouse_earlylate"+suffix+".png")
    elif split == "None":
        plt.savefig(save_path+"/prop_"+lmer_result+"_cohort_mous"+suffix+".png")

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
        plt.savefig(save_path+"/prop_ramp_cohort_mouse_earlylate"+suffix+".png")
    elif split == "None":
        plt.savefig(save_path+"/prop_ramp_cohort_mous"+suffix+".png")
    elif split == "daily":
        plt.savefig(save_path+"/prop_ramp_cohort_mouse_daily"+suffix+".png")

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
        plt.savefig(save_path+"/prop_ramp_rs_cohort_mouse_earlylate"+suffix+".png")
    elif split == "None":
        plt.savefig(save_path+"/prop_ramp_rs_cohort_mous"+suffix+".png")
    elif split == "daily":
        plt.savefig(save_path+"/prop_ramp_rs_cohort_mouse_daily"+suffix+".png")

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

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #------------------------------------------------------------------------------------------------#
    # for looking a head directional proportion of cells in each mouse
    '''
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
    new = add_cohort_mouse_label(new)
    new = add_recording_day(new)
    hd_per_mouse(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", threshold=0.4)
    gc_per_mouse(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", threshold=0.4)
    percentage_boccara(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", split="daily", suffix="")
    percentage_boccara(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", split="two", suffix="")
    percentage_boccara(new, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs", split="None", suffix="")
    '''
    #=================================================================================================#

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
    for trial_type in ["beaconed"]:
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
        for trial_type in ["beaconed"]:
            for ramp_region in ["outbound", "homebound", "all"]:
                location_ramp(data, collumn="ramp_score", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)
                location_ramp(data, collumn="abs_ramp_score", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)

                mouse_ramp(data, collumn="ramp_score", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)
                mouse_ramp(data, collumn="abs_ramp_score", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)

    # theta comparison by region
    location_ramp(data, collumn="ThetaIndex", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, print_p=True)
    location_ramp(data, collumn="ThetaPower", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, print_p=True)
    mouse_ramp(data, collumn="ThetaIndex", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, print_p=True)
    mouse_ramp(data, collumn="ThetaPower", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, print_p=True)

    print("look now")

if __name__ == '__main__':
    main()


