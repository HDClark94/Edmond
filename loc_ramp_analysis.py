import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import itertools
import scipy
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

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

def simple_bar(data, collumn, save_path=None, ramp_region=None, trial_type=None, p=None, print_p=False, filter_by_slope=False):
    fig, ax = plt.subplots(figsize=(3,6))
    p_str = get_p_text(p, ns=True)
    #ax.set_title("rr= "+ramp_region+", tt= "+trial_type +", p="+p_str, fontsize=12)

    PS = data[data.tetrode_location == "PS"]
    MEC = data[data.tetrode_location == "MEC"]
    UN = data[data.tetrode_location == "UN"]

    objects = ("PS", "MEC", "UN")
    x_pos = np.arange(len(objects))

    ax.bar(x_pos, [np.mean(np.asarray(PS[collumn])), np.mean(np.asarray(MEC[collumn])), np.mean(np.asarray(UN[collumn]))],
           yerr=  [stats.sem(np.asarray(PS[collumn])), stats.sem(np.asarray(MEC[collumn])), stats.sem(np.asarray(UN[collumn]))],
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10,
           color =['b', 'r', 'grey'])

    #ax.text(0.95, 1, p_str, ha='left', va='top', transform=ax.transAxes, fontsize=20)
    plt.xticks(x_pos, objects, fontsize=15)
    plt.ylabel(get_tidy_title(collumn),  fontsize=20)
    plt.xlim((-0.5,2.5))
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

    objects = ("PS", "MEC", "UN")
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
    plt.xticks(x_pos, objects, fontsize=15)
    plt.ylabel("Percent of neurons",  fontsize=20)
    plt.xlim((-0.5,2.5))
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
        row["ThetaIndex"] = thetaIdx
        row["ThetaPower"] = thetaPwr
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

    PS = data[data.tetrode_location == "PS"]
    MEC = data[data.tetrode_location == "MEC"]
    UN = data[data.tetrode_location == "UN"]

    p_ks = stats.ks_2samp(np.asarray(PS[collumn]), np.asarray(MEC[collumn]))[1]
    p_st = stats.ttest_ind(np.asarray(PS[collumn]), np.asarray(MEC[collumn]))[1]

    simple_histogram(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=p_ks, filter_by_slope=filter_by_slope)
    simple_boxplot(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=p_st, filter_by_slope=filter_by_slope)
    simple_bar(data, collumn, save_path, ramp_region=ramp_region, trial_type=trial_type, p=p_st, print_p=print_p, filter_by_slope=filter_by_slope)
    return

def location_slope(data, collumn, save_path, ramp_region="outbound", trial_type="beaconed", print_p=False):
    data = data[(data[collumn] != np.nan)]

    # only look at beacoend and outbound
    data = data[(data.trial_type == trial_type) &
                (data.ramp_region == ramp_region)]

    PS = data[data.tetrode_location == "PS"]
    MEC = data[data.tetrode_location == "MEC"]
    UN = data[data.tetrode_location == "UN"]

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

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')#

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

    # to analyse theta modulate between cue dependant and independant neurons
    cue_theta_location_bar(data, save_path)
    cue_theta_location_hist(data, save_path)

    # chi squared slope by region
    for trial_type in ["beaconed"]:
        for ramp_region in ["outbound", "homebound", "all"]:
            location_slope(data, collumn="lm_result_ob", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)
            location_slope(data, collumn="lm_result_hb", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)
            location_slope(data, collumn="lmer_result_ob", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)
            location_slope(data, collumn="lmer_result_hb", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type)

    location_slope(data, collumn="ramp_driver", save_path=save_path, ramp_region="outbound", trial_type="all")

    for filter_by_slope in [True, False]:
        for trial_type in ["beaconed"]:
            for ramp_region in ["outbound", "homebound", "all"]:
                location_ramp(data, collumn="ramp_score", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)
                location_ramp(data, collumn="abs_ramp_score", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, filter_by_slope=filter_by_slope)

    # theta comparison by region
    location_ramp(data, collumn="ThetaIndex", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, print_p=True)
    location_ramp(data, collumn="ThetaPower", save_path=save_path, ramp_region=ramp_region, trial_type=trial_type, print_p=True)

    print("look now")

if __name__ == '__main__':
    main()


