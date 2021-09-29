import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pylab as plt
import math
import os
import pandas as pd
import PostSorting.parameters
import pickle
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import traceback
import warnings
import sys
from PostSorting.theta_modulation import *
from Edmond.ramp_cells_of import *
import scipy
from Edmond.loc_ramp_analysis import remove_mouse
warnings.filterwarnings('ignore')

test_params = PostSorting.parameters.Parameters()

def track_theta(path_to_recording_list, theta_df):

    try: # this is if path to recording list is a folder name
        list_of_recordings_tmp = [f.path for f in os.scandir(path_to_recording_list) if f.is_dir()]
        list_of_recordings = []
        for i in range(len(list_of_recordings_tmp)):
            list_of_recordings.append(list_of_recordings_tmp[i].split("/mnt/datastore/")[1])

    except:
        recordings_file_reader = open(path_to_recording_list, 'r')
        recordings = recordings_file_reader.readlines()
        list_of_recordings = list([x.strip() for x in recordings])

    for i in range(len(list_of_recordings)):
        try:
            recording_path = list_of_recordings[i]
            spatial_firing_path = recording_path + "/MountainSort/DataFrames/spatial_firing.pkl"
            if os.path.exists('/mnt/datastore/'+spatial_firing_path) is False:
                spatial_firing_path = recording_path + "/MountainSort/DataFrames/spatial_firing_all.pkl"

            spatial_firing = pd.read_pickle('/mnt/datastore/'+spatial_firing_path)
            spatial_firing= spatial_firing.sort_values(by=['cluster_id'])

            if "Curated" in list(spatial_firing):
                spatial_firing = spatial_firing[spatial_firing["Curated"] == 1]

            if len(spatial_firing)>0:
                if "Theta_index" not in list(spatial_firing):
                    test_params.set_output_path('/mnt/datastore/Harry/Mouse_data_for_sarah_paper/theta_index_figs/'+recording_path+"/MountainSort")
                    #test_params.set_output_path('/mnt/datastore/'+recording_path+"/MountainSort")
                    spatial_firing = calculate_theta_index(spatial_firing, test_params)

                for j in range(len(spatial_firing)):
                    spatial_firing_cluster = spatial_firing.iloc[j]

                    row = pd.DataFrame()
                    row["cohort_mouse"] = [get_cohort_mouse(recording_path)]
                    row["recording_day"] = [get_day(recording_path)]
                    row["cluster_id"] = [spatial_firing_cluster["cluster_id"]]
                    row["tetrode"] =    [spatial_firing_cluster["tetrode"]]
                    row["ThetaIndex"] = [spatial_firing_cluster["ThetaIndex"]]
                    row["ThetaPower"] = [spatial_firing_cluster["ThetaPower"]]
                    row["Boccara_theta_class"] = [spatial_firing_cluster["Boccara_theta_class"]]
                    row["session_id"] = [spatial_firing_cluster["session_id"]]
                    theta_df = pd.concat([row, theta_df], ignore_index=True)

            print("successful with ", recording_path)

        except Exception as ex:
            print("failed on recording, ", recording_path)
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)

    return theta_df


def get_day(full_session_id):
    session_id = full_session_id.split("/")[-1]
    training_day = session_id.split("_")[1]
    training_day = training_day.split("D")[1]
    return int(training_day)

def get_cohort_mouse(full_session_id):
    session_id = full_session_id.split("/")[-1]
    mouse = session_id.split("_D")[0]

    if "Junji" in full_session_id:
        cohort = "C6"
    elif "Ian"  in full_session_id:
        cohort = "C7"
    elif ("Harry" in full_session_id) and ("2020" in full_session_id):
        cohort = "C8"
    elif ("Harry" in full_session_id) and ("2019" in full_session_id):
        cohort = "C9"
    else:
        cohort = get_tidy_title(full_session_id.split("/")[-4])
    return cohort+"_"+mouse

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

    fig, ax = plt.subplots(figsize=(9,6))

    for mouse in np.unique(theta_df["cohort_mouse"]):
        mouse_df = theta_df[theta_df["cohort_mouse"] == mouse]
        mouse_df = mouse_df.dropna()

        mean = mouse_df.groupby('recording_day')["ThetaPower"].mean().reset_index()
        sem = mouse_df.groupby('recording_day')["ThetaPower"].sem().reset_index()
        max_by_day = mouse_df.groupby('recording_day')["ThetaPower"].max().reset_index()
        mean = mean.fillna(0)
        sem = sem.fillna(0)
        max_by_day = max_by_day.fillna(0)

        gauss_kernel = Gaussian1DKernel(2) # 2 days
        smoothed_max_theta = convolve(max_by_day["ThetaPower"], gauss_kernel)

        plt.plot(max_by_day["recording_day"], smoothed_max_theta, label=mouse)

    ax.set_xlabel("Training day", fontsize=15)
    ax.set_ylabel("Maximum Theta Power", fontsize=15)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.tight_layout()
    plt.savefig(save_path+"/tracked_theta_power.png", dpi=300)
    plt.show()

def plot_theta_at_max_rs(theta_df, save_path):
    fig, ax = plt.subplots(figsize=(9,6))
    gauss_kernel = Gaussian1DKernel(2) # 2 days

    for mouse in np.unique(theta_df["cohort_mouse"]):
        mouse_df = theta_df[theta_df["cohort_mouse"] == mouse]
        # only look at outbound and beaconed trials
        mouse_df = mouse_df[(mouse_df.trial_type == "beaconed") & (mouse_df.ramp_region == "outbound")]

        max_by_day_rs = mouse_df.groupby('recording_day')["abs_ramp_score"].max().reset_index()
        smoothed_max_rs = convolve(max_by_day_rs["abs_ramp_score"], gauss_kernel)

        max_theta_at_max_by_day_rs = pd.DataFrame()
        for index, row in max_by_day_rs.iterrows():
            row =  row.to_frame().T.reset_index(drop=True)

            tmp = mouse_df[(mouse_df["abs_ramp_score"] == row["abs_ramp_score"].iloc[0])]
            row["ThetaIndex"] = max(tmp["ThetaIndex"])

            max_theta_at_max_by_day_rs = pd.concat([max_theta_at_max_by_day_rs, row], ignore_index=True)

        smoothed_max_theta = convolve(max_theta_at_max_by_day_rs["ThetaIndex"], gauss_kernel)

        plt.scatter(smoothed_max_theta, smoothed_max_rs, label=mouse)
        plot_regression(ax, pd.Series(smoothed_max_theta), pd.Series(smoothed_max_rs))

    ax.set_ylim(bottom=0, top=0.5)
    ax.vlines(x=0.07, ymin=0, ymax=0.5, linestyles="--") # threshold specified for rythmic cells Kornienko et al.
    ax.set_xlabel("Theta Index", fontsize=15)
    ax.set_ylabel("Max Abs Ramp Score by Day", fontsize=15)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=20)
    #plt.tight_layout()
    plt.savefig(save_path+"/theta_at_max_rs.png", dpi=300)
    plt.show()

def plot_max_rs_at_max_theta(theta_df, save_path):
    fig, ax = plt.subplots(figsize=(9,6))
    max_rs = 0
    gauss_kernel = Gaussian1DKernel(2) # 2 days

    for mouse in np.unique(theta_df["cohort_mouse"]):
        mouse_df = theta_df[theta_df["cohort_mouse"] == mouse]
        # only look at outbound and beaconed trials
        mouse_df = mouse_df[(mouse_df.trial_type == "beaconed") & (mouse_df.ramp_region == "outbound")]

        max_by_day_theta = mouse_df.groupby('recording_day')["ThetaIndex"].max().reset_index()
        smoothed_max_theta = convolve(max_by_day_theta["ThetaIndex"], gauss_kernel)

        max_ramp_at_max_by_day_theta = pd.DataFrame()
        for index, row in max_by_day_theta.iterrows():
            row =  row.to_frame().T.reset_index(drop=True)

            tmp = mouse_df[(mouse_df["ThetaIndex"] == row["ThetaIndex"].iloc[0])]
            row["abs_ramp_score"] = max(tmp["abs_ramp_score"])

            max_ramp_at_max_by_day_theta = pd.concat([max_ramp_at_max_by_day_theta, row], ignore_index=True)

        smoothed_max_rs = convolve(max_ramp_at_max_by_day_theta["abs_ramp_score"], gauss_kernel)

        plt.scatter(smoothed_max_theta, smoothed_max_rs, label=mouse)
        plot_regression(ax, pd.Series(smoothed_max_theta), pd.Series(smoothed_max_rs))

        if max(max_ramp_at_max_by_day_theta.abs_ramp_score) > max_rs:
            max_rs = max(max_ramp_at_max_by_day_theta.abs_ramp_score)

    ax.set_ylim(bottom=0, top=0.3)
    ax.vlines(x=0.07, ymin=0, ymax=1, linestyles="--") # threshold specified for rythmic cells Kornienko et al.
    ax.set_xlabel("Max Theta Index by Day", fontsize=15)
    ax.set_ylabel("Abs Ramp Score", fontsize=15)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=20)
    #plt.tight_layout()
    plt.savefig(save_path+"/max_rs_at_max_theta.png", dpi=300)
    plt.show()


def plot_max_ramp_vs_max_theta(theta_df, save_path):
    fig, ax = plt.subplots(figsize=(9,6))
    gauss_kernel = Gaussian1DKernel(2) # 2 days
    max_rs = 0

    for mouse in np.unique(theta_df["cohort_mouse"]):
        mouse_df = theta_df[theta_df["cohort_mouse"] == mouse]
        # only look at outbound and beaconed trials
        mouse_df = mouse_df[(mouse_df.trial_type == "beaconed") & (mouse_df.ramp_region == "outbound")]

        max_by_day_theta = mouse_df.groupby('recording_day')["ThetaIndex"].max().reset_index()
        smoothed_max_theta = convolve(max_by_day_theta["ThetaIndex"], gauss_kernel)

        max_by_day_rs = mouse_df.groupby('recording_day')["abs_ramp_score"].max().reset_index()
        smoothed_max_rs = convolve(max_by_day_rs["abs_ramp_score"], gauss_kernel)

        plt.scatter(smoothed_max_theta, smoothed_max_rs, label=mouse)
        plot_regression(ax, pd.Series(smoothed_max_theta), pd.Series(smoothed_max_rs))

        if max(smoothed_max_rs) > max_rs:
            max_rs = max(smoothed_max_rs)

    ax.set_ylim(bottom=0, top=max_rs)
    ax.vlines(x=0.07, ymin=0, ymax=max_rs, linestyles="--") # threshold specified for rythmic cells Kornienko et al.
    ax.set_xlabel("Max Theta Index by Day", fontsize=15)
    ax.set_ylabel("Max Abs Ramp Score by Day", fontsize=15)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=20)
    #plt.tight_layout()
    plt.savefig(save_path+"/max_theta_vs_max_abs_rs.png", dpi=300)
    plt.show()

def plot_max_ramp_score(theta_df, save_path):

    fig, ax = plt.subplots(figsize=(9,6))

    for mouse in np.unique(theta_df["cohort_mouse"]):
        mouse_df = theta_df[theta_df["cohort_mouse"] == mouse]

        # only look at outbound and beaconed trials
        mouse_df = mouse_df[(mouse_df.trial_type == "beaconed") & (mouse_df.ramp_region == "outbound")]
        #mouse_df = mouse_df.dropna()

        mean = mouse_df.groupby('recording_day')["abs_ramp_score"].mean().reset_index()
        sem = mouse_df.groupby('recording_day')["abs_ramp_score"].sem().reset_index()
        max_by_day = mouse_df.groupby('recording_day')["abs_ramp_score"].max().reset_index()
        mean = mean.fillna(0)
        sem = sem.fillna(0)
        max_by_day = max_by_day.fillna(0)

        gauss_kernel = Gaussian1DKernel(2) # 2 days

        if len(max_by_day["abs_ramp_score"]) > 0:
            smoothed_max_theta = convolve(max_by_day["abs_ramp_score"], gauss_kernel)
            plt.plot(max_by_day["recording_day"], smoothed_max_theta, label=mouse)

        #plt.fill_between(mean["recording_day"], mean["ThetaIndex"]-sem["ThetaIndex"], mean["ThetaIndex"]+sem["ThetaIndex"], alpha=0.5)

    #ax.hlines(y=0.07, xmin=0, xmax=max(theta_df["recording_day"]), linestyles="--") # threshold specified for rythmic cells Kornienko et al.
    ax.set_xlabel("Training day", fontsize=15)
    ax.set_ylabel("Maximum Abs Ramp Score", fontsize=15)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.tight_layout()
    plt.savefig(save_path+"/tracked_ramp_score.png", dpi=300)
    plt.show()

def add_ramp_scores(theta_df, ramp_lm, ramp_scores, tetrode_locations):
    new=pd.DataFrame()

    for index_j, row_j in theta_df.iterrows():
        row_j =  row_j.to_frame().T.reset_index(drop=True)
        session_id = row_j["session_id"].iloc[0]
        cluster_id = row_j["cluster_id"].iloc[0]
        recording_day = row_j["recording_day"].iloc[0]

        paired_ramp_lm = ramp_lm[((ramp_lm["session_id"] == session_id) & (ramp_lm["cluster_id"] == cluster_id))]
        paired_ramp_score = ramp_scores[((ramp_scores["session_id"] == session_id) & (ramp_scores["cluster_id"] == cluster_id))]
        paired_location = tetrode_locations[(tetrode_locations["session_id"] == session_id)]

        for index_i, row_i in paired_ramp_score.iterrows():
            row_i =  row_i.to_frame().T.reset_index(drop=True)
            if len(paired_location) < 1:
                row_i["tetrode_depth"] = np.nan
            else:
                row_i["tetrode_depth"] = paired_location["tetrode_depth"].iloc[0]
            row_i["ThetaIndex"] = row_j["ThetaIndex"]
            row_i["ThetaPower"] = row_j["ThetaPower"]
            row_i["cohort_mouse"] = row_j["cohort_mouse"]
            row_i["lm_result_b_outbound"] = paired_ramp_lm["lm_result_b_outbound"].iloc[0]
            row_i["lm_result_b_homebound"] = paired_ramp_lm["lm_result_b_homebound"].iloc[0]
            row_i["lm_result_nb_outbound"] = paired_ramp_lm["lm_result_nb_outbound"].iloc[0]
            row_i["lm_result_nb_homebound"] = paired_ramp_lm["lm_result_nb_homebound"].iloc[0]
            row_i["lm_result_p_outbound"] = paired_ramp_lm["lm_result_p_outbound"].iloc[0]
            row_i["lm_result_p_homebound"] = paired_ramp_lm["lm_result_p_homebound"].iloc[0]
            if "lmer_result_outbound" in list(paired_ramp_lm):
                row_i["lmer_result_outbound"] = paired_ramp_lm["lmer_result_outbound"].iloc[0]
                row_i["lmer_result_homebound"] = paired_ramp_lm["lmer_result_homebound"].iloc[0]
            row_i["recording_day"] = recording_day
            new = pd.concat([new, row_i], ignore_index=True)

    new = correct_datatypes(new, ignore_of=True)
    new = analyse_ramp_driver(new)
    #new = get_best_ramp_score(new) #TODO this doesn't work with the new ramp score format
    new = get_best_theta_scores_VR(new)
    new = absolute_ramp_score(new)
    return new

def boxplot_theta(theta_df_VR, save_path, theta_threshold=0.07, best_theta=False):

    for trial_type in np.unique(theta_df_VR.trial_type):
        for ramp_region in np.unique(theta_df_VR.ramp_region):

            trial_type_theta_df = theta_df_VR[(theta_df_VR.trial_type == trial_type)]
            trial_type_theta_df = trial_type_theta_df[(trial_type_theta_df.ramp_region == ramp_region)]

            if best_theta:
                rythmic = trial_type_theta_df[(trial_type_theta_df.best_theta_idx_vr > theta_threshold)]
                non_rythmic = trial_type_theta_df[(trial_type_theta_df.best_theta_idx_vr < theta_threshold)]
            else:
                rythmic = trial_type_theta_df[(trial_type_theta_df.ThetaIndex > theta_threshold)]
                non_rythmic = trial_type_theta_df[(trial_type_theta_df.ThetaIndex < theta_threshold)]

            fig, ax = plt.subplots(figsize=(3,6))
            objects = ('NR', 'TR')
            x_pos = np.arange(len(objects))

            boxprops = dict(linewidth=3, color='k')
            medianprops = dict(linewidth=3, color='k')
            capprops = dict(linewidth=3, color='k')
            whiskerprops = dict(linewidth=3, color='k')

            bplot1 = ax.boxplot(np.asarray(non_rythmic["ramp_score"]), positions = [0], widths=0.9,
                        boxprops=boxprops, medianprops=medianprops,
                        whiskerprops=whiskerprops, capprops=capprops, patch_artist=True)

            bplot2 = ax.boxplot(np.asarray(rythmic["ramp_score"]), positions = [1], widths=0.9,
                                 boxprops=boxprops, medianprops=medianprops,
                                 whiskerprops=whiskerprops, capprops=capprops, patch_artist=True)

            # fill with colors
            colors = ['r', 'grey']
            i=0
            for bplot in (bplot1, bplot2):
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(colors[i])
                i+=1

            print("trial type = ", trial_type, ", ramp_region = ", ramp_region)
            p = scipy.stats.ttest_ind(np.asarray(non_rythmic["ramp_score"]), np.asarray(rythmic["ramp_score"]))[1]
            print(p)

            ax.text(0.95, 1.25, "p= "+str(np.round(p, decimals=4)), ha='right', va='top', transform=ax.transAxes, fontsize=20)

            plt.xticks(x_pos, objects, fontsize=15)
            plt.ylabel("Ramp Score",  fontsize=15)
            plt.xlim((-1,2))
            #plt.ylim((0,1))
            #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
            #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
            #plt.title('Programming language usage')
            #ax.legend()
            ax.tick_params(axis='both', which='major', labelsize=20)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            if best_theta:
                plt.savefig(save_path+"/"+trial_type+"_"+ramp_region+"_best_theta_boxplot.png", dpi=300)
            else:
                plt.savefig(save_path+"/"+trial_type+"_"+ramp_region+"_absolute_theta_boxplot.png", dpi=300)
            plt.show()


            fig, ax = plt.subplots(figsize=(3,6))
            ax.hist(np.asarray(rythmic["ramp_score"]), bins=20, alpha=0.5, color="k", density=True)
            ax.hist(np.asarray(non_rythmic["ramp_score"]), bins=20, alpha=0.5, color="r", density=True)
            plt.xlabel("Ramp Score",  fontsize=15)
            plt.ylabel("Density",  fontsize=15)
            #plt.xlim((-1,2))
            #plt.ylim((0,1))
            #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
            #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
            ax.tick_params(axis='both', which='major', labelsize=20)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()

            if best_theta:
                plt.savefig(save_path+"/"+trial_type+"_"+ramp_region+"_best_theta_ramp_hist.png", dpi=300)
            else:
                plt.savefig(save_path+"/"+trial_type+"_"+ramp_region+"_absolute_theta_ramp_hist.png", dpi=300)
            plt.show()

def plot_lm_proportions(theta_df_VR, save_path, best_theta=False):

    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        if trial_type == "beaconed":
            collumn = "lm_result_b_outbound"
        elif trial_type == "non-beaconed":
            collumn = "lm_result_nb_outbound"
        elif trial_type == "probe":
            collumn = "lm_result_p_outbound"

        trial_type_theta_df = theta_df_VR[(theta_df_VR.trial_type == trial_type)]
        trial_type_theta_df = trial_type_theta_df[(trial_type_theta_df.ramp_region == "outbound")]

        if best_theta:
            rythmic = trial_type_theta_df[(trial_type_theta_df.best_theta_idx_vr > 0.07)]
            no_rythmic = trial_type_theta_df[(trial_type_theta_df.best_theta_idx_vr < 0.07)]
        else:
            rythmic = trial_type_theta_df[(trial_type_theta_df.ThetaIndex > 0.07)]
            no_rythmic = trial_type_theta_df[(trial_type_theta_df.ThetaIndex < 0.07)]

        fig, ax = plt.subplots(figsize=(3,6))

        pos_rythimc = len(rythmic[rythmic[collumn] == "Positive"])/len(rythmic)
        neg_rythmic = len(rythmic[rythmic[collumn] == "Negative"])/len(rythmic)
        non_rythmic = len(rythmic[rythmic[collumn] == "None"])/len(rythmic)

        pos_norythimc = len(no_rythmic[no_rythmic[collumn] == "Positive"])/len(no_rythmic)
        neg_norythmic = len(no_rythmic[no_rythmic[collumn] == "Negative"])/len(no_rythmic)
        non_norythmic = len(no_rythmic[no_rythmic[collumn] == "None"])/len(no_rythmic)

        ax.bar(x=1, height=non_rythmic, bottom=0, color="grey")
        ax.bar(x=1, height=neg_rythmic, bottom=non_rythmic, color="red")
        ax.bar(x=1, height=pos_rythimc, bottom=non_rythmic+ neg_rythmic, color="blue")

        ax.bar(x=0, height=non_norythmic, bottom=0, color="grey")
        ax.bar(x=0, height=neg_norythmic, bottom=non_norythmic, color="red")
        ax.bar(x=0, height=pos_norythimc, bottom=non_norythmic+ neg_norythmic, color="blue")

        ax.text(x=1 , y=0+0.05, s=str(len(rythmic[rythmic[collumn] == "None"])), color="white", fontsize=12, horizontalalignment='center')
        ax.text(x=1 , y=non_rythmic+0.05, s=str(len(rythmic[rythmic[collumn] == "Negative"])), color="white", fontsize=12, horizontalalignment='center')
        ax.text(x=1 , y=non_rythmic+ neg_rythmic+0.05, s=str(len(rythmic[rythmic[collumn] == "Positive"])), color="white", fontsize=12, horizontalalignment='center')
        ax.text(x=0 , y=0+0.05, s=str(len(no_rythmic[no_rythmic[collumn] == "None"])), color="white", fontsize=12, horizontalalignment='center')
        ax.text(x=0 , y=non_norythmic+0.05, s=str(len(no_rythmic[no_rythmic[collumn] == "Negative"])), color="white", fontsize=12, horizontalalignment='center')
        ax.text(x=0 , y=non_norythmic+ neg_norythmic+0.05, s=str(len(no_rythmic[no_rythmic[collumn] == "Positive"])), color="white", fontsize=12, horizontalalignment='center')

        ax.text(x=1 , y=1.05, s=str(np.round((len(rythmic)/(len(rythmic)+len(no_rythmic)))*100, decimals=0))+"%", color="black", fontsize=12, horizontalalignment='center')
        ax.text(x=0 , y=1.05, s=str(np.round((len(no_rythmic)/(len(rythmic)+len(no_rythmic)))*100, decimals=0))+"%", color="black", fontsize=12, horizontalalignment='center')


        objects = ('NR', 'TR')
        x_pos = np.arange(len(objects))
        plt.xticks(x_pos, objects, fontsize=15)

        plt.ylabel("Proportion Cells",  fontsize=15)
        plt.xlim((-1,2))
        plt.ylim((0,1))
        plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
        plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
        #plt.title("Trial type = "+get_tidy_title(trial_type))
        #ax.legend()
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        if best_theta:
            plt.savefig(save_path+"/"+trial_type+"_"+collumn+"_best_theta_proportion.png", dpi=300)
        else:
            plt.savefig(save_path+"/"+trial_type+"_"+collumn+"theta_proportion.png", dpi=300)
        plt.show()


def plot_theta_histogram(data, save_path):
    trial_type_theta_df = data[(data.trial_type == "beaconed")]
    trial_type_theta_df = trial_type_theta_df[(trial_type_theta_df.ramp_region == "outbound")]
    # remove cells which does not have any local theta cells
    #trial_type_theta_df = trial_type_theta_df[(trial_type_theta_df["best_theta_idx_vr"] > 0.07)]

    rythmic = trial_type_theta_df[(trial_type_theta_df["ThetaIndex"] > 0.07)]
    no_rythmic = trial_type_theta_df[(trial_type_theta_df["ThetaIndex"] < 0.07)]

    fig, ax = plt.subplots(figsize=(3,6))
    ax.hist(np.asarray(rythmic["ThetaIndex"]), bins=20, alpha=0.5, color="k")
    ax.hist(np.asarray(no_rythmic["ThetaIndex"]), bins=20, alpha=0.5, color="r")
    plt.xlabel("Theta Index",  fontsize=15)
    plt.ylabel("Number of Cells",  fontsize=15)
    #plt.xlim((-1,2))
    #plt.ylim((0,1))
    #plt.axvline(x=-1, ymax=1, ymin=0, linewidth=3, color="k")
    #plt.axhline(y=0, xmin=-1, xmax=2, linewidth=3, color="k")
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"/theta_histo.png", dpi=300)
    plt.show()

def main():
    test_params.set_sampling_rate(30000)

    ramp_path_lm = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_results_linearmodel_trialtypes.txt"
    ramp_scores_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ramp_score_coeff_export.csv"
    tetrode_location_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/tetrode_locations.csv"
    ramp_scores = pd.read_csv(ramp_scores_path)
    ramp_lm = pd.read_csv(ramp_path_lm, sep = "\t")
    tetrode_locations = pd.read_csv(tetrode_location_path)
                
    # VR cells
    '''

    theta_df = pd.DataFrame()

    # junjis 2019 cohort
    theta_df = track_theta("/mnt/datastore/Junji/Data/2019cohort1/vr", theta_df)
    theta_df = track_theta("/mnt/datastore/Junji/Data/2019cohort1/m1_part1", theta_df)
    theta_df = track_theta("/mnt/datastore/Junji/Data/2019cohort1/m2_part1", theta_df)

    # ians 2019 cohort
    theta_df = track_theta("/mnt/datastore/Ian/Ephys/VR", theta_df)

    theta_df = track_theta("/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M1_sorted", theta_df)
    theta_df = track_theta("/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort5/VirtualReality/M2_sorted", theta_df)

    theta_df = track_theta("/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/VirtualReality/M2_sorted", theta_df)
    theta_df = track_theta("/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/VirtualReality/M3_sorted", theta_df)

    theta_df = track_theta("/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/VirtualReality/M1_sorted", theta_df)
    theta_df = track_theta("/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort3/VirtualReality/M6_sorted", theta_df)

    theta_df = track_theta("/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort2/VirtualReality/245_sorted", theta_df)
    theta_df = track_theta("/mnt/datastore/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort2/VirtualReality/1124_sorted", theta_df)

    theta_df.to_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta/theta_df_VR.pkl")
    
    theta_df = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta/theta_df_VR.pkl")
    theta_df = track_theta("/mnt/datastore/Harry/Cohort6_july2020/vr", theta_df)
    theta_df = track_theta("/mnt/datastore/Harry/MouseVR/data/Cue_conditioned_cohort1_190902", theta_df)
    '''
    theta_df_VR = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta/theta_df_VR.pkl")
    #theta_df.to_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta/theta_df_VR.pkl")

    data = add_ramp_scores(theta_df_VR, ramp_lm, ramp_scores, tetrode_locations)
    data = remove_mouse(data, cohort_mouse_list="C2_1124")
    plot_theta_histogram(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    plot_lm_proportions(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", best_theta=False)
    plot_lm_proportions(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", best_theta=True)
    plot_theta(theta_df_VR, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    plot_max_ramp_score(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    plot_max_ramp_vs_max_theta(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    plot_max_rs_at_max_theta(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    plot_theta_at_max_rs(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")
    boxplot_theta(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", theta_threshold=0.07, best_theta=False)
    boxplot_theta(data, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta", theta_threshold=0.07, best_theta=True)

    '''
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs"
    correlation_save_path = save_path+"/rampscores_correlations/theta"
    for trial_type in ["beaconed", "non-beaconed", "probe", "all"]:
        for collumn_a in ["ramp_score", "abs_ramp_score"]:
            for ramp_region in ["outbound", "homebound", "all"]:
                for collumn_b in ["ThetaIndex", 'best_theta_idx_vr']:
                    for label_collumn in ["cohort_mouse"]:
                        ramp_score_correleation(data, correlation_save_path, collumn_a=collumn_a, collumn_b=collumn_b, ramp_region=ramp_region, label_collumn=label_collumn, trial_type=trial_type, of_n_spike_thres=None, by_mouse=False)
    print("Finished Correlation plots")
    '''
    
    # OF + VR cells
    theta_df = pd.DataFrame()
    theta_df = track_theta("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/vr_list.txt", theta_df)
    theta_df = track_theta("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/vrlist.txt", theta_df)
    theta_df = track_theta("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/vrlist_cohort3.txt", theta_df)
    theta_df = track_theta("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/with_of_recordings.txt", theta_df)
    theta_df.to_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta/theta_df.pkl")

    theta_df = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta/theta_df.pkl")
    plot_theta(theta_df, save_path="/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/theta")


    print("look now")

if __name__ == '__main__':
    main()