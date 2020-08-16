import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import itertools
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

def add_curation_flag(data):
    curated_together = []
    curated_seperately = []
    full_of_session_id = "" # used to speed up
    for i, row in data.iterrows():
        # read from spatial firing
        full_of_session_id_row = row["full_of_session_id"]
        cluster_id = row["sorted_together_vr_cluster_ids"]
        if full_of_session_id_row != full_of_session_id:
            full_of_session_id = full_of_session_id_row
            spatial_firing = pd.read_pickle(full_of_session_id+"/MountainSort/DataFrames/spatial_firing.pkl")

        spatial_firing_cluster = spatial_firing[spatial_firing.cluster_id == cluster_id]
        curated_seperately.append(True) # all vr seperate are curated
        if "Curated" in list(spatial_firing_cluster):
            curated_together.append(spatial_firing_cluster.Curated.iloc[0])
        else:
            curated_together.append(True)

    data["curated_seperately"] = curated_seperately
    data["curated_together"] = curated_together
    return data

def get_tetrode_location(tetrode_locations, session_id):
    tetrode_location = tetrode_locations[tetrode_locations["vr_Session_id"] == session_id]
    if len(tetrode_location)>0:
        return tetrode_location["tetrode_depth"].iloc[0]
    else:
        return np.nan

def get_cohort_mouse(row):
    full_session_id = row["full_session_id"].iloc[0]
    session_id = full_session_id.split("/")[-1]
    mouse = session_id.split("_D")[0]
    cohort = get_tidy_title(full_session_id.split("/")[-4])
    return cohort+"_"+mouse

def concatenate_all(ramp_path, ramp_scores_path, tetrode_location_path, all_of_paths, include_unmatch=True, ignore_of=False):
    data = pd.read_csv(ramp_path, sep = "\t")
    ramp_scores = pd.read_csv(ramp_scores_path)
    tetrode_locations = pd.read_csv(tetrode_location_path)

    new=pd.DataFrame()
    matches=0

    if ignore_of:
        for index_j, row_ramp_score in ramp_scores.iterrows():
            row_ramp_score =  row_ramp_score.to_frame().T.reset_index(drop=True)
            session_id = row_ramp_score["session_id"].iloc[0]
            cluster_id = row_ramp_score["cluster_id"].iloc[0]

            paired_cluster = data[((data["session_id"] == session_id) & (data["cluster_id"] == cluster_id))]

            if len(paired_cluster)==1:
                paired_cluster["trial_type"] = [row_ramp_score['trial_type'].iloc[0]]
                paired_cluster['ramp_score_out'] = [row_ramp_score['ramp_score_out'].iloc[0]]
                paired_cluster['ramp_score_home'] = [row_ramp_score['ramp_score_home'].iloc[0]]
                paired_cluster['ramp_score'] = [row_ramp_score['ramp_score'].iloc[0]]
                paired_cluster['fr_range'] = [row_ramp_score['fr_range'].iloc[0]]
                paired_cluster['meanVar'] = [row_ramp_score['meanVar'].iloc[0]]
                paired_cluster["tetrode_location"] = [get_tetrode_location(tetrode_locations, session_id)]

            new = pd.concat([new, paired_cluster], ignore_index=True)
        new = correct_datatypes(new, ignore_of=ignore_of)
        new = analyse_ramp_driver(new)
        new = get_best_ramp_score(new)
        return new

    # now add open field statistics before plotting
    for i in range(len(all_of_paths)):
        of_frame = pd.read_pickle(all_of_paths[i])
        matches += len(of_frame)

        for index, row in of_frame.iterrows():
            row =  row.to_frame().T.reset_index(drop=True)
            session_id = row["session_id"].iloc[0]
            cluster_id = row["sorted_seperately_vr_cluster_ids"].iloc[0]

            paired_cluster = data[((data["session_id"] == session_id) & (data["cluster_id"] == cluster_id))]
            paried_cluster_ramp_scores = ramp_scores[((ramp_scores["session_id"] == session_id) & (ramp_scores["cluster_id"] == cluster_id))]

            if len(paired_cluster) != 1:
                continue
            else:
                row["lmer_result_outbound"] = [paired_cluster["lmer_result_outbound"].iloc[0]]
                row["lmer_result_homebound"] = [paired_cluster["lmer_result_homebound"].iloc[0]]
                row['lm_result_b_outbound'] = [paired_cluster['lm_result_b_outbound'].iloc[0]]
                row['lm_result_b_homebound'] = [paired_cluster['lm_result_b_homebound'].iloc[0]]
                row['lm_result_nb_outbound'] = [paired_cluster['lm_result_nb_outbound'].iloc[0]]
                row['lm_result_nb_homebound'] = [paired_cluster['lm_result_nb_homebound'].iloc[0]]
                row['lm_result_p_outbound'] = [paired_cluster['lm_result_p_outbound'].iloc[0]]
                row['lm_result_P_homebound'] = [paired_cluster['lm_result_p_homebound'].iloc[0]]
                row["tetrode_location"] = [get_tetrode_location(tetrode_locations, session_id)]
                row["cohort_mouse"] = [get_cohort_mouse(row)]

                for index_j, row_ramp_score in paried_cluster_ramp_scores.iterrows():
                    row_ramp_score =  row_ramp_score.to_frame().T.reset_index(drop=True)
                    trial_type_row = row
                    trial_type_row["trial_type"] = [row_ramp_score['trial_type'].iloc[0]]
                    trial_type_row['ramp_score_out'] = [row_ramp_score['ramp_score_out'].iloc[0]]
                    trial_type_row['ramp_score_home'] = [row_ramp_score['ramp_score_home'].iloc[0]]
                    trial_type_row['ramp_score'] = [row_ramp_score['ramp_score'].iloc[0]]
                    trial_type_row['fr_range'] = [row_ramp_score['fr_range'].iloc[0]]
                    trial_type_row['meanVar'] = [row_ramp_score['meanVar'].iloc[0]]

                    new = pd.concat([new, trial_type_row], ignore_index=True)

    new = correct_datatypes(new)
    new = correct_speed_score(new)
    new = analyse_ramp_driver(new)
    new = get_best_ramp_score(new)
    new = get_matching_spike_ratio(new)
    new = get_best_theta_scores(new)
    return new

def get_best_theta_scores_VR(data):
    best_theta_idx = []
    best_theta_pwr = []

    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        session_id = row["session_id"].iloc[0]

        same_session = data[(data["session_id"] == session_id)]
        best_theta_idx.append(max(same_session['ThetaIndex']))
        best_theta_pwr.append(max(same_session['ThetaPower']))

    data["best_theta_idx_vr"] = best_theta_idx
    data["best_theta_pwr_vr"] = best_theta_pwr
    return data

def get_best_theta_scores(data):
    best_theta_idx_vr = []
    best_theta_idx_of = []
    best_theta_idx_combined = []
    best_theta_pwr_vr = []
    best_theta_pwr_of = []
    best_theta_pwr_combined = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        session_id = row["session_id"].iloc[0]

        same_session = data[(data["session_id"] == session_id)]
        best_theta_idx_vr.append(max(same_session['ThetaIndex_vr']))
        best_theta_idx_of.append(max(same_session['ThetaIndex']))
        best_theta_idx_combined.append(max([max(same_session['ThetaIndex_vr']) , max(same_session['ThetaIndex'])]))

        best_theta_pwr_vr.append(max(same_session['ThetaPower_vr']))
        best_theta_pwr_of.append(max(same_session['ThetaPower']))
        best_theta_pwr_combined.append(max([max(same_session['ThetaPower_vr']) , max(same_session['ThetaPower'])]))

    data["best_theta_idx_vr"] = best_theta_idx_vr
    data["best_theta_idx_of"] = best_theta_idx_of
    data["best_theta_idx_combined"] = best_theta_idx_combined
    data["best_theta_pwr_vr"] = best_theta_pwr_vr
    data["best_theta_pwr_of"] = best_theta_pwr_of
    data["best_theta_pwr_combined"] = best_theta_pwr_combined
    return data

def get_matching_spike_ratio(data):
    spike_ratios = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        n_spikes_vr = row["n_spikes_vr"].iloc[0]
        n_spikes_vr_original = row["n_spikes_vr_original"].iloc[0]
        spike_ratio = n_spikes_vr_original/n_spikes_vr
        spike_ratios.append(spike_ratio)
    data["spike_ratio"] = spike_ratios
    return data

def get_best_ramp_score(data):
    best_ramp_half = []
    best_ramp_score_labels = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        ramp_score_outbound = row["ramp_score_out"].iloc[0]
        ramp_score_homebound = row["ramp_score_home"].iloc[0]
        ramp_score_full_track = row["ramp_score"].iloc[0]

        best_ramp_score = ramp_score_outbound
        best_ramp_score_label = "outbound"
        if ramp_score_homebound>best_ramp_score:
            best_ramp_score=ramp_score_homebound
            best_ramp_score_label = "homebound"
        if ramp_score_full_track>best_ramp_score:
            best_ramp_score=ramp_score_full_track
            best_ramp_score_label = "full_track"

        best_ramp_score_labels.append(best_ramp_score_label)
        best_ramp_half.append(best_ramp_score)

    data["max_ramp_score"] = best_ramp_half
    data["max_ramp_score_label"] = best_ramp_score_labels
    return data

def analyse_ramp_driver(data):
    # look at lm results and label ramp driver as PI, Cue or None
    pi_count=0
    cue_count=0
    ramp_driver=[]
    for index, row in data.iterrows():
        label= "None"
        row = row.to_frame().T.reset_index(drop=True)

        lm_result_b_outbound = row["lm_result_b_outbound"].iloc[0]
        lm_result_nb_outbound = row["lm_result_nb_outbound"].iloc[0]

        if (lm_result_b_outbound == "Negative") or (lm_result_b_outbound == "Positive"): # significant on beaconed
            if (lm_result_nb_outbound == "Negative") or (lm_result_nb_outbound == "Positive"): # significant on non_beaconed
                label="PI"
                pi_count+=1
            elif (lm_result_nb_outbound == "None") : # not significant on non beaconed
                label="Cue"
                cue_count+=1

        ramp_driver.append(label)

    print(pi_count, " = pi_count")
    print(cue_count, " = cue_count")
    data["ramp_driver"] = ramp_driver
    return data

def absolute_ramp_score(data):
    absolute_ramp_scores = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        ramp_score = row["ramp_score"].iloc[0]
        absolute_ramp_scores.append(np.abs(ramp_score))
    data["abs_ramp_score"] = absolute_ramp_scores
    return data

def correct_speed_score(df):
    df['speed_score'] = df['speed_score'].abs()
    return df

def correct_datatypes(dataframe, ignore_of=False):

    if ignore_of:
        dataframe["ramp_score"] = pd.to_numeric(dataframe["ramp_score"])
        dataframe["fr_range"] = pd.to_numeric(dataframe["fr_range"])
        dataframe["meanVar"] = pd.to_numeric(dataframe["meanVar"])
        return dataframe

    # I'm first fixing the dtype of the prefered HD, this is a size-1 np.ndarray atm
    for i in range(len(dataframe)):
        if type(dataframe["preferred_HD"].iloc[i]) is np.ndarray:
            if len(dataframe["preferred_HD"].iloc[i]) > 1:
                dataframe["preferred_HD"].iloc[i] = np.nan
            else:
                dataframe["preferred_HD"].iloc[i] = dataframe["preferred_HD"].iloc[i][0]

    dataframe["rate_map_correlation_first_vs_second_half"] = pd.to_numeric(dataframe["rate_map_correlation_first_vs_second_half"])
    dataframe["mean_firing_rate"] = pd.to_numeric(dataframe["mean_firing_rate"])
    dataframe["isolation"] = pd.to_numeric(dataframe["isolation"])
    dataframe["noise_overlap"] = pd.to_numeric(dataframe["noise_overlap"])
    dataframe["peak_snr"] = pd.to_numeric(dataframe["peak_snr"])
    dataframe["peak_amp"] = pd.to_numeric(dataframe["peak_amp"])
    dataframe["speed_score"] = pd.to_numeric(dataframe["speed_score"])
    dataframe["speed_score_p_values"] = pd.to_numeric(dataframe["speed_score_p_values"])
    dataframe["max_firing_rate_hd"] = pd.to_numeric(dataframe["max_firing_rate_hd"])
    dataframe["preferred_HD"] = pd.to_numeric(dataframe["preferred_HD"])
    dataframe["rayleigh_score"] = pd.to_numeric(dataframe["rayleigh_score"])
    dataframe["grid_spacing"] = pd.to_numeric(dataframe["grid_spacing"])
    dataframe["hd_score"] = pd.to_numeric(dataframe["hd_score"])
    dataframe["field_size"] = pd.to_numeric(dataframe["field_size"])
    dataframe["grid_score"] = pd.to_numeric(dataframe["grid_score"])
    dataframe["border_score"] = pd.to_numeric(dataframe["border_score"])
    dataframe["corner_score"] = pd.to_numeric(dataframe["corner_score"])
    dataframe["percent_excluded_bins_rate_map_correlation_first_vs_second_half_p"] = pd.to_numeric(dataframe["percent_excluded_bins_rate_map_correlation_first_vs_second_half_p"])
    dataframe["hd_correlation_first_vs_second_half"] = pd.to_numeric(dataframe["hd_correlation_first_vs_second_half"])
    dataframe["hd_correlation_first_vs_second_half_p"] = pd.to_numeric(dataframe["hd_correlation_first_vs_second_half_p"])
    dataframe["split_cluster"] = pd.to_numeric(dataframe["split_cluster"])
    dataframe["n_spikes_vr"] = pd.to_numeric(dataframe["n_spikes_vr"])
    dataframe["n_spikes_of"] = pd.to_numeric(dataframe["n_spikes_of"])
    dataframe["ramp_score_out"] = pd.to_numeric(dataframe["ramp_score_out"])
    dataframe["ramp_score_home"] = pd.to_numeric(dataframe["ramp_score_home"])
    dataframe["ramp_score"] = pd.to_numeric(dataframe["ramp_score"])
    dataframe["fr_range"] = pd.to_numeric(dataframe["fr_range"])
    dataframe["meanVar"] = pd.to_numeric(dataframe["meanVar"])
    dataframe["ThetaIndex_vr"] = pd.to_numeric(dataframe["ThetaIndex_vr"])
    dataframe["ThetaIndex"] = pd.to_numeric(dataframe["ThetaIndex"])

    return dataframe

def plot_histogram(data, save_path, collumn_a=None, collumn_b=None,  trial_type=None, of_n_spike_thres=1000, cumulative=False):
    data = data[(data["trial_type"]==trial_type)]

    # only use for lm results not lmer
    data1 = data.dropna(subset=[collumn_b])
    # remove clusters that have very few spikes in of to calculate spatial scores on
    data2 = data1[data1["n_spikes_of"]>=of_n_spike_thres]

    fig, ax = plt.subplots(figsize=(9,6))
    threshold = get_score_threshold(collumn_b)
    ax.set_title(get_tidy_title(collumn_a), fontsize=15)

    non_ramps = data2[(data2[collumn_a] == "None")]
    ramps =     data2[((data2[collumn_a] == "Negative") | (data2[collumn_a] == "Positive"))]
    p = stats.ks_2samp(np.asarray(non_ramps[collumn_b]), np.asarray(ramps[collumn_b]))[1]
    p_text = get_p_text(p)

    n_ramps = len(data[((data[collumn_a] == "Negative") | (data[collumn_a] == "Positive"))])
    n_non_ramps = len(data[(data[collumn_a] == "None")])
    n_ramp_over_thres = np.sum(ramps[collumn_b]>threshold)
    n_non_ramps_over_thres = np.sum(non_ramps[collumn_b]>threshold)
    percentage_ramps_over_thres = np.round((n_ramp_over_thres/n_ramps)*100, decimals=1)
    percentage_non_ramps_over_thres = np.round((n_non_ramps_over_thres/n_non_ramps)*100, decimals=1)

    ax.text(1.6, 0.6, 'OF Matches = '+str(len(data))+'\n'+
            'Ramps = '+str(n_ramps)+'\n' +
            'Non Ramps = '+str(n_non_ramps)+'\n' +
            'Ramps > thres = '+str(n_ramp_over_thres)+', '+str(percentage_ramps_over_thres)+r'$\%$'+'\n'+
            'Non Ramps > thres = '+str(n_non_ramps_over_thres)+', '+str(percentage_non_ramps_over_thres)+r'$\%$'+'\n' +
            'KS test, p = '+ str(np.round(p, decimals=2)) +" "+str(p_text)+'\n',
        ha='right', va='top',
        transform=ax.transAxes)

    if cumulative:
        _, _, patches1= ax.hist(np.asarray(non_ramps[collumn_b]), bins=50, label="Non Ramp", alpha=0.5, color="k", cumulative=cumulative, density=True, histtype='step')
        _, _, patches2= ax.hist(np.asarray(ramps[collumn_b]), bins=50, label="Ramp", alpha=0.5, color="b", cumulative=cumulative, density=True, histtype='step')
        patches1[0].set_xy(patches1[0].get_xy()[:-1])
        patches2[0].set_xy(patches2[0].get_xy()[:-1])
        ax.set_ylabel("Cumulative Density", fontsize=15)
    else:
        _, _, patches1= ax.hist(np.asarray(non_ramps[collumn_b]), bins=50, label="Non Ramp", alpha=0.5, color="k", cumulative=cumulative)
        _, _, patches2= ax.hist(np.asarray(ramps[collumn_b]), bins=50, label="Ramp", alpha=0.5, color="b", cumulative=cumulative)
        ax.set_ylabel("Frequency", fontsize=15)

    ax.set_xlabel(get_tidy_title(collumn_b), fontsize=15)
    ax.legend(bbox_to_anchor=(0.65,0.8), \
              bbox_transform=plt.gcf().transFigure,
              fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if threshold is not None:
        ax.axvline(threshold, color="k")
    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.6, top=0.8, bottom=0.2)
    if cumulative:
        plt.savefig(save_path+"/cumhistogram_"+collumn_a+"_"+collumn_b+".png")
    else:
        plt.savefig(save_path+"/histogram_"+collumn_a+"_"+collumn_b+".png")
    plt.show()


    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_title(get_tidy_title(collumn_a), fontsize=15)
    non_ramp = data2[(data2[collumn_a] == "None")]
    neg_ramp =  data2[(data2[collumn_a] == "Negative")]
    pos_ramp =  data2[(data2[collumn_a] == "Positive")]
    p = stats.ks_2samp(np.asarray(neg_ramp[collumn_b]), np.asarray(pos_ramp[collumn_b]))[1]
    p_text = get_p_text(p)

    n_neg_ramps = len(data[((data[collumn_a] == "Negative"))])
    n_pos_ramps = len(data[((data[collumn_a] == "Positive"))])
    n_non_ramps = len(data[(data[collumn_a] == "None")])
    n_neg_ramp_over_thres = np.sum(neg_ramp[collumn_b]>threshold)
    n_pos_ramp_over_thres = np.sum(pos_ramp[collumn_b]>threshold)
    n_non_ramps_over_thres = np.sum(non_ramp[collumn_b]>threshold)
    percentage_neg_ramps_over_thres = np.round((n_neg_ramp_over_thres/n_neg_ramps)*100, decimals=1)
    percentage_pos_ramps_over_thres = np.round((n_pos_ramp_over_thres/n_pos_ramps)*100, decimals=1)
    percentage_non_ramps_over_thres = np.round((n_non_ramps_over_thres/n_non_ramps)*100, decimals=1)

    if cumulative:
        _, _, patches1 = ax.hist(np.asarray(non_ramp[collumn_b]), bins=50, label="Non Ramp", alpha=0.5, color="k", cumulative=cumulative, density=True, histtype='step')
        _, _, patches2 = ax.hist(np.asarray(neg_ramp[collumn_b]), bins=50, label="Negative Ramp", alpha=0.5, color="r", cumulative=cumulative, density=True, histtype='step')
        _, _, patches3 = ax.hist(np.asarray(pos_ramp[collumn_b]), bins=50, label="Positive Ramp", alpha=0.5, color="b", cumulative=cumulative, density=True, histtype='step')
        patches1[0].set_xy(patches1[0].get_xy()[:-1])
        patches2[0].set_xy(patches2[0].get_xy()[:-1])
        patches3[0].set_xy(patches3[0].get_xy()[:-1])
        ax.set_ylabel("Cumulative Density", fontsize=15)
    else:
        _, _, patches1 = ax.hist(np.asarray(non_ramp[collumn_b]), bins=50, label="Non Ramp", alpha=0.5, color="k", cumulative=cumulative)
        _, _, patches2 = ax.hist(np.asarray(neg_ramp[collumn_b]), bins=50, label="Negative Ramp", alpha=0.5, color="r", cumulative=cumulative)
        _, _, patches3 = ax.hist(np.asarray(pos_ramp[collumn_b]), bins=50, label="Positive Ramp", alpha=0.5, color="b", cumulative=cumulative)
        ax.set_ylabel("Frequency", fontsize=15)

    ax.legend(bbox_to_anchor=(0.65,0.8), \
               bbox_transform=plt.gcf().transFigure,
              fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if threshold is not None:
        ax.axvline(threshold, color="k")
    ax.set_xlabel(get_tidy_title(collumn_b), fontsize=15)

    ax.text(1.6, 0.6, 'OF Matches = '+str(len(data))+'\n'+
            '+ve Ramps = '+str(n_pos_ramps)+'\n' +
            '-ve Ramps = '+str(n_neg_ramps)+'\n' +
            'Non Ramps = '+str(n_non_ramps)+'\n' +
            '+ve Ramps > thres = '+str(n_pos_ramp_over_thres)+', '+str(percentage_pos_ramps_over_thres)+r'$\%$'+'\n'+
            '-ve Ramps > thres = '+str(n_neg_ramp_over_thres)+', '+str(percentage_neg_ramps_over_thres)+r'$\%$'+'\n'+
            'Non Ramps > thres = '+str(n_non_ramps_over_thres)+', '+str(percentage_non_ramps_over_thres)+r'$\%$'+'\n'+
            'KS test, p = '+ str(np.round(p, decimals=2)) +" "+str(p_text)+'\n',
            ha='right', va='top',
            transform=ax.transAxes)

    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.6, top=0.8, bottom=0.2)

    if cumulative:
        plt.savefig(save_path+"/cumhistogram_neg_pos_"+collumn_a+"_"+collumn_b+".png")
    else:
        plt.savefig(save_path+"/histogram_neg_pos_"+collumn_a+"_"+collumn_b+".png")
    plt.show()


    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_title(get_tidy_title(collumn_a), fontsize=15)
    none_ramp = data2[(data2["ramp_driver"] == "None")]
    pi_ramp =  data2[(data2["ramp_driver"] == "PI")]
    cue_ramp =  data2[(data2["ramp_driver"] == "Cue")]
    p = stats.ks_2samp(np.asarray(pi_ramp[collumn_b]), np.asarray(cue_ramp[collumn_b]))[1]
    p_text = get_p_text(p)

    n_none_ramp = len(data[((data["ramp_driver"] == "None"))])
    n_pi_ramp = len(data[((data["ramp_driver"] == "PI"))])
    n_cue_ramp = len(data[(data["ramp_driver"] == "Cue")])
    n_none_ramp_over_thres = np.sum(none_ramp[collumn_b]>threshold)
    n_pi_ramp_over_thres = np.sum(pi_ramp[collumn_b]>threshold)
    n_cue_ramp_over_thres = np.sum(cue_ramp[collumn_b]>threshold)
    percentage_none_ramps_over_thres = np.round((n_none_ramp_over_thres/n_none_ramp)*100, decimals=1)
    percentage_pi_ramps_over_thres = np.round((n_pi_ramp_over_thres/n_pi_ramp)*100, decimals=1)
    percentage_cue_ramps_over_thres = np.round((n_cue_ramp_over_thres/n_cue_ramp)*100, decimals=1)

    if cumulative:
        _, _, patches1 = ax.hist(np.asarray(none_ramp[collumn_b]), bins=50, label="None", alpha=0.5, color="k", cumulative=cumulative, density=True, histtype='step')
        _, _, patches2 = ax.hist(np.asarray(pi_ramp[collumn_b]), bins=50, label="PI Ramp", alpha=0.5, color="y", cumulative=cumulative, density=True, histtype='step')
        _, _, patches3 = ax.hist(np.asarray(cue_ramp[collumn_b]), bins=50, label="Cue Ramp", alpha=0.5, color="g", cumulative=cumulative, density=True, histtype='step')
        patches1[0].set_xy(patches1[0].get_xy()[:-1])
        patches2[0].set_xy(patches2[0].get_xy()[:-1])
        patches3[0].set_xy(patches3[0].get_xy()[:-1])
        ax.set_ylabel("Cumulative Density", fontsize=15)
    else:
        _, _, patches1 = ax.hist(np.asarray(none_ramp[collumn_b]), bins=50, label="None", alpha=0.5, color="k", cumulative=cumulative)
        _, _, patches2 = ax.hist(np.asarray(pi_ramp[collumn_b]), bins=50, label="PI Ramp", alpha=0.5, color="y", cumulative=cumulative)
        _, _, patches3 = ax.hist(np.asarray(cue_ramp[collumn_b]), bins=50, label="Cue Ramp", alpha=0.5, color="g", cumulative=cumulative)
        ax.set_ylabel("Frequency", fontsize=15)

    ax.legend(bbox_to_anchor=(0.65,0.8), \
              bbox_transform=plt.gcf().transFigure,
              fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if threshold is not None:
        ax.axvline(threshold, color="k")
    ax.set_xlabel(get_tidy_title(collumn_b), fontsize=15)

    ax.text(1.6, 0.6, 'OF Matches = '+str(len(data))+'\n'+
            'PI Ramps = '+str(n_pi_ramp)+'\n' +
            'Cue Ramps = '+str(n_cue_ramp)+'\n' +
            'None = '+str(n_none_ramp)+'\n' +
            'PI Ramps > thres = '+str(n_pi_ramp_over_thres)+', '+str(percentage_pi_ramps_over_thres)+r'$\%$'+'\n'+
            'Cue Ramps > thres = '+str(n_cue_ramp_over_thres)+', '+str(percentage_cue_ramps_over_thres)+r'$\%$'+'\n'+
            'None > thres = '+str(n_none_ramp_over_thres)+', '+str(percentage_none_ramps_over_thres)+r'$\%$'+'\n'+
            'KS test, p = '+ str(np.round(p, decimals=2)) +" "+str(p_text)+'\n',
            ha='right', va='top',
            transform=ax.transAxes)

    #plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.6, top=0.8, bottom=0.2)

    if cumulative:
        plt.savefig(save_path+"/cumhistogram_rampDriver_"+collumn_a+"_"+collumn_b+".png")
    else:
        plt.savefig(save_path+"/histogram_rampDriver_"+collumn_a+"_"+collumn_b+".png")
    plt.show()

    print("finished plotting histogram")


    return

def plot_ramp_correlations(data, save_path, collumn_a=None, collumn_b=None, of_n_spike_thres=1000):
    data = data.dropna(subset=[collumn_b])
    # remove clusters that have very few spikes in of to calculate spatial scores on
    data = data[data["n_spikes_of"]>=of_n_spike_thres]
    subset_mean = data.groupby([collumn_a])[collumn_b].mean().reset_index()
    subset_sem = data.groupby([collumn_a])[collumn_b].sem().reset_index()

    fig, ax = plt.subplots()
    x_pos = np.arange(1,len(subset_mean[collumn_a])+1,1)
    means = subset_mean[collumn_b]
    sems = subset_sem[collumn_b]

    if collumn_a == "lm_result_outbound":
        colors = ["red", "grey", "blue"]
    elif collumn_a == "lmer_result_outbound":
        colors =["lightskyblue", "grey", "deeppink", "orchid", "indianred", "forestgreen", "salmon", "mediumaquamarine"]

    for x, mean, sem, color in zip(x_pos, means, sems, colors):
        ax.errorbar(x, mean, sem, lw=2, capsize=20, capthick=2, color=color)

    counter = 1
    for condition in np.unique(data[collumn_a]):
        condition_data = data[data[collumn_a]==condition]

        for i in range(len(condition_data)):
            ax.scatter(counter, condition_data[collumn_b].iloc[i], marker="x", color=colors[counter-1])
        counter+=1

    c = list(itertools.combinations(np.unique(data[collumn_a]), 2))
    counter=0.1
    for combo in c:
        tip = max(data[collumn_b]) + 1
        group_1 = data[data[collumn_a]==combo[0]][collumn_b]
        group_2 = data[data[collumn_a]==combo[1]][collumn_b]
        t, p = stats.ttest_ind(group_1, group_2, axis=0, equal_var=True, nan_policy='propagate')

        p_text = get_p_text(p)
        y = tip + counter
        x1 = x_pos[np.unique(data[collumn_a])==combo[0]][0]
        x2 = x_pos[np.unique(data[collumn_a])==combo[1]][0]

        cc1 = (x1, y)
        cc2 = (x2, y)
        #plt.plot([cc1[0], cc2[0]], [cc1[1], cc2[1]], lw=1.5, color="k")
        #plt.text((cc1[0]+cc2[0])*.5, cc1[1]+0.05, p_text, ha='center', va='bottom', color="k")
        cc1 = (x1, y+0.1)
        cc2 = (x2, y+0.1)
        counter+=0.2

    ax.set_xlabel(collumn_a, fontsize=7)
    ax.set_ylabel(collumn_b, fontsize=10)
    #ax.set_title(collumn_b, fontsize=25)
    ax.set_xticks(x_pos)
    ax.set_xlim([0,max(x_pos)+1])
    #ax.set_ylim([0.8, 1])
    ax.set_xticklabels(np.unique(data[collumn_a]))
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path+"/Ramp_correlations_"+collumn_a+"_"+collumn_b+".png")
    plt.show()

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

def plot_split_stats(data, save_path, collumn_a):

    fig, ax = plt.subplots(figsize=(8,5))
    if collumn_a == "lm_result":
        colors_a = ["red", "grey", "blue", "silver"]
        #colors_b = ["firebrick", "black", "midnightblue"]
    elif collumn_a == "model_result_outbound":
        colors_a =["lightskyblue", "grey", "deeppink", "orchid", "indianred", "forestgreen", "salmon", "mediumaquamarine", "silver"]
        #colors_b =["steelblue",   "black", "mediumvioletred", "darkmagenta",    "brown",   "darkgreen", "tomato", "mediumseagreen"]
    subset_mean = data.groupby([collumn_a])["split_cluster"].mean().reset_index()
    x_pos = np.arange(1,len(subset_mean[collumn_a])+1,1)
    means_splits = subset_mean["split_cluster"]
    means_no_splits = 1-subset_mean["split_cluster"]

    counter=0
    for x, mean, color in zip(x_pos, means_no_splits, colors_a):
        ax.bar(x, mean, bottom=means_splits[counter], color=color, edgecolor=color)
        counter+=1
    for x, mean, color in zip(x_pos, means_splits, colors_a):
        ax.bar(x, mean, color="white", edgecolor=color)

    ax.set_xlabel(collumn_a, fontsize=20)
    ax.set_ylabel("proportion", fontsize=20)
    ax.set_xticks(x_pos)
    ax.set_xlim([0,max(x_pos)+1])
    #ax.set_ylim([0.8, 1])
    ax.set_xticklabels(np.unique(data[collumn_a]))
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    legend_elements = [Patch(facecolor='white', edgecolor='k', label=' Split Cluster'),
                       Patch(facecolor='black', edgecolor='k', label=' Complete Cluster')]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(1.2, 0.5))

    plt.tight_layout()
    plt.savefig(save_path+"/Ramp_split_clusters_"+collumn_a+"_"+".png")
    plt.show()

def plot_regression(ax, x, y):
    # x  and y are pandas collumn
    x = x.values
    y = y.values
    x = x[~np.isnan(y)].reshape(-1, 1)
    y = y[~np.isnan(y)].reshape(-1, 1)

    pearson_r = stats.pearsonr(x.flatten(),y.flatten())

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x,y)  # perform linear regression

    x_test = np.linspace(min(x), max(x), 100)

    Y_pred = linear_regressor.predict(x_test.reshape(-1, 1))  # make predictions

    ax.text(  # position text relative to Axes
        0.95, 1.25, "R= "+str(np.round(pearson_r[0], decimals=2))+ ", p = "+str(np.round(pearson_r[1], decimals=2)) +str(get_p_text(pearson_r[1])),
        ha='right', va='top',
        transform=ax.transAxes, fontsize=20)

    ax.plot(x_test, Y_pred)

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

def simple_histogram(data, collumn, title="", save_path=None):
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_title(title, fontsize=15)

    curated = data[data.curated_together == True]
    non_curated = data[data.curated_together == False]

    #_, _, patches1= ax.hist(np.asarray(curated[collumn]), bins=50, alpha=0.5, color="k")

    ax.hist(np.asarray(curated[collumn]), bins=50, alpha=0.5, color="k", label="Passed Curated")
    ax.hist(np.asarray(non_curated[collumn]), bins=50, alpha=0.5, color="r", label="Failed Curated")

    ax.set_ylabel("Count", fontsize=15)
    ax.set_xlabel(get_tidy_title(collumn), fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.2, right=0.6, top=0.8, bottom=0.2)
    ax.legend()
    ax.set_xlim(left=0)
    if save_path is not None:
        plt.savefig(save_path+"/histo_"+collumn+"_.png")

    plt.show()
    plt.close()
    return

def ramp_score_correleation(data, save_path, collumn_a, collumn_b, ramp_region, label_collumn, trial_type, of_n_spike_thres=1000, by_mouse=False):
    data = data[(data["trial_type"]==trial_type)]
    data = data[(data["ramp_region"]==ramp_region)]
    data1 = data.dropna(subset=[collumn_b])

    # remove clusters that have very few spikes in of to calculate spatial scores on
    if of_n_spike_thres is not None:
        data2 = data1[data1["n_spikes_of"]>=of_n_spike_thres]
    else:
        data2 = data1

    if by_mouse:
        for mouse in np.unique(data2["cohort_mouse"]):
            mouse_data = data2[(data2["cohort_mouse"] == mouse)]

            color = label_collumn2color(mouse_data, label_collumn)
            fig, ax = plt.subplots()
            ax.axvline(x=0.07, ymin=0, ymax=1, color="k")
            ax.set_ylim([0.4,1])

            ax.set_title("label="+label_collumn+", tt="+get_tidy_title(trial_type), fontsize=15)
            ax.scatter(mouse_data[collumn_b], mouse_data[collumn_a], edgecolor=color, marker="o", facecolors='none')
            plot_regression(ax, mouse_data[collumn_b], mouse_data[collumn_a])
            #plt.xscale('log')
            #plt.xlim((0.000001, 1))
            plt.ylabel(get_tidy_title(collumn_a), fontsize=20, labelpad=10)
            plt.xlabel(get_tidy_title(collumn_b), fontsize=20, labelpad=10)
            #plt.xlim(0, 2)
            #plt.ylim(-150, 150)
            plt.tick_params(labelsize=20)
            plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
            plt.savefig(save_path+"/"+mouse+"_rs_correlation_"+collumn_a+"_"+collumn_b+"_"+ramp_region+"_"+label_collumn+"_"+trial_type+".png")
            print("plotted ramp score correlation")
    else:

        color = label_collumn2color(data2, label_collumn)
        fig, ax = plt.subplots()
        ax.axvline(x=0.07, ymin=0, ymax=1, color="k")
        #ax.set_ylim([0.4,1])

        ax.set_title("rr="+ramp_region+", tt="+get_tidy_title(trial_type), fontsize=15)
        ax.scatter(data2[collumn_b], data2[collumn_a], edgecolor=color, marker="o", facecolors='none')
        #plot_regression(ax, data2[collumn_b], data2[collumn_a])
        #plt.xscale('log')
        #plt.xlim((0.000001, 1))
        plt.ylabel(get_tidy_title(collumn_a), fontsize=20, labelpad=10)
        plt.xlabel(get_tidy_title(collumn_b), fontsize=20, labelpad=10)
        #plt.xlim(0, 2)
        #plt.ylim(-150, 150)
        plt.tick_params(labelsize=20)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        plt.savefig(save_path+"/ramp_score_correlation_"+collumn_a+"_"+collumn_b+"_"+ramp_region+"_"+label_collumn+"_"+trial_type+".png")
        print("plotted ramp score correlation")


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')#

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    ramp_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/all_results_linearmodel.txt"
    ramp_scores_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/ramp_score_export.csv"
    tetrode_location_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/tetrode_locations.csv"
    save_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/figs/Ramp_figs"
    '''
    c5m1_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M1_sorting_stats.pkl"
    c5m2_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/M2_sorting_stats.pkl"
    c4m2_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M2_sorting_stats.pkl"
    c4m3_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/M3_sorting_stats.pkl"
    c3m1_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M1_sorting_stats.pkl"
    c3m6_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/M6_sorting_stats.pkl"
    c2m245_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/245_sorting_stats.pkl"
    c2m1124_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/1124_sorting_stats.pkl"

    all_of_paths = [c5m1_path, c5m2_path, c4m2_path, c4m3_path, c3m1_path, c3m6_path, c2m245_path, c2m1124_path]

    data = concatenate_all(ramp_path, ramp_scores_path, tetrode_location_path, all_of_paths, include_unmatch=False)
    #data.to_csv('/mnt/datastore/Harry/Mouse_data_for_sarah_paper/match.csv', index = False)
    '''
    data = pd.read_csv('/mnt/datastore/Harry/Mouse_data_for_sarah_paper/match.csv')

    simple_histogram(data[(data["trial_type"]=="beaconed")], collumn="spike_ratio", save_path=save_path)
    tmp = data[(data["trial_type"]=="beaconed")]
    print("@ > 0.5 spike ratio, n = ", len(tmp[(tmp["spike_ratio"])>0.5]), ", agreement is ", np.mean(tmp[(tmp["spike_ratio"])>0.5]["agreement_vr"]), "+- ", np.std(tmp[(tmp["spike_ratio"])>0.5]["agreement_vr"]))
    print("@ < 0.5 spike ratio, n = ", len(tmp[(tmp["spike_ratio"])<0.5]), ", agreement is ", np.mean(tmp[(tmp["spike_ratio"])<0.5]["agreement_vr"]), "+- ", np.std(tmp[(tmp["spike_ratio"])<0.5]["agreement_vr"]))
    data = data[(data["spike_ratio"]>0.5)]

    #correlation_save_path = save_path+"/rampscores_correlations"
    #for trial_type in ["beaconed", "non-beaconed", "probe"]:
    #    for collumn_a in ["ramp_score_out", "ramp_score_home", "ramp_score", "max_ramp_score"]:
    #        for collumn_b in ["hd_score", "speed_score", "grid_score", "border_score", "rayleigh_score", "rate_map_correlation_first_vs_second_half"]:
    #            for label_collumn in ["lm_result_b_homebound", "lm_result_b_outbound", "lmer_result_homebound", "lmer_result_outbound", "ramp_driver", "max_ramp_score_label"]:
    #                ramp_score_correleation(data, correlation_save_path, collumn_a=collumn_a, collumn_b=collumn_b, label_collumn=label_collumn, trial_type=trial_type)

    correlation_save_path = save_path+"/rampscores_correlations/theta"
    for trial_type in ["beaconed", "non-beaconed", "probe"]:
        for collumn_a in ["ramp_score_out", "ramp_score_home", "ramp_score", "max_ramp_score"]:
            for collumn_b in ["ThetaIndex_vr", "ThetaPower_vr", "ThetaIndex", "ThetaPower", 'best_theta_idx_vr', 'best_theta_idx_of', 'best_theta_idx_combined', 'best_theta_pwr_vr', 'best_theta_pwr_of', 'best_theta_pwr_combined']:
                for label_collumn in ["lm_result_b_homebound", "lm_result_b_outbound", "lmer_result_homebound", "lmer_result_outbound", "ramp_driver", "max_ramp_score_label", "cohort_mouse"]:
                    ramp_score_correleation(data, correlation_save_path, collumn_a=collumn_a, collumn_b=collumn_b, label_collumn=label_collumn, trial_type=trial_type)
    print("Finished Correlation plots")

    hist_save_path=save_path+"/hists"
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_outbound", collumn_b="hd_score", trial_type="beaconed", cumulative=True)
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_outbound", collumn_b="speed_score", trial_type="beaconed", cumulative=True)
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_outbound", collumn_b="grid_score", trial_type="beaconed", cumulative=True)
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_outbound", collumn_b="border_score", trial_type="beaconed", cumulative=True)
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_outbound", collumn_b="rate_map_correlation_first_vs_second_half", trial_type="beaconed", cumulative=True)
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_homebound", collumn_b="hd_score", trial_type="beaconed", cumulative=True)
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_homebound", collumn_b="speed_score", trial_type="beaconed", cumulative=True)
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_homebound", collumn_b="grid_score", trial_type="beaconed", cumulative=True)
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_homebound", collumn_b="border_score", trial_type="beaconed", cumulative=True)
    plot_histogram(data,hist_save_path, collumn_a="lm_result_b_homebound", collumn_b="rate_map_correlation_first_vs_second_half", trial_type="beaconed", cumulative=True)

    '''
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="mean_firing_rate")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="speed_score")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="max_firing_rate_hd")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="preferred_HD")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="hd_score")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="rayleigh_score")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="grid_spacing")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="field_size")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="grid_score")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="border_score")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="corner_score")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="rate_map_correlation_first_vs_second_half")
    plot_ramp_correlations(data, save_path, collumn_a="lmer_result_outbound", collumn_b="hd_correlation_first_vs_second_half")

    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="mean_firing_rate")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="speed_score")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="max_firing_rate_hd")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="preferred_HD")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="hd_score")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="rayleigh_score")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="grid_spacing")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="field_size")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="grid_score")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="border_score")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="corner_score")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="rate_map_correlation_first_vs_second_half")
    plot_ramp_correlations(data, save_path, collumn_a="lm_result_outbound", collumn_b="hd_correlation_first_vs_second_half")
    '''
    print("look now")

if __name__ == '__main__':
    main()