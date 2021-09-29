import pandas as pd
import numpy as np
import os
from Edmond.Concatenate_from_server import *

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

def get_mouse(session_id):
    return session_id.split("_")[0]

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

def get_mouse_id(full_session_id):
    session_id = full_session_id.split("/")[-1]
    mouse = get_mouse(session_id)
    return mouse

def add_mouse_label(data):
    mouse = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        mouse_str = get_mouse_id(row["full_session_id"].iloc[0])
        mouse.append(mouse_str)
    data["mouse"] = mouse
    return data

def add_recording_day(data):
    recording_days = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        recording_days.append(get_day(row["session_id"].iloc[0]))
    data["recording_day"] = recording_days
    return data

def add_full_session_id(data, all_recording_paths):
    full_session_ids = []
    for index, row in data.iterrows():
        row =  row.to_frame().T.reset_index(drop=True)
        session_id = row["session_id"].iloc[0]
        full_session_id = [s for s in all_recording_paths if session_id in s]
        full_session_ids.append(full_session_id[0])
    data["full_session_id"] = full_session_ids
    return data

def summarise_experiment(recordings_folder_path, suffix=None, save_path=None):
    '''
    :param recordings_folder_path: path to folder with all the recordings you want to summarise
    :param suffix: should be vr or of
    :param save_path:
    :param prm: parameters Class object
    :return: saves an all_days dataframe @save_path
    '''

    print("summarising")

    recording_paths = get_recording_paths([], recordings_folder_path)
    all_days_df = pd.DataFrame()
    if suffix == "vr":
        all_days_df = load_virtual_reality_spatial_firing(all_days_df, recording_paths, save_path=None)
    elif suffix == "of":
        all_days_df = load_open_field_spatial_firing(all_days_df, recording_paths, save_path=None)

    all_days_df = add_full_session_id(all_days_df, recording_paths)
    all_days_df = add_session_identifiers(all_days_df)

    if save_path is not None:
        all_days_df.to_pickle(save_path+"/All_mice_"+suffix+".pkl")
        all_days_df.to_csv(save_path+"/All_mice_"+suffix+".csv")
    return all_days_df

def check_structure_session_id(session_id):
    # looks at string of session id and returns the correct structure if extra bits are added
    # eg M1_D1_2020-08-03_16-11-14vr should be M1_D1_2020-08-03_16-11-14

    ending = session_id.split("-")[-1]
    corrected_ending = ''.join(filter(str.isdigit, ending))

    if corrected_ending == ending:
        return session_id
    else:
        return session_id.split(ending)[0]+corrected_ending

def add_session_identifiers(all_days_df):
    timestamp_list = []
    date_list = []
    mouse_list = []
    training_day_list = []

    for index, cluster_df in all_days_df.iterrows():
        session_id = cluster_df["session_id"]

        session_id = check_structure_session_id(session_id)
        timestamp_string = session_id.split("_")[-1][0:8]  # eg 14-49-23  time = 14:49, 23rd second
        date_string = session_id.split("_")[-2]
        mouse = session_id.split("_")[0]
        training_day = session_id.split("_")[1]

        timestamp_list.append(timestamp_string)
        date_list.append(date_string)
        mouse_list.append(mouse)
        training_day_list.append(training_day)

    all_days_df["timestamp"] = timestamp_list
    all_days_df["date"] = date_list
    all_days_df["mouse"] = mouse_list
    all_days_df["recording_day"] = training_day_list

    return all_days_df

def plot_summary(days_data, save_path=None):
    '''
    :param days_data: a pandas dataframe of spatial firing from all days of recording for all mice for one experiment
    :param save_path:
    :return: summary plots
    '''

def to_days(days_strings):
    days_int= []
    for i in range(len(days_strings)):
        try:
            days_int.append(int(days_strings[i].split("D")[-1]))
        except Exception as ex:
            days_int.append(np.nan)

    return np.array(days_int)

def plot_stat_across_days(mouse_all_days, collumn="", save_path=None):
    if collumn in list(mouse_all_days):

        fig, ax = plt.subplots(figsize=(6,6))
        plt.scatter(to_days(np.asarray(mouse_all_days["recording_day"])), mouse_all_days[collumn], alpha=0.3, color="b", marker="o")
        max_day = 0
        for day in np.unique(mouse_all_days["recording_day"]):
            try:
                day_data = mouse_all_days[(mouse_all_days["recording_day"] == day)]
                day = int(day.split("D")[-1])
                mean = np.nanmean(day_data[collumn])
                plt.scatter(day, mean, color="k", alpha=1, marker="_")

                if day>max_day:
                    max_day=day

            except Exception as ex:
                print("there was an error with this day")

        mouse = mouse_all_days["mouse"].iloc[0]
        plt.xlabel("Day",  fontsize=20)
        plt.ylabel(get_tidy_title(collumn),  fontsize=20)
        plt.xlim([0,max_day])
        plt.ylim(get_y_lim(collumn))
        plt.title(mouse, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.subplots_adjust(left=0.28, top=0.8, bottom=0.2)

        plt.savefig(save_path+mouse+"_"+collumn+".png", dpi=300)
        plt.show()

def get_y_lim(collumn):
    if collumn == "ThetaIndex":
        return [-0.1, 0.7]
    elif collumn == "rate_map_correlation_first_vs_second_half":
        return [-1,1]
    elif collumn =="grid_score":
        return [-0.75, 1]
    elif collumn =="hd_score":
        return [0, 1]
    elif collumn =="grid_spacing":
        return [-0.1, 0.6]
    elif collumn =="Boccara_theta_class":
        return [-0.05, 1.05]


def plot_cell_counts_across_days(mouse_all_days, save_path=None):

    fig, ax = plt.subplots(figsize=(6,6))

    max_day =0
    for day in np.unique(mouse_all_days["recording_day"]):
        try:
            day_data = mouse_all_days[(mouse_all_days["recording_day"] == day)]
            day = int(day.split("D")[-1])
            count = len(day_data)
            if count == 1:
                # check if the 1 row corresponds to a placeholder for having no cells in the session
                if np.isnan(day_data.cluster_id.iloc[0]):
                    count=0

            if day>max_day:
                max_day=day

            plt.scatter(day, count, color="k", alpha=1, marker="_")
        except Exception as ex:
            print("There was an error with a day")


    mouse = mouse_all_days["mouse"].iloc[0]
    plt.xlabel("Day",  fontsize=20)
    plt.ylabel("Cell Count",  fontsize=20)
    plt.xlim([0,max_day])
    plt.title(mouse, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.28, top=0.8, bottom=0.2)
    plt.savefig(save_path+mouse+"_cel_count.png", dpi=300)
    plt.show()

def plot_summary_per_mouse(days_data, save_path=None):
    '''
    :param days_data: a pandas dataframe of spatial firing from all days of recording for all mice for one experiment
    :param save_path:
    :return: summary plots
    '''

    for mouse in np.unique(days_data["mouse"]):
        mouse_all_days = days_data[(days_data["mouse"] == mouse)]
        plot_cell_counts_across_days(mouse_all_days, save_path)
        plot_stat_across_days(mouse_all_days, collumn="ThetaIndex", save_path=save_path)
        plot_stat_across_days(mouse_all_days, collumn="rate_map_correlation_first_vs_second_half", save_path=save_path)
        plot_stat_across_days(mouse_all_days, collumn="grid_score", save_path=save_path)
        plot_stat_across_days(mouse_all_days, collumn="hd_score", save_path=save_path)
        plot_stat_across_days(mouse_all_days, collumn="Boccara_theta_class", save_path=save_path)

def add_model_classifications(df, linear_model_classifications, mixed_model_classifications):
    """
    :param df: data frame with the structure of spatial firing
    :param linear_model_classifications: dataframe with the linear model classifications
    :param mixed_model_classifications: dataframe with the mixed model classifications
    :return: a dataframe with the structure of the linear model but with the mixed model
    classifications and the spatial firing metrics added
    """
    new_df = pd.DataFrame()

    # rename the classification collumn to be specific to linear model or mixed model
    linear_model_classifications = linear_model_classifications.rename(columns={'classification': ('linear_model_class')})
    mixed_model_classifications = mixed_model_classifications.rename(columns={'classication': ('mixed_model_class')}) # TODO fix typo in R script

    # merge the linear and mixed model dataframes based on shared collumns that are the same
    merged_df = linear_model_classifications.merge(mixed_model_classifications, how = 'inner', on = ['cluster_id', 'session_id',
                                                                                                     'trial_type', 'track_region'])
    collumns_to_add = df.columns.difference(merged_df.columns)

    # find the matching cluster per row and add the spatial firing metrics to the merged df
    for index, row in merged_df.iterrows():
        row = row.to_frame().T.reset_index(drop=True)
        session_id = row["session_id"].iloc[0]
        cluster_id = row["cluster_id"].iloc[0]

        # only add to dataframe if match was found
        matched_row = df[(df["session_id"] == session_id) &
                         (df["cluster_id"] == cluster_id)]

        if len(matched_row)==1:
            # add spatial firing metrics
            for collumn in collumns_to_add:
                row[collumn] = [matched_row[collumn].iloc[0]]

            new_df = pd.concat([new_df, row], ignore_index=True)

    return new_df

def main():
    print("============================================")
    print("============================================")

    # =================== for concatenation ====================================== #
    #vr_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Harry/Cohort8_may2021/vr", suffix="vr", save_path="/mnt/datastore/Harry/Cohort8_may2021/summary/")
    #of_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Harry/Cohort8_may2021/of", suffix="of", save_path="/mnt/datastore/Harry/Cohort8_may2021/summary/")
    #combined_df = combine_of_vr_dataframes(vr_data, of_data)
    #combined_df.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")

    vr_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Harry/Cohort7_october2020/vr", suffix="vr", save_path="/mnt/datastore/Harry/Cohort7_october2020/summary/")
    of_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Harry/Cohort7_october2020/of", suffix="of", save_path="/mnt/datastore/Harry/Cohort7_october2020/summary/")
    combined_df = combine_of_vr_dataframes(vr_data, of_data)
    combined_df.to_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/concatenated_dataframes/combined_Cohort7.pkl")

    vr_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort2/VirtualReality", suffix="vr", save_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort2/")
    of_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort2/OpenField", suffix="of", save_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort2/")
    combined_df = combine_of_vr_dataframes(vr_data, of_data)
    combined_df.to_pickle("/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort2/combined_Cohort2.pkl")

    vr_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort3/VirtualReality", suffix="vr", save_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort3/")
    of_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort3/OpenFeild", suffix="of", save_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort3/")
    combined_df = combine_of_vr_dataframes(vr_data, of_data)
    combined_df.to_pickle("/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort3/combined_Cohort3.pkl")

    vr_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort4/VirtualReality", suffix="vr", save_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort4/")
    of_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort4/OpenFeild", suffix="of", save_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort4/")
    combined_df = combine_of_vr_dataframes(vr_data, of_data)
    combined_df.to_pickle("/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort4/combined_Cohort4.pkl")

    vr_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort5/VirtualReality", suffix="vr", save_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort5/")
    of_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort5/OpenField", suffix="of", save_path="/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort5/")
    combined_df = combine_of_vr_dataframes(vr_data, of_data)
    combined_df.to_pickle("/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort5/combined_Cohort5.pkl")

    #vr_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Harry/Cohort6_july2020/vr", suffix="vr", save_path="/mnt/datastore/Harry/Cohort6_july2020/summary/")
    #of_data = summarise_experiment(recordings_folder_path="/mnt/datastore/Harry/Cohort6_july2020/of", suffix="of", save_path="/mnt/datastore/Harry/Cohort6_july2020/summary/")
    # ============= for loading from concatenated dataframe ====================== #

    #vr_data = pd.read_pickle("/mnt/datastore/Harry/Cohort7_october2020/summary/All_mice_vr.pkl")
    #of_data = pd.read_pickle("/mnt/datastore/Harry/Cohort7_october2020/summary/All_mice_of.pkl")
    #vr_data = pd.read_pickle("/mnt/datastore/Harry/Cohort8_may2021/summary/All_mice_vr.pkl")
    #of_data = pd.read_pickle("/mnt/datastore/Harry/Cohort8_may2021/summary/All_mice_of.pkl")
    #plot_summary_per_mouse(of_data, save_path="/mnt/datastore/Harry/Cohort7_october2020/summary/")
    #plot_summary_per_mouse(vr_data, save_path="/mnt/datastore/Harry/Cohort7_october2020/summary/")
    #linear_model_classifications = pd.read_csv("/mnt/datastore/Harry/Ramp_cells_open_field_paper/linear_model_classifations.csv")
    #mixed_model_classifications = pd.read_csv("/mnt/datastore/Harry/Ramp_cells_open_field_paper/mixed_model_classifations.csv")
    #mixed_model_classifications = pd.read_csv("/mnt/datastore/Harry/Ramp_cells_open_field_paper/mixed_model_classifations_best_score.csv")

    #combined_df = combine_of_vr_dataframes(vr_data, of_data)
    #combined_df = add_model_classifications(combined_df, linear_model_classifications, mixed_model_classifications)
    #combined_df.to_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")

    #vr_data2 = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/summary/All_mice_vr.pkl")
    #of_data2 = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/summary/All_mice_of.pkl")
    #plot_summary_per_mouse(of_data2, save_path="/mnt/datastore/Harry/Cohort6_july2020/summary/")
    #plot_summary_per_mouse(vr_data2, save_path="/mnt/datastore/Harry/Cohort6_july2020/summary/")

    print("============================================")
    print("============================================")

if __name__ == '__main__':
    main()