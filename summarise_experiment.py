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
        recording_days.append(get_suedo_day(row["session_id"].iloc[0]))
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

def summarise_experiment(recordings_folder_path, suffix=None, save_path=None, prm=None):
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
        all_days_df = load_virtual_reality_spatial_firing(all_days_df, recording_paths, save_path=None, suffix="vr", prm=prm)
    elif suffix == "of":
        all_days_df = load_open_field_spatial_firing(all_days_df, recording_paths, save_path=None, suffix="of", prm=prm)

    all_days_df = add_recording_day(all_days_df)
    all_days_df = add_full_session_id(all_days_df, recording_paths)
    all_days_df = add_mouse_label(all_days_df)



    if save_path is not None:
        all_days_df.to_pickle(save_path+"/All_mice_"+suffix+".pkl")
    return

def plot_summary(days_data, save_path=None):
    '''
    :param days_data: a pandas dataframe of spatial firing from all days of recording for all mice for one experiment
    :param save_path:
    :return: summary plots
    '''


def main():
    print("============================================")#
    print("============================================")

    prm = PostSorting.parameters.Parameters()
    prm.set_sampling_rate(30000)
    prm.set_pixel_ratio(440)
    prm.set_vr_grid_analysis_bin_size(20)

    # =================== for concatenation ====================================== #
    save_path = "/mnt/datastore/Harry/Cohort7_october2020/summary/"
    summarise_experiment(recordings_folder_path="/mnt/datastore/Harry/Cohort7_october2020/vr", suffix="vr", save_path=save_path, prm=prm)
    summarise_experiment(recordings_folder_path="/mnt/datastore/Harry/Cohort7_october2020/of", suffix="of", save_path=save_path, prm=prm)

    # ============= for loading from concatenated dataframe ====================== #

    #vr_data = pd.read_pickle("/mnt/datastore/Harry/Cohort7_october2020/summary/All_mice_vr.pkl")
    #of_data = pd.read_pickle("/mnt/datastore/Harry/Cohort7_october2020/summary/All_mice_of.pkl")
    #plot_summary(vr_data, save_path=save_path)
    #plot_summary(of_data, save_path=save_path)


    print("============================================")#
    print("============================================")


if __name__ == '__main__':
    main()