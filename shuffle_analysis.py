import pandas as pd
from joblib import Parallel, delayed
import os
import multiprocessing
from control_sorting_analysis import get_session_type
import PostSorting
import numpy as np
import settings

prm = PostSorting.parameters.Parameters()

def run_shuffle_analysis(list_of_recordings, n_shuffles, prm):

    for recording in list_of_recordings:

        if os.path.exists(recording+"/MountainSort/DataFrames/spatial_firing.pkl") and not \
                os.path.exists(recording+"/MountainSort/DataFrames/spatial_firing_with_shuffled_threshold.pkl"):
            session_type = get_session_type(recording)

            if session_type == "openfield":
                run_shuffle_analysis_open_field(recording, n_shuffles, prm)

            elif session_type == "vr":
                run_shuffle_analysis_vr(recording, n_shuffles, prm)

def run_parallel_of_shuffle(shuffle_id, shuffled_cluster_spike_data, synced_spatial_data, prm):
    prm.set_sampling_rate(30000)
    prm.set_pixel_ratio(440)

    single_shuffle = shuffled_cluster_spike_data[(shuffled_cluster_spike_data["shuffle_id"] == shuffle_id)]

    single_shuffle = PostSorting.open_field_spatial_firing.process_spatial_firing(single_shuffle, synced_spatial_data)
    single_shuffle = PostSorting.speed.calculate_speed_score(synced_spatial_data, single_shuffle, settings.gauss_sd_for_speed_score, settings.sampling_rate)
    _, single_shuffle = PostSorting.open_field_head_direction.process_hd_data(single_shuffle, synced_spatial_data, prm)
    _, single_shuffle = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, single_shuffle, prm)
    single_shuffle = PostSorting.open_field_firing_maps.calculate_spatial_information(single_shuffle)
    single_shuffle = PostSorting.open_field_grid_cells.process_grid_data(single_shuffle)
    single_shuffle = PostSorting.open_field_border_cells.process_border_data(single_shuffle)

    return single_shuffle

def generate_shuffled_times(cluster_firing, n_shuffles):
    cluster_firing = cluster_firing[["cluster_id", "firing_times", "mean_firing_rate", "recording_length_sampling_points"]]

    shuffle_firing = pd.DataFrame()
    for i in range(n_shuffles):
        shuffle = cluster_firing.copy()
        firing_times = shuffle["firing_times"].to_numpy()[0]
        random_firing_additions = np.random.randint(low=int(20*settings.sampling_rate),
                                                    high=int(580*settings.sampling_rate), size=len(firing_times))

        shuffled_firing_times = firing_times + random_firing_additions
        recording_length = int(cluster_firing["recording_length_sampling_points"].iloc[0])
        shuffled_firing_times[shuffled_firing_times > recording_length] = shuffled_firing_times[shuffled_firing_times > recording_length] - recording_length # wrap around the firing times that exceed the length of the recording
        shuffle["firing_times"] = [shuffled_firing_times]

        shuffle_firing = pd.concat([shuffle_firing, shuffle], ignore_index=True)

    shuffle_firing["shuffle_id"] = np.arange(0, n_shuffles)
    return shuffle_firing

def run_shuffle_analysis_open_field(recording, n_shuffles, prm):
    num_cores = int(os.environ['HEATMAP_CONCURRENCY']) if os.environ.get('HEATMAP_CONCURRENCY') else multiprocessing.cpu_count()

    spike_data_spatial = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
    synced_spatial_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position.pkl")

    speed_threshold_poss = []
    speed_threshold_negs = []
    hd_thresholds = []
    rayleigh_thresholds = []
    spatial_thresholds = []
    grid_thresholds = []
    border_thresholds = []

    for cluster_index, cluster_id in enumerate(spike_data_spatial.cluster_id):
        cluster_spike_data = spike_data_spatial[(spike_data_spatial["cluster_id"] == cluster_id)]
        shuffled_cluster_spike_data = generate_shuffled_times(cluster_spike_data, n_shuffles)
        shuffles = shuffled_cluster_spike_data.shuffle_id

        shuffled_cluster_spike_data = Parallel(n_jobs=num_cores)(delayed(run_parallel_of_shuffle)(shuffle, shuffled_cluster_spike_data, synced_spatial_data, prm) for shuffle in shuffles)
        shuffled_cluster_spike_data = pd.concat(shuffled_cluster_spike_data)

        # calculate the 95th percentile threshold for individual clusters
        adjusted_speed_threshold_pos = np.mean(shuffled_cluster_spike_data["speed_score"]) + (np.std(shuffled_cluster_spike_data["speed_score"])*1.645)
        adjusted_speed_threshold_neg = np.mean(shuffled_cluster_spike_data["speed_score"]) + (np.std(shuffled_cluster_spike_data["speed_score"])*-1.645)
        adjusted_hd_threshold = np.mean(shuffled_cluster_spike_data["hd_score"]) + (np.std(shuffled_cluster_spike_data["hd_score"])*1.645)
        adjusted_rayleigh_threshold = np.mean(shuffled_cluster_spike_data["rayleigh_score"]) + (np.std(shuffled_cluster_spike_data["rayleigh_score"])*1.645)
        adjusted_spatial_threshold = np.mean(shuffled_cluster_spike_data["spatial_information_score"]) + (np.std(shuffled_cluster_spike_data["spatial_information_score"])*1.645)
        adjusted_grid_threshold = np.mean(shuffled_cluster_spike_data["grid_score"]) + (np.std(shuffled_cluster_spike_data["grid_score"])*1.645)
        adjusted_border_threshold = np.mean(shuffled_cluster_spike_data["border_score"]) + (np.std(shuffled_cluster_spike_data["border_score"])*1.645)

        speed_threshold_poss.append(adjusted_speed_threshold_pos)
        speed_threshold_negs.append(adjusted_speed_threshold_neg)
        hd_thresholds.append(adjusted_hd_threshold)
        rayleigh_thresholds.append(adjusted_rayleigh_threshold)
        spatial_thresholds.append(adjusted_spatial_threshold)
        grid_thresholds.append(adjusted_grid_threshold)
        border_thresholds.append(adjusted_border_threshold)

    spike_data_spatial["speed_threshold_pos"] = speed_threshold_poss
    spike_data_spatial["speed_threshold_neg"] = speed_threshold_negs
    spike_data_spatial["hd_threshold"] = hd_thresholds
    spike_data_spatial["rayleigh_threshold"] = rayleigh_thresholds
    spike_data_spatial["spatial_threshold"] = spatial_thresholds
    spike_data_spatial["grid_threshold"] = grid_thresholds
    spike_data_spatial["border_threshold"] = border_thresholds

    spike_data_spatial.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
    spike_data_spatial.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing_with_shuffled_threshold.pkl")
    return

def run_shuffle_parallel(recording_path, shuffle_id):
    '''
    creates a single shuffle of each cell and saves it in recording/sorter/dataframes/shuffles/
    :param recording_path: path to a recording folder
    :param shuffle_id: integer id of a single shuffle
    '''

    spike_data_spatial = pd.read_pickle(recording_path+"/MountainSort/DataFrames/spatial_firing.pkl")
    synced_spatial_data = pd.read_pickle(recording_path+"/MountainSort/DataFrames/position.pkl")

    shuffle = pd.DataFrame()
    for cluster_index, cluster_id in enumerate(spike_data_spatial.cluster_id):
        cluster_spike_data = spike_data_spatial[(spike_data_spatial["cluster_id"] == cluster_id)]
        shuffled_cluster_spike_data = generate_shuffled_times(cluster_spike_data, n_shuffles=1)
        shuffled_cluster_spike_data = run_parallel_of_shuffle(0, shuffled_cluster_spike_data, synced_spatial_data, prm)
        shuffle = pd.concat([shuffle, shuffled_cluster_spike_data], ignore_index=True)

    if not os.path.exists(recording_path+"/MountainSort/DataFrames/shuffles"):
        os.mkdir(recording_path+"/MountainSort/DataFrames/shuffles")

    shuffle.to_pickle(recording_path+"/MountainSort/DataFrames/shuffles/shuffle_"+str(shuffle_id)+".pkl")


def run_shuffle_analysis_vr(recording, n_shuffles, prm):
    return

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #n_shuffles = 1000

    # get list of all recordings in the recordings folder
    of_recording_list = [f.path for f in os.scandir("/mnt/datastore/Harry/Cohort8_may2021/of/") if f.is_dir()]

    #run_shuffle_analysis(of_recording_list, n_shuffles, prm)


    recording_path = '/mnt/datastore/Harry/Cohort8_may2021/of/M13_D5_2021-05-14_11-34-47'
    shuffle_id = 1
    run_shuffle_parallel(recording_path, shuffle_id)

    print("dfdf")



if __name__ == '__main__':
    main()