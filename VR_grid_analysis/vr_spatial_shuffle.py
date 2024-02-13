import warnings
import sys
import matplotlib.pylab as plt

warnings.filterwarnings('ignore')
plt.rc('axes', linewidth=3)
from Edmond.eLife_Grid_anchoring_2024.vr_grid_cells import *


# `values` should be sorted
def get_closest(array, values):
    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array)) | (np.fabs(values - array[np.maximum(idxs - 1, 0)]) < np.fabs(
        values - array[np.minimum(idxs, len(array) - 1)])))
    idxs[prev_idx_is_less] -= 1

    # return indices of closest values
    return idxs

def calculate_spatial_information_shuffle(spatial_firing, position_data, track_length, n_shuffles=100):
    position_heatmap = np.zeros(track_length)
    for x in np.arange(track_length):
        bin_occupancy = len(position_data[(position_data["x_position_cm"] > x) &
                                                (position_data["x_position_cm"] <= x+1)])
        position_heatmap[x] = bin_occupancy
    position_heatmap = position_heatmap*np.diff(position_data["time_seconds"])[-1] # convert to real time in seconds
    occupancy_probability_map = position_heatmap/np.sum(position_heatmap) # Pj

    spatial_information_scores_Ispike_percentile_threshold = []
    spatial_information_scores_Isec_percentile_threshold = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        mean_firing_rate = cluster_df.iloc[0]["number_of_spikes"] / np.sum(len(position_data) * np.diff(position_data["time_seconds"])[-1])  # λ
        recording_length = int(cluster_df["recording_length_sampling_points"].iloc[0])

        minimum_shift = int(20 * settings.sampling_rate)  # 20 seconds
        maximum_shift = int(recording_length - 20 * settings.sampling_rate)  # full length - 20 sec

        ISec_shuffle_scores=[]
        Ispike_shuffle_scores=[]
        for i in range(n_shuffles):
            firing_times = np.array(cluster_df['firing_times'].iloc[0])
            random_firing_additions = np.random.randint(low=minimum_shift, high=maximum_shift)

            shuffled_firing_times = firing_times + random_firing_additions
            shuffled_firing_times[shuffled_firing_times >= recording_length] = shuffled_firing_times[shuffled_firing_times >= recording_length] - recording_length  # wrap around the firing times that exceed the length of the recording
            shuffled_firing_times_in_seconds = shuffled_firing_times/settings.sampling_rate
            closest_indices = get_closest(np.array(position_data["time_seconds"]), shuffled_firing_times_in_seconds)
            shuffle_spike_locations = np.array(position_data["x_position_cm"])[closest_indices]

            spikes, _ = np.histogram(shuffle_spike_locations, bins=track_length, range=(0,track_length))
            rates = spikes/position_heatmap
            Isec, Ispike = spatial_info(mean_firing_rate, occupancy_probability_map, rates)

            ISec_shuffle_scores.append(Isec)
            Ispike_shuffle_scores.append(Ispike)

        ISec_shuffle_scores = np.array(ISec_shuffle_scores)
        Ispike_shuffle_scores = np.array(Ispike_shuffle_scores)

        Isec_threshold = np.nanpercentile(ISec_shuffle_scores, 99)
        Ispike_threshold = np.nanpercentile(Ispike_shuffle_scores, 99)

        spatial_information_scores_Ispike_percentile_threshold.append(Isec_threshold)
        spatial_information_scores_Isec_percentile_threshold.append(Ispike_threshold)

    spatial_firing["spatial_information_scores_Ispike_percentile_threshold"] = spatial_information_scores_Ispike_percentile_threshold
    spatial_firing["spatial_information_scores_Isec_percentile_threshold"] = spatial_information_scores_Isec_percentile_threshold
    return spatial_firing

def add_vr_spatial_classifier(spike_data):
    classifiers_based_on_ISec = []
    classifiers_based_on_ISpike = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data.cluster_id == cluster_id)]  # dataframe for that cluster

        if (cluster_spike_data["spatial_information_scores_Isec_percentile_threshold"].iloc[0] > cluster_spike_data["spatial_information_score_Isec"].iloc[0]):
            classifiers_based_on_ISec.append(True)
        else:
            classifiers_based_on_ISec.append(False)

        if (cluster_spike_data["spatial_information_scores_Ispike_percentile_threshold"].iloc[0] > cluster_spike_data["spatial_information_score_Ispike"].iloc[0]):
            classifiers_based_on_ISpike.append(True)
        else:
            classifiers_based_on_ISpike.append(False)

    spike_data["vr_spatial_classifer_Isec"] = classifiers_based_on_ISec
    spike_data["vr_spatial_classifer_Ispike"] = classifiers_based_on_ISpike
    return spike_data

def process_recordings(vr_recording_path_list):
    vr_recording_path_list.sort()
    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            tags = control_sorting_analysis.get_tags_parameter_file(recording)
            sorter_name = control_sorting_analysis.check_for_tag_name(tags, "sorter_name")
            position_data = pd.read_pickle(recording+"/"+sorter_name+"/DataFrames/position_data.pkl")
            spike_data = pd.read_pickle(recording+"/"+sorter_name+"/DataFrames/spatial_firing.pkl")

            if len(spike_data) != 0:
                spike_data = calculate_spatial_information_shuffle(spike_data, position_data, track_length=get_track_length(recording))
                spike_data = add_vr_spatial_classifier(spike_data)
                spike_data.to_pickle(recording+"/"+sorter_name+"/DataFrames/spatial_firing.pkl")
                print("successfully processed and saved vr_grid analysis on "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)


def main():
    print('-------------------------------------------------------------')
    vr_path_list = []
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    process_recordings(vr_path_list)

    print("look now")

if __name__ == '__main__':
    main()
