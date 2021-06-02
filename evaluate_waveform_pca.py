import pandas as pd
import OpenEphys as oe
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import spikeinterface.toolkit as st

import spikeinterface.extractors as se
from Edmond.spikeforest_comparison import file_utility
from tqdm import tqdm
n_comp = 10
whiten=True
seed = 42

def spatial_firing2label(spatial_firing):
    times = []
    labels = []
    for cluster_id in np.unique(spatial_firing["cluster_id"]):
        cluster_spatial_firing = spatial_firing[(spatial_firing["cluster_id"] == cluster_id)]
        cluster_times = list(cluster_spatial_firing["firing_times"].iloc[0])
        cluster_labels = list(cluster_id*np.ones(len(cluster_times)))

        times.extend(cluster_times)
        labels.extend(cluster_labels)
    return np.array(times), np.array(labels)

def create_phy(recording, spatial_firing, output_folder, sampling_rate=30000):
    signal = file_utility.load_OpenEphysRecording(recording)
    dead_channel_path = recording +'/dead_channels.txt'
    bad_channel = file_utility.getDeadChannel(dead_channel_path)
    tetrode_geom = '/home/ubuntu/to_sort/sorting_files/geom_all_tetrodes_original.csv'
    geom = pd.read_csv(tetrode_geom,header=None).values
    recording = se.NumpyRecordingExtractor(signal, sampling_rate, geom)
    recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel)
    recording = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = st.preprocessing.whiten(recording)
    recording = se.CacheRecordingExtractor(recording)
    # reconstruct a sorting extractor
    times, labels = spatial_firing2label(spatial_firing)
    sorting = se.NumpySortingExtractor()
    sorting.set_times_labels(times=times, labels=labels)
    sorting.set_sampling_frequency(sampling_frequency=sampling_rate)

    st.postprocessing.export_to_phy(recording, sorting, output_folder=output_folder, copy_binary=False, ms_before=0.5, ms_after=0.5)
    print("I have created the phy output for ", recording)


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    recording_1= "/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D23_2020-11-28_15-13-28/MountainSort/DataFrames/spatial_firing.pkl"
    output_folder = "/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D23_2020-11-28_15-13-28/MountainSort/DataFrames/phy2/"
    spatial_firing_1 = pd.read_pickle(recording_1)
    recording = "/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D23_2020-11-28_15-13-28"
    create_phy(recording, spatial_firing_1, output_folder, sampling_rate=30000)

if __name__ == '__main__':
    main()