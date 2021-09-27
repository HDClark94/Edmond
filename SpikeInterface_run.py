import EdmondHC.SpikeInterface_fileutility as file_utility
from control_sorting_analysis import get_tags_parameter_file, check_for_paired
import os
import pandas as pd
import copy as cp
from collections import namedtuple
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as sorters
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import json
import pickle
import EdmondHC.spikeinterfaceHelper as spikeinterfaceHelper
from tqdm import tqdm
import numpy as np
import EdmondHC.SpikeInterface_setting as setting
import logging
from types import SimpleNamespace
import matplotlib.pylab as plt


from scipy.signal import butter,filtfilt

def filterRecording(recording, sampling_freq, lp_freq=300,hp_freq=6000,order=3):
    fn = sampling_freq / 2.
    band = np.array([lp_freq, hp_freq]) / fn

    b, a = butter(order, band, btype='bandpass')

    if not (np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1)):
        raise ValueError('Filter is not stable')

    for i in tqdm(range(recording._timeseries.shape[0])):
        recording._timeseries[i,:] = filtfilt(b,a,recording._timeseries[i,:])

    return recording

def plot_waveforms(sorted_df, figure_path, sorter, recording_type=""):
    print('I will plot the waveform shapes for each cluster.')
    for cluster in tqdm(range(len(sorted_df))):
        #extract waveforms from dataframe
        waveforms = sorted_df.waveforms[cluster]
        print(np.shape(waveforms))
        waveforms = np.stack([w for w in waveforms if w is not None])
        max_channel = sorted_df.max_channel.values[cluster]
        cluster_id = sorted_df.cluster_id[cluster]
        tetrode = max_channel//setting.num_tetrodes #get the tetrode number

        #plot spike waveform from the same tetrode
        fig = plt.figure()
        for i in range(4):
            ax = fig.add_subplot(2,2,i+1)
            print(np.shape(waveforms[:,tetrode+i,75:105].T))
            ax.plot(waveforms[:,tetrode+i,75:105].T,color='lightslategray')
            template = waveforms[:,tetrode+i,75:105].mean(0)
            ax.plot(template, color='red')

        plt.savefig(figure_path + '/' + sorter + '_' + recording_type + "_" + sorted_df.session_id[cluster] + '_' + str(cluster_id) + '_waveforms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_waveforms_all_channels(sorted_df, figure_path, sorter, recording_type=""):
    print('I will plot the waveform shapes for each cluster.')
    for cluster in tqdm(range(len(sorted_df))):
        #extract waveforms from dataframe
        waveforms = sorted_df.waveforms[cluster]
        waveforms = np.stack([w for w in waveforms if w is not None])
        max_channel = sorted_df.max_channel.values[cluster]
        cluster_id = sorted_df.cluster_id[cluster]
        tetrode = max_channel//setting.num_tetrodes #get the tetrode number

        #plot spike waveform from the same tetrode
        fig = plt.figure()
        for i in range(16):
            ax = fig.add_subplot(4,4,i+1)
            ax.plot(waveforms[:,i,:].T,color='lightslategray')
            template = waveforms[:,i,:].mean(0)
            ax.plot(template, color='red')

        plt.savefig(figure_path + '/' + sorter + '_' + recording_type + "_" + sorted_df.session_id[cluster] + '_' + str(cluster_id) + '_waveforms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

def post_process(recording, sorting, sinput, soutput, signal,
                 sorter, concatenated_sorting=False, stitch_point=None, suffix=""):

    # collect cluster stats and waveforms
    st.postprocessing.get_unit_max_channels(recording, sorting, max_spikes_per_unit=100)
    st.postprocessing.get_unit_waveforms(recording, sorting, max_spikes_per_unit=100, return_idxs=True)
    for id in sorting.get_unit_ids():
        number_of_spikes = len(sorting.get_unit_spike_train(id))
        mean_firing_rate = number_of_spikes/(len(signal)/setting.sampling_rate)
        sorting.set_unit_property(id, 'number_of_spikes', number_of_spikes)
        sorting.set_unit_property(id, 'mean_firing_rate', mean_firing_rate)
    session_id = sinput.recording_to_sort.split('/')[-1]

    # apply some curation
    if len(sorting.get_unit_ids()) > 0:
        sorting_x_curated = sorting
        sorting_x_curated = st.curation.threshold_snrs(sorting=sorting_x_curated, recording = recording, threshold =2, threshold_sign='less', max_snr_spikes_per_unit=100, apply_filter=False) #remove when less than threshold
        sorting_x_curated = st.curation.threshold_firing_rates(sorting_x_curated, duration_in_frames=len(signal[0]), threshold=0.5, threshold_sign='less')
        sorting_x_curated = st.curation.threshold_isolation_distances(sorting_x_curated, recording=recording, threshold = 0.9, threshold_sign='less')
        curated_sorter_df = spikeinterfaceHelper.sorter2dataframe(sorting_x_curated, session_id)
    else:
        print("no untis to curate")
        # we want an empty dataframe where there is no curated clusters
        sorting_x_curated = sorting
        curated_sorter_df = spikeinterfaceHelper.sorter2dataframe(sorting_x_curated, session_id)

    if concatenated_sorting:
        vr_curated_sorter_df, n_spikes_vr_list = split_df_with_stitchpoint(curated_sorter_df, recording, stitch_point, signal, recording_type="vr", n_spikes_list_in=None)
        of_curated_sorter_df, v_spikes_of_list = split_df_with_stitchpoint(curated_sorter_df, recording, stitch_point, signal, recording_type="of", n_spikes_list_in=n_spikes_vr_list)

        vr_curated_sorter_df.to_pickle(sinput.recording_to_sort +'/processed/'+sorter+'_vrconcat_sorted_df.pkl')
        of_curated_sorter_df.to_pickle(sinput.recording_to_sort +'/processed/'+sorter+'_ofconcat_sorted_df.pkl')
        plot_waveforms(vr_curated_sorter_df, soutput.waveform_figure, sorter, recording_type="vrconcat")
        plot_waveforms(of_curated_sorter_df, soutput.waveform_figure, sorter, recording_type="ofconcat")

    else:
        curated_sorter_df.to_pickle(sinput.recording_to_sort +'/processed/'+sorter+'_'+suffix+'_df.pkl')
        plot_waveforms(curated_sorter_df, soutput.waveform_figure, sorter, recording_type=suffix)

    return

def sort_recording(recording_to_sort, sorter=None, try_dual_sorting=False):
    '''

    :param recording_to_sort: string absolute path of recording folder in which to sort to sort
    :param sorter: string name of sorter
    :param try_dual_sorting: if true, the paired recording will be inspected in the param.txt
    and dual sorting will be attemped, if no paired recording is found, recording is sorted as single session
    :return: sorted and curated spike trains are inserted into /processed subdirectory and
    waveforms plotted
    '''
    if sorter is not None: # ms4 is default
        setting.sorterName = sorter

    sinput = SimpleNamespace()
    soutput = SimpleNamespace()
    sinput.recording_to_sort = recording_to_sort

    paired_recording_to_sort = None
    if try_dual_sorting:
        tags = get_tags_parameter_file(recording_to_sort)
        paired_recording_to_sort, _ = check_for_paired(tags)

    #make output folder
    try:
        os.mkdir(sinput.recording_to_sort+'/processed/')
    except FileExistsError:
        print('Folder already there')

    sinput.probe_file =   '/home/nolanlab/to_sort/sorting_files/tetrode_16.prb'
    sinput.sort_param = '/home/nolanlab/to_sort/sorting_files/params.json'
    sinput.tetrode_geom = '/home/nolanlab/to_sort/sorting_files/geom_all_tetrodes_original.csv'
    sinput.dead_channel = sinput.recording_to_sort +'/dead_channels.txt'
    soutput.sorter_df = sinput.recording_to_sort +'/processed/'+sorter+'_sorted_df.pkl'
    soutput.sorter_curated_df = sinput.recording_to_sort +'/processed/'+sorter+'sorted_curated_df.pkl'
    soutput.waveform_figure = sinput.recording_to_sort+'/processed'
    geom = pd.read_csv(sinput.tetrode_geom, header=None).values
    bad_channel = file_utility.getDeadChannel(sinput.dead_channel)
    base_signal = file_utility.load_OpenEphysRecording(sinput.recording_to_sort)

    if try_dual_sorting and (paired_recording_to_sort is not None):
        stitch_point = len(base_signal[0])
        signal_paired = file_utility.load_OpenEphysRecording(setting.server_path_first_half+paired_recording_to_sort)
        signal = np.concatenate((base_signal, signal_paired), axis=1)

        base_recording = se.NumpyRecordingExtractor(base_signal,setting.sampling_rate,geom)
        filterRecording(base_recording,setting.sampling_rate) #filer recording
        base_recording = st.preprocessing.remove_bad_channels(base_recording, bad_channel_ids=bad_channel) #remove bad channel
        #base_recording = st.preprocessing.whiten(base_recording, chunk_size=30000, cache_chunks=False, seed=0)


        paired_recording = se.NumpyRecordingExtractor(signal_paired,setting.sampling_rate,geom)
        filterRecording(paired_recording,setting.sampling_rate) #filer recording
        paired_recording = st.preprocessing.remove_bad_channels(paired_recording, bad_channel_ids=bad_channel) #remove bad channel
        #paired_recording = st.preprocessing.whiten(paired_recording, chunk_size=30000, cache_chunks=False, seed=0)

    recording = se.NumpyRecordingExtractor(signal,setting.sampling_rate,geom)
    filterRecording(recording,setting.sampling_rate) #filer recording
    recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel) #remove bad channel
    #recording = st.preprocessing.whiten(recording, chunk_size=30000, cache_chunks=False, seed=0)

    # run sorter
    sorting_x = sorters.run_sorter(setting.sorterName, recording, verbose=True, output_folder=setting.sorterName)
    post_process(recording, sorting_x, sinput, soutput, signal,
                 sorter, concatenated_sorting=True, stitch_point=stitch_point, suffix="")

    sorting_x_base = sorters.run_sorter(setting.sorterName, base_recording, verbose=True, output_folder=setting.sorterName)
    post_process(base_recording, sorting_x_base, sinput, soutput, base_signal,
                 sorter, concatenated_sorting=False, stitch_point=None, suffix="vrsingle")

    sorting_x_paired = sorters.run_sorter(setting.sorterName, paired_recording, verbose=True, output_folder=setting.sorterName)
    post_process(paired_recording, sorting_x_paired, sinput, soutput, signal_paired,
                 sorter, concatenated_sorting=False, stitch_point=None, suffix="ofsingle")

def split_df_with_stitchpoint(curated_sorter_df, recording, stitch_point, signal, recording_type, n_spikes_list_in=None):
    split_curated_sorter_df = curated_sorter_df.copy()

    local_firing_rate = []
    n_spikes_list = []
    i=0
    for cluster in tqdm(range(len(curated_sorter_df))):
        spike_train = curated_sorter_df.spike_train[cluster]

        if recording_type=="vr":
            spike_train = spike_train[spike_train<stitch_point]

            number_of_spikes = len(spike_train)
            local_firing_rate.append(number_of_spikes/(stitch_point/setting.sampling_rate))
            waveforms_idxs = curated_sorter_df.waveforms_idxs[cluster]
            curated_sorter_df.waveforms_idxs[cluster] = waveforms_idxs[waveforms_idxs<number_of_spikes]
            curated_sorter_df.waveforms[cluster] = curated_sorter_df.waveforms[cluster][waveforms_idxs<number_of_spikes]

        elif recording_type=="of":
            spike_train = spike_train[spike_train>stitch_point]

            number_of_spikes = len(spike_train)
            local_firing_rate.append(number_of_spikes/((len(signal)-stitch_point)/setting.sampling_rate))
            waveforms_idxs = curated_sorter_df.waveforms_idxs[cluster]
            curated_sorter_df.waveforms_idxs[cluster] = waveforms_idxs[waveforms_idxs>n_spikes_list_in[i]]
            curated_sorter_df.waveforms[cluster] = curated_sorter_df.waveforms[cluster][waveforms_idxs>n_spikes_list_in[i]]

        split_curated_sorter_df.spike_train[cluster] = spike_train
        n_spikes_list.append(number_of_spikes)
        i+=1

        # now recompute firing rate

    split_curated_sorter_df["local_firing_rate"] = local_firing_rate

    return split_curated_sorter_df, n_spikes_list


def run_multisorter_recordings(parent_recording_dir):
    recordings_list = [f.path for f in os.scandir(parent_recording_dir) if f.is_dir()]

    for i in range(len(recordings_list)):
        recording_to_sort = recordings_list[i]
        sort_recording(recording_to_sort, sorter="mountainsort4", try_dual_sorting=True)
        sort_recording(recording_to_sort, sorter="klusta", try_dual_sorting=True)
        sort_recording(recording_to_sort, sorter="tridesclous", try_dual_sorting=True)


def main():
    recording_to_sort = setting.server_path_first_half+"Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted/M1_D8_2019-06-26_13-31-11"

    #pad=pd.read_pickle(recording_to_sort+"/processed/mountainsort4_sorted_df.pkl")
    sort_recording(recording_to_sort, sorter="mountainsort4", try_dual_sorting=True)
    #sort_recording(recording_to_sort, sorter="klusta", try_dual_sorting=True)
    #sort_recording(recording_to_sort, sorter="tridesclous", try_dual_sorting=True)

    '''
    run_multisorter_recordings("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted")
    run_multisorter_recordings("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted")
    run_multisorter_recordings("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M2_sorted")
    run_multisorter_recordings("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort4/VirtualReality/M3_sorted")
    run_multisorter_recordings("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M1_sorted")
    run_multisorter_recordings("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M6_sorted")
    run_multisorter_recordings("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/245_sorted")
    run_multisorter_recordings("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/1124_sorted")
    '''
    print("hello there")

    # possible sorter names
    '''
    'mountainsort4'
    'klusta'
    'tridesclous'
    
    'hdsort'
    'ironclust'
    'kilosort'
    'kilosort2'
    'spykingcircus'
    'herdingspikes'
    'waveclus'
    '''

if __name__ == '__main__':
    main()