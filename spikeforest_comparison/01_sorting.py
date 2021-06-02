#%%
import json
import logging
import os
import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc
import spikeinterface.extractors as se
import spikeinterface.sorters as sorters
import spikeinterface.toolkit as st
import spikeinterface.widgets as sw
from tqdm import tqdm

import file_utility
import Logger
import setting
import SnakeIOHelper
import spikeinterfaceHelper
from PostSorting.make_plots import plot_waveforms
from PreClustering.pre_process_ephys_data import filterRecording

#for logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__)+':'+__name__)

#%% define input and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'vr_workflow.smk', [setting.debug_folder+'/processed/mountainsort4/sorter_curated_df.pkl'],
    'sort_spikes')

#%% Load data and create recording extractor

signal = file_utility.load_OpenEphysRecording(sinput.recording_to_sort)
geom = pd.read_csv(sinput.tetrode_geom,header=None).values

dead_channel_path =  Path(sinput.recording_to_sort+'/dead_channels.txt')
if dead_channel_path.exists():
    bad_channel = file_utility.getDeadChannel(dead_channel_path)
else:
    bad_channel = []


#%% Create and filter the recording
logger.info('Filtering files') #TODO logging not show correctly

recording = se.NumpyRecordingExtractor(signal,setting.sampling_rate,geom)
recording = recording.load_probe_file(sinput.probe_file) #load probe definition
filterRecording(recording,setting.sampling_rate) #for faster operation later

#%% Remove some bad channels
recording = st.preprocessing.remove_bad_channels(recording, bad_channel_ids=bad_channel) #remove bad channel
tetrodeNum = np.array(recording.get_channel_ids())//setting.num_tetrodes
#%% perform sorting
with open(sinput.sort_param) as f:
    param = json.load(f)
sorting_ms4 = sorters.run_sorter(setting.sorterName,recording, output_folder=setting.sorterName,
    adjacency_radius=param['adjacency_radius'], detect_sign=param['detect_sign'],verbose=True)

with open(soutput.sorter,'wb') as f:
    pickle.dump(sorting_ms4,f)
    

#%% compute some property of the sorting
st.postprocessing.get_unit_max_channels(recording, sorting_ms4, save_as_property=True,max_spikes_per_unit=100)
st.postprocessing.get_unit_waveforms(recording, sorting_ms4,save_as_features=True, max_spikes_per_unit=100)

for id in sorting_ms4.get_unit_ids():
    number_of_spikes = len(sorting_ms4.get_unit_spike_train(id))
    mean_firing_rate = number_of_spikes/(recording._recording._timeseries.shape[1]/setting.sampling_rate)
    sorting_ms4.set_unit_property(id,'number_of_spikes',number_of_spikes)
    sorting_ms4.set_unit_property(id, 'mean_firing_rate', mean_firing_rate)


#%% save data
with open(soutput.sorter_curated,'wb') as f:
    pickle.dump(sorting_ms4,f)
session_id = sinput.recording_to_sort.split('/')[-1]
sorter_df=spikeinterfaceHelper.sorter2dataframe(sorting_ms4,session_id)
sorter_df.to_pickle(soutput.sorter_df)

#%% Do some simple curation for now
# less to remove
sorting_ms4_curated = st.curation.threshold_snr(sorting=sorting_ms4, recording = recording,
  threshold = 2, threshold_sign='less',
    max_snr_spikes_per_unit=100, apply_filter=False) #remove when less than threshold
print(sorting_ms4_curated.get_unit_ids())

sorting_ms4_curated=st.curation.threshold_firing_rate(sorting_ms4_curated,
    threshold=0.5, threshold_sign='less')
print(sorting_ms4_curated.get_unit_ids())

sorting_ms4_curated=st.curation.threshold_isi_violations(sorting_ms4_curated, 
    threshold = 0.9)
print(sorting_ms4_curated.get_unit_ids())


#%%
#save curated data
curated_sorter_df = spikeinterfaceHelper.sorter2dataframe(sorting_ms4_curated, session_id)
curated_sorter_df.to_pickle(soutput.sorter_curated_df)
sorting_ms4_curated = se.SubSortingExtractor(sorting_ms4,unit_ids=sorting_ms4_curated.get_unit_ids())
with open(soutput.sorter_curated,'wb') as f:
    pickle.dump(sorting_ms4_curated, f)

#%% Plot spike waveforms
plot_waveforms(curated_sorter_df, tetrodeNum, soutput.waveform_figure)
