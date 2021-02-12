
#%%
import json
from pathlib import Path

import hither_sf as hither
import kachery as ka
import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.comparison as sc
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
import spikeinterface.widgets as sw
from spikeforest2_utils import (AutoRecordingExtractor, AutoSortingExtractor,
                                MdaRecordingExtractor)

import hither_sf as hither
import kachery as ka
from spikeforest2 import processing, sorters
import file_utility
import pickle
import os

# os.environ['KACHERY_STORAGE_DIR'] = '/home/ubuntu'

#%% Load data

# recording_to_sort = 'E:\\pipeline_testing_data\\M1_D31_2018-11-01_12-28-25_short'
recording_to_sort = '/mnt/e/pipeline_testing_data/M1_D31_2018-11-01_12-28-25_short'
tetrode_geom = 'sorting_files/geom_all_tetrodes_original.csv'
signal = file_utility.load_OpenEphysRecording(recording_to_sort)
geom = pd.read_csv(tetrode_geom,header=None).values
Fs = 30000
recording = se.NumpyRecordingExtractor(signal,Fs,geom)

#%%
# register the recording in the kachery database
print('I will now convert the recording to MDA')

def register_recording(*, recording, output_fname, label):
    with hither.TemporaryDirectory() as tmpdir:
        recdir = tmpdir + '/recording'
        MdaRecordingExtractor.write_recording(recording=recording, save_path=recdir) # write recording as MDA file
        raw_path = ka.store_file(recdir + '/raw.mda') # store the MDA file in kachery
        obj = dict(
            raw=raw_path,
            params=ka.load_object(recdir + '/params.json'),
            geom=np.genfromtxt(ka.load_file(recdir + '/geom.csv'), delimiter=',').tolist()
        )

        obj['params']['spike_sign'] = -1 #for tridesclous
        obj['self_reference'] = ka.store_object(obj, basename='{}.json'.format(label))
        with open(output_fname, 'w') as f:
            json.dump(obj, f, indent=4)

        return obj

recording_info = register_recording(
    recording=recording,
    output_fname='new_recording.json',
    label='new_recording',
)

recording_path = recording_info['self_reference']

print(f'Recording file saved succesffully. Address: {recording_path}')

#%% Perform sorting using spikeforest
print("Sorting with spikeforest")
sorter_result={}

with open('new_recording.json','r') as f:
    recording_path = json.load(f)['self_reference']

#%% herding spikes
with ka.config(fr='default_readonly'):
    #with hither.config(cache='default_readwrite'):
        with hither.config(container='default'):

            result= sorters.herdingspikes2.run(
                recording_path=recording_path,
                sorting_out=hither.File()
            )

sorter_result['herding'] = AutoSortingExtractor(result.outputs.sorting_out._path)

#%% mountainsort
with ka.config(fr='default_readonly'):
    with hither.config(container='default'):

        result = sorters.mountainsort4.run(
            recording_path=recording_path,
            sorting_out=hither.File()
        )

sorter_result['mountainsort'] = AutoSortingExtractor(result.outputs.sorting_out._path)

#%% Tridesclous
# with ka.config(fr='default_readonly'):
#     with hither.config(container='docker://teristam/sf-tridesclous:1.6.0'):
#         result = sorters.tridesclous.run(
#             recording_path=recording_path,
#             sorting_out=hither.File()
#         )

# sorter_result['tridesclous'] = AutoSortingExtractor(result.outputs.sorting_out._path)

#%% klusta
with ka.config(fr='default_readonly'):
    with hither.config(container='default'):
        result = sorters.klusta.run(
            recording_path=recording_path,
            sorting_out=hither.File()
        )

sorter_result['klusta'] = AutoSortingExtractor(result.outputs.sorting_out._path)

#%% ironclust
with ka.config(fr='default_readonly'):
    with hither.config(container='default'):
        result = sorters.ironclust.run(
            recording_path=recording_path,
            sorting_out=hither.File()
        )

sorter_result['ironclust'] = AutoSortingExtractor(result.outputs.sorting_out._path)


#%% Kilosort2
with ka.config(fr='default_readonly'):
    with hither.config(container='default', gpu=True):
        result = sorters.kilosort2.run(
            recording_path=recording_path,
            sorting_out=hither.File()
        )

sorter_result['kilsort2'] = AutoSortingExtractor(result.outputs.sorting_out._path)


#%% spyking
with ka.config(fr='default_readonly'):
    with hither.config(container='default'):
        result= sorters.spykingcircus.run(
            recording_path=recording_path,
            sorting_out=hither.File()
        )

sorter_result['spyking'] = AutoSortingExtractor(result.outputs.sorting_out._path)

#%% save sorter results
# with open('sorter_results.pkl','wb') as f:
#     pickle.dump(sorter_result,f)

#%%
with open('sorter_results.pkl','rb') as f:
    sorter_result = pickle.load(f)

#%% Create extractor form sorting results

# construct sorting extractor for each sorting results


sortingExtractor = AutoSortingExtractor(result.outputs.sorting_out._path)
print(sortingExtractor.get_unit_ids())
print(sortingExtractor.get_unit_spike_train(0))

# %%
