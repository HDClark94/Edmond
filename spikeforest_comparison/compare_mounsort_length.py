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

#%% Load data

recording_to_sort = 'E:\\pipeline_testing_data\\M1_D31_2018-11-01_12-28-25_short'
# recording_to_sort = '/mnt/e/pipeline_testing_data/M1_D31_2018-11-01_12-28-25_short'
tetrode_geom = 'sorting_files/geom_all_tetrodes_original.csv'
signal = file_utility.load_OpenEphysRecording(recording_to_sort)
geom = pd.read_csv(tetrode_geom,header=None).values
Fs = 30000

recordings = []

for i in [2,4,6,8,10]:
    recordings.append(se.NumpyRecordingExtractor(signal[:,:i*60*Fs],Fs,geom))



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

recording_infos = []

for i,recording in enumerate(recordings):
    recording_info = register_recording(
        recording=recording,
        output_fname=f'recording_{i}.json',
        label=f'recording_{i}',
    )


# %% Perform sorting and extract the results

sorter_results = []
for i in range(5):
    with open(f'recording_{i}.json','r') as f:
        recording_path = json.load(f)['self_reference']

    with ka.config(fr='default_readonly'):
        with hither.config(container='default'):

            result = sorters.mountainsort4.run(
                recording_path=recording_path,
                sorting_out=hither.File()
            )

    sorter_results.append(AutoSortingExtractor(result.outputs.sorting_out._path))

# %%
with open('sorter_length_compare.pkl','wb') as f:
    pickle.dump(sorter_results, f)
# %%
