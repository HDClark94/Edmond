#%%


'''
- Inject different level of noise in the second half trying to simulate motion artifact in openfield recording

'''


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
from tqdm import tqdm
from utils import addNoise

#%% Load data

recording_to_sort = 'E:\\pipeline_testing_data\\M1_D31_2018-11-01_12-28-25_short'
# recording_to_sort = '/mnt/e/pipeline_testing_data/M1_D31_2018-11-01_12-28-25_short'
tetrode_geom = 'sorting_files/geom_all_tetrodes_original.csv'
signal = file_utility.load_OpenEphysRecording(recording_to_sort)
geom = pd.read_csv(tetrode_geom,header=None).values
Fs = 30000

signal = signal[:,:5*60*Fs]

#%% create recordings
recordings=[]
params = [0,0.2,0.6,0.8]
for noise_ratio in params:
    signaln = addNoise(signal,noise_ratio==noise_ratio)
    recordings.append(se.NumpyRecordingExtractor(signaln,Fs,geom))

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

        obj['self_reference'] = ka.store_object(obj, basename='{}.json'.format(label))
        with open(output_fname, 'w') as f:
            json.dump(obj, f, indent=4)

        return obj

recording_infos = []

for i,recording in enumerate(recordings):
    recording_info = register_recording(
        recording=recording,
        output_fname=f'recording_noise_{i}.json',
        label=f'recording_{i}',
    )

# %% perform the sorting
sorter_results = []
for i in range(len(recordings)):
    with open(f'recording_noise_{i}.json','r') as f:
        recording_path = json.load(f)['self_reference']

    with ka.config(fr='default_readonly'):
        with hither.config(container='default'):

            result = sorters.mountainsort4.run(
                recording_path=recording_path,
                sorting_out=hither.File()
            )

    sorter_results.append(AutoSortingExtractor(result.outputs.sorting_out._path))

# %%
with open('sorter_noise_compare.pkl','wb') as f:
    pickle.dump(sorter_results, f)


#%% Load sorting data
with open('sorter_noise_compare.pkl','rb') as f:
    sorters_list = pickle.load(f)

#%% only compare the first half (5min) of data

sorters_crop =[]
Fs = 30000

for s in sorters_list:
    s.add_epoch('first_section',0,2.5*60*Fs)
    sorters_crop.append(s.get_epoch('first_section'))

# %%
comparisons=[]
label =[f'noiseRatio = {i}' for i in params]

for i in range(len(label)):
    cmp = sc.compare_two_sorters(sorting1=sorters_crop[0], sorting2=sorters_crop[i],
                                                sorting1_name=label[0], sorting2_name=label[i],sampling_frequency=30000)

    comparisons.append(cmp)

#%%
fig,ax = plt.subplots(2,2,figsize=(10*2,10*2))

for i,c in enumerate(comparisons):
    sw.plot_agreement_matrix(c,ax=ax.flat[i])
    ax.flat[i].set_title(f'Num cells in x: {len(sorters_crop[i].get_unit_ids())} y: {len(sorters_crop[0].get_unit_ids())}')

fig.tight_layout()

plt.savefig('comparison_noise_compare.png')

# %%
