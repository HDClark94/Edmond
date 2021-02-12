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
from ShadowExtractor import ShadowSortingExtractor

import seaborn as sns 

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
params = [2,1,0.1,0.01]
for snr in params:
    signaln = addNoise(signal,snr=snr)
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
        output_fname=f'recording_noise_snr_{i}.json',
        label=f'recording_{i}',
    )

# %% perform the sorting
sorter_results = []
for i in range(len(recordings)):
    with open(f'recording_noise_snr_{i}.json','r') as f:
        recording_path = json.load(f)['self_reference']

    with ka.config(fr='default_readonly'):
        with hither.config(container='default'):

            result = sorters.mountainsort4.run(
                recording_path=recording_path,
                sorting_out=hither.File()
            )

    sorter_results.append(AutoSortingExtractor(result.outputs.sorting_out._path))

# %%
with open('sorter_noise_snr_compare.pkl','wb') as f:
    pickle.dump(sorter_results, f)


#%% Load sorting data
with open('sorter_noise_snr_compare.pkl','rb') as f:
    sorters_list = pickle.load(f)

#%% only compare the first half (5min) of data

sorters_crop =[]
Fs = 30000

for s in sorters_list:
    s.add_epoch('first_section',0,2.5*60*Fs)
    sorters_crop.append(s.get_epoch('first_section'))

# %%
comparisons=[]
label =[f'snr = {i}' for i in params]

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

plt.savefig('comparison_noise_snr_compare.png')

# %% #Compute the unit PCA score
pca_scores=[]

for recording, sorting in zip(recordings, sorters_crop):    

    score=st.postprocessing.compute_unit_pca_scores(recording,ShadowSortingExtractor(original_sorter=sorting),
        by_electrode=False, whiten=True,grouping_property='group', 
        compute_property_from_recording = True,
        recompute_info=True, memmap=False, verbose=True)
    pca_scores.append(score)
    print('.')

# %% # Convert the PCA to dataframe for easy plotting later
dfs = []

for sorter_idx in range(len(pca_scores)):
    scores = pca_scores[sorter_idx]
    for cluster_id in range(len(scores)):
        df = pd.DataFrame(scores[cluster_id])
        df['cluster_id']= cluster_id
        df['snr'] = label[sorter_idx]
        dfs.append(df)

df_all = pd.concat(dfs)

sns.relplot(x=0,y=1,hue='cluster_id',style='cluster_id', col='snr',
    legend='full',col_wrap=2,facet_kws={'sharex': False, 'sharey':False},data=df_all)
# %%
fig, ax = plt.subplots()
ax.plot(pca_scores[0][0][:, 0], pca_scores[0][0][:, 1], 'r*')
ax.plot(pca_scores[0][1][:, 0], pca_scores[0][1][:, 1], 'b*')
# %%
w_wf = sw.plot_unit_waveforms(recordings[0], sorters_crop[0], max_spikes_per_unit=100)

#%%
w_feat = sw.plot_pca_features(recordings[0], sorters_crop[0], colormap='rainbow', nproj=3, max_spikes_per_unit=100)

# %%
