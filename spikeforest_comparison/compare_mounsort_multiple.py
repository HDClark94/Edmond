'''
Run the sorting multiplle times and compare the results

'''

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
import seaborn as sns
#%% Load data

recording_to_sort = 'E:\\pipeline_testing_data\\M1_D31_2018-11-01_12-28-25_short'
# recording_to_sort = '/mnt/e/pipeline_testing_data/M1_D31_2018-11-01_12-28-25_short'
tetrode_geom = 'sorting_files/geom_all_tetrodes_original.csv'
signal = file_utility.load_OpenEphysRecording(recording_to_sort)
geom = pd.read_csv(tetrode_geom,header=None).values
Fs = 30000


recording=se.NumpyRecordingExtractor(signal[:,:10*60*Fs],Fs,geom)

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

recording_info = register_recording(
    recording=recording,
    output_fname=f'recording.json',
    label=f'recording',
)

# recording_path = recording_info['self_reference']

# print(f'Recording file saved succesffully. Address: {recording_path}')
# %%
sorter_results = []
for i in range(4):
    with open(f'recording.json','r') as f:
        recording_path = json.load(f)['self_reference']

    with ka.config(fr='default_readonly'):
        with hither.config(container='default'):

            result = sorters.mountainsort4.run(
                recording_path=recording_path,
                sorting_out=hither.File()
            )

    sorter_results.append(AutoSortingExtractor(result.outputs.sorting_out._path))

# %%
with open('sorter_multiple_sort_compare.pkl','wb') as f:
    pickle.dump(sorter_results, f)

#%% only compare the first 2 minutes of data
Fs = 30000

# %%
comparisons=[]
label =['1st','2nd','3rd','4th']

for i in range(1,len(label)):
    cmp = sc.compare_two_sorters(sorting1=sorter_results[0], sorting2=sorter_results[i],
                                                sorting1_name=label[0], sorting2_name=label[i],sampling_frequency=30000)

    comparisons.append(cmp)

#%%
plt.rc('font', size=10)
plt.rc('figure', titlesize=30)

fig,ax = plt.subplots(1,3,figsize=(10*3,10*1))

for i,c in enumerate(comparisons):
    sw.plot_agreement_matrix(c,ax=ax.flat[i])
    ax.flat[i].set_title(f'Num cells in x-axis: {len(sorter_results[i].get_unit_ids())} ')

fig.tight_layout()
plt.savefig('comparison_multiple_compare.png')


# %% Find the metrics of the sorting

metrics_list = st.validation.get_quality_metrics_list() # compute all available metrics


from ShadowExtractor import ShadowSortingExtractor #temp fix to upstream bug

sorter_ms4s = ShadowSortingExtractor(original_sorter=sorter_results[0])

quality_metrics = st.validation.compute_quality_metrics(sorter_ms4s, recording,
     metric_names=metrics_list, as_dataframe=True, n_jobs=4, verbose=True,recompute_info=True)

quality_metrics.to_pickle('quality_metrics_multiple_sort.pkl')

# %% calculate the mean agreemenet score
scores = []
for c in comparisons:
    score = np.max(c.agreement_scores, axis=1)
    scores.append(score)

scores = np.stack(scores).T.mean(axis=1)
quality_metrics['agreement_score'] = scores
quality_metrics['cluster_id'] = quality_metrics.index

#%% plot the agreement score with cluster metrics
plt.rc('font',size=12)
qm2plot = quality_metrics.melt(id_vars=['cluster_id','agreement_score'])
g = sns.relplot(x='value',y='agreement_score',col_wrap=4,col='variable',
    hue='cluster_id', style='cluster_id',legend = 'full',height=3,facet_kws={'sharex': False}, data=qm2plot)

leg = g._legend
leg.set_bbox_to_anchor([1.1, 0.5])  # coordinates of lower left of bounding box
plt.tight_layout()
plt.savefig('figures/metrics_multiple_sort.pdf',dpi=100, bbox_inches = "tight")

# %%
