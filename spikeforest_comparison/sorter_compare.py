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
from matplotlib import pylab as plt
import seaborn as sns

#%%
with open('sorter_results.pkl','rb') as f:
    sorter_result = pickle.load(f)

sorter_ms4 = sorter_result['mountainsort']
sorter_spyking = sorter_result['spyking']
sorter_herding = sorter_result['herding']
sorter_klusta = sorter_result['klusta']
sorter_ironclust = sorter_result['ironclust']
sorter_kilsort2 = sorter_result['kilsort2']

# %%

cmp_ms4_spyking = sc.compare_two_sorters(sorting1=sorter_ms4, sorting2=sorter_spyking,
                                               sorting1_name='ms4', sorting2_name='spyking',sampling_frequency=30000)

cmp_ms4_herding = sc.compare_two_sorters(sorting1=sorter_ms4, sorting2=sorter_herding,
                                               sorting1_name='ms4', sorting2_name='herding',sampling_frequency=30000)

cmp_ms4_klusta = sc.compare_two_sorters(sorting1=sorter_ms4, sorting2=sorter_klusta,
                                               sorting1_name='ms4', sorting2_name='klusta',sampling_frequency=30000)

cmp_ms4_ironclust = sc.compare_two_sorters(sorting1=sorter_ms4, sorting2=sorter_ironclust,
                                               sorting1_name='ms4', sorting2_name='ironclust',sampling_frequency=30000)

cmp_ms4_kilosort2 = sc.compare_two_sorters(sorting1=sorter_ms4, sorting2=sorter_kilsort2,
                                               sorting1_name='ms4', sorting2_name='kilsort2',sampling_frequency=30000) 

#%%

#%%

sorter_name = ['spyking','herding','klusta','ironclust','kilosort2']
sorter = [cmp_ms4_spyking, cmp_ms4_herding, cmp_ms4_klusta, cmp_ms4_ironclust, cmp_ms4_kilosort2]

for ss_name, ss_cm in zip(sorter_name, sorter):
    fig,ax = plt.subplots(1,1,figsize=(12,12), dpi=100)
    sw.plot_agreement_matrix(ss_cm,ax=ax)
    plt.savefig(f'figures/ms4_vs_{ss_name}.png')



# %%
fig,ax = plt.subplots(1,2,figsize=(10*2,10))
sw.plot_agreement_matrix(cmp_ms4_klusta,ax=ax[0])
sw.plot_agreement_matrix(cmp_ms4_ironclust,ax=ax[1])
plt.savefig('comparison2.png')

# %% compare the consistency of sorters with some cluster metrics

# Load recording
recording_to_sort = 'E:\\pipeline_testing_data\\M1_D31_2018-11-01_12-28-25_short'
# recording_to_sort = '/mnt/e/pipeline_testing_data/M1_D31_2018-11-01_12-28-25_short'
tetrode_geom = 'sorting_files/geom_all_tetrodes_original.csv'
signal = file_utility.load_OpenEphysRecording(recording_to_sort)
geom = pd.read_csv(tetrode_geom,header=None).values
Fs = 30000
recording = se.NumpyRecordingExtractor(signal,Fs,geom)
recording= se.CacheRecordingExtractor(recording)

# %% Get some of the sorting properties of the clusters

metrics_list = st.validation.get_quality_metrics_list() # compute all available metrics


from ShadowExtractor import ShadowSortingExtractor #temp fix to upstream bug

sorter_ms4s = ShadowSortingExtractor(original_sorter=sorter_ms4)

quality_metrics = st.validation.compute_quality_metrics(sorter_ms4s, recording,
     metric_names=metrics_list, as_dataframe=True, n_jobs=4, verbose=True,recompute_info=True)

quality_metrics.to_pickle('ms4_quality_metrics.pkl')

# %% Get the agreement score of all sorters
Fs = 30000
quality_metrics = pd.read_pickle('ms4_quality_metrics.pkl')
sorter_list = list(sorter_result.keys())
mcmp = sc.compare_multiple_sorters(sorting_list=list(sorter_result.values()),
                                   name_list=sorter_list, verbose=True, sampling_frequency=Fs)
# check which metrics best predict consistency

# %%

# Get the average max agreement for all sorters
sorter_agreement={}
for c in mcmp.comparisons:
    if 'mountainsort' in c.name_list:
        ms_idx = c.name_list.index('mountainsort')
        if ms_idx ==0:
            # ms result should in rows
            agreement_score = c.agreement_scores
        else:
            agreement_score = c.agreement_scores.T

        sorter_agreement[c.name_list[1-ms_idx]] = agreement_score.max(axis=1) #store the max agreement


df_agree = pd.DataFrame(sorter_agreement)
df_agree_mean = df_agree.mean(axis=1)

# add the agreement score to metrics
quality_metrics['agreement_score'] = df_agree_mean
quality_metrics['cluster_id'] = quality_metrics.index
qm2plot = quality_metrics.melt(id_vars=['cluster_id','agreement_score'])
# %% Plot 
g = sns.relplot(x='value',y='agreement_score',col_wrap=4,col='variable',
    hue='cluster_id', style='cluster_id',legend = 'full',height=3,facet_kws={'sharex': False}, data=qm2plot)

leg = g._legend
leg.set_bbox_to_anchor([1.1, 0.5])  # coordinates of lower left of bounding box
# leg._loc = 4  # if required you can set the 

plt.tight_layout()
plt.savefig('figures/metrics_agreements_score.pdf',dpi=100, bbox_inches = "tight")
# %%
