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


#%%
with open('sorter_length_compare.pkl','rb') as f:
    sorters_list = pickle.load(f)

#%% only compare the first 2 minutes of data

sorters_crop =[]
Fs = 30000

for s in sorters_list:
    s.add_epoch('first_section',0,2*60*Fs)
    sorters_crop.append(s.get_epoch('first_section'))

# %%
comparisons=[]
label =['2min','4min','6min','8min','10min']

for i in range(len(label)-1):
    cmp = sc.compare_two_sorters(sorting1=sorters_crop[-1], sorting2=sorters_crop[i],
                                                sorting1_name=label[-1], sorting2_name=label[i],sampling_frequency=30000)

    comparisons.append(cmp)

#%%
fig,ax = plt.subplots(2,2,figsize=(10*2,10*2))

for i,c in enumerate(comparisons):
    sw.plot_agreement_matrix(c,ax=ax.flat[i])
    ax.flat[i].set_title(f'Num cells in x-axis: {len(sorters_crop[i].get_unit_ids())} ')

fig.tight_layout()

plt.savefig('comparison_length_compare.png')



