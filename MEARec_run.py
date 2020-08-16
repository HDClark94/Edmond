import Edmond.SpikeInterface_fileutility as file_utility
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
import Edmond.spikeinterfaceHelper as spikeinterfaceHelper
from tqdm import tqdm
import numpy as np
import Edmond.SpikeInterface_setting as setting
import logging
from types import SimpleNamespace
import matplotlib.pylab as plt
from scipy.signal import butter,filtfilt
import MEArec as mr

params = {}

def sort_recording(recording_to_sort, sorter=None, try_dual_sorting=False):
    print("do something")
    # generate templates
    templates_params = mr.get_default_templates_params()
    cell_models_folder = mr.get_default_cell_models_folder()
    templates_params['probe'] = 'tetrode-mea-l'
    templates_params['n'] = 30
    templates_params['seed']=0
    tempgen = mr.gen_templates(cell_models_folder=cell_models_folder,
                              params=templates_params)
    mr.save_template_generator(tempgen, 'path-to-templates-file.h5')

    # generate recordings
    recordings_params = mr.get_default_recordings_params()
    recordings_params['spiketrains']['n_exc']=4
    recordings_params['spiketrains']['n_inh']=2
    recordings_params['spiketrains']['duration'] = 30
    recordings_params['seeds']['spiketrains']=0
    recordings_params['seeds']['templates']=1
    recordings_params['seeds']['noise']=2
    recordings_params['seeds']['convolution']=3
    recgen = mr.gen_recordings(params=recordings_params,
                           templates='path-to-templates-file.h5')
    mr.save_recording_generator(recgen, 'path-to-recordings-file.h5')

    recgen = mr.gen_recordings(params=params,
                               templates=None,
                               tempgen=None,
                               n_jobs=None,
                               verbose=False)

def test():
    import MEArec as mr
    import os

    cell_folder = mr.get_default_cell_models_folder()
    params = mr.get_default_templates_params()

    target_spikes = [3, 50]
    params['target_spikes'] = target_spikes
    cells = os.listdir(cell_folder)
    cell_name = [c for c in cells if 'TTPC1' in c][0]
    cell_path = os.path.join(cell_folder, cell_name)

    cell, v, i = mr.run_cell_model(cell_model_folder=cell_path, sim_folder=None, verbose=True, save=False, return_vi=True, **params)

    print(i.shape)

    import matplotlib.pyplot as plt
    plt.plot(i[0,0])
    plt.show()

def main():

    recording_to_sort = setting.server_path_first_half+"Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted/M1_D8_2019-06-26_13-31-11"

    test()
    sort_recording(recording_to_sort, sorter="klusta", try_dual_sorting=True)
    print("hello there")

if __name__ == '__main__':
    main()