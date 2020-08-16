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
import open_ephys_IO


from scipy.signal import butter,filtfilt

def main():
    recording_to_sort = setting.server_path_first_half+"Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M1_sorted/M1_D8_2019-06-26_13-31-11"
    #a = pd.read_pickle(recording_to_sort+"/processed/klusta_ofsingle_df.pkl")
    print("hello there")



    recording_to_sort = setting.server_path_first_half+"Harry/Cohort6_july2020/vr/M1_D5_2020-08-07_14-27-26"
    #recording_to_sort = setting.server_path_first_half+"Harry/Cohort6_july2020/vr/M1_D1_2020-08-03_16-11-14vr"
    prm=None

    for ADC_string in ["ADC6", "ADC7", "ADC8"]:
        ADC = open_ephys_IO.get_data_continuous(prm, recording_to_sort+"/100_"+ADC_string+".continuous")
        plt.plot(ADC)
        plt.savefig(recording_to_sort + '/Figures/'+ADC_string+'.png')
        plt.close()


    # ADC5 is the movement channel for this recording.


if __name__ == '__main__':
    main()