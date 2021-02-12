#This file contains the parameters for analysis


##########
# Folder locations
basefolder = '/Users/teristam/Documents/Data'
mountainsort_tmp_folder = basefolder+'/tmp/mountainlab/'
sorting_folder = basefolder+'/to_sort/recordings/'
to_sort_folder = basefolder+'/to_sort/'
output_folder = basefolder +'/output'
server_path_first_half = basefolder+'/log/'
downtime_lists_path = basefolder+'/to_sort/sort_downtime/'

##########
# Recording setting
sampling_rate = 30000
num_tetrodes = 4
movement_ch = '100_ADC2.continuous' #channel that contains the movement data
opto_ch = '100_ADC3.continuous'
data_file_prefix = '100_CH' #prefix of data files
data_file_suffix = '' 
wave_form_size = 40 

#########
# sorter configuration
sorterName = 'mountainsort4'
is_tetrode_by_tetrode = False #set to True if you want the spike sorting to be done tetrode by tetrode
all_tetrode_together = True #set to True if you want the spike sorting done on all tetrodes combined


############
# Analysis
spike_bin_size = 20 #the bin size to group spike together to calculate spike count, in ms
location_bin_num = 200 #number of location bin
stop_threshold = 4.7 #threshold for detecting stop
location_ds_rate = 1000 #the sampleing frequency in Hz to downsample the location signal to 

##########
# VR
track_length = 200
first_trial_channel = '100_ADC4.continuous' #channel for the start of trial
second_trial_channel = '100_ADC5.continuous' #channel for the stp of trial
reward_start = 88 #position for the reward
reward_end = 110 #position for the reward

##########
# Experiment
session_type = 'vr'


##########
# open field
opto_tagging_start_index = None
pixel_ratio = 440
sync_channel = '100_ADC1.continuous' #channel for the sync pulse
bonsai_sampling_rate = 30

############
# Debug
# debug_folder = 'testData/M6_2018-03-06_16-10-00_of' #recording for debug purpose
# debug_folder ='/media/data2/pipeline_testing_data/M1_D31_2018-11-01_12-28-25'
debug_folder ='/home/ubuntu/to_sort/recordings/M5_2018-03-06_15-34-44_of'

