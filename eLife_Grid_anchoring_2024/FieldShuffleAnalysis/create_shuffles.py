from pathlib import Path
import subprocess
import sys
import os
import pandas as pd

recording_folder_to_process = sys.argv[1] # something like /mnt/datastore/Harry/test_recordings or /mnt/datastore/Harry/Cohort7_october2020/of

N_SHUFFLES = 1000
local_scratch_path = '/exports/eddie/scratch/s1228823/recordings'
ELEANOR_HOST = 'ubuntu@172.16.49.217'
RUN_SHUFFLE_PATH = '/home/s1228823/Edmond/VR_grid_analysis/FieldShuffleAnalysis/run_shuffle.sh'
ELEANOR_RECORDINGS_PATH = Path(recording_folder_to_process)
LOCAL_SCRATCH_PATH = Path(local_scratch_path)

# copy the recording file structure and the pickled dataframes
print(f'Copying spatial_firing.pkl and position.pkl from Datastore (via Eleanor) to Eddie Scratch')
subprocess.check_call(f'rsync -avP --include "*/" --include="*.pkl" --exclude="*" {ELEANOR_HOST}:{ELEANOR_RECORDINGS_PATH} {LOCAL_SCRATCH_PATH}', shell=True)
print(f'Recordings copied... Submitting shuffle jobs to Eddie')

# get the recording names off the scratch space
recordings_scratch_path = local_scratch_path+"/"+recording_folder_to_process.split("/")[-1]
recording_list = [f.path for f in os.scandir(recordings_scratch_path) if f.is_dir()]

n_jobs_submitted = 0
for recording_path in recording_list: # eg. M1_D1_2020-01-31_00-00-00
    print("processing ", recording_path)

    if os.path.isfile(recording_path + "/MountainSort/DataFrames/spatial_firing.pkl"):
        spatial_firing = pd.read_pickle(recording_path + "/MountainSort/DataFrames/spatial_firing.pkl")
        print("I have found ", len(spatial_firing), "cells")

        for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):

            if os.path.isfile(recording_path + "/MountainSort/DataFrames/shuffles/"+str(int(cluster_id))+"_shuffle.pkl"):
                print("shuffle dataframe found, I will only submit a job if more shuffles are required")
                shuffle = pd.read_pickle(recording_path + "/MountainSort/DataFrames/shuffles/"+str(int(cluster_id))+"_shuffle.pkl")
                print("I have found ", len(shuffle), " shuffles")

                #only submit a job if the shuffle dataframe is incomplete
                if (N_SHUFFLES - len(shuffle)) > 0:
                    print(f'Submitting shuffle job')
                    cmd = f'qsub -v RECORDING_PATH={recording_path} -v SHUFFLE_NUMBER={N_SHUFFLES} -v CLUSTER_ID={cluster_id} {RUN_SHUFFLE_PATH}'
                    subprocess.check_call(cmd, shell=True)
                    n_jobs_submitted += 1
                else:
                    print("No job was submitted because shuffle.pkl is complete")

            else:
                print("shuffle dataframe not found, I will submit a new shuffle job")
                cmd = f'qsub -v RECORDING_PATH={recording_path} -v SHUFFLE_NUMBER={N_SHUFFLES} -v CLUSTER_ID={cluster_id} {RUN_SHUFFLE_PATH}'
                subprocess.check_call(cmd, shell=True)
                n_jobs_submitted += 1


print("A total of ", n_jobs_submitted, " have been submitted, good luck!")