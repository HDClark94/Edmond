import pandas as pd
import os
import OpenEphys
import numpy as np

'''
This script enables allows you to pass the path of a recording and specify how much to clip the recording by, 
this should only be used when there is a issue with the recording and you know it is corrupted. This will permanently
delete data off the server. Only use on recordings you absolutely know need fixing
'''
def clip_recording(broken_recording, clip_end_sampling_points=None, clip_start_sampling_points=None):

    if os.path.exists(broken_recording):
        continuous_files = [f for f in os.listdir(broken_recording) if f.endswith(".continuous")]
        for i in range(len(continuous_files)):

            ch = OpenEphys.loadContinuous(broken_recording+"/"+continuous_files[i])

            if clip_end_sampling_points is not None:
                ch['data'] = ch['data'][:clip_end_sampling_points]

            elif clip_start_sampling_points is not None:
                ch['data'] = ch['data'][clip_start_sampling_points:]

            OpenEphys.writeContinuousFile(broken_recording+"/"+continuous_files[i], ch['header'], ch['timestamps'], ch['data'], ch['recordingNumber'])



def print_recording_lengths(recording):
    if os.path.exists(recording):
        continuous_files = [f for f in os.listdir(recording) if f.endswith(".continuous")]
        ch = OpenEphys.loadContinuous(recording+"/"+continuous_files[0])
        print(len(ch["data"]))
        return len(ch["data"])

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    broken_recording = "/mnt/datastore/Harry/Cohort8_may2021/of/M11_D44_2021-07-08_11-24-55"
    #clip_recording(broken_recording, clip_end_sampling_points=54032384, clip_start_sampling_points=None)

    broken_recordings = ["/mnt/datastore/Harry/Cohort9_february2023/tmp/M16_D1_2023-02-28_17-42-27", "/mnt/datastore/Harry/Cohort9_february2023/tmp/M16_D2_2023-03-01_17-12-03",
                         "/mnt/datastore/Harry/Cohort9_february2023/tmp/M16_D3_2023-03-02_16-18-48", "/mnt/datastore/Harry/Cohort9_february2023/tmp/M16_D4_2023-03-03_16-47-29"]
    tmp_recordings = ["/mnt/datastore/Harry/Cohort9_february2023/tmp/1",
                         "/mnt/datastore/Harry/Cohort9_february2023/tmp/2",
                         "/mnt/datastore/Harry/Cohort9_february2023/tmp/3",
                         "/mnt/datastore/Harry/Cohort9_february2023/tmp/4"]

    #for broken_recording, tmp_recording in zip(broken_recordings, tmp_recordings):
    #    length = print_recording_lengths(broken_recording)
    #    clip_recording(tmp_recording, clip_end_sampling_points=length, clip_start_sampling_points=None)

if __name__ == '__main__':
    main()