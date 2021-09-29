import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil

def copy_MountainSort_folder(recordings_folder_path, paste_path):

    if os.path.exists(recordings_folder_path):
        recordings = [f.path for f in os.scandir(recordings_folder_path) if f.is_dir()]

        for recording_folder in recordings:
            folder = recording_folder.split("\\")[6]
            if folder.startswith("M"):
                MS_path = recording_folder+"\MountainSort"
                if os.path.exists(recording_folder+"\MountainSort"):

                    new_paste_path = paste_path+"/"+folder
                    if os.path.exists(new_paste_path) is False:
                        os.makedirs(new_paste_path)

                        MS_paste_path = new_paste_path+"\MountainSort"
                        if os.path.exists(MS_paste_path) is False:

                            if os.path.exists(MS_path+"\Figures"):
                                print("I am copying a MountainSort Folder to a new location")
                                shutil.copytree(MS_path+"\Figures", MS_paste_path)

def main():

    print('-------------------------------------------------------------')

    server_path = "Z:\ActiveProjects\Harry\MouseVR\data\Cue_conditioned_cohort1_190902"
    paste_path = server_path+"\Sorted_Seperately"
    copy_MountainSort_folder(server_path, paste_path)


    server_path = "Z:\ActiveProjects\Harry\MouseOF\data\Cue_conditioned_cohort1_190902"
    paste_path = server_path+"\Sorted_Seperately"
    copy_MountainSort_folder(server_path, paste_path)



    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()
