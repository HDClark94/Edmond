import pandas as pd
import os
import traceback
import sys

def rewrite_vr_param_file(vr_paths, of_paths):

    vr_recording_paths = [f.path for f in os.scandir(vr_paths) if f.is_dir()]
    of_recording_paths = [f.path for f in os.scandir(of_paths) if f.is_dir()]

    for vr_recording_path in vr_recording_paths:

        id = vr_recording_path.split("/")
        of_path = of_paths+"/"+id[-1]
        mouse_and_day = of_path.split("_20")[0].split("/")[-1] + "_20"
        try:
            matching_of_path = list(filter(lambda x: mouse_and_day in x, of_recording_paths))[0]
            matching_of_path_from_person = matching_of_path.split("/mnt/datastore/")[1]

            path_from_person = vr_recording_path.split("/mnt/datastore/")[1]
            param_path = vr_recording_path + '/parameters.txt'

            with open(param_path, 'r') as file:
                # read a list of lines into data
                data = file.readlines()

                data[0] = data[0] # no change here
                data[1] = path_from_person+"\n"
                #data[1] = data[1].replace("\\", "/")

                if len(data[2].split("*")) > 1:
                    stop_threshold = data[2].split("*stop_threshold=")[-1]
                else:
                    stop_threshold = data[2]

                data[2] = "paired="+matching_of_path_from_person+"*session_type_paired=openfield*stop_threshold="+stop_threshold
                #data[2] = data[2].replace("\\", "/")

                # and write everything back
                with open(param_path, 'w') as file:
                    file.writelines(data)

            #print("Success with ", vr_recording_path)
            print(vr_recording_path.split("/mnt/datastore/")[-1])
        except:
            a=1
            #print("Failed with ", vr_recording_path)

def rewrite_of_param_file(of_paths):
    of_recording_paths = [f.path for f in os.scandir(of_paths) if f.is_dir()]

    for of_recording_path in of_recording_paths:
        try:
            of_path_from_person = of_recording_path.split("/mnt/datastore/")[1]
            param_path = of_recording_path + '/parameters.txt'

            with open(param_path, 'r') as file:
                # read a list of lines into data
                data = file.readlines()

                data[0] = data[0] # no change here
                data[1] = of_path_from_person+"\n"
                #data[1] = data[1].replace("\\", "/")

                # and write everything back
                with open(param_path, 'w') as file:
                    file.writelines(data)

            print("Success with ", of_recording_path)
            #print(of_recording_path.split("/mnt/datastore/")[-1])
        except Exception as ex:
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("Failed with ", of_recording_path)


def main():

    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    vr_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/VirtualReality/245_sorted"
    of_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort2/OpenField"
    rewrite_of_param_file(of_paths)


if __name__ == '__main__':
    main()