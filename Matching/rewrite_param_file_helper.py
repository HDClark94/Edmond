import pandas as pd
import os

def rewrite_param_file(vr_paths, of_paths):

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

            print("Success with ", vr_recording_path)
        except:
            a=1
            #print("Failed with ", vr_recording_path)

def main():

    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #vr_paths = r"Z:\ActiveProjects\Harry\Mouse_data_for_sarah_paper\_cohort4\VirtualReality\M2_sorted"
    #of_paths = r"Z:\ActiveProjects\Harry\Mouse_data_for_sarah_paper\_cohort4\OpenFeild"
    #rewrite_param_file(vr_paths, of_paths)
    #vr_paths = r"Z:\ActiveProjects\Harry\Mouse_data_for_sarah_paper\_cohort4\VirtualReality\M3_sorted"
    #of_paths = r"Z:\ActiveProjects\Harry\Mouse_data_for_sarah_paper\_cohort4\OpenFeild"
    #rewrite_param_file(vr_paths, of_paths)

    #vr_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M1_sorted"
    #of_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild"
    #rewrite_param_file(vr_paths, of_paths)

    #vr_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M2"
    #of_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild"
    #rewrite_param_file(vr_paths, of_paths)

    vr_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M3"
    of_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild"
    rewrite_param_file(vr_paths, of_paths)

    #vr_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M4"
    #of_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild"
    #rewrite_param_file(vr_paths, of_paths)

    #vr_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/VirtualReality/M6_sorted"
    #of_paths = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort3/OpenFeild"
    #rewrite_param_file(vr_paths, of_paths)


if __name__ == '__main__':
    main()