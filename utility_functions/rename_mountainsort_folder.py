import os

def rename_mountainsort(folder_path, rename_this, rename_to_this):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for name in dirnames:
            if (rename_this == name):
                print("renaming "+dirpath+"/"+rename_this)

                os.rename(dirpath+"/"+rename_this, dirpath+"/"+rename_to_this)

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    path_to_folder = "/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort4/VirtualReality/"
    rename_mountainsort(path_to_folder, rename_this="MountainSort", rename_to_this="MountainSort_sorted_together")

    path_to_folder = "/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort3/VirtualReality/"
    rename_mountainsort(path_to_folder, rename_this="MountainSort", rename_to_this="MountainSort_sorted_together")

    path_to_folder = "/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort2/VirtualReality/"
    rename_mountainsort(path_to_folder, rename_this="MountainSort", rename_to_this="MountainSort_sorted_together")

if __name__ == '__main__':
    main()