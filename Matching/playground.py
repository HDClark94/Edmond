import pandas as pd

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    path = r"Z:\\ActiveProjects\\Harry\\Mouse_data_for_sarah_paper\\_cohort5\\OpenField\\M1_D1_2019-06-17_13-54-35\\MountainSort\\DataFrames\\spatial_firing.pkl"
    spatial_firing = pd.read_pickle(path)
    print("look now")

if __name__ == '__main__':
    main()