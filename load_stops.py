import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import Mouse_paths


def load_stops(data_frame_path):
    processed_position = pd.read_pickle(data_frame_path)
    print("hello there")

def main():

    print('-------------------------------------------------------------')

    server_path = r"Z:\ActiveProjects\Harry\2019cohort1\vr\M1_D13_2019-11-27_13-34-37\MountainSort\DataFrames\processed_position_data.pkl"
    load_stops(server_path)

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()