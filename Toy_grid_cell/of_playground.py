import matplotlib.pyplot as plt
import numpy as np


def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array



def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    save_path = "/mnt/datastore/Harry/Vr_grid_cells"

    plot_cue_anchored_grid_cells_spatial_autocorreologram_anchored(n_trials=n_trials, save_path=save_path, p_scalar=p_scalar)

    print("look now")

if __name__ == '__main__':
    main()
