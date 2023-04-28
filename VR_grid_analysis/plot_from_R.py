import numpy as np
import pandas as pd
import pickle
import shutil
from statsmodels.stats.multitest import fdrcorrection as fdrcorrection
import Edmond.VR_grid_analysis.analysis_settings as Settings
from matplotlib.markers import TICKDOWN
import PostSorting.parameters
from astropy.nddata import block_reduce
from scipy.signal import correlate
from matplotlib import colors
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_spatial_data
from scipy import interpolate
from scipy import stats
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from Edmond.VR_grid_analysis.hit_miss_try_firing_analysis import hmt2collumn
import matplotlib.patches as patches
import matplotlib.colors as colors
from sklearn.linear_model import LinearRegression
from PostSorting.vr_spatial_firing import bin_fr_in_space, bin_fr_in_time, add_position_x
from scipy import signal
from astropy.convolution import convolve, Gaussian1DKernel
import os
import traceback
import warnings
import matplotlib.ticker as ticker
import sys
import scipy
import Edmond.plot_utility2
import Edmond.VR_grid_analysis.hit_miss_try_firing_analysis
from Edmond.VR_grid_analysis.vr_grid_cells import *
import settings
import matplotlib.pylab as plt
import matplotlib as mpl
import control_sorting_analysis
import PostSorting.post_process_sorted_data_vr
from Edmond.utility_functions.array_manipulations import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
warnings.filterwarnings('ignore')
from scipy.stats.stats import pearsonr
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from Edmond.VR_grid_analysis.hit_miss_try_firing_analysis import significance_bar, get_p_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg


def plot_coefs(coef_df, save_path):

    fig, ax = plt.subplots(figsize=(6,6))

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xticks([1,2,4,5])
    ax.set_xticklabels(["B", "B", "NB", "NB"])
    ax.set_xlim(left=0, right=6)
    ax.set_ylim(bottom=0, top=100)
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.25, bottom=0.2)
    #ax.set_ylabel("% hit trials", fontsize=20)
    plt.savefig(save_path + '.png', dpi=300)
    plt.close()

def main():
    print('-------------------------------------------------------------')
    print("hello")
    beaconed_coefs = pd.read_csv("/mnt/datastore/Harry/Vr_grid_cells/beaconed_coefs.csv")
    non_beaconed_coefs = pd.read_csv("/mnt/datastore/Harry/Vr_grid_cells/non_beaconed_coefs.csv")
    probe_coefs = pd.read_csv("/mnt/datastore/Harry/Vr_grid_cells/probe_coefs.csv")

    print("hello")

    plot_coefs(beaconed_coefs, save_path ="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/fast_hit_analysis/beaconed_coefs")
    plot_coefs(beaconed_coefs, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/fast_hit_analysis/nonbeaconed_coefs")
    plot_coefs(beaconed_coefs, save_path="/mnt/datastore/Harry/Vr_grid_cells/lomb_classifiers/fast_hit_analysis/probe_coefs")

if __name__ == '__main__':
    main()