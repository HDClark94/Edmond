import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import itertools
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
import os
warnings.filterwarnings('ignore')

def style_vr_plot(ax, x_max=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=True,
        labelleft=True,
        labelbottom=True)  # labels along the bottom edge are off

    #ax.set_aspect('equal')

    ax.axvline(0, linewidth=2.5, color='black') # bold line on the y axis
    ax.axhline(0, linewidth=2.5, color='black') # bold line on the x axis
    if x_max is not None:
        plt.ylim(0, x_max)

    return ax

def get_tidy_title(collumn):
    if collumn == "speed_score":
        return "Speed Score"
    elif collumn == "grid_score":
        return "Grid Score"
    elif collumn == "border_score":
        return "Border Score"
    elif collumn == "corner_score":
        return "Corner Score"
    elif collumn == "hd_score":
        return "HD Score"
    elif collumn == "spatial_information_score":
        return "Spatial Information"
    elif collumn == "hd_threshold":
        return "HD Cut-Off"
    elif collumn == "grid_threshold":
        return "Grid Score Cut-Off"
    elif collumn == "border_threshold":
        return "Border Score Cut-Off"
    elif collumn == "ramp_score_out":
        return "Ramp Score Outbound"
    elif collumn == "ramp_score_home":
        return "Ramp Score Homebound"
    elif collumn == "ramp_score":
        return "Ramp Score"
    elif collumn == "abs_ramp_score":
        return "Abs Ramp Score"
    elif collumn == "max_ramp_score":
        return "Max Ramp Score"
    elif collumn == 'rayleigh_score':
        return 'Rayleigh Score'
    elif collumn == "rate_map_correlation_first_vs_second_half":
        return "Spatial Stability"
    elif collumn == "lm_result_b_outbound":
        return "LM Outbound fit"
    elif collumn == "lm_result_b_homebound":
        return "LM Homebound fit"
    elif collumn == "lmer_result_b_outbound":
        return "LMER Outbound fit"
    elif collumn == "lmer_result_b_homebound":
        return "LMER Homebound fit"
    elif collumn == "beaconed":
        return "Beaconed"
    elif collumn == "non-beaconed":
        return "Non Beaconed"
    elif collumn == "probe":
        return "Probe"
    elif collumn == "all":
        return "All Trial Types"
    elif collumn == "spike_ratio":
        return "Spike Ratio"
    elif collumn == "_cohort5":
        return "C5"
    elif collumn == "_cohort4":
        return "C4"
    elif collumn == "_cohort3":
        return "C3"
    elif collumn == "_cohort2":
        return "C2"
    elif collumn == "ThetaIndex_vr":
        return "Theta Index VR"
    elif collumn == "ThetaPower_vr":
        return "Theta Power VR"
    elif collumn == "ThetaIndex":
        return "Theta Index"
    elif collumn == "ThetaPower":
        return "Theta Power"
    elif collumn == 'best_theta_idx_vr':
        return "Max Theta Index VR"
    elif collumn == 'best_theta_idx_of':
        return "Max Theta Index OF"
    elif collumn == 'best_theta_idx_combined':
        return "Max Theta Index VR+OF"
    elif collumn == 'best_theta_pwr_vr':
        return "Max Theta Power VR"
    elif collumn == 'best_theta_pwr_of':
        return "Max Theta Power OF"
    elif collumn == 'best_theta_pwr_combined':
        return "Max Theta Power VR+OF"
    else:
        print("collumn title not found!")

def get_score_threshold(collumn):
    if collumn == "speed_score":
        return 0.18
    elif collumn == "grid_score":
        return 0.4
    elif collumn == "border_score":
        return 0.5
    elif collumn == "corner_score":
        return 0.5
    elif collumn == "hd_score":
        return 0.5
    elif collumn == "rate_map_correlation_first_vs_second_half":
        return None

def get_cohort_mouse(row):
    full_session_id = row["full_session_id"].iloc[0]
    session_id = full_session_id.split("/")[-1]
    mouse = session_id.split("_D")[0]
    cohort = get_tidy_title(full_session_id.split("/")[-4])
    return cohort+"_"+mouse

def correct_datatypes(dataframe, ignore_of=False):

    if ignore_of:
        dataframe["ramp_score"] = pd.to_numeric(dataframe["ramp_score"])
        dataframe["fr_range"] = pd.to_numeric(dataframe["fr_range"])
        dataframe["meanVar"] = pd.to_numeric(dataframe["meanVar"])
        return dataframe

    # I'm first fixing the dtype of the prefered HD, this is a size-1 np.ndarray atm
    for i in range(len(dataframe)):
        if type(dataframe["preferred_HD"].iloc[i]) is np.ndarray:
            if len(dataframe["preferred_HD"].iloc[i]) > 1:
                dataframe["preferred_HD"].iloc[i] = np.nan
            else:
                dataframe["preferred_HD"].iloc[i] = dataframe["preferred_HD"].iloc[i][0]

    dataframe["rate_map_correlation_first_vs_second_half"] = pd.to_numeric(dataframe["rate_map_correlation_first_vs_second_half"])
    dataframe["mean_firing_rate"] = pd.to_numeric(dataframe["mean_firing_rate"])
    dataframe["isolation"] = pd.to_numeric(dataframe["isolation"])
    dataframe["noise_overlap"] = pd.to_numeric(dataframe["noise_overlap"])
    dataframe["peak_snr"] = pd.to_numeric(dataframe["peak_snr"])
    dataframe["peak_amp"] = pd.to_numeric(dataframe["peak_amp"])
    dataframe["speed_score"] = pd.to_numeric(dataframe["speed_score"])
    dataframe["speed_score_p_values"] = pd.to_numeric(dataframe["speed_score_p_values"])
    dataframe["max_firing_rate_hd"] = pd.to_numeric(dataframe["max_firing_rate_hd"])
    dataframe["preferred_HD"] = pd.to_numeric(dataframe["preferred_HD"])
    dataframe["rayleigh_score"] = pd.to_numeric(dataframe["rayleigh_score"])
    dataframe["grid_spacing"] = pd.to_numeric(dataframe["grid_spacing"])
    dataframe["hd_score"] = pd.to_numeric(dataframe["hd_score"])
    dataframe["field_size"] = pd.to_numeric(dataframe["field_size"])
    dataframe["grid_score"] = pd.to_numeric(dataframe["grid_score"])
    dataframe["border_score"] = pd.to_numeric(dataframe["border_score"])
    dataframe["corner_score"] = pd.to_numeric(dataframe["corner_score"])
    dataframe["percent_excluded_bins_rate_map_correlation_first_vs_second_half_p"] = pd.to_numeric(dataframe["percent_excluded_bins_rate_map_correlation_first_vs_second_half_p"])
    dataframe["hd_correlation_first_vs_second_half"] = pd.to_numeric(dataframe["hd_correlation_first_vs_second_half"])
    dataframe["hd_correlation_first_vs_second_half_p"] = pd.to_numeric(dataframe["hd_correlation_first_vs_second_half_p"])
    dataframe["split_cluster"] = pd.to_numeric(dataframe["split_cluster"])
    dataframe["n_spikes_vr"] = pd.to_numeric(dataframe["n_spikes_vr"])
    dataframe["n_spikes_of"] = pd.to_numeric(dataframe["n_spikes_of"])
    dataframe["ramp_score_out"] = pd.to_numeric(dataframe["ramp_score_out"])
    dataframe["ramp_score_home"] = pd.to_numeric(dataframe["ramp_score_home"])
    dataframe["ramp_score"] = pd.to_numeric(dataframe["ramp_score"])
    dataframe["fr_range"] = pd.to_numeric(dataframe["fr_range"])
    dataframe["meanVar"] = pd.to_numeric(dataframe["meanVar"])
    dataframe["ThetaIndex_vr"] = pd.to_numeric(dataframe["ThetaIndex_vr"])
    dataframe["ThetaIndex"] = pd.to_numeric(dataframe["ThetaIndex"])
    return dataframe


def get_p_text(p, ns=False):

    if p<0.0001:
        return "****"
    elif p<0.001:
        return "***"
    elif p<0.01:
        return "**"
    elif p<0.05:
        return "*"
    elif ns:
        return "ns"
    else:
        return " "

def plot_regression(ax, x, y):
    # x  and y are pandas collumn
    x = x.values
    y = y.values
    x = x[~np.isnan(y)].reshape(-1, 1)
    y = y[~np.isnan(y)].reshape(-1, 1)

    pearson_r = stats.pearsonr(x.flatten(),y.flatten())

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x,y)  # perform linear regression

    x_test = np.linspace(min(x), max(x), 100)

    Y_pred = linear_regressor.predict(x_test.reshape(-1, 1))  # make predictions

    ax.text(  # position text relative to Axes
        0.95, 1.25, "R= "+str(np.round(pearson_r[0], decimals=2))+ ", p = "+str(np.round(pearson_r[1], decimals=2)) +str(get_p_text(pearson_r[1])),
        ha='right', va='top',
        transform=ax.transAxes, fontsize=20)

    ax.plot(x_test, Y_pred)

def lmer_result_color(lmer_result):
    if lmer_result=="PA":
        return "orchid"
    elif lmer_result=="PS":
        return "indianred"
    elif lmer_result=="A":
        return "lightskyblue"
    elif lmer_result=="S":
        return "salmon"
    elif lmer_result=="P":
        return "deeppink"
    elif lmer_result=="PSA":
        return "forestgreen"
    elif lmer_result=="SA":
        return "mediumaquamarine"
    elif lmer_result=="None":
        return "grey"

def lm_result_color(lm_result):
    if lm_result=="None":
        return "grey"
    elif lm_result=="Negative":
        return "red"
    elif lm_result=="Positive":
        return "blue"

def ramp_driver_color(ramp_driver):
    if ramp_driver == "PI":
        return "yellow"
    elif ramp_driver == "Cue":
        return "green"
    elif ramp_driver == "None":
        return "grey"

def cohort_mouse_label_color(cohort_mouse_label):
    if cohort_mouse_label == "C2_1124":
        return "C0"
    elif cohort_mouse_label == "C2_245":
        return "C1"
    elif cohort_mouse_label == "C3_M1":
        return "C2"
    elif cohort_mouse_label == "C3_M6":
        return "C3"
    elif cohort_mouse_label == "C4_M2":
        return "C4"
    elif cohort_mouse_label == "C4_M3":
        return "C5"
    elif cohort_mouse_label == "C5_M1":
        return "C6"
    elif cohort_mouse_label == "C5_M2":
        return "C7"

def concatenate_spatial_firing(parent_dir, save_path, df=pd.DataFrame()):

    recording_list = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    for recording_path in recording_list:
        print("processing ", recording_path)
        if os.path.isfile(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl"):
            spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

            column_list_to_drop = ["firing_times", "firing_times_opto", "all_snippets", "random_snippets",
                                   "position_x", "position_x_pixels", "position_y", "position_y_pixels", "hd", "speed"]
            for column in column_list_to_drop:
                if column in list(spatial_firing):
                    del spatial_firing[column]
            df = pd.concat([df, spatial_firing], ignore_index=True)
    df.to_pickle(save_path)
    return df

def plot_of_shuffle_metrics(data, save_path, metric):
    threshold_name = get_metric_threshold_name(metric)
    x = data[threshold_name]
    y = data[metric]
    max_v = max([max(x), max(y)])
    max_v = max_v+(0.1*max_v)
    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(5, 5, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.scatter(x, y, marker="o", color="black", alpha=0.4)
    ax.plot(np.arange(0, max_v, 0.1), np.arange(0, max_v, 0.1), linestyle="dashed", color="black")
    ax.set_ylim(0, max_v)
    ax.set_xlim(0, max_v)
    style_vr_plot(ax)
    plt.ylabel(get_tidy_title(metric), fontsize=15, labelpad = 10)
    plt.xlabel(get_tidy_title(threshold_name), fontsize=15, labelpad = 10)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig(save_path + 'shuffle_cutoff_'+metric+'.png', dpi=200)
    plt.close()

def get_metric_threshold_name(metric):
    if metric == "hd_score":
        return "hd_threshold"
    elif metric == "spatial_information_score":
        return "spatial_threshold"
    elif metric == "grid_score":
        return "grid_threshold"
    elif metric == "rayleigh_score":
        return "rayleigh_threshold"
    elif metric == "border_score":
        return "border_threshold"
    else:
        print("metric passed doesnt have a threshold name")

def plot_pie_chart(data, save_path):
    grid_share = int(np.round((len(data[data["classifier"] == "G"])/len(data))*100))
    border_share = int(np.round((len(data[data["classifier"] == "B"])/len(data))*100))
    non_grid_share = int(np.round((len(data[data["classifier"] == "NG"])/len(data))*100))
    hd_share = int(np.round((len(data[data["classifier"] == "HD"])/len(data))*100))
    non_spatial_share = int(np.round((len(data[data["classifier"] == "NS"])/len(data))*100))

    pieLabels = ['Grid: '+str(grid_share)+"%",
                 'Border: '+str(border_share)+"%",
                 'Non-Grid Spatial: '+str(non_grid_share)+"%",
                 'Pure HD: '+str(hd_share)+"%",
                 'Non-Spatial: '+str(non_spatial_share)+"%"]

    # Population data
    population_colors = ["b", "g", "r", "c", "m"]
    populationShare     = [grid_share, border_share, non_grid_share, hd_share, non_spatial_share]
    plt.rcParams["figure.figsize"] = [8, 4]
    plt.rcParams["figure.autolayout"] = True

    # Draw the pie chart
    patches, texts = plt.pie(populationShare,
                                    labels=["", "", "", "", ""],
                                    startangle=90,
                                    colors=population_colors,
                                    wedgeprops={"edgecolor":"k",'linewidth': 2, 'linestyle': 'solid', 'antialiased': True},
                                    pctdistance=1, radius = 0.5)
    #autopct='%1i%%',
    # Aspect ratio - equal means pie is a circle
    plt.axis('equal')
    plt.legend(patches, pieLabels, loc="best")
    plt.tight_layout()
    plt.savefig(save_path + 'pie.png', dpi=200)
    plt.close()


    print((len(data[data["classifier"] == "G"])/len(data))*100, "% of the cells are grid cells")
    print((len(data[data["classifier"] == "B"])/len(data))*100, "% of the cells are border cells")
    print((len(data[data["classifier"] == "NG"])/len(data))*100, "% of the cells are non-grid spatial cells")
    print((len(data[data["classifier"] == "HD"])/len(data))*100, "% of the cells are head direction cells")
    print((len(data[data["classifier"] == "NS"])/len(data))*100, "% of the cells are non-spatial cells")
    return

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    df=pd.DataFrame()
    ramp_scores_path = ""
    tetrode_location_path = "/mnt/datastore/Harry/Mouse_data_for_sarah_paper/tetrode_locations.csv"
    save_path = r"/mnt/datastore/Harry/Mouse_data_for_sarah_paper/open_field_analysis/"
    #data = concatenate_spatial_firing(parent_dir=r"/mnt/datastore/Harry/Cohort7_october2020/of", save_path=save_path+"Cohort7_october2020_of.pkl", df=df)
    data = pd.read_pickle(save_path+"Cohort7_october2020_of.pkl")
    plot_of_shuffle_metrics(data, save_path, metric="spatial_information_score")
    plot_of_shuffle_metrics(data, save_path, metric="hd_score")
    plot_of_shuffle_metrics(data, save_path, metric="grid_score")
    plot_of_shuffle_metrics(data, save_path, metric="border_score")

    plot_pie_chart(data, save_path)

    # now we want to only look at the positionally encoded neurons in vr
    # add a partition step for positional neurons



    print("look now")

if __name__ == '__main__':
    main()