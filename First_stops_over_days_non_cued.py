import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utility
import matplotlib.ticker as ticker
import Mouse_paths
import scipy.stats as st
from matplotlib.lines import Line2D

def plot_first_stop_days(processed_position_path, stop_type, save_path, Mouse):

    if stop_type == "all_stops":
        trial_type_collumn = "stop_trial_type"
        stop_collumn = "stop_location_cm"
        ylimits=(0, 200)
    elif stop_type == "first_stops":
        trial_type_collumn = "first_series_trial_type"
        stop_collumn = "first_series_location_cm"
        ylimits=(0, 200)

    trial_number_collumn = "stop_trial_number"

    title = processed_position_path.split("\\")[5].split(".")[0]
    save_path = save_path+"\AvgFirstStopByDay_"+title+".png"

    all_days_processed_position = pd.read_pickle(processed_position_path)

    cue_days = Mouse_paths.get_cue_days(Mouse)

    beaconed_first_stop_means = []
    beaconed_first_stop_stds = []
    non_beaconed_first_stops_means = []
    non_beaconed_first_stops_stds = []
    beaconed_all_stops_days = []
    non_beaconed_all_stops_days = []

    days =[]

    for day in range(len(all_days_processed_position)):
        days.append(day+1)
        beaconed_idx = all_days_processed_position.iloc[day][trial_type_collumn]==0 # beaconed
        non_beaconed_idx = all_days_processed_position.iloc[day][trial_type_collumn]==1 # non_beaconed

        beaconed_all_stops = np.array(all_days_processed_position.iloc[day][stop_collumn][beaconed_idx])
        non_beaconed_all_stops = np.array(all_days_processed_position.iloc[day][stop_collumn][non_beaconed_idx])

        beaconed_all_stops = beaconed_all_stops[beaconed_all_stops>30]
        non_beaconed_all_stops = non_beaconed_all_stops[non_beaconed_all_stops>30]

        beaconed_all_stops_days.append(beaconed_all_stops)
        non_beaconed_all_stops_days.append(non_beaconed_all_stops)

        beaconed_first_stop_means.append(np.nanmean(beaconed_all_stops))
        non_beaconed_first_stops_means.append(np.nanmean(non_beaconed_all_stops))

        beaconed_first_stop_stds.append(np.nanstd(all_days_processed_position.iloc[day][stop_collumn][beaconed_idx]))
        non_beaconed_first_stops_stds.append(np.nanstd(all_days_processed_position.iloc[day][stop_collumn][non_beaconed_idx]))

    # first stop plot
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,2,1) #stops per trial
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    #ax.plot(days, beaconed_first_stop_means, 'o', color="k", label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
    ax.errorbar(days,beaconed_first_stop_means,beaconed_first_stop_stds, fmt = 'o', color = '0.5', capsize = 1.5, markersize = 2, elinewidth = 1.5)

    #for day in days:
    #    ax.scatter(np.ones(len(beaconed_all_stops_days[day-1]))*day, beaconed_all_stops_days[day-1], color='0.8', marker='o')
    for i in range(len(days)):
        ax.scatter(days[i], beaconed_first_stop_means[i], marker='o', color="k")

    plot_utility.style_vr_plot_offset(ax, 100)
    plot_utility.style_track_plot(ax, 200, flipped=True)
    ax.set_ylabel('First Stop (cm)', fontsize=14, labelpad = 18)
    ax.set_xlim(0,len(days)+1)
    ax.set_ylim(ylimits)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax = fig.add_subplot(1,2,2)
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom', style='italic')
    #ax.plot(days, non_beaconed_first_stops_means, 'o', color = "k"", label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
    ax.errorbar(days,non_beaconed_first_stops_means,non_beaconed_first_stops_stds, fmt = 'o', color = '0.5', capsize = 1.5, markersize = 2, elinewidth = 1.5)

    #for day in days:
    #    ax.scatter(np.ones(len(non_beaconed_all_stops_days[day-1]))*day, non_beaconed_all_stops_days[day-1], color='0.8', marker='o')
    for i in range(len(days)):
        ax.scatter(days[i], non_beaconed_first_stops_means[i], marker='o', color="k")

    plot_utility.style_vr_plot_offset(ax, 100)
    plot_utility.style_track_plot(ax, 200, flipped=True)
    ax.set_xlim(0,len(days)+1)
    ax.set_ylim(ylimits)  # -82s
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.6, left = 0.15, right = 0.92, top = 0.95)
    plt.xlabel(" ", fontsize=16)
    fig.text(0.5, 0.04, 'Training Day', ha='center', fontsize=16)
    fig.text(0.05, 0.94, Mouse, ha='center', fontsize=16)

    plt.show()
    fig.savefig(save_path,  dpi=200)
    plt.close()


def main():

    print('-------------------------------------------------------------')

    server_path = r"Z:\ActiveProjects\Harry\2019cohort1\vr"
    save_path = server_path+ "\Summary"

    plot_first_stop_days(server_path+"\All_days_processed_position_M1.pkl", "first_stops", save_path, "M1")
    plot_first_stop_days(server_path+"\All_days_processed_position_M2.pkl", "first_stops", save_path, "M2")

    plot_first_stop_days(server_path+"\All_days_processed_position_M1.pkl", "all_stops", save_path, "M1")
    plot_first_stop_days(server_path+"\All_days_processed_position_M2.pkl", "all_stops", save_path, "M2")

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()