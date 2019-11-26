import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utility
import matplotlib.ticker as ticker

def plot_first_stop_days(processed_position_path, save_path):
    title = processed_position_path.split("\\")[6].split(".")[0]
    save_path = save_path+"\AvgFirstStopByDay_"+title+".png"

    all_days_processed_position = pd.read_pickle(processed_position_path)

    beaconed_first_stop_means = []
    beaconed_first_stop_stds = []
    non_beaconed_first_stops_means = []
    non_beaconed_first_stops_stds = []
    days =[]

    for day in range(len(all_days_processed_position)):
        days.append(day+1)
        beaconed_idx = all_days_processed_position.iloc[day]["first_series_trial_type_postcue"]==0 # beaconed
        non_beaconed_idx = all_days_processed_position.iloc[day]["first_series_trial_type_postcue"]==1 # non_beaconed

        beaconed_first_stop_means.append(np.nanmean(all_days_processed_position.iloc[day]["first_series_location_cm_postcue"][beaconed_idx]))
        non_beaconed_first_stops_means.append(np.nanmean(all_days_processed_position.iloc[day]["first_series_location_cm_postcue"][non_beaconed_idx]))
        beaconed_first_stop_stds.append(np.nanstd(all_days_processed_position.iloc[day]["first_series_location_cm_postcue"][beaconed_idx]))
        non_beaconed_first_stops_stds.append(np.nanstd(all_days_processed_position.iloc[day]["first_series_location_cm_postcue"][non_beaconed_idx]))

    # first stop plot
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,2,1) #stops per trial
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.plot(days, beaconed_first_stop_means, 'o',color = '0.3', label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
    ax.errorbar(days,beaconed_first_stop_means,beaconed_first_stop_stds, fmt = 'o', color = '0.3', capsize = 1.5, markersize = 2, elinewidth = 1.5)
    plot_utility.style_vr_plot_offset(ax, 100)
    plot_utility.style_track_plot_cue_conditioned(ax, 300, flipped=True)
    ax.set_ylabel('First Stop', fontsize=16, labelpad = 18)
    ax.set_xlim(0,len(days)+1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_ylim(-82, 70)

    ax = fig.add_subplot(1,2,2)
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom', style='italic')
    ax.plot(days, non_beaconed_first_stops_means, 'o', color = '0.3', label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
    ax.errorbar(days,non_beaconed_first_stops_means,non_beaconed_first_stops_stds, fmt = 'o', color = '0.3', capsize = 1.5, markersize = 2, elinewidth = 1.5)
    plot_utility.style_vr_plot_offset(ax, 100)
    plot_utility.style_track_plot_cue_conditioned(ax, 300, flipped=True)
    ax.set_xlim(0,len(days)+1)
    ax.set_ylim(-82, 70)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.6, left = 0.15, right = 0.82, top = 0.95)
    plt.xlabel(" ", fontsize=16)
    fig.text(0.5, 0.04, 'Training Day', ha='center', fontsize=16)
    plt.show()
    fig.savefig(save_path,  dpi=200)
    plt.close()


def main():

    print('-------------------------------------------------------------')

    server_path = "Z:\ActiveProjects\Harry\MouseVR\data\Cue_conditioned_cohort1_190902"
    save_path = server_path+ "\Summary"

    plot_first_stop_days(server_path+"\All_days_processed_position_M2.pkl", save_path)
    plot_first_stop_days(server_path+"\All_days_processed_position_M3.pkl", save_path)
    plot_first_stop_days(server_path+"\All_days_processed_position_M4.pkl", save_path)
    plot_first_stop_days(server_path+"\All_days_processed_position_M5.pkl", save_path)

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()