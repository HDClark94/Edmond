import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utility


def plot_first_stop_days(processed_position_path, save_path, cummulative=False):
    title = processed_position_path.split("\\")[6].split(".")[0]
    if cummulative:
        save_path = save_path+"\FirstStopCumHistogram_"+title+".png"
    else:
        save_path = save_path+"\FirstStopHistogram_"+title+".png"

    all_days_processed_position = pd.read_pickle(processed_position_path)

    bin_size = 2
    bins = np.arange(-200, 200, bin_size)
    bin_centres = 0.5*(bins[1:]+bins[:-1])
    beaconed_stops_early = []
    non_beaconed_stops_early = []

    beaconed_stops_late = []
    non_beaconed_stops_late = []

    for day in range(len(all_days_processed_position)):
        beaconed_idx = all_days_processed_position.iloc[day]["first_series_trial_type_postcue"]==0 # beaconed
        non_beaconed_idx = all_days_processed_position.iloc[day]["first_series_trial_type_postcue"]==1 # non_beaconed
        n_beaconed_trials = all_days_processed_position.iloc[day]["beaconed_total_trial_number"][0]
        n_non_beaconed_trials = all_days_processed_position.iloc[day]["nonbeaconed_total_trial_number"][0]

        beaconed_stop_histogram, _ = np.histogram(all_days_processed_position.iloc[day]["first_series_location_cm_postcue"][beaconed_idx], bins)/n_beaconed_trials
        non_beaconed_stop_histogram, _ =  np.histogram(all_days_processed_position.iloc[day]["first_series_location_cm_postcue"][non_beaconed_idx], bins)/n_non_beaconed_trials

        # remove stops 60 cm after reward zone
        beaconed_stop_histogram[int(260/bin_size):] = 0
        non_beaconed_stop_histogram[int(260/bin_size):] = 0

        beaconed_stop_histogram = beaconed_stop_histogram/np.sum(beaconed_stop_histogram)
        non_beaconed_stop_histogram = non_beaconed_stop_histogram/np.sum(non_beaconed_stop_histogram)

        if cummulative:
            beaconed_stop_histogram = np.cumsum(beaconed_stop_histogram)
            non_beaconed_stop_histogram = np.cumsum(non_beaconed_stop_histogram)

        if day<5:
            beaconed_stops_early.append(beaconed_stop_histogram)
            non_beaconed_stops_early.append(non_beaconed_stop_histogram)
        elif day>20:
            beaconed_stops_late.append(beaconed_stop_histogram)
            non_beaconed_stops_late.append(non_beaconed_stop_histogram)

    beaconed_stops_early = np.array(beaconed_stops_early)
    non_beaconed_stops_early = np.array(non_beaconed_stops_early)
    beaconed_stops_late = np.array(beaconed_stops_late)
    non_beaconed_stops_late = np.array(non_beaconed_stops_late)

    avg_beaconed_stops_early = np.nanmean(beaconed_stops_early, axis=0)
    avg_non_beaconed_stops_early = np.nanmean(non_beaconed_stops_early, axis=0)
    avg_beaconed_stops_late = np.nanmean(beaconed_stops_late, axis=0)
    avg_non_beaconed_stops_late = np.nanmean(non_beaconed_stops_late, axis=0)

    std_beaconed_stops_early = np.nanstd(beaconed_stops_early, axis=0)
    std_non_beaconed_stops_early = np.nanstd(non_beaconed_stops_early, axis=0)
    std_beaconed_stops_late = np.nanstd(beaconed_stops_late, axis=0)
    std_non_beaconed_stops_late = np.nanstd(non_beaconed_stops_late, axis=0)

    # first stop histogram
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,2,1) #stops per trial
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.plot(bin_centres,avg_beaconed_stops_early,color = 'blue',label = 'Early', linewidth = 2) #plot becaoned trials
    ax.fill_between(bin_centres,avg_beaconed_stops_early-std_beaconed_stops_early,avg_beaconed_stops_early+std_beaconed_stops_early, facecolor = 'blue', alpha = 0.3)
    ax.plot(bin_centres,avg_beaconed_stops_late,color = 'red',label = 'Late', linewidth = 2) #plot becaoned trials
    ax.fill_between(bin_centres,avg_beaconed_stops_late-std_beaconed_stops_late,avg_beaconed_stops_late+std_beaconed_stops_late, facecolor = 'red', alpha = 0.3)
    ax.set_xlim(-200,200)
    ax.set_ylabel('1st stop probability', fontsize=16, labelpad = 18)
    plot_utility.style_vr_plot_offset(ax, max(avg_beaconed_stops_late))
    plot_utility.style_track_plot_cue_conditioned(ax, 300)
    ax.legend(loc="upper left")

    ax = fig.add_subplot(1,2,2) #stops per trial
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.plot(bin_centres,avg_non_beaconed_stops_early,color = 'blue', label = 'Early', linewidth = 2) #plot becaoned trials
    ax.fill_between(bin_centres,avg_non_beaconed_stops_early-std_non_beaconed_stops_early,avg_non_beaconed_stops_early+std_non_beaconed_stops_early, facecolor = 'blue', alpha = 0.3)
    ax.plot(bin_centres,avg_non_beaconed_stops_late,color = 'red', label = 'Late', linewidth = 2) #plot becaoned trials
    ax.fill_between(bin_centres,avg_non_beaconed_stops_late-std_non_beaconed_stops_late,avg_non_beaconed_stops_late+std_non_beaconed_stops_late, facecolor = 'red', alpha = 0.3)
    ax.set_xlim(-200,200)
    #ax.set_xlabel("Track Position relative to goal (cm)", fontsize=16, labelpad = 18)
    plot_utility.style_vr_plot_offset(ax, max(avg_beaconed_stops_late))
    plot_utility.style_track_plot_cue_conditioned(ax, 300)

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.6, left = 0.15, right = 0.82, top = 0.85)
    fig.text(0.5, 0.04, 'Track Position relative to goal (cm)', ha='center', fontsize=16)
    plt.show()
    fig.savefig(save_path,  dpi=200)
    plt.close()


def main():

    print('-------------------------------------------------------------')

    server_path = "Z:\ActiveProjects\Harry\MouseVR\data\Cue_conditioned_cohort1_190902"
    save_path = server_path+ "\Summary"
    all_days_processed_position_path = server_path + "\All_days_processed_position_M4.pkl"
    plot_first_stop_days(all_days_processed_position_path, save_path, cummulative=True)
    plot_first_stop_days(all_days_processed_position_path, save_path, cummulative=False)

    #all_days_processed_position_path = server_path + "\All_days_processed_position_M2.pkl"
    #plot_first_stop_days(all_days_processed_position_path, save_path, cummulative=True)
    #plot_first_stop_days(all_days_processed_position_path, save_path, cummulative=False)


    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()