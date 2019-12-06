import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utility
import Mouse_paths

def get_n_stop_trial_types(trial_numbers, trial_types, goal_trial_numbers, goal_trial_types):
    trial_numbers = trial_numbers[~np.isnan(trial_numbers)]
    trial_types = trial_types[~np.isnan(trial_types)]
    goal_trial_types = goal_trial_types[~np.isnan(goal_trial_types)]
    goal_trial_numbers = goal_trial_numbers[~np.isnan(goal_trial_numbers)]

    n_b_with_stops=0
    n_nb_with_stops=0

    newValue=100 # starting on the wrong trial number
    for i in range(len(trial_numbers)):
        t = int(trial_numbers[i])
        if newValue != t:
            visited=False
        else:
            visited=True

        newValue = t
        if goal_trial_types[goal_trial_numbers==t] == 0.0 and visited==False:
            n_b_with_stops += 1
        elif goal_trial_types[goal_trial_numbers==t] == 1.0 and visited==False:
            n_nb_with_stops += 1

    return n_b_with_stops, n_nb_with_stops


def plot_histogram(processed_position_path, save_path, stop_type, Mouse, cummulative=False):
    if stop_type == "all_stops":
        trial_type_collumn = "stop_trial_type"
        stop_collumn = "stop_location_cm"
    elif stop_type == "first_stops":
        trial_type_collumn = "first_series_trial_type"
        stop_collumn = "first_series_location_cm"
    elif stop_type == "first_stops_post_cue":
        trial_type_collumn = "first_series_trial_type_postcue"
        stop_collumn = "first_series_location_cm_postcue"
    trial_number_collumn = "stop_trial_number"

    title = processed_position_path.split("\\")[6].split(".")[0]
    if cummulative:
        save_path = save_path+"\FirstStopCumHistogram_"+stop_type+title+".png"
    else:
        save_path = save_path+"\FirstStopHistogram_"+stop_type+title+".png"

    all_days_processed_position = pd.read_pickle(processed_position_path)

    cue_days = Mouse_paths.get_cue_days(Mouse)
    last2_control_days = Mouse_paths.find_last2_control_days(cue_days)

    bin_size = 5
    bins = np.arange(-200, 200, bin_size)
    bin_centres = 0.5*(bins[1:]+bins[:-1])
    beaconed_stops_early = []
    non_beaconed_stops_early = []
    b_proportion_early = []
    nb_proportion_early = []

    last_2_control_beaconed_stops = []
    last_2_control_non_beaconed_stops = []
    last_2_control_b_proportion = []
    last_2_control_nb_proportion = []

    beaconed_stops_late = []
    non_beaconed_stops_late = []
    b_proportion_late = []
    nb_proportion_late = []

    n_days = len(all_days_processed_position)

    for day in range(n_days):
        if day in last2_control_days:
            last2 = True
        else:
            last2 = False

        beaconed_idx = all_days_processed_position.iloc[day][trial_type_collumn]==0 # beaconed
        non_beaconed_idx = all_days_processed_position.iloc[day][trial_type_collumn]==1 # non_beaconed

        n_trial_with_a_stop_b, n_trial_with_a_stop_nb = get_n_stop_trial_types(all_days_processed_position.iloc[day][trial_number_collumn],
                                                                               all_days_processed_position.iloc[day][trial_type_collumn],
                                                                               all_days_processed_position.iloc[day]['goal_location_trial_numbers'],
                                                                               all_days_processed_position.iloc[day]['goal_location_trial_types'])

        n_beaconed_trials = all_days_processed_position.iloc[day]["beaconed_total_trial_number"][0]
        n_non_beaconed_trials = all_days_processed_position.iloc[day]["nonbeaconed_total_trial_number"][0]

        beaconed_stops = all_days_processed_position.iloc[day][stop_collumn][beaconed_idx]
        non_beaconed_stops = all_days_processed_position.iloc[day][stop_collumn][non_beaconed_idx]

        beaconed_stop_histogram, _ = np.histogram(beaconed_stops, bins)/n_beaconed_trials
        non_beaconed_stop_histogram, _ = np.histogram(non_beaconed_stops, bins)/n_non_beaconed_trials
        # TODO consider if the divide by n_trials is a good idea

        b_proportion_stopped = n_trial_with_a_stop_b/n_beaconed_trials
        nb_proportion_stopped = n_trial_with_a_stop_nb/n_non_beaconed_trials
        #b_proportion_stopped = 1   # comment these out to get the right
        #nb_proportion_stopped = 1

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
            b_proportion_early.append(b_proportion_stopped)
            nb_proportion_early.append(nb_proportion_stopped)
        elif n_days-day<=5:
            beaconed_stops_late.append(beaconed_stop_histogram)
            non_beaconed_stops_late.append(non_beaconed_stop_histogram)
            b_proportion_late.append(b_proportion_stopped)
            nb_proportion_late.append(nb_proportion_stopped)

        if last2:
            last_2_control_beaconed_stops.append(beaconed_stop_histogram)
            last_2_control_non_beaconed_stops.append(non_beaconed_stop_histogram)
            last_2_control_b_proportion.append(b_proportion_stopped)
            last_2_control_nb_proportion.append(nb_proportion_stopped)

    beaconed_stops_early = np.array(beaconed_stops_early)
    non_beaconed_stops_early = np.array(non_beaconed_stops_early)
    beaconed_stops_late = np.array(beaconed_stops_late)
    non_beaconed_stops_late = np.array(non_beaconed_stops_late)
    b_proportion_early = np.array(b_proportion_early)
    nb_proportion_early = np.array(nb_proportion_early)
    b_proportion_late = np.array(b_proportion_late)
    nb_proportion_late = np.array(nb_proportion_late)
    last_2_control_beaconed_stops = np.array(last_2_control_beaconed_stops)
    last_2_control_non_beaconed_stops = np.array(last_2_control_non_beaconed_stops)
    last_2_control_b_proportion = np.array(last_2_control_b_proportion)
    last_2_control_nb_proportion = np.array(last_2_control_nb_proportion)

    avg_beaconed_stops_early = np.nanmean(beaconed_stops_early, axis=0)
    avg_non_beaconed_stops_early = np.nanmean(non_beaconed_stops_early, axis=0)
    avg_beaconed_stops_late = np.nanmean(beaconed_stops_late, axis=0)
    avg_non_beaconed_stops_late = np.nanmean(non_beaconed_stops_late, axis=0)

    std_beaconed_stops_early = np.nanstd(beaconed_stops_early, axis=0)
    std_non_beaconed_stops_early = np.nanstd(non_beaconed_stops_early, axis=0)
    std_beaconed_stops_late = np.nanstd(beaconed_stops_late, axis=0)
    std_non_beaconed_stops_late = np.nanstd(non_beaconed_stops_late, axis=0)

    if len(last_2_control_beaconed_stops)>0:
        last_2_control_b_proportion = np.nanmean(last_2_control_b_proportion)
        last_2_control_nb_proportion = np.nanmean(last_2_control_nb_proportion)

        avg_last_2_control_beaconed_stops = np.nanmean(last_2_control_beaconed_stops, axis=0)
        avg_last_2_control_non_beaconed_stops = np.nanmean(last_2_control_non_beaconed_stops, axis=0)
        std_last_2_control_beaconed_stops = np.nanstd(last_2_control_beaconed_stops, axis=0)
        std_last_2_control_non_beaconed_stops = np.nanstd(last_2_control_non_beaconed_stops, axis=0)
        avg_last_2_control_beaconed_stops = avg_last_2_control_beaconed_stops*last_2_control_b_proportion
        avg_last_2_control_non_beaconed_stops = avg_last_2_control_non_beaconed_stops*last_2_control_nb_proportion
        std_last_2_control_beaconed_stops = std_last_2_control_beaconed_stops*last_2_control_b_proportion
        std_last_2_control_non_beaconed_stops = std_last_2_control_non_beaconed_stops*last_2_control_nb_proportion

    avg_b_proportion_early = np.nanmean(b_proportion_early)
    avg_nb_proportion_early = np.nanmean(nb_proportion_early)
    avg_b_proportion_late = np.nanmean(b_proportion_late)
    avg_nb_proportion_late = np.nanmean(nb_proportion_late)

    avg_beaconed_stops_early=avg_beaconed_stops_early*avg_b_proportion_early
    avg_non_beaconed_stops_early=avg_non_beaconed_stops_early*avg_nb_proportion_early
    avg_beaconed_stops_late= avg_beaconed_stops_late*avg_b_proportion_late
    avg_non_beaconed_stops_late=avg_non_beaconed_stops_late*avg_nb_proportion_late

    std_beaconed_stops_early=std_beaconed_stops_early*avg_b_proportion_early
    std_non_beaconed_stops_early=std_non_beaconed_stops_early*avg_nb_proportion_early
    std_beaconed_stops_late= std_beaconed_stops_late*avg_b_proportion_late
    std_non_beaconed_stops_late=std_non_beaconed_stops_late*avg_nb_proportion_late

    b_y_max = max(max(avg_beaconed_stops_early), max(avg_beaconed_stops_late))
    nb_y_max = max(max(avg_non_beaconed_stops_early), max(avg_non_beaconed_stops_late))

    # first stop histogram
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,2,1) #stops per trial
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.plot(bin_centres,avg_beaconed_stops_early,color = 'black',label = 'First 5 days', linewidth = 2) #plot becaoned trials
    ax.fill_between(bin_centres,avg_beaconed_stops_early-std_beaconed_stops_early,avg_beaconed_stops_early+std_beaconed_stops_early, facecolor = 'black', alpha = 0.3)
    ax.plot(bin_centres,avg_beaconed_stops_late,color = 'red',label = 'Last 5 days', linewidth = 2) #plot becaoned trials
    ax.fill_between(bin_centres,avg_beaconed_stops_late-std_beaconed_stops_late,avg_beaconed_stops_late+std_beaconed_stops_late, facecolor = 'red', alpha = 0.3)
    if len(last_2_control_beaconed_stops)>0:
        ax.plot(bin_centres,avg_last_2_control_beaconed_stops,color = 'green',label = 'Last 2 days of fixed Cue', linewidth = 2) #plot becaoned trials
        ax.fill_between(bin_centres,
                        avg_last_2_control_beaconed_stops-std_last_2_control_beaconed_stops,
                        avg_last_2_control_beaconed_stops+std_last_2_control_beaconed_stops, facecolor = 'green', alpha = 0.3)
    ax.set_xlim(-200,200)
    ax.set_ylabel('First Stop Probability', fontsize=16, labelpad = 18)
    plot_utility.style_vr_plot_offset(ax, max(avg_beaconed_stops_late))
    plot_utility.style_track_plot_cue_conditioned(ax, 300)
    ax.set_ylim(0, b_y_max+0.01)

    # non beaconed stop histogram
    ax = fig.add_subplot(1,2,2) #stops per trial
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.plot(bin_centres,avg_non_beaconed_stops_early,color = 'black', label = 'First 5 days', linewidth = 2) #plot becaoned trials
    ax.fill_between(bin_centres,avg_non_beaconed_stops_early-std_non_beaconed_stops_early,avg_non_beaconed_stops_early+std_non_beaconed_stops_early, facecolor = 'black', alpha = 0.3)
    ax.plot(bin_centres,avg_non_beaconed_stops_late,color = 'red', label = 'Last 5 days', linewidth = 2) #plot becaoned trials
    ax.fill_between(bin_centres,avg_non_beaconed_stops_late-std_non_beaconed_stops_late,avg_non_beaconed_stops_late+std_non_beaconed_stops_late, facecolor = 'red', alpha = 0.3)
    if len(last_2_control_beaconed_stops)>0:
        ax.plot(bin_centres,avg_last_2_control_non_beaconed_stops,color = 'green', label = 'Last 2 days of fixed Cue', linewidth = 2) #plot becaoned trials
        ax.fill_between(bin_centres,
                        avg_last_2_control_non_beaconed_stops-std_last_2_control_non_beaconed_stops,
                        avg_last_2_control_non_beaconed_stops+std_last_2_control_non_beaconed_stops, facecolor = 'green', alpha = 0.3)
    ax.set_xlim(-200,200)
    #ax.set_xlabel("Track Position relative to goal (cm)", fontsize=16, labelpad = 18)
    plot_utility.style_vr_plot_offset(ax, max(avg_beaconed_stops_late))
    plot_utility.style_track_plot_cue_conditioned(ax, 300)
    ax.set_ylim(0, nb_y_max+0.01)

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.6, left = 0.15, right = 0.82, top = 0.85)
    fig.text(0.5, 0.04, 'Track Position Relative to Goal (cm)', ha='center', fontsize=16)
    fig.text(0.05, 0.94, Mouse, ha='center', fontsize=16)
    ax.legend(loc=(0.99, 0.5))
    plt.show()
    fig.savefig(save_path,  dpi=200)
    plt.close()


def main():

    print('-------------------------------------------------------------')

    server_path = "Z:\ActiveProjects\Harry\MouseVR\data\Cue_conditioned_cohort1_190902"
    save_path = server_path+ "\Summary"

    plot_histogram(server_path + "\All_days_processed_position_M2.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M2", cummulative=True)
    plot_histogram(server_path + "\All_days_processed_position_M2.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M2", cummulative=False)
    #plot_histogram(server_path + "\All_days_processed_position_M2.pkl", save_path, stop_type="first_stops", Mouse="M2", cummulative=True)
    #plot_histogram(server_path + "\All_days_processed_position_M2.pkl", save_path, stop_type="first_stops", Mouse="M2", cummulative=False)
    #plot_histogram(server_path + "\All_days_processed_position_M2.pkl", save_path, stop_type="all_stops", Mouse="M2", cummulative=True)
    #plot_histogram(server_path + "\All_days_processed_position_M2.pkl", save_path, stop_type="all_stops", Mouse="M2", cummulative=False)

    plot_histogram(server_path + "\All_days_processed_position_M3.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M3", cummulative=True)
    plot_histogram(server_path + "\All_days_processed_position_M3.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M3", cummulative=False)
    #plot_histogram(server_path + "\All_days_processed_position_M3.pkl", save_path, stop_type="first_stops", Mouse="M3", cummulative=True)
    #plot_histogram(server_path + "\All_days_processed_position_M3.pkl", save_path, stop_type="first_stops", Mouse="M3", cummulative=False)
    #plot_histogram(server_path + "\All_days_processed_position_M3.pkl", save_path, stop_type="all_stops", Mouse="M3", cummulative=True)
    #plot_histogram(server_path + "\All_days_processed_position_M3.pkl", save_path, stop_type="all_stops", Mouse="M3", cummulative=False)

    plot_histogram(server_path + "\All_days_processed_position_M4.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M4", cummulative=True)
    plot_histogram(server_path + "\All_days_processed_position_M4.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M4", cummulative=False)
    #plot_histogram(server_path + "\All_days_processed_position_M4.pkl", save_path, stop_type="first_stops", Mouse="M4", cummulative=True)
    #plot_histogram(server_path + "\All_days_processed_position_M4.pkl", save_path, stop_type="first_stops", Mouse="M4", cummulative=False)
    #plot_histogram(server_path + "\All_days_processed_position_M4.pkl", save_path, stop_type="all_stops", Mouse="M4", cummulative=True)
    #plot_histogram(server_path + "\All_days_processed_position_M4.pkl", save_path, stop_type="all_stops", Mouse="M4", cummulative=False)

    plot_histogram(server_path + "\All_days_processed_position_M5.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M5", cummulative=True)
    plot_histogram(server_path + "\All_days_processed_position_M5.pkl", save_path, stop_type="first_stops_post_cue",  Mouse="M5", cummulative=False)
    #plot_histogram(server_path + "\All_days_processed_position_M5.pkl", save_path, stop_type="first_stops", Mouse="M5", cummulative=True)
    #plot_histogram(server_path + "\All_days_processed_position_M5.pkl", save_path, stop_type="first_stops", Mouse="M5", cummulative=False)
    #plot_histogram(server_path + "\All_days_processed_position_M5.pkl", save_path, stop_type="all_stops", Mouse="M5", cummulative=True)
    #plot_histogram(server_path + "\All_days_processed_position_M5.pkl", save_path, stop_type="all_stops", Mouse="M5", cummulative=False)

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()