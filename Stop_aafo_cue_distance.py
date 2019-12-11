import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utility
import Mouse_paths
from sklearn.linear_model import LinearRegression

def get_goal_locations(trial_numbers, goal_trial_numbers, goal_locations):
    non_nan_length = len(trial_numbers[~np.isnan(trial_numbers)])
    new_goal_locations = trial_numbers.copy()
    # this function gives the goal locations in a stop by stop format
    for i in range(non_nan_length):
        trial_number = trial_numbers[i]
        tmp = goal_locations[goal_trial_numbers==trial_number]
        new_goal_locations[i] = tmp
    return new_goal_locations


def plot_cue_distance_vs_stops(processed_position_path, save_path, stop_type, Mouse, cummulative=False):
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
    save_path = save_path+"\Fcue_distance"+stop_type+title+".png"
    all_days_processed_position = pd.read_pickle(processed_position_path)
    cue_days = Mouse_paths.get_cue_days(Mouse)

    beaconed_stops_days = np.array([])
    non_beaconed_stops_days = np.array([])

    beaconed_goal_location_days = np.array([])
    non_beaconed_goal_location_days = np.array([])

    n_days = len(all_days_processed_position)
    for day in range(n_days):
        if n_days-day<5:
            goal_locations = get_goal_locations(all_days_processed_position.iloc[day][trial_number_collumn],
                                                all_days_processed_position.iloc[day]['goal_location_trial_numbers'],
                                                all_days_processed_position.iloc[day]['goal_location'])

            beaconed_idx = all_days_processed_position.iloc[day][trial_type_collumn]==0 # beaconed
            non_beaconed_idx = all_days_processed_position.iloc[day][trial_type_collumn]==1 # non_beaconed

            beaconed_all_stops = np.array(all_days_processed_position.iloc[day][stop_collumn][beaconed_idx])
            non_beaconed_all_stops = np.array(all_days_processed_position.iloc[day][stop_collumn][non_beaconed_idx])

            beaconed_goal_locations = np.array(goal_locations[beaconed_idx])
            non_beaconed_goal_locations = np.array(goal_locations[non_beaconed_idx])

            # remove stops 60 cm after reward zone
            b_tmp = beaconed_all_stops<60
            nb_tmp = non_beaconed_all_stops<60
            beaconed_all_stops = beaconed_all_stops[b_tmp]
            non_beaconed_all_stops = non_beaconed_all_stops[nb_tmp]

            beaconed_goal_locations = beaconed_goal_locations[b_tmp]
            non_beaconed_goal_locations = non_beaconed_goal_locations[nb_tmp]

            beaconed_stops_days = np.append(beaconed_stops_days, beaconed_all_stops)
            non_beaconed_stops_days = np.append(non_beaconed_stops_days, non_beaconed_all_stops)
            beaconed_goal_location_days = np.append(beaconed_goal_location_days, beaconed_goal_locations)
            non_beaconed_goal_location_days = np.append(non_beaconed_goal_location_days, non_beaconed_goal_locations)

    # first
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,2,1) #stops per trial
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.scatter(beaconed_stops_days,beaconed_goal_location_days, color = 'black',label = 'First 5 days', linewidth = 2) #plot becaoned trials
    ax.set_xlim(-90,40)
    #plot_utility.style_vr_plot_offset(ax, 100)
    plot_utility.style_track_plot_cue_conditioned(ax, 300)
    ax.set_ylabel('True Goal Location (cm)', fontsize=16, labelpad = 18)
    ax.set_ylim(100, 200)
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
    #ax.set_ylim(0, nb_y_max+0.01)


    # non beaconed
    ax = fig.add_subplot(1,2,2) #stops per trial
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.scatter(non_beaconed_stops_days,non_beaconed_goal_location_days,color = 'black', label = 'First 5 days', linewidth = 2) #plot becaoned trials
    ax.set_xlim(-90,40)
    ax.set_ylim(100, 200)
    #plot_utility.style_vr_plot_offset(ax, 100)
    plot_utility.style_track_plot_cue_conditioned(ax, 300)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.6, left = 0.15, right = 0.82, top = 0.85)
    fig.text(0.5, 0.04, 'Track Position Relative to Goal (cm)', ha='center', fontsize=16)
    fig.text(0.05, 0.94, Mouse, ha='center', fontsize=16)
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
    #ax.legend(loc=(0.99, 0.5))
    plt.show()
    fig.savefig(save_path,  dpi=200)
    plt.close()



def main():

    print('-------------------------------------------------------------')

    server_path = "Z:\ActiveProjects\Harry\MouseVR\data\Cue_conditioned_cohort1_190902"
    save_path = server_path+ "\Summary"

    plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M2.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M2")
    #plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M2.pkl", save_path, stop_type="first_stops", Mouse="M2")
    #plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M2.pkl", save_path, stop_type="all_stops", Mouse="M2")

    plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M3.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M3")
    #plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M3.pkl", save_path, stop_type="first_stops", Mouse="M3")
    #plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M3.pkl", save_path, stop_type="all_stops", Mouse="M3")

    plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M4.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M4")
    #plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M4.pkl", save_path, stop_type="first_stops", Mouse="M4")
    #plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M4.pkl", save_path, stop_type="all_stops", Mouse="M4")

    plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M5.pkl", save_path, stop_type="first_stops_post_cue", Mouse="M5")
    #plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M5.pkl", save_path, stop_type="first_stops", Mouse="M5")
    #plot_cue_distance_vs_stops(server_path + "\All_days_processed_position_M5.pkl", save_path, stop_type="all_stops", Mouse="M5")

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()