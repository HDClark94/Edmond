import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utility
import Mouse_paths

account_for_no_stop_runs = False
remove_late_stops = False

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


def plot_stuff(processed_position_path):

    prop_bs =   []
    prop_nbs = []
    prop_ps =  []

    for path in processed_position_path:
        all_days_processed_position = pd.read_pickle(path)

        tmp_b = np.unique(np.array(all_days_processed_position['stop_trial_number'][all_days_processed_position["stop_trial_type"]==0]))
        tmp_nb = np.unique(np.array(all_days_processed_position['stop_trial_number'][all_days_processed_position["stop_trial_type"]==1]))
        tmp_p = np.unique(np.array(all_days_processed_position['stop_trial_number'][all_days_processed_position["stop_trial_type"]==2]))

        tmp_b = tmp_b[~np.isnan(tmp_b)]
        tmp_nb = tmp_nb[~np.isnan(tmp_nb)]
        tmp_p = tmp_p[~np.isnan(tmp_p)]

        prop_b = len(tmp_b)/all_days_processed_position['beaconed_total_trial_number'][0]
        prop_nb = len(tmp_nb)/all_days_processed_position['nonbeaconed_total_trial_number'][0]
        prop_p = len(tmp_p)/all_days_processed_position['probe_total_trial_number'][0]

        prop_bs.append(prop_b)
        prop_nbs.append(prop_nb)
        prop_ps.append(prop_p)

    avg_prop_b = np.mean(prop_bs)
    avg_prop_nb= np.mean(prop_nbs)
    avg_prop_p= np.mean(prop_ps)

    std_prop_b= np.std(prop_bs)
    std_prop_nb= np.std(prop_bs)
    std_prop_p= np.std(prop_bs)

    # first stop histogram
    fig = plt.figure(figsize = (4,4))
    ax = fig.add_subplot(1,2,1) #stops per trial
    #ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    objects = ('B', 'NB')
    y_pos = np.arange(len(objects))*0.25

    plt.bar(y_pos, [avg_prop_b, avg_prop_nb], yerr=[std_prop_b, std_prop_nb], alpha=0.8, align='center', color=["black", "red", "blue"], width=0.25, capsize=10)
    plt.xticks(y_pos, objects)

    ax.set_ylim(0,1)
    ax.set_ylabel('Stop Proportions', fontsize=16, labelpad = 18)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.6, left = 0.15, right = 0.82, top = 0.85)
    ax.legend(loc=(0.99, 0.5))
    plt.show()
    #plt.close()


def main():

    print('-------------------------------------------------------------')

    #server_path = "Z:\ActiveProjects\Sarah\Data\PIProject_OptoEphys\Data\OpenEphys\_cohort5\VirtualReality\M1_sorted\M1_D19_2019-07-11_13-29-24\MountainSort\DataFrames"
    #plot_stuff(server_path + "\processed_position_data.pkl")

    paths = ['Z:\ActiveProjects\Sarah\Data\PIProject_OptoEphys\Data\OpenEphys\_cohort5\VirtualReality\M2_D17_2019-07-09_13-57-58\MountainSort\DataFrames\processed_position_data.pkl',
             'Z:\ActiveProjects\Sarah\Data\PIProject_OptoEphys\Data\OpenEphys\_cohort5\VirtualReality\M2_D18_2019-07-10_14-18-22\MountainSort\DataFrames\processed_position_data.pkl',
             'Z:\ActiveProjects\Sarah\Data\PIProject_OptoEphys\Data\OpenEphys\_cohort5\VirtualReality\M2_D19_2019-07-11_14-06-30\MountainSort\DataFrames\processed_position_data.pkl',
             'Z:\ActiveProjects\Sarah\Data\PIProject_OptoEphys\Data\OpenEphys\_cohort5\VirtualReality\M2_D20_2019-07-15_13-48-37\MountainSort\DataFrames\processed_position_data.pkl']
    plot_stuff(paths)

    paths = ['Z:\ActiveProjects\Sarah\Data\PIProject_OptoEphys\Data\OpenEphys\_cohort5\VirtualReality\M1_sorted\M1_D19_2019-07-11_13-29-24\MountainSort\DataFrames\processed_position_data.pkl',
             'Z:\ActiveProjects\Sarah\Data\PIProject_OptoEphys\Data\OpenEphys\_cohort5\VirtualReality\M1_sorted\M1_D18_2019-07-10_13-33-11\MountainSort\DataFrames\processed_position_data.pkl',
             'Z:\ActiveProjects\Sarah\Data\PIProject_OptoEphys\Data\OpenEphys\_cohort5\VirtualReality\M1_D17_2019-07-09_13-22-48\MountainSort\DataFrames\processed_position_data.pkl',
             'Z:\ActiveProjects\Sarah\Data\PIProject_OptoEphys\Data\OpenEphys\_cohort5\VirtualReality\M1_D15_2019-07-05_13-25-15\MountainSort\DataFrames\processed_position_data.pkl']
    plot_stuff(paths)

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()