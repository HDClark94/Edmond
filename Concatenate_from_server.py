import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import Mouse_paths

def load_processed_position_all_days(recordings_folder_path, paths, mouse):
    processed_position_path = "\MountainSort\DataFrames\processed_position_data.pkl"

    all_days = pd.DataFrame()

    for recording in paths:
        data_frame_path = recordings_folder_path+recording+processed_position_path
        if os.path.exists(data_frame_path):
            print('I found a spatial data frame.')
            session_id = recording

            processed_position = pd.read_pickle(data_frame_path)

            all_days = all_days.append({"session_id": session_id,
                                        'beaconed_total_trial_number': np.array(processed_position['beaconed_total_trial_number']),
                                        'nonbeaconed_total_trial_number': np.array(processed_position['nonbeaconed_total_trial_number']),
                                        'probe_total_trial_number': np.array(processed_position['probe_total_trial_number']),
                                        'goal_location': np.array(processed_position['goal_location']),
                                        'goal_location_trial_numbers': np.array(processed_position['goal_location_trial_numbers']),
                                        'goal_location_trial_types': np.array(processed_position['goal_location_trial_types']),
                                        'stop_location_cm': np.array(processed_position['stop_location_cm']),
                                        'stop_trial_number': np.array(processed_position['stop_trial_number']),
                                        'stop_trial_type': np.array(processed_position['stop_trial_type']),
                                        'first_series_location_cm': np.array(processed_position['first_series_location_cm']),
                                        'first_series_trial_number': np.array(processed_position['first_series_trial_number']),
                                        'first_series_trial_type': np.array(processed_position['first_series_trial_type']),
                                        'first_series_location_cm_postcue': np.array(processed_position['first_series_location_cm_postcue']),
                                        'first_series_trial_number_postcue': np.array(processed_position['first_series_trial_number_postcue']),
                                        'first_series_trial_type_postcue': np.array(processed_position['first_series_trial_type_postcue'])}, ignore_index=True)

            print('Position data extracted from frame')

    all_days.to_pickle(recordings_folder_path + '/all_days_processed_position_' + mouse + '.pkl')

def plot_first_stop_days():

    '''
    bins = np.arange(0.5,n_bins+0.5,1)

    # first stop histogram
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,3,1) #stops per trial
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.axvspan(rz_start, rz_end, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
    ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
    ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
    ax.plot(bins,con_beac_w1,color = 'blue',label = 'Beaconed', linewidth = 2) #plot becaoned trials
    ax.fill_between(bins,con_beac_w1-sd_con_beac_w1,con_beac_w1+sd_con_beac_w1, facecolor = 'blue', alpha = 0.3)
    #ax.plot(bins,con_beac_w4,color = 'red',label = 'Beaconed', linewidth = 2) #plot becaoned trials
    #ax.fill_between(bins,con_beac_w4-sd_con_beac_w4,con_beac_w4+sd_con_beac_w4, facecolor = 'red', alpha = 0.3)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
    ax.set_xlim(0,20)
    ax.set_ylim(0,0.95)
    adjust_spines(ax, ['left','bottom']) # removes top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax.set_xticklabels(['0', '10', '20'])
    ax.set_yticklabels(['0','0.3','0.6','0.9'])
    ax.set_ylabel('1st stop probability', fontsize=16, labelpad = 18)

    ax = fig.add_subplot(1,3,2) #stops per trial
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.axvspan(rz_start, rz_end, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
    ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
    ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
    ax.plot(bins,con_nbeac_w1,color = 'blue', linewidth = 2) #plot becaoned trials
    ax.fill_between(bins,con_nbeac_w1-sd_con_nbeac_w1,con_nbeac_w1+sd_con_nbeac_w1, facecolor = 'blue', alpha = 0.3)
    #ax.plot(bins,con_nbeac_w4,color = 'red', linewidth = 2) #plot becaoned trials
    #ax.fill_between(bins,con_nbeac_w4-sd_con_nbeac_w4,con_nbeac_w4+sd_con_nbeac_w4, facecolor = 'red', alpha = 0.3)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
    ax.set_xlim(0,20)
    ax.set_ylim(0,0.95)
    adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax.set_xticklabels(['0', '10', '20'])
    ax.set_yticklabels(['', '', '',''])
    ax.set_xlabel('Distance (VU)', fontsize=16, labelpad=18)

    ax = fig.add_subplot(1,3,3) #stops per trial
    ax.set_title('Probe', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.axvspan(rz_start, rz_end, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
    ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
    ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
    ax.plot(bins,con_probe_w1,color = 'blue', label = 'Beaconed', linewidth = 2) #plot becaoned trials
    ax.fill_between(bins,con_probe_w1-sd_con_probe_w1,con_probe_w1+sd_con_probe_w1, facecolor = 'blue', alpha = 0.3)
    #ax.plot(bins,con_probe_w4,color = 'red', label = 'Beaconed', linewidth = 2) #plot becaoned trials
    #ax.fill_between(bins,con_probe_w4-sd_con_probe_w4,con_probe_w4+sd_con_probe_w4, facecolor = 'red', alpha = 0.3)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
    ax.set_xlim(0,20)
    ax.set_ylim(0,0.95)
    adjust_spines(ax, ['left','bottom']) # removes top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax.set_yticklabels(['', '', '',''])
    ax.set_xticklabels(['0', '10', '20'])

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.15, right = 0.82, top = 0.85)

    fig.savefig(save_path,  dpi = 200)
    plt.close()
    '''

def main():

    print('-------------------------------------------------------------')

    server_path = "Z:\ActiveProjects\Harry\MouseVR\data\Cue_conditioned_cohort1_190902"
    load_processed_position_all_days(server_path, Mouse_paths.M2_paths(), mouse="M2")
    load_processed_position_all_days(server_path, Mouse_paths.M3_paths(), mouse="M3")
    load_processed_position_all_days(server_path, Mouse_paths.M4_paths(), mouse="M4")
    load_processed_position_all_days(server_path, Mouse_paths.M5_paths(), mouse="M5")

    plot_first_stop_days()


    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()