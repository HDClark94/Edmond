import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utility
import Mouse_paths
import spikeinterface as si
import settings
from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe
from probeinterface import get_probe
import probeinterface as pi
from scipy.spatial import distance
import spikeinterface.widgets as sw


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

def get_n_closest_waveforms(waveforms, number_of_channels, primary_channel, probe_id, shank_id, n=4):
    if number_of_channels == 16: # presume tetrodes
        geom = pd.read_csv(settings.tetrode_geom_path, header=None).values
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': 5})
        probe.set_device_channel_indices(np.arange(number_of_channels))
        probe_df = probe.to_dataframe()
        probe_df["channel"] = np.arange(1,16+1)

    else: # presume cambridge neurotech P1 probes
        probegroup = ProbeGroup()
        n_probes = int(number_of_channels/64)
        for i in range(n_probes):
            probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
            probe.move([i*2000, 0]) # move the probes far away from eachother
            probe.set_device_channel_indices(np.arange(64)+(64*i))
            probe.set_contact_ids(np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64)+(64*i))
            probegroup.add_probe(probe)
        probe_df = probegroup.to_dataframe()
        probe_df["probe_index"] = np.asarray(probe_df["probe_index"])+1
        probe_df["shank_ids"] = np.asarray(probe_df["shank_ids"], dtype=np.uint8)+1
        probe_df = probe_df.astype({"probe_index": str})
        probe_df = probe_df.astype({"shank_ids": str})
        probe_df = probe_df[(probe_df["probe_index"] == probe_id) & (probe_df["shank_ids"] == shank_id)]
        probe_df["channel"] = np.arange(1,len(probe_df)+1)

    primary_x = probe_df[probe_df["channel"] == primary_channel]["x"].iloc[0]
    primary_y = probe_df[probe_df["channel"] == primary_channel]["y"].iloc[0]

    channel_ids = []
    channel_distances = []
    for i, channel in probe_df.iterrows():
        channel_id = channel["channel"]
        channel_x = channel["x"]
        channel_y = channel["y"]
        dst = distance.euclidean((primary_x, primary_y), (channel_x, channel_y))
        channel_distances.append(dst)
        channel_ids.append(channel_id)
    channel_distances = np.array(channel_distances)
    channel_ids = np.array(channel_ids)
    closest_channel_ids = channel_ids[np.argsort(channel_distances)]
    closest_n = closest_channel_ids[:n]
    closest_n_as_indices = closest_n-1

    # closest n includes primary channel and n-1 of the closest contacts
    #return waveforms[:, :, :]
    return waveforms[:,:, closest_n_as_indices], closest_n, probe_df


def plot_waveforms_from_sever2(sorter_dir_path, save_path, load_closest_channels=False, n_spikes=30):
    import os
    sorter_paths = [f.path for f in os.scandir(sorter_dir_path) if f.is_dir()]

    for sub_path in sorter_paths:
        probe_id = sub_path.split("probe")[-1][0]
        shank_id = sub_path.split("shank")[-1][0]

        Sorter = si.load_extractor(settings.temp_storage_path + '/sorter_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0')
        Recording = si.load_extractor(settings.temp_storage_path + '/processed_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0')
        we = si.extract_waveforms(Recording, Sorter,folder=settings.temp_storage_path + '/waveforms_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0', ms_before=1, ms_after=1, load_if_exists=False, overwrite=True)

        for i in Sorter.get_unit_ids():
            sw.plot_unit_waveforms(we, unit_ids=np.array([int(i)]))
            plt.savefig(save_path + '/by_geometry_waveforms_' + str(probe_id) + str(shank_id) + str(i) + '_plot_waveforms_from_sever2.png', dpi=300)
            plt.close()
    return


def plot_waveforms_from_sever(waveform_path, save_path, load_closest_channels=False, n_spikes=30):
    import os
    waveform_paths = [f for f in os.listdir(waveform_path) if f.endswith('.npy')]
    #waveform_paths = ["waveforms_112.npy"]
    for sub_path in waveform_paths:
        cluster_id = sub_path.split("_")[-1].split(".npy")[0]
        probe_id= cluster_id[0]
        shank_id=cluster_id[1]
        on_shank_id = cluster_id[2:]

        Sorter = si.load_extractor(settings.temp_storage_path + '/sorter_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0')
        Recording = si.load_extractor(settings.temp_storage_path + '/processed_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0')
        we = si.extract_waveforms(Recording, Sorter,folder=settings.temp_storage_path + '/waveforms_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0', ms_before=1, ms_after=1, load_if_exists=False, overwrite=True)
        primary_channel_ids = si.get_template_extremum_channel(we)
        primary_channel = primary_channel_ids[int(on_shank_id)]

        waveforms = np.load(waveform_path+"/"+sub_path)
        if load_closest_channels:
            waveforms, closest_channel_ids, probe_df = get_n_closest_waveforms(waveforms, 128, primary_channel+1, probe_id, shank_id, n=16)
            primary_channel=0

        fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
        y=0; x=0
        for i in range(len(waveforms[0][0])):
            axs[y,x].text(0.9, 0.9, str(closest_channel_ids[i]),horizontalalignment='center',verticalalignment='center', transform=axs[y,x].transAxes)

            spikes_to_plot = np.random.randint(0, len(waveforms), size=n_spikes)
            waveforms_to_plot = waveforms[spikes_to_plot,:, i]
            for j in range(len(waveforms_to_plot)):
                axs[y,x].plot(np.linspace(-1,1,len(waveforms_to_plot[j,:])), waveforms_to_plot[j,:], color="grey")

            if primary_channel == i:
                axs[y,x].plot(np.linspace(-1,1,len(waveforms_to_plot[j,:])), np.nanmean(waveforms_to_plot, axis=0), color="red")
            else:
                axs[y,x].plot(np.linspace(-1, 1, len(waveforms_to_plot[j, :])), np.nanmean(waveforms_to_plot, axis=0), color="black")

            x += 1
            if x % 4 == 0:
                y += 1
                x = 0

        #ax.set_ylabel("% Bias", fontsize=30, labelpad=10)
        #ax.set_xlabel('frequency_thresholds', fontsize=30, labelpad=10)
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #ax.set_xticks([0, 0.5])
        #ax.set_xlim([0, 0.5])
        #ax.set_ylim([-100, 100])
        #ax.yaxis.set_tick_params(labelsize=20)
        #ax.xaxis.set_tick_params(labelsize=20)
        #plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.3, right=0.87, top=0.92)
        #ax.legend(loc='best')
        for ax in fig.get_axes():
            ax.label_outer()

        if load_closest_channels:
            plt.savefig(save_path + '/waveforms_'+str(cluster_id)+'_top.png', dpi=300)
        else:
            plt.savefig(save_path + '/waveforms_' + str(cluster_id) + '.png', dpi=300)
        plt.close()
    return


def plot_waveforms_from_sever_using_probe_geometry(waveform_path, save_path, load_closest_channels=False, n_spikes=30):
    a = pi.get_available_pathways()

    #probe.wiring_to_device('H32>RHD2132')








    import os
    waveform_paths = [f for f in os.listdir(waveform_path) if f.endswith('.npy')]
    #waveform_paths = ["waveforms_112.npy"]
    for sub_path in waveform_paths:
        cluster_id = sub_path.split("_")[1].split(".npy")[0]
        probe_id= cluster_id[0]
        shank_id=cluster_id[1]
        on_shank_id = cluster_id[2:]

        Sorter = si.load_extractor(settings.temp_storage_path + '/sorter_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0')
        Recording = si.load_extractor(settings.temp_storage_path + '/processed_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0')
        we = si.extract_waveforms(Recording, Sorter,folder=settings.temp_storage_path + '/waveforms_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0', ms_before=1, ms_after=1, load_if_exists=False, overwrite=True)

        sw.plot_unit_waveforms(we, unit_ids=np.array([int(on_shank_id)]))
        plt.savefig(save_path + '/by_geometry_waveforms_' + str(cluster_id) + '_2.png', dpi=300)
        plt.close()

        primary_channel_ids = si.get_template_extremum_channel(we)
        primary_channel = primary_channel_ids[int(on_shank_id)]

        waveforms = np.load(waveform_path+"/"+sub_path)
        _, closest_channel_ids, probe_df = get_n_closest_waveforms(waveforms, 128, primary_channel+1, probe_id, shank_id, n=16)

        x_stretch_factor = 10
        y_stretch_factor = 5
        fig = plt.figure()
        fig.set_size_inches(2, 6, forward=True)
        ax = fig.add_subplot(1, 1, 1)
        for i in range(len(waveforms[0][0])):
            x = probe_df["x"].iloc[i]
            y = probe_df["y"].iloc[i]
            spikes_to_plot = np.random.randint(0, len(waveforms), size=n_spikes)
            waveforms_to_plot = waveforms[spikes_to_plot, :, i]
            for j in range(len(waveforms_to_plot)):
                ax.plot(x+(np.linspace(-1, 1, len(waveforms_to_plot[j, :]))*x_stretch_factor), y+(waveforms_to_plot[j, :]*y_stretch_factor), color="grey")
            ax.plot(x+(np.linspace(-1, 1, len(waveforms_to_plot[j,:]))*x_stretch_factor), y+(np.nanmean(waveforms_to_plot, axis=0)*y_stretch_factor), color="red")

        plt.savefig(save_path + '/by_geometry_waveforms_' + str(cluster_id) + '_plot_waveforms_from_sever_using_probe_geometry.png', dpi=300)
        plt.close()

    return

def plot_waveforms_again(waveform_path, save_path):
    import os
    waveform_paths = [f for f in os.listdir(waveform_path) if f.endswith('.npy')]
    #waveform_paths = ["waveforms_112.npy"]
    for sub_path in waveform_paths:
        cluster_id = sub_path.split("_")[1].split(".npy")[0]
        probe_id= cluster_id[0]
        shank_id=cluster_id[1]
        on_shank_id = cluster_id[2:]

        Sorter = si.load_extractor(settings.temp_storage_path + '/sorter_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0')
        Recording = si.load_extractor(settings.temp_storage_path + '/processed_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0')
        we = si.extract_waveforms(Recording, Sorter,folder=settings.temp_storage_path + '/waveforms_probe' + str(probe_id) + '_shank' + str(shank_id) + '_segment0', ms_before=1, ms_after=1, load_if_exists=False, overwrite=True)

        waveforms = we.get_waveforms(unit_id=int(on_shank_id))
        reshaped_waveforms = np.swapaxes(waveforms, 0, 2)

        fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
        y = 0;x = 0
        for i in range(16):
            axs[y,x].plot(reshaped_waveforms[i, :, :], color='lightslategray')
            axs[y,x].plot(np.mean(reshaped_waveforms[i, :, :], 1), color="red")
            x += 1
            if x % 4 == 0:
                y += 1
                x = 0
        plt.savefig(save_path + '/by_geometry_waveforms_' + str(cluster_id) + '_plot_waveforms_again1.png', dpi=300)
        plt.close()



        sw.plot_unit_waveforms(we, unit_ids=np.array([int(on_shank_id)]))
        plt.savefig(save_path + '/by_geometry_waveforms_' + str(cluster_id) + '__plot_waveforms_again2.png', dpi=300)
        plt.close()


def adjust_pvals(p_values):
    import statsmodels.stats.multitest as multitest
    _, correct_p_values, _, _ = multitest.multipletests(p_values, alpha=0.05, method='bonferroni')
    print(correct_p_values)
    return

def main():

    print('-------------------------------------------------------------')
    adjust_pvals(p_values=[0.26, 0.03, 0.001])

    plot_waveforms_again(waveform_path="/home/ubuntu/to_sort/recordings/tmp/waveform_arrays",
                              save_path="/mnt/datastore/Harry/Cohort9_february2023/vr/M16_D1_2023-02-28_17-42-27/Figures/")

    plot_waveforms_from_sever2(sorter_dir_path="/home/ubuntu/to_sort/recordings/tmp",
                              save_path="/mnt/datastore/Harry/Cohort9_february2023/vr/M16_D1_2023-02-28_17-42-27/Figures/",
                              load_closest_channels=False)

    plot_waveforms_from_sever_using_probe_geometry(waveform_path="/home/ubuntu/to_sort/recordings/tmp/waveform_arrays",
                              save_path="/mnt/datastore/Harry/Cohort9_february2023/vr/M16_D1_2023-02-28_17-42-27/Figures/",
                              load_closest_channels=False)


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