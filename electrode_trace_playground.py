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
import spikeinterface.extractors as se
import spikeinterface.sorters as sorters
import spikeinterface.preprocessing as spre
import os
import OpenEphys
import glob
import shutil
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from functools import partial
import math

def getDeadChannel(deadChannelFile):
    deadChannels = []
    if os.path.exists(deadChannelFile):
        with open(deadChannelFile,'r') as f:
            deadChannels = [int(s) for s in f.readlines()]
    return deadChannels

def count_files_that_match_in_folder(folder, data_file_prefix, data_file_suffix):
    file_names = os.listdir(folder)
    matches=0
    for i in range(len(file_names)):
        if ((data_file_prefix in  file_names[i]) and (file_names[i].endswith(data_file_suffix))):
            corrected_data_file_suffix = file_names[i].split(data_file_prefix)[0]+data_file_prefix # corrects the data prefix, important if recordings vary e.g. 100_CH1, 101_CH1
            matches += 1
    return matches, corrected_data_file_suffix

def add_probe_info_by_shank(recordingExtractor, shank_df):
    shank_df = shank_df.sort_values(by="channel", ascending=True)
    x = shank_df["x"].values
    y = shank_df["y"].values
    geom = np.transpose(np.vstack((x,y)))
    probe = Probe(ndim=2, si_units='um')
    if shank_df["contact_shapes"].iloc[0] == "rect":
        probe.set_contacts(positions=geom, shapes='rect', shape_params={'width': shank_df["width"].iloc[0],
                                                                        'height': shank_df["height"].iloc[0]})
    elif shank_df["contact_shapes"].iloc[0] == "circle":
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': shank_df["radius"].iloc[0]})
    probe.set_device_channel_indices(np.arange(len(shank_df)))
    recordingExtractor.set_probe(probe, in_place=True)
    return recordingExtractor

def load_OpenEphysRecording(folder, channel_ids=None):
    number_of_channels, corrected_data_file_suffix = count_files_that_match_in_folder(folder, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    if channel_ids is None:
        channel_ids = np.arange(1, number_of_channels+1)

    channel_ids = np.sort(channel_ids)
    signal = []
    for i, channel_id in enumerate(channel_ids):
        fname = folder+'/'+corrected_data_file_suffix+str(channel_id)+settings.data_file_suffix+'.continuous'
        x = OpenEphys.loadContinuousFast(fname)['data']
        if i==0:
            #preallocate array on first run
            signal = np.zeros((x.shape[0], len(channel_ids)))
        signal[:,i] = x
    return [signal]

def get_wiring_for_cambridgeneurotech_ASSY_236_P_1(contact_id, probe_id):
    contact_id = contact_id-(64*probe_id)

    if contact_id == 1:
        return 41 + (64*probe_id)
    elif contact_id == 2:
        return 39 + (64*probe_id)
    elif contact_id == 3:
        return 38 + (64*probe_id)
    elif contact_id == 4:
        return 37 + (64*probe_id)
    elif contact_id == 5:
        return 35 + (64*probe_id)
    elif contact_id == 6:
        return 34 + (64*probe_id)
    elif contact_id == 7:
        return 33 + (64*probe_id)
    elif contact_id == 8:
        return 32 + (64*probe_id)
    elif contact_id == 9:
        return 29 + (64*probe_id)
    elif contact_id == 10:
        return 30 + (64*probe_id)
    elif contact_id == 11:
        return 28 + (64*probe_id)
    elif contact_id == 12:
        return 26 + (64*probe_id)
    elif contact_id == 13:
        return 25 + (64*probe_id)
    elif contact_id == 14:
        return 24 + (64*probe_id)
    elif contact_id == 15:
        return 22 + (64*probe_id)
    elif contact_id == 16:
        return 20 + (64*probe_id)
    elif contact_id == 17:
        return 46 + (64*probe_id)
    elif contact_id == 18:
        return 45 + (64*probe_id)
    elif contact_id == 19:
        return 44 + (64*probe_id)
    elif contact_id == 20:
        return 43 + (64*probe_id)
    elif contact_id == 21:
        return 42 + (64*probe_id)
    elif contact_id == 22:
        return 40 + (64*probe_id)
    elif contact_id == 23:
        return 36 + (64*probe_id)
    elif contact_id == 24:
        return 31 + (64*probe_id)
    elif contact_id == 25:
        return 27 + (64*probe_id)
    elif contact_id == 26:
        return 23 + (64*probe_id)
    elif contact_id == 27:
        return 21 + (64*probe_id)
    elif contact_id == 28:
        return 18 + (64*probe_id)
    elif contact_id == 29:
        return 19 + (64*probe_id)
    elif contact_id == 30:
        return 17 + (64*probe_id)
    elif contact_id == 31:
        return 16 + (64*probe_id)
    elif contact_id == 32:
        return 14 + (64*probe_id)
    elif contact_id == 33:
        return 55 + (64*probe_id)
    elif contact_id == 34:
        return 53 + (64*probe_id)
    elif contact_id == 35:
        return 54 + (64*probe_id)
    elif contact_id == 36:
        return 52 + (64*probe_id)
    elif contact_id == 37:
        return 51 + (64*probe_id)
    elif contact_id == 38:
        return 50 + (64*probe_id)
    elif contact_id == 39:
        return 49 + (64*probe_id)
    elif contact_id == 40:
        return 48 + (64*probe_id)
    elif contact_id == 41:
        return 47 + (64*probe_id)
    elif contact_id == 42:
        return 15 + (64*probe_id)
    elif contact_id == 43:
        return 13 + (64*probe_id)
    elif contact_id == 44:
        return 12 + (64*probe_id)
    elif contact_id == 45:
        return 11 + (64*probe_id)
    elif contact_id == 46:
        return 9 + (64*probe_id)
    elif contact_id == 47:
        return 10 + (64*probe_id)
    elif contact_id == 48:
        return 8 + (64*probe_id)
    elif contact_id == 49:
        return 63 + (64*probe_id)
    elif contact_id == 50:
        return 62 + (64*probe_id)
    elif contact_id == 51:
        return 61 + (64*probe_id)
    elif contact_id == 52:
        return 60 + (64*probe_id)
    elif contact_id == 53:
        return 59 + (64*probe_id)
    elif contact_id == 54:
        return 58 + (64*probe_id)
    elif contact_id == 55:
        return 57 + (64*probe_id)
    elif contact_id == 56:
        return 56 + (64*probe_id)
    elif contact_id == 57:
        return 7 + (64*probe_id)
    elif contact_id == 58:
        return 6 + (64*probe_id)
    elif contact_id == 59:
        return 5 + (64*probe_id)
    elif contact_id == 60:
        return 4 + (64*probe_id)
    elif contact_id == 61:
        return 3 + (64*probe_id)
    elif contact_id == 62:
        return 2 + (64*probe_id)
    elif contact_id == 63:
        return 1 + (64*probe_id)
    elif contact_id == 64:
        return 0 + (64*probe_id)
    else:
        print("contact is invalid")

def get_wiring(contact_ids, probes_ids, probe_manufacturer, probe_type):
    # add wiring info here when new experiments use different probes
    wiring_ids = []
    if probe_manufacturer == "cambridgeneurotech" and probe_type == "ASSY-236-P-1":
        for contact_id, probe_id in zip(contact_ids, probes_ids):
            corresponding_wiring_id = get_wiring_for_cambridgeneurotech_ASSY_236_P_1(contact_id, probe_id)
            wiring_ids.append(corresponding_wiring_id)
    else:
        print("The given probe_manufacturer and probe_type do not have the wiring set yet"
              "Check the arguments and add the wiring here if not yet added")
    return np.array(wiring_ids)

def get_probe_dataframe(number_of_channels):
    if number_of_channels == 16: # presume tetrodes
        geom = pd.read_csv(settings.tetrode_geom_path, header=None).values
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': 5})
        probe.set_device_channel_indices(np.arange(number_of_channels))
        probe_df = probe.to_dataframe()
        probe_df["channel"] = np.arange(1,16+1)
        probe_df["shank_ids"] = 1
        probe_df["probe_index"] = 1

    else: # presume cambridge neurotech P1 probes
        assert number_of_channels%64==0
        probegroup = ProbeGroup()
        n_probes = int(number_of_channels/64)
        for i in range(n_probes):
            probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
            probe.move([i*2000, 0]) # move the probes far away from eachother
            contact_ids = np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64)+(64*i)
            #probe.set_device_channel_indices(get_wiring(contact_ids, 'cambridgeneurotech', 'ASSY-236-P-1', probe_i=i))
            probe.set_contact_ids(contact_ids)
            probegroup.add_probe(probe)
        probe_df = probegroup.to_dataframe()
        probe_df = probe_df.astype({"probe_index": int, "shank_ids": int, "contact_ids": int})
        probe_df["channel"] = get_wiring(probe_df["contact_ids"], probe_df["probe_index"], 'cambridgeneurotech', 'ASSY-236-P-1')+1
        probe_df["shank_ids"] = (np.asarray(probe_df["shank_ids"])+1).tolist()
        probe_df["probe_index"] = (np.asarray(probe_df["probe_index"])+1).tolist()
    return probe_df


def plot_voltage_traces_from_recording_by_shank(recording, save_path):
    # by shark
    n_channels, _ = count_files_that_match_in_folder(recording, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    probe_group_df = get_probe_dataframe(n_channels)
    bad_channel_ids = getDeadChannel(recording +'/dead_channels.txt')

    for probe_index in np.unique(probe_group_df["probe_index"]):
        print("I am subsetting the recording and analysing probe "+str(probe_index))
        probe_df = probe_group_df[probe_group_df["probe_index"] == probe_index]
        for shank_id in np.unique(probe_df["shank_ids"]):
            print("I am looking at shank "+str(shank_id))
            shank_df = probe_df[probe_df["shank_ids"] == shank_id]
            channels_in_shank = np.array(shank_df["channel"])
            base_signal_shank = load_OpenEphysRecording(recording, channel_ids=channels_in_shank)
            base_shank_recording = se.NumpyRecording(base_signal_shank,settings.sampling_rate)
            base_shank_recording = add_probe_info_by_shank(base_shank_recording, shank_df)
            base_shank_recording = spre.whiten(base_shank_recording)
            base_shank_recording = spre.bandpass_filter(base_shank_recording, freq_min=300, freq_max=2000)
            base_shank_recording.remove_channels(bad_channel_ids)
            base_shank_recording = base_shank_recording.save(folder= '/home/ubuntu/tmp/processed_probe'+str(probe_index)+'_shank'+str(shank_id)+'_segment0',
                                                             n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)

            traces = base_shank_recording.get_traces(start_frame=4*60*settings.sampling_rate, end_frame=7*60*settings.sampling_rate)

            plt.rcParams["figure.figsize"] = [24, 12]
            plt.rcParams["figure.autolayout"] = True

            fig = plt.figure()
            ax = plt.axes(xlim=(0, 2), ylim=(0, 17))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            plt.axis('off')

            N = len(traces[0])
            x = np.linspace(0, 2, settings.sampling_rate*2)
            lines = [plt.plot(x, np.zeros(len(x)))[0] for _ in range(N)]
            patches = lines
            interval_in_ms = 100

            def animate(i, x=None, traces=None, interval_in_ms=None):
                gain_factor = 0.1
                for j, line in enumerate(lines):
                    line.set_data(x, j+1+(traces[(60*settings.sampling_rate)+(i*int(settings.sampling_rate*(interval_in_ms/1000))):
                                                 (60*settings.sampling_rate)+int(len(x))+(i*int(settings.sampling_rate*(interval_in_ms/1000))), j] *gain_factor))
                return patches

            anim = FuncAnimation(fig, partial(animate, x=x, traces=traces, interval_in_ms=interval_in_ms), frames=1000, interval=interval_in_ms, blit=True)

            anim.save(save_path+'traces_probe'+str(probe_index)+'_shank'+str(shank_id)+'_by_shank.mp4',writer='ffmpeg', fps=15, dpi=50)
            print("I have made a video snippet")
    return

def plot_voltage_traces_from_recording_by_probe(recording, save_path, min_start=0, min_end=2):
    # by shark
    n_channels, _ = count_files_that_match_in_folder(recording, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    probe_group_df = get_probe_dataframe(n_channels)
    bad_channel_ids = getDeadChannel(recording +'/dead_channels.txt')

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cycle_colors = [np.concatenate((np.repeat(cycle[0], 16),np.repeat(cycle[1], 16), np.repeat(cycle[2], 16), np.repeat(cycle[3], 16))),
                    np.concatenate((np.repeat(cycle[4], 16),np.repeat(cycle[5], 16), np.repeat(cycle[6], 16), np.repeat(cycle[7], 16)))]

    for pi, probe_index in enumerate(np.unique(probe_group_df["probe_index"])):
        print("I am subsetting the recording and analysing probe "+str(probe_index))
        probe_df = probe_group_df[probe_group_df["probe_index"] == probe_index]
        probe_df = probe_df.sort_values(by="shank_ids")
        channels_on_probe = np.array(probe_df["channel"])
        base_signal_probe = load_OpenEphysRecording(recording, channel_ids=channels_on_probe)
        base_probe_recording = se.NumpyRecording(base_signal_probe,settings.sampling_rate)
        base_probe_recording = add_probe_info_by_shank(base_probe_recording, probe_df)
        base_probe_recording = spre.whiten(base_probe_recording)
        base_probe_recording = spre.bandpass_filter(base_probe_recording, freq_min=300, freq_max=2000)
        base_probe_recording.remove_channels(bad_channel_ids)
        base_probe_recording = base_probe_recording.save(folder= '/home/ubuntu/tmp/processed_probe'+str(probe_index)+'_segment0',
                                                         n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)

        traces = base_probe_recording.get_traces(start_frame=min_start*60*settings.sampling_rate, end_frame=min_end*60*settings.sampling_rate)

        plt.rcParams["figure.figsize"] = [24, 48]
        plt.rcParams["figure.autolayout"] = True

        fig = plt.figure()
        ax = plt.axes(xlim=(0, 2), ylim=(0, 65))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.axis('off')

        N = len(traces[0])
        x = np.linspace(0, 2, settings.sampling_rate*2)
        lines = [plt.plot(x, np.zeros(len(x)), color=cycle_colors[pi][i])[0] for i in range(N)]
        patches = lines
        interval_in_ms = 100

        def animate(i, x=None, traces=None, interval_in_ms=None):
            gain_factor = 0.1
            for j, line in enumerate(lines):
                line.set_data(x, j+1+(traces[(60*settings.sampling_rate)+(i*int(settings.sampling_rate*(interval_in_ms/1000))):
                                             (60*settings.sampling_rate)+int(len(x))+(i*int(settings.sampling_rate*(interval_in_ms/1000))), j] *gain_factor))
            return patches

        anim = FuncAnimation(fig, partial(animate, x=x, traces=traces, interval_in_ms=interval_in_ms), frames=1000, interval=interval_in_ms, blit=True)

        anim.save(save_path+'traces_probe'+str(probe_index)+'_by_probe.mp4',writer='ffmpeg', fps=10, dpi=50)
        print("I have made a video snippet")
    return

def remove_tmp_files():
    for path in glob.glob('/home/ubuntu/tmp/*'):
        shutil.rmtree(path)

def plot_spike_raster_vs_track_position(spatial_firing, save_path):
    mec_color="blue"
    ca1_color="green"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax_i=0
    for probe, color in zip([2, 1], [ca1_color, mec_color]):
        for shank in [1,2,3,4]:
            shank_firing = spatial_firing[(spatial_firing["probe_id"] == str(probe)) &
                                          (spatial_firing["shank_id"] == str(shank))]

            for i in range(len(shank_firing)):
                ax.scatter(x=shank_firing["x_position_cm"].iloc[i],
                           y=np.ones(len(shank_firing['x_position_cm'].iloc[i]))+ax_i,
                           marker="|", color=color, linewidths=0.2, alpha=0.2)
                ax_i+=0.1

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0)
    ax.margins(y=0)
    ax.set_xlim(0,200)
    ax.axvspan(0, 30, facecolor='k', linewidth=0, alpha=.15)  # black box
    ax.axvspan(200 - 110, 200 - 90, facecolor='DarkGreen', alpha=.15, linewidth=0)
    ax.axvspan(200 - 30, 200, facecolor='k', linewidth=0, alpha=.15)  # black box
    #ax.set_xlim(0, 5*60*30000)
    #ax.set_x_lim[time_interval_min, time_interval_max]
    plt.subplots_adjust(hspace=None, wspace=None, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/spike_raster_vs_track_position.png', dpi=800 ,bbox_inches='tight', pad_inches=0)
    plt.close()

    for tt in np.arange(1,45):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax_i=0
        for probe, color in zip([2, 1], [ca1_color, mec_color]):
            for shank in [1,2,3,4]:
                shank_firing = spatial_firing[(spatial_firing["probe_id"] == str(probe)) &
                                              (spatial_firing["shank_id"] == str(shank))]

                for i in range(len(shank_firing)):
                    x_position_cm = np.array(shank_firing["x_position_cm"].iloc[i])
                    trial_numbers = np.array(shank_firing["trial_number"].iloc[i])
                    x_position_cm = x_position_cm[trial_numbers==tt]

                    ax.scatter(x=x_position_cm,
                               y=np.ones(len(x_position_cm))+ax_i,
                               marker="|", color=color, linewidths=0.2, alpha=0.2)
                    ax_i+=0.1

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(x=0)
        ax.margins(y=0)
        ax.set_xlim(0,200)
        ax.axvspan(0, 30, facecolor='k', linewidth=0, alpha=.15)  # black box
        ax.axvspan(200 - 110, 200 - 90, facecolor='DarkGreen', alpha=.15, linewidth=0)
        ax.axvspan(200 - 30, 200, facecolor='k', linewidth=0, alpha=.15)  # black box
        #ax.set_xlim(0, 5*60*30000)
        #ax.set_x_lim[time_interval_min, time_interval_max]
        plt.subplots_adjust(hspace=None, wspace=None, bottom=None, left=None, right=None, top=None)
        plt.savefig(save_path + '/spike_raster_vs_track_position_'+str(tt)+'.png', dpi=800,bbox_inches='tight', pad_inches=0)
        plt.close()
    return

def plot_spike_raster2(spatial_firing, save_path, suffix=""):
    mec_color="blue"
    ca1_color="green"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax_i=0
    for probe, color in zip([2, 1], [ca1_color, mec_color]):
        for shank in [1,2,3,4]:
            shank_firing = spatial_firing[(spatial_firing["probe_id"] == str(probe)) &
                                          (spatial_firing["shank_id"] == str(shank))]

            for i in range(len(shank_firing)):
                ax.scatter(x=shank_firing["firing_times"].iloc[i],
                           y=np.ones(len(shank_firing["firing_times"].iloc[i]))+ax_i,
                           marker="|", color=color, linewidths=0.5, alpha=1)
                ax_i+=0.1

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0)
    ax.margins(y=0)
    #ax.set_ylim(1.5, 3.4)
    ax.set_xlim(0, 20*60*30000)
    #ax.set_x_lim[time_interval_min, time_interval_max]
    plt.subplots_adjust(hspace=None, wspace=None, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/spike_raster2'+suffix+'.png', dpi=800)
    plt.close()
    return

def plot_spike_raster(spatial_firing, save_path):
    cm = plt.get_cmap('gist_rainbow')

    fig, ax = plt.subplots(figsize=(6, 2))
    ax_i = 0
    ax_jump = True
    for probe in [2, 1]:
        # handle the middle gap between probes
        if (probe ==2) and ax_jump:
            ax_i+=0.7
            ax_jump=False

        for shank in [1,2,3,4]:
            shank_firing = spatial_firing[(spatial_firing["probe_id"] == str(probe)) &
                                          (spatial_firing["shank_id"] == str(shank))]

            # generate colors
            NUM_COLORS = len(shank_firing)
            colors=[]
            for i in range(len(shank_firing)):
                colors.append(cm(i/NUM_COLORS))
            colors=np.array(colors)
            np.random.shuffle(colors)

            for i in range(len(shank_firing)):
                ax.scatter(x=shank_firing["firing_times"].iloc[i],
                           y=np.ones(len(shank_firing["firing_times"].iloc[i]))+ax_i,
                           marker="|", color=colors[i], linewidths=0.2)
            ax_i+=0.2

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0)
    ax.margins(y=0)
    ax.set_ylim(1.5, 3.4)
    ax.set_xlim(0, 5*60*30000)
    #ax.set_x_lim[time_interval_min, time_interval_max]
    plt.subplots_adjust(hspace=None, wspace=None, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/spike_raster.png', dpi=800)
    plt.close()
    return


def calculate_autocorrelogram_hist(spikes, bin_size, window):
    half_window = int(window/2)
    number_of_bins = int(math.ceil(spikes[-1]*1000))
    train = np.zeros(number_of_bins)
    bins = np.zeros(len(spikes))

    for spike in range(len(spikes)-1):
        bin = math.floor(spikes[spike]*1000)
        train[bin] = train[bin] + 1
        bins[spike] = bin

    counts = np.zeros(window+1)
    counted = 0
    for b in range(len(bins)):
        bin = int(bins[b])
        window_start = int(bin - half_window)
        window_end = int(bin + half_window + 1)
        if (window_start > 0) and (window_end < len(train)):
            counts = counts + train[window_start:window_end]
            counted = counted + sum(train[window_start:window_end]) - train[bin]

    counts[half_window] = 0
    if max(counts) == 0 and counted == 0:
        counted = 1

    corr = counts / counted
    time = np.arange(-half_window, half_window + 1, bin_size)
    return corr, time


def get_10ms_autocorr(firing_times_cluster, sampling_rate):
    corr1, time1 = calculate_autocorrelogram_hist(np.array(firing_times_cluster) / sampling_rate, 1, 20)
    return corr1, time1


def get_250ms_autocorr(firing_times_cluster, sampling_rate):
    corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster) / sampling_rate, 1, 500)
    return corr, time

def plot_autocorrelations(spike_data, save_path):
    fig, axes = plt.subplots(nrows=len(spike_data), ncols=2, figsize=(1, 6))
    i=0
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        firing_times_cluster = cluster_df['firing_times'].iloc[0]
        if len(firing_times_cluster)>1: # only calculate autocorr if there are any spikes
            corr_10, time_10 = get_10ms_autocorr(firing_times_cluster, 30000)
            corr_250, time_250 = get_250ms_autocorr(firing_times_cluster, 30000)
            axes[i, 0].bar(time_10, corr_10, width=1, color='black')
            axes[i, 1].bar(time_250, corr_250, width=1,  color='black')
        for j in [0,1]:
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].spines['bottom'].set_visible(False)
            axes[i, j].spines['left'].set_visible(False)
            axes[i, j].xaxis.set_ticks_position('bottom')
            axes[i, j].yaxis.set_ticks_position('left')
            axes[i, j].set_yticks([])
            axes[i, j].set_xticks([])
        i+=1
    #plt.xticks([0, 30], [0, 1])
    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)
    #plt.xlabel('Time (ms)', fontsize=14)
    #plt.ylabel('Voltage (µV)', fontsize=14)
    plt.subplots_adjust(hspace=None, wspace=None, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/autocorrelations.png', dpi=800,bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_all_waveforms(spike_data, save_path):
    fig, axes = plt.subplots(nrows=len(spike_data), ncols=4, figsize=(1, 6))
    i=0
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        max_channel = cluster_df['primary_channel'].iloc[0]
        sd = np.std(cluster_df['random_snippets'].iloc[0][max_channel - 1, :, :] * -1)
        highest_value = np.mean(cluster_df['random_snippets'].iloc[0][max_channel - 1, :, :] * -1) + (sd * 4)
        lowest_value = np.mean(cluster_df['random_snippets'].iloc[0][max_channel - 1, :, :] * -1) - (sd * 4)

        for channel in range(4):
            axes[i, channel].plot(cluster_df["random_snippets"].iloc[0][channel, :, :] * -1, color='lightslategray', linewidth=0.5, alpha=0.3)
            axes[i, channel].plot(np.mean(cluster_df["random_snippets"].iloc[0][channel, :, :], 1) * -1, color="red", linewidth=0.5)
            axes[i, channel].spines['top'].set_visible(False)
            axes[i, channel].spines['right'].set_visible(False)
            axes[i, channel].spines['bottom'].set_visible(False)
            axes[i, channel].spines['left'].set_visible(False)
            axes[i, channel].xaxis.set_ticks_position('bottom')
            axes[i, channel].yaxis.set_ticks_position('left')
            axes[i, channel].set_yticks([])
            axes[i, channel].set_xticks([])
        i+=1
    #plt.xticks([0, 30], [0, 1])
    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)
    #plt.xlabel('Time (ms)', fontsize=14)
    #plt.ylabel('Voltage (µV)', fontsize=14)
    plt.subplots_adjust(hspace=None, wspace=None, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/waveforms.png', dpi=600,bbox_inches='tight', pad_inches=0)
    plt.close()

def main():

    print('-------------------------------------------------------------')

    plot_spike_raster2(spatial_firing=pd.read_pickle(
        "/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D1_2023-10-30_12-38-29/mountainsort4/DataFrames/spatial_firing.pkl"),
                      save_path="/mnt/datastore/Harry/test_recording/M18_D1_2023-10-30_12-38-29/Figures", suffix="_vr")

    plot_spike_raster2(spatial_firing=pd.read_pickle(
        "/mnt/datastore/Harry/Cohort10_october2023/of/M18_D1_2023-10-30_13-25-44/mountainsort4/DataFrames/spatial_firing.pkl"),
                      save_path="/mnt/datastore/Harry/test_recording/M18_D1_2023-10-30_12-38-29/Figures", suffix="_of")


    #remove_tmp_files()
    #plot_voltage_traces_from_recording_by_probe(recording="/mnt/datastore/Harry/Cohort9_february2023/of/M17_2023-05-16_15-43-04",
    #                          save_path="/mnt/datastore/Harry/Cohort9_february2023/of/M17_2023-05-16_15-43-04/Figures/", min_start=1, min_end=2)

    #plot_voltage_traces_from_recording_by_probe(recording="/mnt/datastore/Harry/Cohort9_february2023/vr/M16_D1_2023-02-28_17-42-27",
    #                                            save_path="/mnt/datastore/Harry/Cohort9_february2023/vr/M16_D1_2023-02-28_17-42-27/Figures/", min_start=4, min_end=7)

    #plot_spike_raster(spatial_firing=pd.read_pickle("/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D1_2023-10-30_12-38-29/mountainsort4/DataFrames/spatial_firing.pkl"),
    #                  save_path="/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D1_2023-10-30_12-38-29/mountainsort4/Figures")
    plot_autocorrelations(spike_data=pd.read_pickle(
        "/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D1_2023-10-30_12-38-29/mountainsort4/DataFrames/spatial_firing.pkl"),
                          save_path="/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D1_2023-10-30_12-38-29/mountainsort4/Figures")

    plot_all_waveforms(spike_data=pd.read_pickle("/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D1_2023-10-30_12-38-29/mountainsort4/DataFrames/spatial_firing.pkl"),
                       save_path="/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D1_2023-10-30_12-38-29/mountainsort4/Figures")

    plot_spike_raster_vs_track_position(spatial_firing=pd.read_pickle(
        "/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D1_2023-10-30_12-38-29/mountainsort4/DataFrames/spatial_firing.pkl"),
        save_path="/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D1_2023-10-30_12-38-29/mountainsort4/Figures")

    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()