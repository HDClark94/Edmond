import numpy as np
from matplotlib.ticker import MaxNLocator
from Edmond.Toy_grid_cell.plot_example_periodic_cells import *
from Edmond.Toy_grid_cell.Generate_simulated_grid_cells import generate_spatial_periodogram
import pandas as pd
import scipy
from Edmond.VR_grid_analysis.vr_grid_population_level_plots import add_max_block_lengths, add_mean_block_lengths, add_median_block_lengths
from Edmond.utility_functions.array_manipulations import *
from Edmond.VR_grid_analysis.vr_grid_cells import get_max_SNR, distance_from_integer, get_lomb_classifier, get_rolling_lomb_classifier_for_centre_trial, \
    compress_rolling_stats, get_rolling_lomb_classifier_for_centre_trial2, get_rolling_lomb_classifier_for_centre_trial_frequentist, get_block_lengths_any_code

import matplotlib

def get_classifications(cells, threshold):
    peak_frequency_delta_int = np.array(cells['peak_frequency_delta_int'], dtype=np.float16)
    classified_position = peak_frequency_delta_int<=threshold
    classifications = np.tile(np.array(["D"]), len(cells))
    classifications[classified_position] = "P"
    return classifications

def plot_switch_prediction_accuracy(sim_data, save_path, switch_code="block"):
    sim_data = sim_data[(sim_data["switch_coding"] == switch_code)]
    rolling_window_thresholds = np.array([1, 50, 100, 200, 400, 800, 1000])
    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    alphas = [0.333, 0.666, 1]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]

    # fetch the centre trials (they will be the same for all cells)
    _, centre_trials, _ = generate_spatial_periodogram(sim_data["rate_maps_smoothened"].iloc[0])

    for m, p_scalar in enumerate(np.unique(sim_data["p_scalar"])):
        for i, noise in enumerate(np.unique(sim_data["field_noise_sigma"])):
            subset_sim_data = sim_data[(sim_data["field_noise_sigma"] == noise) & (sim_data["p_scalar"] == p_scalar)]
            true_classifications = np.array(subset_sim_data['true_classification'])

            P_accuracies = np.zeros((len(rolling_window_thresholds), len(subset_sim_data)))
            D_accuracies = np.zeros((len(rolling_window_thresholds), len(subset_sim_data)))
            overall_accuracies = np.zeros((len(rolling_window_thresholds), len(subset_sim_data)))

            for j, rolling_window in enumerate(rolling_window_thresholds):
                for n in range(len(subset_sim_data)):
                    powers = subset_sim_data['spatial_periodogram'].iloc[i]

                    rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
                        get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials,
                                                                     powers=powers,
                                                                     power_threshold=0.05,
                                                                     power_step=Settings.power_estimate_step,
                                                                     track_length=200,
                                                                     n_window_size=rolling_window,
                                                                     lomb_frequency_threshold=0.05)
                    rolling_centre_trials, rolling_classifiers, _ = compress_rolling_stats(centre_trials,
                                                                                           rolling_lomb_classifier)

                    # compare the true classifications to the estimated classifications
                    cell_true_classifications = true_classifications[n]
                    # cell_true_classifications = ignore_end_trials_in_block(cell_true_classifications)
                    cell_true_classifications = cell_true_classifications[1:len(rolling_classifiers) + 1]
                    predicted_classifications = rolling_classifiers

                    total_number_of_position_trials = len(cell_true_classifications[cell_true_classifications == "P"])
                    total_number_of_distance_trials = len(cell_true_classifications[cell_true_classifications == "D"])

                    position_trial_accuracy = 100 * (np.sum(predicted_classifications[cell_true_classifications == "P"] == cell_true_classifications[cell_true_classifications == "P"]) / total_number_of_position_trials)
                    distance_trial_accuracy = 100 * (np.sum(predicted_classifications[cell_true_classifications == "D"] == cell_true_classifications[cell_true_classifications == "D"]) / total_number_of_distance_trials)
                    overall_trial_accuracy = 100 * (np.sum(predicted_classifications == cell_true_classifications) / len(cell_true_classifications))

                    P_accuracies[j, n] = position_trial_accuracy
                    D_accuracies[j, n] = distance_trial_accuracy
                    overall_accuracies[j, n] = overall_trial_accuracy


            ax.plot(rolling_window_thresholds, np.nanmean(P_accuracies, axis=1), alpha=alphas[m], color=Settings.allocentric_color, clip_on=False, linestyle=linestyles[i])
            ax.plot(rolling_window_thresholds, np.nanmean(D_accuracies, axis=1), alpha=alphas[m], color=Settings.egocentric_color, clip_on=False, linestyle=linestyles[i])
            ax.plot(rolling_window_thresholds, np.nanmean(overall_accuracies, axis=1), alpha=alphas[m], color="black", clip_on=False, linestyle=linestyles[i])

    #ax.set_ylabel("% Accuracy", fontsize=30, labelpad = 10)
    #ax.set_xlabel('Freq threshold', fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_xlim([0, 0.5])
    #ax.set_ylim([0, 100])
    #ax.set_yticks([0, 1])
    #ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    #ax.legend(loc='best')
    plt.savefig(save_path + 'switch_accuracy_'+switch_code+'.png', dpi=500)
    plt.close()
    return

def plot_switch_bias(sim_data, save_path, switch_code="block"):
    sim_data = sim_data[(sim_data["switch_coding"] == switch_code)]
    rolling_window_thresholds = np.array([1, 50, 100, 200, 400, 800, 1000])
    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    alphas=[0.333,0.666,1]
    linestyles = ["solid","dashed", "dashdot","dotted"]
    ax.axhline(y=0, color="black", linewidth=3, linestyle="solid", alpha=0.1)

    # fetch the centre trials (they will be the same for all cells)
    _, centre_trials, _ = generate_spatial_periodogram(sim_data["rate_maps_smoothened"].iloc[0])

    for m, p_scalar in enumerate(np.unique(sim_data["p_scalar"])):
        for i, noise in enumerate(np.unique(sim_data["field_noise_sigma"])):
            subset_sim_data = sim_data[(sim_data["field_noise_sigma"] == noise) & (sim_data["p_scalar"] == p_scalar)]
            true_classifications = np.array(subset_sim_data['true_classification'])

            Biases_in_coding = np.zeros((len(rolling_window_thresholds), len(subset_sim_data)))
            for j, rolling_window in enumerate(rolling_window_thresholds):
                for n in range(len(subset_sim_data)):
                    powers = subset_sim_data['spatial_periodogram'].iloc[i]

                    rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
                        get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials,
                                                                     powers=powers,
                                                                     power_threshold=0.05,
                                                                     power_step=Settings.power_estimate_step,
                                                                     track_length=200,
                                                                     n_window_size=rolling_window,
                                                                     lomb_frequency_threshold=0.05)
                    rolling_centre_trials, rolling_classifiers, _ = compress_rolling_stats(centre_trials,
                                                                                           rolling_lomb_classifier)

                    # compare the true classifications to the estimated classifications
                    cell_true_classifications = true_classifications[n]
                    # cell_true_classifications = ignore_end_trials_in_block(cell_true_classifications)
                    cell_true_classifications = cell_true_classifications[1:len(rolling_classifiers) + 1]
                    predicted_classifications = rolling_classifiers

                    total_number_of_position_trials = len(cell_true_classifications[cell_true_classifications == "P"])
                    total_number_of_distance_trials = len(cell_true_classifications[cell_true_classifications == "D"])

                    total_number_of_predicted_position_trials = len(
                        predicted_classifications[predicted_classifications == "P"])
                    total_number_of_predicted_distance_trials = len(
                        predicted_classifications[predicted_classifications == "D"])

                    actual_difference_of_position_and_distance_trials = total_number_of_position_trials - total_number_of_distance_trials
                    predicted_difference_of_position_and_distance_trials = total_number_of_predicted_position_trials - total_number_of_predicted_distance_trials

                    bias = actual_difference_of_position_and_distance_trials - predicted_difference_of_position_and_distance_trials
                    Biases_in_coding[j, n] = bias

            #ax.plot(frequency_thresholds, biases, label= "s="+str(noise)+",gs="+grid_spacings+",ps="+str(p_scalar),
            #                      marker=markers[m], color=cmap(i/len(np.unique(sim_data["field_noise_sigma"]))), clip_on=False, linestyle=linestyle)

            ax.plot(rolling_window_thresholds, np.nanmean(Biases_in_coding,axis=1), color="black", clip_on=False, linestyle=linestyles[i],alpha=alphas[m])

    ax.set_ylabel("Bias", fontsize=30, labelpad = 10)
    ax.set_xlabel('frequency_thresholds', fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    #ax.set_xlim([0,0.5])
    #ax.set_ylim([-1005, 1005])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    #ax.legend(loc='best')
    plt.savefig(save_path + 'switch_bias_'+switch_code+'.png', dpi=500)
    plt.close()
    return


def plot_bias(sim_data, save_path):
    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    markers=[".", "+", "x"]
    alphas=[0.333,0.666,1]
    linestyles = ["solid","dashed", "dashdot","dotted"]
    cmap = matplotlib.cm.get_cmap('autumn')
    ax.axhline(y=0, color="black", linewidth=3, linestyle="solid", alpha=0.1)
    for m, p_scalar in enumerate(np.unique(sim_data["p_scalar"])):
        p_scalar_sim_data = sim_data[sim_data["p_scalar"] == p_scalar]
        for i, noise in enumerate(np.unique(sim_data["field_noise_sigma"])):
            subset_sim_data = p_scalar_sim_data[p_scalar_sim_data["field_noise_sigma"] == noise]

            # take only 500 of each
            subset_sim_data = pd.concat([subset_sim_data[subset_sim_data["true_classification"]=="P"].head(500),
                                         subset_sim_data[subset_sim_data["true_classification"]=="D"].head(500)], ignore_index=True)

            true_classifications = np.array(subset_sim_data['true_classification'])

            biases = []
            frequency_thresholds = np.arange(0,0.52, 0.02)
            for frequency_threshold in frequency_thresholds:
                classsications = get_classifications(subset_sim_data, frequency_threshold)

                total_number_of_cells = len(true_classifications)
                total_number_of_position_cells = len(true_classifications[true_classifications == "P"])
                total_number_of_distance_cells = len(true_classifications[true_classifications == "D"])
                total_number_of_predicted_position_cells = len(classsications[classsications == "P"])
                total_number_of_predicted_distance_cells = len(classsications[classsications == "D"])

                actual_difference_of_position_and_distance_cells = total_number_of_position_cells - total_number_of_distance_cells
                predicted_difference_of_position_and_distance_cells = total_number_of_predicted_position_cells - total_number_of_predicted_distance_cells

                bias = actual_difference_of_position_and_distance_cells - predicted_difference_of_position_and_distance_cells
                biases.append(bias)

            #ax.plot(frequency_thresholds, biases, label= "s="+str(noise)+",gs="+grid_spacings+",ps="+str(p_scalar),
            #                      marker=markers[m], color=cmap(i/len(np.unique(sim_data["field_noise_sigma"]))), clip_on=False, linestyle=linestyle)

            ax.plot(frequency_thresholds, biases, label= "s="+str(noise)+",ps="+str(p_scalar), color="black", clip_on=False, linestyle=linestyles[i],alpha=alphas[m])

    #ax.set_ylabel("Bias", fontsize=30, labelpad = 10)
    #ax.set_xlabel('frequency_thresholds', fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlim([0,0.5])
    ax.set_ylim([-1000, 1000])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    #ax.legend(loc='best')
    plt.savefig(save_path + 'bias.png', dpi=500)
    plt.close()
    return

def plot_prediction_accuracy(sim_data, save_path):
    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    markers = [".", "+", "x"]
    alphas=[0.333,0.666,1]
    linestyles = ["solid","dashed", "dashdot","dotted"]
    ax.axhline(y=50, color="red", linewidth=3, linestyle="dashed")
    for m, p_scalar in enumerate(np.unique(sim_data["p_scalar"])):
        p_scalar_sim_data = sim_data[sim_data["p_scalar"] == p_scalar]
        for i, noise in enumerate(np.unique(sim_data["field_noise_sigma"])):
            subset_sim_data = p_scalar_sim_data[p_scalar_sim_data["field_noise_sigma"] == noise]

            # take only 500 of each
            subset_sim_data = pd.concat([subset_sim_data[subset_sim_data["true_classification"]=="P"].head(500),
                                         subset_sim_data[subset_sim_data["true_classification"]=="D"].head(500)], ignore_index=True)

            true_classifications = np.array(subset_sim_data['true_classification'])

            P_accuracies = []
            D_accuracies = []
            overall_accuracies = []
            frequency_thresholds = np.arange(0,0.52, 0.02)
            for frequency_threshold in frequency_thresholds:
                classsications = get_classifications(subset_sim_data, frequency_threshold)
                acc = 100*((np.sum(classsications == true_classifications))/len(true_classifications))
                P_acc = 100*((np.sum(classsications[true_classifications=="P"] == true_classifications[true_classifications=="P"]))/len(true_classifications[true_classifications=="P"]))
                D_acc = 100*((np.sum(classsications[true_classifications=="D"] == true_classifications[true_classifications=="D"]))/len(true_classifications[true_classifications=="D"]))
                P_accuracies.append(P_acc)
                D_accuracies.append(D_acc)
                overall_accuracies.append(acc)
            P_accuracies = np.array(P_accuracies)
            D_accuracies = np.array(D_accuracies)
            overall_accuracies = np.array(overall_accuracies)

            ax.plot(frequency_thresholds, P_accuracies, label= "s="+str(noise)+",ps="+str(p_scalar), alpha=alphas[m], color=Settings.allocentric_color, clip_on=False, linestyle=linestyles[i])
            ax.plot(frequency_thresholds, D_accuracies, label= "s="+str(noise)+",ps="+str(p_scalar), alpha=alphas[m], color=Settings.egocentric_color, clip_on=False, linestyle=linestyles[i])
            ax.plot(frequency_thresholds, overall_accuracies, label= "s="+str(noise)+",ps="+str(p_scalar), alpha=alphas[m], color="black", clip_on=False, linestyle=linestyles[i])

    #ax.set_ylabel("% Accuracy", fontsize=30, labelpad = 10)
    #ax.set_xlabel('Freq threshold', fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 100])
    #ax.set_yticks([0, 1])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    #ax.legend(loc='best')
    plt.savefig(save_path + 'accuracy.png', dpi=500)
    plt.close()
    return

def plot_ROC(sim_data, save_path, mode="P"):
    import matplotlib
    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(-0.05, 1.05, 0.001), np.arange(-0.05, 1.05, 0.001), color="black", linewidth=1, linestyle="dashed")
    cmap = matplotlib.cm.get_cmap('autumn')
    markers = [".", "+", "x"]
    for m, p_scalar in enumerate(np.unique(sim_data["p_scalar"])):
        for i, noise in enumerate(np.unique(sim_data["field_noise_sigma"])):
            noise_sim_data = sim_data[sim_data["field_noise_sigma"] == noise]
            for grid_spacings, linestyle in zip(["less", "greater"], ["solid", "dotted"]):
                if grid_spacings == "greater":
                    subset_sim_data = noise_sim_data[noise_sim_data["grid_spacing"] > 200]
                elif grid_spacings == "less":
                    subset_sim_data = noise_sim_data[noise_sim_data["grid_spacing"] < 200]
                true_classifications = np.array(subset_sim_data['true_classification'])

                FPRs = []
                TPRs = []
                frequency_thresholds = np.arange(0,0.52, 0.02)
                for frequency_threshold in frequency_thresholds:
                    classsications = get_classifications(subset_sim_data, frequency_threshold)
                    true_positive_rate =  np.sum(classsications[true_classifications == mode] == true_classifications[true_classifications == mode])/len(true_classifications[true_classifications == mode])
                    false_positive_rate=  np.sum(classsications[classsications == mode] != true_classifications[classsications == mode])            /len(true_classifications[true_classifications != mode])
                    FPRs.append(false_positive_rate)
                    TPRs.append(true_positive_rate)
                FPRs = np.array(FPRs)
                TPRs = np.array(TPRs)

                ax.plot(FPRs, TPRs, label= "s="+str(noise)+",gs="+grid_spacings+",ps="+str(p_scalar), marker=markers[m], color=cmap(i/len(np.unique(sim_data["field_noise_sigma"]))), clip_on=False, linestyle=linestyle)

    ax.set_ylabel("TPR", fontsize=30, labelpad = 10)
    ax.set_xlabel('FPR', fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_yticks([0, 1])
    ax.set_xticks([0, 1])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    ax.legend(loc='best')
    plt.savefig(save_path + '/ROC_'+mode+'.png', dpi=300)
    plt.close()
    return



def plot_peak_frequencies(sim_data, real_data, save_path):

    fig = plt.figure()
    fig.set_size_inches(5, 5, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    d_0_simdata = sim_data[(sim_data["true_classification"] == "D") & (sim_data["field_noise_sigma"] == 0)]
    d_5_simdata = sim_data[(sim_data["true_classification"] == "D") & (sim_data["field_noise_sigma"] == 5)]
    d_10_simdata = sim_data[(sim_data["true_classification"] == "D") & (sim_data["field_noise_sigma"] == 10)]
    p_0_simdata = sim_data[(sim_data["true_classification"] == "P") & (sim_data["field_noise_sigma"] == 0)]
    p_5_simdata = sim_data[(sim_data["true_classification"] == "P") & (sim_data["field_noise_sigma"] == 5)]
    p_10_simdata = sim_data[(sim_data["true_classification"] == "P") & (sim_data["field_noise_sigma"] == 10)]

    _, bin_edges = np.histogram(d_0_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

    ax.plot(bin_centres, np.histogram(d_0_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.egocentric_color, linestyle="solid", linewidth=3, label="D,fstd=0")
    ax.plot(bin_centres, np.histogram(d_5_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.egocentric_color, linestyle="dashed", linewidth=3, label="D,fstd=5")
    ax.plot(bin_centres, np.histogram(d_10_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.egocentric_color, linestyle="dotted", linewidth=3, label="D,fstd=10")
    ax.plot(bin_centres, np.histogram(p_0_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.allocentric_color, linestyle="solid", linewidth=3, label="P,fstd=0")
    ax.plot(bin_centres, np.histogram(p_5_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.allocentric_color, linestyle="dashed", linewidth=3, label="P,fstd=5")
    ax.plot(bin_centres, np.histogram(p_10_simdata["peak_frequency_delta_int"], range=(0,0.5), bins=25)[0], color=Settings.allocentric_color, linestyle="dotted", linewidth=3, label="P,fstd=10")

    #ax.set_title("min="+str(np.round(total_accuracies.min(), decimals=2))+"@Hz="+str(frequency_thresholds[np.where(total_accuracies==total_accuracies.min())[0][0]]), fontsize=20)
    ax.set_ylabel("Number of cells", fontsize=30, labelpad = 10)
    ax.set_xlabel('Spatial Frequency', fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, 0.5])
    #ax.set_ylim([0, 1])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    ax.legend(loc='best')
    plt.savefig(save_path + '/peak_frequency_distribution.png', dpi=300)
    plt.close()
    return

def compute_peak_statistics(sim_data):
    peak_frequencies = []
    peak_frequencies_delta_int = []
    peak_powers = []
    for index, row in sim_data.iterrows():
        spatial_periodogram = row["spatial_periodogram"]
        avg_power = np.nanmean(spatial_periodogram, axis=0)
        max_SNR, max_SNR_freq = get_max_SNR(Settings.frequency, avg_power)
        max_SNR_freq_delta_int = distance_from_integer(max_SNR_freq)

        peak_frequencies.append(max_SNR_freq)
        peak_frequencies_delta_int.append(max_SNR_freq_delta_int)
        peak_powers.append(max_SNR)

    sim_data["peak_frequency"] = peak_frequencies
    sim_data["peak_frequency_delta_int"] = peak_frequencies_delta_int
    sim_data["peak_power"] = peak_powers
    return sim_data


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


    data_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/grid_data/simulated_grid_cells.pkl"
    #switch_data_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/grid_data/simulated_remapped_grid_cells.pkl"
    save_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/"
    sim_data = pd.read_pickle(data_path)
    #switch_data = pd.read_pickle(switch_data_path)
    #switch_data = switch_data[switch_data["field_noise_sigma"] != 5]
    #plot_switch_bias(switch_data, save_path, switch_code="block")
    #plot_switch_prediction_accuracy(switch_data, save_path, switch_code="block")
    #plot_switch_bias(switch_data, save_path, switch_code="by_trial")
    #plot_switch_prediction_accuracy(switch_data, save_path, switch_code="by_trial")

    #sim_data = compute_peak_statistics(sim_data)
    sim_data = sim_data[sim_data["field_noise_sigma"] != 5]
    plot_bias(sim_data, save_path)
    plot_prediction_accuracy(sim_data, save_path)


    plot_ROC(sim_data, save_path, mode="P")
    plot_ROC(sim_data, save_path, mode="D")

    real_data = pd.DataFrame()
    real_data = pd.concat([real_data, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort6.pkl")], ignore_index=True)
    real_data = pd.concat([real_data, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort7.pkl")], ignore_index=True)
    real_data = pd.concat([real_data, pd.read_pickle("/mnt/datastore/Harry/Vr_grid_cells/combined_cohort8.pkl")], ignore_index=True)
    real_data = real_data[real_data["snippet_peak_to_trough"] < 500] # uV
    real_data = real_data[real_data["track_length"] == 200]
    real_data = real_data[real_data["n_trials"] >= 10]


    plot_peak_frequencies(sim_data, real_data, save_path=save_path)

if __name__ == '__main__':
    main()
