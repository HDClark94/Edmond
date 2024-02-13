from sklearn.metrics import accuracy_score
import pandas as pd

from Edmond.eLife_Grid_anchoring_2024.ToyGridCells.generate_periodic_cells import *
from Edmond.eLife_Grid_anchoring_2024.Helpers.array_manipulations import *
from Edmond.eLife_Grid_anchoring_2024.vr_grid_cells import get_max_SNR, distance_from_integer, get_lomb_classifier


def get_classifications(cells, threshold):
    peak_frequency_delta_int = np.array(cells['peak_frequency_delta_int'], dtype=np.float16)
    classified_position = peak_frequency_delta_int<=threshold
    classifications = np.tile(np.array(["D"]), len(cells))
    classifications[classified_position] = "P"
    return classifications

def plot_bias(sim_data, save_path, N_p_cells=500, N_d_cells=500):
    fig = plt.figure()
    fig.set_size_inches(5, 4, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    alphas=[0.333,0.666,1]
    linestyles = ["solid","dashed", "dashdot","dotted"]
    ax.axhline(y=0, color="black", linewidth=3, linestyle="solid", alpha=0.1)
    for m, p_scalar in enumerate(np.unique(sim_data["p_scalar"])):
        p_scalar_sim_data = sim_data[sim_data["p_scalar"] == p_scalar]
        for i, noise in enumerate(np.unique(sim_data["field_noise_sigma"])):
            subset_sim_data = p_scalar_sim_data[p_scalar_sim_data["field_noise_sigma"] == noise]

            # take only 500 of each
            subset_sim_data = pd.concat([subset_sim_data[subset_sim_data["true_classification"]=="P"].head(N_p_cells),
                                         subset_sim_data[subset_sim_data["true_classification"]=="D"].head(N_d_cells)], ignore_index=True)

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

                bias_percentage = (100*(total_number_of_position_cells/total_number_of_cells)) - \
                                  (100*(total_number_of_predicted_position_cells/total_number_of_cells))
                biases.append(bias_percentage)

            ax.plot(frequency_thresholds, biases, label= "s="+str(noise)+",ps="+str(p_scalar), color="black", clip_on=False, linestyle=linestyles[i],alpha=alphas[m])

    #ax.set_ylabel("Bias", fontsize=30, labelpad = 10)
    #ax.set_xlabel('frequency_thresholds', fontsize=30, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlim([0,0.5])
    ax.set_ylim([-100, 100])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    #ax.legend(loc='best')
    plt.savefig(save_path + 'stable_bias.png', dpi=500)
    plt.close()
    return

def plot_prediction_accuracy(sim_data, save_path, N_p_cells=500, N_d_cells=500):
    fig = plt.figure()
    fig.set_size_inches(5, 4, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    alphas=[0.333,0.666,1]
    linestyles = ["solid","dashed", "dashdot","dotted"]
    ax.axhline(y=50, color="red", linewidth=3, linestyle="dashed")
    for m, p_scalar in enumerate(np.unique(sim_data["p_scalar"])):
        p_scalar_sim_data = sim_data[sim_data["p_scalar"] == p_scalar]
        for i, noise in enumerate(np.unique(sim_data["field_noise_sigma"])):
            subset_sim_data = p_scalar_sim_data[p_scalar_sim_data["field_noise_sigma"] == noise]

            # take only 500 of each
            subset_sim_data = pd.concat([subset_sim_data[subset_sim_data["true_classification"]=="P"].head(N_p_cells),
                                         subset_sim_data[subset_sim_data["true_classification"]=="D"].head(N_d_cells)], ignore_index=True)

            true_classifications = np.array(subset_sim_data['true_classification'])

            P_accuracies = []
            D_accuracies = []
            overall_accuracies = []
            frequency_thresholds = np.arange(0,0.52, 0.02)
            for frequency_threshold in frequency_thresholds:
                classsications = get_classifications(subset_sim_data, frequency_threshold)
                acc = 100 * accuracy_score(true_classifications, classsications)
                P_acc = 100 * accuracy_score(true_classifications[true_classifications == "P"], classsications[true_classifications == "P"])
                D_acc = 100 * accuracy_score(true_classifications[true_classifications == "D"], classsications[true_classifications == "D"])

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
    ax.set_xlim([0,0.5])
    ax.set_ylim([0, 100])
    #ax.set_yticks([0, 1])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    #ax.legend(loc='best')
    plt.savefig(save_path + 'stable_accuracy.png', dpi=500)
    plt.close()
    return


def generate_stable_grid_cells(field_noise, p_scalars, save_path, grid_spacings, n_cells, save_large_columns=True):
    # this function creates a dataframe of cells of a certain parametisation and creates the spatial periodograms and rate maps
    simulated_data_set = pd.DataFrame()
    for j in range(len(p_scalars)):
        for i in range(len(field_noise)):
            # calculate the stats for the most random simulated cell
            grid_stability="imperfect"
            switch_coding = "block"
            # generate_cells, set switch probability to 0 so it remains anchored or non-anchored for all trials
            powers_all_cells, centre_trials, track_length, true_classifications_all_cells =\
                switch_grid_cells(switch_coding, grid_stability, grid_spacings, n_cells,
                                  trial_switch_probability=0, field_noise_std=field_noise[i], p_scalar=p_scalars[j])

            for n in range(n_cells):
                avg_power = np.nanmean(powers_all_cells[n], axis=0)
                max_SNR, max_SNR_freq = get_max_SNR(Settings.frequency, avg_power)
                lomb_classifier = get_lomb_classifier(max_SNR, max_SNR_freq, 0, 0.05, numeric=False)

                cell_row = pd.DataFrame()
                #cell_row["mean_firing_rate"] = [np.nanmean(rate_maps[n])]
                cell_row["track_length"] = [track_length]
                cell_row["field_noise_sigma"] = [field_noise[i]]
                cell_row["p_scalar"] = [p_scalars[j]]
                cell_row["grid_spacing"] = [grid_spacings[n]]
                cell_row["lomb_classification"] = [lomb_classifier]
                cell_row["true_classification"] = [true_classifications_all_cells[n][0]]
                #cell_row["rate_maps_smoothened"] = [rate_maps[n]]
                cell_row["spatial_periodogram"] = [powers_all_cells[n]]
                cell_row["centre_trials"] = [centre_trials[n]]
                cell_row=compute_peak_statistics(cell_row)

                if not save_large_columns:
                    #cell_row = cell_row.drop('rate_maps_smoothened', axis=1)
                    cell_row = cell_row.drop('spatial_periodogram', axis=1)
                    cell_row = cell_row.drop('centre_trials', axis=1)
                simulated_data_set = pd.concat([simulated_data_set, cell_row], ignore_index=True)
                print("added_cell_to_dataframe")

    simulated_data_set.to_pickle(save_path+"simulated_grid_cells.pkl")
    return simulated_data_set


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
        peak_frequencies_delta_int.append(max_SNR_freq_delta_int[0])
        peak_powers.append(max_SNR)

    sim_data["peak_frequency"] = peak_frequencies
    sim_data["peak_frequency_delta_int"] = peak_frequencies_delta_int
    sim_data["peak_power"] = peak_powers
    return sim_data

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    save_path = "/mnt/datastore/Harry/Grid_anchoring_eLife_2023/simulated/"
    np.random.seed(0)

    N_p_cells = 500
    N_d_cells = 500

    n_cells = N_p_cells+N_d_cells+200 # make slightly more N_p_cells + N_d_cells as cells are assigned randomly
    grid_spacing_low = 40
    grid_spacing_high = 400
    grid_spacings = np.random.uniform(low=grid_spacing_low, high=grid_spacing_high, size=n_cells)
    sim_data = generate_stable_grid_cells(field_noise=[0, 10, 20, 30], p_scalars=[0.01, 0.1, 1], save_path=save_path,
                                    grid_spacings=grid_spacings, n_cells=n_cells, save_large_columns=False)
    #data_path = "/mnt/datastore/Harry/Vr_grid_cells/simulated_data/grid_data/simulated_grid_cells.pkl"
    #sim_data = pd.read_pickle(data_path)

    sim_data = sim_data[sim_data["field_noise_sigma"] != 5]
    plot_bias(sim_data, save_path, N_p_cells=N_p_cells, N_d_cells=N_d_cells)
    plot_prediction_accuracy(sim_data, save_path, N_p_cells=N_p_cells, N_d_cells=N_d_cells)

if __name__ == '__main__':
    main()
