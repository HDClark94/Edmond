from sklearn.metrics import accuracy_score

from Edmond.eLife_Grid_anchoring_2024.ToyGridCells.generate_periodic_cells import *
from Edmond.eLife_Grid_anchoring_2024.Helpers.array_manipulations import *
from Edmond.eLife_Grid_anchoring_2024.vr_grid_cells import get_rolling_lomb_classifier_for_centre_trial, compress_rolling_stats

def run_assay(switch_coding_mode, grid_stability, save_path, grid_spacing_low, grid_spacing_high, n_cells, trial_switch_probability, freq_thres=0.05, plot_suffix=""):
    grid_spacings = np.random.uniform(low=grid_spacing_low, high=grid_spacing_high, size=n_cells);
    rolling_window_sizes = np.array([1, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000])

    field_noise_stds = [0, 10, 20, 30]
    p_scalars=[0.01, 0.1, 1]

    Biases_in_coding = np.zeros((len(p_scalars), len(field_noise_stds), len(rolling_window_sizes), n_cells))
    overall_coding_accuracy = np.zeros((len(p_scalars), len(field_noise_stds), len(rolling_window_sizes), n_cells))
    position_coding_accuracy = np.zeros((len(p_scalars), len(field_noise_stds), len(rolling_window_sizes), n_cells))
    distance_coding_accuracy = np.zeros((len(p_scalars), len(field_noise_stds), len(rolling_window_sizes), n_cells))

    for j, p_scalar in enumerate(np.unique(p_scalars)):
        for n, noise in enumerate(np.unique(field_noise_stds)):

            # first lets generate n cells
            powers_all_cells, centre_trials, track_length, true_classifications = \
                switch_grid_cells(switch_coding_mode, grid_stability, grid_spacings, n_cells,
                                  trial_switch_probability=trial_switch_probability, field_noise_std=noise, p_scalar=p_scalar)

            for m in range(len(rolling_window_sizes)):
                for i in range(n_cells):
                    rolling_lomb_classifier, _, _, rolling_frequencies, rolling_points = \
                        get_rolling_lomb_classifier_for_centre_trial(centre_trials=centre_trials, powers=powers_all_cells[i], power_threshold=0.05,
                                                                 power_step=Settings.power_estimate_step, track_length=track_length,
                                                                 n_window_size=rolling_window_sizes[m], lomb_frequency_threshold=freq_thres)
                    rolling_centre_trials, rolling_classifiers, _ = compress_rolling_stats(centre_trials, rolling_lomb_classifier)

                    # compare the true classifications to the estimated classifications
                    cell_true_classifications = true_classifications[i]
                    #cell_true_classifications = ignore_end_trials_in_block(cell_true_classifications)
                    cell_true_classifications = cell_true_classifications[1:len(rolling_classifiers)+1]
                    predicted_classifications = rolling_classifiers

                    total_number_of_trials = len(cell_true_classifications)
                    total_number_of_position_trials = len(cell_true_classifications[cell_true_classifications=="P"])
                    total_number_of_distance_trials = len(cell_true_classifications[cell_true_classifications=="D"])

                    total_number_of_predicted_position_trials = len(predicted_classifications[predicted_classifications=="P"])
                    total_number_of_predicted_distance_trials = len(predicted_classifications[predicted_classifications=="D"])

                    bias_percentage = (100 * (total_number_of_position_trials / total_number_of_trials)) - \
                                      (100 * (total_number_of_predicted_position_trials / total_number_of_trials))

                    Biases_in_coding[j, n, m, i] = bias_percentage
                    overall_coding_accuracy[j, n, m, i] = 100*accuracy_score(cell_true_classifications, predicted_classifications)
                    position_coding_accuracy[j, n, m, i] = 100*accuracy_score(cell_true_classifications[cell_true_classifications=="P"], predicted_classifications[cell_true_classifications=="P"])
                    distance_coding_accuracy[j, n, m, i] = 100*accuracy_score(cell_true_classifications[cell_true_classifications=="D"], predicted_classifications[cell_true_classifications=="D"])

    fig = plt.figure()
    fig.set_size_inches(5, 3, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    alphas = [0.333, 0.666, 1]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    ax.axhline(y=0, color="black", linewidth=3, linestyle="solid", alpha=0.1)
    for j, p_scalar in enumerate(np.unique(p_scalars)):
        for n, noise in enumerate(np.unique(field_noise_stds)):
            ax.plot(rolling_window_sizes, np.nanmean(Biases_in_coding[j,n], axis=1), label="s=" + str(noise) + ",ps=" + str(p_scalar), color="black",
                    clip_on=False, linestyle=linestyles[n], alpha=alphas[j])
    ax.set_ylabel('bias', fontsize=20, labelpad = 10)
    ax.set_xlabel('Rolling window size', fontsize=20, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([1, 1000])
    ax.set_ylim([-100, 100])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/bias_trial'+grid_stability+'_cell'+plot_suffix+'.png', dpi=300)
    plt.close()


    fig = plt.figure()
    fig.set_size_inches(5, 3, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    alphas = [0.333, 0.666, 1]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    ax.axhline(y=50, color="red", linewidth=3, linestyle="dashed")
    for j, p_scalar in enumerate(np.unique(p_scalars)):
        for n, noise in enumerate(np.unique(field_noise_stds)):
            ax.plot(rolling_window_sizes, np.nanmean(position_coding_accuracy[j,n], axis=1), label="s=" + str(noise) + ",ps=" + str(p_scalar), color=Settings.allocentric_color,
                    clip_on=False, linestyle=linestyles[n], alpha=alphas[j])
            ax.plot(rolling_window_sizes, np.nanmean(distance_coding_accuracy[j,n], axis=1), label="s=" + str(noise) + ",ps=" + str(p_scalar), color=Settings.egocentric_color,
                    clip_on=False, linestyle=linestyles[n], alpha=alphas[j])
            ax.plot(rolling_window_sizes, np.nanmean(overall_coding_accuracy[j, n], axis=1),label="s=" + str(noise) + ",ps=" + str(p_scalar), color="black",
                    clip_on=False, linestyle=linestyles[n], alpha=alphas[j])

    ax.set_ylabel('accuracy', fontsize=20, labelpad = 10)
    ax.set_xlabel('Rolling window size', fontsize=20, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([1, 1000])
    ax.set_ylim([0, 100])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/accuracy_trial'+grid_stability+'_cell'+plot_suffix+'.png', dpi=300)
    plt.close()
    return


def run_assay_alt_method(switch_coding_mode, grid_stability, save_path, grid_spacing_low, grid_spacing_high, n_cells, trial_switch_probability, freq_thres=0.05, plot_suffix=""):
    grid_spacings = np.random.uniform(low=grid_spacing_low, high=grid_spacing_high, size=n_cells);
    correlation_thresholds = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    field_noise_stds = [0, 10, 20, 30]
    p_scalars=[0.01, 0.1, 1]

    Biases_in_coding = np.zeros((len(p_scalars), len(field_noise_stds), len(correlation_thresholds), n_cells))
    overall_coding_accuracy = np.zeros((len(p_scalars), len(field_noise_stds), len(correlation_thresholds), n_cells))
    position_coding_accuracy = np.zeros((len(p_scalars), len(field_noise_stds), len(correlation_thresholds), n_cells))
    distance_coding_accuracy = np.zeros((len(p_scalars), len(field_noise_stds), len(correlation_thresholds), n_cells))

    for j, p_scalar in enumerate(np.unique(p_scalars)):
        for n, noise in enumerate(np.unique(field_noise_stds)):

            # first lets generate n cells
            correlations_all_cells, true_classifications = \
                switch_grid_cells_alt_method(switch_coding_mode, grid_stability, grid_spacings, n_cells, trial_switch_probability,
                                  field_noise_std=noise, p_scalar=p_scalar)

            for m in range(len(correlation_thresholds)):
                for i in range(n_cells):

                    correlations = correlations_all_cells[i]
                    anchored = correlations >= correlation_thresholds[m]
                    predicted_classifications = np.empty(len(correlations), dtype=np.str0)
                    predicted_classifications[anchored] = "P"
                    predicted_classifications[~anchored] = "D"

                    # compare the true classifications to the estimated classifications
                    cell_true_classifications = true_classifications[i]

                    total_number_of_trials = len(cell_true_classifications)
                    total_number_of_position_trials = len(cell_true_classifications[cell_true_classifications=="P"])
                    total_number_of_distance_trials = len(cell_true_classifications[cell_true_classifications=="D"])

                    total_number_of_predicted_position_trials = len(predicted_classifications[predicted_classifications=="P"])
                    total_number_of_predicted_distance_trials = len(predicted_classifications[predicted_classifications=="D"])

                    bias_percentage = (100 * (total_number_of_position_trials / total_number_of_trials)) - \
                                      (100 * (total_number_of_predicted_position_trials / total_number_of_trials))

                    Biases_in_coding[j, n, m, i] = bias_percentage
                    overall_coding_accuracy[j, n, m, i] = 100*accuracy_score(cell_true_classifications, predicted_classifications)
                    position_coding_accuracy[j, n, m, i] = 100*accuracy_score(cell_true_classifications[cell_true_classifications=="P"], predicted_classifications[cell_true_classifications=="P"])
                    distance_coding_accuracy[j, n, m, i] = 100*accuracy_score(cell_true_classifications[cell_true_classifications=="D"], predicted_classifications[cell_true_classifications=="D"])

    fig = plt.figure()
    fig.set_size_inches(5, 3, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    alphas = [0.333, 0.666, 1]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    ax.axhline(y=0, color="black", linewidth=3, linestyle="solid", alpha=0.1)
    for j, p_scalar in enumerate(np.unique(p_scalars)):
        for n, noise in enumerate(np.unique(field_noise_stds)):
            ax.plot(correlation_thresholds, np.nanmean(Biases_in_coding[j,n], axis=1), label="s=" + str(noise) + ",ps=" + str(p_scalar), color="black",
                    clip_on=False, linestyle=linestyles[n], alpha=alphas[j])
    ax.set_ylabel('bias', fontsize=20, labelpad = 10)
    ax.set_xlabel('correlation thresholds', fontsize=20, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1,1])
    ax.set_ylim([-100, 100])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/bias_trial'+grid_stability+'_cell'+plot_suffix+'.png', dpi=300)
    plt.close()


    fig = plt.figure()
    fig.set_size_inches(5, 3, forward=True)
    ax = fig.add_subplot(1, 1, 1)
    alphas = [0.333, 0.666, 1]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    ax.axhline(y=50, color="red", linewidth=3, linestyle="dashed")
    for j, p_scalar in enumerate(np.unique(p_scalars)):
        for n, noise in enumerate(np.unique(field_noise_stds)):
            ax.plot(correlation_thresholds, np.nanmean(position_coding_accuracy[j,n], axis=1), label="s=" + str(noise) + ",ps=" + str(p_scalar), color=Settings.allocentric_color,
                    clip_on=False, linestyle=linestyles[n], alpha=alphas[j])
            ax.plot(correlation_thresholds, np.nanmean(distance_coding_accuracy[j,n], axis=1), label="s=" + str(noise) + ",ps=" + str(p_scalar), color=Settings.egocentric_color,
                    clip_on=False, linestyle=linestyles[n], alpha=alphas[j])
            ax.plot(correlation_thresholds, np.nanmean(overall_coding_accuracy[j, n], axis=1),label="s=" + str(noise) + ",ps=" + str(p_scalar), color="black",
                    clip_on=False, linestyle=linestyles[n], alpha=alphas[j])

    ax.set_ylabel('accuracy', fontsize=20, labelpad = 10)
    ax.set_xlabel('correlation thresholds', fontsize=20, labelpad = 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-1,1])
    ax.set_ylim([0, 100])
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/accuracy_trial'+grid_stability+'_cell'+plot_suffix+'.png', dpi=300)
    plt.close()
    return

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    save_path = "/mnt/datastore/Harry/Grid_anchoring_eLife_2023/simulated"
    n_cells = 100
    grid_spacing_low = 40
    grid_spacing_high = 400

    # Figure 3, Figure Supplement 2
    for switch_coding in ["block", "by_trial"]:
        for freq_thres in [0.05]:
            np.random.seed(0)
            run_assay(switch_coding_mode=switch_coding, grid_stability="imperfect", save_path=save_path, grid_spacing_low=grid_spacing_low, grid_spacing_high=grid_spacing_high,
                      n_cells=n_cells, trial_switch_probability=0.1, freq_thres=freq_thres,
                      plot_suffix="_grid_spacings-"+str(grid_spacing_low)+"-"+str(grid_spacing_high)+"sigma_switch_coding="+switch_coding+"_freq_thres="+str(freq_thres))

            run_assay_alt_method(switch_coding_mode=switch_coding, grid_stability="imperfect", save_path=save_path, grid_spacing_low=grid_spacing_low, grid_spacing_high=grid_spacing_high,
                                 n_cells=n_cells, trial_switch_probability=0.1, freq_thres=freq_thres,
                                  plot_suffix="alt_method_grid_spacings-"+str(grid_spacing_low)+"-"+str(grid_spacing_high)+"sigma_switch_coding="+switch_coding+"_freq_thres="+str(freq_thres))

if __name__ == '__main__':
    main()
