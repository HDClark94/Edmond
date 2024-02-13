from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from Edmond.eLife_Grid_anchoring_2024.vr_grid_cells import *
from Edmond.eLife_Grid_anchoring_2024.Helpers.array_manipulations import *
import Edmond.eLife_Grid_anchoring_2024.Helpers.plot_utility as plot_utility

warnings.filterwarnings('ignore')

def get_hmt_color(hmt):
    if hmt == "hit":
        return "green"
    elif hmt == "miss":
        return "red"
    elif hmt == "try":
        return "orange"
    else:
        return "black"

def plot_lomb_classifiers_vs_shuffle(concantenated_dataframe, suffix="", save_path=""):
    print('plotting lomb classifiers...')

    distance_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Distance"]
    position_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Position"]
    null_cells = concantenated_dataframe[concantenated_dataframe["Lomb_classifier_"+suffix] == "Null"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,4), gridspec_kw={'width_ratios': [1, 0.3]})
    ax1.set_ylabel("Peak power vs \n false alarm rate",color="black",fontsize=25, labelpad=10)
    ax1.set_xlabel("Track frequency", color="black", fontsize=25, labelpad=10)
    ax1.set_xticks(np.arange(0, 11, 1.0))
    ax1.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
    ax2.set_xticks([0, 0.5])
    ax2.set_xticklabels(["0", "0.5"])
    ax2.set_yticks([])
    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax2.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    for f in range(1,6):
        ax1.axvline(x=f, color="gray", linewidth=2,linestyle="solid", alpha=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.scatter(x=null_cells["ML_Freqs"+suffix], y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax1.scatter(x=distance_cells["ML_Freqs"+suffix], y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax1.scatter(x=position_cells["ML_Freqs"+suffix], y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax1.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set_xlim([0,5.02])
    ax1.set_ylim([-0.1,0.4])
    ax2.set_xlim([-0.05,0.55])
    ax2.set_ylim([-0.1,0.4])
    ax2.set_xlabel(r'$\Delta$ from Int', color="black", fontsize=25, labelpad=10)
    ax2.scatter(x=distance_from_integer(null_cells["ML_Freqs"+suffix]), y=null_cells["ML_SNRs"+suffix]-null_cells["power_threshold"], color=Settings.null_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(distance_cells["ML_Freqs"+suffix]), y=distance_cells["ML_SNRs"+suffix]-distance_cells["power_threshold"], color=Settings.egocentric_color, marker="o", alpha=0.3)
    ax2.scatter(x=distance_from_integer(position_cells["ML_Freqs"+suffix]), y=position_cells["ML_SNRs"+suffix]-position_cells["power_threshold"], color=Settings.allocentric_color, marker="o", alpha=0.3)
    ax2.axhline(y=0, color="red", linewidth=3,linestyle="dashed")
    plt.tight_layout()
    plt.savefig(save_path + '/lomb_classifiers_vs_shuffle_PDN_'+suffix+'.png', dpi=200)
    plt.close()
    return

def plot_lomb_classifier_powers_vs_groups(concantenated_dataframe, suffix="", save_path="", fig_size=(6,6)):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]["ML_SNRs"])
    g_position_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Position"]["ML_SNRs"])
    g_null_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Null"]["ML_SNRs"])
    ng_distance_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Distance"]["ML_SNRs"])
    ng_position_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Position"]["ML_SNRs"])
    ng_null_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Null"]["ML_SNRs"])

    fig, ax = plt.subplots(figsize=fig_size)
    data = [g_position_cells[~np.isnan(g_position_cells)],
            g_distance_cells[~np.isnan(g_distance_cells)],
            g_null_cells[~np.isnan(g_null_cells)],
            ng_position_cells[~np.isnan(ng_position_cells)],
            ng_distance_cells[~np.isnan(ng_distance_cells)],
            ng_null_cells[~np.isnan(ng_null_cells)]]
    data = add_zero_to_data_if_empty(data)
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3,5,6,7], boxprops=boxprops, medianprops=medianprops,
               whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(bottom=0, top=0.3)
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Peak power", fontsize=20)
    plt.savefig(save_path + '/lomb_classifier_powers_vs_groups'+suffix+'.png', dpi=300)
    plt.close()
    return

def plot_lomb_classifier_spatinfo_vs_groups(concantenated_dataframe, suffix="", save_path="", fig_size=(6,6), score="spatial_information_score_Ispike_vr"):
    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Distance"][score])
    g_position_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Position"][score])
    g_null_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Null"][score])
    ng_distance_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Distance"][score])
    ng_position_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Position"][score])
    ng_null_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Null"][score])

    fig, ax = plt.subplots(figsize=fig_size)
    data = [g_position_cells[~np.isnan(g_position_cells)],
            g_distance_cells[~np.isnan(g_distance_cells)],
            g_null_cells[~np.isnan(g_null_cells)],
            ng_position_cells[~np.isnan(ng_position_cells)],
            ng_distance_cells[~np.isnan(ng_distance_cells)],
            ng_null_cells[~np.isnan(ng_null_cells)]]
    data = add_zero_to_data_if_empty(data)
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlim(left=0.5, right=3.5)
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Spatial info", fontsize=20)
    plt.savefig(save_path + '/lomb_classifier_spatinfo_vs_groups_'+score+''+suffix+'.png', dpi=300)
    plt.close()
    return

def plot_lomb_classifier_mfr_vs_groups(concantenated_dataframe, suffix="", save_path="", fig_size=(6,6)):

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]['mean_firing_rate_vr'])
    g_position_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Position"]['mean_firing_rate_vr'])
    g_null_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Null"]['mean_firing_rate_vr'])
    ng_distance_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Distance"]['mean_firing_rate_vr'])
    ng_position_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Position"]['mean_firing_rate_vr'])
    ng_null_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Null"]['mean_firing_rate_vr'])

    fig, ax = plt.subplots(figsize=fig_size)
    data = [g_position_cells[~np.isnan(g_position_cells)],
            g_distance_cells[~np.isnan(g_distance_cells)],
            g_null_cells[~np.isnan(g_null_cells)],
            ng_position_cells[~np.isnan(ng_position_cells)],
            ng_distance_cells[~np.isnan(ng_distance_cells)],
            ng_null_cells[~np.isnan(ng_null_cells)]]
    data = add_zero_to_data_if_empty(data)
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(bottom=0, top=15)
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_yticks([0, 5, 10, 15])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Mean firing rate", fontsize=20)
    plt.savefig(save_path + '/lomb_classifier_mfr_vs_groups'+suffix+'.png', dpi=300)
    plt.close()
    return


def plot_lomb_classifier_peak_width_vs_groups(concantenated_dataframe, suffix="", save_path="", fig_size=(6,6)):

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    g_distance_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Distance"]['ML_peak_width'])
    g_position_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Position"]['ML_peak_width'])
    g_null_cells = np.asarray(grid_cells[grid_cells["Lomb_classifier_"] == "Null"]['ML_peak_width'])
    ng_distance_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Distance"]['ML_peak_width'])
    ng_position_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Position"]['ML_peak_width'])
    ng_null_cells = np.asarray(non_grid_cells[non_grid_cells["Lomb_classifier_"] == "Null"]['ML_peak_width'])

    fig, ax = plt.subplots(figsize=fig_size)
    data = [g_position_cells[~np.isnan(g_position_cells)],
            g_distance_cells[~np.isnan(g_distance_cells)],
            g_null_cells[~np.isnan(g_null_cells)],
            ng_position_cells[~np.isnan(ng_position_cells)],
            ng_distance_cells[~np.isnan(ng_distance_cells)],
            ng_null_cells[~np.isnan(ng_null_cells)]]
    data = add_zero_to_data_if_empty(data)
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color, Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,3,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(bottom=0, top=3)
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.set_yticks([0, 1, 2, 3])
    fig.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.2)
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Peak width", fontsize=20)
    plt.savefig(save_path + '/lomb_classifier_peak_width_vs_groups'+suffix+'.png', dpi=300)
    plt.close()
    return

def plot_percentage_encoding_by_trial_category_each_mouse_weighted(combined_df, save_path="", trial_classification_column_name="", suffix=""):
    for tt in [0, 1, 2]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for code, code_color, code_z_order in zip(["P", "D"], [Settings.allocentric_color, get_mode_color_from_classification_column_name(trial_classification_column_name, "D")], [1,2]):
            for i, mouse_id in enumerate(np.unique(combined_df["mouse"])):
                mouse_cells = combined_df[combined_df["mouse"] == mouse_id]
                mouse_hit, hit_weights = get_percentage_from_rolling_classifier_column(mouse_cells, code=code, tt=tt, hmt="hit", column_name=trial_classification_column_name)
                mouse_try, try_weights = get_percentage_from_rolling_classifier_column(mouse_cells, code=code, tt=tt, hmt="try", column_name=trial_classification_column_name)
                mouse_miss, miss_weights = get_percentage_from_rolling_classifier_column(mouse_cells, code=code, tt=tt, hmt="miss", column_name=trial_classification_column_name)

                mask = ~np.isnan(mouse_hit) & ~np.isnan(mouse_try) & ~np.isnan(mouse_miss)
                if len(mouse_hit[mask])>0:
                    ax.plot([2, 6, 10], [np.average(mouse_hit[mask], weights=hit_weights[mask]),
                                         np.average(mouse_try[mask], weights=try_weights[mask]),
                                         np.average(mouse_miss[mask], weights=miss_weights[mask])], clip_on=False, color=code_color, linewidth=1.5, alpha=0.3, zorder=-1)

            All_hit, hit_weights = get_percentage_from_rolling_classifier_column(combined_df, code=code, tt=tt, hmt="hit", column_name=trial_classification_column_name)
            All_try, try_weights = get_percentage_from_rolling_classifier_column(combined_df, code=code, tt=tt, hmt="try", column_name=trial_classification_column_name)
            All_miss, miss_weights = get_percentage_from_rolling_classifier_column(combined_df, code=code, tt=tt, hmt="miss", column_name=trial_classification_column_name)
            mask = ~np.isnan(All_hit) & ~np.isnan(All_try) & ~np.isnan(All_miss)

            ax.errorbar(x=[2, 6, 10], y=[np.average(All_hit[mask], weights=hit_weights[mask]),
                                         np.average(All_try[mask], weights=try_weights[mask]),
                                         np.average(All_miss[mask], weights=miss_weights[mask])],
                        yerr=[stats.sem(All_hit[mask], axis=0, nan_policy="omit"),
                              stats.sem(All_try[mask], axis=0, nan_policy="omit"),
                              stats.sem(All_miss[mask], axis=0, nan_policy="omit")], capsize=25, linewidth=3, color=code_color, zorder=code_z_order)

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.set_xticks([2,6,10])
        ax.set_yticks([0,25,50,75,100])
        ax.xaxis.set_tick_params(labelbottom=False)
        #ax.set_xticks([2,6,10])
        #ax.set_xticklabels(["Hit", "Try", "Run"])
        ax.set_xlim(left=0, right=12)
        ax.set_ylim(bottom=0, top=100)
        plt.savefig(save_path + '/rolling_percent_encooding_for_'+str(tt)+'_for_each_mouse_'+suffix+'_weigthed.png', dpi=300)
        plt.close()
    return

def get_percentage_from_rolling_classifier_column(df, code, tt, hmt, column_name):
    percentages = []
    weights = []
    for index, cell in df.iterrows():
        cell = cell.to_frame().T.reset_index(drop=True)
        session_id = cell["session_id_vr"].iloc[0]
        behaviours = np.array(cell["behaviour_hit_try_miss"].iloc[0])
        trial_types = np.array(cell["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cell[column_name].iloc[0])

        # mask by trial type
        tt_mask = trial_types == tt
        trial_types = trial_types[tt_mask]
        behaviours = behaviours[tt_mask]
        rolling_classifiers = rolling_classifiers[tt_mask]

        # mask by behaviour
        beh_mask = behaviours == hmt
        trial_types = trial_types[beh_mask]
        hmt_behaviours = behaviours[beh_mask]
        rolling_classifiers = rolling_classifiers[beh_mask]

        # mask by code
        code_mask = (rolling_classifiers != "") &\
                    (rolling_classifiers != "nan") &\
                    (rolling_classifiers != "N")
        rolling_classifiers = rolling_classifiers[code_mask]

        code_mask = rolling_classifiers == code
        rolling_classifiers_masked = rolling_classifiers[code_mask]

        if len(rolling_classifiers) == 0:
            percentages.append(np.nan)
            weights.append(0)
        else:
            percentage_encoding = (len(rolling_classifiers_masked)/len(rolling_classifiers))*100
            percentages.append(percentage_encoding)

            hmt_weight = len(hmt_behaviours)/len(behaviours)
            session_weight = 1/len(df[df["session_id_vr"]==session_id])

            weights.append(hmt_weight*session_weight)

    return np.array(percentages), np.array(weights)


def avg_over_cells_and_then_session(array, session_ids, mouse_ids, return_nans=True):
    session_ids = np.asarray(session_ids)
    mouse_ids = np.asarray(mouse_ids)

    avg_values = []
    for mouse in np.unique(mouse_ids):

        session_avg_values = []
        for session_id in np.unique(session_ids[mouse_ids == mouse]):
            values_for_session = array[(mouse_ids == mouse) & (session_ids == session_id)]
            session_avg_values.append(np.nanmean(values_for_session, axis=0))
        session_avg_values = np.array(session_avg_values)

        avg_values.append(np.nanmean(session_avg_values, axis=0))

    avg_values = np.array(avg_values)
    if return_nans:
        return avg_values
    else:
        return avg_values[~np.isnan(avg_values)]

def plot_percentage_hits_for_remapped_encoding_cells_averaged_over_cells(cells, save_path="", trial_classification_column_name="", suffix=""):
    mouse_colors = ['darkturquoise', 'salmon', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    p_b_hit = get_percentage_from_rolling_classification(cells, code="P", tt=0, trial_classification_column_name=trial_classification_column_name)
    d_b_hit = get_percentage_from_rolling_classification(cells, code="D", tt=0, trial_classification_column_name=trial_classification_column_name)
    p_nb_hit = get_percentage_from_rolling_classification(cells, code="P", tt=1, trial_classification_column_name=trial_classification_column_name)
    d_nb_hit = get_percentage_from_rolling_classification(cells, code="D", tt=1, trial_classification_column_name=trial_classification_column_name)
    p_p_hit = get_percentage_from_rolling_classification(cells, code="P", tt=2, trial_classification_column_name=trial_classification_column_name)
    d_p_hit = get_percentage_from_rolling_classification(cells, code="D", tt=2, trial_classification_column_name=trial_classification_column_name)

    b_mask = ~np.isnan(p_b_hit) & ~np.isnan(d_b_hit)
    nb_mask = ~np.isnan(p_nb_hit) & ~np.isnan(d_nb_hit)
    p_mask = ~np.isnan(p_p_hit) & ~np.isnan(d_p_hit)

    data = [p_b_hit[~np.isnan(p_b_hit)], d_b_hit[~np.isnan(d_b_hit)], p_nb_hit[~np.isnan(p_nb_hit)],
            d_nb_hit[~np.isnan(d_nb_hit)], p_p_hit[~np.isnan(p_p_hit)], d_p_hit[~np.isnan(d_p_hit)]]

    colors = [Settings.allocentric_color, get_mode_color_from_classification_column_name(trial_classification_column_name, "D"),
              Settings.allocentric_color, get_mode_color_from_classification_column_name(trial_classification_column_name, "D"),
              Settings.allocentric_color, get_mode_color_from_classification_column_name(trial_classification_column_name, "D")]
    fig, ax = plt.subplots(figsize=(8, 6))
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2,4,5,7,8], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    p_b_hit = avg_over_cells_and_then_session(p_b_hit, cells["session_id"], cells["mouse"])
    d_b_hit = avg_over_cells_and_then_session(d_b_hit, cells["session_id"], cells["mouse"])
    p_nb_hit = avg_over_cells_and_then_session(p_nb_hit, cells["session_id"], cells["mouse"])
    d_nb_hit = avg_over_cells_and_then_session(d_nb_hit, cells["session_id"], cells["mouse"])
    p_p_hit = avg_over_cells_and_then_session(p_p_hit, cells["session_id"], cells["mouse"])
    d_p_hit = avg_over_cells_and_then_session(d_p_hit, cells["session_id"], cells["mouse"])
    data = [p_b_hit, d_b_hit, p_nb_hit, d_nb_hit, p_p_hit, d_p_hit]
    r_i = np.array([0,0])
    for i in range(len(data[0])):
        ax.plot([1 + r_i[0],2 + r_i[1]], [data[0][i], data[1][i]], color=mouse_colors[i], alpha=1, linewidth=2)
    for i in range(len(data[2])):
        ax.plot([4 + r_i[0], 5 + r_i[1]], [data[2][i], data[3][i]], color=mouse_colors[i], alpha=1, linewidth=2)
    for i in range(len(data[4])):
        ax.plot([7 + r_i[0], 8 + r_i[1]], [data[4][i], data[5][i]], color=mouse_colors[i], alpha=1, linewidth=2)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=25)
    ax.set_yticks([0,25, 50, 75, 100])
    ax.set_xticks([1,2,4,5,7,8])
    ax.set_xticklabels(["B", "B", "NB", "NB", "P", "P"])
    ax.set_xlim(left=0, right=9)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.savefig(save_path + '/percentage_hit_trials_in_coded_trials_' + suffix + '.png', dpi=300)
    plt.close()
    return

def plot_lomb_classifiers_proportions(concantenated_dataframe, suffix="", save_path=""):
    print('plotting lomb classifers proportions...')

    grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] == "G"]
    non_grid_cells = concantenated_dataframe[concantenated_dataframe["classifier"] != "G"]

    fig, ax = plt.subplots(figsize=(4,6))
    groups = ["Position", "Distance", "Null"]
    colors_lm = [Settings.allocentric_color,  Settings.egocentric_color, Settings.null_color, "black"]
    objects = ["G", "NG"]
    x_pos = np.arange(len(objects))
    for object, x in zip(objects, x_pos):
        if object == "G":
            df = grid_cells
        elif object == "NG":
            df = non_grid_cells
        bottom=0
        for color, group in zip(colors_lm, groups):
            count = len(df[(df["Lomb_classifier_"+suffix] == group)])
            percent = (count/len(df))*100
            ax.bar(x, percent, bottom=bottom, color=color, edgecolor=color)
            ax.text(x,bottom+0.5, str(count), color="k", fontsize=15, ha="center")
            bottom = bottom+percent
    plt.xticks(x_pos, objects, fontsize=15)
    plt.ylabel("Percent of neurons",  fontsize=25)
    plt.xlim((-0.5, len(objects)-0.5))
    plt.ylim((0,100))
    ax.set_yticks([0,25,50,75,100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.4)
    ax.tick_params(axis='both', which='major', labelsize=30)
    plt.savefig(save_path + '/lomb_classifiers_proportions_'+suffix+'.png', dpi=200)
    plt.close()
    return


def plot_regression(ax, x, y, c, y_text_pos, x_test_min=None, x_test_max=None):
    # x  and y are pandas collumn
    try:
        x = x.values
        y = y.values
    except Exception as ex:
        print("")

    x = x[~np.isnan(y)].reshape(-1, 1)
    y = y[~np.isnan(y)].reshape(-1, 1)

    pearson_r = stats.pearsonr(x.flatten(),y.flatten())

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x,y)  # perform linear regression

    if x_test_min is None:
        x_test_min = min(x)
    if x_test_max is None:
        x_test_max = max(x)

    x_test = np.linspace(x_test_min, x_test_max, 100)

    Y_pred = linear_regressor.predict(x_test.reshape(-1, 1))  # make predictions
    ax.text(  # position text relative to Axes
        0.05, y_text_pos, "R= "+str(np.round(pearson_r[0], decimals=2))+ ", p = "+str(np.round(pearson_r[1], decimals=4)),
        ha='left', va='top', color=c,
        transform=ax.transAxes, fontsize=10)
    ax.plot(x_test, Y_pred, color=c)


def get_percentage_hit_column(df, tt, return_nans=False):
    percentage_hits = []
    for index, cluster_df in df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        behaviours = np.array(cluster_df["behaviour_hit_try_miss"].iloc[0])
        trial_types = np.array(cluster_df["behaviour_trial_types"].iloc[0])
        valid_behaviours = behaviours[(trial_types == tt) & (behaviours != "rejected")]
        if len(valid_behaviours)>0:
            percentage = 100*(len(valid_behaviours[valid_behaviours == "hit"])/len(valid_behaviours))
        else:
            percentage = np.nan
        percentage_hits.append(percentage)

    percentage_hits = np.array(percentage_hits)
    if return_nans:
        return percentage_hits
    else:
        return percentage_hits[~np.isnan(percentage_hits)]


def get_percentage_from_rolling_classification(df, code, tt, trial_classification_column_name="", hmt=""):
    percentages = []
    for index, cluster_df in df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        behaviours = np.array(cluster_df["behaviour_hit_try_miss"].iloc[0])
        trial_types = np.array(cluster_df["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cluster_df[trial_classification_column_name].iloc[0])

        valid_trials = behaviours[(rolling_classifiers == code) &
                                  (trial_types == tt) &
                                  (rolling_classifiers != "nan") &
                                  (behaviours != "rejected")]

        if len(valid_trials) == 0:
            percentage = np.nan
        else:
            percentage = 100*(len(valid_trials[valid_trials==hmt])/len(valid_trials))
        percentages.append(percentage)

    return np.array(percentages)


def get_mode_color_from_classification_column_name(column_name, mode):
    if mode == "P" and column_name == "alternative_classifications":
        return Settings.allocentric_color
    elif mode == "P" and column_name == "rolling:classifier_by_trial_number":
        return Settings.allocentric_color
    elif mode == "D" and column_name == "alternative_classifications":
        return Settings.alt_egocentric_color
    elif mode == "D" and column_name == "rolling:classifier_by_trial_number":
        return Settings.egocentric_color
    else:
        print("There is an error extracting a color based on the classification column name")


def plot_stop_histogram_for_remapped_encoding_cells_averaged_over_cells(cells, save_path, suffix="", trial_classification_column_name="", lock_y_axis=True):
    for tt in [0,1,2]:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.axhline(y=0, linestyle="dashed", linewidth=2, color="black")
        remapped_position_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt, coding_scheme="P", shuffle=False, trial_classification_column_name=trial_classification_column_name)
        remapped_distance_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt, coding_scheme="D", shuffle=False, trial_classification_column_name=trial_classification_column_name)
        remapped_position_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt,coding_scheme="P",shuffle=True, trial_classification_column_name=trial_classification_column_name)
        remapped_distance_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(cells, tt=tt,coding_scheme="D",shuffle=True, trial_classification_column_name=trial_classification_column_name)

        # get session weights
        P_weigths = get_session_weights(cells, tt = tt, coding_scheme = "P")
        D_weigths = get_session_weights(cells, tt = tt, coding_scheme = "D")

        remapped_position_grid_cells_stop_histogram_tt = np.array(remapped_position_grid_cells_stop_histogram_tt)
        remapped_distance_grid_cells_stop_histogram_tt = np.array(remapped_distance_grid_cells_stop_histogram_tt)
        remapped_position_grid_cells_shuffled_histogram_tt = np.array(remapped_position_grid_cells_shuffled_histogram_tt)
        remapped_distance_grid_cells_shuffled_histogram_tt = np.array(remapped_distance_grid_cells_shuffled_histogram_tt)

        p_nan_mask = ~np.isnan(remapped_position_grid_cells_stop_histogram_tt)[:,0]
        d_nan_mask = ~np.isnan(remapped_distance_grid_cells_stop_histogram_tt)[:,0]

        # normalise to baseline
        remapped_position_grid_cells_stop_histogram_tt = remapped_position_grid_cells_stop_histogram_tt-remapped_position_grid_cells_shuffled_histogram_tt
        remapped_distance_grid_cells_stop_histogram_tt = remapped_distance_grid_cells_stop_histogram_tt-remapped_distance_grid_cells_shuffled_histogram_tt

        # apply mask
        remapped_position_grid_cells_stop_histogram_tt = remapped_position_grid_cells_stop_histogram_tt[p_nan_mask,:]
        remapped_distance_grid_cells_stop_histogram_tt = remapped_distance_grid_cells_stop_histogram_tt[d_nan_mask,:]
        P_weigths = P_weigths[p_nan_mask]
        D_weigths = D_weigths[d_nan_mask]


        # plot position grid cell session stop histogram
        ax.plot(bin_centres, np.average(remapped_position_grid_cells_stop_histogram_tt, weights=P_weigths, axis=0), color= Settings.allocentric_color, linewidth=3)
        ax.fill_between(bin_centres, np.average(remapped_position_grid_cells_stop_histogram_tt, weights=P_weigths, axis=0)-stats.sem(remapped_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.average(remapped_position_grid_cells_stop_histogram_tt, weights=P_weigths, axis=0)+stats.sem(remapped_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.allocentric_color, alpha=0.3)

        # plot distance grid cell session stop histogram
        ax.plot(bin_centres, np.average(remapped_distance_grid_cells_stop_histogram_tt, weights=D_weigths, axis=0), color= get_mode_color_from_classification_column_name(trial_classification_column_name, "D"), linewidth=3)
        ax.fill_between(bin_centres, np.average(remapped_distance_grid_cells_stop_histogram_tt, weights=D_weigths, axis=0)-stats.sem(remapped_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.average(remapped_distance_grid_cells_stop_histogram_tt, weights=D_weigths, axis=0)+stats.sem(remapped_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=get_mode_color_from_classification_column_name(trial_classification_column_name, "D"), alpha=0.3)

        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        #plt.ylabel('Stops (/cm)', fontsize=20, labelpad = 20)
        #plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
        plt.xlim(0, 200)
        tick_spacing = 100
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plot_utility.style_vr_plot(ax)
        if lock_y_axis:
            ax.set_ylim([-0.05,0.15])
            ax.set_yticks([0, 0.1])

        plt.locator_params(axis = 'y', nbins  = 3)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        plt.savefig(save_path + '/stop_histogram_for_remapped_cells_encoding_position_and_distance_'+str(tt)+'_'+suffix+'.png', dpi=300)
        plt.close()
    return


def plot_stop_histogram_for_stable_cells_averaged_over_cells(cells, save_path, suffix=""):
    position_cells = cells[cells["Lomb_classifier_"] == "Position"]
    distance_cells = cells[cells["Lomb_classifier_"] == "Distance"]

    stable_position_cells = position_cells[position_cells["rolling:proportion_encoding_position"] >= 0.85]
    stable_distance_cells = distance_cells[distance_cells["rolling:proportion_encoding_distance"] >= 0.85]

    stable_position_cells = drop_duplicate_sessions(stable_position_cells)
    stable_distance_cells = drop_duplicate_sessions(stable_distance_cells)

    for tt in [0,1,2]:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.axhline(y=0, linestyle="dashed", linewidth=2, color="black")
        stable_position_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_cells, tt=tt, coding_scheme=None, shuffle=False)
        stable_distance_grid_cells_stop_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_distance_cells, tt=tt, coding_scheme=None, shuffle=False)
        stable_position_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_position_cells, tt=tt, coding_scheme=None, shuffle=True)
        stable_distance_grid_cells_shuffled_histogram_tt, _, bin_centres, _, _, _ = get_stop_histogram(stable_distance_cells, tt=tt, coding_scheme=None, shuffle=True)
        stable_position_grid_cells_stop_histogram_tt = np.array(stable_position_grid_cells_stop_histogram_tt)
        stable_distance_grid_cells_stop_histogram_tt = np.array(stable_distance_grid_cells_stop_histogram_tt)
        stable_position_grid_cells_shuffled_histogram_tt = np.array(stable_position_grid_cells_shuffled_histogram_tt)
        stable_distance_grid_cells_shuffled_histogram_tt = np.array(stable_distance_grid_cells_shuffled_histogram_tt)

        # apply normalisation with baseline
        stable_position_grid_cells_stop_histogram_tt = stable_position_grid_cells_stop_histogram_tt-stable_position_grid_cells_shuffled_histogram_tt
        stable_distance_grid_cells_stop_histogram_tt = stable_distance_grid_cells_stop_histogram_tt-stable_distance_grid_cells_shuffled_histogram_tt

        # plot position grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(stable_position_grid_cells_stop_histogram_tt, axis=0), color= Settings.allocentric_color,linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(stable_position_grid_cells_stop_histogram_tt, axis=0)-stats.sem(stable_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(stable_position_grid_cells_stop_histogram_tt, axis=0)+stats.sem(stable_position_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.allocentric_color, alpha=0.3)

        # plot distance grid cell session stop histogram
        ax.plot(bin_centres, np.nanmean(stable_distance_grid_cells_stop_histogram_tt, axis=0), color= Settings.egocentric_color,linewidth=3)
        ax.fill_between(bin_centres, np.nanmean(stable_distance_grid_cells_stop_histogram_tt, axis=0)-stats.sem(stable_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"),
                        np.nanmean(stable_distance_grid_cells_stop_histogram_tt, axis=0)+stats.sem(stable_distance_grid_cells_stop_histogram_tt, axis=0, nan_policy="omit"), color=Settings.egocentric_color, alpha=0.3)

        if tt == 0:
            style_track_plot(ax, 200)
        else:
            style_track_plot_no_RZ(ax, 200)
        #plt.ylabel('Stops (/cm)', fontsize=20, labelpad = 20)
        #plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
        plt.xlim(0, 200)
        tick_spacing = 100
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plot_utility.style_vr_plot(ax)
        ax.set_ylim([-0.05,0.15])
        ax.set_yticks([0, 0.1])
        #ax.set_ylim([-0.05,0.05])
        #ax.set_yticks([0, 0.05])

        plt.locator_params(axis = 'y', nbins  = 3)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()
        plt.subplots_adjust(bottom = 0.2, left=0.2)
        plt.savefig(save_path + '/stop_histogram_for_stable_cells_encoding_position_and_distance_'+str(tt)+'_'+suffix+'_averaged_over_cells.png', dpi=300)
        plt.close()
    return


def drop_duplicate_sessions(cells_df):
    sessions = []
    new_df = pd.DataFrame()
    for index, cluster_df in cells_df.iterrows():
        cluster_df = cluster_df.to_frame().T.reset_index(drop=True)
        session_id = cluster_df["session_id"].iloc[0]
        if session_id not in sessions:
            new_df = pd.concat([new_df, cluster_df], ignore_index=True)
            sessions.append(session_id)
    return new_df


def get_mouse_colors(cells, mouse_colors, all_mouse_ids):
    mouse_ids_from_data_frame = np.asarray(cells["mouse"])
    mouse_colors_for_dataframe = []
    for i in range(len(mouse_ids_from_data_frame)):
        mouse_colors_for_dataframe.append(mouse_colors[np.argwhere(all_mouse_ids == mouse_ids_from_data_frame[i])[0][0]])
    return np.array(mouse_colors_for_dataframe)


def plot_percentage_hits_for_stable_encoding_cells_averaged_over_cells(cells, save_path, suffix=""):
    mouse_colors = ['darkturquoise', 'salmon', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',u'#bcbd22', u'#17becf']
    mouse_ids = np.unique(cells["mouse"])[:len(mouse_colors)]

    position_cells = cells[cells["Lomb_classifier_"] == "Position"]
    distance_cells = cells[cells["Lomb_classifier_"] == "Distance"]

    stable_position_cells = position_cells[position_cells["rolling:proportion_encoding_position"] >= 0.85]
    stable_distance_cells = distance_cells[distance_cells["rolling:proportion_encoding_distance"] >= 0.85]

    stable_position_cells = drop_duplicate_sessions(stable_position_cells)
    stable_distance_cells = drop_duplicate_sessions(stable_distance_cells)
    stable_position_colors = get_mouse_colors(stable_position_cells, mouse_colors, mouse_ids)
    stable_distance_colors = get_mouse_colors(stable_distance_cells, mouse_colors, mouse_ids)

    print("n session for stable position grid cells, n = ", str(len(stable_position_cells)))
    print("n session for stable distance grid cells, n = ", str(len(stable_distance_cells)))

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    beaconed_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_cells, tt=0, return_nans=True)
    non_beaconed_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_cells, tt=1, return_nans=True)
    beaconed_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_cells, tt=0, return_nans=True)
    non_beaconed_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_cells, tt=1, return_nans=True)
    probe_percentage_hits_stable_position_grid_cells = get_percentage_hit_column(stable_position_cells, tt=2, return_nans=True)
    probe_percentage_hits_stable_distance_grid_cells = get_percentage_hit_column(stable_distance_cells, tt=2, return_nans=True)

    beaconed_percentage_hits_stable_position_colors = stable_position_colors[~np.isnan(beaconed_percentage_hits_stable_position_grid_cells)]
    non_beaconed_percentage_hits_stable_position_colors = stable_position_colors[~np.isnan(non_beaconed_percentage_hits_stable_position_grid_cells)]
    beaconed_percentage_hits_stable_distance_colors = stable_distance_colors[~np.isnan(beaconed_percentage_hits_stable_distance_grid_cells)]
    non_beaconed_percentage_hits_stable_distance_colors = stable_distance_colors[~np.isnan(non_beaconed_percentage_hits_stable_distance_grid_cells)]
    probe_percentage_hits_stable_position_colors = stable_position_colors[~np.isnan(probe_percentage_hits_stable_position_grid_cells)]
    probe_percentage_hits_stable_distance_colors = stable_distance_colors[~np.isnan(probe_percentage_hits_stable_distance_grid_cells)]

    colors = [Settings.allocentric_color, Settings.egocentric_color, Settings.allocentric_color,
              Settings.egocentric_color, Settings.allocentric_color, Settings.egocentric_color]

    data = [beaconed_percentage_hits_stable_position_grid_cells[~np.isnan(beaconed_percentage_hits_stable_position_grid_cells)],
            beaconed_percentage_hits_stable_distance_grid_cells[~np.isnan(beaconed_percentage_hits_stable_distance_grid_cells)],
            non_beaconed_percentage_hits_stable_position_grid_cells[~np.isnan(non_beaconed_percentage_hits_stable_position_grid_cells)],
            non_beaconed_percentage_hits_stable_distance_grid_cells[~np.isnan(non_beaconed_percentage_hits_stable_distance_grid_cells)],
            probe_percentage_hits_stable_position_grid_cells[~np.isnan(probe_percentage_hits_stable_position_grid_cells)],
            probe_percentage_hits_stable_distance_grid_cells[~np.isnan(probe_percentage_hits_stable_distance_grid_cells)]]

    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1, 2, 4, 5, 7, 8], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=-1)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    data_colors = [beaconed_percentage_hits_stable_position_colors, beaconed_percentage_hits_stable_distance_colors,
                   non_beaconed_percentage_hits_stable_position_colors, non_beaconed_percentage_hits_stable_distance_colors,
                   probe_percentage_hits_stable_position_colors, probe_percentage_hits_stable_distance_colors]
    for i, x in enumerate([1, 2, 4, 5, 7, 8]):
        ax.scatter((np.ones(len(data[i])) * x) + np.random.uniform(low=-0.1, high=+0.1, size=len(data[i])), data[i],
                   color=data_colors[i], alpha=1, zorder=1)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1, 2, 4, 5, 7, 8])
    ax.set_xticklabels(["B", "B", "NB", "NB", "P", "P"])
    ax.set_yticks([0, 25, 50, 75, 100])
    # ax.set_ylim([0, 100])
    ax.set_xlim([0, 9])
    # ax.set_xlabel("Encoding grid cells", fontsize=20)
    # ax.set_ylabel("Percentage hits", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=25)
    # plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.3, right = 0.87, top = 0.92)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.savefig(save_path + '/percentage_hits_for_stable_cells_'+suffix+'.png', dpi=300)
    plt.close()
    return


def plot_rolling_lomb_block_sizes(combined_df, save_path, suffix=""):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_xticks([0,1])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    _, _, patches0 = ax.hist(pandas_collumn_to_numpy_array(grid_cells["rolling:proportion_encoding_position"]), density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color=(255 / 255, 127 / 255, 14 / 255),linewidth=2)
    _, _, patches1 = ax.hist(pandas_collumn_to_numpy_array(non_grid_cells["rolling:proportion_encoding_position"]), density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color=(0 / 255, 154 / 255, 255 / 255),linewidth=2)
    patches0[0].set_xy(patches0[0].get_xy()[:-1])
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    ks, p = stats.ks_2samp(pandas_collumn_to_numpy_array(grid_cells["rolling:proportion_encoding_position"]),
                                 pandas_collumn_to_numpy_array(non_grid_cells["rolling:proportion_encoding_position"]))
    print("position, ks=", str(ks), " , p=", str(p), " , Ng= ", str(len(grid_cells)), ", Nng= ", str(len(non_grid_cells)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/block_length_encoding_position_cumsum'+suffix+'.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_xticks([0,1])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    _, _, patches0 = ax.hist(pandas_collumn_to_numpy_array(grid_cells["rolling:proportion_encoding_distance"]), density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color=(255 / 255, 127 / 255, 14 / 255),linewidth=2)
    _, _, patches1 = ax.hist(pandas_collumn_to_numpy_array(non_grid_cells["rolling:proportion_encoding_distance"]), density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color=(0 / 255, 154 / 255, 255 / 255),linewidth=2)
    patches0[0].set_xy(patches0[0].get_xy()[:-1])
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    ks, p = stats.ks_2samp(pandas_collumn_to_numpy_array(grid_cells["rolling:proportion_encoding_distance"]),
                                 pandas_collumn_to_numpy_array(non_grid_cells["rolling:proportion_encoding_distance"]))
    print("distance, ks=", str(ks), " , p=", str(p), " , Ng= ", str(len(grid_cells)), ", Nng= ", str(len(non_grid_cells)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/block_length_encoding_distance_cumsum'+suffix+'.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_xticks([0,1])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    _, _, patches0 = ax.hist(pandas_collumn_to_numpy_array(grid_cells["rolling:proportion_encoding_null"]), density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color=(255 / 255, 127 / 255, 14 / 255),linewidth=2)
    _, _, patches1 = ax.hist(pandas_collumn_to_numpy_array(non_grid_cells["rolling:proportion_encoding_null"]), density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color=(0 / 255, 154 / 255, 255 / 255),linewidth=2)
    patches0[0].set_xy(patches0[0].get_xy()[:-1])
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    ks, p = stats.ks_2samp(pandas_collumn_to_numpy_array(grid_cells["rolling:proportion_encoding_null"]),
                                 pandas_collumn_to_numpy_array(non_grid_cells["rolling:proportion_encoding_null"]))
    print("null, ks=", str(ks), " , p=", str(p), " , Ng= ", str(len(grid_cells)), ", Nng= ", str(len(non_grid_cells)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=30)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(save_path + '/block_length_encoding_null_cumsum'+suffix+'.png', dpi=300)
    plt.close()
    return

def plot_proportion_of_session_encoding_mode(combined_df, save_path, suffix=""):

    Position_cells = combined_df[combined_df["Lomb_classifier_"] == "Position"]
    Distance_cells = combined_df[combined_df["Lomb_classifier_"] == "Distance"]
    Null_cells = combined_df[combined_df["Lomb_classifier_"] == "Null"]

    for encoding_name in ["position", "distance", "null"]:
        encoding_column_name = "rolling:proportion_encoding_"+encoding_name

        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.set_xticks([0,1])
        ax.set_ylim([0, 20]) # change the y limit if out of bounds
        if suffix=="_nongrid_cells":
            ax.set_ylim([0, 600])
        ax.set_xlim([0, 1])
        colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.null_color]
        ax.hist([pandas_collumn_to_numpy_array(Position_cells[encoding_column_name]),
                 pandas_collumn_to_numpy_array(Distance_cells[encoding_column_name]),
                 pandas_collumn_to_numpy_array(Null_cells[encoding_column_name])], bins=20, range=(0,1), color=colors, stacked=True)
        #ax.set_xlabel("frac. session", fontsize=20)
        #ax.set_ylabel("Number of cells", fontsize=20, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(length=0)
        ax.tick_params(axis='both', which='both', labelsize=30)
        plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
        plt.savefig(save_path + '/block_length_encoding_'+encoding_name+suffix+'.png', dpi=300)
        plt.close()
    return


def plot_rolling_lomb_block_lengths_vs_shuffled(combined_df, save_path):
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"]
    grid_cells = combined_df[combined_df["classifier"] == "G"]

    fig, ax = plt.subplots(1,1, figsize=(4,4))
    block_lengths = pandas_collumn_to_numpy_array(grid_cells["rolling:block_lengths"])
    block_lengths_shuffled = pandas_collumn_to_numpy_array(grid_cells["rolling:block_lengths_shuffled"])
    _, _, patches0 = ax.hist(block_lengths, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="red",linewidth=2)
    _, _, patches1 = ax.hist(block_lengths_shuffled, density=True, bins=1000, cumulative=True, range=(0,1), histtype="step", color="grey",linewidth=2)
    patches0[0].set_xy(patches0[0].get_xy()[:-1])
    patches1[0].set_xy(patches1[0].get_xy()[:-1])
    #ax.set_xlabel("Frac. session", fontsize=20)
    #ax.set_ylabel("Density", fontsize=20, labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=20)
    plt.savefig(save_path + '/block_lengths_vs_shuffled_trials.png', dpi=300)
    plt.close()
    ks, p = stats.ks_2samp(block_lengths, block_lengths_shuffled)
    return


def add_peak_width(combined_df):
    peak_widths = []
    for index, row in combined_df.iterrows():
        avg_powers = row["MOVING_LOMB_avg_power"]
        if np.sum(np.isnan(avg_powers))==0:

            width, _, _, _ = signal.peak_widths(avg_powers, np.array([np.nanargmax(avg_powers)]))
            width = width[0]*Settings.frequency_step

            peak_index = np.nanargmax(avg_powers)
            trough_indices = get_peak_indices(avg_powers, [peak_index])

            # in cases where only 1 trough can be located, a trough indices will be at 0 or -1 (i.e. len(avg_powers))
            if trough_indices[0][0] == 0:
                # ignore this index and use only valid trough index and multiple this peak-to-trough x2
                width_according_to_peak_indices = (trough_indices[0][1] - peak_index)           * Settings.frequency_step * 2
            elif trough_indices[0][1] == len(avg_powers):
                # ignore this index and use only valid trough index and multiple this peak-to-trough x2
                width_according_to_peak_indices = (peak_index           - trough_indices[0][0]) * Settings.frequency_step * 2
            else:
                width_according_to_peak_indices = (trough_indices[0][1] - trough_indices[0][0]) * Settings.frequency_step

            width = width_according_to_peak_indices
            if width == 0:
                print("stop here")
        else:
            width = np.nan
        peak_widths.append(width)
    combined_df["ML_peak_width"] = peak_widths
    return combined_df


def plot_mean_firing_rates_on_position_and_distance_trials(combined_df, save_path, fig_size, suffix="", unlock_y=False):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    position_scores = grid_cells["rolling:position_mean_firing_rate"]
    distance_scores = grid_cells["rolling:distance_mean_firing_rate"]
    nan_mask = ~np.isnan(position_scores) & ~np.isnan(distance_scores)
    position_scores = position_scores[nan_mask]
    distance_scores = distance_scores[nan_mask]

    fig, ax = plt.subplots(figsize=fig_size)
    data = [position_scores, distance_scores, position_scores, distance_scores, position_scores, distance_scores, position_scores]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2, 4,4,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if unlock_y:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(bottom=0, top=15)
        ax.set_yticks([0,5,10,15])
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_xticks([1,2])
    plt.xticks(rotation = 30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_ylabel("Mean firing rate", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=25)
    fig.tight_layout()
    plt.savefig(save_path + '/' +'position_vs_distance_trials_mfr_'+suffix+'.png', dpi=300)
    plt.close()

def plot_spatial_information_on_position_and_distance_trials(combined_df, save_path, fig_size, suffix="", unlock_y=False):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    position_scores = grid_cells["rolling:position_spatial_information_scores_Isec"]
    distance_scores = grid_cells["rolling:distance_spatial_information_scores_Isec"]
    nan_mask = ~np.isnan(position_scores) & ~np.isnan(distance_scores)
    position_scores = position_scores[nan_mask]
    distance_scores = distance_scores[nan_mask]

    fig, ax = plt.subplots(figsize=fig_size)
    data = [position_scores, distance_scores, position_scores, distance_scores, position_scores, distance_scores, position_scores]
    colors=[Settings.allocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color, Settings.egocentric_color]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2, 4,4,5,6,7], boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if unlock_y:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(bottom=0, top=3)
    ax.set_xlim(left=0.5, right=3.5)
    ax.set_xticks([1,2])
    ax.set_yticks([0,1, 2, 3])
    ax.set_yticklabels(["00", "01", "02", "03"])
    plt.xticks(rotation = 30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_ylabel("Spatial info bits/s", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=25)
    fig.tight_layout()
    plt.savefig(save_path + '/' +'position_vs_distance_trials_spatial_scores_'+suffix+'.png', dpi=300)
    plt.close()

def behaviours_classification_to_numeric(behaviours_classifications):
    numeric_classifications = []
    for i in range(len(behaviours_classifications)):
        if behaviours_classifications[i] == "hit":
            numeric_classifications.append(1)
        elif behaviours_classifications[i] == "try":
            numeric_classifications.append(0.5)
        elif behaviours_classifications[i] == "miss":
            numeric_classifications.append(0)
        else:
            numeric_classifications.append(np.nan)
    return np.array(numeric_classifications)


def generate_long_form_for_R_by_session(combined_df):
    position_sessions = combined_df[combined_df["rolling:proportion_encoding_position"] >= 0.85]
    distance_sessions = combined_df[combined_df["rolling:proportion_encoding_distance"] >= 0.85]

    df = pd.DataFrame()
    # iterate over position_sessions
    for session_dfs, session_type in zip([position_sessions, distance_sessions], ["Position", "Distance"]):
        for index, session_row in session_dfs.iterrows():
            session_row = session_row.to_frame().T.reset_index(drop=True)
            session_id = session_row["session_id"].iloc[0]
            cluster_id = session_row["cluster_id"].iloc[0]
            classifier = session_row["classifier"].iloc[0]
            mouse_id = session_row["mouse"].iloc[0]
            behaviours = np.array(session_row["behaviour_hit_try_miss"].iloc[0])
            hits_binary = np.array(behaviours == "hit", dtype=np.int0)
            trial_types = np.array(session_row["behaviour_trial_types"].iloc[0])
            trial_numbers = np.array(session_row["behaviour_trial_numbers"].iloc[0])

            # remove rejected trials
            valid_trials_mask = behaviours != "rejected"
            behaviours = behaviours[valid_trials_mask]
            hits_binary = hits_binary[valid_trials_mask]
            trial_types = trial_types[valid_trials_mask]
            trial_numbers = trial_numbers[valid_trials_mask]

            for i, tn in enumerate(trial_numbers):
                trial_df = pd.DataFrame()
                trial_df["mouse_id"] = [mouse_id]
                trial_df["session_id"] = [session_id]
                trial_df["cluster_id"] = [cluster_id]
                trial_df["classifier"] = [classifier]
                trial_df["trial_type"] = [trial_types[i]]
                trial_df["Lomb_classifier"] = [session_type]
                trial_df["trial_number"] = [tn]
                trial_df["hit"] = [hits_binary[i]]
                trial_df["behavioural_outcome"] = [behaviours[i]]
                df = pd.concat([df, trial_df], ignore_index=True)
    return df

def generate_long_form_for_R(combined_df):
    df = pd.DataFrame()
    # iterate over clusters
    for index, cluster_row in combined_df.iterrows():
        cluster_row = cluster_row.to_frame().T.reset_index(drop=True)
        cluster_id = cluster_row["cluster_id"].iloc[0]
        session_id = cluster_row["session_id"].iloc[0]
        classifier = cluster_row["classifier"].iloc[0]
        grid_cell = cluster_row["grid_cell"].iloc[0]
        mouse_id = cluster_row["mouse"].iloc[0]
        trial_numbers = np.array(cluster_row["behaviour_trial_numbers"].iloc[0])
        behaviours = np.array(cluster_row["behaviour_hit_try_miss"].iloc[0])
        behaviours_numeric = behaviours_classification_to_numeric(behaviours)
        hits_binary = np.array(behaviours == "hit", dtype=np.int0)
        trial_types = np.array(cluster_row["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cluster_row["rolling:classifier_by_trial_number"].iloc[0])
        alternative_classifiers = np.array(cluster_row["alternative_classifications"].iloc[0])
        correlation_to_template = np.array(cluster_row["rolling:position_correlation_by_trial_number_t2tmethod"].iloc[0])
        p_coding = np.array(rolling_classifiers == "P", dtype=np.int0)
        d_coding = np.array(rolling_classifiers == "D", dtype=np.int0)
        n_coding = np.array(rolling_classifiers == "N", dtype=np.int0)
        p_coding_alt = np.array(alternative_classifiers == "P", dtype=np.int0)

        # remove rejected trials
        valid_trials_mask = behaviours != "rejected"
        trial_numbers = trial_numbers[valid_trials_mask]
        behaviours = behaviours[valid_trials_mask]
        behaviours_numeric = behaviours_numeric[valid_trials_mask]
        hits_binary = hits_binary[valid_trials_mask]
        trial_types = trial_types[valid_trials_mask]
        rolling_classifiers = rolling_classifiers[valid_trials_mask]
        alternative_classifiers = alternative_classifiers[valid_trials_mask]
        correlation_to_template = correlation_to_template[valid_trials_mask]
        p_coding_alt = p_coding_alt[valid_trials_mask]
        p_coding = p_coding[valid_trials_mask]
        d_coding = d_coding[valid_trials_mask]
        n_coding = n_coding[valid_trials_mask]

        cell_df = pd.DataFrame()
        cell_df["trial_number"] = np.array(trial_numbers)
        cell_df["hit"] = np.array(hits_binary)
        cell_df["behaviour_numeric"] = np.array(behaviours_numeric)
        cell_df["behavioural_outcome"] = np.array(behaviours)
        cell_df["trial_type"] = np.array(trial_types)
        cell_df["rolling_classifier"] = np.array(rolling_classifiers)
        cell_df["rolling_classifier_alt"] = np.array(alternative_classifiers)
        cell_df["p_coding_alt"] = np.array(p_coding_alt)
        cell_df["template_corr"] = np.array(correlation_to_template)
        cell_df["p_coding"] = np.array(p_coding)
        cell_df["d_coding"] = np.array(d_coding)
        cell_df["n_coding"] = np.array(n_coding)
        cell_df["mouse_id"] = mouse_id
        cell_df["session_id"] = session_id
        cell_df["cluster_id"] = cluster_id
        cell_df["classifier"] = classifier
        cell_df["grid_cell"] = grid_cell
        df = pd.concat([df, cell_df], ignore_index=True)
    return df

def plot_odds_ratio(odds_ratio_data, save_path,suffix=""):
    beaconed_odds_ratio = odds_ratio_data[odds_ratio_data["trial_type"] == "beaconed"]
    non_beaconed_odds_ratio = odds_ratio_data[odds_ratio_data["trial_type"] == "non_beaconed"]
    probe_odds_ratio = odds_ratio_data[odds_ratio_data["trial_type"] == "probe"]

    fig, ax = plt.subplots(figsize=(4,4))
    y = 9.3
    y_increment = 0.6
    linewidth=3
    s=75

    cell_types = ["g"]
    cell_colors = ["black"]

    for cell_type, cell_color in zip(cell_types, cell_colors):
        ax.plot([beaconed_odds_ratio[beaconed_odds_ratio["cell_type"] == cell_type]["ci_lower"].iloc[0],
                 beaconed_odds_ratio[beaconed_odds_ratio["cell_type"] == cell_type]["ci_upper"].iloc[0]], [y, y], color=cell_color, linewidth=linewidth)
        ax.scatter(y=y, x=beaconed_odds_ratio[beaconed_odds_ratio["cell_type"] == cell_type]["odds_ratio"].iloc[0],color=cell_color, s=s)
        y -= y_increment

    y = 6.3

    for cell_type, cell_color in zip(cell_types, cell_colors):
        ax.plot([non_beaconed_odds_ratio[non_beaconed_odds_ratio["cell_type"] == cell_type]["ci_lower"].iloc[0],
                 non_beaconed_odds_ratio[non_beaconed_odds_ratio["cell_type"] == cell_type]["ci_upper"].iloc[0]], [y, y], color=cell_color, linewidth=linewidth)
        ax.scatter(y=y, x=non_beaconed_odds_ratio[non_beaconed_odds_ratio["cell_type"] == cell_type]["odds_ratio"].iloc[0],color=cell_color, s=s)
        y -= y_increment

    y = 3.3

    for cell_type, cell_color in zip(cell_types, cell_colors):
        ax.plot([probe_odds_ratio[probe_odds_ratio["cell_type"] == cell_type]["ci_lower"].iloc[0],
                 probe_odds_ratio[probe_odds_ratio["cell_type"] == cell_type]["ci_upper"].iloc[0]], [y, y], color=cell_color, linewidth=linewidth)
        ax.scatter(y=y, x=probe_odds_ratio[probe_odds_ratio["cell_type"] == cell_type]["odds_ratio"].iloc[0],color=cell_color, s=s)
        y -= y_increment


    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=25)
    #ax.set_yticks([0,0.5,1])
    ax.set_xscale('log')
    ax.set_xticks([      0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1,  2,  3,  4,  5,   6,  7,  8,  9,  10])
    ax.set_xticklabels(["0.1",  "",  "",  "","0.5", "",  "",  "",  "", "1","", "", "", "5", "", "", "", "", "10"])
    ax.set_ylim(1, 11)
    ax.set_yticks([3.3, 6.3, 9.3])
    ax.set_yticklabels(["P", "N", "B"])
    #ax.set_xlim(left=0, right=9)
    ax.axvline(x=1, color="black",linestyle="dotted")
    #ax.set_ylim(bottom=0, top=100)
    plt.savefig(save_path + '/odds_ratio'+suffix+'.png', dpi=300)
    plt.close()
    return

def concatenate_dataframe_from_path_list(path_list):
    df = pd.DataFrame()
    for i in range(len(path_list)):
        df_from_list = pd.read_pickle(path_list[i])
        df = pd.concat([df, df_from_list], ignore_index=True)
    return df

def compile_shuffled_agreements(agreement_dataframe):
    # this function averages over shuffled agreement scores from the same cell pair
    compiled_df = pd.DataFrame()
    for session_id in np.unique(agreement_dataframe["session_id"]):
        session_df = agreement_dataframe[agreement_dataframe["session_id"] == session_id]
        for cluster_id_i in np.unique(session_df["cluster_id_i"]):
            cluster_id_i_df = session_df[session_df["cluster_id_i"] == cluster_id_i]
            for cluster_id_j in np.unique(cluster_id_i_df["cluster_id_j"]):
                pair_df = cluster_id_i_df[cluster_id_i_df["cluster_id_j"] == cluster_id_j]
                true_pair_df = pair_df[pair_df["shuffled"] == 0]
                shuffled_pair_df = pair_df[pair_df["shuffled"] == 1]
                shuffle_avg = np.nanmean(np.array(shuffled_pair_df["agreement"], dtype=np.float64))
                new_shuffled_df = true_pair_df.copy()
                new_shuffled_df["shuffled"] = 1
                new_shuffled_df["agreement"] = shuffle_avg
                compiled_df = pd.concat([compiled_df, true_pair_df], ignore_index=True)
                compiled_df = pd.concat([compiled_df, new_shuffled_df], ignore_index=True)
    return compiled_df

def generate_agreement_dataframe(combined_df, n_shuffles=10):
    agreement_df = pd.DataFrame()
    session_ids = np.unique(combined_df["session_id_vr"])
    for session_id in session_ids:
        session_df = combined_df[combined_df["session_id_vr"] == session_id]
        mouse_id = session_df["mouse"].iloc[0]

        for cluster_id_i in session_df["cluster_id"]:
            for cluster_id_j in session_df["cluster_id"]:
                if cluster_id_i != cluster_id_j:

                    cell_rolling_classifiers_i = np.array(session_df[session_df["cluster_id"] == cluster_id_i]["rolling:classifier_by_trial_number"].iloc[0])
                    cell_rolling_classifiers_j = np.array(session_df[session_df["cluster_id"] == cluster_id_j]["rolling:classifier_by_trial_number"].iloc[0])
                    cell_rolling_classifiers_i = cell_rolling_classifiers_i[cell_rolling_classifiers_i != "nan"]
                    cell_rolling_classifiers_j = cell_rolling_classifiers_j[cell_rolling_classifiers_j != "nan"]

                    cell_classifier_i = np.array(session_df[session_df["cluster_id"] == cluster_id_i]["classifier"].iloc[0])
                    cell_classifier_j = np.array(session_df[session_df["cluster_id"] == cluster_id_j]["classifier"].iloc[0])
                    true_agreement = np.sum(cell_rolling_classifiers_i==cell_rolling_classifiers_j)/len(cell_rolling_classifiers_i)

                    true_df = pd.DataFrame()
                    true_df["mouse_id"] = [mouse_id]
                    true_df["session_id"] = [session_id]
                    true_df["cluster_id_i"] = [cluster_id_i]
                    true_df["cluster_id_j"] = [cluster_id_j]
                    true_df["classifier_i"] = [cell_classifier_i]
                    true_df["classifier_j"] = [cell_classifier_j]
                    true_df["shuffled"] = [0]
                    true_df["agreement"] = [true_agreement]
                    agreement_df = pd.concat([agreement_df, true_df], ignore_index=True)

                    for n in range(n_shuffles):
                        shuffled_cell_rolling_classifiers_j = shuffle_blocks(cell_rolling_classifiers_j)
                        shuffled_agreement = np.sum(cell_rolling_classifiers_i==shuffled_cell_rolling_classifiers_j)/len(cell_rolling_classifiers_i)

                        shuffled_df = pd.DataFrame()
                        shuffled_df["mouse_id"] = [mouse_id]
                        shuffled_df["session_id"] = [session_id]
                        shuffled_df["cluster_id_i"] = [cluster_id_i]
                        shuffled_df["cluster_id_j"] = [cluster_id_j]
                        shuffled_df["classifier_i"] = [cell_classifier_i]
                        shuffled_df["classifier_j"] = [cell_classifier_j]
                        shuffled_df["shuffled"] = [1]
                        shuffled_df["agreement"] = [shuffled_agreement]
                        agreement_df = pd.concat([agreement_df, shuffled_df], ignore_index=True)

    # function to fix df, currently will have duplicates
    def remove_duplicate_pairs(agreement_df):
        new_df = pd.DataFrame()
        session_ids = np.unique(agreement_df["session_id"])
        for session_id in session_ids:
            new_session_df = pd.DataFrame()
            session_df = agreement_df[agreement_df["session_id"] == session_id]
            pairs=[]
            for index, pair_row in session_df.iterrows():
                pair_row = pair_row.to_frame().T.reset_index(drop=True)
                cell_i = pair_row["cluster_id_i"].iloc[0]
                cell_j = pair_row["cluster_id_j"].iloc[0]

                include_pair = True
                # check the pair labels
                for pair in list(set(pairs)):
                    if (pair[0] == cell_j) and (pair[1] == cell_i):
                        include_pair=False
                if include_pair:
                    pairs.append((cell_i, cell_j))
                    new_session_df = pd.concat([new_session_df, pair_row], ignore_index=True)
            new_df = pd.concat([new_df, new_session_df], ignore_index=True)

        return new_df

    agreement_df = remove_duplicate_pairs(agreement_df)
    return agreement_df

def get_agreements_from_df(agreement_df, celltypei="G", celltypeii="G", shuffled=0):
    agreement_df[agreement_df["classifier_i"] != "G"]["classifier_i"] = "NG"
    agreement_df[agreement_df["classifier_j"] != "G"]["classifier_i"] = "NG"
    new_df = agreement_df[(agreement_df["classifier_i"] == celltypei) &
                          (agreement_df["classifier_j"] == celltypeii)]
    new_df = new_df[new_df["shuffled"] == shuffled]
    return np.array(new_df["agreement"], dtype=np.float64)

def plot_agreement_between_grid_and_non_grids(agreement_df, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    colors = ["dimgrey", "lightcoral", "dimgrey", "lightcoral", "dimgrey", "lightcoral"]
    gc_gc_agreement_true = get_agreements_from_df(agreement_df, celltypei="G", celltypeii="G", shuffled=0)*100
    gc_ngc_agreement_true = get_agreements_from_df(agreement_df, celltypei="G", celltypeii="NG", shuffled=0)*100
    ngc_ngc_agreement_true = get_agreements_from_df(agreement_df, celltypei="NG", celltypeii="NG", shuffled=0)*100
    gc_gc_agreement_shuffle = get_agreements_from_df(agreement_df, celltypei="G", celltypeii="G", shuffled=1)*100
    gc_ngc_agreement_shuffle = get_agreements_from_df(agreement_df, celltypei="G", celltypeii="NG", shuffled=1)*100
    ngc_ngc_agreement_shuffle = get_agreements_from_df(agreement_df, celltypei="NG", celltypeii="NG", shuffled=1)*100

    data = [gc_gc_agreement_true, gc_gc_agreement_shuffle, ngc_ngc_agreement_true, ngc_ngc_agreement_shuffle,
            gc_ngc_agreement_true, gc_ngc_agreement_shuffle]
    boxprops = dict(linewidth=3, color='k')
    medianprops = dict(linewidth=3, color='k')
    capprops = dict(linewidth=3, color='k')
    whiskerprops = dict(linewidth=3, color='k')
    box = ax.boxplot(data, positions=[1,2, 4,5, 7,8], vert=True, widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=2)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 100])
    ax.set_yticklabels([0, 25,50,75,100])
    ax.tick_params(axis='both', which='both', labelsize=20)
    plt.savefig(save_path + '/agreements_between_grid_cells_and_non_grid_cells_boxplot.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 1))
    data = [gc_gc_agreement_true-gc_gc_agreement_shuffle, ngc_ngc_agreement_true-ngc_ngc_agreement_shuffle,
            gc_ngc_agreement_true-gc_ngc_agreement_shuffle]
    ax.axhline(y=0, color="black",linestyle="dashed")
    parts = ax.violinplot(data, positions=[1.5, 4.5, 7.5], widths=2, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('black')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax.set_xticklabels(["", "", "", ""])
    ax.set_yticks([0, 50])
    ax.set_yticklabels([0,50])
    ax.set_xlim([0, 12])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=20)
    plt.savefig(save_path + '/delta_agreements_between_grid_cells_and_non_grid_cells_violinplot.png', dpi=300)
    plt.close()
    return

def plot_spatial_information_during_dominant_PandD_grid_modes(combined_df, save_path):
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    spatial_info_P_G = np.array(grid_cells["spatial_information_during_P"])
    spatial_info_D_G = np.array(grid_cells["spatial_information_during_D"])
    spatial_info_P_NG = np.array(non_grid_cells["spatial_information_during_P"])
    spatial_info_D_NG = np.array(non_grid_cells["spatial_information_during_D"])

    not_nan_mask_G = ~np.isnan(spatial_info_P_G) & ~np.isnan(spatial_info_D_G)
    not_nan_mask_NG = ~np.isnan(spatial_info_P_NG) & ~np.isnan(spatial_info_D_NG)

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.plot(np.arange(0,45), np.arange(0,45), linestyle="dashed", color="black")
    ax.scatter(spatial_info_P_NG[not_nan_mask_NG], spatial_info_D_NG[not_nan_mask_NG], marker="o", s=40, facecolors=np.array([0/255, 154/255, 255/255, 1]).reshape(1,4), linewidths=0.5, edgecolors="black")
    ax.scatter(spatial_info_P_G[not_nan_mask_G], spatial_info_D_G[not_nan_mask_G], marker="o", s=40, facecolors=np.array([255/255,127/255,14/255, 1]).reshape(1,4), linewidths=0.5,  edgecolors="black")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.get_major_locator().set_params(numticks=99)
    ax.xaxis.get_minor_locator().set_params(numticks=99, subs=[.1, .2,.3, .4,.5, .6, .7,.8, .9])
    ax.yaxis.get_major_locator().set_params(numticks=99)
    ax.yaxis.get_minor_locator().set_params(numticks=99, subs=[.1, .2,.3, .4,.5, .6, .7,.8, .9])
    ax.tick_params(axis='both', which='major', labelsize=12)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.subplots_adjust(hspace=None, wspace=None, bottom=None, left=0.2, right=None, top=None)
    plt.savefig(save_path + '/spatial_info_P_vs_spatial_info_D.png', dpi=300)
    plt.close()

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(2.5, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.axhline(y=0, linestyle="dashed", color="black", linewidth=2)
    colors = [np.array([255/255,127/255,14/255, 1]).reshape(1,4), np.array([0/255, 154/255, 255/255, 1]).reshape(1,4)]

    data = [np.asarray((spatial_info_P_G[not_nan_mask_G] - spatial_info_D_G[not_nan_mask_G]) / spatial_info_P_G[not_nan_mask_G])*100,
            np.asarray((spatial_info_P_NG[not_nan_mask_NG]-spatial_info_D_NG[not_nan_mask_NG]) / spatial_info_P_NG[not_nan_mask_NG])*100]
    boxprops = dict(linewidth=2, color='k')
    medianprops = dict(linewidth=2, color='k')
    capprops = dict(linewidth=2, color='k')
    whiskerprops = dict(linewidth=2, color='k')
    box = ax.boxplot(data, positions=[1,3], widths=1, boxprops=boxprops, medianprops=medianprops,
                     whiskerprops=whiskerprops, capprops=capprops, patch_artist=True, showfliers=False, zorder=2)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([1,3])
    ax.set_xticklabels(["G", "NG"])
    ax.set_yticks([-100, 0, 100])
    ax.set_ylim([-120, 120])
    #ax.set_yticks([-1, 0, 1])
    #ax.set_ylim([-2, 2])
    ax.set_xlim([0, 4])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.tick_params(axis='both', which='both', labelsize=15)
    plt.subplots_adjust(hspace = None, wspace = None,  bottom = None, left = 0.4, right =None, top = None)
    plt.savefig(save_path + '/spatial_info_P_vs_spatial_info_D_boxplot.png', dpi=300)
    plt.close()
    return


def plot_field_distributions_stable_cells(combined_df, save_path):
    combined_df = combined_df[combined_df["classifier"] == "G"]
    
    position_cells = combined_df[combined_df["Lomb_classifier_"] == "Position"]
    distance_cells = combined_df[combined_df["Lomb_classifier_"] == "Distance"]

    stable_position_cells = position_cells[position_cells["rolling:proportion_encoding_position"] >= 0.85]
    stable_distance_cells = distance_cells[distance_cells["rolling:proportion_encoding_distance"] >= 0.85]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for cells, color in zip([stable_position_cells, stable_distance_cells],
                            [Settings.allocentric_color, Settings.egocentric_color]):
        field_dist = []
        weights = []
        for i in range(len(cells)):
            session_id = cells["session_id"].iloc[0]
            session_weight = 1 / len(cells[cells["session_id"] == session_id])
            field_locations = np.array(cells["field_locations"].iloc[i])
            hist, bin_edges = np.histogram(field_locations, bins=40,range=(0, 200), density=True)
            hist = hist / hist.sum()
            if np.isnan(np.sum(hist)):
                session_weight = 0

            field_dist.append(hist)
            weights.append(session_weight)

        bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        field_dist = np.array(field_dist)
        field_dist[np.isnan(field_dist)] = 0
        weights = np.array(weights)

        ax.plot(bin_centres, np.average(field_dist, axis=0, weights=weights), color=color)
        ax.fill_between(bin_centres,
                        np.average(field_dist, axis=0, weights=weights) - stats.sem(field_dist, axis=0,
                                                                                        nan_policy="omit"),
                        np.average(field_dist, axis=0, weights=weights) + stats.sem(field_dist, axis=0,
                                                                                        nan_policy="omit"),
                        color=color, alpha=0.3)

    style_track_plot(ax, 200)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=20)
    #ax.set_yticks([0.02, 0.03, 0.04])
    ax.set_xticks([0, 100, 200])
    ax.set_xlim(left=0, right=200)
    ax.set_ylim(top=0.05)
    plt.subplots_adjust(hspace=.25, wspace=.25, bottom=None, left=0.2, right=None, top=None)
    plt.savefig(save_path + '/field_dist_p_and_d_stable.png', dpi=300)
    plt.close()
    return


def plot_field_distributions(combined_df, save_path):
    combined_df = combined_df[combined_df["classifier"] == "G"]

    fig, ax = plt.subplots(1,1, figsize=(6,4))

    field_dist_p = []
    field_dist_d = []
    weights_p = []
    weights_d = []
    for i in range(len(combined_df)):
        session_id = combined_df["session_id"].iloc[0]
        session_weight = 1 / len(combined_df[combined_df["session_id"] == session_id])

        field_locations = np.array(combined_df["field_locations"].iloc[i])
        field_trial_numbers = np.array(combined_df["field_trial_numbers"].iloc[i])

        behaviour_trial_numbers = np.array(combined_df["behaviour_trial_numbers"].iloc[i])
        trial_modes = np.asarray(combined_df["rolling:classifier_by_trial_number"].iloc[i])

        valid_trial_numbers_p = behaviour_trial_numbers[(trial_modes == "P")]
        valid_trial_numbers_d = behaviour_trial_numbers[(trial_modes == "D")]

        p_weight = len(valid_trial_numbers_p) / len(behaviour_trial_numbers)
        d_weight = len(valid_trial_numbers_d) / len(behaviour_trial_numbers)

        hist_p, bin_edges = np.histogram(field_locations[np.isin(field_trial_numbers, valid_trial_numbers_p)], bins=40, range=(0, 200), density=True)
        hist_d, bin_edges = np.histogram(field_locations[np.isin(field_trial_numbers, valid_trial_numbers_d)], bins=40, range=(0, 200), density=True)
        hist_p = hist_p / hist_p.sum()
        hist_d = hist_d / hist_d.sum()

        final_weight_p = session_weight * p_weight
        final_weight_d = session_weight * d_weight
        weights_p.append(final_weight_p)
        weights_d.append(final_weight_d)
        field_dist_p.append(hist_p)
        field_dist_d.append(hist_d)

    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    field_dist_p = np.array(field_dist_p)
    field_dist_d = np.array(field_dist_d)
    field_dist_p[np.isnan(field_dist_p)] = 0
    field_dist_d[np.isnan(field_dist_d)] = 0
    weights_p = np.array(weights_p)
    weights_d = np.array(weights_d)

    ax.plot(bin_centres, np.average(field_dist_p, axis=0, weights=weights_p), color=Settings.allocentric_color)
    ax.fill_between(bin_centres,
                                 np.average(field_dist_p, axis=0, weights=weights_p) - stats.sem(field_dist_p, axis=0, nan_policy="omit"),
                                 np.average(field_dist_p, axis=0, weights=weights_p) + stats.sem(field_dist_p, axis=0, nan_policy="omit"),
                                 color=Settings.allocentric_color, alpha=0.3)
    ax.plot(bin_centres, np.average(field_dist_d, axis=0, weights=weights_d), color=Settings.egocentric_color)

    ax.fill_between(bin_centres,
                                 np.average(field_dist_d, axis=0, weights=weights_d) - stats.sem(field_dist_d, axis=0, nan_policy="omit"),
                                 np.average(field_dist_d, axis=0, weights=weights_d) + stats.sem(field_dist_d, axis=0, nan_policy="omit"),
                                 color=Settings.egocentric_color, alpha=0.3)
    style_track_plot(ax, 200)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_yticks([0.02, 0.03, 0.04])
    ax.set_xticks([0, 100, 200])
    ax.set_xlim(left=0, right=200)
    ax.set_ylim(top=0.05)
    plt.subplots_adjust(hspace=.25, wspace=.25, bottom=None, left=0.2, right=None, top=None)
    plt.savefig(save_path + '/field_dist_p_and_d.png',dpi=300)
    plt.close()
    return

def plot_speed_profiles_between_p_and_d_groups(df, suffix="", modes=["all"], save_path=""):
    # if not concerned with looking at coding groups then drop duplicate sessions
    if modes[0] == "all":
        df= drop_duplicate_sessions(df)

    fig, ax = plt.subplots(3,3, figsize=(13,8), sharex=True, sharey=True)
    for tt_i, tt in enumerate([0,1,2]):
        for hmt_i, hmt in enumerate(["hit", "try", "miss"]):
            for mode in modes:
                speeds_to_plot = []
                for index, row in df.iterrows():
                    row = row.to_frame().T.reset_index(drop=True)
                    behaviour_trial_numbers = np.array(row["behaviour_trial_numbers"].iloc[0])
                    behaviour_hit_try_miss = np.array(row["behaviour_hit_try_miss"].iloc[0])
                    behaviour_trial_types = np.array(row["behaviour_trial_types"].iloc[0])
                    trial_modes = np.asarray(row["rolling:classifier_by_trial_number"].iloc[0])
                    speeds = np.asarray(row["speeds_binned_in_space_smoothed"].iloc[0])
                    mask = (behaviour_hit_try_miss == hmt) & (behaviour_trial_types == tt)

                    if mode == "all":
                        mode_mask = mask
                    else:
                        mode_mask = mask & (trial_modes == mode)
                    if len(speeds[mode_mask])>0:
                        speeds_to_plot.extend(speeds[mode_mask])
                speeds_to_plot = np.array(speeds_to_plot)

                if mode == "P":
                    color = Settings.allocentric_color
                elif mode == "D":
                    color = Settings.egocentric_color
                else:
                    color = get_hmt_color(hmt)
                ax[tt_i, hmt_i].plot(np.arange(0.5, 200.5, 1), np.nanmean(speeds_to_plot, axis=0), color=color)
                ax[tt_i, hmt_i].fill_between(np.arange(0.5, 200.5, 1), np.nanmean(speeds_to_plot, axis=0)-np.nanstd(speeds_to_plot, axis=0),
                                np.nanmean(speeds_to_plot, axis=0)+np.nanstd(speeds_to_plot, axis=0), color=color, alpha=0.3)

            if tt == 0:
                style_track_plot(ax[tt_i, hmt_i], 200)
            else:
                style_track_plot_no_RZ(ax[tt_i, hmt_i], 200)
            ax[tt_i, hmt_i].spines['top'].set_visible(False)
            ax[tt_i, hmt_i].spines['right'].set_visible(False)
            ax[tt_i, hmt_i].yaxis.set_ticks_position('left')
            ax[tt_i, hmt_i].xaxis.set_ticks_position('bottom')
            ax[tt_i, hmt_i].tick_params(axis='both', which='both', labelsize=20)
            ax[tt_i, hmt_i].set_yticks([0, 50, 100])
            ax[tt_i, hmt_i].set_xticks([0, 100, 200])
            ax[tt_i, hmt_i].set_xlim(left=0, right=200)
            ax[tt_i, hmt_i].set_ylim(bottom=0, top=100)
    plt.subplots_adjust(hspace=.25, wspace=.25, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/speed_profiles' + '_' + suffix + '.png', dpi=300)
    plt.close()
    return

def plot_trial_type_ratio_against_task_anchored_and_independent_ratio(df, save_path):

    trial_type_ratios = []
    task_anchored_and_independent_ratios = []
    for index, cell in df.iterrows():
        cell = cell.to_frame().T.reset_index(drop=True)

        trial_types = np.array(cell["behaviour_trial_types"].iloc[0])
        rolling_classifiers = np.array(cell["rolling:classifier_by_trial_number"].iloc[0])
        trial_type_ratio = np.sum(trial_types==0)/len(trial_types)#
        task_anchored_and_independent_ratio = np.sum(rolling_classifiers=="P")/len(rolling_classifiers)

        trial_type_ratios.append(trial_type_ratio)
        task_anchored_and_independent_ratios.append(task_anchored_and_independent_ratio)

    trial_type_ratios=np.array(trial_type_ratios)*100
    task_anchored_and_independent_ratios=np.array(task_anchored_and_independent_ratios)*100

    spikes_on_track = plt.figure()
    spikes_on_track.set_size_inches(3, 3, forward=True)
    ax = spikes_on_track.add_subplot(1, 1, 1)
    ax.plot(np.arange(-50,500), np.arange(-50,500), linestyle="dashed", color="black")
    ax.scatter(trial_type_ratios, task_anchored_and_independent_ratios, marker="o", s=20, facecolors=np.array([255/255, 255/255, 255/ 255, 1]).reshape(1, 4), linewidths=0.5,edgecolors="black")
    plot_regression(ax, trial_type_ratios, task_anchored_and_independent_ratios, c="red", y_text_pos=1.1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.subplots_adjust(bottom=.2, left=.2, right=None, top=None)
    plt.savefig(save_path + '/grid_cells_trial_type_ratio_against_task_anchored_and_independent_ratio.png', dpi=300)
    plt.close()


def plot_histogram_of_position_template_correlations(spike_data, save_path):
    correlations = pandas_collumn_to_numpy_array(spike_data["rolling:position_correlation_by_trial_number_t2tmethod"])
    original_classification = pandas_collumn_to_numpy_array(spike_data["rolling:classifier_by_trial_number"])

    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(correlations[original_classification=="P"], range=(-1, 1), bins=100, color=Settings.allocentric_color, density=True)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(-1, 1)
    ax.tick_params(axis='both', which='both', labelsize=20)
    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.2, right=0.87, top=0.92)
    plt.savefig(save_path + '/correlations_hist_against_position_template_P.png', dpi=300)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(correlations[original_classification=="D"], range=(-1, 1), bins=100, color=Settings.egocentric_color, density=True)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(-1, 1)
    ax.tick_params(axis='both', which='both', labelsize=20)
    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.2, right=0.87, top=0.92)
    plt.savefig(save_path + '/correlations_hist_against_position_template_D.png', dpi=300)
    plt.close()
    return

def add_alternative_classification(df, column_name="rolling:position_correlation_by_trial_number_t2tmethod", threshold=0.5):
    proportion_p = []
    proportion_d = []
    alternative_classifications_all_clusters=[]
    for index, cell in df.iterrows():
        cell = cell.to_frame().T.reset_index(drop=True)
        correlations = np.array(cell[column_name].iloc[0])
        P = correlations >= threshold
        D = correlations < threshold
        classifications = np.empty(len(correlations), dtype=object)
        classifications[P] = "P" # P label denotes TA + in the manuscript
        classifications[D] = "D" # D label denotes TA - in the manuscript
        classifications[np.isnan(correlations)] = "D"
        alternative_classifications_all_clusters.append(classifications)
        proportion_p.append(len(classifications[classifications == "P"]) / len(classifications))
        proportion_d.append(len(classifications[classifications == "D"]) / len(classifications))

    df["alternative_classifications"] = alternative_classifications_all_clusters
    df["rolling:proportion_encoding_position_alternative_method"] = proportion_p
    df["rolling:proportion_encoding_distance_alternative_method"] = proportion_d
    return df


def plot_avg_speed_profiles(processed_position_data, save_path):
    # remove low speed trials
    processed_position_data = processed_position_data[processed_position_data["hit_miss_try"] != "rejected"]

    session_avg = []
    for session_id in np.unique(processed_position_data["session_id_vr"]):
        session_processed_position_data = processed_position_data[processed_position_data["session_id_vr"] == session_id]
        session_speed_profiles = pandas_collumn_to_2d_numpy_array(session_processed_position_data['speeds_binned_in_space_smoothed'])
        avg_session_speed_profile = np.nanmean(session_speed_profiles, axis=0)
        session_avg.append(avg_session_speed_profile.tolist())
    session_avg = np.array(session_avg)
    locations = np.array(processed_position_data['position_bin_centres'].iloc[0])

    speed_histogram = plt.figure(figsize=(6, 4))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    for i in range(len(session_avg)):
        ax.plot(locations, session_avg[i], color="grey", linewidth=2,alpha=0.5)
    ax.plot(locations, np.nanmean(session_avg, axis=0), color="black", linewidth=4)
    ax.axhline(y=4.7, color="black", linestyle="dashed", linewidth=2)
    plt.xlim(0, 200)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    style_track_plot(ax, 200)
    tick_spacing = 100
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    x_max = 115
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.savefig(save_path + '/avg_session_speed_profile.png', dpi=300)
    plt.close()

    mouse_colors = ['darkturquoise', 'salmon', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',u'#bcbd22', u'#17becf']
    for mouse, mouse_color in zip(np.unique(processed_position_data["mouse_id"]), mouse_colors):
        mouse_processed_position_data = processed_position_data[processed_position_data["mouse_id"] == mouse]

        session_avg = []
        for session_id in np.unique(mouse_processed_position_data["session_id_vr"]):
            session_processed_position_data = mouse_processed_position_data[
                mouse_processed_position_data["session_id_vr"] == session_id]
            session_speed_profiles = pandas_collumn_to_2d_numpy_array(
                session_processed_position_data['speeds_binned_in_space_smoothed'])
            avg_session_speed_profile = np.nanmean(session_speed_profiles, axis=0)
            session_avg.append(avg_session_speed_profile.tolist())
        session_avg = np.array(session_avg)
        locations = np.array(mouse_processed_position_data['position_bin_centres'].iloc[0])

        speed_histogram = plt.figure(figsize=(6, 4))
        ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        for i in range(len(session_avg)):
            ax.plot(locations, session_avg[i], color=mouse_color, linewidth=2, alpha=0.5)
        ax.plot(locations, np.nanmean(session_avg, axis=0), color="black", linewidth=4)
        ax.axhline(y=4.7, color="black", linestyle="dashed", linewidth=2)
        plt.xlim(0, 200)
        ax.set_yticks([0, 50, 100])
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        style_track_plot(ax, 200)
        tick_spacing = 100
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        x_max = 115
        plot_utility.style_vr_plot(ax, x_max)
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.savefig(save_path + '/avg_session_speed_profile_'+str(mouse)+'.png', dpi=300)
        plt.close()
    return

def main():
    print("------------------------------------------------------------------------------------------------")
    print("-----------------------------------------Hello there--------------------------------------------")
    print("------------------------------------------------------------------------------------------------")
    print("---- This analysis was run on a ubuntu 16.04 image on a 96GB ram virtual machine (with 30GB ----")
    print("---- extra swap space) using the conda environment provided in grid_behaviour.yml           ----")
    print("------------------------------------------------------------------------------------------------")
    print("---- This script (1) loads spatial firing dataframes                                        ----")
    print("----             (2) filters out cells based on an exclusion criteria                       ----")
    print("----             (3) creates dataframes ready to be exported to R formats                   ----")
    print("----             (4) creates population-level plots as seen in Clark and Nolan 2023         ----")
    print("------------------------------------------------------------------------------------------------")

    # load dataframe, each row represents one cell
    combined_df = pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/combined_cohorts.pkl")

    """
    combined_df = pd.DataFrame()
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/combined_cohort6.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/combined_cohort7.pkl")], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/combined_cohort8.pkl")], ignore_index=True)

    # TODO DELETE ONCE DONE ONCE #######
    collumn_names_to_keep = ['cluster_id', 'primary_channel', 'session_id', 'session_id_vr', 'session_id_of',
                             'recording_length_sampling_points', 'recording_length_seconds',
                             'tetrode', 'number_of_spikes', 'mean_firing_rate_of', 'isolation', 'noise_overlap',
                             'peak_snr', 'peak_amp', 'snippet_peak_to_trough',
                             'speed_score', 'hd_score', 'spatial_information_score', 'grid_spacing', 'grid_score',
                             'border_score',
                             'rate_map_correlation_first_vs_second_half',
                             'percent_excluded_bins_rate_map_correlation_first_vs_second_half_p',
                             'firing_maps_of', 'occupancy_maps_of', 'rate_map_autocorrelogram_of', 
                             'grid_cell', 'border_cell', 'hd_cell', 'spatial_cell', 'speed_cell', 'classifier',
                             'spatial_information_score_Isec_vr',
                             'spatial_information_score_Ispike_vr', 'firing_times_vr', 'fr_binned_in_space_smoothed',
                             'mean_firing_rate_vr', 'n_trials', 'track_length',
                             'timestamp', 'date', 'mouse', 'recording_day',
                             'Lomb_classifier_', 'ML_Freqs', 'ML_SNRs',
                             'MOVING_LOMB_all_centre_trials', 'MOVING_LOMB_all_powers', 'MOVING_LOMB_avg_power',
                             'MOVING_LOMB_freqs',
                             'rolling:block_lengths', 'rolling:block_lengths_shuffled',
                             'rolling:distance_mean_firing_rate',
                             'rolling:distance_spatial_information_scores',
                             'rolling:distance_spatial_information_scores_Isec',
                             'rolling:position_mean_firing_rate',
                             'rolling:position_spatial_information_scores',
                             'rolling:position_spatial_information_scores_Isec',
                             'rolling:proportion_encoding_distance',
                             'rolling:proportion_encoding_null',
                             'rolling:proportion_encoding_position',
                             'rolling:rolling_centre_trials',
                             'rolling:rolling_classifiers',
                             'rolling:grid_code_global',
                             'rolling_threshold',
                             'rolling:classifier_by_trial_number',
                             'behaviour_hit_try_miss',
                             'behaviour_trial_numbers',
                             'behaviour_trial_types',
                             'agreement_between_cell_and_global',
                             'agreement_between_cell_and_grid_global',
                             'field_locations',
                             'field_trial_numbers',
                             'stop_locations',
                             'stop_trial_numbers',
                             'power_threshold',
                             'spatial_information_during_D',
                             'spatial_information_during_P',
                             'speeds_binned_in_space_smoothed',
                             'rolling:position_correlation_by_trial_number_t2tmethod']
    combined_df = combined_df[collumn_names_to_keep]
    combined_df.to_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/combined_cohorts.pkl")
    # TODO DELETE ONCE DONE ONCE #######
    """

    # remove artefacts, low firing rates in the open field, sessions with non 200cm track lengths and sessions with less than 10 trials
    combined_df = combined_df[combined_df["snippet_peak_to_trough"] < 500] # uV
    combined_df = combined_df[combined_df["mean_firing_rate_of"] > 0.2] # Hz
    combined_df = combined_df[combined_df["track_length"] == 200]
    combined_df = combined_df[combined_df["n_trials"] >= 10]
    combined_df = add_lomb_classifier(combined_df,suffix="")
    combined_df = add_peak_width(combined_df)
    combined_df = combined_df[combined_df["Lomb_classifier_"] != "Unclassified"] # removes cells without firing in the virtual reality track

    # remove mice without any grid cells
    combined_df = combined_df[combined_df["mouse"] != "M2"]
    combined_df = combined_df[combined_df["mouse"] != "M4"]
    combined_df = combined_df[combined_df["mouse"] != "M15"]
    combined_df = add_alternative_classification(combined_df, column_name="rolling:position_correlation_by_trial_number_t2tmethod", threshold=0.5)

    # print n numbers for dataset
    grid_cells = combined_df[combined_df["classifier"] == "G"]
    non_grid_cells = combined_df[combined_df["classifier"] != "G"]

    print("There is ", len(np.unique(combined_df["mouse"])), " mice in the dataset")
    print("There is ", len(combined_df), " clusters in the dataset")
    print("There is ", len(np.unique(combined_df["session_id"])), " sessions in this dataset")
    print("There is ", len(grid_cells), " grid clusters in the dataset")
    print("There is ", len(np.unique(grid_cells["session_id"])), " sessions in this dataset with grid cells")

    # make dataframes for R Figure 2, 3, 4
    collumn_names_to_keep = ["cluster_id", "session_id", "mouse", "recording_day", "classifier", "n_trials", "Lomb_classifier_",
                             "ML_Freqs", "ML_SNRs", "mean_firing_rate_vr","number_of_spikes", "grid_score", "rate_map_correlation_first_vs_second_half",
                             "percent_excluded_bins_rate_map_correlation_first_vs_second_half_p","grid_cell",
                             "rolling:position_spatial_information_scores_Isec", "rolling:distance_spatial_information_scores_Isec",
                             "rolling:proportion_encoding_distance","rolling:proportion_encoding_null", "rolling:proportion_encoding_position",
                             "spatial_information_during_D","spatial_information_during_P", "spatial_information_score_Isec_vr",
                             "spatial_information_score_Ispike_vr", "ML_peak_width", "rolling:position_mean_firing_rate", "rolling:distance_mean_firing_rate"]
    combined_df_with_singular_values = combined_df[collumn_names_to_keep]
    combined_df_with_singular_values.to_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/cells_df_singular_values.pkl")

    agreement_dataframe = generate_agreement_dataframe(combined_df)
    avg_agreement_dataframe = compile_shuffled_agreements(agreement_dataframe)
    agreement_dataframe.to_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/agreement_df.pkl")
    avg_agreement_dataframe.to_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/compiled_agreement_df.pkl")

    # make dataframes for R (Figure 6 and Figure 6, Figure Supplement 2
    stable_grid_cells_sessions = grid_cells[((grid_cells["rolling:proportion_encoding_position"] >= 0.85)
                                            |(grid_cells["rolling:proportion_encoding_distance"] >= 0.85))]
    stable_grid_cells_sessions = drop_duplicate_sessions(stable_grid_cells_sessions)
    cells_for_R_longform_by_session = generate_long_form_for_R_by_session(stable_grid_cells_sessions)
    cells_for_R_longform = generate_long_form_for_R(combined_df)
    cells_for_R_longform_by_session.to_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/all_cells_for_R_longform_by_session.pkl")
    cells_for_R_longform.to_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/all_cells_for_R_longform.pkl")
    save_path = "/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/plots"

    fig_size = (3.5, 6)
    # Figure 2D-F
    plot_lomb_classifiers_vs_shuffle(combined_df, suffix="", save_path=save_path)
    plot_lomb_classifiers_proportions(combined_df, suffix="", save_path=save_path)
    plot_lomb_classifier_powers_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size)
    plot_lomb_classifier_mfr_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size)
    plot_lomb_classifier_spatinfo_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size, score="spatial_information_score_Isec_vr")
    plot_lomb_classifier_spatinfo_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size, score="spatial_information_score_Ispike_vr")
    plot_lomb_classifier_peak_width_vs_groups(combined_df, suffix="", save_path=save_path, fig_size=fig_size)

    # Figure 3F-G
    plot_proportion_of_session_encoding_mode(combined_df[combined_df["classifier"] == "G"], save_path=save_path, suffix="_grid_cells")
    plot_spatial_information_on_position_and_distance_trials(combined_df, save_path=save_path, fig_size=(3.5,6))
    plot_mean_firing_rates_on_position_and_distance_trials(combined_df, save_path=save_path, fig_size=(3.5,6))

    # Figure 3, Figure Supplement 4
    stable_cells_sessions = combined_df[((combined_df["rolling:proportion_encoding_position"] >= 0.85) |
                                         (combined_df["rolling:proportion_encoding_distance"] >= 0.85) |
                                         (combined_df["rolling:proportion_encoding_null"] >= 0.85))]
    plot_lomb_classifier_powers_vs_groups(stable_cells_sessions, suffix="stable_sessions", save_path=save_path, fig_size=fig_size)
    plot_lomb_classifier_mfr_vs_groups(stable_cells_sessions, suffix="stable_sessions", save_path=save_path, fig_size=fig_size)
    plot_lomb_classifier_spatinfo_vs_groups(stable_cells_sessions, suffix="stable_sessions", save_path=save_path, fig_size=fig_size, score="spatial_information_score_Isec_vr")
    plot_lomb_classifier_spatinfo_vs_groups(stable_cells_sessions, suffix="stable_sessions", save_path=save_path, fig_size=fig_size, score="spatial_information_score_Ispike_vr")
    plot_lomb_classifier_peak_width_vs_groups(stable_cells_sessions, suffix="stable_sessions", save_path=save_path, fig_size=fig_size)

    # Figure 3, Figure Supplement 5
    plot_proportion_of_session_encoding_mode(combined_df[combined_df["classifier"] == "G"], save_path=save_path, suffix="_grid_cells") # duplicated from Figure 3F
    plot_proportion_of_session_encoding_mode(combined_df[combined_df["classifier"] != "G"], save_path=save_path, suffix="_nongrid_cells")
    plot_rolling_lomb_block_sizes(combined_df, save_path=save_path, suffix="")

    # Figure 3, Figure Supplement 6
    plot_rolling_lomb_block_lengths_vs_shuffled(combined_df, save_path=save_path)

    # Figure 3, Figure Supplement 7
    plot_trial_type_ratio_against_task_anchored_and_independent_ratio(grid_cells, save_path=save_path)

    # Figure 4E-F
    #agreement_dataframe = pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/agreement_df.pkl")
    plot_agreement_between_grid_and_non_grids(avg_agreement_dataframe, save_path=save_path)
    plot_spatial_information_during_dominant_PandD_grid_modes(combined_df, save_path=save_path)

    # Figure 5A and Figure 5, Figure Supplement 2
    all_behaviour = pd.read_pickle("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/all_behaviour_200cm.pkl") # load behaviour-only dataframe
    all_behaviour = all_behaviour[np.isin(all_behaviour["session_id_vr"], combined_df["session_id_vr"])]
    plot_avg_speed_profiles(all_behaviour, save_path="/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/plots")

    # Figure 6A-F
    plot_stop_histogram_for_stable_cells_averaged_over_cells(combined_df[combined_df["classifier"] == "G"], save_path=save_path, suffix="grid_cells") # B
    plot_percentage_hits_for_stable_encoding_cells_averaged_over_cells(combined_df[combined_df["classifier"] == "G"], save_path=save_path, suffix="grid_cells") # A
    plot_stop_histogram_for_remapped_encoding_cells_averaged_over_cells(combined_df[combined_df["classifier"] == "G"], save_path=save_path, trial_classification_column_name="rolling:classifier_by_trial_number", suffix="grid_cells") # C
    plot_percentage_hits_for_remapped_encoding_cells_averaged_over_cells(combined_df[combined_df["classifier"] == "G"], save_path=save_path, trial_classification_column_name="rolling:classifier_by_trial_number", suffix="grid_cells") # D
    plot_percentage_encoding_by_trial_category_each_mouse_weighted(combined_df[combined_df["classifier"] == "G"], save_path=save_path, trial_classification_column_name="rolling:classifier_by_trial_number", suffix="grid_cells") # E

    odds_ratio_data = pd.read_csv("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/odds_ratio_glmer_model.csv")
    plot_odds_ratio(odds_ratio_data, save_path="/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/plots", suffix="") # F

    # Figure 6 and Figure 6 Supplemental 1
    plot_speed_profiles_between_p_and_d_groups(combined_df[combined_df["classifier"] == "G"], suffix="", modes=["all"], save_path=save_path)
    plot_speed_profiles_between_p_and_d_groups(combined_df[combined_df["classifier"] == "G"], suffix="grid_cells_P_and_D", modes=["P", "D"], save_path=save_path)

    # Figure 6 and Figure 6 Supplemental 2B, E-H
    plot_histogram_of_position_template_correlations(grid_cells, save_path=save_path) # B

    plot_stop_histogram_for_remapped_encoding_cells_averaged_over_cells(combined_df[combined_df["classifier"] == "G"], save_path=save_path, trial_classification_column_name="alternative_classifications", suffix="grid_cells_altmethod") # E
    plot_percentage_hits_for_remapped_encoding_cells_averaged_over_cells(combined_df[combined_df["classifier"] == "G"], save_path=save_path, trial_classification_column_name="alternative_classifications",  suffix="grid_cells_altmethod") # F
    plot_percentage_encoding_by_trial_category_each_mouse_weighted(combined_df[combined_df["classifier"] == "G"], save_path=save_path, trial_classification_column_name="alternative_classifications",  suffix="grid_cells_altmethod") # G

    odds_ratio_data = pd.read_csv("/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/data/odds_ratio_glmer_model_alt.csv")
    plot_odds_ratio(odds_ratio_data, save_path="/mnt/datastore/Harry/Grid_anchoring_eLife_2023/real/plots", suffix="_alt") # H

    # distribution of grid fields as requested in response to reviewer # 3
    plot_field_distributions(combined_df, save_path=save_path)
    plot_field_distributions_stable_cells(combined_df, save_path=save_path)

    print("------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    main()