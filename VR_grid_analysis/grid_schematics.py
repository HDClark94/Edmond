import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
plt.rc('axes', linewidth=3)

def plot_power_hypotheses_mfr(output_path):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xs = np.linspace(-np.pi, 4*np.pi, 10000)
    ax.plot(xs, np.sin(2*xs), label="Trial 1", linewidth=2, color="black")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=-1, top=1.05)
    ax.set_xlim(left=0, right=10)
    fig.tight_layout()
    ax.set_ylabel('FR (Hz)', fontsize=25, labelpad=8)
    ax.set_xlabel('Position', fontsize=25, labelpad=8)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(output_path + '/grid_cell_jitter_mfr.png', dpi=200)
    plt.close()

def plot_power_hypotheses(jitter, output_path):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    colors = cm.rainbow(np.linspace(0, 1, 8))
    xs = np.linspace(-np.pi, 4*np.pi, 10000)
    ax.plot(xs, np.sin(2*xs), label="Trial 1", linewidth=2, color=colors[0])
    ax.plot(xs+jitter, np.sin(2*xs), label="Trial 2", linewidth=2, color=colors[1])
    ax.plot(xs-jitter, np.sin(2*xs), label="Trial 3", linewidth=2, color=colors[2])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=-1, top=1.05)
    ax.set_xlim(left=0, right=10)
    fig.tight_layout()
    ax.set_ylabel('FR (Hz)', fontsize=25, labelpad=8)
    ax.set_xlabel('Position', fontsize=25, labelpad=8)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(output_path + '/grid_cell_jitter'+str(jitter)+'.png', dpi=200)
    plt.close()

def plot_zero_inflation_hypotheses(jitter, output_path):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    colors = cm.rainbow(np.linspace(0, 1, 8))
    xs = np.linspace(-np.pi, 4*np.pi, 10000)
    y1 = np.sin(2*xs)
    y2 = np.sin(2*xs); y2[3500:5550] = -1
    y3 = np.sin(2*xs); y3[5550:7500] = -1
    ax.plot(xs+jitter, y1, label="Trial 1", linewidth=2, color=colors[0])
    ax.plot(xs, y2, label="Trial 2", linewidth=2, color=colors[1])
    ax.plot(xs-jitter, y3, label="Trial 3", linewidth=2, color=colors[2])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(bottom=-1, top=1.05)
    ax.set_xlim(left=0, right=10)
    fig.tight_layout()
    ax.set_ylabel('FR (Hz)', fontsize=25, labelpad=8)
    ax.set_xlabel('Position', fontsize=25, labelpad=8)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(output_path + '/grid_cell_skipped_fields.png', dpi=200)
    plt.close()

def main():
    print('-------------------------------------------------------------')
    output_path = "/mnt/datastore/Harry/VR_grid_cells/paper_figures/grid_schematics"
    plot_power_hypotheses_mfr(output_path=output_path)
    plot_power_hypotheses(jitter=0.1, output_path=output_path)
    plot_power_hypotheses(jitter=0.4, output_path=output_path)
    plot_zero_inflation_hypotheses(jitter=0.1, output_path=output_path)
    print("look now")


if __name__ == '__main__':
    main()
