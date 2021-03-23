import numpy as np
import matplotlib.pyplot as plt
import Python_PostSorting.ExtractFiringData
import os


"""

## the following code calculates the mean spike width for each cluster

1. extract snippets
2. calculate mean snippet
3. calculate half width of mean snippet
4. insert into dataframe

"""

def plot_snippet_method(mean_snippet,snippet_height,half_height, intercept_line):
    plt.plot(mean_snippet)
    plt.plot(snippet_height, 'o', color='r', markersize=5)
    plt.plot(half_height, 'o', color='b', markersize=5)
    plt.plot(intercept_line, '-', color='r', markersize=5)
    plt.close()
    return


def extract_mean_spike_width_for_channel(spike_data, cluster, channel):
    mean_snippet = np.mean(spike_data.random_snippets[cluster][channel, :, :], 1) * -1
    snippet_height = np.max(mean_snippet) - np.min(mean_snippet)
    half_height = snippet_height/2
    intercept_line = np.repeat(half_height/2, mean_snippet.shape[0])
    intercept=find_intercept(mean_snippet, intercept_line)
    try:
        width=intercept[1]-intercept[0]
    except IndexError:
        width=0
    plot_snippet_method(mean_snippet,snippet_height,half_height, intercept_line)
    return width


def find_intercept(mean_snippet, intercept_line):
    idx = np.argwhere(np.diff(np.sign(mean_snippet - intercept_line))).flatten()
    return idx


def calculate_spike_width(spike_data):
    print("calculating spike half width...")
    for cluster in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster] - 1
        #max_channel = spike_data.primary_channel[cluster]
        spike_width_on_channels=np.zeros(4)
        for channel in range(4):
            width= extract_mean_spike_width_for_channel(spike_data, cluster, channel)
            spike_width_on_channels[channel] = width

        spike_data.at[cluster, "spike_width"] = max(spike_width_on_channels)
    return spike_data
