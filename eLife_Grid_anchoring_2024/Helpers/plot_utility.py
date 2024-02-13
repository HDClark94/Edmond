import matplotlib.pylab as plt
import math
import numpy as np
import random
from astropy.nddata import block_reduce

def draw_reward_zone():
    for stripe in range(8):
        if stripe % 2 == 0:
            plt.axvline(91.25+stripe*2.5, color='limegreen', linewidth=5.5, alpha=0.4, zorder=0)
        else:
            plt.axvline(91.25+stripe*2.5, color='k', linewidth=5.5, alpha=0.4, zorder=0)

def draw_black_boxes():
    plt.axvline(15, color='k', linewidth=66, alpha=0.25, zorder=0)
    plt.axvline(185, color='k', linewidth=66, alpha=0.25, zorder=0)

def style_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return plt, ax

def style_open_field_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off
    ax.set_aspect('equal')
    return ax

def style_polar_plot(ax):
    ax.spines['polar'].set_visible(False)
    ax.set_yticklabels([])  # remove yticklabels
    # ax.grid(None)
    plt.xticks([math.radians(0), math.radians(90), math.radians(180), math.radians(270)])
    ax.axvline(math.radians(90), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(180), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(270), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(0), color='black', linewidth=1, alpha=0.6)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)
    ax.xaxis.set_tick_params(labelsize=25)
    return ax

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def style_vr_plot_offset(ax, x_max):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=True,
        labelleft=True,
        labelbottom=True)  # labels along the bottom edge are off
    plt.ylim(0, x_max)
    return ax

def style_track_plot(ax, track_length,alpha=0.25):
    ax.axvspan(track_length-60-30-20, track_length-60-30, facecolor='DarkGreen', alpha=alpha, linewidth =0)
    ax.axvspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axvspan(track_length-30, track_length, facecolor='k', linewidth =0, alpha=alpha)# black box

def makelegend(fig,ax, x_location):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="upper right", bbox_to_anchor=(0.989, x_location), fontsize = "xx-large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def adjust_spine_thickness(ax):
    for axis in ['left','bottom']:
        ax.spines[axis].set_linewidth(1)

def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0)) # outward by 10 points
            #spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def get_weights_normalized_hist(array_in):
    weights = np.ones_like(array_in) / float(len(array_in))
    return weights

def style_vr_plot(ax, x_max=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=True,
        labelleft=True,
        labelbottom=True)  # labels along the bottom edge are off
    if x_max is not None:
        plt.ylim(0, x_max)
    return ax

def get_vmin_vmax(cluster_firing_maps, bin_cm=8):
    cluster_firing_maps_reduced = []
    for i in range(len(cluster_firing_maps)):
        cluster_firing_maps_reduced.append(block_reduce(cluster_firing_maps[i], bin_cm, func=np.mean))
    cluster_firing_maps_reduced = np.array(cluster_firing_maps_reduced)
    vmin= 0
    vmax= np.max(cluster_firing_maps_reduced)
    return vmin, vmax
