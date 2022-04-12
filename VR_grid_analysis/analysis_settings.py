#This file contains the parameters for analysis
import numpy as np
import matplotlib

allocentric_color = '#{:02x}{:02x}{:02x}{:02x}'.format(109, 47, 255, 248)
egocentric_color = '#{:02x}{:02x}{:02x}{:02x}'.format(0, 220, 184, 255)
null_color = '#{:02x}{:02x}{:02x}{:02x}'.format(128, 128, 128, 255)
rate_map_cmap = "BuPu"
#rate_map_cmap = cmap_reversed = matplotlib.cm.get_cmap('cudehelix_r')
rate_map_cmap = "viridis"

# Periodgram settings
frequency_step = 0.02
frequency = np.arange(0.1, 5+frequency_step, frequency_step)
window_length_in_laps = 3
rolling_window_size_for_lomb_classifier = 1000
measured_far = 0.023
power_estimate_step = 10
minimum_peak_distance = 20
rate_map_extra_smooth_gauss_kernel_std = 4
rate_map_gauss_kernel_std = 2


# Hit Miss Try
track_speed_threshold = 20
