import numpy as np

allocentric_color = '#{:02x}{:02x}{:02x}{:02x}'.format(109, 47, 255, 248) # a.k.a task-anchored
egocentric_color = '#{:02x}{:02x}{:02x}{:02x}'.format(0, 220, 184, 255) # a.k.a task-independent
null_color = '#{:02x}{:02x}{:02x}{:02x}'.format(128, 128, 128, 255) # a.k.a aperiodic
alt_egocentric_color = '#{:02x}{:02x}{:02x}{:02x}'.format(109, 217, 255, 255)
rate_map_cmap = "viridis"

# Periodgram settings
frequency_step = 0.02
frequency = np.arange(0.1, 5+frequency_step, frequency_step)
window_length_in_laps = 3
rolling_window_size_for_lomb_classifier = 200
lomb_frequency_threshold = 0.05
power_estimate_step = 10
minimum_peak_distance = 20
rate_map_extra_smooth_gauss_kernel_std = 4
rate_map_gauss_kernel_std = 2

# Hit Miss Try # a.k.a hit, run, try
track_speed_threshold = 10

# Default simulated data parameters
sim_cmps = 10
sim_n_trials=100
sim_bin_size_cm=1
sim_sampling_rate=1000
sim_avg_speed_cmps=10
sim_p_scalar=0.1
sim_track_length=200
sim_field_spacing=90
sim_gauss_kernel_std=2
sim_step=0.000001
sim_field_noise_std=5
sim_switch_code_prob=0.05