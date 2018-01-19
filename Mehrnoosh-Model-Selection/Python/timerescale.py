
# This script is psedu code for learning how to use time_rescaling library


import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix
from statsmodels.api import GLM, families
from time_rescale import TimeRescaling

def simulate_poisson_process(rate, sampling_frequency):
    return np.random.poisson(rate / sampling_frequency)

n_time, n_trials = 1500, 1000
SAMPLING_FREQUENCY = 1500
sampling_frequency = 1500;

# Firing rate starts at 5 Hz and switches to 10 Hz
firing_rate = np.ones((n_time, n_trials)) * 10
firing_rate[:n_time // 2, :] = 5
spike_train = simulate_poisson_process(
    firing_rate, sampling_frequency)
time = (np.arange(0, n_time)[:, np.newaxis] / sampling_frequency *
        np.ones((1, n_trials)))
trial_id = (np.arange(n_trials)[np.newaxis, :]
           * np.ones((n_time, 1)))

# Fit a spline model to the firing rate
design_matrix = dmatrix('bs(time, df=5)', dict(time=time.ravel()))
fit = GLM(spike_train.ravel(), design_matrix,
          family=families.Poisson()).fit()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].pcolormesh(np.unique(time), np.unique(trial_id),
                   spike_train.T, cmap='viridis')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Trials')
axes[0].set_title('Simulated Spikes')
conditional_intensity = fit.mu

axes[1].plot(np.unique(time), firing_rate[:, 0],
             linestyle='--', color='black',
             linewidth=4, label='True Rate')
axes[1].plot(time.ravel(), conditional_intensity * sampling_frequency,
             linewidth=4, label='model conditional intensity')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Firing Rate (Hz)')
axes[1].set_title('True Rate vs. Model')
plt.legend()

###############################################################################

conditional_intensity = fit.mu
rescaled = TimeRescaling(conditional_intensity,
                         spike_train.ravel(),
                         trial_id.ravel())

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
rescaled.plot_ks(ax=axes[0])
rescaled.plot_rescaled_ISI_autocorrelation(ax=axes[1])








