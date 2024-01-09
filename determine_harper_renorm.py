"""
Section 3 of Harper+ 2017:
"These proper motions are presented in Table 3 where the uncertainties are 
calculated assuming the adopted astrometric model is the correct one."

I'm assuming this means that the radio errors were adjusted so that chi2_red=1
for the adopted solution (which uses both Hipparcos and radio data).

Ref: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Correcting_for_over-_or_under-dispersion
"""
from orbitize.hipparcos import PMPlx_Motion
from pandas import read_csv
from fit_orbit import beetle_Hip
import numpy as np

# table 4
# adopted_solution = {
#     "alpha0": 0.09,
#     "delta0": 1.4,
#     "plx": 4.51,
#     "pm_ra": 26.42,
#     "pm_dec": 9.6,
# }

# table 3, 2.4 mas jitter
adopted_solution = {
    "alpha0": 0,
    "delta0": 0,
    "plx": 3.33,
    "pm_ra": 25.77,
    "pm_dec": 9.55,
}

param_idx = {}
samples_array = np.empty(len(adopted_solution.keys()))
for i, param in enumerate(adopted_solution.keys()):
    param_idx[param] = i
    samples_array[i] = adopted_solution[param]

# read in radio data
data = read_csv("data/data.csv")

# compute the model predictions for this fit
myPlx_model = PMPlx_Motion(data.epoch.values, beetle_Hip.alpha0, beetle_Hip.delta0)
ra_pred, dec_pred = myPlx_model.compute_astrometric_model(samples_array, param_idx)

# compute chi2
chi2_ra = ((ra_pred - data.raoff) / data.raoff_err) ** 2
chi2_dec = ((dec_pred - data.decoff) / data.decoff_err) ** 2

chi2_total = np.sum(chi2_ra) + np.sum(chi2_dec)
n_data = 2 * len(data)  # 2 data points per epoch
chi2_reduced = chi2_total / (n_data - 1)

# renormalize errors
error_factor = np.sqrt(chi2_reduced)
print(error_factor)
