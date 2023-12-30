import orbitize.results
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# choose which fit to inspect
fit_planet = False
dvd = True
burn_steps = 50
total_steps = 100_000
renorm_hip = True

run_name = "planet{}_dvd{}_renormHIP{}_burn{}_total{}".format(
    fit_planet, dvd, renorm_hip, burn_steps, total_steps
)


# read in results
beetle_results = (
    orbitize.results.Results()
)  # create a blank results object to load the data
beetle_results.load_results("results/{}.hdf5".format(run_name))

harper17_solution = {
    "alpha0": [0.09, 0.19],
    "delta0": [1.4, 0.48],
    "plx": [4.51, 0.8],
    "pm_ra": [26.42, 0.25],
    "pm_dec": [9.6, 0.12],
}

# harper17_solution = {  # table 3, 2.4 mas radio noise
#     "alpha0": [0.09, 0.19],
#     "delta0": [1.4, 0.48],
#     "plx": [3.77, 2.20],
#     "pm_ra": [25.53, 0.31],
#     "pm_dec": [9.37, 0.28],
# }

fig, ax = plt.subplots(5, 1, figsize=(8, 11))
for i, a in enumerate(ax):
    idx = 6 + i
    a.hist(
        beetle_results.post[:, idx],
        bins=50,
        color="grey",
        density=True,
        label="orbitize! refit",
    )
    param = beetle_results.labels[idx]
    a.set_xlabel(param)
    xs = np.linspace(
        harper17_solution[param][0] - 5 * harper17_solution[param][1],
        harper17_solution[param][0] + 5 * harper17_solution[param][1],
    )
    a.plot(
        xs,
        norm(harper17_solution[param][0], harper17_solution[param][1]).pdf(xs),
        color="k",
        label="Harper+ 17 solution",
    )
ax[0].legend()

plt.tight_layout()
plt.savefig("plots/{}.png".format(run_name))
