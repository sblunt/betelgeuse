import orbitize.results
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# choose which fit to inspect
fit_planet = False
dvd = False 
burn_steps = 100
total_steps = 500000 
renorm_hip = True


run_name = "planet{}_dvd{}_renormHIP{}_burn{}_total{}".format(
    fit_planet, dvd, renorm_hip, burn_steps, total_steps
)


# read in results
beetle_results = (
    orbitize.results.Results()
)  # create a blank results object to load the data
beetle_results.load_results("results/{}.hdf5".format(run_name))

harper17_solution = {  # table 3, 2.4 mas radio noise
    "alpha0": [0.09, 0.19],
    "delta0": [1.4, 0.48],
    "plx": [3.77, 2.20],
    "pm_ra": [25.53, 0.31],
    "pm_dec": [9.37, 0.28],
}

fig, ax = plt.subplots(5, 1, figsize=(5, 11))
for i, a in enumerate(ax):
    idx = 6 + i
    param = beetle_results.labels[idx]

    if param == 'pm_ra':
        a.hist(
            beetle_results.post[:, idx] / np.cos(np.radians(7.40703653)),
            bins=50,
            color="grey",
            density=True,
            label="orbitize! refit",
        )
    else:
        a.hist(
            beetle_results.post[:, idx],
            bins=50,
            color="grey",
            density=True,
            label="orbitize! refit",
        )
    

    a.set_xlabel(param)
    if param not in ["alpha0", "delta0"]:
        xs = np.linspace(
            harper17_solution[param][0] - 5 * harper17_solution[param][1],
            harper17_solution[param][0] + 5 * harper17_solution[param][1],
        )
        a.plot(
            xs,
            norm(harper17_solution[param][0], harper17_solution[param][1]).pdf(xs),
            color="k",
            label="Harper+ 17 table 4, $\sigma_{{radio}}$ = 2.4mas",
        )

ax[3].set_xlabel("$\Delta\\alpha$cos($\delta$) [mas]")
ax[4].set_xlabel("$\Delta\\delta$ [mas]")


# # plx vs pm_ra
# ax[3].hist2d(beetle_results.post[:, 6], beetle_results.post[:, 7], bins=30)
# ax[3].set_xlabel(beetle_results.labels[6])
# ax[3].set_ylabel(beetle_results.labels[7])

# # plx vs pm_dec
# ax[4].hist2d(beetle_results.post[:, 6], beetle_results.post[:, 8], bins=30)
# ax[4].set_xlabel(beetle_results.labels[6])
# ax[4].set_ylabel(beetle_results.labels[8])

ax[0].legend()

plt.tight_layout()
plt.savefig("plots/{}.png".format(run_name))
