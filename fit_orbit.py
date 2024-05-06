import os.path
from orbitize import read_input, hipparcos, system, priors, sampler
import numpy as np

"""
Note: in Harper+ 2017 fit, I believe they assume the same cosmic jitter for the
Hipparcos data as Hipparcos does.

Note: using orbitize! branch: option-to-exclude-hipIAD-from-absastromlike

Fits to run:
1. "standard" fit-- all data, no jitter or error inflation. 
a. with planet: planetTrue_dvdFalse_renormHIPFalse_fitradioTrue_burn100_total5_000_000.hdf5 TODO: run 5x longer
b. no planet: planetFalse_dvdFalse_renormHIPFalse_fitradioTrue_burn100_total5000000.hdf5 TODO: run 5x longer

2. "Hipparcos only" fit -- no jitter or error inflation.
a. with planet: planetTrue_dvdFalse_renormHIPFalse_fitradioFalse_burn100_total5000000.hdf5 TODO: run 5x longer
b. no planet: planetFalse_dvdFalse_renormHIPFalse_fitradioFalse_burn100_total5000000.hdf5 TODO: run 5x longer

3. "radio only" -- no jitter or error inflation: no planet, use for 4
a. planetFalse_dvdFalse_renormHIPFalse_fitradioTrue_fithipparcosFalse_burn100_total5000000.hdf5 TODO: think through why PM is so broad

4. "Hipparcos only, radio PM" fit -- no jitter or error inflation. PM constrained by radio fit.
a. with planet: TODO: run later 
b. no planet: TODO: run later 

5. "no bad Hipparcos" -- remove first two Hipparcos points
a. with planet: RUNNING: planetTrue_dvdFalse_renormHIPFalse_fitradioTrue_fithipparcosTrue_nofirstIAD_burn100_total25000000
b. no planet: RUNNING: planetFalse_dvdFalse_renormHIPFalse_fitradioTrue_fithipparcosTrue_nofirstIAD_burn100_total25000000

Fits to show for comparison:
A. Hipparcos reproduction: plot shown in plots/betelgeuse_IADrefit_dvd.png
B. Harper+ 17 reproduction: plot shown in plots/radio_refit_2.4mas_planetFalse_dvdFalse_renormHIPTrue_burn100_total500000.png

"""

fit_planet = True  # if True, fit for planet parameters
radio_jit = (
    0  # 2.4  # [mas] Harper+ 17 fit adds in a jitter term to the radio positions
)
hip_dvd = False
normalizie_hip_errs = False
fit_radio = True
error_norm_factor = 0  # 1.2957671  # this is the number Graham scales by for the 2.4mas radio-only fit (private comm) [mas]
fit_hipparcos = True
no_bad_hipparcos = True

fit_name = "planet{}_dvd{}_renormHIP{}_fitradio{}_fithipparcos{}".format(
    fit_planet, hip_dvd, normalizie_hip_errs, fit_radio, fit_hipparcos
)

if no_bad_hipparcos:
    fit_name += "_nofirstIAD"

input_file = os.path.join("data/data.csv")
data_table = read_input.read_file(input_file)

data_table["quant1_err"] = error_norm_factor * np.sqrt(
    data_table["quant1_err"] ** 2 + radio_jit**2
)
data_table["quant2_err"] = error_norm_factor * np.sqrt(
    data_table["quant2_err"] ** 2 + radio_jit**2
)

if not fit_radio:
    # replace the radio astrometry with an empty data table
    data_table = data_table[:0]

num_secondary_bodies = 1  # number of planets/companions orbiting your primary
hip_num = "027989"  # Betelgeuse
if hip_dvd:
    IAD_file = "data/HIP{}_dvd.d".format(hip_num)
else:
    IAD_file = "data/H{}.d".format(hip_num)

# the angular size of Beetle is 55 mas, and the Hipparcos jitter is 0.15 mas, for reference

beetle_Hip = hipparcos.HipparcosLogProb(
    IAD_file, hip_num, num_secondary_bodies, renormalize_errors=normalizie_hip_errs
)

# remove the first two outlier Hipparcos points
if no_bad_hipparcos:
    good_mask = beetle_Hip.epochs > 1990.5
    beetle_Hip.epochs = beetle_Hip.epochs[good_mask]
    beetle_Hip.eps = beetle_Hip.eps[good_mask]
    beetle_Hip.cos_phi = beetle_Hip.cos_phi[good_mask]
    beetle_Hip.sin_phi = beetle_Hip.sin_phi[good_mask]

if not fit_hipparcos:
    beetle_Hip.include_hip_iad_in_likelihood = False

# we'll overwrite these in a sec
m0 = 1e-10  # median mass of primary [M_sol]
plx = 1e-10  # [mas]
mass_err = 1e-10
plx_err = 1e-10
#####

fit_secondary_mass = True  # tell orbitize! we want to get dynamical masses

beetle_system = system.System(
    num_secondary_bodies,
    data_table,
    m0,
    plx,
    hipparcos_IAD=beetle_Hip,
    fit_secondary_mass=fit_secondary_mass,
    mass_err=mass_err,
    plx_err=plx_err,
    fitting_basis="Period",
    restrict_angle_ranges=True,
)

if fit_radio:
    # make sure orbitize! knows to fit for the proper motion and parallax
    assert beetle_system.pm_plx_predictor is not None

    # make sure orbitize! correctly figures out which epochs are absolute astrometry
    assert len(beetle_system.stellar_astrom_epochs) == 18

if not fit_hipparcos:
    assert not beetle_system.hipparcos_IAD.include_hip_iad_in_likelihood

"""
Change priors
"""

# set uniform parallax prior
plx_index = beetle_system.param_idx["plx"]
beetle_system.sys_priors[plx_index] = priors.UniformPrior(-10, 15)


m0_index = beetle_system.param_idx["m0"]
m1_index = beetle_system.param_idx["m1"]
p1_index = beetle_system.param_idx["per1"]
e1_index = beetle_system.param_idx["ecc1"]
pan1_index = beetle_system.param_idx["pan1"]
aop1_index = beetle_system.param_idx["aop1"]
inc1_index = beetle_system.param_idx["inc1"]
tau1_index = beetle_system.param_idx["tau1"]
alpha0_index = beetle_system.param_idx["alpha0"]
delta0_index = beetle_system.param_idx["delta0"]


if fit_planet:
    # set uniform primary mass prior
    beetle_system.sys_priors[m0_index] = priors.UniformPrior(
        10, 25
    )  # [Msun] big range, lots of unc for Betelgeuse

    # set log-uniform secondary mass prior
    beetle_system.sys_priors[m1_index] = priors.LogUniformPrior(0.1, 10)  # [Msun]

    # set period prior
    beetle_system.sys_priors[p1_index] = priors.UniformPrior(
        1000 / 365.25, 2700 / 365.25
    )  # [yr]

    # fix e=0 and aop (undefined for circular orbit)
    beetle_system.sys_priors[e1_index] = 0
    beetle_system.sys_priors[aop1_index] = 0


else:
    # set the planet parameters so that the planet signal is 0 (i.e. only model the astrometry)
    beetle_system.sys_priors[m0_index] = 10
    beetle_system.sys_priors[m1_index] = 0  # test particle secondary
    beetle_system.sys_priors[p1_index] = 1
    beetle_system.sys_priors[e1_index] = 0
    beetle_system.sys_priors[pan1_index] = 0
    beetle_system.sys_priors[aop1_index] = 0
    beetle_system.sys_priors[inc1_index] = 0
    beetle_system.sys_priors[tau1_index] = 0

    # increase prior limits (useful for radio-only fits)
    beetle_system.sys_priors[alpha0_index].minval = -20
    beetle_system.sys_priors[delta0_index].maxval = 20
    beetle_system.sys_priors[alpha0_index].maxval = 20
    beetle_system.sys_priors[delta0_index].minval = -20


# print out the priors to make sure everything looks fine
print(list(zip(beetle_system.labels, beetle_system.sys_priors)))

"""
Run MCMC
"""

if __name__ == "__main__":
    num_threads = 20
    num_temps = 20
    num_walkers = 1000
    n_steps_per_walker = 25_000
    num_steps = num_walkers * n_steps_per_walker
    burn_steps = 100
    thin = 100

    beetle_sampler = sampler.MCMC(
        beetle_system,
        num_threads=num_threads,
        num_temps=num_temps,
        num_walkers=num_walkers,
    )

    assert not beetle_sampler.system.hipparcos_IAD.renormalize_errors

    beetle_sampler.run_sampler(num_steps, burn_steps=burn_steps, thin=thin)
    beetle_sampler.results.save_results(
        "results/{}_burn{}_total{}.hdf5".format(fit_name, burn_steps, num_steps)
    )
