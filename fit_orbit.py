import os.path
from orbitize import read_input, hipparcos, system, priors, sampler


"""
Set some options for the fit
"""

fit_planet = False  # if True, fit for planet parameters

input_file = os.path.join("data/data.csv")
data_table = read_input.read_file(input_file)

num_secondary_bodies = 1  # number of planets/companions orbiting your primary
hip_num = "027989"  # Betelgeuse
IAD_file = "data/H{}.d".format(hip_num)

# the angular size of Beetle is 55 mas, and the Hipparcos jitter is 0.15 mas, for reference

beetle_Hip = hipparcos.HipparcosLogProb(IAD_file, hip_num, num_secondary_bodies)

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
)

# make sure orbitize! knows to fit for the proper motion and parallax
assert beetle_system.pm_plx_predictor is not None

# make sure orbitize! correctly figures out which epochs are absolute astrometry
assert len(beetle_system.stellar_astrom_epochs) == 18
"""
Change priors
"""

# set uniform parallax prior
plx_index = beetle_system.param_idx["plx"]
beetle_system.sys_priors[plx_index] = priors.UniformPrior(7.6 - 5.0, 7.6 + 5.0)


m0_index = beetle_system.param_idx["m0"]
m1_index = beetle_system.param_idx["m1"]
p1_index = beetle_system.param_idx["per1"]
e1_index = beetle_system.param_idx["ecc1"]
pan1_index = beetle_system.param_idx["pan1"]
aop1_index = beetle_system.param_idx["aop1"]
inc1_index = beetle_system.param_idx["inc1"]
tau1_index = beetle_system.param_idx["tau1"]


if fit_planet:
    # set uniform primary mass prior
    beetle_system.sys_priors[m0_index] = priors.UniformPrior(
        15, 20
    )  # big range, lots of unc for Betelgeuse

    # set log-uniform secondary mass prior
    beetle_system.sys_priors[m1_index] = priors.LogUniformPrior(0.1, 10)

    # set period prior between 3 and 10 years
    beetle_system.sys_priors[p1_index] = priors.UniformPrior(3, 10)

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

# print out the priors to make sure everything looks fine
print(list(zip(beetle_system.labels, beetle_system.sys_priors)))

"""
Run MCMC
"""
num_threads = 10  # 50
num_temps = 5  # 20
num_walkers = 10  # 1000
n_steps_per_walker = 1_000  # 10_000
num_steps = num_walkers * n_steps_per_walker
burn_steps = 1_000  # 10_000
thin = 1  # 100


beetle_sampler = sampler.MCMC(
    beetle_system,
    num_threads=num_threads,
    num_temps=num_temps,
    num_walkers=num_walkers,
)
beetle_sampler.run_sampler(num_steps, burn_steps=burn_steps, thin=thin)
