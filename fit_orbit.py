import os.path
import orbitize
from orbitize import read_input, hipparcos, gaia, system, priors, sampler

input_file = os.path.join("data/data.csv")
data_table = read_input.read_file(input_file)

num_secondary_bodies = 1  # number of planets/companions orbiting your primary
hip_num = "027989"  # Betelgeuse
IAD_file = "data/H{}.d".format(hip_num)

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

"""
Change priors
"""

# set uniform parallax prior
plx_index = beetle_system.param_idx["plx"]
beetle_system.sys_priors[plx_index] = priors.UniformPrior(7.6 - 5.0, 7.6 + 5.0)

# set uniform primary mass prior
m0_index = beetle_system.param_idx["m0"]
beetle_system.sys_priors[m0_index] = priors.UniformPrior(
    15, 20
)  # big range, lots of unc for Betelgeuse

# set log-uniform secondary mass prior
m1_index = beetle_system.param_idx["m1"]
beetle_system.sys_priors[m1_index] = priors.LogUniformPrior(0.1, 10)

# set period prior between 3 and 10 years
p1_index = beetle_system.param_idx["per1"]
beetle_system.sys_priors[p1_index] = priors.UniformPrior(3, 10)


"""
Run MCMC
"""
num_threads = 50
num_temps = 20
num_walkers = 1000
num_steps = 10000000  # n_walkers x n_steps_per_walker
burn_steps = 10000
thin = 100

beetle_sampler = sampler.MCMC(
    beetle_system,
    num_threads=num_threads,
    num_temps=num_temps,
    num_walkers=num_walkers,
)
# beetle_sampler.run_sampler(num_steps, burn_steps=burn_steps, thin=thin)
