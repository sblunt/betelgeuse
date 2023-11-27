from datetime import datetime

from orbitize.hipparcos import nielsen_iad_refitting_test

"""
It wasn't immediately clear to me how the van Leeuwen+ team described stochastic fits.
I tried refitting by adding the var term in quadrature and subtracting it in quadrature,
but both of those looked incorrect for the DVD refit. Not incorporating the var
term into the refit in any way worked, so my interpretation is that the van 
Leeuwen team doesn't actually incorporate the var term into the fit; it's just
a "leftover" error term to indicate that the fit is not great.

TODO: I don't think this is right: see footnote 8 of van Leeuwen+ 2017. Update documentation here.
"""

use_dvd = True
if use_dvd:
    saveplot_append = "_dvd"
else:
    saveplot_append = ""

hip_num = "027989"  # Betelgeuse

# Name/path for the plot this function will make
saveplot = "plots/betelgeuse_IADrefit{}.png".format(saveplot_append)

# Location of the Hipparcos IAD file.
if use_dvd:
    IAD_file = "data/HIP027989_dvd.d"
else:
    IAD_file = "data/H027989.d"

burn_steps = 100
mcmc_steps = 5000

start = datetime.now()

# run the fit
nielsen_iad_refitting_test(
    IAD_file,
    hip_num=hip_num,
    saveplot=saveplot,
    burn_steps=burn_steps,
    mcmc_steps=mcmc_steps,
)

end = datetime.now()
duration_mins = (end - start).total_seconds() / 60

print("Done! This fit took {:.1f} mins on my machine.".format(duration_mins))
