from datetime import datetime
import matplotlib.pyplot as plt
from orbitize.hipparcos import nielsen_iad_refitting_test

"""

TODO: see footnote 8 of van Leeuwen+ 2017 & Update documentation here.
"""

plt.rcParams["font.family"] = "stixgeneral"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 11
plt.rcParams["figure.facecolor"] = "white"


use_dvd = False
if use_dvd:
    saveplot_append = "_dvd"
else:
    saveplot_append = ""

hip_num = "027989"  # Betelgeuse

# Name/path for the plot this function will make
saveplot = "plots/betelgeuse_IADrefit{}.pdf".format(saveplot_append)

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
