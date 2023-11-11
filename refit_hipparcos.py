from datetime import datetime

from orbitize.hipparcos import nielsen_iad_refitting_test

hip_num = "027989"  # Betelgeuse

# Name/path for the plot this function will make
saveplot = "plots/betelgeuse_IADrefit.png"

# Location of the Hipparcos IAD file.
IAD_file = "/data/user/sblunt/HipIAD-2021/ResRec_JavaTool_2014/H{}/H{}.d".format(
    hip_num[0:3], hip_num
)

# These `emcee` settings are sufficient for the 5-parameter fits we're about to run,
#   although I'd probably run it for 5,000-10,000 steps if I wanted to publish it.
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
