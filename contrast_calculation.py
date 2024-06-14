import numpy as np
import astropy.units as u, astropy.constants as cst
from sbpy.units import VEGAmag, spectral_density_vega
from sbpy.photometry import bandpass

V = bandpass("Johnson V")

# from Simbad
beetle_mags = {"V": 0.42}

# total flux of betelgeuse (unresolved)
vmag_flux = (beetle_mags["V"] * VEGAmag).to("erg/(cm2 s AA)", spectral_density_vega(V))

print(vmag_flux)

# resolution elements of different instruments
beetle_ang_diam = 0.055 * u.arcsec
beetle_ang_area = np.pi * (beetle_ang_diam / 2) ** 2
chara_res_elem = 200 * u.uas
sphere_res_elem = 0.02 * u.arcsec

# calculate number of CHARA resolution elements across betelgeuse
num_res_elem_chara = beetle_ang_area / (np.pi * (chara_res_elem) ** 2)
print(num_res_elem_chara.to(""))
num_res_elem_sphere = beetle_ang_area / (np.pi * (sphere_res_elem) ** 2)
print(num_res_elem_sphere.to(""))


# total flux of companion per resolution element
# total flux of companion
