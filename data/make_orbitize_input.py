import numpy as np
import pandas as pd
from pandas import DataFrame
from astropy.time import Time
from astropy.coordinates import SkyCoord, ICRS
from astropy import units as u
import matplotlib.pyplot as plt

"""
Convenience script to convert data from download format to orbitize format.

Hipparcos measurements (from 2007 van Leeuwen and more modern reductions)
are given in ICRS frame. All radio data are also in this frame. This script:
- subtracts the reported Hipparcos position from the radio positions. This is the
  format orbitize! expects. 

Note: this file uses the new (2014) Hipparcos reduction value to subtract, but the
derived values (except var) are all the same as the DVD values, so it doesn't matter.

# NOTE: The input is delta(RA)*cos(delta0). This is output from the astropy utility.
"""


table1 = pd.read_csv(
    "radio_pos_table1.csv",
    delim_whitespace=True,
    skiprows=5,
    engine="python",
    names=[
        "date",
        "ra_hr",
        "ra_min",
        "ra_sec",
        "raoff_err",
        "dec_deg",
        "dec_arcmin",
        "dec_arcsec",
        "decoff_err",
    ],
    usecols=range(9),
    on_bad_lines="skip",
)
table1 = table1[:-2]

table2 = pd.read_csv(
    "radio_pos_table2.csv",
    delim_whitespace=True,
    skiprows=6,
    engine="python",
    names=[
        "date",
        "ra_hr",
        "ra_min",
        "ra_sec",
        "dec_deg",
        "dec_arcmin",
        "dec_arcsec",
        "error",
    ],
    usecols=range(8),
)

table2["raoff_err"] = table2["error"]
table2["decoff_err"] = table2["error"]

df = pd.concat([table1, table2], ignore_index=True)

n_obs = len(df)

df_orbitize = DataFrame(
    Time(df["date"].astype("float"), format="decimalyear").mjd, columns=["epoch"]
)
df_orbitize["object"] = np.zeros_like(df_orbitize["epoch"]).astype(int)
df_orbitize["raoff"] = np.ones_like(df_orbitize["epoch"])
df_orbitize["decoff"] = np.ones_like(df_orbitize["epoch"])

# read the Hipparcos best-fit solution from the IAD DVD file
astrometric_solution = pd.read_csv("H027989.d", skiprows=9, sep="\s+", nrows=1)
hipparcos_alpha0 = astrometric_solution["RAdeg"].values[0]  # [deg]
hipparcos_delta0 = astrometric_solution["DEdeg"].values[0]  # [deg]
hipparcos_coord = SkyCoord(
    hipparcos_alpha0, hipparcos_delta0, unit=(u.deg, u.deg), frame=ICRS
)

for i in range(n_obs):
    coord = SkyCoord(
        "{} {} {} {} {} {}".format(
            df["ra_hr"][i],
            df["ra_min"][i],
            df["ra_sec"][i],
            df["dec_deg"][i],
            df["dec_arcmin"][i],
            df["dec_arcsec"][i],
        ),
        unit=(u.hourangle, u.deg),
        frame=ICRS,
    )

    # take difference between reported Hipparcos position and convert to mas
    raoff, deoff = hipparcos_coord.spherical_offsets_to(coord)
    df_orbitize["raoff"][i] = raoff.to(u.mas).value
    df_orbitize["decoff"][i] = deoff.to(u.mas).value

df_orbitize["decoff_err"] = df["decoff_err"].astype(float)
df_orbitize["raoff_err"] = df["raoff_err"].astype(float)

# save
df_orbitize.to_csv("data.csv", index=False)

# make a quick plot of the positions (compare to Fig 1 of Harper+ 2017)
plt.figure()
plt.errorbar(
    df_orbitize["raoff"],
    df_orbitize["decoff"],
    df_orbitize["raoff_err"],
    df_orbitize["decoff_err"],
    ls="",
    color="rebeccapurple",
)

plt.xlabel("$\Delta\\alpha$cos($\delta$) [mas]")
plt.ylabel("$\Delta\delta$ [mas]")
plt.savefig("../plots/radio_data.png", dpi=250)
