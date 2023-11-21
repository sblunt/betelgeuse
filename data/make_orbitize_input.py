import numpy as np
import pandas as pd
from pandas import DataFrame
from astropy.time import Time
from astropy.coordinates import SkyCoord, FK5, ICRS
from astropy import units as u
import matplotlib.pyplot as plt

"""
Convenience script to convert data from download format to orbitize format.

Hipparcos measurements (from 2007 van Leeuwen and more modern reductions)
are given in ICRS frame. The second radio positions table is also given in ICRS, but
the first one is given in FK5 at equinox J2000. This script:
- converts the older radio data to ICRS, and
- subtracts the reported Hipparcos position from the radio positions. This is the
  format orbitize! expects.

Note: this file uses the new (2014) Hipparcos reduction to subtract, but the
values (except var) are all the same as the DVD values, so it doesn't matter.
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
n_obs1 = len(table1)
n_obs2 = len(table2)

df_orbitize = DataFrame(
    Time(df["date"].astype("float"), format="decimalyear").mjd, columns=["epoch"]
)
df_orbitize["object"] = np.zeros_like(df_orbitize["epoch"]).astype(int)
df_orbitize["raoff"] = np.ones_like(df_orbitize["epoch"])
df_orbitize["decoff"] = np.ones_like(df_orbitize["epoch"])

for i in range(n_obs1):
    coord = SkyCoord(
        "{} {} {} {} {} {}".format(
            table1["ra_hr"][i],
            table1["ra_min"][i],
            table1["ra_sec"][i],
            table1["dec_deg"][i],
            table1["dec_arcmin"][i],
            table1["dec_arcsec"][i],
        ),
        unit=(u.hourangle, u.deg),
        frame=FK5,
        equinox="J2000.0",
    )
    coord = coord.transform_to(ICRS)
    ra = coord.ra.deg
    dec = coord.dec.deg
    df_orbitize["raoff"][i] = float(ra)  # [deg]
    df_orbitize["decoff"][i] = float(dec)  # [deg]

for i in range(n_obs2):
    coord = SkyCoord(
        "{} {} {} {} {} {}".format(
            table2["ra_hr"][i],
            table2["ra_min"][i],
            table2["ra_sec"][i],
            table2["dec_deg"][i],
            table2["dec_arcmin"][i],
            table2["dec_arcsec"][i],
        ),
        unit=(u.hourangle, u.deg),
        frame=ICRS,
    )
    ra = coord.ra.deg
    dec = coord.dec.deg
    df_orbitize["raoff"][i + n_obs1] = float(ra)  # [deg]
    df_orbitize["decoff"][i + n_obs1] = float(dec)  # [deg]

# take difference between reported Hipparcos position and convert to mas

# read the Hipparcos best-fit solution from the IAD file
astrometric_solution = pd.read_csv("H027989.d", skiprows=9, sep="\s+", nrows=1)
hipparcos_alpha0 = astrometric_solution["RAdeg"].values[0]  # [deg]
hipparcos_delta0 = astrometric_solution["DEdeg"].values[0]  # [deg]

df_orbitize["raoff"] = (df_orbitize["raoff"] - hipparcos_alpha0) * u.deg.to(
    u.mas, equivalencies=u.dimensionless_angles()
)
df_orbitize["decoff"] = (df_orbitize["decoff"] - hipparcos_delta0) * u.deg.to(
    u.mas, equivalencies=u.dimensionless_angles()
)
df_orbitize["decoff_err"] = df["decoff_err"].astype(float)
df_orbitize["raoff_err"] = df["decoff_err"].astype(float)

# save
df_orbitize.to_csv("data.csv", index=False)

# make a quick plot of the positions
plt.figure()
plt.errorbar(
    df_orbitize["raoff"],
    df_orbitize["decoff"],
    df_orbitize["raoff_err"],
    df_orbitize["decoff_err"],
    ls="",
    color="rebeccapurple",
)
plt.xlabel("RA [mas]")
plt.ylabel("decl. [mas]")
plt.savefig("../plots/radio_data.png", dpi=250)
