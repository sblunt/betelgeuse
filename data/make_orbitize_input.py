import numpy as np
import pandas as pd
from pandas import DataFrame
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt

"""
Convenience script to convert data from download format to orbitize format.
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
        "ra_err",
        "dec_deg",
        "dec_arcmin",
        "dec_arcsec",
        "dec_err",
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

table2["ra_err"] = table2["error"]
table2["dec_err"] = table2["error"]

df = pd.concat([table1, table2], ignore_index=True)
n_obs = len(df)

df_orbitize = DataFrame(
    Time(df["date"].astype("float"), format="decimalyear").mjd, columns=["epoch"]
)
df_orbitize["object"] = np.ones_like(df_orbitize["epoch"]).astype(int)
df_orbitize["ra"] = np.ones_like(df_orbitize["epoch"])
df_orbitize["dec"] = np.ones_like(df_orbitize["epoch"])

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
    )
    ra = coord.ra.deg
    dec = coord.dec.deg
    df_orbitize["ra"][i] = float(ra)  # [deg]
    df_orbitize["dec"][i] = float(dec)  # [deg]

# take difference between epoch 0 and convert to mas
df_orbitize["ra"] = (df_orbitize["ra"] - df_orbitize["ra"][0]) * u.deg.to(
    u.mas, equivalencies=u.dimensionless_angles()
)
df_orbitize["dec"] = (df_orbitize["dec"] - df_orbitize["dec"][0]) * u.deg.to(
    u.mas, equivalencies=u.dimensionless_angles()
)
df_orbitize["dec_err"] = df["dec_err"].astype(float)
df_orbitize["ra_err"] = df["dec_err"].astype(float)

# save
df_orbitize.to_csv("data.csv", index=False)

# make a quick plot of the positions
plt.figure()
plt.errorbar(
    df_orbitize["ra"],
    df_orbitize["dec"],
    df_orbitize["ra_err"],
    df_orbitize["dec_err"],
    ls="",
    color="rebeccapurple",
)
plt.xlabel("RA [mas]")
plt.ylabel("decl. [mas]")
plt.savefig("radio_data.png", dpi=250)
