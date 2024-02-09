"""
I keep getting confused about whether the astropy `spherical_offsets_to` function
returns delta_ra * cos(delta) or delta_ra, so I wrote this test as a sanity check.

It shows that `spherical_offsets_to` returns delta_ra, not delta_ra * cos(delta)
"""
from astropy.coordinates import SkyCoord, ICRS
from astropy import units as u
import numpy as np

def test_spherical_offsets_to():
    delta1 = 2 * u.deg
    coord1 = SkyCoord(ICRS(ra=1*u.deg, dec=delta1))
    coord2 = SkyCoord(ICRS(ra=2*u.deg, dec=delta1))

    delta2 = 10 * u.deg
    coord3 = SkyCoord(ICRS(ra=4*u.deg, dec=delta2))
    coord4 = SkyCoord(ICRS(ra=5*u.deg, dec=delta2))

    raoff1, deoff1 = coord3.spherical_offsets_to(coord4)
    raoff2, deoff2 = coord1.spherical_offsets_to(coord2)

    # if spherical_offsets_to() returns delta_RA cos(delta), these should be the same
    # (they're not)
    assert not np.isclose(raoff1.to(u.arcsec), raoff2.to(u.arcsec))

    # if spherical_offsets_to() returns delta_RA, these should be the same
    # (they are)
    ra1_cosdelta = raoff1.to(u.arcsec) * np.cos(delta1.to(u.rad, equivalencies=u.dimensionless_angles()).value)
    ra2_cosdelta = raoff2.to(u.arcsec) * np.cos(delta2.to(u.rad, equivalencies=u.dimensionless_angles()).value)
    assert np.isclose(ra1_cosdelta, ra2_cosdelta)


if __name__ == '__main__':
    test_spherical_offsets_to()