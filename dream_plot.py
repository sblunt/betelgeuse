import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from orbitize import results
from astropy.time import Time

plt.rcParams["font.family"] = "stixgeneral"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 11
plt.rcParams["figure.facecolor"] = "white"


def compute_iad_tangent_points(
    ra_st_predictions,
    dec_predictions,
    ra_absc_st,
    dec_absc,
    cosphi,
    sinphi,
):
    """
    Args:
        ra_st_predictions (np.array of float): n_epochs x n_orbits array of RA*cos(delta0)
            model predictions to be compared to the IAD
        dec_predictions (np.array of float): n_epochs x n_orbits array of decl
            model predictions to be compared to the IAD
        ra_absc_st (np.array of float): array of RA*cos(delta0) absciscca points
            of Hipparcos data relative to alphadec0_epoch
        dec_absc (np.array of float): array of decl absciscca points
            of Hipparcos data relative to alphadec0_epoch
        cosphi (np.array of float): cosine of scan angles of Hipparcos scans
        sinphi (np.array of float): sine of scan angles of Hipparcos scans

    Returns:
        tuple of:
            np.array of float: n_epochs x n_orbits array of RA tangent points
            np.array of float:  n_epochs x n_orbits array of decl tangent points
    """
    n_orbits = ra_st_predictions.shape[1]
    n_epochs = ra_st_predictions.shape[0]
    tangent_points_ra_st = np.empty((n_epochs, n_orbits))
    tangent_points_decra_st = np.empty((n_epochs, n_orbits))

    for i in range(n_orbits):
        m = sinphi / cosphi
        a1 = 1 / m
        b1 = 1
        c1 = -(dec_predictions[:, i] + 1 / m * ra_st_predictions[:, i])
        a2 = -m
        b2 = 1
        c2 = -(dec_absc - m * ra_absc_st)
        tangent_points_ra_st[:, i] = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
        tangent_points_decra_st[:, i] = (c1 * a2 - c2 * a1) / (a1 * b2 - a2 * b1)

    return tangent_points_ra_st, tangent_points_decra_st


def compute_orbit_prediction(post, epochs_mjd, system):
    """
    Args:
        post (np.array of float): n_params x n_orbits array subset of orbitize posterior
        epochs_mjd (np.array of float): times [in mjd] at which to compute predictions
        system (orbitize.system): system object used for orbit fit from which post was generated

    Returns:
        tuple of:
            np.array of float: n_epochs x n_orbits array of RA*cos(delta0) predictions
            np.array of float:  n_epochs x n_orbits array of decl predictions
    """

    raoff_orbit, deoff_orbit, _ = system.compute_all_orbits(
        post,
        epochs_mjd,
    )

    return (
        raoff_orbit[:, 0, :] * np.cos(np.radians(system.hipparcos_IAD.delta0)),
        deoff_orbit[:, 0, :],
    )


def compute_pm_prediction(post, epochs_mjd, system):
    """
    Args:
        post (np.array of float): n_params x n_orbits array subset of orbitize posterior
        epochs_mjd (np.array of float): times [in mjd] at which to compute predictions
        system (orbitize.system): system object used for orbit fit from which post was generated

    Returns:
        tuple of:
            np.array of float: n_epochs x n_orbits array of RA predictions
            np.array of float:  n_epochs x n_orbits array of decl predictions
    """
    pm_ra_idx = system.param_idx["pm_ra"]
    pm_ra = post[pm_ra_idx, :]
    pm_dec_idx = system.param_idx["pm_dec"]
    pm_dec = post[pm_dec_idx, :]

    n_orbits = len(pm_ra)

    pm_racosdelta_prediction = np.empty((len(epochs_mjd), n_orbits))
    pm_dec_prediction = np.empty((len(epochs_mjd), n_orbits))

    for i in range(n_orbits):
        pm_racosdelta_prediction[:, i] = (
            Time(epochs_mjd, format="mjd").decimalyear
            - system.hipparcos_IAD.alphadec0_epoch
        ) * pm_ra[i]

        pm_dec_prediction[:, i] = (
            Time(epochs_mjd, format="mjd").decimalyear
            - system.hipparcos_IAD.alphadec0_epoch
        ) * pm_dec[i]

    return pm_racosdelta_prediction, pm_dec_prediction


def compute_plx_prediction(post, epochs_mjd, system):
    """
    Args:
        post (np.array of float): n_params x n_orbits array subset of orbitize posterior
        epochs_mjd (np.array of float): times [in mjd] at which to compute predictions
        system (orbitize.system): system object used for orbit fit from which post was generated

    Returns:
        tuple of:
            np.array of float: n_epochs x n_orbits array of RA predictions
            np.array of float:  n_epochs x n_orbits array of decl predictions
    """
    n_orbits = len(post[0, :])
    racosdelta_plx_prediction = np.empty((len(epochs_mjd), n_orbits))
    dec_plx_prediction = np.empty((len(epochs_mjd), n_orbits))

    pm_ra, pm_dec = compute_pm_prediction(post, epochs_mjd, system)
    for i in range(n_orbits):
        pm_plx_ra, pm_plx_dec = system.pm_plx_predictor.compute_astrometric_model(
            post[:, i],
            system.param_idx,
            epochs=Time(epochs_mjd, format="mjd").decimalyear,
        )
        racosdelta_plx_prediction[:, i] = pm_plx_ra - pm_ra[:, i]
        dec_plx_prediction[:, i] = pm_plx_dec - pm_dec[:, i]

    return racosdelta_plx_prediction, dec_plx_prediction


def plot_top_panel(
    post,
    param_idx,
    system,
    epochs2plot_mjd,
    ax,
    astr_data_times_mjd,
    ra_data_values,
    dec_data_values,
    ra_data_errs,
    dec_data_errs,
    epochs_absc,
    ra_absc_st,
    dec_absc,
    cosphi,
    sinphi,
    epsilon,
):
    """
    Args:
        post (np.array of float): n_params x n_orbits array subset of orbitize posterior
        param_idx (dict): orbitize.system.param_idx mapping labels to indices
        system (orbitize.system): system object used for orbit fit from which post was generated
        epochs_mjd (np.array of float): times [in mjd] at which to compute predictions
        ax (np.array of matplotlib.axis.Axis): where to put the plot: ax[0] should be where ra goes,
            and ax[1] should be where dec goes
        astr_data_times_mjd (np.array of float): times [in mjd] at which observed astrometry were
            taken
        ra_data_values (np.array of float): RA values of observed astrometry [mas]
        dec_data_values (np.array of float): dec values of observed astrometry [mas]
        ra_data_errs (np.array of float): RA errors on observed astrometry [mas]
        dec_data_errs (np.array of float): dec errors on observed astrometry [mas]
        epochs_absc (np.array of float): epochs [mjd] of Hipparcos scans
        ra_absc_st (np.array of float): RA*cosdelta0 values of Hipparcos scan abscissca [mas] relative to
            alphadec0_epoch
        dec_absc: decl values of Hipparcos scan abscissca [mas] relative to
            alphadec0_epoch
        cosphi (np.array of float): cos of Hipparcos scans
        sinphi (np.array of float): sine of Hipparcos scans
        epsilon (np.array of float): error on Hipparcos scans
    """

    ra_pm, dec_pm = compute_pm_prediction(post, epochs2plot_mjd, system)
    ra_plx, dec_plx = compute_plx_prediction(post, epochs2plot_mjd, system)
    ra_orbit, dec_orbit = compute_orbit_prediction(post, epochs2plot_mjd, system)
    ra_pm_plx_orbit = ra_pm + ra_plx + ra_orbit
    dec_pm_plx_orbit = dec_pm + dec_plx + dec_orbit

    for i in range(ra_pm_plx_orbit.shape[1]):
        ax[0].plot(
            Time(epochs2plot_mjd, format="mjd").decimalyear,
            ra_pm_plx_orbit[:, i],
            color="grey",
            alpha=0.2,
        )
        ax[1].plot(
            Time(epochs2plot_mjd, format="mjd").decimalyear,
            dec_pm_plx_orbit[:, i],
            color="grey",
            alpha=0.2,
        )

    ax[0].errorbar(
        Time(astr_data_times_mjd, format="mjd").decimalyear,
        ra_data_values,
        ra_data_errs,
        color="purple",
        ls="",
        marker="o",
        alpha=0.2,
    )
    ax[1].errorbar(
        Time(astr_data_times_mjd, format="mjd").decimalyear,
        dec_data_values,
        dec_data_errs,
        color="purple",
        ls="",
        marker="o",
        alpha=0.2,
    )

    ra_pm_iad, dec_pm_iad = compute_pm_prediction(post, epochs_absc, system)
    ra_plx_iad, dec_plx_iad = compute_plx_prediction(post, epochs_absc, system)
    ra_orbit_iad, dec_orbit_iad = compute_orbit_prediction(post, epochs_absc, system)
    ra_pm_plx_orbit_iad = ra_pm_iad + ra_plx_iad + ra_orbit_iad
    dec_pm_plx_orbit_iad = dec_pm_iad + dec_plx_iad + dec_orbit_iad

    # compute tangent points of hip scans wrt sampled posterior values
    ra_tangent_pts, dec_tangent_pts = compute_iad_tangent_points(
        ra_pm_plx_orbit_iad,
        dec_pm_plx_orbit_iad,
        ra_absc_st,
        dec_absc,
        cosphi,
        sinphi,
    )
    ax[0].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(ra_tangent_pts, axis=1),
        np.std(ra_tangent_pts, axis=1),
        ls="",
        marker="o",
        color="blue",
        elinewidth=5,
        alpha=0.2,
    )
    ax[1].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(dec_tangent_pts, axis=1),
        np.std(dec_tangent_pts, axis=1),
        ls="",
        marker="o",
        color="blue",
        elinewidth=5,
        alpha=0.2,
    )

    ra_scan_errors = np.abs(epsilon / sinphi)
    dec_scan_errors = np.abs(epsilon / cosphi)

    ax[0].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(ra_tangent_pts, axis=1),
        ra_scan_errors,
        alpha=0.2,
        ls="",
    )
    ax[1].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(dec_tangent_pts, axis=1),
        dec_scan_errors,
        alpha=0.2,
        ls="",
    )


def plot_middle_panel(
    post,
    param_idx,
    system,
    epochs2plot_mjd,
    ax,
    astr_data_times_mjd,
    ra_data_values,
    dec_data_values,
    ra_data_errs,
    dec_data_errs,
    epochs_absc,
    ra_absc_st,
    dec_absc,
    cosphi,
    sinphi,
    epsilon,
):
    """
    Args:
        post (np.array of float): n_params x n_orbits array subset of orbitize posterior
        param_idx (dict): orbitize.system.param_idx mapping labels to indices
        system (orbitize.system): system object used for orbit fit from which post was generated
        epochs_mjd (np.array of float): times [in mjd] at which to compute predictions
        ax (np.array of matplotlib.axis.Axis): where to put the plot: ax[0] should be where ra goes,
            and ax[1] should be where dec goes
        astr_data_times_mjd (np.array of float): times [in mjd] at which observed astrometry were
            taken
        ra_data_values (np.array of float): RA values of observed astrometry [mas]
        dec_data_values (np.array of float): dec values of observed astrometry [mas]
        ra_data_errs (np.array of float): RA errors on observed astrometry [mas]
        dec_data_errs (np.array of float): dec errors on observed astrometry [mas]
        epochs_absc (np.array of float): epochs [mjd] of Hipparcos scans
        ra_absc_st (np.array of float): RA*cosdelta0 values of Hipparcos scan abscissca [mas] relative to
            alphadec0_epoch
        dec_absc: decl values of Hipparcos scan abscissca [mas] relative to
            alphadec0_epoch
        cosphi (np.array of float): cos of Hipparcos scans
        sinphi (np.array of float): sine of Hipparcos scans
        epsilon (np.array of float): error on Hipparcos scans
    """
    ra_plx, dec_plx = compute_plx_prediction(post, epochs2plot_mjd, system)
    ra_orbit, dec_orbit = compute_orbit_prediction(post, epochs2plot_mjd, system)
    ra_plx_orbit = ra_plx + ra_orbit
    dec_plx_orbit = dec_plx + dec_orbit

    for i in range(ra_plx_orbit.shape[1]):
        ax[0].plot(
            Time(epochs2plot_mjd, format="mjd").decimalyear,
            ra_plx_orbit[:, i],
            color="grey",
            alpha=0.2,
        )
        ax[1].plot(
            Time(epochs2plot_mjd, format="mjd").decimalyear,
            dec_plx_orbit[:, i],
            color="grey",
            alpha=0.2,
        )

    # compute PM predictions at astrometric epochs to plot residuals
    ra_pm, dec_pm = compute_pm_prediction(post, astr_data_times_mjd, system)

    # plot residuals and observational errors
    ax[0].errorbar(
        Time(astr_data_times_mjd, format="mjd").decimalyear,
        ra_data_values - np.median(ra_pm, axis=1),  # n_epochs x n_orbits
        ra_data_errs,
        color="purple",
        ls="",
        marker="o",
        alpha=0.2,
    )
    ax[1].errorbar(
        Time(astr_data_times_mjd, format="mjd").decimalyear,
        dec_data_values - np.median(dec_pm, axis=1),  # n_epochs x n_orbits
        dec_data_errs,
        color="purple",
        ls="",
        marker="o",
        alpha=0.2,
    )

    # plot residuals and model subtraction errors
    for i in range(len(astr_data_times_mjd)):
        ax[0].errorbar(
            Time(astr_data_times_mjd[i], format="mjd").decimalyear,
            # ra_data_values[i] - np.median(ra_pm[i, :]),  # n_epochs x n_orbits
            # np.abs(np.median(ra_data_values[i] - ra_pm[i, :])),
            np.median(ra_data_values[i] - ra_pm[i, :]),  # n_epochs x n_orbits
            np.std(ra_data_values[i] - ra_pm[i, :]),
            color="purple",
            ls="",
            marker="o",
            alpha=0.2,
            elinewidth=5,
        )
        ax[1].errorbar(
            Time(astr_data_times_mjd[i], format="mjd").decimalyear,
            # dec_data_values[i] - np.median(dec_pm[i, :]),  # n_epochs x n_orbits
            # np.abs(np.median(dec_data_values[i] - dec_pm[i, :])),
            np.median(dec_data_values[i] - dec_pm[i, :]),  # n_epochs x n_orbits
            np.std(dec_data_values[i] - dec_pm[i, :]),
            color="purple",
            ls="",
            marker="o",
            alpha=0.2,
            elinewidth=5,
        )

    ra_pm_iad, dec_pm_iad = compute_pm_prediction(post, epochs_absc, system)
    ra_plx_iad, dec_plx_iad = compute_plx_prediction(post, epochs_absc, system)
    ra_orbit_iad, dec_orbit_iad = compute_orbit_prediction(post, epochs_absc, system)
    ra_pm_plx_orbit_iad = ra_pm_iad + ra_plx_iad + ra_orbit_iad
    dec_pm_plx_orbit_iad = dec_pm_iad + dec_plx_iad + dec_orbit_iad

    # compute tangent points of hip scans wrt sampled posterior values
    ra_tangent_pts, dec_tangent_pts = compute_iad_tangent_points(
        ra_pm_plx_orbit_iad,
        dec_pm_plx_orbit_iad,
        ra_absc_st,
        dec_absc,
        cosphi,
        sinphi,
    )
    ax[0].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(ra_tangent_pts - ra_pm_iad, axis=1),
        np.std(ra_tangent_pts - ra_pm_iad, axis=1),
        # np.sqrt(np.std(ra_tangent_pts, axis=1) ** 2 + np.std(ra_pm_iad, axis=1) ** 2),
        ls="",
        marker="o",
        color="blue",
        elinewidth=5,
        alpha=0.2,
    )
    ax[1].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(dec_tangent_pts - dec_pm_iad, axis=1),
        np.std(dec_tangent_pts - dec_pm_iad, axis=1),
        # np.sqrt(np.std(dec_tangent_pts, axis=1) ** 2 + np.std(dec_pm_iad, axis=1) ** 2),
        ls="",
        marker="o",
        color="blue",
        elinewidth=5,
        alpha=0.2,
    )

    ra_scan_errors = np.abs(epsilon / sinphi)
    dec_scan_errors = np.abs(epsilon / cosphi)

    ax[0].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(ra_tangent_pts - ra_pm_iad, axis=1),
        ra_scan_errors,
        alpha=0.2,
        ls="",
    )
    ax[1].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(dec_tangent_pts - dec_pm_iad, axis=1),
        dec_scan_errors,
        alpha=0.2,
        ls="",
    )


def plot_bottom_panel(
    post,
    param_idx,
    system,
    epochs2plot_mjd,
    ax,
    astr_data_times_mjd,
    ra_data_values,
    dec_data_values,
    ra_data_errs,
    dec_data_errs,
    epochs_absc,
    ra_absc_st,
    dec_absc,
    cosphi,
    sinphi,
    epsilon,
):
    """
    Args:
        post (np.array of float): n_params x n_orbits array subset of orbitize posterior
        param_idx (dict): orbitize.system.param_idx mapping labels to indices
        system (orbitize.system): system object used for orbit fit from which post was generated
        epochs_mjd (np.array of float): times [in mjd] at which to compute predictions
        ax (np.array of matplotlib.axis.Axis): where to put the plot: ax[0] should be where ra goes,
            and ax[1] should be where dec goes
        astr_data_times_mjd (np.array of float): times [in mjd] at which observed astrometry were
            taken
        ra_data_values (np.array of float): RA values of observed astrometry [mas]
        dec_data_values (np.array of float): dec values of observed astrometry [mas]
        ra_data_errs (np.array of float): RA errors on observed astrometry [mas]
        dec_data_errs (np.array of float): dec errors on observed astrometry [mas]
        epochs_absc (np.array of float): epochs [mjd] of Hipparcos scans
        ra_absc_st (np.array of float): RA*cosdelta0 values of Hipparcos scan abscissca [mas] relative to
            alphadec0_epoch
        dec_absc: decl values of Hipparcos scan abscissca [mas] relative to
            alphadec0_epoch
        cosphi (np.array of float): cos of Hipparcos scans
        sinphi (np.array of float): sine of Hipparcos scans
        epsilon (np.array of float): error on Hipparcos scans
    """
    ra_orbit, dec_orbit = compute_orbit_prediction(post, epochs2plot_mjd, system)

    orb_period_min = 4
    orb_period_range = 7 - 4

    for i in range(ra_orbit.shape[1]):

        orb_per_i = post[param_idx["per1"], i]  # [yr]
        orb_per_fraction = (orb_per_i - orb_period_min) / orb_period_range

        ax[0].plot(
            Time(epochs2plot_mjd, format="mjd").decimalyear,
            ra_orbit[:, i],
            color=cm.RdBu_r(orb_per_fraction),
            alpha=0.2,
        )
        ax[1].plot(
            Time(epochs2plot_mjd, format="mjd").decimalyear,
            dec_orbit[:, i],
            color=cm.RdBu_r(orb_per_fraction),
            alpha=0.2,
        )

    # compute PM & plx predictions at astrometric epochs to plot residuals
    ra_pm, dec_pm = compute_pm_prediction(post, astr_data_times_mjd, system)
    ra_plx, dec_plx = compute_plx_prediction(post, astr_data_times_mjd, system)
    ra_pm_plx = ra_pm + ra_plx
    dec_pm_plx = dec_pm + dec_plx

    # plot residuals and observational errors
    ax[0].errorbar(
        Time(astr_data_times_mjd, format="mjd").decimalyear,
        ra_data_values - np.median(ra_pm_plx, axis=1),  # n_epochs x n_orbits
        ra_data_errs,
        color="purple",
        ls="",
        marker="o",
        alpha=0.2,
    )
    ax[1].errorbar(
        Time(astr_data_times_mjd, format="mjd").decimalyear,
        dec_data_values - np.median(dec_pm_plx, axis=1),  # n_epochs x n_orbits
        dec_data_errs,
        color="purple",
        ls="",
        marker="o",
        alpha=0.2,
    )

    # plot residuals and model subtraction errors
    for i in range(len(astr_data_times_mjd)):
        ax[0].errorbar(
            Time(astr_data_times_mjd[i], format="mjd").decimalyear,
            # ra_data_values[i] - np.median(ra_pm_plx[i, :]),  # n_epochs x n_orbits
            # np.abs(np.median(ra_data_values[i] - ra_pm_plx[i, :])),
            np.median(ra_data_values[i] - ra_pm_plx[i, :]),  # n_epochs x n_orbits
            np.std(ra_data_values[i] - ra_pm_plx[i, :]),
            color="purple",
            ls="",
            marker="o",
            elinewidth=5,
            alpha=0.2,
        )
        ax[1].errorbar(
            Time(astr_data_times_mjd[i], format="mjd").decimalyear,
            # dec_data_values[i] - np.median(dec_pm_plx[i, :]),  # n_epochs x n_orbits
            # np.abs(np.median(dec_data_values[i] - dec_pm_plx[i, :])),
            np.median(dec_data_values[i] - dec_pm_plx[i, :]),  # n_epochs x n_orbits
            np.std(dec_data_values[i] - dec_pm_plx[i, :]),
            color="purple",
            ls="",
            marker="o",
            elinewidth=5,
            alpha=0.2,
        )

    ra_pm_iad, dec_pm_iad = compute_pm_prediction(post, epochs_absc, system)
    ra_plx_iad, dec_plx_iad = compute_plx_prediction(post, epochs_absc, system)
    ra_orbit_iad, dec_orbit_iad = compute_orbit_prediction(post, epochs_absc, system)
    ra_pm_plx_orbit_iad = ra_pm_iad + ra_plx_iad + ra_orbit_iad
    dec_pm_plx_orbit_iad = dec_pm_iad + dec_plx_iad + dec_orbit_iad
    ra_pm_plx_iad = ra_pm_iad + ra_plx_iad
    dec_pm_plx_iad = dec_pm_iad + dec_plx_iad

    # compute tangent points of hip scans wrt sampled posterior values
    ra_tangent_pts, dec_tangent_pts = compute_iad_tangent_points(
        ra_pm_plx_orbit_iad,
        dec_pm_plx_orbit_iad,
        ra_absc_st,
        dec_absc,
        cosphi,
        sinphi,
    )
    ax[0].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(ra_tangent_pts - ra_pm_plx_iad, axis=1),
        np.std(ra_tangent_pts - ra_pm_plx_iad, axis=1),
        # np.sqrt(
        #     np.std(ra_tangent_pts, axis=1) ** 2
        #     + np.std(ra_pm_plx_orbit_iad, axis=1) ** 2
        # ),
        ls="",
        marker="o",
        color="blue",
        elinewidth=5,
        alpha=0.2,
    )
    ax[1].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(dec_tangent_pts - dec_pm_plx_iad, axis=1),
        np.std(dec_tangent_pts - dec_pm_plx_iad, axis=1),
        # np.sqrt(
        #     np.std(dec_tangent_pts, axis=1) ** 2
        #     + np.std(dec_pm_plx_orbit_iad, axis=1) ** 2
        # ),
        ls="",
        marker="o",
        color="blue",
        elinewidth=5,
        alpha=0.2,
    )

    ra_scan_errors = np.abs(epsilon / sinphi)
    dec_scan_errors = np.abs(epsilon / cosphi)

    ax[0].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(ra_tangent_pts - ra_pm_plx_iad, axis=1),
        ra_scan_errors,
        alpha=0.2,
        ls="",
    )
    ax[1].errorbar(
        Time(epochs_absc, format="mjd").decimalyear,
        np.median(dec_tangent_pts - dec_pm_plx_iad, axis=1),
        dec_scan_errors,
        alpha=0.2,
        ls="",
    )


if __name__ == "__main__":

    font = {"size": 20}

    matplotlib.rc("font", **font)

    zoomout = False

    no_bad_hipparcos = False
    no_hip = False
    restrict_period = False

    # 1A
    run_name = "planetTrue_dvdFalse_renormHIPFalse_fitradioTrue_fithipparcosTrue_burn100_total25000000"

    # 1B
    # run_name = "planetFalse_dvdFalse_renormHIPFalse_fitradioTrue_fithipparcosTrue_burn100_total25000000"

    # 3
    # no_hip = True
    # run_name = "planetFalse_dvdFalse_renormHIPFalse_fitradioTrue_fithipparcosFalse_burn100_total25000000"

    beetle_results = results.Results()
    beetle_results.load_results("results/{}.hdf5".format(run_name))

    # aliases for radio astrometry data
    ra_data = beetle_results.system.data_table["quant1"]
    dec_data = beetle_results.system.data_table["quant2"]
    ra_err_data = beetle_results.system.data_table["quant1_err"]
    dec_err_data = beetle_results.system.data_table["quant2_err"]
    epochs_data = beetle_results.system.data_table["epoch"]

    # aliases for Hipparcos IAD data
    hip_epochs = beetle_results.system.hipparcos_IAD.epochs_mjd
    if no_bad_hipparcos:
        good_mask = Time(hip_epochs, format="mjd").decimalyear > 1990.5
    else:
        good_mask = Time(hip_epochs, format="mjd").decimalyear > 0
    if no_hip:
        good_mask = []

    hip_epochs = hip_epochs[good_mask]
    hip_ra_absc = beetle_results.system.hipparcos_IAD.alpha_abs_st[good_mask]
    hip_dec_absc = beetle_results.system.hipparcos_IAD.delta_abs[good_mask]
    cosphi = beetle_results.system.hipparcos_IAD.cos_phi[good_mask]
    sinphi = beetle_results.system.hipparcos_IAD.sin_phi[good_mask]
    epsilon = beetle_results.system.hipparcos_IAD.eps[good_mask]

    # pick some random orbits from the posterior to plot
    num2plot = 50
    if restrict_period:
        good_post = beetle_results.post[beetle_results.post[:, 0] > 5]
        plot_indices = np.random.randint(0, len(good_post), size=num2plot)
        post = good_post[plot_indices].T
    else:
        plot_indices = np.random.randint(0, len(beetle_results.post), size=num2plot)
        post = beetle_results.post[plot_indices].T

    # pick time at which to plot models
    epochs2plot = np.linspace(44000, 61000, int(1e3))

    # make plot vs time
    fig, ax = plt.subplots(3, 2, figsize=(30, 10), dpi=250, sharex=True)
    plt.subplots_adjust(hspace=0)

    plot_top_panel(
        post,
        beetle_results.system.param_idx,
        beetle_results.system,
        epochs2plot,
        ax[0],
        epochs_data,
        ra_data,
        dec_data,
        ra_err_data,
        dec_err_data,
        hip_epochs,
        hip_ra_absc,
        hip_dec_absc,
        cosphi,
        sinphi,
        epsilon,
    )
    plot_middle_panel(
        post,
        beetle_results.system.param_idx,
        beetle_results.system,
        epochs2plot,
        ax[1],
        epochs_data,
        ra_data,
        dec_data,
        ra_err_data,
        dec_err_data,
        hip_epochs,
        hip_ra_absc,
        hip_dec_absc,
        cosphi,
        sinphi,
        epsilon,
    )
    plot_bottom_panel(
        post,
        beetle_results.system.param_idx,
        beetle_results.system,
        epochs2plot,
        ax[2],
        epochs_data,
        ra_data,
        dec_data,
        ra_err_data,
        dec_err_data,
        hip_epochs,
        hip_ra_absc,
        hip_dec_absc,
        cosphi,
        sinphi,
        epsilon,
    )

    ax[-1, 0].set_xlabel("Time [yr]")
    ax[-1, 1].set_xlabel("Time [yr]")
    ax[1, 0].set_ylabel("$\\Delta$R.A. $\\cos{\\delta_0}$ [mas]")
    ax[1, 1].set_ylabel("$\\Delta$decl. [mas]")

    if zoomout:
        if restrict_period:
            plt.savefig(
                "plots/{}/dreamplot_restrictPeriod.pdf".format(run_name), dpi=250
            )
        else:
            plt.savefig("plots/{}/dreamplot.pdf".format(run_name), dpi=250)
    else:

        for a in ax.flatten():
            a.set_xlim(1990, 1992.5)
        for a in ax[0]:
            a.set_ylim(-100, 100)
        for a in ax[1]:
            a.set_yticks([-5, 0, 5])
            a.set_ylim(-10, 10)
        for a in ax[2]:
            a.set_ylim(-5, 5)
            a.set_yticks([-2.5, 0, 2.5])
        if restrict_period:
            plt.savefig(
                "plots/{}/dreamplot_zoomin_restrictPeriod.pdf".format(run_name), dpi=250
            )
        else:
            plt.savefig("plots/{}/dreamplot_zoomin.pdf".format(run_name), dpi=250)
