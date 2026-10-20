from __future__ import annotations

from collections.abc import Mapping

import astropy.units as u
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle
except ImportError as exc:  # pragma: no cover - only without matplotlib
    raise ImportError(
        "eelrfit.plotting needs matplotlib: pip install 'eelrfit[plot]'"
    ) from exc

from .lines import get_line

__all__ = [
    "KYR_PER_KPC",
    "demarcation_lines",
    "plot_bpt",
    "plot_line_map",
    "plot_line_ratios",
    "plot_luminosity_history",
    "plot_parameter_profiles",
    "plot_posterior",
    "plot_posterior_panels",
    "plot_residuals",
]

# Light-travel time across 1 kpc, in kyr
KYR_PER_KPC = float((1 * u.kpc / u.Quantity(299792.458, u.km / u.s)).to_value(u.kyr))

PARAMETER_LABELS = {
    "log_U": r"$\log U$",
    "log_nH": r"$\log n_\mathrm{H}$ (cm$^{-3}$)",
    "A_V": r"$A_V$ (mag)",
    "log_N_scale": r"$\Delta \log$ N/O (dex)",
}
LUMINOSITY_LABELS = {
    "log_Q": r"$\log Q(\mathrm{H})$ (s$^{-1}$)",
    "log_L_ion": r"$\log L_\mathrm{ion}$, > 13.6 eV (erg s$^{-1}$)",
}
BALMER_NAMES = {"HI_6563": "Hα", "HI_4861": "Hβ", "HI_4340": "Hγ", "HI_4102": "Hδ"}


# --- shared helpers -----------------------------------------------------------------------


def _axes(ax, figsize):
    """The given axes, or new axes on a new figure."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def _radius(table):
    """Radius of each region, in kpc if available (else arcsec), with its unit."""
    if "r_mean_kpc" in table.colnames:
        return u.Quantity(table["r_mean_kpc"]).to_value(u.kpc), "kpc"
    return u.Quantity(table["r_mean"]).to_value(u.arcsec), "arcsec"


def _line_label(line_id):
    try:
        line = get_line(line_id)
    except KeyError:
        return line_id
    return BALMER_NAMES.get(line.id, line.label)


def _column(table, name, default):
    """A table column as an array, or ``default`` repeated if the column is missing."""
    if name in table.colnames:
        return np.asarray(table[name])
    return np.full(len(table), default)


def _draw_psf(ax, table, unit):
    """Mark the PSF FWHM along the bottom of a radial profile; annuli closer than this are not independent."""
    fwhm = table.meta.get("psf_fwhm_arcsec")
    scale = table.meta.get("kpc_per_arcsec") if unit == "kpc" else 1.0
    if not fwhm or not scale:
        return
    size = fwhm * scale
    x0 = ax.get_xlim()[1] - 1.1 * size
    # x in data coordinates, y in axes coordinates (0 = bottom of the axes)
    ax.plot(
        [x0, x0 + size],
        [0.06, 0.06],
        color="0.3",
        linewidth=2,
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        x0 + size / 2,
        0.08,
        f"PSF FWHM {size:.1f} {unit}",
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.3",
        transform=ax.get_xaxis_transform(),
    )


# --- line maps ----------------------------------------------------------------------------


def plot_line_map(
    maps,
    line,
    *,
    regions=None,
    snr_threshold=3.0,
    ax=None,
    log=True,
    show_center=True,
    colorbar=True,
):
    """Emission-line flux map, with spaxels below ``snr_threshold`` left blank and annuli overlaid.

    The axes are offsets in arcsec from the center of ``regions`` (or of the
    map), with x increasing to the right of the image.
    """
    ax = _axes(ax, (4.6, 3.8))
    flux = maps.flux_map(line).value
    good = maps.valid(line) & (maps.snr(line) >= snr_threshold)
    if log:
        good &= flux > 0
    image = np.where(good, flux, np.nan)

    ny, nx = maps.shape
    sx, sy = maps.pixel_scale.to_value(u.arcsec)
    cx, cy = (
        (regions.center.x, regions.center.y)
        if regions is not None
        else ((nx - 1) / 2, (ny - 1) / 2)
    )
    extent = [
        (-0.5 - cx) * sx,
        (nx - 0.5 - cx) * sx,
        (-0.5 - cy) * sy,
        (ny - 0.5 - cy) * sy,
    ]

    norm = None
    if log and good.any():
        norm = LogNorm(*np.nanpercentile(image, [2, 99.5]))
    shown = ax.imshow(
        image, origin="lower", extent=extent, norm=norm, interpolation="nearest"
    )
    if colorbar:
        plt.colorbar(
            shown, ax=ax, label=f"flux per spaxel ({maps.flux.unit.to_string()})"
        )
    if regions is not None:
        for radius in regions.edges[1:].to_value(u.arcsec):
            ax.add_patch(Circle((0, 0), radius, fill=False, color="0.3", linewidth=0.7))
    if show_center:
        ax.plot(0, 0, marker="+", color="k", markersize=10)
    ax.set(xlabel="Δx (arcsec)", ylabel="Δy (arcsec)", aspect="equal")
    ax.set_title(
        f"{maps.meta.target_id}: {_line_label(get_line(line).id)}, S/N ≥ {snr_threshold:g}"
    )
    return ax


# --- BPT diagrams -------------------------------------------------------------------------

BPT_NUMERATORS = {
    "NII": ("NII_6584",),
    "SII": ("SII_6717", "SII_6731"),
    "OI": ("OI_6300",),
}
BPT_XLABELS = {
    "NII": "log [N II] 6584 / Hα",
    "SII": "log [S II] 6717+6731 / Hα",
    "OI": "log [O I] 6300 / Hα",
}
BPT_XLIMITS = {"NII": (-1.6, 0.8), "SII": (-1.4, 0.8), "OI": (-2.4, 0.4)}


def demarcation_lines(kind):
    """Standard BPT demarcation curves for ``kind`` in {"NII", "SII", "OI"}.

    Returns a list of dicts with ``label``, ``x``, and ``y`` (log ratios): the
    maximum-starburst line of Kewley et al. (2001); for [N II], the empirical
    star-forming limit of Kauffmann et al. (2003) and the Seyfert-LINER line of
    Schawinski et al. (2007); for [S II] and [O I], the Seyfert-LINER lines of
    Kewley et al. (2006).
    """
    if kind == "NII":
        x1, x2, x3 = (
            np.linspace(-2.2, 0.46, 300),
            np.linspace(-2.2, 0.04, 300),
            np.linspace(-0.18, 0.8, 100),
        )
        return [
            {"label": "Kewley+01", "x": x1, "y": 0.61 / (x1 - 0.47) + 1.19},
            {"label": "Kauffmann+03", "x": x2, "y": 0.61 / (x2 - 0.05) + 1.30},
            {"label": "Schawinski+07", "x": x3, "y": 1.05 * x3 + 0.45},
        ]
    if kind == "SII":
        x1, x2 = np.linspace(-2.0, 0.31, 300), np.linspace(-0.31, 0.8, 100)
        return [
            {"label": "Kewley+01", "x": x1, "y": 0.72 / (x1 - 0.32) + 1.30},
            {"label": "Kewley+06", "x": x2, "y": 1.89 * x2 + 0.76},
        ]
    if kind == "OI":
        x1, x2 = np.linspace(-2.8, -0.60, 300), np.linspace(-1.12, 0.3, 100)
        return [
            {"label": "Kewley+01", "x": x1, "y": 0.73 / (x1 + 0.59) + 1.33},
            {"label": "Kewley+06", "x": x2, "y": 1.18 * x2 + 1.30},
        ]
    raise ValueError(f"kind must be 'NII', 'SII' or 'OI'; got {kind!r}")


def _bpt_ratios(flux, kind):
    """log [O III]/Hβ and log (numerator)/Hα from a mapping of line id to flux arrays."""
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(sum(flux[line] for line in BPT_NUMERATORS[kind]) / flux["HI_6563"])
        y = np.log10(flux["OIII_5007"] / flux["HI_4861"])
    return x, y


def plot_bpt(
    maps=None,
    *,
    kinds=("NII", "SII", "OI"),
    regions_table=None,
    grid=None,
    snr_threshold=3.0,
    axes=None,
    max_tracks=5,
    grid_at=None,
):
    """BPT diagrams with spaxels, annuli, and model tracks against the standard demarcation lines.

    Parameters
    ----------
    maps : EmissionLineMaps, optional
        Spaxels with S/N >= ``snr_threshold`` in all four lines are shown as gray points.
    regions_table : QTable, optional
        Region values from `eelrfit.regions.aggregate`, colored by radius; open
        markers have a line below the region S/N threshold.
    grid : CloudyGrid, optional
        Draw model tracks: one per density (at most ``max_tracks``), with log U
        increasing along the track. These ratios pair lines close in wavelength,
        so the tracks hardly depend on attenuation.
    grid_at : mapping, optional
        Values of other grid parameters (e.g. ``log_N_scale``) for the tracks;
        unlisted parameters are held at 0 (solar).
    """
    kinds = tuple(kinds)
    if axes is None:
        _, axes = plt.subplots(
            1,
            len(kinds),
            figsize=(3.6 * len(kinds), 3.6),
            sharey=True,
            layout="constrained",
        )
    axes = np.atleast_1d(axes)
    points = None

    for ax, kind in zip(axes, kinds):
        needed = ("HI_4861", "OIII_5007", "HI_6563", *BPT_NUMERATORS[kind])
        for curve, style in zip(demarcation_lines(kind), ("-", "--", ":")):
            ax.plot(
                curve["x"],
                curve["y"],
                color="k",
                linestyle=style,
                linewidth=1,
                label=curve["label"],
            )

        if grid is not None and all(line in grid.line_ids for line in needed):
            _draw_tracks(ax, grid, kind, max_tracks, grid_at or {})

        if maps is not None and all(line in maps for line in needed):
            good = np.all(
                [
                    maps.valid(line) & (maps.snr(line) >= snr_threshold)
                    for line in needed
                ],
                axis=0,
            )
            x, y = _bpt_ratios(
                {line: maps.flux_map(line).value for line in needed}, kind
            )
            ax.scatter(
                x[good],
                y[good],
                s=4,
                color="0.6",
                label=f"spaxels, S/N ≥ {snr_threshold:g}",
            )

        if regions_table is not None and all(
            f"{line}_flux" in regions_table.colnames for line in needed
        ):
            flux = {
                line: u.Quantity(regions_table[f"{line}_flux"]).value for line in needed
            }
            x, y = _bpt_ratios(flux, kind)
            detected = np.all(
                [_column(regions_table, f"{line}_detected", True) for line in needed],
                axis=0,
            )
            radius, unit = _radius(regions_table)
            points = ax.scatter(
                x[detected],
                y[detected],
                c=radius[detected],
                vmin=0,
                vmax=np.nanmax(radius),
                marker="D",
                s=40,
                edgecolors="k",
                linewidths=0.5,
                zorder=3,
                label="annuli",
            )
            ax.scatter(
                x[~detected],
                y[~detected],
                marker="D",
                s=40,
                facecolors="none",
                edgecolors="k",
                linewidths=0.5,
                zorder=3,
                label="annuli, a line below S/N",
            )

        ax.set(xlabel=BPT_XLABELS[kind], xlim=BPT_XLIMITS[kind], ylim=(-1.4, 1.5))

    axes[0].set_ylabel("log [O III] 5007 / Hβ")
    axes[0].legend(loc="lower left", fontsize=7)
    if points is not None:
        plt.colorbar(points, ax=list(axes), label=f"radius ({unit})")
    return axes


def _draw_tracks(ax, grid, kind, max_tracks, grid_at):
    intensity = grid.intensity()
    for name in grid.parameters:
        if name not in ("log_U", "log_nH"):
            intensity = intensity.sel({name: grid_at.get(name, 0.0)}, method="nearest")
    densities = grid.dataset["log_nH"].values
    if len(densities) > max_tracks:
        densities = densities[
            np.linspace(0, len(densities) - 1, max_tracks).round().astype(int)
        ]
    for k, density in enumerate(densities):
        model = intensity.sel(log_nH=density)
        flux = {line: model.sel(line=line).values for line in model["line"].values}
        x, y = _bpt_ratios(flux, kind)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.plot(
            x[ok],
            y[ok],
            color="C1",
            linewidth=1,
            label="models (log nH labeled)" if k == 0 else None,
        )
        start = np.flatnonzero(ok)[0]  # the low-U end, inside the usual axis limits
        ax.text(
            x[start],
            y[start],
            f"{density:g} ",
            color="C1",
            fontsize=7,
            ha="right",
            va="center",
            clip_on=True,
        )


# --- posteriors -----------------------------------------------------------------------------


def _cell_edges(values):
    """Edges of grid cells centered on ``values``."""
    values = np.asarray(values, float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    middle = (values[1:] + values[:-1]) / 2
    return np.concatenate(
        [[2 * values[0] - middle[0]], middle, [2 * values[-1] - middle[-1]]]
    )


def _credible_levels(posterior, fractions):
    """Posterior values whose contours enclose the given fractions of the probability."""
    p = np.sort(posterior.ravel())[::-1]
    cumulative = np.cumsum(p)
    return sorted(
        float(p[min(np.searchsorted(cumulative, f), len(p) - 1)]) for f in fractions
    )


def plot_posterior(fit, *, ax=None, levels=(0.68, 0.95), colorbar=True, title=None):
    """Posterior of one region over log U and log nH, with credible contours and the best fit.

    Other grid parameters (e.g. abundance scales) and A_V are marginalized. If
    the grid has no log nH, its next parameter is used; for a fit with a single
    free parameter, its marginal posterior is shown.
    """
    ax = _axes(ax, (4.4, 3.6))
    if fit.status != "ok":
        ax.text(
            0.5, 0.5, f"not fitted ({fit.status})", ha="center", transform=ax.transAxes
        )
        return ax

    if len(fit.parameters) == 1:
        (name,) = fit.parameters
        values, probability = fit.marginal(name)
        ax.stairs(probability, _cell_edges(values))
        ax.axvline(fit.best[name], color="k", linewidth=0.8)
        ax.set(xlabel=PARAMETER_LABELS.get(name, name), ylabel="posterior probability")
    else:
        others = [p for p in fit.parameters if p != "log_U"]
        xname = "log_nH" if "log_nH" in others else others[0]
        ix, iu = fit.parameters.index(xname), fit.parameters.index("log_U")
        # sum over every other axis, leaving a (log U, x) array
        joint = fit.posterior.sum(
            axis=tuple(i for i in range(fit.posterior.ndim) if i not in (ix, iu))
        )
        if ix < iu:
            joint = joint.T
        x, y = fit.coords[xname], fit.coords["log_U"]
        shown = np.where(
            joint > 1e-4 * joint.max(), joint, np.nan
        )  # leave negligible probability blank
        mesh = ax.pcolormesh(_cell_edges(x), _cell_edges(y), shown, cmap="Blues")
        if np.count_nonzero(joint) > 1 and len(x) > 1 and len(y) > 1:
            contours = ax.contour(
                x,
                y,
                joint,
                levels=_credible_levels(joint, levels),
                colors="k",
                linewidths=0.8,
            )
            ax.clabel(
                contours,
                fmt=dict(
                    zip(
                        contours.levels,
                        [f"{f:.0%}" for f in sorted(levels, reverse=True)],
                    )
                ),
                fontsize=7,
            )
        ax.plot(
            fit.best[xname],
            fit.best["log_U"],
            marker="*",
            color="C1",
            markersize=12,
            linestyle="none",
            label="best fit",
        )
        ax.set(
            xlabel=PARAMETER_LABELS.get(xname, xname), ylabel=PARAMETER_LABELS["log_U"]
        )
        ax.legend(loc="upper right", fontsize=7)
        if colorbar:
            plt.colorbar(mesh, ax=ax, label="posterior probability")

    if title is None:
        title = (
            f"χ²_min = {fit.chi2_min:.1f}, {fit.dof} dof, {len(fit.lines_used)} lines"
        )
    ax.set_title(title, fontsize=9)
    return ax


def plot_posterior_panels(results, fits, *, n_panels=6):
    """Posteriors of up to ``n_panels`` regions spread over radius, preferring constrained regions.

    Dotted lines mark constant n_H U, i.e. constant ionizing luminosity at the
    region's radius. Where the contours run off the low-density edge along
    log U, the luminosity is only an upper limit.
    """
    fitted = [k for k, fit in enumerate(fits) if fit.status == "ok"]
    constrained = [k for k in fitted if _column(results, "constrained", True)[k]]
    candidates = constrained or fitted
    picks = np.unique(
        np.linspace(0, len(candidates) - 1, min(n_panels, len(candidates)))
        .round()
        .astype(int)
    )
    chosen = [candidates[i] for i in picks]

    ncols = max(min(3, len(chosen)), 1)
    nrows = max(-(-len(chosen) // ncols), 1)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 3.4 * nrows),
        layout="constrained",
        squeeze=False,
    )
    radius, unit = _radius(results)
    limits = _column(results, "log_L_limit", "")
    status = {
        "upper": "L_ion upper limit",
        "lower": "L_ion lower limit",
        "unconstrained": "L_ion unconstrained",
    }

    for ax, k in zip(axes.flat, chosen):
        fit = fits[k]
        title = f"r = {radius[k]:.1f} {unit}: {status.get(str(limits[k]), 'L_ion measured')}"
        plot_posterior(
            fit,
            ax=ax,
            colorbar=False,
            title=f"{title}\nχ²_min = {fit.chi2_min:.1f}, {fit.dof} dof",
        )
        if "log_nH" in fit.parameters:
            nh, lu = fit.coords["log_nH"], fit.coords["log_U"]
            for total in np.arange(
                np.floor(nh.min() + lu.min()), np.ceil(nh.max() + lu.max()) + 1
            ):
                ax.plot(nh, total - nh, color="0.5", linewidth=0.6, linestyle=":")
            ax.set(xlim=(nh.min(), nh.max()), ylim=(lu.min(), lu.max()))
        if ax is not axes.flat[0] and ax.get_legend():
            ax.get_legend().remove()
    if chosen:
        axes.flat[0].plot(
            [],
            [],
            color="0.5",
            linewidth=0.6,
            linestyle=":",
            label="constant L_ion (1 dex steps)",
        )
        axes.flat[0].legend(loc="upper right", fontsize=7)
    for ax in axes.flat[len(chosen) :]:
        ax.set_visible(False)
    return fig


# --- radial profiles ------------------------------------------------------------------------


def _as_models(results):
    """A results table, or a mapping of label to results table, as a mapping."""
    return dict(results) if isinstance(results, Mapping) else {"": results}


def _plot_profile(ax, radius, low, mid, high, kind, constrained, color, label):
    """Draw one profile: values with 16-84% error bars, limits as arrows, unconstrained regions faded.

    ``kind`` holds, per region, "upper" or "lower" for limits, "unconstrained"
    for both, and "" otherwise.
    """
    quoted = np.where(kind == "upper", high, np.where(kind == "lower", low, mid))
    usable = np.isfinite(mid) & constrained
    ax.plot(
        radius[usable], quoted[usable], color=color, linewidth=1, label=label or None
    )
    arrow = 0.3  # length of a limit arrow, in data units
    for r, lo, m, hi, k, good in zip(radius, low, mid, high, kind, constrained):
        if not np.isfinite(m):
            continue
        style = dict(color=color, marker="o", markersize=5, alpha=1.0 if good else 0.3)
        if k == "upper":
            ax.errorbar(r, hi, yerr=[[arrow], [0]], uplims=True, **style)
        elif k == "lower":
            ax.errorbar(r, lo, yerr=[[0], [arrow]], lolims=True, **style)
        else:
            ax.errorbar(
                r,
                m,
                yerr=[[m - lo], [hi - m]],
                markerfacecolor="white" if k else color,
                **style,
            )


def plot_parameter_profiles(
    results,
    *,
    parameters=("log_U", "log_nH", "A_V"),
    axes=None,
    show_grid_range=True,
    show_psf=True,
):
    """Fitted parameters against radius: median and 16-84th percentiles of each region.

    ``results`` is a results table or a mapping of label to results table, to
    compare models. A parameter whose probability piles up at a grid edge is
    drawn as a limit (an arrow from the bound), one at both edges with an open
    marker. Unconstrained regions are faded. The grid's range is drawn as gray
    lines, and the PSF FWHM along the bottom panel.
    """
    models = _as_models(results)
    parameters = [
        p for p in parameters if any(f"{p}_p50" in t.colnames for t in models.values())
    ]
    if axes is None:
        _, axes = plt.subplots(
            len(parameters),
            1,
            figsize=(5.2, 2 * len(parameters) + 0.5),
            sharex=True,
            layout="constrained",
        )
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, parameters):
        for k, (label, table) in enumerate(models.items()):
            if f"{name}_p50" not in table.colnames:
                continue
            radius, unit = _radius(table)
            low, mid, high = (
                np.asarray(table[f"{name}_{q}"], float) for q in ("p16", "p50", "p84")
            )
            # probability at the lower edge of the grid means the value is an upper limit
            edge = _column(table, f"{name}_edge", "").astype(str)
            kind = np.select(
                [edge == "lower", edge == "upper", edge == "lower+upper"],
                ["upper", "lower", "unconstrained"],
                "",
            )
            constrained = _column(table, "constrained", True).astype(bool)
            _plot_profile(ax, radius, low, mid, high, kind, constrained, f"C{k}", label)
        grid_range = (
            next(iter(models.values()))
            .meta.get("grid", {})
            .get("parameters", {})
            .get(name)
        )
        if show_grid_range and grid_range:
            for bound in grid_range:
                ax.axhline(bound, color="0.7", linewidth=0.8, zorder=0)
        ax.set_ylabel(PARAMETER_LABELS.get(name, name))

    first = next(iter(models.values()))
    axes[-1].set_xlabel(f"projected radius ({_radius(first)[1]})")
    if show_psf:
        _draw_psf(axes[-1], first, _radius(first)[1])
    if len(models) > 1:
        axes[0].legend(fontsize=7)
    return axes


def plot_luminosity_history(
    results, *, quantity=None, current=None, ax=None, lookback=True, show_psf=True
):
    """Ionizing luminosity against radius: the luminosity history of the nucleus.

    Each model is drawn as its quoted values (limits at their bound) with the
    16-84th percentile band. ``quantity`` defaults to ``log_L_band`` if present,
    else ``log_L_ion``. ``current`` adds horizontal reference lines, e.g.
    ``{"IR": 42.6}``. With radii in kpc, the top axis gives the light-travel
    time r/c, a lower limit on the look-back time since radii are projected.
    """
    ax = _axes(ax, (5.6, 3.8))
    models = _as_models(results)
    first = next(iter(models.values()))
    if quantity is None:
        quantity = "log_L_band" if "log_L_band_p50" in first.colnames else "log_L_ion"

    for k, (label, table) in enumerate(models.items()):
        if f"{quantity}_p50" not in table.colnames:
            raise KeyError(
                f"{quantity} is not in the results (it needs radii in kpc and an SED)"
            )
        radius, unit = _radius(table)
        low, mid, high = (
            np.asarray(table[f"{quantity}_{q}"], float) for q in ("p16", "p50", "p84")
        )
        kind = _column(table, "log_L_limit", "").astype(str)
        constrained = _column(table, "constrained", True).astype(bool)
        usable = np.isfinite(mid) & constrained
        ax.fill_between(
            radius[usable],
            low[usable],
            high[usable],
            color=f"C{k}",
            alpha=0.2,
            linewidth=0,
        )
        _plot_profile(ax, radius, low, mid, high, kind, constrained, f"C{k}", label)

    references = (
        current
        if isinstance(current, Mapping)
        else ({"current": current} if current is not None else {})
    )
    for name, value in references.items():
        ax.axhline(value, color="0.3", linewidth=0.9)
        ax.annotate(
            name,
            xy=(0.99, value),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=8,
        )

    unit = _radius(first)[1]
    if quantity == "log_L_band" and (
        band := first.meta.get("fit_config", {}).get("band_ev")
    ):
        ylabel = rf"$\log L$, {band[0]:g}–{band[1]:g} eV (erg s$^{{-1}}$)"
    else:
        ylabel = LUMINOSITY_LABELS.get(quantity, quantity)
    ax.set(xlabel=f"projected radius ({unit})", ylabel=ylabel)
    if show_psf:
        _draw_psf(ax, first, unit)
    if lookback and unit == "kpc":
        top = ax.secondary_xaxis(
            "top", functions=(lambda r: r * KYR_PER_KPC, lambda t: t / KYR_PER_KPC)
        )
        top.set_xlabel("light-travel time r/c (kyr)")
    if len(models) > 1:
        ax.legend(fontsize=7)
    return ax


# --- fit quality ----------------------------------------------------------------------------


def plot_line_ratios(
    regions_table, fits, *, lines=None, normalize="HI_4861", axes=None, ncols=3
):
    """Observed line ratios against radius, with the best-fit model, one panel per line.

    Both are observed (attenuated) ratios: the model includes the fitted A_V.
    Open markers are lines below the region S/N threshold.
    """
    normalize = get_line(normalize).id
    if lines is None:
        lines = [
            line
            for line in dict.fromkeys(l for fit in fits for l in fit.lines_used)
            if line != normalize
        ]
    lines = [
        get_line(line).id
        for line in lines
        if f"{get_line(line).id}_flux" in regions_table.colnames
    ]
    if axes is None:
        nrows = -(-len(lines) // ncols)
        _, grid = plt.subplots(
            nrows,
            ncols,
            figsize=(3.2 * ncols, 2.4 * nrows),
            sharex=True,
            layout="constrained",
            squeeze=False,
        )
        for ax in grid.flat[len(lines) :]:
            ax.set_visible(False)
        axes = grid.flat[: len(lines)]

    radius, unit = _radius(regions_table)
    reference = u.Quantity(regions_table[f"{normalize}_flux"]).value
    reference_error = u.Quantity(regions_table[f"{normalize}_error"]).value
    for ax, line in zip(axes, lines):
        flux = u.Quantity(regions_table[f"{line}_flux"]).value
        error = u.Quantity(regions_table[f"{line}_error"]).value
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = flux / reference
            ratio_error = np.abs(ratio) * np.hypot(
                error / flux, reference_error / reference
            )
        model = np.array(
            [
                (
                    fit.best_model_flux.get(line, np.nan)
                    / fit.best_model_flux.get(normalize, np.nan)
                    if fit.status == "ok"
                    else np.nan
                )
                for fit in fits
            ]
        )
        detected = _column(regions_table, f"{line}_detected", True).astype(bool)
        positive = ratio > 0
        ax.errorbar(
            radius[positive & detected],
            ratio[positive & detected],
            yerr=ratio_error[positive & detected],
            color="C0",
            marker="o",
            markersize=4,
            linestyle="none",
            label="observed",
        )
        ax.errorbar(
            radius[positive & ~detected],
            ratio[positive & ~detected],
            yerr=ratio_error[positive & ~detected],
            color="C0",
            marker="o",
            markersize=4,
            markerfacecolor="white",
            linestyle="none",
            label="observed, S/N below threshold",
        )
        ax.plot(
            radius, model, color="C1", marker="D", markersize=3, label="best-fit model"
        )
        ax.set_yscale("log")
        ax.set_title(f"{_line_label(line)} / {_line_label(normalize)}", fontsize=9)
    for ax in axes[-ncols:]:
        ax.set_xlabel(f"projected radius ({unit})")
        ax.tick_params(labelbottom=True)
    axes[0].legend(fontsize=7)
    return axes


def plot_residuals(results, fits, regions_table, *, ax=None, vlim=5.0):
    """Normalized residuals, (observed − model) / σ, for every fitted line in every region.

    σ includes the model uncertainty from the fit configuration. Empty cells
    were not fitted.
    """
    ax = _axes(ax, (5.8, 3.8))
    floor = float(results.meta.get("fit_config", {}).get("model_floor", 0.0))
    lines = list(dict.fromkeys(line for fit in fits for line in fit.lines_used))
    residual = np.full((len(fits), len(lines)), np.nan)
    for i, fit in enumerate(fits):
        for j, line in enumerate(lines):
            if fit.status == "ok" and line in fit.best_model_flux:
                flux = float(u.Quantity(regions_table[f"{line}_flux"][i]).value)
                error = float(u.Quantity(regions_table[f"{line}_error"][i]).value)
                residual[i, j] = (flux - fit.best_model_flux[line]) / np.hypot(
                    error, floor * flux
                )

    mesh = ax.pcolormesh(
        residual, cmap="RdBu_r", vmin=-vlim, vmax=vlim, edgecolors="white", linewidth=1
    )
    radius, unit = _radius(regions_table)
    ax.set_xticks(
        np.arange(len(lines)) + 0.5,
        [_line_label(line) for line in lines],
        rotation=40,
        ha="right",
    )
    ax.set_yticks(np.arange(len(fits)) + 0.5, [f"{r:.1f}" for r in radius])
    ax.set_ylabel(f"region radius ({unit})")
    plt.colorbar(mesh, ax=ax, extend="both", label="(observed − model) / σ")
    return ax
