from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import astropy.units as u
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    from matplotlib.patches import Circle
except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "eelrfit.plotting needs matplotlib: pip install 'eelrfit[plot]'"
    ) from exc

from .lines import get_line

__all__ = [
    "CATEGORICAL",
    "DIVERGING_CMAP",
    "MARKERS",
    "SEQUENTIAL_CMAP",
    "SEQUENTIAL_POINTS_CMAP",
    "STYLE",
    "demarcation_lines",
    "plot_bpt",
    "plot_line_map",
    "plot_line_ratios",
    "plot_luminosity_history",
    "plot_parameter_profiles",
    "plot_posterior",
    "plot_posterior_panels",
    "plot_residuals",
    "style",
]

# --- palette and style ------------------------------------------------------------------

# Categorical colors in fixed order (validated: adjacent-pair CVD ΔE >= 9.1 for lines;
# the first three also pass all-pairs, for scatter-type plots).
CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
# Marker shapes paired with the categorical slots, so identity never relies on color alone.
MARKERS = ("o", "D", "s", "^", "v", "P", "X", "*")
INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}
_BLUE_RAMP = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)
# Single-hue sequential map (blue, light to dark) for magnitudes.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "eelrfit_blue", _BLUE_RAMP
).with_extremes(bad="#ffffff")
# Diverging map: blue and red poles with a neutral gray midpoint.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "eelrfit_diverging", ("#184f95", "#6da7ec", "#f0efec", "#ec8b8a", "#b73232")
).with_extremes(bad="#ffffff")
# The same ramp starting at a darker step, for markers on a white surface (>= 2:1 contrast).
SEQUENTIAL_POINTS_CMAP = LinearSegmentedColormap.from_list(
    "eelrfit_blue_points", _BLUE_RAMP[2:]
)
for _cmap in (SEQUENTIAL_CMAP, SEQUENTIAL_POINTS_CMAP, DIVERGING_CMAP):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)

STYLE: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK["axis"],
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK["primary"],
    "axes.titlesize": 10,
    "axes.titlecolor": INK["primary"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
    "grid.color": INK["grid"],
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": INK["axis"],
    "ytick.color": INK["axis"],
    "xtick.labelcolor": INK["secondary"],
    "ytick.labelcolor": INK["secondary"],
    "font.size": 9,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "errorbar.capsize": 0,
    "image.cmap": "eelrfit_blue",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def style():
    """Context manager applying EELRFit's figure style (`STYLE`)."""
    return plt.rc_context(STYLE)


def _styled(func):
    @functools.wraps(func)
    def wrapper(*args, style: bool = True, **kwargs):
        if not style:
            return func(*args, **kwargs)
        with plt.rc_context(STYLE):
            return func(*args, **kwargs)

    wrapper.__doc__ = (
        (func.__doc__ or "")
        + "\n\n    Pass ``style=False`` to use the current matplotlib style instead.\n"
    )
    return wrapper


def _new_ax(ax, figsize=(5.0, 3.6)):
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def _as_mapping(results) -> dict[str, Any]:
    if isinstance(results, Mapping):
        return dict(results)
    return {"": results}


def _radius(table) -> tuple[np.ndarray, str, bool]:
    """Representative radius of each region, its axis label, and whether it is in kpc."""
    if "r_mean_kpc" in table.colnames:
        return (
            np.asarray(u.Quantity(table["r_mean_kpc"]).to_value(u.kpc)),
            "Projected radius (kpc)",
            True,
        )
    return (
        np.asarray(u.Quantity(table["r_mean"]).to_value(u.arcsec)),
        "Projected radius (arcsec)",
        False,
    )


def _psf_bar(ax, tables, kpc: bool, *, loc: str = "lower right") -> bool:
    """Draw the PSF FWHM as a scale bar in radius units; True if drawn.

    Annuli narrower than the PSF sample the profile finely, but points closer
    than this are not independent: the bar shows the real resolution.
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    for table in tables:
        fwhm = table.meta.get("psf_fwhm_arcsec")
        scale = table.meta.get("kpc_per_arcsec") if kpc else 1.0
        if fwhm and scale:
            size = fwhm * scale
            bar = AnchoredSizeBar(
                ax.transData,
                size,
                f"PSF FWHM {size:.1f} {'kpc' if kpc else '″'}",
                loc,
                pad=0.4,
                color=INK["secondary"],
                frameon=False,
                size_vertical=0,
                label_top=True,
                fontproperties={"size": 7},
            )
            bar.size_bar.get_children()[0].set_linewidth(1.6)
            ax.add_artist(bar)
            return True
    return False


def _line_label(line_id: str) -> str:
    try:
        line = get_line(line_id)
    except KeyError:
        return line_id
    names = {"HI_6563": "Hα", "HI_4861": "Hβ", "HI_4340": "Hγ", "HI_4102": "Hδ"}
    return names.get(line.id, line.label)


def _plain_log_ticks(ax) -> None:
    """Readable log-axis ticks: 1-2-3-5 within a decade, 1-2-5 up to two, powers of ten beyond."""
    lo, hi = ax.get_ylim()
    if lo <= 0 or hi <= lo:
        return
    decades = np.log10(hi / lo)
    subs = (
        (1.0, 2.0, 3.0, 5.0)
        if decades < 1
        else (1.0, 2.0, 5.0) if decades < 2 else (1.0,)
    )
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _ring(color: str) -> dict[str, Any]:
    """Marker style with a thin surface-colored ring, for markers that overlap."""
    return {"color": color, "markeredgecolor": "white", "markeredgewidth": 0.8}


# --- maps -------------------------------------------------------------------------------


@_styled
def plot_line_map(
    maps,
    line,
    *,
    regions=None,
    snr_threshold: float = 3.0,
    ax=None,
    log: bool = True,
    show_center: bool = True,
    colorbar: bool = True,
):
    """Emission-line flux map with low-S/N spaxels grayed out and annuli overlaid.

    Axes are offsets in arcsec from the region center (or the map center),
    with x increasing to the right of the image (east is to the left on the sky).
    """
    ax = _new_ax(ax, (4.4, 3.8))
    flux = maps.flux_map(line).value
    snr = maps.snr(line)
    good = maps.valid(line) & (snr >= snr_threshold)
    covered = maps.masks.get("coverage", np.ones(maps.shape, bool))
    sx, sy = maps.pixel_scale.to_value(u.arcsec)
    ny, nx = maps.shape
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

    background = np.where(covered & ~good, 1.0, np.nan)
    ax.imshow(
        background,
        origin="lower",
        extent=extent,
        cmap=LinearSegmentedColormap.from_list("g", [INK["grid"], INK["grid"]]),
        interpolation="nearest",
    )
    shown = np.where(good & (flux > 0 if log else True), flux, np.nan)
    finite = shown[np.isfinite(shown)]
    norm = None
    if log and finite.size:
        norm = LogNorm(vmin=np.percentile(finite, 2), vmax=np.percentile(finite, 99.5))
    image = ax.imshow(
        shown,
        origin="lower",
        extent=extent,
        cmap=SEQUENTIAL_CMAP,
        norm=norm,
        interpolation="nearest",
    )
    if colorbar:
        cb = plt.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(f"flux per spaxel ({maps.flux.unit.to_string('unicode')})")
        cb.outline.set_visible(False)
    if regions is not None:
        for edge in regions.edges[1:].to_value(u.arcsec):
            ax.add_patch(
                Circle((0, 0), edge, fill=False, color=INK["secondary"], linewidth=0.7)
            )
    if show_center:
        ax.plot(
            0,
            0,
            marker="+",
            color=INK["primary"],
            markersize=9,
            markeredgewidth=1.2,
            linestyle="none",
        )
    ax.set_xlabel("Δx (arcsec)")
    ax.set_ylabel("Δy (arcsec)")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(
        f"{maps.meta.target_id}: {_line_label(maps.lines[maps.index(line)].id)} (S/N ≥ {snr_threshold:g}; gray: below)"
    )
    return ax


# --- BPT --------------------------------------------------------------------------------


def demarcation_lines(kind: str) -> list[dict[str, Any]]:
    """Standard BPT demarcation curves for ``kind`` in {"NII", "SII", "OI"}.

    Each entry has ``label``, ``x`` and ``y`` (log ratios). Kewley et al. (2001)
    maximum starburst; Kauffmann et al. (2003) empirical star-forming limit (NII);
    Kewley et al. (2006) Seyfert/LINER division (SII, OI); Schawinski et al.
    (2007) Seyfert/LINER division (NII).
    """
    out = []
    if kind == "NII":
        x = np.linspace(-2.2, 0.46, 300)
        out.append(
            {
                "label": "Kewley+01",
                "x": x,
                "y": 0.61 / (x - 0.47) + 1.19,
                "role": "primary",
            }
        )
        x = np.linspace(-2.2, 0.04, 300)
        out.append(
            {
                "label": "Kauffmann+03",
                "x": x,
                "y": 0.61 / (x - 0.05) + 1.30,
                "role": "secondary",
            }
        )
        x = np.linspace(-0.18, 0.8, 100)
        out.append(
            {"label": "Schawinski+07", "x": x, "y": 1.05 * x + 0.45, "role": "muted"}
        )
    elif kind == "SII":
        x = np.linspace(-2.0, 0.31, 300)
        out.append(
            {
                "label": "Kewley+01",
                "x": x,
                "y": 0.72 / (x - 0.32) + 1.30,
                "role": "primary",
            }
        )
        x = np.linspace(-0.31, 0.8, 100)
        out.append(
            {"label": "Kewley+06", "x": x, "y": 1.89 * x + 0.76, "role": "muted"}
        )
    elif kind == "OI":
        x = np.linspace(-2.8, -0.60, 300)
        out.append(
            {
                "label": "Kewley+01",
                "x": x,
                "y": 0.73 / (x + 0.59) + 1.33,
                "role": "primary",
            }
        )
        x = np.linspace(-1.12, 0.3, 100)
        out.append(
            {"label": "Kewley+06", "x": x, "y": 1.18 * x + 1.30, "role": "muted"}
        )
    else:
        raise ValueError(f"kind must be 'NII', 'SII' or 'OI'; got {kind!r}")
    return out


_BPT_LINES = {"NII": ("NII_6584",), "SII": ("SII_6717", "SII_6731"), "OI": ("OI_6300",)}
_BPT_XLABEL = {
    "NII": "log [N II] 6584 / Hα",
    "SII": "log [S II] 6717+6731 / Hα",
    "OI": "log [O I] 6300 / Hα",
}


def _bpt_spaxels(maps, kind, snr_threshold):
    needed = ("HI_4861", "OIII_5007", "HI_6563", *_BPT_LINES[kind])
    if any(line not in maps for line in needed):
        return None
    good = np.ones(maps.shape, bool)
    for line in needed:
        good &= maps.valid(line) & (maps.snr(line) >= snr_threshold)
    f = {line: maps.flux_map(line).value for line in needed}
    x_num = sum(f[l] for l in _BPT_LINES[kind])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(x_num / f["HI_6563"])
        y = np.log10(f["OIII_5007"] / f["HI_4861"])
    good &= np.isfinite(x) & np.isfinite(y)
    return x, y, good


def _bpt_regions(table, kind):
    needed = ("HI_4861", "OIII_5007", "HI_6563", *_BPT_LINES[kind])
    if any(f"{l}_flux" not in table.colnames for l in needed):
        return None
    f = {l: np.asarray(u.Quantity(table[f"{l}_flux"]).value, float) for l in needed}
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(sum(f[l] for l in _BPT_LINES[kind]) / f["HI_6563"])
        y = np.log10(f["OIII_5007"] / f["HI_4861"])
    detected = np.ones(len(table), bool)
    for l in needed:
        if f"{l}_detected" in table.colnames:
            detected &= np.asarray(table[f"{l}_detected"], bool)
    return x, y, detected


@_styled
def plot_bpt(
    maps=None,
    *,
    kinds: Sequence[str] = ("NII", "SII", "OI"),
    regions=None,
    regions_table=None,
    grid=None,
    snr_threshold: float = 3.0,
    axes=None,
    max_tracks: int = 5,
    grid_at: Mapping[str, float] | None = None,
):
    """BPT diagrams: spaxels and region values against the standard demarcation lines.

    Parameters
    ----------
    maps : EmissionLineMaps, optional
        Spaxels with S/N >= ``snr_threshold`` in all four lines are shown,
        colored by projected radius when ``regions`` is given.
    regions_table : QTable, optional
        Region values from `eelrfit.regions.aggregate`, shown as larger markers
        (hollow where a line is below the region S/N threshold).
    grid : CloudyGrid, optional
        Overlay model tracks: constant log nH, varying log U (at most
        ``max_tracks`` densities). These ratios pair lines close in wavelength,
        so the tracks are nearly independent of attenuation.
    grid_at : mapping, optional
        Values of further grid parameters (e.g. ``log_N_scale``) at which the
        tracks are drawn (nearest grid value); unlisted ones are drawn at 0 (solar).
    """
    kinds = tuple(kinds)
    if axes is None:
        fig, axes = plt.subplots(
            1,
            len(kinds),
            figsize=(3.4 * len(kinds), 3.4),
            layout="constrained",
            sharey=True,
        )
        axes = np.atleast_1d(axes)
    radius_norm = None
    if regions is not None:
        rmax = float(regions.edges[-1].to_value(u.arcsec))
        radius_norm = plt.Normalize(0, rmax)
    scatter = None
    for ax, kind in zip(axes, kinds):
        for curve in demarcation_lines(kind):
            ax.plot(
                curve["x"],
                curve["y"],
                color=INK[curve["role"]],
                linewidth=1.0 if curve["role"] == "primary" else 0.8,
                label=curve["label"],
                zorder=2,
            )
        if grid is not None:
            _bpt_tracks(ax, grid, kind, max_tracks, grid_at or {})
        if maps is not None:
            spax = _bpt_spaxels(maps, kind, snr_threshold)
            if spax is not None:
                x, y, good = spax
                if regions is not None:
                    c = regions.radius.to_value(u.arcsec)[good]
                    scatter = ax.scatter(
                        x[good],
                        y[good],
                        c=c,
                        cmap=SEQUENTIAL_POINTS_CMAP,
                        norm=radius_norm,
                        s=6,
                        linewidths=0,
                        zorder=3,
                    )
                else:
                    ax.scatter(
                        x[good],
                        y[good],
                        color=INK["muted"],
                        s=6,
                        linewidths=0,
                        zorder=3,
                        label="spaxels",
                    )
        if regions_table is not None:
            reg = _bpt_regions(regions_table, kind)
            if reg is not None:
                x, y, det = reg
                r = np.asarray(u.Quantity(regions_table["r_mean"]).to_value(u.arcsec))
                colors = (
                    SEQUENTIAL_POINTS_CMAP(radius_norm(r))
                    if radius_norm is not None
                    else [CATEGORICAL[0]] * len(r)
                )  # data blue, models orange
                for xi, yi, di, ci in zip(x, y, det, colors):
                    if np.isfinite(xi) and np.isfinite(yi):
                        ax.plot(
                            xi,
                            yi,
                            marker="D",
                            markersize=7,
                            linestyle="none",
                            zorder=5,
                            markerfacecolor=ci if di else "white",
                            markeredgecolor=INK["primary"],
                            markeredgewidth=0.8,
                        )
        ax.set_xlabel(_BPT_XLABEL[kind])
        ax.set_xlim({"NII": (-1.6, 0.8), "SII": (-1.4, 0.8), "OI": (-2.4, 0.4)}[kind])
        ax.set_ylim(-1.4, 1.5)
    axes[0].set_ylabel("log [O III] 5007 / Hβ")
    from matplotlib.lines import Line2D

    proxies = []
    if maps is not None and regions is not None:
        proxies.append(
            Line2D(
                [],
                [],
                marker="o",
                markersize=3.5,
                linestyle="none",
                color=SEQUENTIAL_POINTS_CMAP(0.6),
                label=f"spaxels, S/N ≥ {snr_threshold:g}",
            )
        )
    if regions_table is not None:
        proxies.append(
            Line2D(
                [],
                [],
                marker="D",
                markersize=6,
                linestyle="none",
                markerfacecolor=SEQUENTIAL_POINTS_CMAP(0.6),
                markeredgecolor=INK["primary"],
                markeredgewidth=0.8,
                label="annulus value",
            )
        )
        proxies.append(
            Line2D(
                [],
                [],
                marker="D",
                markersize=6,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=INK["primary"],
                markeredgewidth=0.8,
                label="annulus, a line below S/N",
            )
        )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles + proxies, labels + [p.get_label() for p in proxies], loc="lower left"
    )
    if scatter is not None:
        cb = plt.colorbar(scatter, ax=list(axes), shrink=0.85, pad=0.01)
        cb.set_label("Projected radius (arcsec)")
        cb.outline.set_visible(False)
    return axes


def _bpt_tracks(ax, grid, kind, max_tracks, grid_at):
    needed = ("HI_4861", "OIII_5007", "HI_6563", *_BPT_LINES[kind])
    if any(l not in grid.line_ids for l in needed) or "log_U" not in grid.parameters:
        return
    inten = grid.intensity()
    # Further grid parameters (e.g. abundance scales) are held at ``grid_at``, else at 0 (solar).
    for name in grid.parameters:
        if name not in ("log_U", "log_nH"):
            inten = inten.sel({name: grid_at.get(name, 0.0)}, method="nearest")
    densities = grid.dataset["log_nH"].values if "log_nH" in grid.parameters else [None]
    if len(densities) > max_tracks:
        densities = densities[
            np.linspace(0, len(densities) - 1, max_tracks).round().astype(int)
        ]
    for k, nh in enumerate(densities):
        sel = inten.sel(log_nH=nh) if nh is not None else inten
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log10(
                np.asarray(
                    sum(sel.sel(line=l) for l in _BPT_LINES[kind])
                    / sel.sel(line="HI_6563"),
                    float,
                )
            )
            y = np.log10(
                np.asarray(sel.sel(line="OIII_5007") / sel.sel(line="HI_4861"), float)
            )
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        ax.plot(
            x[ok], y[ok], color=CATEGORICAL[1], linewidth=0.9, alpha=0.8, zorder=2.5
        )
        if nh is not None:
            # label the low-U end, which lies inside the usual axis limits
            i = int(np.flatnonzero(ok)[0])
            ax.text(
                x[i],
                y[i],
                f"{nh:g} ",
                color=INK["secondary"],
                fontsize=7,
                ha="right",
                va="center",
                clip_on=True,
            )
    ax.plot(
        [],
        [],
        color=CATEGORICAL[1],
        linewidth=0.9,
        label="model (fixed log nH, labeled; U rises upward)",
    )


# --- posteriors -------------------------------------------------------------------------


def _edges(values):
    values = np.asarray(values, float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    mid = 0.5 * (values[1:] + values[:-1])
    return np.concatenate([[2 * values[0] - mid[0]], mid, [2 * values[-1] - mid[-1]]])


def _credible_levels(posterior, levels):
    flat = np.sort(posterior.ravel())[::-1]
    cdf = np.cumsum(flat)
    return sorted(
        float(flat[min(np.searchsorted(cdf, level), len(flat) - 1)]) for level in levels
    )


_PARAM_LABEL = {
    "log_U": "log U",
    "log_nH": r"log $n_{\rm H}$ (cm$^{-3}$)",
    "A_V": r"$A_V$ (mag)",
    "log_N_scale": "Δlog N/O (dex, vs. solar)",
    "log_S_scale": "Δlog S/O (dex, vs. solar)",
}


def _joint_axes(fit) -> tuple[str, str]:
    """(x, y) parameters of the joint posterior panel: log nH (or another grid parameter) vs log U."""
    others = [p for p in fit.parameters if p != "log_U"]
    return ("log_nH" if "log_nH" in others else others[0]), "log_U"


@_styled
def plot_posterior(
    fit,
    *,
    ax=None,
    levels: Sequence[float] = (0.68, 0.95),
    colorbar: bool = True,
    title: str | None = None,
):
    """Posterior over the grid (log U vs log nH) with credible-region contours and the best fit.

    Further grid parameters (e.g. abundance scales) are marginalized; without
    log nH, the next grid parameter takes its place. For one-parameter fits the
    marginal posterior is shown instead.
    """
    ax = _new_ax(ax, (4.2, 3.4))
    if fit.status != "ok":
        ax.text(
            0.5,
            0.5,
            f"not fitted ({fit.status})",
            transform=ax.transAxes,
            ha="center",
            color=INK["secondary"],
        )
        return ax
    if len(fit.parameters) == 1:
        name = fit.parameters[0]
        values, prob = fit.marginal(name)
        ax.stairs(prob, _edges(values), color=CATEGORICAL[0], linewidth=1.5)
        ax.axvline(fit.best[name], color=INK["secondary"], linewidth=0.8)
        ax.set_xlabel(_PARAM_LABEL.get(name, name))
        ax.set_ylabel("posterior probability")
    else:
        post = fit.posterior
        xname, yname = _joint_axes(fit)
        iu, inh = fit.parameters.index(yname), fit.parameters.index(xname)
        p2 = post.sum(axis=tuple(i for i in range(post.ndim) if i not in (iu, inh)))
        p2 = p2 if iu < inh else p2.T
        U, N = fit.coords[yname], fit.coords[xname]
        shown = np.where(
            p2 > 1e-4 * p2.max(), p2, np.nan
        )  # negligible probability is left blank
        mesh = ax.pcolormesh(
            _edges(N), _edges(U), shown, cmap=SEQUENTIAL_CMAP, shading="flat"
        )
        if np.count_nonzero(p2) > 1 and min(len(U), len(N)) > 1:
            lv = _credible_levels(p2, levels)
            cs = ax.contour(
                N, U, p2, levels=sorted(set(lv)), colors=INK["primary"], linewidths=0.8
            )
            fmt = {
                l: f"{int(round(level * 100))}%"
                for l, level in zip(sorted(set(lv)), sorted(levels, reverse=True))
            }
            ax.clabel(cs, fmt=fmt, fontsize=7)
        ax.plot(
            fit.best[xname],
            fit.best[yname],
            marker="*",
            markersize=12,
            linestyle="none",
            markerfacecolor=CATEGORICAL[1],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="best fit",
        )
        ax.set_xlabel(_PARAM_LABEL.get(xname, xname))
        ax.set_ylabel(_PARAM_LABEL[yname])
        ax.grid(False)
        ax.legend(loc="upper right")
        if colorbar:
            cb = plt.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02)
            cb.set_label("posterior probability (A_V marginalized)")
            cb.outline.set_visible(False)
    ax.set_title(
        title
        if title is not None
        else f"χ²_min = {fit.chi2_min:.1f}, {fit.dof} dof, {len(fit.lines_used)} lines"
    )
    return ax


@_styled
def plot_posterior_panels(results, fits, *, n_panels: int = 6):
    """log U - log nH posteriors of up to ``n_panels`` regions spread over radius.

    Constrained regions are preferred. Dotted lines are constant
    n_H U, i.e. constant L_ion at a given radius: where the contours run along
    log U but off the low-density edge, L_ion is only bounded above.
    """
    ok = [k for k, fit in enumerate(fits) if fit.status == "ok"]
    if "constrained" in results.colnames:
        ok = [k for k in ok if results["constrained"][k]] or ok
    chosen = [
        ok[i]
        for i in np.unique(
            np.linspace(0, len(ok) - 1, min(n_panels, len(ok))).round().astype(int)
        )
    ]
    ncols = min(3, max(len(chosen), 1))
    nrows = max(int(np.ceil(len(chosen) / ncols)), 1)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.7 * ncols, 3.4 * nrows),
        layout="constrained",
        squeeze=False,
    )
    radius, _, kpc = _radius(results)
    unit = "kpc" if kpc else "arcsec"
    for ax, k in zip(axes.ravel(), chosen):
        limit = (
            str(results["log_L_limit"][k]) if "log_L_limit" in results.colnames else ""
        )
        status = {
            "upper": "L_ion upper limit",
            "lower": "L_ion lower limit",
            "unconstrained": "L_ion unconstrained",
        }.get(limit, "L_ion measured")
        plot_posterior(
            fits[k],
            ax=ax,
            colorbar=False,
            style=False,
            title=f"r = {radius[k]:.1f} {unit}: {status}\nχ²_min = {fits[k].chi2_min:.1f}, {fits[k].dof} dof",
        )
        if "log_nH" in fits[k].parameters and "log_U" in fits[k].parameters:
            nh, lu = fits[k].coords["log_nH"], fits[k].coords["log_U"]
            x = np.array([nh.min(), nh.max()])
            for c in np.arange(
                np.floor(nh.min() + lu.min()), np.ceil(nh.max() + lu.max()) + 1
            ):
                ax.plot(
                    x, c - x, color=INK["muted"], linewidth=0.6, linestyle=":", zorder=1
                )
            ax.set_xlim(nh.min(), nh.max())
            ax.set_ylim(lu.min(), lu.max())
        if ax is not axes.ravel()[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    first = axes.ravel()[0]
    if chosen and first.get_legend() is not None:
        first.plot(
            [],
            [],
            color=INK["muted"],
            linewidth=0.6,
            linestyle=":",
            label="constant L_ion (1 dex steps)",
        )
        first.legend(loc="upper right", fontsize=7)
    for extra in axes.ravel()[len(chosen) :]:
        extra.set_visible(False)
    return fig


# --- radial profiles ----------------------------------------------------------------------


def _limits(edge: str):
    """errorbar limit flags: probability at the lower edge means the value is an upper limit."""
    edge = edge or ""
    return (
        ("lower" in edge and "upper" not in edge),
        ("upper" in edge and "lower" not in edge),
        ("lower" in edge and "upper" in edge),
    )


@_styled
def plot_parameter_profiles(
    results,
    *,
    parameters: Sequence[str] = ("log_U", "log_nH", "A_V"),
    axes=None,
    show_grid_range: bool = True,
    show_psf: bool = True,
):
    """Radial profiles of fitted parameters (median and 16-84th percentile range).

    ``results`` is a `fit_regions`/`analyze` results table, or a mapping of
    label to table to compare models. Parameters whose probability piles up
    at a grid edge are drawn as limits (arrow from the bound); hollow markers
    mean both edges, i.e. unconstrained. ``show_psf`` draws the PSF FWHM as a
    scale bar on the bottom panel, when the results record it.
    """
    models = _as_mapping(results)
    parameters = [
        p for p in parameters if any(f"{p}_p50" in t.colnames for t in models.values())
    ]
    if axes is None:
        fig, axes = plt.subplots(
            len(parameters),
            1,
            figsize=(5.2, 1.9 * len(parameters) + 0.6),
            sharex=True,
            layout="constrained",
        )
        axes = np.atleast_1d(axes)
    span = []
    for k, (label, table) in enumerate(models.items()):
        r, xlabel, _ = _radius(table)
        span.append(np.nanmax(r) - np.nanmin(r) if len(r) > 1 else 1.0)
    offset_step = 0.012 * (max(span) if span else 1.0)
    for ax, name in zip(axes, parameters):
        for k, (label, table) in enumerate(models.items()):
            if f"{name}_p50" not in table.colnames:
                continue
            r, xlabel, _ = _radius(table)
            x = r + (k - (len(models) - 1) / 2) * offset_step
            p16, p50, p84 = (
                np.asarray(table[f"{name}_{q}"], float) for q in ("p16", "p50", "p84")
            )
            edges = (
                table[f"{name}_edge"]
                if f"{name}_edge" in table.colnames
                else [""] * len(table)
            )
            color, marker = CATEGORICAL[k % 8], MARKERS[k % 8]
            shown = []
            for lo, mid, hi, edge in zip(p16, p50, p84, edges):
                upper_limit, lower_limit, _ = _limits(str(edge))
                shown.append(hi if upper_limit else lo if lower_limit else mid)
            shown = np.asarray(shown, float)
            constrained = (
                np.asarray(table["constrained"], bool)
                if "constrained" in table.colnames
                else np.ones(len(x), bool)
            )
            ok = np.isfinite(shown) & constrained
            ax.plot(x[ok], shown[ok], color=color, linewidth=1.0, alpha=0.6)
            first = True
            for xi, lo, mid, hi, edge, good in zip(
                x, p16, p50, p84, edges, constrained
            ):
                if not np.isfinite(mid):
                    continue
                upper_limit, lower_limit, unconstrained = _limits(str(edge))
                kw = dict(
                    color=color,
                    marker=marker,
                    markersize=6,
                    linestyle="none",
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=(label or None) if first else None,
                    elinewidth=1.0,
                    alpha=1.0 if good else 0.35,
                )
                if upper_limit:
                    ax.errorbar(
                        xi, hi, yerr=[[0.25 * (hi - lo) or 0.2], [0]], uplims=True, **kw
                    )
                elif lower_limit:
                    ax.errorbar(
                        xi, lo, yerr=[[0], [0.25 * (hi - lo) or 0.2]], lolims=True, **kw
                    )
                elif unconstrained:
                    kw.update(markerfacecolor="white", markeredgecolor=color)
                    ax.errorbar(xi, mid, yerr=[[mid - lo], [hi - mid]], **kw)
                else:
                    ax.errorbar(xi, mid, yerr=[[mid - lo], [hi - mid]], **kw)
                first = False
        meta_range = None
        for table in models.values():
            meta_range = (table.meta.get("grid", {}).get("parameters", {}) or {}).get(
                name
            )
            if meta_range:
                break
        if show_grid_range and meta_range:
            for bound in meta_range:
                ax.axhline(bound, color=INK["axis"], linewidth=0.8, zorder=0)
        ax.set_ylabel(_PARAM_LABEL.get(name, name))
    axes[-1].set_xlabel(xlabel)
    if show_psf:
        _psf_bar(axes[-1], models.values(), _radius(next(iter(models.values())))[2])
    if len(models) > 1:
        axes[0].legend(loc="best")
    return axes


_LUM_LABEL = {
    "log_L_ion": r"log $L_{\rm ion}$ (> 13.6 eV) [erg s$^{-1}$]",
    "log_Q": r"log $Q({\rm H})$ [s$^{-1}$]",
}


def _luminosity_label(quantity: str, table) -> str:
    if quantity == "log_L_band":
        band = (table.meta.get("fit_config") or {}).get("band_ev")
        if band:
            return rf"log $L$ ({band[0]:g}–{band[1]:g} eV) [erg s$^{{-1}}$]"
    return _LUM_LABEL.get(quantity, quantity)


# Light-travel time per kpc, in kyr (1 kpc / c = 3.2616 kyr).
KYR_PER_KPC = float((1 * u.kpc / u.Quantity(299792.458, u.km / u.s)).to_value(u.kyr))


@_styled
def plot_luminosity_history(
    results,
    *,
    quantity: str | None = None,
    current: Mapping[str, float] | float | None = None,
    ax=None,
    lookback: bool = True,
    show_psf: bool = True,
):
    """Ionizing luminosity against radius: the AGN's luminosity history.

    Shows the posterior median with its 16-84th percentile band for each
    model; limits are drawn at their quoted bound. ``quantity`` defaults to
    ``log_L_band`` when the results have it, else ``log_L_ion``. ``current``
    adds reference luminosities (e.g. ``{"IR": 42.6}``), each labeled directly. With radii in kpc, the top axis gives the
    light-travel time r/c (a lower limit on the look-back time, since radii are
    projected). ``show_psf`` draws the PSF FWHM as a scale bar, when the
    results record it: points closer than that are not independent.
    """
    ax = _new_ax(ax, (5.4, 3.6))
    models = _as_mapping(results)
    if quantity is None:
        first = next(iter(models.values()))
        quantity = "log_L_band" if "log_L_band_p50" in first.colnames else "log_L_ion"
    in_kpc = True
    any_weak = any_limit = False
    for k, (label, table) in enumerate(models.items()):
        if f"{quantity}_p50" not in table.colnames:
            raise KeyError(
                f"{quantity} is not in the results (need physical radii and an SED)"
            )
        r, xlabel, kpc = _radius(table)
        in_kpc &= kpc
        lo, mid, hi = (
            np.asarray(table[f"{quantity}_{q}"], float) for q in ("p16", "p50", "p84")
        )
        constrained = (
            np.asarray(table["constrained"], bool)
            if "constrained" in table.colnames
            else np.ones(len(r), bool)
        )
        limits = (
            np.asarray(table["log_L_limit"])
            if "log_L_limit" in table.colnames
            else np.array([""] * len(r))
        )
        color, marker = CATEGORICAL[k % 8], MARKERS[k % 8]
        ok = np.isfinite(mid) & constrained
        # the line joins the plotted positions: limits are drawn at their bound, not at the median
        drawn = np.where(limits == "upper", hi, np.where(limits == "lower", lo, mid))
        ax.fill_between(r[ok], lo[ok], hi[ok], color=color, alpha=0.18, linewidth=0)
        ax.plot(r[ok], drawn[ok], color=color, linewidth=1.5, label=label or None)
        for ri, l_, m_, h_, good, lim in zip(r, lo, mid, hi, constrained, limits):
            if not np.isfinite(m_):
                continue
            if not good:
                any_weak = True
                ax.errorbar(
                    ri,
                    m_,
                    yerr=[[m_ - l_], [h_ - m_]],
                    marker=marker,
                    markersize=5,
                    linestyle="none",
                    color=color,
                    alpha=0.45,
                    markerfacecolor="white",
                    elinewidth=0.8,
                )
            elif lim in ("upper", "lower"):
                any_limit = True
                y = h_ if lim == "upper" else l_
                ax.errorbar(
                    ri,
                    y,
                    yerr=[[0.3], [0]] if lim == "upper" else [[0], [0.3]],
                    uplims=lim == "upper",
                    lolims=lim == "lower",
                    marker=marker,
                    markersize=6,
                    linestyle="none",
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                )
            else:
                ax.plot(
                    ri,
                    m_,
                    marker=marker,
                    markersize=6,
                    linestyle="none",
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                )
    if current is not None:
        refs = current if isinstance(current, Mapping) else {"current": current}
        for name, value in refs.items():
            ax.axhline(value, color=INK["secondary"], linewidth=0.9, zorder=1)
            ax.annotate(
                f"{name} ",
                xy=(1.0, value),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=INK["secondary"],
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_luminosity_label(quantity, next(iter(models.values()))))
    from matplotlib.lines import Line2D

    extra = []
    if any_limit:
        extra.append(
            Line2D(
                [],
                [],
                marker="$\\downarrow$",
                linestyle="none",
                color=INK["secondary"],
                markersize=8,
                label="limit (density at grid edge)",
            )
        )
    if any_weak:
        extra.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=INK["muted"],
                markerfacecolor="white",
                markersize=5,
                label="not constrained (Hβ or < 4 lines detected)",
            )
        )
    if show_psf:
        _psf_bar(ax, models.values(), in_kpc)
    if lookback and in_kpc:
        top = ax.secondary_xaxis(
            "top", functions=(lambda r: r * KYR_PER_KPC, lambda t: t / KYR_PER_KPC)
        )
        top.set_xlabel("light-travel time r/c (kyr)", color=INK["secondary"])
        top.tick_params(colors=INK["axis"], labelcolor=INK["secondary"])
    handles, labels = ax.get_legend_handles_labels()
    if handles or extra:
        ax.legend(
            handles + extra,
            labels + [e.get_label() for e in extra],
            loc="best",
            fontsize=7,
        )
    return ax


@_styled
def plot_line_ratios(
    regions_table,
    fits,
    *,
    lines: Iterable[str] | None = None,
    normalize: str = "HI_4861",
    axes=None,
    ncols: int = 3,
):
    """Observed line ratios against radius with the best-fit model, one panel per line.

    Both the data and the model are *attenuated* (observed) ratios: the model
    curve includes the fitted A_V, so the two are directly comparable. Hollow
    markers are lines below the region S/N threshold.
    """
    normalize = get_line(normalize).id
    used = [l for fit in fits for l in fit.lines_used]
    lines = (
        [get_line(l).id for l in lines]
        if lines is not None
        else list(dict.fromkeys(l for l in used if l != normalize))
    )
    lines = [l for l in lines if f"{l}_flux" in regions_table.colnames]
    n = len(lines)
    if axes is None:
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.0 * ncols, 2.3 * nrows),
            sharex=True,
            layout="constrained",
            squeeze=False,
        )
        for extra in axes.ravel()[n:]:
            extra.set_visible(False)
        axes = axes.ravel()[:n]
    r, xlabel, _ = _radius(regions_table)
    fn = np.asarray(u.Quantity(regions_table[f"{normalize}_flux"]).value, float)
    en = np.asarray(u.Quantity(regions_table[f"{normalize}_error"]).value, float)
    for ax, line in zip(axes, lines):
        f = np.asarray(u.Quantity(regions_table[f"{line}_flux"]).value, float)
        e = np.asarray(u.Quantity(regions_table[f"{line}_error"]).value, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = f / fn
            err = np.abs(ratio) * np.sqrt((e / f) ** 2 + (en / fn) ** 2)
        det = (
            np.asarray(regions_table[f"{line}_detected"], bool)
            if f"{line}_detected" in regions_table.colnames
            else np.ones(len(r), bool)
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
        positive = np.isfinite(ratio) & (ratio > 0)
        for mask, face in (
            (positive & det, CATEGORICAL[0]),
            (positive & ~det, "white"),
        ):
            if mask.any():
                ax.errorbar(
                    r[mask],
                    ratio[mask],
                    yerr=err[mask],
                    marker="o",
                    linestyle="none",
                    markersize=5,
                    color=CATEGORICAL[0],
                    markerfacecolor=face,
                    markeredgecolor=CATEGORICAL[0],
                    elinewidth=1.0,
                    zorder=3,
                    label=(
                        "observed" if face != "white" else "observed (S/N < threshold)"
                    ),
                )
        okm = np.isfinite(model) & (model > 0)
        ax.plot(
            r[okm],
            model[okm],
            color=CATEGORICAL[1],
            marker="D",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.2,
            zorder=4,
            label="best-fit model",
        )
        ax.set_yscale("log")
        _plain_log_ticks(ax)
        ax.set_title(f"{_line_label(line)} / {_line_label(normalize)}")
    for ax in axes[-ncols:]:
        ax.set_xlabel(xlabel)
        ax.tick_params(labelbottom=True)
    found = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    order = [
        l
        for l in ("observed", "observed (S/N < threshold)", "best-fit model")
        if l in found
    ]
    if order:
        axes[0].figure.legend(
            [found[l] for l in order],
            order,
            loc="outside lower center",
            ncols=len(order),
        )
    return axes


@_styled
def plot_residuals(results, fits, regions_table, *, ax=None, vlim: float = 5.0):
    """Normalized residuals (observed − model) / σ of every fitted line in every region.

    σ includes the fractional model floor (relative to the observed flux) from
    the fit configuration. Blank cells were not fitted (no valid measurement or
    region not fitted).
    """
    ax = _new_ax(ax, (5.6, 3.6))
    floor = float(results.meta.get("fit_config", {}).get("model_floor", 0.0))
    lines = list(dict.fromkeys(l for fit in fits for l in fit.lines_used))
    z = np.full((len(fits), len(lines)), np.nan)
    for i, fit in enumerate(fits):
        if fit.status != "ok":
            continue
        for j, line in enumerate(lines):
            if line in fit.best_model_flux:
                f = float(u.Quantity(regions_table[f"{line}_flux"][i]).value)
                e = float(u.Quantity(regions_table[f"{line}_error"][i]).value)
                mu = fit.best_model_flux[line]
                z[i, j] = (f - mu) / np.sqrt(e**2 + (floor * f) ** 2)
    r, xlabel, kpc = _radius(regions_table)
    mesh = ax.pcolormesh(
        np.arange(len(lines) + 1),
        np.arange(len(fits) + 1),
        z,
        cmap=DIVERGING_CMAP,
        norm=TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim),
        edgecolors="white",
        linewidth=1.5,
    )
    ax.set_xticks(
        np.arange(len(lines)) + 0.5,
        [_line_label(l) for l in lines],
        rotation=40,
        ha="right",
    )
    unit = "kpc" if kpc else "″"
    ax.set_yticks(np.arange(len(fits)) + 0.5, [f"{x:.1f} {unit}" for x in r])
    ax.set_ylabel("region (mean radius)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02, extend="both")
    cb.set_label("(observed − model) / σ")
    cb.outline.set_visible(False)
    return ax


# --- SEDs ---------------------------------------------------------------------------------
