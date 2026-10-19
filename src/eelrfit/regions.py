from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.table import QTable
from scipy import ndimage

from .lines import EmissionLine, get_line
from .maps import EmissionLineMaps

__all__ = [
    "AnnularRegions",
    "Center",
    "aggregate",
    "annuli",
    "continuum_center",
    "spaxel_correlation",
]

# Masks (from `EmissionLineMaps.masks`) excluded from regions by default.
DEFAULT_EXCLUDED_MASKS = ("gaia_star",)
# Offset from the catalog position above which a note is recorded.
CENTER_OFFSET_WARNING = 0.5 * u.arcsec


# Gaussian width, in spaxels, of the spaxel-noise correlation ρ(d) = exp(−d²/2σ²) of
# reconstructed MaNGA cubes. Fitting the Westfall et al. (2019, AJ 158, 231) correction
# 1 + 1.62 log10 N for compact bins of N = 2-100 spaxels gives σ = 1.917, matching the
# 1.92 spaxels the MaNGA DAP uses for its cube covariance.
SPAXEL_CORRELATION_SIGMA = 1.92

# Default number of annuli per PSF FWHM.
SAMPLES_PER_FWHM = 3


def spaxel_correlation(sigma: float = SPAXEL_CORRELATION_SIGMA) -> np.ndarray:
    """Spaxel-noise correlation kernel ρ(d) = exp(−d²/2σ²) on the spaxel grid (σ in spaxels).

    Truncated at 4σ. Convolving a map with it gives Σ_j ρ_ij a_j at every
    spaxel i, the ingredient of the variance and covariance of region means.
    """
    h = int(np.ceil(4 * sigma))
    y, x = np.mgrid[-h : h + 1, -h : h + 1]
    return np.exp(-0.5 * (x**2 + y**2) / sigma**2)


def _region_covariance(
    error: np.ndarray, members: list[np.ndarray], kernel: np.ndarray
) -> np.ndarray:
    """Covariance matrix of the inverse-variance weighted means of ``members`` (boolean maps).

    With weights w_i = 1/σ_i² and correlation ρ_ij between spaxel errors,
    Cov(mean_r, mean_q) = Σ_{i∈r, j∈q} ρ_ij / (σ_i σ_j) / (S_r S_q), where S = Σ w.
    """
    inv = np.zeros_like(error)
    ok = np.isfinite(error) & (error > 0)
    inv[ok] = 1.0 / error[ok]
    a = [np.where(m, inv, 0.0) for m in members]
    s = np.array([np.sum(x**2) for x in a])
    smooth = [ndimage.convolve(x, kernel, mode="constant") for x in a]
    cov = np.array(
        [[np.sum(a[r] * smooth[q]) for q in range(len(a))] for r in range(len(a))]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.outer(s, s)


@dataclasses.dataclass(frozen=True)
class Center:
    """A center on the spaxel grid."""

    x: float
    y: float
    coord: SkyCoord
    method: str
    notes: tuple[str, ...] = ()

    @property
    def pixel(self) -> tuple[float, float]:
        return (self.x, self.y)


def _pixel_scale_arcsec(maps: EmissionLineMaps) -> tuple[float, float]:
    sx, sy = maps.pixel_scale.to_value(u.arcsec)
    return float(sx), float(sy)


def _excluded(maps: EmissionLineMaps, exclude: Iterable[str]) -> np.ndarray:
    out = np.zeros(maps.shape, dtype=bool)
    for name in exclude:
        if name in maps.masks:
            out |= maps.masks[name]
    return out


def continuum_center(
    maps: EmissionLineMaps,
    *,
    radius: u.Quantity | None = None,
    exclude: Iterable[str] = DEFAULT_EXCLUDED_MASKS,
    max_iterations: int = 20,
) -> Center:
    """Flux-weighted centroid of the continuum map around its peak.

    The peak of the lightly smoothed (3×3) continuum map is taken as the
    starting point; the centroid of positive continuum within ``radius``
    (default: one PSF FWHM, else 1.5″) is then iterated until it moves by less
    than 0.01 spaxel. An offset from the catalog position attached to the maps
    is recorded in ``notes``.
    """
    if maps.continuum is None:
        raise ValueError(
            "These maps have no continuum map; pass an explicit center instead"
        )
    image = np.asarray(maps.continuum.value, dtype=float).copy()
    image[~np.isfinite(image) | (image < 0) | _excluded(maps, exclude)] = 0.0
    if not np.any(image > 0):
        raise ValueError("The continuum map has no positive values")

    if radius is None:
        radius = (
            maps.meta.psf_fwhm if maps.meta.psf_fwhm is not None else 1.5 * u.arcsec
        )
    sx, sy = _pixel_scale_arcsec(maps)
    r_pix = radius.to_value(u.arcsec) / np.sqrt(sx * sy)

    smoothed = ndimage.uniform_filter(image, size=3, mode="constant")
    y0, x0 = np.unravel_index(np.argmax(smoothed), image.shape)
    yy, xx = np.indices(image.shape)
    x, y = float(x0), float(y0)
    for _ in range(max_iterations):
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= r_pix**2
        weights = image * inside
        total = weights.sum()
        if total <= 0:
            break
        nx, ny = float((weights * xx).sum() / total), float(
            (weights * yy).sum() / total
        )
        moved = np.hypot(nx - x, ny - y)
        x, y = nx, ny
        if moved < 0.01:
            break

    coord = maps.wcs.pixel_to_world(x, y)
    notes = []
    candidates: dict[str, SkyCoord] = {}
    if maps.properties is not None:
        candidates.update(maps.properties.centers)
    if maps.meta.object_coord is not None:
        candidates.setdefault("header_object", maps.meta.object_coord)
    for name, other in candidates.items():
        sep = coord.separation(other).to(u.arcsec)
        if name == "catalog" and sep > CENTER_OFFSET_WARNING:
            notes.append(f"continuum centroid is {sep:.2f} from the catalog position")
    return Center(x, y, coord, "continuum_centroid", tuple(notes))


def _resolve_center(maps: EmissionLineMaps, center) -> Center:
    if center is None or center == "continuum":
        return continuum_center(maps)
    if isinstance(center, Center):
        return center
    if isinstance(center, str):
        candidates: dict[str, SkyCoord] = (
            dict(maps.properties.centers) if maps.properties is not None else {}
        )
        if maps.meta.object_coord is not None:
            candidates.setdefault("header_object", maps.meta.object_coord)
        if maps.meta.pointing_coord is not None:
            candidates.setdefault("header_pointing", maps.meta.pointing_coord)
        if center not in candidates:
            raise KeyError(
                f"Unknown center {center!r}; available: continuum, {', '.join(candidates)}"
            )
        coord, method = candidates[center], center
    elif isinstance(center, SkyCoord):
        coord, method = center, "user"
    else:
        raise TypeError(
            f"center must be 'continuum', a center name, a SkyCoord or a Center; got {center!r}"
        )
    x, y = maps.wcs.world_to_pixel(coord)
    return Center(float(x), float(y), coord, method)


@dataclasses.dataclass(frozen=True, eq=False)
class AnnularRegions:
    """Concentric circular annuli on the spaxel grid.

    ``labels`` holds the annulus index of every spaxel (−1 outside all annuli
    or outside the coverage mask); annulus ``i`` spans ``edges[i]`` to
    ``edges[i + 1]`` in projected radius.
    """

    center: Center
    edges: u.Quantity
    labels: np.ndarray
    radius: u.Quantity

    @property
    def n_regions(self) -> int:
        return len(self.edges) - 1

    @property
    def width(self) -> u.Quantity:
        return np.diff(self.edges)


def annuli(
    maps: EmissionLineMaps,
    *,
    center: Center | SkyCoord | str | None = "continuum",
    width: u.Quantity | None = None,
    samples_per_fwhm: float = SAMPLES_PER_FWHM,
    max_radius: u.Quantity | None = None,
    exclude: Iterable[str] = DEFAULT_EXCLUDED_MASKS,
) -> AnnularRegions:
    """Projected circular annuli around ``center``.

    Parameters
    ----------
    center : "continuum", center name, SkyCoord or Center
        ``"continuum"`` (default) uses `continuum_center`; a name selects one
        of ``maps.properties.centers`` (which include the header positions).
    width : Quantity, optional
        Annulus width; default ``maps.meta.psf_fwhm / samples_per_fwhm``.
    samples_per_fwhm : float
        Annuli per PSF FWHM when ``width`` is not given (default 3). Adjacent
        annuli then share light through the PSF; `aggregate` reports their
        noise correlation, and the PSF FWHM sets the real resolution.
    max_radius : Quantity, optional
        Outer limit; default the largest radius of any covered spaxel.
    exclude : iterable of str
        Masks whose spaxels are left out of every annulus.
    """
    c = _resolve_center(maps, center)
    if width is None:
        if maps.meta.psf_fwhm is None:
            raise ValueError(
                "No PSF FWHM is recorded for these maps; pass width explicitly"
            )
        if samples_per_fwhm <= 0:
            raise ValueError("samples_per_fwhm must be positive")
        width = maps.meta.psf_fwhm / samples_per_fwhm
    width = width.to(u.arcsec)
    if width <= 0 * u.arcsec:
        raise ValueError("width must be positive")

    sx, sy = _pixel_scale_arcsec(maps)
    yy, xx = np.indices(maps.shape)
    radius = np.hypot((xx - c.x) * sx, (yy - c.y) * sy)
    usable = maps.masks.get("coverage", np.ones(maps.shape, bool)) & ~_excluded(
        maps, exclude
    )
    if not usable.any():
        raise ValueError("No spaxels are covered")
    rmax = (
        max_radius.to_value(u.arcsec)
        if max_radius is not None
        else radius[usable].max()
    )
    n = max(int(np.ceil(rmax / width.value - 1e-9)), 1)
    edges = np.arange(n + 1) * width.value
    labels = np.floor(radius / width.value).astype(int)
    labels[(labels >= n) | ~usable] = -1
    return AnnularRegions(c, edges * u.arcsec, labels, radius * u.arcsec)


def aggregate(
    maps: EmissionLineMaps,
    regions: AnnularRegions,
    *,
    lines: Iterable[str | EmissionLine] | None = None,
    snr_threshold: float = 3.0,
    covariance: str | None = "gaussian",
    correlation_sigma: float = SPAXEL_CORRELATION_SIGMA,
    cosmology=Planck18,
) -> QTable:
    """Combine the spaxels of each region into one measurement per line.

    Returns
    -------
    QTable
        One row per region with ``region``, ``r_inner``, ``r_outer``,
        ``r_mean`` (mean radius of covered spaxels) and ``n_spaxels``, plus, if
        ``maps.properties`` has a redshift, the same radii in kpc. For each
        line ``<id>``: ``<id>_flux`` and ``<id>_error`` (weighted mean and its
        covariance-corrected uncertainty, per spaxel), ``<id>_n`` (valid
        spaxels used), ``<id>_snr``, ``<id>_detected`` (S/N >= threshold),
        ``<id>_scatter`` (weighted standard deviation of the spaxels) and
        ``<id>_rchi2`` (reduced chi-square of the spaxels about the mean).
        ``meta`` records the center, binning, threshold, covariance model,
        cosmology and source provenance.

    ``covariance`` sets how correlated spaxel noise enters the uncertainties:
    ``"gaussian"`` (default) evaluates the correlation ρ(d) = exp(−d²/2σ²),
    σ = ``correlation_sigma`` spaxels, for each region's actual spaxels and
    weights; None assumes independent spaxels. The default σ is calibrated
    for MaNGA cubes; other instruments may need their own value. With ``"gaussian"``, ``meta``
    also holds ``region_noise_correlation``: the correlation matrix between
    region means for uniform spaxel errors, for fits that combine regions.
    """
    ids = (
        [get_line(line).id for line in lines]
        if lines is not None
        else list(maps.line_ids)
    )
    n_regions = regions.n_regions
    labels = regions.labels
    table = QTable()
    table["region"] = np.arange(n_regions)
    table["r_inner"] = regions.edges[:-1]
    table["r_outer"] = regions.edges[1:]
    radius = regions.radius.to_value(u.arcsec)
    table["r_mean"] = (
        np.array(
            [
                radius[labels == i].mean() if np.any(labels == i) else np.nan
                for i in range(n_regions)
            ]
        )
        * u.arcsec
    )
    table["n_spaxels"] = np.array([np.sum(labels == i) for i in range(n_regions)])

    kpc_per_arcsec = None
    props = maps.properties
    if props is not None and props.redshift is not None and cosmology is not None:
        d_a = cosmology.angular_diameter_distance(props.redshift)
        kpc_per_arcsec = (d_a * (1 * u.arcsec).to_value(u.rad)).to(u.kpc) / u.arcsec
        for name in ("r_inner", "r_outer", "r_mean"):
            table[f"{name}_kpc"] = (table[name] * kpc_per_arcsec).to(u.kpc)

    gaussian = covariance == "gaussian"
    if covariance is not None and not gaussian:
        raise ValueError(
            f"Unknown covariance model {covariance!r}; use 'gaussian' or None"
        )
    kernel = spaxel_correlation(correlation_sigma) if gaussian else None
    flux_unit = maps.flux.unit
    for line_id in ids:
        i = maps.index(line_id)
        flux = maps.flux.value[i]
        error = maps.error.value[i]
        valid = maps.valid(line_id)
        cols = {
            k: np.full(n_regions, np.nan)
            for k in ("flux", "error", "snr", "scatter", "rchi2")
        }
        counts = np.zeros(n_regions, dtype=int)
        if gaussian:
            region_var = np.diag(
                _region_covariance(
                    error, [(labels == r) & valid for r in range(n_regions)], kernel
                )
            )
        for r in range(n_regions):
            sel = (labels == r) & valid
            n = int(sel.sum())
            counts[r] = n
            if n == 0:
                continue
            f, w = flux[sel], 1.0 / error[sel] ** 2
            mean = float(np.sum(w * f) / np.sum(w))
            sigma = float(np.sqrt(1.0 / np.sum(w)))
            if gaussian:
                sigma = float(np.sqrt(region_var[r]))
            resid = np.sum(w * (f - mean) ** 2)
            cols["flux"][r] = mean
            cols["error"][r] = sigma
            cols["snr"][r] = mean / sigma
            cols["scatter"][r] = float(np.sqrt(resid / np.sum(w)))
            cols["rchi2"][r] = float(resid / (n - 1)) if n > 1 else np.nan
        table[f"{line_id}_flux"] = cols["flux"] * flux_unit
        table[f"{line_id}_error"] = cols["error"] * flux_unit
        table[f"{line_id}_n"] = counts
        table[f"{line_id}_snr"] = cols["snr"]
        table[f"{line_id}_detected"] = (
            np.nan_to_num(cols["snr"], nan=-np.inf) >= snr_threshold
        )
        table[f"{line_id}_scatter"] = cols["scatter"] * flux_unit
        table[f"{line_id}_rchi2"] = cols["rchi2"]

    c = regions.center
    table.meta.update(
        {
            "lines": ids,
            "flux_per": "spaxel",
            "center": {
                "method": c.method,
                "x_pixel": c.x,
                "y_pixel": c.y,
                "ra_deg": float(c.coord.ra.deg),
                "dec_deg": float(c.coord.dec.deg),
                "notes": list(c.notes),
            },
            "region_shape": "projected circular annuli",
            "width_arcsec": float(regions.width[0].to_value(u.arcsec)),
            "psf_fwhm_arcsec": (
                float(maps.meta.psf_fwhm.to_value(u.arcsec))
                if maps.meta.psf_fwhm is not None
                else None
            ),
            "snr_threshold": snr_threshold,
            "snr_applied_to": "region values (no per-spaxel cut)",
            "covariance_correction": (
                f"gaussian spaxel correlation, sigma = {correlation_sigma:g} spaxels"
                if gaussian
                else "none"
            ),
            "cosmology": (
                getattr(cosmology, "name", str(cosmology))
                if kpc_per_arcsec is not None
                else None
            ),
            "kpc_per_arcsec": (
                float(kpc_per_arcsec.value) if kpc_per_arcsec is not None else None
            ),
            "source": maps.meta.to_dict(),
            "properties": (
                maps.properties.to_dict() if maps.properties is not None else None
            ),
            "line_provenance": {
                k: dict(v) for k, v in maps.line_provenance.items() if k in ids
            },
        }
    )
    if gaussian:
        covered = [labels == r for r in range(n_regions)]
        cov = _region_covariance(np.ones(maps.shape), covered, kernel)
        d = np.sqrt(np.diag(cov))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(d, d)
        table.meta["region_noise_correlation"] = [
            [None if not np.isfinite(v) else round(float(v), 4) for v in row]
            for row in corr
        ]
    return table
