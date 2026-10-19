from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import QTable
from dust_extinction.parameter_averages import O94
from scipy.special import logsumexp

from .cloudy.grid import CloudyGrid
from .lines import LINES, get_line
from .sed import CLOUDY_MAX_RYD, RYDBERG_EV, SED, log_q_from_u

__all__ = ["DEFAULT_FIT_LINES", "FitConfig", "RegionFit", "fit_region", "fit_regions"]

# Default fitted lines: no redundant doublet members ([O III] 4959, [N II] 6548 have fixed ratios to 5007, 6584).
DEFAULT_FIT_LINES = (
    "HI_6563",
    "HI_4861",
    "HI_4102",
    "OI_6300",
    "OIII_5007",
    "NII_6584",
    "SII_6717",
    "SII_6731",
)

# Probability within the outermost grid cell above which a parameter is flagged as at the edge.
EDGE_PROBABILITY = 0.05


@dataclasses.dataclass(frozen=True)
class FitConfig:
    """Settings of a grid fit.

    The defaults are explained in the Methodology section of the documentation.

    Attributes
    ----------
    lines : tuple of str
        Lines to fit, if measured and present in the grid.
    model_floor : float
        Fractional model uncertainty, as a fraction of each line's observed
        flux, added in quadrature to its error.
    ratio_floor : mapping
        ``{(line_a, line_b): ε_r}``: fractional model uncertainty of the ratio
        of a pair of lines (default 5% for the [S II] doublet, whose
        ratio depends almost only on density and atomic data). The pair's model
        uncertainties are correlated so that each line keeps ``model_floor``
        but their ratio has ε_r. It only tightens: the ratio never gets more
        than the √2 × ``model_floor`` that independent errors give. Each line
        may be in one pair only; ``{}`` makes all lines independent.
    allow_warned_models : bool
        Use grid models that Cloudy completed with warnings. By default they
        are excluded like failed models (Cloudy's warnings mean the result may
        be unreliable); `CloudyGrid.warnings` flags them.
    attenuation : callable or None
        Returns A(λ)/A_V for wavelengths given as a Quantity, e.g. any
        `dust_extinction` model (default ``O94(Rv=3.1)``); None fits without dust.
    a_v : (start, stop, step)
        Grid of A_V values marginalized over.
    fixed : mapping
        Grid parameters to hold at one grid value, e.g. ``{"log_nH": 2.5}``.
    min_lines : int
        Minimum number of fitted lines; regions with fewer are not fitted.
    min_detected_lines : int
        A fitted region is flagged ``constrained`` only if at least this many
        of its fitted lines are detected (region S/N above threshold) and all
        of ``required_detections`` are. Unconstrained regions are still fitted
        and reported, but should not drive summaries.
    required_detections : tuple of str
        Lines that must be detected for a region to be ``constrained``.
    band_ev : (float, float) or None
        Energy band (eV) for an additional luminosity column ``log_L_band``.
    """

    lines: tuple[str, ...] = DEFAULT_FIT_LINES
    model_floor: float = 0.10
    ratio_floor: Mapping[tuple[str, str], float] = dataclasses.field(
        default_factory=lambda: {("SII_6717", "SII_6731"): 0.05}
    )
    allow_warned_models: bool = False
    attenuation: Callable[[u.Quantity], np.ndarray] | None = dataclasses.field(
        default_factory=lambda: O94(Rv=3.1)
    )
    a_v: tuple[float, float, float] = (0.0, 4.0, 0.05)
    fixed: Mapping[str, float] = dataclasses.field(default_factory=dict)
    min_lines: int = 3
    min_detected_lines: int = 4
    required_detections: tuple[str, ...] = ("HI_4861",)
    band_ev: tuple[float, float] | None = None

    def __post_init__(self):
        object.__setattr__(self, "lines", tuple(get_line(l).id for l in self.lines))
        object.__setattr__(
            self,
            "required_detections",
            tuple(get_line(l).id for l in self.required_detections),
        )
        if self.model_floor < 0:
            raise ValueError("model_floor must be non-negative")
        pairs = {}
        for pair, value in dict(self.ratio_floor).items():
            a, b = (get_line(l).id for l in pair)
            if a == b or {a, b} & {l for p in pairs for l in p}:
                raise ValueError(
                    f"ratio_floor: each line may appear in one pair only ({a}, {b})"
                )
            if value < 0:
                raise ValueError(f"ratio_floor for ({a}, {b}) must be non-negative")
            pairs[(a, b)] = float(value)
        object.__setattr__(self, "ratio_floor", pairs)
        if self.attenuation is not None and not callable(self.attenuation):
            raise TypeError(
                "attenuation must be a callable returning A(λ)/A_V, e.g. dust_extinction's O94(Rv=3.1)"
            )
        object.__setattr__(self, "fixed", dict(self.fixed))

    @property
    def a_v_values(self) -> np.ndarray:
        if self.attenuation is None:
            return np.array([0.0])
        start, stop, step = self.a_v
        n = int(np.floor((stop - start) / step + 1e-6)) + 1
        return np.round(start + step * np.arange(n), 10)

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d["lines"] = list(self.lines)
        d["ratio_floor"] = {f"{a}/{b}": v for (a, b), v in self.ratio_floor.items()}
        d["attenuation"] = _describe(self.attenuation)
        d["fixed"] = dict(self.fixed)
        return d


@dataclasses.dataclass(frozen=True, eq=False)
class RegionFit:
    """Result of fitting one region.

    ``log_posterior`` has one axis per varied grid parameter (in
    ``parameters`` order) and is normalized; ``a_v_posterior`` is the marginal
    distribution of A_V on ``a_v_values``.
    """

    parameters: tuple[str, ...]
    coords: Mapping[str, np.ndarray]
    fixed: Mapping[str, float]
    log_posterior: np.ndarray
    a_v_values: np.ndarray
    a_v_posterior: np.ndarray
    lines_used: tuple[str, ...]
    lines_missing: Mapping[str, str]
    chi2_min: float
    dof: int
    best: Mapping[str, float]
    best_model_flux: Mapping[str, float]
    status: str = "ok"

    @property
    def posterior(self) -> np.ndarray:
        return np.exp(self.log_posterior)

    def marginal(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(values, probabilities) of one parameter (grid parameter or ``"A_V"``)."""
        if name == "A_V":
            return self.a_v_values, self.a_v_posterior
        axis = self.parameters.index(name)
        others = tuple(i for i in range(len(self.parameters)) if i != axis)
        return self.coords[name], (
            self.posterior.sum(axis=others) if others else self.posterior
        )

    def summary(self, name: str) -> dict[str, Any]:
        values, prob = self.marginal(name)
        return _summarize(values, prob, best=self.best.get(name))

    def weighted_percentiles(self, values: np.ndarray) -> dict[str, float]:
        """Percentiles of a derived quantity defined on the grid, under the posterior."""
        return _weighted_percentiles(values.ravel(), self.posterior.ravel())


def _weighted_percentiles(
    values: np.ndarray, weights: np.ndarray, qs=(16, 50, 84)
) -> dict[str, float]:
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return {f"p{q}": np.nan for q in qs}
    v, w = values[ok], weights[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return {f"p{q}": float(np.interp(q / 100, cdf, v)) for q in qs}


def _summarize(
    values: np.ndarray, prob: np.ndarray, best: float | None
) -> dict[str, Any]:
    values = np.asarray(values, float)
    prob = np.asarray(prob, float)
    out = _weighted_percentiles(values, prob)
    out["best"] = best if best is not None else np.nan
    edges = []
    if len(values) > 1:
        if prob[0] > EDGE_PROBABILITY:
            edges.append("lower")
        if prob[-1] > EDGE_PROBABILITY:
            edges.append("upper")
    out["edge"] = "+".join(edges)
    return out


def _measurements(row, lines: Iterable[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for line in lines:
        f, e = row.get(f"{line}_flux"), row.get(f"{line}_error")
        if f is None or e is None:
            continue
        f = float(u.Quantity(f).value)
        e = float(u.Quantity(e).value)
        if np.isfinite(f) and np.isfinite(e) and e > 0:
            out[line] = (f, e)
    return out


def fit_region(
    measurements: Mapping[str, tuple[float, float]],
    grid: CloudyGrid,
    config: FitConfig | None = None,
) -> RegionFit:
    """Fit one region's line fluxes ``{line_id: (flux, error)}`` with ``grid``."""
    config = config or FitConfig()
    missing: dict[str, str] = {}
    used = []
    for line in config.lines:
        if line not in measurements:
            missing[line] = "no valid measurement"
        elif line not in grid.line_ids:
            missing[line] = "not in grid"
        else:
            used.append(line)

    intensity = grid.intensity()
    for name, value in config.fixed.items():
        if name not in grid.parameters:
            raise KeyError(
                f"Cannot fix {name!r}: grid parameters are {grid.parameters}"
            )
        intensity = intensity.sel({name: value})
    params = tuple(d for d in intensity.dims if d != "line")
    coords = {p: intensity[p].values.astype(float) for p in params}
    fixed = {**grid.fixed_parameters, **{k: float(v) for k, v in config.fixed.items()}}
    a_v = config.a_v_values
    shape = tuple(len(coords[p]) for p in params)

    n_free = 1 + (1 if config.attenuation is not None else 0) + len(params)
    if len(used) < max(config.min_lines, 1):
        empty = np.full(shape, np.nan)
        return RegionFit(
            params,
            coords,
            fixed,
            empty,
            a_v,
            np.full(len(a_v), np.nan),
            tuple(used),
            missing,
            np.nan,
            len(used) - n_free,
            {},
            {},
            status="too_few_lines",
        )

    model = (
        intensity.sel(line=used)
        .transpose(*params, "line")
        .values.reshape(-1, len(used))
    )
    usable = (
        grid.usable
        if config.allow_warned_models
        else grid.usable & ~grid.dataset["warnings"]
    )
    for name, value in config.fixed.items():
        usable = usable.sel({name: value})
    usable = usable.transpose(*params).values.reshape(-1) & np.all(
        np.isfinite(model), axis=1
    )

    f = np.array([measurements[l][0] for l in used])
    sigma = np.array([measurements[l][1] for l in used])
    if config.attenuation is not None:
        a_lambda = np.asarray(
            config.attenuation(np.array([LINES[l].wavelength_aa for l in used]) * u.AA),
            float,
        )
    else:
        a_lambda = np.zeros(len(used))
    trans = 10.0 ** (-0.4 * np.outer(a_v, a_lambda))  # (n_av, L)
    g = trans[:, None, :] * model[None, :, :]  # (n_av, P, L)

    cov = np.diag(
        sigma**2 + (config.model_floor * f) ** 2
    )  # (L, L): independent of θ, A_V and s
    for (a, b), ratio_floor in config.ratio_floor.items():
        if a in used and b in used:
            i, j = used.index(a), used.index(b)
            ratio_floor = min(
                ratio_floor, np.sqrt(2) * config.model_floor
            )  # only tightens: correlation >= 0
            cov[i, j] = cov[j, i] = (
                (config.model_floor**2 - ratio_floor**2 / 2) * f[i] * f[j]
            )
    weight = np.linalg.inv(cov)
    gw = g @ weight
    num = gw @ f
    den = np.sum(gw * g, axis=-1)
    s = np.where(den > 0, np.maximum(num / den, 0.0), 0.0)  # exact profile of s >= 0
    mu = s[..., None] * g
    resid = f - mu
    chi2 = np.sum((resid @ weight) * resid, axis=-1)
    loglike = -0.5 * (chi2 + np.linalg.slogdet(2 * np.pi * cov)[1])
    loglike[:, ~usable] = -np.inf

    log_theta = logsumexp(loglike, axis=0)  # marginalize A_V
    norm = logsumexp(log_theta)
    log_post = (log_theta - norm).reshape(shape)
    a_v_post = np.exp(logsumexp(loglike, axis=1) - norm)

    i_av, i_p = np.unravel_index(np.argmax(loglike), loglike.shape)
    best_index = np.unravel_index(i_p, shape)
    best = {p: float(coords[p][best_index[k]]) for k, p in enumerate(params)}
    best["A_V"] = float(a_v[i_av])
    best_flux = {l: float(mu[i_av, i_p, k]) for k, l in enumerate(used)}
    return RegionFit(
        params,
        coords,
        fixed,
        log_post,
        a_v,
        a_v_post,
        tuple(used),
        missing,
        float(chi2[i_av, i_p]),
        len(used) - n_free,
        best,
        best_flux,
    )


def fit_regions(
    table: QTable,
    grid: CloudyGrid,
    config: FitConfig | None = None,
    *,
    sed: SED | None = None,
) -> tuple[QTable, list[RegionFit]]:
    """Fit every region of an `eelrfit.regions.aggregate` table.

    Returns
    -------
    results : QTable
        One row per region: radii, ``n_lines``, ``lines_used``, ``chi2_min``,
        ``dof``, ``status``, ``n_detected`` and ``constrained`` (see
        `FitConfig.min_detected_lines`), ``log_L_limit`` ("upper"/"lower" when
        the density, and hence Q and L, is limited by a grid edge); for each
        grid parameter (``log_U``, ``log_nH``, ``log_N_scale``, ...), ``A_V``,
        ``log_Q`` and the luminosities, ``<name>_p16``, ``_p50``, ``_p84``,
        ``_best`` and (for fitted parameters) ``_edge``; for ``log_Q`` and the
        luminosities also ``_quoted``: the p84 (p16) bound of an upper (lower)
        limit, else the median. Q needs the region radii in kpc; luminosities
        also an ``sed``: ``log_L_ion`` (> 1 Ryd) and, if ``config.band_ev`` is
        set, ``log_L_band``, in erg/s.
    fits : list of RegionFit
        Full posteriors, for diagnostics and plots.
    """
    config = config or FitConfig()
    fits = [
        fit_region(_measurements(row, config.lines), grid, config)
        for row in _rows(table)
    ]
    out = QTable()
    for name in ("region", "r_inner", "r_outer", "r_mean", "r_mean_kpc"):
        if name in table.colnames:
            out[name] = table[name]
    out["status"] = [fit.status for fit in fits]
    out["n_lines"] = [len(fit.lines_used) for fit in fits]
    out["lines_used"] = [",".join(fit.lines_used) for fit in fits]
    out["chi2_min"] = [fit.chi2_min for fit in fits]
    out["dof"] = [fit.dof for fit in fits]
    n_detected, constrained = [], []
    for k, fit in enumerate(fits):
        det = {
            l: bool(table[f"{l}_detected"][k])
            for l in fit.lines_used
            if f"{l}_detected" in table.colnames
        }
        n = sum(det.values())
        required = all(
            (
                bool(table[f"{l}_detected"][k])
                if f"{l}_detected" in table.colnames
                else False
            )
            for l in config.required_detections
        )
        n_detected.append(n)
        constrained.append(
            fit.status == "ok" and required and n >= config.min_detected_lines
        )
    out["n_detected"] = n_detected
    out["constrained"] = constrained

    names = list(dict.fromkeys(p for fit in fits for p in fit.parameters)) + ["A_V"]
    for name in names:
        rows = [
            (
                fit.summary(name)
                if fit.status == "ok" and (name == "A_V" or name in fit.parameters)
                else {}
            )
            for fit in fits
        ]
        for key in ("p16", "p50", "p84", "best"):
            out[f"{name}_{key}"] = [r.get(key, np.nan) for r in rows]
        out[f"{name}_edge"] = [r.get("edge", "") for r in rows]

    radii = table["r_mean_kpc"] if "r_mean_kpc" in table.colnames else None
    log_l_per_q = {}
    if sed is not None:
        log_l_per_q["log_L_ion"] = np.log10(
            sed.band_luminosity_per_q(
                (RYDBERG_EV, CLOUDY_MAX_RYD * RYDBERG_EV)
            ).to_value(u.erg)
        )
        if config.band_ev is not None:
            log_l_per_q["log_L_band"] = np.log10(
                sed.band_luminosity_per_q(config.band_ev).to_value(u.erg)
            )
    derived = {"log_Q": [], **{name: [] for name in log_l_per_q}}
    for k, fit in enumerate(fits):
        if fit.status != "ok" or radii is None or not np.isfinite(radii[k].value):
            for name in derived:
                derived[name].append({})
            continue
        log_u, log_nh = _grid_values(fit)
        log_q = log_q_from_u(log_u, log_nh, radii[k])
        best_q = float(
            log_q_from_u(
                fit.best.get("log_U", fit.fixed.get("log_U", np.nan)),
                fit.best.get("log_nH", fit.fixed.get("log_nH", np.nan)),
                radii[k],
            )
        )
        derived["log_Q"].append({**fit.weighted_percentiles(log_q), "best": best_q})
        for name, offset in log_l_per_q.items():
            derived[name].append(
                {**fit.weighted_percentiles(log_q + offset), "best": best_q + offset}
            )
    if radii is not None and "log_nH_edge" in out.colnames:
        # Q and L scale with n_H: if the density posterior piles up at a grid edge, so do they.
        out["log_L_limit"] = [
            (
                "upper"
                if e == "lower"
                else "lower" if e == "upper" else "unconstrained" if e else ""
            )
            for e in out["log_nH_edge"]
        ]
    for name, rows in derived.items():
        if radii is None:
            continue
        for key in ("p16", "p50", "p84", "best"):
            out[f"{name}_{key}"] = [r.get(key, np.nan) for r in rows]
        # A limit is quoted at its bound: its median depends on the prior range beyond the grid edge.
        limit = (
            np.asarray(out["log_L_limit"])
            if "log_L_limit" in out.colnames
            else np.full(len(out), "")
        )
        out[f"{name}_quoted"] = np.where(
            limit == "upper",
            out[f"{name}_p84"],
            np.where(limit == "lower", out[f"{name}_p16"], out[f"{name}_p50"]),
        )

    for key in (
        "psf_fwhm_arcsec",
        "width_arcsec",
        "kpc_per_arcsec",
    ):  # spatial resolution, for plots
        if table.meta.get(key) is not None:
            out.meta[key] = table.meta[key]
    out.meta.update(
        {
            "fit_config": config.to_dict(),
            "grid": {
                "cloudy_version": grid.cloudy_version,
                "parameters": {
                    p: [float(grid.dataset[p].min()), float(grid.dataset[p].max())]
                    for p in grid.parameters
                },
                "fixed": grid.fixed_parameters,
                "provenance": grid.provenance,
            },
            "sed": (
                {"path": sed.path, "sha256": sed.sha256} if sed is not None else None
            ),
            "radius_for_Q": "r_mean_kpc (mean projected radius of the region's spaxels)",
            "luminosity_units": "log10 erg/s",
            "regions": dict(table.meta),
        }
    )
    return out, fits


def _rows(table: QTable):
    for row in table:
        yield {name: row[name] for name in table.colnames}


def _grid_values(fit: RegionFit) -> tuple[np.ndarray, np.ndarray]:
    """log U and log nH at every grid point of the posterior (fixed values broadcast)."""
    mesh = np.meshgrid(*(fit.coords[p] for p in fit.parameters), indexing="ij")
    values = dict(zip(fit.parameters, mesh))
    shape = fit.log_posterior.shape
    log_u = values.get("log_U", np.full(shape, fit.fixed.get("log_U", np.nan)))
    log_nh = values.get("log_nH", np.full(shape, fit.fixed.get("log_nH", np.nan)))
    return log_u, log_nh


def _describe(curve) -> str | None:
    """A readable name for an attenuation curve, e.g. ``"O94(Rv=3.1)"``."""
    if curve is None:
        return None
    name = getattr(curve, "__name__", None) or type(curve).__name__
    rv = getattr(curve, "Rv", None)
    return f"{name}(Rv={float(getattr(rv, 'value', rv)):g})" if rv is not None else name
