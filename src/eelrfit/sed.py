from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import astropy.constants as const
import astropy.units as u
import numpy as np

__all__ = [
    "CLOUDY_MAX_RYD",
    "CLOUDY_MIN_RYD",
    "RYDBERG_EV",
    "SED",
    "log_q_from_u",
    "read_cloudy_sed",
    "write_cloudy_sed",
]

RYDBERG_EV = (1 * u.Ry).to_value(u.eV)
# Upper end of Cloudy's continuum energy mesh (7.354e6 Ryd, about 100 MeV).
CLOUDY_MAX_RYD = 7.354e6
# Lower end of Cloudy's continuum energy mesh (1e-8 Ryd).
CLOUDY_MIN_RYD = 1.0e-8

_ENERGY_UNITS = {
    "ryd": ("energy", RYDBERG_EV),
    "ev": ("energy", 1.0),
    "kev": ("energy", 1e3),
    "mev": ("energy", 1e6),
    "hz": ("frequency", None),
    "angstrom": ("wavelength", 1e-10),
    "angstroms": ("wavelength", 1e-10),
    "nm": ("wavelength", 1e-9),
    "micron": ("wavelength", 1e-6),
    "microns": ("wavelength", 1e-6),
    "cm": ("wavelength", 1e-2),
}


@dataclasses.dataclass(frozen=True, eq=False)
class SED:
    """An SED as photon energy (eV) and f_ν (arbitrary normalization)."""

    energy_ev: np.ndarray
    fnu: np.ndarray
    path: str = ""
    sha256: str = ""
    meta: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        order = np.argsort(self.energy_ev)
        e, f = (
            np.asarray(self.energy_ev, float)[order],
            np.asarray(self.fnu, float)[order],
        )
        if len(e) < 2 or np.any(np.diff(e) <= 0):
            raise ValueError("SED energies must be distinct and at least two")
        if np.any(f < 0):
            raise ValueError("SED fluxes must be non-negative")
        object.__setattr__(self, "energy_ev", e)
        object.__setattr__(self, "fnu", f)

    def fnu_at(self, energy_ev: np.ndarray) -> np.ndarray:
        """f_ν interpolated linearly in log-log space (zero outside the tabulated range)."""
        e = np.asarray(energy_ev, float)
        positive = self.fnu > 0
        if not positive.any():
            return np.zeros_like(e)
        # Zero-flux points are floored at a tiny positive value so that log-log
        # interpolation stays finite; exp() of the result is then effectively zero.
        floor = 1e-300
        logf = np.interp(
            np.log(e),
            np.log(self.energy_ev),
            np.log(np.maximum(self.fnu, floor)),
            left=-np.inf,
            right=-np.inf,
        )
        out = np.exp(logf)
        out[out <= floor * (1 + 1e-9)] = 0.0
        return out

    def _integrate(
        self, lo_ev: float, hi_ev: float, power: int, n: int = 20000
    ) -> float:
        """∫ f_ν ν^power dν between two energies (frequencies in Hz)."""
        hi_ev = min(hi_ev, self.energy_ev[-1])
        lo_ev = max(lo_ev, self.energy_ev[0])
        if hi_ev <= lo_ev:
            return 0.0
        e = np.geomspace(lo_ev, hi_ev, n)
        nu = (e * u.eV).to_value(u.Hz, u.spectral())
        y = self.fnu_at(e) * nu**power
        # Integrate in log ν for accuracy over many decades.
        return float(np.trapezoid(y * nu, np.log(nu)))

    def photon_rate_per_unit(self, max_ryd: float = CLOUDY_MAX_RYD) -> float:
        """∫_{1 Ryd} f_ν / (hν) dν: the ionizing photon rate for this normalization."""
        h = const.h.cgs.value
        return self._integrate(RYDBERG_EV, max_ryd * RYDBERG_EV, power=-1) / h

    def band_luminosity_per_q(self, band_ev: tuple[float, float]) -> u.Quantity:
        """Luminosity in an energy band per ionizing photon per second (erg).

        A grid's ionization parameter fixes Q(H) = 4π r² n_H c U, counting all
        photons above 1 Ryd, so L_band = Q(H) × ∫_band L_ν dν / ∫_{1 Ryd} (L_ν / hν) dν.
        """
        energy = self._integrate(band_ev[0], band_ev[1], power=0)
        return energy / self.photon_rate_per_unit() * u.erg

    def mean_ionizing_photon_energy(self) -> u.Quantity:
        """⟨hν⟩ of hydrogen-ionizing photons (L(>1 Ryd) / Q(H))."""
        return self.band_luminosity_per_q((RYDBERG_EV, CLOUDY_MAX_RYD * RYDBERG_EV)).to(
            u.eV
        )


def read_cloudy_sed(path: str | Path) -> SED:
    """Read a Cloudy ``table SED`` file, interpreting it exactly as Cloudy does.

    The first column is in Rydberg unless a ``units`` keyword says otherwise;
    the second is f_ν unless the ``nuFnu`` keyword is given (keywords may
    appear on any line and apply to the whole table); ``#`` starts a comment
    anywhere; reading stops at ``***``. Values are interpolated linearly in
    log-log space.
    """
    path = Path(path)
    text = path.read_text()
    energies, values = [], []
    unit_kind, unit_scale, nufnu = "energy", RYDBERG_EV, False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()  # comments may appear anywhere
        if not line:
            continue
        if line.startswith("***"):
            break
        tokens = line.split()
        try:
            x, y = float(tokens[0]), float(tokens[1])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse SED line in {path}: {raw!r}") from None
        # Keywords may appear on any line and apply to the whole table (Cloudy data/SED/README.md).
        words = [w.lower() for w in tokens[2:]]
        if "nufnu" in words or "nfn" in words:
            nufnu = True
        if "units" in words:
            i = words.index("units")
            name = words[i + 1] if i + 1 < len(words) else ""
            if name not in _ENERGY_UNITS:
                raise ValueError(f"Unsupported SED units {name!r} in {path}")
            unit_kind, unit_scale = _ENERGY_UNITS[name]
        energies.append(x)
        values.append(y)
    if len(energies) < 2:
        raise ValueError(f"No SED data in {path}")
    x = np.array(energies)
    y = np.array(values)
    if unit_kind == "energy":
        e_ev = x * unit_scale
    elif unit_kind == "frequency":
        e_ev = (x * u.Hz).to_value(u.eV, u.spectral())
    else:
        e_ev = (x * unit_scale * u.m).to_value(u.eV, u.spectral())
    nu = (e_ev * u.eV).to_value(u.Hz, u.spectral())
    fnu = y / nu if nufnu else y
    return SED(e_ev, fnu, str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def log_q_from_u(log_u, log_nh, radius: u.Quantity) -> np.ndarray:
    """log10 Q(H) [s^-1] = log10(4π r² n_H c U)."""
    r_cm = radius.to_value(u.cm)
    return (
        np.log10(4 * math.pi * r_cm**2 * const.c.cgs.value)
        + np.asarray(log_nh)
        + np.asarray(log_u)
    )


def write_cloudy_sed(
    path: str | os.PathLike,
    energy_ev: np.ndarray,
    fnu: np.ndarray,
    *,
    comments: Iterable[str] = (),
    floor_fraction: float = 1e-30,
) -> Path:
    """Write a Cloudy ``table SED`` file (energies in eV, f_ν).

    Zero or negative fluxes are raised to ``floor_fraction`` times the peak,
    because Cloudy interpolates in log space. Each comment becomes a ``#``
    line at the top of the file.
    """
    e = np.asarray(energy_ev, float)
    f = np.asarray(fnu, float)
    order = np.argsort(e)
    e, f = e[order], f[order]
    if len(e) < 2 or np.any(np.diff(e) <= 0):
        raise ValueError("SED energies must be distinct and at least two")
    peak = f[f > 0].max() if np.any(f > 0) else None
    if peak is None:
        raise ValueError("SED has no positive flux")
    f = np.maximum(f, floor_fraction * peak)
    lines = [f"# {c}" for comment in comments for c in str(comment).splitlines()]
    lines.append(f"{e[0]:.8e} {f[0]:.8e} units ev")
    lines += [f"{x:.8e} {y:.8e}" for x, y in zip(e[1:], f[1:])]
    lines.append("***")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path
