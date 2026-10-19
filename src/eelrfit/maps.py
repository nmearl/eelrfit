from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .lines import EmissionLine, get_line

__all__ = ["EmissionLineMaps", "GalaxyProperties", "SourceMetadata"]


def _coord_dict(c: SkyCoord | None) -> dict[str, float] | None:
    return (
        None if c is None else {"ra_deg": float(c.ra.deg), "dec_deg": float(c.dec.deg)}
    )


@dataclass(frozen=True)
class SourceMetadata:
    """Where a set of maps came from, recorded for provenance.

    Attributes
    ----------
    survey : str
        Survey or program the product belongs to, e.g. ``"MaNGA"``, ``"KILOGAS"``.
    instrument : str
        Instrument that produced the cube, e.g. ``"MaNGA"``, ``"SAMI"``.
    target_id : str
        Identifier of the target within the survey; catalogs are matched on it.
    aliases : mapping
        Other identifiers for the same target, keyed by naming scheme.
    object_coord : SkyCoord or None
        Position of the target galaxy recorded in the header (not necessarily
        the center that defines radii).
    pointing_coord : SkyCoord or None
        Nominal IFU pointing (the WCS reference position).
    redshift : float or None
        Redshift recorded in the header, if any.
    psf_fwhm : Quantity or None
        PSF FWHM recorded in the header.
    galactic_extinction_corrected : bool or None
        Whether the product states that Galactic extinction was corrected
        before the lines were measured; ``None`` if it does not say.
    software : mapping
        Versions of the software that made the product.
    path, sha256 : str
        The file read and its checksum.
    extra : mapping
        Other header values worth keeping (quality flags, ...).
    """

    survey: str
    instrument: str
    target_id: str
    aliases: Mapping[str, str] = field(default_factory=dict)
    object_coord: SkyCoord | None = None
    pointing_coord: SkyCoord | None = None
    redshift: float | None = None
    psf_fwhm: u.Quantity | None = None
    galactic_extinction_corrected: bool | None = None
    software: Mapping[str, str] = field(default_factory=dict)
    path: str = ""
    sha256: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable representation."""
        return {
            "survey": self.survey,
            "instrument": self.instrument,
            "target_id": self.target_id,
            "aliases": dict(self.aliases),
            "object_coord": _coord_dict(self.object_coord),
            "pointing_coord": _coord_dict(self.pointing_coord),
            "redshift": self.redshift,
            "psf_fwhm_arcsec": (
                None
                if self.psf_fwhm is None
                else float(self.psf_fwhm.to_value(u.arcsec))
            ),
            "galactic_extinction_corrected": self.galactic_extinction_corrected,
            "software": dict(self.software),
            "path": self.path,
            "sha256": self.sha256,
            "extra": {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in self.extra.items()
            },
        }


@dataclass(frozen=True)
class GalaxyProperties:
    """Galaxy-level values the analysis adopts, each with its source.

    Distances are not stored: they follow from the redshift and a cosmology
    when radii are converted to kpc.

    Attributes
    ----------
    redshift : float or None
    centers : mapping of str to SkyCoord
        Candidate galaxy centers by origin (e.g. ``"catalog"``,
        ``"image_barycenter"``, ``"header_object"``). Which one defines radii
        is an analysis choice made in `eelrfit.regions`.
    sources : mapping of str to str
        Where each value came from (``"<catalog>:<column>"``, ``"header"``,
        ...). Values given without a source are recorded as ``"user"``.
    notes : tuple of str
        Disagreements between sources.
    catalogs : mapping of str to mapping
        Catalog files used: source, local path and SHA-256 checksum.
    """

    redshift: float | None = None
    centers: Mapping[str, SkyCoord] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    catalogs: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        if self.redshift is not None and not (
            np.isfinite(self.redshift) and self.redshift >= 0
        ):
            raise ValueError(
                f"redshift must be finite and non-negative; got {self.redshift}"
            )
        sources = dict(self.sources)
        if self.redshift is not None:
            sources.setdefault("redshift", "user")
        for name in self.centers:
            sources.setdefault(f"centers.{name}", "user")
        object.__setattr__(self, "centers", MappingProxyType(dict(self.centers)))
        object.__setattr__(self, "sources", MappingProxyType(sources))
        object.__setattr__(
            self,
            "catalogs",
            MappingProxyType({k: dict(v) for k, v in self.catalogs.items()}),
        )
        object.__setattr__(self, "notes", tuple(self.notes))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable representation."""
        return {
            "redshift": self.redshift,
            "centers": {k: _coord_dict(c) for k, c in self.centers.items()},
            "sources": dict(self.sources),
            "notes": list(self.notes),
            "catalogs": {k: dict(v) for k, v in self.catalogs.items()},
        }


@dataclass(frozen=True, eq=False)
class EmissionLineMaps:
    """Flux and uncertainty maps for a set of emission lines on one spatial grid.

    The interface between data and analysis: loaders produce it, and regions,
    fitting and plotting consume only it, so data from any source can be
    analyzed by constructing one from arrays.

    Fluxes are stored as delivered by the input product. A measurement is
    *valid* where both flux and error are finite and the error is positive.
    Negative fluxes are kept: they are legitimate noisy measurements of faint
    lines, and discarding them biases averages high.

    Attributes
    ----------
    lines : tuple of EmissionLine
        Lines present, in the order of the first axis of `flux` and `error`.
    flux, error : Quantity
        Arrays of shape ``(n_lines, ny, nx)``; stored read-only.
    wcs : WCS
        Two-dimensional celestial WCS of the spatial axes.
    meta : SourceMetadata
        Provenance of the product.
    masks : mapping of str to ndarray
        Boolean ``(ny, nx)`` masks, documented by the loader that made them;
        none has been applied to `flux`.
    continuum : Quantity or None
        A ``(ny, nx)`` continuum map, used to locate the nucleus.
    properties : GalaxyProperties or None
        Adopted galaxy properties (redshift, candidate centers).
    unavailable : mapping of str to str
        Lines that were looked for but not found, with the reason.
    line_provenance : mapping of str to mapping
        For each line id, where in the product it was read from.
    """

    lines: tuple[EmissionLine, ...]
    flux: u.Quantity
    error: u.Quantity
    wcs: WCS
    meta: SourceMetadata
    masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    continuum: u.Quantity | None = None
    properties: GalaxyProperties | None = None
    unavailable: Mapping[str, str] = field(default_factory=dict)
    line_provenance: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        lines = tuple(get_line(line) for line in self.lines)
        n = len(lines)
        if len({line.id for line in lines}) != n:
            raise ValueError("lines must be unique")
        if self.flux.ndim != 3 or self.flux.shape[0] != n:
            raise ValueError(
                f"flux must have shape (n_lines={n}, ny, nx); got {self.flux.shape}"
            )
        if self.error.shape != self.flux.shape:
            raise ValueError(
                f"error shape {self.error.shape} does not match flux shape {self.flux.shape}"
            )
        if not self.error.unit.is_equivalent(self.flux.unit):
            raise u.UnitsError(
                f"error unit {self.error.unit} is not compatible with flux unit {self.flux.unit}"
            )
        shape = self.flux.shape[1:]
        if self.continuum is not None and u.Quantity(self.continuum).shape != shape:
            raise ValueError(
                f"continuum must have shape {shape}; got {u.Quantity(self.continuum).shape}"
            )
        for name, mask in self.masks.items():
            if np.shape(mask) != shape or np.asarray(mask).dtype != bool:
                raise ValueError(
                    f"mask {name!r} must be a boolean array of shape {shape}"
                )

        # Read-only copies, so no downstream step can modify the input data in place.
        flux = u.Quantity(self.flux, copy=True)
        error = u.Quantity(self.error.to(self.flux.unit), copy=True)
        masks = {name: np.array(mask, copy=True) for name, mask in self.masks.items()}
        for array in (flux, error, *masks.values()):
            array.flags.writeable = False
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "masks", MappingProxyType(masks))
        object.__setattr__(
            self, "unavailable", MappingProxyType(dict(self.unavailable))
        )
        object.__setattr__(
            self, "line_provenance", MappingProxyType(dict(self.line_provenance))
        )
        if self.continuum is not None:
            continuum = u.Quantity(self.continuum, copy=True)
            continuum.flags.writeable = False
            object.__setattr__(self, "continuum", continuum)

    def with_properties(self, properties: GalaxyProperties | None) -> EmissionLineMaps:
        """A copy with different galaxy properties attached."""
        return dataclasses.replace(self, properties=properties)

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial shape ``(ny, nx)``."""
        return self.flux.shape[1:]

    @property
    def line_ids(self) -> tuple[str, ...]:
        return tuple(line.id for line in self.lines)

    @property
    def pixel_scale(self) -> u.Quantity:
        """Angular size of a spaxel along each spatial axis ``(x, y)``."""
        scales = proj_plane_pixel_scales(self.wcs) * u.Unit(
            self.wcs.wcs.cunit[0] or "deg"
        )
        return scales.to(u.arcsec)

    @property
    def spaxel_area(self) -> u.Quantity:
        scale = self.pixel_scale
        return (scale[0] * scale[1]).to(u.arcsec**2)

    def __contains__(self, line: str | EmissionLine) -> bool:
        try:
            return get_line(line).id in self.line_ids
        except KeyError:
            return False

    def index(self, line: str | EmissionLine) -> int:
        """Position of a line along the first axis of `flux`."""
        line = get_line(line)
        try:
            return self.line_ids.index(line.id)
        except ValueError:
            reason = self.unavailable.get(line.id, "it was not requested")
            raise KeyError(f"{line.id} is not in these maps: {reason}") from None

    def flux_map(self, line: str | EmissionLine) -> u.Quantity:
        return self.flux[self.index(line)]

    def error_map(self, line: str | EmissionLine) -> u.Quantity:
        return self.error[self.index(line)]

    def valid(self, line: str | EmissionLine) -> np.ndarray:
        """Spaxels with a usable measurement: finite flux, finite positive error."""
        i = self.index(line)
        flux, error = self.flux.value[i], self.error.value[i]
        with np.errstate(invalid="ignore"):
            return np.isfinite(flux) & np.isfinite(error) & (error > 0)

    def snr(self, line: str | EmissionLine) -> np.ndarray:
        """Signal-to-noise ratio, NaN where the measurement is not valid."""
        i = self.index(line)
        valid = self.valid(line)
        out = np.full(self.shape, np.nan)
        out[valid] = self.flux.value[i][valid] / self.error.value[i][valid]
        return out

    def __repr__(self) -> str:
        return f"<EmissionLineMaps {self.meta.survey}:{self.meta.target_id} shape={self.shape} lines=[{', '.join(self.line_ids)}]>"
