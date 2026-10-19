from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from ..lines import LINES, EmissionLine, get_line, species_from_label
from ..maps import EmissionLineMaps, SourceMetadata
from .loader import Loader

__all__ = [
    "AmbiguousLineError",
    "EXTENSIONS",
    "LineNotFoundError",
    "Pipe3DLoader",
    "plane_catalog",
]

# Emission-line extensions that share the NAME/WAVE/UNIT plane layout.
EXTENSIONS = ("FLUX_ELINES", "FLUX_ELINES_LONG")


class LineNotFoundError(LookupError):
    """A requested line has no matching plane in the product."""


class AmbiguousLineError(LookupError):
    """More than one plane matches a requested line."""


def parse_unit(text: str) -> u.UnitBase:
    """Parse a pyPipe3D ``UNIT<i>`` value such as ``"10^-16 erg/s/cm^2"``."""
    text = text.strip()
    scale = 1.0
    match = re.match(r"^10\^\(?([+-]?\d+)\)?\s*(.*)$", text)
    if match:
        scale = 10.0 ** int(match.group(1))
        text = match.group(2)
    # "erg/s/cm^2" means erg / (s cm^2); parse each factor separately to avoid
    # astropy's warning about chained slashes.
    numerator, *denominators = (
        part.strip().replace("^", "") for part in text.split("/")
    )
    unit = u.Unit(numerator)
    for part in denominators:
        unit = unit / u.Unit(part)
    return u.CompositeUnit(scale, [unit], [1]) if scale != 1.0 else unit


# Planes are always located from their NAME/WAVE keywords, never by fixed index:
# pyPipe3D labels are not unique ("[SII]" appears four times in FLUX_ELINES) and
# differ between the extensions ("Hb" vs "Hbeta"), so lines are matched by
# wavelength, with the label as a cross-check (`resolve_plane`).
def plane_catalog(source: str | Path | fits.HDUList) -> Table:
    """Describe every plane of the emission-line extensions.

    Returns
    -------
    Table
        One row per plane with columns ``extension``, ``plane``,
        ``quantity`` (e.g. ``flux``, ``e_flux``), ``label`` (pyPipe3D's
        free-text line label), ``wavelength`` (Angstrom), and ``unit``.
    """
    if not isinstance(source, fits.HDUList):
        with fits.open(source) as hdul:
            return plane_catalog(hdul)

    rows = []
    for ext in EXTENSIONS:
        if ext not in source:
            continue
        header = source[ext].header
        for i in range(header["NAXIS3"]):
            try:
                name = str(header[f"NAME{i}"]).strip()
                wave = float(header[f"WAVE{i}"])
                unit = str(header.get(f"UNIT{i}", "")).strip()
            except KeyError as exc:
                raise ValueError(
                    f"{ext} plane {i} has no {exc.args[0]} keyword; not a pyPipe3D product?"
                ) from None
            quantity, _, label = name.partition(" ")
            rows.append((ext, i, quantity, label.strip(), wave, unit))
    if not rows:
        raise ValueError(
            f"None of the extensions {EXTENSIONS} were found; not a pyPipe3D product?"
        )
    table = Table(
        rows=rows,
        names=("extension", "plane", "quantity", "label", "wavelength", "unit"),
    )
    table["wavelength"].unit = u.AA
    return table


def resolve_plane(
    catalog: Table,
    line: str | EmissionLine,
    quantity: str,
    extension: str,
    tolerance: u.Quantity = 1.0 * u.AA,
) -> int:
    """Find the plane holding ``quantity`` (e.g. ``"flux"``) for ``line``.

    A plane matches when its header wavelength is within ``tolerance`` of the
    line's rest wavelength. When several planes match, the one whose label
    names the same species is chosen; if that does not single one out, or if
    the only match is labeled with a different species, an error is raised.

    Raises
    ------
    LineNotFoundError, AmbiguousLineError
    """
    line = get_line(line)
    tol = tolerance.to_value(u.AA)
    rows = catalog[
        (catalog["extension"] == extension) & (catalog["quantity"] == quantity)
    ]
    if len(rows) == 0:
        raise LineNotFoundError(f"No {quantity!r} planes in extension {extension!r}")
    offsets = np.abs(np.asarray(rows["wavelength"], dtype=float) - line.wavelength_aa)
    candidates = rows[offsets <= tol]

    def describe(rs):
        return ", ".join(
            f"plane {r['plane']} {r['label']!r} at {r['wavelength']:.2f} A" for r in rs
        )

    if len(candidates) == 0:
        nearest = rows[int(np.argmin(offsets))]
        raise LineNotFoundError(
            f"No {quantity} plane for {line.id} ({line.wavelength_aa:.2f} A) within {tol} A in {extension}; "
            f"nearest is {describe([nearest])}"
        )

    species = [species_from_label(str(label)) for label in candidates["label"]]
    if len(candidates) > 1:
        same = [i for i, s in enumerate(species) if s == line.species]
        if len(same) != 1:
            raise AmbiguousLineError(
                f"{len(candidates)} {quantity} planes in {extension} match {line.id}: {describe(candidates)}"
            )
        return int(candidates[same[0]]["plane"])

    if species[0] is not None and species[0] != line.species:
        raise LineNotFoundError(
            f"The only {quantity} plane near {line.id} in {extension} is labeled as {species[0]}: "
            f"{describe(candidates)}"
        )
    return int(candidates[0]["plane"])


def celestial_wcs(header: fits.Header) -> WCS:
    """Build the 2-D celestial WCS from the spatial keywords of a cube header.

    Only the spatial keywords are copied, so malformed spectral keywords (the
    SAMI products have ``CDELT3 = ''``) cannot break the parse.
    """
    clean = fits.Header()
    clean["NAXIS"] = 2
    for i in (1, 2):
        for key in (f"CTYPE{i}", f"CRPIX{i}", f"CRVAL{i}", f"CUNIT{i}"):
            if key in header:
                clean[key] = header[key]
    cd_keys = [f"CD{i}_{j}" for i in (1, 2) for j in (1, 2)]
    if any(key in header for key in cd_keys):
        for key in cd_keys:
            clean[key] = float(header.get(key, 0.0))
    else:
        for key in ("CDELT1", "CDELT2", "PC1_1", "PC1_2", "PC2_1", "PC2_2"):
            if key in header:
                clean[key] = float(header[key])
    for key in ("RADESYS", "EQUINOX"):
        if key in header:
            clean[key] = header[key]
    wcs = WCS(clean)
    if not wcs.has_celestial:
        raise ValueError("The cube header does not define a celestial WCS")
    return wcs


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _common_unit(units: list[str], quantity: str) -> u.UnitBase:
    if len(set(units)) != 1:
        raise ValueError(
            f"The selected {quantity} planes have inconsistent units: {sorted(set(units))}"
        )
    return parse_unit(units[0])


def header_coord(header: fits.Header, ra_key: str, dec_key: str) -> SkyCoord | None:
    """An ICRS position from two header keywords in degrees, if both are present."""
    if ra_key in header and dec_key in header:
        return SkyCoord(
            float(header[ra_key]) * u.deg, float(header[dec_key]) * u.deg, frame="icrs"
        )
    return None


class Pipe3DLoader(Loader):
    """Base loader for pyPipe3D cubes; subclasses implement `identify` and `metadata`.

    Parameters
    ----------
    extension : str
        Emission-line extension to read: ``"FLUX_ELINES_LONG"`` (default; 192
        lines with updated wavelengths) or ``"FLUX_ELINES"`` (56 lines).
    tolerance : Quantity
        Maximum offset between header and catalog wavelengths.

    Notes
    -----
    pyPipe3D produces the same layout for MaNGA (the SDSS DR17 value-added
    catalog) and KILOGAS (MaNGA- and SAMI-based cubes):

    * ``ORG_HDR`` (primary): header of the original survey cube, incl. the WCS.
    * ``SSP``: stellar-population maps; plane 0, the pseudo-V-band continuum,
      becomes ``maps.continuum``.
    * ``FLUX_ELINES`` (56 lines) and ``FLUX_ELINES_LONG`` (192 lines):
      emission lines measured by a weighted moment analysis, in 8 blocks of
      planes (flux, vel, disp, EW, e_flux, e_vel, e_disp, e_EW), each plane
      described by ``NAME<i>``, ``WAVE<i>``, and ``UNIT<i>`` keywords.
    * ``GAIA_MASK``: nonzero where a bright Gaia star is masked.
    * ``SELECT_REG``: 1 inside the field of view *and* with continuum S/N > 1.

    The ``masks`` of the maps are ``"coverage"`` (spaxels with any valid line
    measurement or nonzero continuum), ``"select_reg"``, and ``"gaia_star"``,
    none applied.
    """

    def __init__(
        self, extension: str = "FLUX_ELINES_LONG", tolerance: u.Quantity = 1.0 * u.AA
    ):
        if extension not in EXTENSIONS:
            raise ValueError(
                f"Unknown emission-line extension {extension!r}; expected one of {EXTENSIONS}"
            )
        self.extension = extension
        self.tolerance = u.Quantity(tolerance, u.AA)

    @staticmethod
    def header(path: Path) -> fits.Header:
        """The ``ORG_HDR`` (primary) header."""
        return fits.getheader(path, 0)

    def metadata(self, header: fits.Header, path: Path) -> dict[str, Any]:
        """`SourceMetadata` fields from the ``ORG_HDR`` header."""
        raise NotImplementedError

    def read(
        self, path: Path, lines: Iterable[str | EmissionLine] | None = None
    ) -> EmissionLineMaps:
        """Read the requested lines (by default every catalog line the product has).

        A requested line that cannot be found raises `LineNotFoundError`; with
        the default, missing lines are listed in ``unavailable``.
        """
        path = Path(path)
        explicit = lines is not None
        wanted = [get_line(line) for line in (lines if explicit else LINES.values())]
        if len({line.id for line in wanted}) != len(wanted):
            raise ValueError("Duplicate lines requested")

        with fits.open(path) as hdul:
            org_hdr = hdul[0].header
            catalog = plane_catalog(hdul)
            found, planes, provenance, unavailable = [], [], {}, {}
            for line in wanted:
                try:
                    flux_plane = resolve_plane(
                        catalog, line, "flux", self.extension, self.tolerance
                    )
                    error_plane = resolve_plane(
                        catalog, line, "e_flux", self.extension, self.tolerance
                    )
                except LineNotFoundError as exc:
                    if explicit:
                        raise
                    unavailable[line.id] = str(exc)
                    continue
                row = catalog[
                    (catalog["extension"] == self.extension)
                    & (catalog["plane"] == flux_plane)
                ][0]
                found.append(line)
                planes.append((flux_plane, error_plane))
                provenance[line.id] = {
                    "extension": self.extension,
                    "flux_plane": flux_plane,
                    "error_plane": error_plane,
                    "label": str(row["label"]),
                    "wavelength_aa": float(row["wavelength"]),
                }
            if not found:
                raise LineNotFoundError(
                    f"None of the requested lines were found in {path.name}"
                )

            rows = catalog[catalog["extension"] == self.extension]
            unit_of = dict(
                zip((int(p) for p in rows["plane"]), (str(t) for t in rows["unit"]))
            )
            flux_unit, error_unit = (
                _common_unit([unit_of[pair[k]] for pair in planes], name)
                for k, name in ((0, "flux"), (1, "error"))
            )
            data = hdul[self.extension].data
            flux = np.stack([np.asarray(data[f], dtype=float) for f, _ in planes])
            error = np.stack([np.asarray(data[e], dtype=float) for _, e in planes])

            with np.errstate(invalid="ignore"):
                coverage = np.any(
                    np.isfinite(flux) & np.isfinite(error) & (error > 0), axis=0
                )
            masks, continuum = {}, None
            if "SSP" in hdul:
                vband = np.asarray(hdul["SSP"].data[0], dtype=float)
                coverage |= np.isfinite(vband) & (vband != 0)
                unit_text = str(hdul["SSP"].header.get("UNITS_0", "")).strip()
                continuum = vband * (
                    parse_unit(unit_text) if unit_text else u.dimensionless_unscaled
                )
            masks["coverage"] = coverage
            if "SELECT_REG" in hdul:
                masks["select_reg"] = np.asarray(hdul["SELECT_REG"].data) > 0
            if "GAIA_MASK" in hdul:
                masks["gaia_star"] = np.asarray(hdul["GAIA_MASK"].data) != 0

            fields = self.metadata(org_hdr, path)
            software = dict(fields.pop("software", {}))
            if "PIPELINE" in hdul[self.extension].header:
                software[f"{self.extension} pipeline"] = str(
                    hdul[self.extension].header["PIPELINE"]
                ).strip()
            if "Pipe3D_VER" in org_hdr:
                software.setdefault("Pipe3D", str(org_hdr["Pipe3D_VER"]).strip())
            meta = SourceMetadata(
                software=software, path=str(path), sha256=file_sha256(path), **fields
            )
            wcs = celestial_wcs(org_hdr)

        return EmissionLineMaps(
            lines=tuple(found),
            flux=flux * flux_unit,
            error=error * error_unit,
            wcs=wcs,
            meta=meta,
            masks=masks,
            continuum=continuum,
            unavailable=unavailable,
            line_provenance=provenance,
        )
