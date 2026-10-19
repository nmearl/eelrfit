from __future__ import annotations

import functools
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import download_file

from .maps import GalaxyProperties, SourceMetadata

__all__ = ["Catalog", "lookup", "merge_header"]

# Header/catalog disagreements smaller than these are not noted.
_REDSHIFT_TOLERANCE = 1e-4
_CENTER_TOLERANCE = 0.5 * u.arcsec


@dataclass(frozen=True)
class Catalog:
    """A FITS table of galaxy properties, matched on a target identifier.

    Loaders list the catalogs their targets are in (`eelrfit.io.Loader.catalogs`),
    and `eelrfit.io.read` looks them up when the data are read. Remote tables
    are downloaded once and cached by astropy.

    Parameters
    ----------
    name : str
        Short name, used in the provenance of the values read.
    source : str
        URL (downloaded once and cached) or local path of the table.
    id_column : str
        Column matched against `SourceMetadata.target_id`.
    redshift : str, optional
        Column holding the redshift.
    centers : mapping of str to (str, str)
        Candidate centers: name -> (RA column, Dec column), in degrees.
    hdu : int
        HDU holding the table.
    """

    name: str
    source: str
    id_column: str
    redshift: str | None = None
    centers: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    hdu: int = 1

    def path(self, cache: bool | str = True) -> Path:
        """Local path of the table, downloading it if needed (``cache="update"`` re-downloads)."""
        if "://" in self.source:
            return Path(download_file(self.source, cache=cache, pkgname="eelrfit"))
        path = Path(self.source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Catalog {self.name!r} not found: {path}")
        return path

    def lookup(
        self, target_id: str, cache: bool | str = True
    ) -> GalaxyProperties | None:
        """Properties of ``target_id``, or None if the table has no row for it."""
        path = self.path(cache)
        columns = [self.redshift] if self.redshift else []
        columns += [c for pair in self.centers.values() for c in pair]
        row = _row(path, self.hdu, self.id_column, str(target_id), tuple(columns))
        if row is None:
            return None
        redshift, centers, sources = None, {}, {}
        if self.redshift and _finite(row[self.redshift]) and row[self.redshift] >= 0:
            redshift = float(row[self.redshift])
            sources["redshift"] = f"{self.name}:{self.redshift}"
        for key, (ra, dec) in self.centers.items():
            if _finite(row[ra]) and _finite(row[dec]):
                centers[key] = SkyCoord(float(row[ra]) * u.deg, float(row[dec]) * u.deg)
                sources[f"centers.{key}"] = f"{self.name}:{ra},{dec}"
        stat = path.stat()
        provenance = {
            "source": self.source,
            "path": str(path),
            "sha256": _sha256(str(path), stat.st_mtime, stat.st_size),
        }
        return GalaxyProperties(
            redshift=redshift,
            centers=centers,
            sources=sources,
            catalogs={self.name: provenance},
        )


def lookup(
    catalogs: Iterable[Catalog], target_id: str, cache: bool | str = True
) -> GalaxyProperties:
    """Merge the properties of ``target_id`` from several catalogs; the first value found wins.

    Raises
    ------
    LookupError
        If none of the catalogs has the target.
    """
    catalogs = tuple(catalogs)
    found = [(c, p) for c in catalogs if (p := c.lookup(target_id, cache)) is not None]
    if catalogs and not found:
        raise LookupError(
            f"{target_id!r} is in none of the catalogs {[c.name for c in catalogs]}"
        )
    redshift, centers, sources, provenance = None, {}, {}, {}
    for _, props in found:
        if redshift is None and props.redshift is not None:
            redshift = props.redshift
            sources["redshift"] = props.sources["redshift"]
        for key, coord in props.centers.items():
            if key not in centers:
                centers[key] = coord
                sources[f"centers.{key}"] = props.sources[f"centers.{key}"]
        provenance.update(props.catalogs)
    return GalaxyProperties(
        redshift=redshift, centers=centers, sources=sources, catalogs=provenance
    )


def merge_header(props: GalaxyProperties, meta: SourceMetadata) -> GalaxyProperties:
    """Fill gaps in ``props`` from the product header and note disagreements.

    The header redshift is used only if ``props`` has none; header positions
    are added as the centers ``"header_object"`` and ``"header_pointing"``.
    """
    redshift, centers = props.redshift, dict(props.centers)
    sources, notes = dict(props.sources), list(props.notes)
    if meta.redshift is not None:
        if redshift is None:
            redshift = meta.redshift
            sources["redshift"] = "header"
        elif abs(redshift - meta.redshift) > _REDSHIFT_TOLERANCE:
            notes.append(
                f"header redshift {meta.redshift:.5f} differs from the adopted {redshift:.5f} ({sources['redshift']})"
            )
    for key, coord in (
        ("header_object", meta.object_coord),
        ("header_pointing", meta.pointing_coord),
    ):
        if coord is not None and key not in centers:
            centers[key] = coord
            sources[f"centers.{key}"] = "header"
    if meta.object_coord is not None and "catalog" in props.centers:
        sep = meta.object_coord.separation(props.centers["catalog"]).to(u.arcsec)
        if sep > _CENTER_TOLERANCE:
            notes.append(
                f"header object position is {sep:.2f} from the catalog position"
            )
    return GalaxyProperties(
        redshift=redshift,
        centers=centers,
        sources=sources,
        notes=tuple(notes),
        catalogs=props.catalogs,
    )


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


@functools.lru_cache(maxsize=16)
def _sha256(path: str, mtime: float, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=16)
def _columns(
    path: str, mtime: float, hdu: int, columns: tuple[str, ...]
) -> dict[str, np.ndarray]:
    # Cached on the file's mtime: large tables are read once per session.
    with fits.open(path, memmap=True) as hdul:
        data = hdul[hdu].data
        missing = [c for c in columns if c not in data.names]
        if missing:
            raise KeyError(f"{Path(path).name} has no column(s) {missing}")
        out = {}
        for c in columns:
            values = np.array(data[c])
            out[c] = (
                np.char.strip(values.astype(str))
                if values.dtype.kind in "SU"
                else values
            )
        return out


def _row(
    path: Path, hdu: int, id_column: str, target: str, columns: tuple[str, ...]
) -> dict[str, Any] | None:
    cols = _columns(
        str(path),
        path.stat().st_mtime,
        hdu,
        tuple(dict.fromkeys((id_column, *columns))),
    )
    match = np.flatnonzero(cols[id_column] == target)
    if len(match) == 0:
        return None
    if len(match) > 1:
        raise ValueError(f"{len(match)} rows for {target!r} in {path.name}")
    return {c: cols[c][match[0]] for c in columns}
