from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import astropy.units as u
from astropy.io import fits

from ..catalogs import Catalog
from .loader import register_loader
from .pipe3d import Pipe3DLoader, header_coord

__all__ = ["KILOGASLoader", "MaNGALoader", "SAMILoader"]

_PIPE3D_NAME = re.compile(r"\.Pipe3D\.cube\.fits(\.gz)?$")
_KILOGAS_NAME = re.compile(r"^(KG-(MaNGA|SAMI)-[\w-]+?)\.Pipe3D\.cube\.fits(\.gz)?$")

MANGA_DR17 = Catalog(
    name="manga_dr17",
    source="https://data.sdss.org/sas/dr17/manga/spectro/pipe3d/v3_1_1/3.1.1/SDSS17Pipe3D_v3_1_1.fits",
    id_column="plateifu",
    redshift="nsa_redshift",
    centers={"catalog": ("objra", "objdec")},
)
KILOGAS_NSA = Catalog(
    name="kilogas_nsa",
    source="https://ifs.astroscu.unam.mx/KILOGAS/Pipe3D/tables/KG_NSA.fits",
    id_column="cubename",
    redshift="Z",
    centers={"catalog": ("RA", "DEC")},
)
KILOGAS_BARYCENTER = Catalog(
    name="kilogas_barycenter",
    source="https://ifs.astroscu.unam.mx/KILOGAS/Pipe3D/tables/PS_img_barycenter.fits",
    id_column="name",
    centers={
        "image_barycenter": ("RA_img", "DEC_img"),
        "cube_barycenter": ("RA_cube", "DEC_cube"),
    },
)


def _instrument(path: Path) -> str:
    """``INSTRUME`` of a pyPipe3D cube's ``ORG_HDR``, or "" if the file is not one."""
    if not _PIPE3D_NAME.search(path.name):
        return ""
    try:
        return str(Pipe3DLoader.header(path).get("INSTRUME", "")).strip().upper()
    except OSError:
        return ""


def _flag(value: Any) -> bool | None:
    """A header boolean that may be stored as a string ('True')."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return (
        True
        if text in ("true", "t", "1", "yes")
        else False if text in ("false", "f", "0", "no") else None
    )


def _keep(header: fits.Header, keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: header[key] for key in keys if key in header}


def manga_metadata(header: fits.Header) -> dict[str, Any]:
    plateifu = str(header.get("PLATEIFU", "")).strip()
    aliases = {
        "plateifu": plateifu,
        **({"mangaid": str(header["MANGAID"]).strip()} if "MANGAID" in header else {}),
    }
    return {
        "survey": "MaNGA",
        "instrument": "MaNGA",
        "target_id": plateifu,
        "aliases": aliases,
        "object_coord": header_coord(header, "OBJRA", "OBJDEC"),
        "pointing_coord": header_coord(header, "IFURA", "IFUDEC"),
        "psf_fwhm": float(header["RFWHM"]) * u.arcsec if "RFWHM" in header else None,
        "galactic_extinction_corrected": (
            _flag(header["GALEXTC"]) if "GALEXTC" in header else None
        ),
        "software": (
            {"MaNGA DRP": str(header["VERSDRP3"]).strip()}
            if "VERSDRP3" in header
            else {}
        ),
        "extra": _keep(
            header,
            (
                "DRP3QUAL",
                "EBVGAL",
                "GFWHM",
                "RFWHM",
                "IFWHM",
                "ZFWHM",
                "MJDMED",
                "EXPTIME",
                "BUNIT",
            ),
        ),
    }


def sami_metadata(header: fits.Header) -> dict[str, Any]:
    catid = str(header.get("NAME", "")).strip()
    return {
        "survey": "SAMI",
        "instrument": "SAMI",
        "target_id": catid,
        "aliases": {"sami_catid": catid},
        "object_coord": header_coord(header, "CATARA", "CATADEC"),
        "pointing_coord": header_coord(header, "CRVAL1", "CRVAL2"),
        "redshift": float(header["Z_SPEC"]) if "Z_SPEC" in header else None,
        "psf_fwhm": (
            float(header["PSFFWHM"]) * u.arcsec if "PSFFWHM" in header else None
        ),
        "software": (
            {"SAMI cubing": str(header["HGCUBING"]).strip()}
            if "HGCUBING" in header
            else {}
        ),
        "extra": _keep(
            header,
            (
                "WCS_SRC",
                "PSFFWHM",
                "PSFALPHA",
                "PSFBETA",
                "TOTALEXP",
                "PLATEID",
                "IFUPROBE",
                "BUNIT",
            ),
        ),
    }


@register_loader
class MaNGALoader(Pipe3DLoader):
    """SDSS DR17 MaNGA Pipe3D cubes, ``manga-<plate>-<ifu>.Pipe3D.cube.fits.gz``.

    ``ORG_HDR`` is the MaNGA DRP cube header, which has no redshift; it comes
    from the DR17 Pipe3D summary table. The target id is the plate-IFU. See
    `Pipe3DLoader` for options.
    """

    name = "manga"
    catalogs = (MANGA_DR17,)

    @classmethod
    def identify(cls, path: Path) -> bool:
        return not path.name.startswith("KG-") and _instrument(path).startswith("MANGA")

    def metadata(self, header: fits.Header, path: Path) -> dict[str, Any]:
        return manga_metadata(header)


@register_loader
class SAMILoader(Pipe3DLoader):
    """SAMI-based pyPipe3D cubes.

    ``ORG_HDR`` is the header of the SAMI (blue-arm) cube, with the redshift in
    ``Z_SPEC``. Its ``CDELT3`` is an empty string (only the spatial WCS
    keywords are used), the WCS is the nominal catalog position
    (``WCS_SRC = 'Nominal'``), and it does not state whether Galactic
    extinction was corrected. The target id is the SAMI CATID. See
    `Pipe3DLoader` for options.
    """

    name = "sami"

    @classmethod
    def identify(cls, path: Path) -> bool:
        return not path.name.startswith("KG-") and "SAMI" in _instrument(path)

    def metadata(self, header: fits.Header, path: Path) -> dict[str, Any]:
        return sami_metadata(header)


@register_loader
class KILOGASLoader(Pipe3DLoader):
    """KILOGAS pyPipe3D cubes, ``KG-MaNGA-<mangaid>`` and ``KG-SAMI-<catid>``.

    From https://ifs.astroscu.unam.mx/KILOGAS/Pipe3D/dataproducts/. The files
    are MaNGA or SAMI products; a KILOGAS-MaNGA header is identical to a DR17
    one, so only the file name tells them apart. Redshifts and centers come
    from the KILOGAS tables. The target id is the cube name, e.g.
    ``KG-SAMI-396833``; the MaNGA or SAMI identifiers are kept in
    ``aliases``. See `Pipe3DLoader` for options.
    """

    name = "kilogas"
    catalogs = (KILOGAS_NSA, KILOGAS_BARYCENTER)

    @classmethod
    def identify(cls, path: Path) -> bool:
        return bool(_KILOGAS_NAME.match(path.name))

    def metadata(self, header: fits.Header, path: Path) -> dict[str, Any]:
        instrument = str(header.get("INSTRUME", "")).strip().upper()
        if instrument.startswith("MANGA"):
            fields, cubename = (
                manga_metadata(header),
                f"KG-MaNGA-{str(header.get('MANGAID', '')).strip()}",
            )
        elif "SAMI" in instrument:
            fields, cubename = (
                sami_metadata(header),
                f"KG-SAMI-{str(header.get('NAME', '')).strip()}",
            )
        else:
            raise ValueError(
                f"{path.name} is neither a MaNGA- nor a SAMI-based product (INSTRUME={instrument!r})"
            )
        match = _KILOGAS_NAME.match(path.name)
        if match and match.group(1) != cubename:
            raise ValueError(
                f"File name {path.name!r} does not match the header identifiers ({cubename})"
            )
        return {
            **fields,
            "survey": "KILOGAS",
            "target_id": cubename,
            "aliases": {**fields["aliases"], "kilogas_cubename": cubename},
        }
