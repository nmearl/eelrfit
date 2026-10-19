# Synthetic pyPipe3D products reproduce the layout of the real files (extension
# structure, header conventions and known quirks: duplicate labels, an empty plane,
# SAMI's malformed CDELT3) in a few kB. Tests marked `example_data` use the real
# cubes, looked for in $EELRFIT_EXAMPLE_DATA or example_data/.
from __future__ import annotations

import os
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.io import fits
from astropy.table import QTable
from astropy.wcs import WCS
from dust_extinction.parameter_averages import O94

from eelrfit.cloudy import CloudyGrid
from eelrfit.lines import get_line
from eelrfit.maps import EmissionLineMaps, SourceMetadata

EXAMPLE_DATA = Path(os.environ.get("EELRFIT_EXAMPLE_DATA", Path(__file__).parents[1] / "example_data"))


def example_file(name: str) -> Path:
    """A real cube from the example data, or skip the test."""
    path = EXAMPLE_DATA / name
    if not path.exists():
        pytest.skip(f"{name} not available")
    return path


QUANTITIES = ("flux", "vel", "disp", "EW", "e_flux", "e_vel", "e_disp", "e_EW")
UNITS = {
    "flux": "10^-16 erg/s/cm^2",
    "e_flux": "10^-16 erg/s/cm^2",
    "vel": "km/s",
    "e_vel": "km/s",
    "disp": "km/s",
    "e_disp": "km/s",
    "EW": "Angstrom",
    "e_EW": "Angstrom",
}

# (label, wavelength) pairs copied from the real headers, in a shortened list.
SHORT_LINES = [
    ("[OII]3727", 3727.4),
    ("He", 3970.1),  # pyPipe3D's label for H-epsilon
    ("[SII]", 4069.17),
    ("[SII]", 4076.72),
    ("Hd", 4101.74),
    ("[OIII]5007", 5006.84),
    ("[OIII]4959", 4958.91),
    ("Hb", 4861.32),
    ("Ha", 6562.68),
    ("[NII]6584", 6583.41),
    ("[NII]6548", 6548.08),
    ("[SII]6717", 6716.39),
    ("[SII]6731", 6730.74),
    ("", 0.0),  # the real FLUX_ELINES ends with an empty plane
]
LONG_LINES = [
    ("[OII]", 3726.03),
    ("[OII]", 3728.82),
    ("HeI", 3888.65),
    ("HI", 3889.05),
    ("Hdelta", 4101.77),
    ("Hbeta", 4861.36),
    ("[OIII]", 4958.91),
    ("[OIII]", 5006.84),
    ("[NII]", 6548.05),
    ("Halpha", 6562.85),
    ("[NII]", 6583.45),
    ("[SII]", 6716.44),
    ("[SII]", 6730.82),
    ("[ArIII]", 7135.8),
]

SHAPE = (5, 6)  # (ny, nx)


def line_value(extension: str, j: int) -> float:
    """Flux assigned to every in-footprint spaxel of line j in an extension."""
    return (j + 1) * (10.0 if extension == "FLUX_ELINES_LONG" else 1.0)


def footprint(shape=SHAPE) -> np.ndarray:
    """Spaxels with data; the rest are zero, as in the real products."""
    mask = np.ones(shape, dtype=bool)
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    return mask


def _elines_hdu(extension: str, lines, shape) -> fits.ImageHDU:
    n = len(lines)
    data = np.zeros((len(QUANTITIES) * n, *shape))
    header = fits.Header()
    header["PIPELINE"] = "pyFIT3D v1.0.0"
    inside = footprint(shape)
    for q, quantity in enumerate(QUANTITIES):
        for j, (label, wave) in enumerate(lines):
            i = q * n + j
            header[f"NAME{i}"] = f"{quantity} {label}"
            header[f"WAVE{i}"] = f"{wave}"
            header[f"UNIT{i}"] = UNITS[quantity]
            if quantity == "flux":
                data[i][inside] = line_value(extension, j)
            elif quantity == "e_flux":
                data[i][inside] = 0.1 * line_value(extension, j)
    # A noisy faint measurement: negative flux with a valid error.
    data[0, 1, 1] = -0.5
    return fits.ImageHDU(data=data, header=header, name=extension)


def _org_header(instrument: str) -> fits.Header:
    h = fits.Header()
    if instrument == "manga":
        h["INSTRUME"] = "MaNGA"
        h["PLATEIFU"] = "11759-3701"
        h["MANGAID"] = "1-207"
        h["OBJRA"], h["OBJDEC"] = 145.594030155, 0.309414123675
        h["IFURA"], h["IFUDEC"] = 145.59403, 0.3094141
        h["RFWHM"] = 2.59434
        h["DRP3QUAL"] = 4096
        h["EBVGAL"] = 0.101185
        h["VERSDRP3"] = "v3_1_1"
        h["CTYPE3"], h["CRPIX3"], h["CRVAL3"], h["CD3_3"] = "LINEAR", 1.0, 3622.0, 1.5
        h["CRPIX1"], h["CRPIX2"] = 3.0, 3.0
        h["CRVAL1"], h["CRVAL2"] = 145.59403, 0.3094141
        h["CD1_1"], h["CD2_2"] = -0.000138889, 0.000138889
        h["GALEXTC"] = "True"
    else:
        h["INSTRUME"] = "AAOMEGA-SAMI"
        h["NAME"] = "15847"
        h["CATARA"], h["CATADEC"] = 216.9662083333334, 0.75425
        h["Z_SPEC"] = 0.05705999955534935
        h["PSFFWHM"] = 2.100327120447356
        h["WCS_SRC"] = "Nominal"
        h["CRPIX1"], h["CRPIX2"] = 3.5, 3.0
        h["CRVAL1"], h["CRVAL2"] = 216.966208, 0.75425
        h["CDELT1"], h["CDELT2"] = -0.00013888888888889, 0.00013888888888889
        h["CTYPE3"], h["CRPIX3"], h["CRVAL3"] = "LINEAR", 1.0, 3720.0
        h["CD3_3"] = 1.0
        h["CDELT3"] = ""  # malformed in the real SAMI products
        h["CD1_1"], h["CD2_2"] = -0.000138889, 0.000138889
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
    h["RADESYS"], h["EQUINOX"] = "FK5", 2000.0
    h["HIERARCH Pipe3D_VER"] = "v3_1_1"
    return h


def write_pipe3d(path: Path, instrument: str = "manga", shape=SHAPE) -> Path:
    """Write a small synthetic pyPipe3D cube and return its path."""
    primary = fits.PrimaryHDU(header=_org_header(instrument))
    primary.header["EXTNAME"] = "ORG_HDR"
    ssp = np.zeros((21, *shape))
    ssp[0][footprint(shape)] = 1.0
    gaia = np.zeros(shape)
    gaia[2, 2] = 1.0
    select = footprint(shape).astype(float)
    select[1, 4] = 0.0  # continuum S/N < 1 but still inside the FoV
    hdul = fits.HDUList(
        [
            primary,
            fits.ImageHDU(ssp, name="SSP"),
            _elines_hdu("FLUX_ELINES", SHORT_LINES, shape),
            _elines_hdu("FLUX_ELINES_LONG", LONG_LINES, shape),
            fits.ImageHDU(gaia, name="GAIA_MASK"),
            fits.ImageHDU(select, name="SELECT_REG"),
        ]
    )
    hdul.writeto(path)
    return path


@pytest.fixture
def manga_file(tmp_path) -> Path:
    return write_pipe3d(tmp_path / "manga-11759-3701.Pipe3D.cube.fits.gz", "manga")


@pytest.fixture
def sami_file(tmp_path) -> Path:
    return write_pipe3d(tmp_path / "sami-15847.Pipe3D.cube.fits.gz", "sami")


@pytest.fixture
def kilogas_sami_file(tmp_path) -> Path:
    return write_pipe3d(tmp_path / "KG-SAMI-15847.Pipe3D.cube.fits.gz", "sami")


@pytest.fixture
def kilogas_manga_file(tmp_path) -> Path:
    return write_pipe3d(tmp_path / "KG-MaNGA-1-207.Pipe3D.cube.fits.gz", "manga")


# --- synthetic maps --------------------------------------------------------------------------

UNIT = 1e-16 * u.erg / u.s / u.cm**2
NY, NX = 31, 33
CX, CY = 16.3, 14.6  # true nucleus (0-based pixel coordinates)


def make_wcs() -> WCS:
    h = fits.Header()
    h["NAXIS"] = 2
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRPIX1"], h["CRPIX2"] = 17.0, 16.0
    h["CRVAL1"], h["CRVAL2"] = 150.0, 2.0
    h["CD1_1"], h["CD2_2"] = -0.5 / 3600, 0.5 / 3600
    h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
    return WCS(h)


def make_maps(flux=None, error=None, psf=2.0, properties=None, masks=None, lines=("HI_4861", "OIII_5007"), continuum=None):
    """Maps on a 31 x 33 grid of 0.5" spaxels with a Gaussian continuum nucleus at (CX, CY)."""
    yy, xx = np.indices((NY, NX))
    if continuum is None:
        continuum = 10 * np.exp(-((xx - CX) ** 2 + (yy - CY) ** 2) / (2 * 2.0**2))
    n = len(lines)
    flux = np.stack([np.full((NY, NX), 1.0 + k) for k in range(n)]) if flux is None else flux
    error = np.full((n, NY, NX), 0.1) if error is None else error
    wcs = make_wcs()
    return EmissionLineMaps(
        lines=lines,
        flux=flux * UNIT,
        error=error * UNIT,
        wcs=wcs,
        meta=SourceMetadata(survey="test", instrument="test", target_id="synthetic", psf_fwhm=psf * u.arcsec,
                            object_coord=wcs.pixel_to_world(CX + 0.4, CY)),  # 0.2" from the true nucleus
        masks={"coverage": np.ones((NY, NX), bool), **(masks or {})},
        continuum=None if continuum is False else continuum * UNIT,
        properties=properties,
    )


# --- a toy model grid ------------------------------------------------------------------------

LOG_U = np.round(np.arange(-4, -0.99, 0.25), 6)
LOG_NH = np.round(np.arange(0, 4.01, 0.5), 6)
TRUTH = {"log_U": -2.5, "log_nH": 2.5, "A_V": 0.5}
FIT_LINES = ("HI_6563", "HI_4861", "HI_4102", "OI_6300", "OIII_5007", "NII_6584", "SII_6717", "SII_6731")
AGN_SED = Path(__file__).parents[1] / "src" / "eelrfit" / "cloudy" / "specs" / "starecheski_agn.sed"


def model_intensities(log_u, log_nh):
    """Smooth toy photoionization model (relative to H-beta)."""
    x = log_u + 2.5
    s_total = 1.0 * 10 ** (-0.6 * x)
    ratio = 0.45 + 1.0 / (1 + 10 ** (log_nh - 2.8))  # [S II] 6717/6731: 1.45 (low nH) to 0.45 (high nH)
    return {
        "HI_4861": np.ones_like(x),
        "HI_6563": 2.80 - 0.02 * (log_nh - 2),
        "HI_4102": 0.26 + 0 * x,
        "OI_6300": 0.3 * 10 ** (-0.8 * x),
        "OIII_4959": 3.0 * 10 ** (1.2 * x) / 2.98,
        "OIII_5007": 3.0 * 10 ** (1.2 * x),
        "NII_6584": 1.5 * 10 ** (-0.5 * x),
        "SII_6717": s_total * ratio / (1 + ratio),
        "SII_6731": s_total / (1 + ratio),
    }


def make_grid(scale=None, n_scales=None) -> CloudyGrid:
    """The toy model on a log U x log nH grid; with ``n_scales``, an N/O axis ([N II] scales as 10**log_N_scale)."""
    U, N = np.meshgrid(LOG_U, LOG_NH, indexing="ij")
    m = model_intensities(U, N)
    lines = list(m)
    intensity = np.stack([m[l] for l in lines], axis=-1) * (1 if scale is None else scale[..., None])
    ds = xr.Dataset(
        {"intensity": (("log_U", "log_nH", "line"), intensity),
         "failed": (("log_U", "log_nH"), np.zeros(U.shape, bool)),
         "warnings": (("log_U", "log_nH"), np.zeros(U.shape, bool))},
        coords={"log_U": LOG_U, "log_nH": LOG_NH, "line": lines},
        attrs={"cloudy_version": "toy", "provenance": "{}"},
    )
    if n_scales is not None:
        n_axis = xr.DataArray(np.asarray(n_scales, float), coords={"log_N_scale": np.asarray(n_scales, float)})
        factor = xr.DataArray(np.where(ds["line"] == "NII_6584", 1.0, 0.0), coords={"line": ds["line"]})
        ds = ds.expand_dims(log_N_scale=n_axis.values).transpose("log_U", "log_nH", "log_N_scale", "line")
        ds["intensity"] = ds["intensity"] * 10 ** (n_axis * factor)
        for name in ("failed", "warnings"):
            ds[name] = ds[name].transpose("log_U", "log_nH", "log_N_scale")
    return CloudyGrid(ds)


def observe(truth=TRUTH, s=50.0, rel_err=0.03, seed=None, lines=FIT_LINES):
    """Region fluxes {line: (flux, error)} of the toy model, reddened with the default curve."""
    m = model_intensities(np.array(truth["log_U"]), np.array(truth["log_nH"]))
    a = O94(Rv=3.1)(np.array([get_line(l).wavelength_aa for l in lines]) * u.AA)
    flux = np.array([s * m[l] for l in lines]) * 10 ** (-0.4 * truth["A_V"] * a)
    err = rel_err * flux + 0.05
    if seed is not None:
        flux = flux + np.random.default_rng(seed).normal(0, err)
    return {l: (float(f), float(e)) for l, f, e in zip(lines, flux, err)}


def region_table(observations, radii_kpc=None, detected=None) -> QTable:
    """A regions table (as from `eelrfit.regions.aggregate`) holding the given observations."""
    table = QTable({"region": np.arange(len(observations))})
    if radii_kpc is not None:
        table["r_mean_kpc"] = np.asarray(radii_kpc, float) * u.kpc
    for line in observations[0]:
        table[f"{line}_flux"] = [o[line][0] for o in observations] * UNIT
        table[f"{line}_error"] = [o[line][1] for o in observations] * UNIT
        if detected is not None:
            table[f"{line}_detected"] = [d(line) for d in detected]
    return table
