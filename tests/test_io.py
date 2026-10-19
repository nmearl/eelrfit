from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from conftest import LONG_LINES, SHAPE, SHORT_LINES, example_file, footprint, line_value, write_pipe3d
from eelrfit import GalaxyProperties, read
from eelrfit.catalogs import Catalog
import eelrfit.io.loader as loader_module
from eelrfit.io import AmbiguousLineError, LineNotFoundError, Loader, MaNGALoader, Pipe3DLoader, loaders, register_loader
from eelrfit.io.pipe3d import celestial_wcs, parse_unit, plane_catalog, resolve_plane
from eelrfit.lines import EmissionLine


def plane(label, wave, lines=LONG_LINES):
    return lines.index((label, wave))


# --- pyPipe3D plane matching -----------------------------------------------------------------


def test_parse_unit():
    assert parse_unit("10^-16 erg/s/cm^2") == 1e-16 * u.erg / u.s / u.cm**2
    assert parse_unit("km/s") == u.km / u.s


def test_planes_are_matched_by_wavelength(manga_file):
    cat = plane_catalog(manga_file)
    n = len(LONG_LINES)
    assert resolve_plane(cat, "Ha", "e_flux", "FLUX_ELINES_LONG") == 4 * n + plane("Halpha", 6562.85)
    # The same line has another label and plane in the other extension.
    assert resolve_plane(cat, "Ha", "flux", "FLUX_ELINES") == plane("Ha", 6562.68, SHORT_LINES)
    # Duplicate "[SII]" labels are told apart by wavelength.
    assert resolve_plane(cat, "SII_6731", "flux", "FLUX_ELINES_LONG") == plane("[SII]", 6730.82)
    # The [O II] blend only matches the blend.
    with pytest.raises(LineNotFoundError, match="nearest"):
        resolve_plane(cat, "OII_3727", "flux", "FLUX_ELINES_LONG")


def test_ambiguous_and_mislabeled_planes_are_errors(manga_file):
    cat = plane_catalog(manga_file)
    # HeI 3888.65 and HI 3889.05 both lie within 1 A; the species breaks the tie when it can.
    with pytest.raises(AmbiguousLineError, match="2 flux planes"):
        resolve_plane(cat, EmissionLine("NeIII_3889", "Ne", 3, 3888.9), "flux", "FLUX_ELINES_LONG")
    assert resolve_plane(cat, EmissionLine("HeI_3889", "He", 1, 3888.9), "flux", "FLUX_ELINES_LONG") == plane("HeI", 3888.65)
    with pytest.raises(LineNotFoundError, match="labeled as OIII"):
        resolve_plane(cat, EmissionLine("FeII_5007", "Fe", 2, 5006.9), "flux", "FLUX_ELINES_LONG")


def test_celestial_wcs_ignores_malformed_spectral_keywords(sami_file):
    header = fits.getheader(sami_file, 0)
    assert header["CDELT3"] == ""
    sky = celestial_wcs(header).pixel_to_world(2.5, 2.0)  # CRPIX is 1-indexed
    assert (sky.ra.deg, sky.dec.deg) == pytest.approx((216.966208, 0.75425))


# --- reading ---------------------------------------------------------------------------------


def test_read_requested_lines(manga_file):
    maps = read(manga_file, ["Ha", "Hb", "SII_6731"], catalogs=False)
    assert maps.line_ids == ("HI_6563", "HI_4861", "SII_6731")
    assert maps.flux.unit == 1e-16 * u.erg / u.s / u.cm**2
    expected = line_value("FLUX_ELINES_LONG", plane("Halpha", 6562.85))
    np.testing.assert_allclose(maps.flux_map("Ha").value[footprint()], expected)
    np.testing.assert_allclose(maps.error_map("Ha").value[footprint()], 0.1 * expected)
    assert maps.line_provenance["HI_6563"]["label"] == "Halpha"
    np.testing.assert_allclose(maps.pixel_scale.to_value(u.arcsec), 0.5, rtol=1e-5)
    with pytest.raises(LineNotFoundError, match="HeII_4686"):
        read(manga_file, ["Ha", "HeII_4686"], catalogs=False)
    short = read(manga_file, ["OII_3727"], loader=MaNGALoader(extension="FLUX_ELINES"), catalogs=False)
    assert short.line_provenance["OII_3727"]["extension"] == "FLUX_ELINES"


def test_default_lines_record_what_is_missing(manga_file):
    maps = read(manga_file, catalogs=False)
    assert "HI_6563" in maps and "HeII_4686" not in maps
    with pytest.raises(KeyError, match="HeII_4686 is not in these maps"):
        maps.flux_map("HeII_4686")


def test_validity_masks_and_immutability(manga_file):
    maps = read(manga_file, ["OII_3726", "Ha"], catalogs=False)
    valid = maps.valid("OII_3726")
    np.testing.assert_array_equal(valid, footprint())  # zero error outside the footprint
    assert maps.flux_map("OII_3726")[1, 1].value == -0.5 and valid[1, 1]  # negative fluxes are kept
    assert np.isnan(maps.snr("OII_3726")[0, 0])
    np.testing.assert_array_equal(maps.masks["coverage"], footprint())
    assert maps.masks["gaia_star"].sum() == 1
    assert not maps.masks["select_reg"][1, 4] and maps.valid("Ha")[1, 4]
    with pytest.raises(ValueError):
        maps.flux.value[0, 2, 2] = 0.0
    with pytest.raises(TypeError):
        maps.masks["new"] = np.ones(SHAPE, bool)


def test_metadata(manga_file, sami_file, kilogas_manga_file, kilogas_sami_file):
    manga = read(manga_file, ["Ha"], catalogs=False)
    assert (manga.meta.survey, manga.meta.target_id, manga.meta.aliases["mangaid"]) == ("MaNGA", "11759-3701", "1-207")
    assert manga.meta.psf_fwhm == 2.59434 * u.arcsec and manga.meta.galactic_extinction_corrected is True
    assert manga.meta.software == {"MaNGA DRP": "v3_1_1", "FLUX_ELINES_LONG pipeline": "pyFIT3D v1.0.0", "Pipe3D": "v3_1_1"}
    assert len(manga.meta.sha256) == 64
    sami = read(sami_file, ["Ha"], catalogs=False)
    assert (sami.meta.survey, sami.meta.target_id, sami.meta.galactic_extinction_corrected) == ("SAMI", "15847", None)
    assert sami.properties.redshift == pytest.approx(0.05706) and sami.properties.sources["redshift"] == "header"
    assert read(kilogas_manga_file, ["Ha"], catalogs=False).meta.target_id == "KG-MaNGA-1-207"
    kg = read(kilogas_sami_file, ["Ha"], catalogs=False)
    assert (kg.meta.survey, kg.meta.instrument, kg.meta.aliases["sami_catid"]) == ("KILOGAS", "SAMI", "15847")


def test_loader_choice(manga_file, kilogas_manga_file, tmp_path):
    # A KILOGAS-MaNGA header is identical to a DR17 one: only the file name decides.
    assert read(kilogas_manga_file, ["Ha"], catalogs=False).meta.survey == "KILOGAS"
    assert read(kilogas_manga_file, ["Ha"], loader="manga", catalogs=False).meta.survey == "MaNGA"
    wrong = write_pipe3d(tmp_path / "KG-MaNGA-1-999.Pipe3D.cube.fits.gz", "manga")
    with pytest.raises(ValueError, match="does not match the header"):
        read(wrong, ["Ha"], catalogs=False)
    other = tmp_path / "notes.txt"
    other.write_text("not a cube")
    with pytest.raises(ValueError, match="matching: none"):
        read(other)


def test_custom_loader(manga_file, tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "_REGISTRY", dict(loader_module._REGISTRY))  # undone after the test

    @register_loader
    class Renamed(Pipe3DLoader):
        name = "test-renamed"

        @classmethod
        def identify(cls, path: Path) -> bool:
            return path.suffix == ".p3d"

        def metadata(self, header, path):
            return {"survey": "Test", "instrument": "MaNGA", "target_id": path.stem}

    path = tmp_path / "galaxy.p3d"
    path.write_bytes(manga_file.read_bytes())
    maps = read(path, ["Ha"])
    assert (maps.meta.survey, maps.meta.target_id) == ("Test", "galaxy")
    assert issubclass(loaders()["test-renamed"], Loader)


# --- catalogs ----------------------------------------------------------------------------------


@pytest.fixture
def dr17(tmp_path):
    """A two-row stand-in for the DR17 Pipe3D table."""
    path = tmp_path / "dr17.fits"
    Table({"plateifu": ["9495-3702", "11759-3701"], "nsa_redshift": [0.022224, 0.0457],
           "objra": [123.320626, 145.594030155], "objdec": [22.648319, 0.309414123675]}).write(path)
    return Catalog("dr17", str(path), "plateifu", redshift="nsa_redshift", centers={"catalog": ("objra", "objdec")})


def test_catalog_lookup_at_read_time(manga_file, dr17, tmp_path):
    maps = read(manga_file, ["Ha"], catalogs=[dr17])
    props = maps.properties
    assert props.redshift == pytest.approx(0.0457) and props.sources["redshift"] == "dr17:nsa_redshift"
    assert set(props.centers) == {"catalog", "header_object", "header_pointing"}
    assert props.catalogs["dr17"]["sha256"] and props.notes == ()
    # Several catalogs: the first value wins and centers are merged.
    bary_path = tmp_path / "bary.fits"
    Table({"name": ["11759-3701"], "z": [0.05], "ra": [145.5941], "dec": [0.3093]}).write(bary_path)
    bary = Catalog("bary", str(bary_path), "name", redshift="z", centers={"barycenter": ("ra", "dec")})
    merged = read(manga_file, ["Ha"], catalogs=[dr17, bary]).properties
    assert merged.redshift == pytest.approx(0.0457) and "barycenter" in merged.centers
    with pytest.raises(LookupError, match="none of the catalogs"):
        read(write_pipe3d(tmp_path / "sami-15847.Pipe3D.cube.fits", "sami"), ["Ha"], catalogs=[dr17, bary])


def test_user_properties_and_header_disagreements(sami_file, tmp_path):
    maps = read(sami_file, ["Ha"], properties=GalaxyProperties(redshift=0.06, centers={"mine": SkyCoord(216.97, 0.754, unit="deg")}))
    props = maps.properties
    assert props.redshift == 0.06 and props.sources["redshift"] == "user" and props.sources["centers.mine"] == "user"
    assert "header redshift 0.05706 differs" in props.notes[0]
    with pytest.raises(FileNotFoundError):
        Catalog("missing", str(tmp_path / "none.fits"), "id").lookup("x")


# --- real cubes --------------------------------------------------------------------------------


@pytest.mark.example_data
@pytest.mark.parametrize("name, loader", [("manga-9495-3702.Pipe3D.cube.fits.gz", "manga"), ("KG-SAMI-396833.Pipe3D.cube.fits.gz", "kilogas")])
def test_real_cubes(name, loader):
    maps = read(example_file(name), ["Ha", "Hb", "SII_6717", "SII_6731"], catalogs=False)
    assert maps.valid("Ha").sum() > 100
    np.testing.assert_allclose(maps.pixel_scale.to_value(u.arcsec), 0.5, rtol=1e-5)
    # The WCS reference pixel points at the target.
    assert maps.wcs.pixel_to_world(*(maps.wcs.wcs.crpix - 1)).separation(maps.meta.object_coord) < 0.1 * u.arcsec
    # Both emission-line extensions measure the same H-alpha.
    short = read(example_file(name), ["Ha"], loader=loaders()[loader](extension="FLUX_ELINES"), catalogs=False)
    good = short.snr("Ha") > 10
    assert np.median(maps.flux_map("Ha").value[good] / short.flux_map("Ha").value[good]) == pytest.approx(1.0, abs=0.01)
