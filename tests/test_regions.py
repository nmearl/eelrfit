import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.table import QTable
from scipy import ndimage

from conftest import CX, CY, NX, NY, UNIT, example_file, make_maps, make_wcs
from eelrfit import GalaxyProperties, read
from eelrfit.regions import _region_covariance, aggregate, annuli, continuum_center, spaxel_correlation


def test_continuum_center():
    c = continuum_center(make_maps())
    # Within 0.1 spaxel; the residual is the pixelization of the circular aperture.
    assert (c.x, c.y) == pytest.approx((CX, CY), abs=0.1) and c.method == "continuum_centroid"
    # A bright foreground star is ignored when masked.
    cont = make_maps().continuum.value.copy()
    cont[3, 3] = 1e4
    star = np.zeros((NY, NX), bool)
    star[2:5, 2:5] = True
    assert (continuum_center(make_maps(masks={"gaia_star": star}, continuum=cont)).x) == pytest.approx(CX, abs=0.1)
    # An offset from the catalog position is noted.
    far = GalaxyProperties(centers={"catalog": make_wcs().pixel_to_world(CX + 3, CY)})
    assert "catalog position" in continuum_center(make_maps(properties=far)).notes[0]
    with pytest.raises(ValueError, match="no continuum map"):
        continuum_center(make_maps(continuum=False))


def test_annuli():
    maps = make_maps(psf=2.0)
    assert annuli(maps).width[0].to_value(u.arcsec) == pytest.approx(2.0 / 3)  # a third of the PSF FWHM
    reg = annuli(maps, samples_per_fwhm=1)
    yy, xx = np.indices((NY, NX))
    r = np.hypot(xx - reg.center.x, yy - reg.center.y) * 0.5
    inside = reg.labels >= 0
    np.testing.assert_array_equal(reg.labels[inside], np.floor(r[inside] / 2.0).astype(int))
    star = np.zeros((NY, NX), bool)
    star[0:3, 0:3] = True
    reg = annuli(make_maps(masks={"gaia_star": star}), center="header_object", width=0.75 * u.arcsec, max_radius=5 * u.arcsec)
    assert reg.center.method == "header_object" and reg.n_regions == 7 and (reg.labels[star] == -1).all()
    assert annuli(maps, center=SkyCoord(150 * u.deg, 2 * u.deg)).center.method == "user"
    with pytest.raises(KeyError, match="Unknown center"):
        annuli(maps, center="nowhere")


def test_weighted_mean_and_correlated_errors():
    rng = np.random.default_rng(4)
    err = rng.uniform(0.05, 0.3, size=(1, NY, NX))
    flux = rng.normal(2.0, err)
    maps = make_maps(flux=flux, error=err, lines=("OIII_5007",), psf=2.4)
    reg = annuli(maps)
    t = aggregate(maps, reg)
    sel = reg.labels == 2
    w = 1 / err[0][sel] ** 2
    assert t["OIII_5007_flux"][2].to_value(UNIT) == pytest.approx(np.sum(w * flux[0][sel]) / np.sum(w))
    assert t["OIII_5007_n"][2] == sel.sum() == t["n_spaxels"][2]
    assert 0.5 < t["OIII_5007_rchi2"][2] < 1.6  # pure noise about a constant
    # The error of the mean with correlated spaxel noise, computed directly.
    a = np.where(sel, 1 / err[0], 0.0)
    expected = np.sqrt(np.sum(a * ndimage.convolve(a, spaxel_correlation(), mode="constant"))) / np.sum(a**2)
    assert t["OIII_5007_error"][2].to_value(UNIT) == pytest.approx(expected)
    independent = aggregate(maps, reg, covariance=None)
    assert independent["OIII_5007_error"][2].to_value(UNIT) == pytest.approx(np.sqrt(1 / np.sum(w)))
    # Neighboring annuli share noise, distant ones do not.
    corr = np.array(t.meta["region_noise_correlation"], dtype=float)
    assert 0.2 < corr[2, 3] < 0.9 and abs(corr[2, 6]) < 0.05


def test_correlation_kernel_matches_the_manga_calibration():
    # For compact bins it reproduces Westfall et al. (2019): 1 + 1.62 log10 N (to 10%).
    size = 41
    yy, xx = np.indices((size, size))
    order = np.argsort(np.hypot(yy - size // 2, xx - size // 2).ravel(), kind="stable")
    for n in (5, 12, 30, 70, 100):
        m = np.zeros(size * size, bool)
        m[order[:n]] = True
        var = _region_covariance(np.ones((size, size)), [m.reshape(size, size)], spaxel_correlation())[0, 0]
        assert np.sqrt(var * n) == pytest.approx(1 + 1.62 * np.log10(n), rel=0.1)


def test_snr_is_applied_to_regions_not_spaxels():
    rng = np.random.default_rng(2)
    flux = rng.normal(1.5, 1.0, size=(1, NY, NX))  # S/N 1.5 per spaxel
    maps = make_maps(flux=flux, error=np.ones((1, NY, NX)), lines=("OI_6300",))
    t = aggregate(maps, annuli(maps, width=6 * u.arcsec), snr_threshold=3)
    assert (flux < 0).sum() > 30 and t["OI_6300_n"][1] == t["n_spaxels"][1]  # negative spaxels are used
    assert t["OI_6300_detected"][1]
    empty = make_maps(error=np.zeros((1, NY, NX)), lines=("OIII_5007",))
    t = aggregate(empty, annuli(empty))
    assert np.all(t["OIII_5007_n"] == 0) and not t["OIII_5007_detected"].any()  # kept and flagged


def test_physical_radii_and_metadata(tmp_path):
    maps = make_maps(properties=GalaxyProperties(redshift=0.05))
    t = aggregate(maps, annuli(maps, samples_per_fwhm=1), lines=["OIII_5007"])
    scale = (Planck18.angular_diameter_distance(0.05) * (1 * u.arcsec).to_value(u.rad)).to_value(u.kpc)
    assert t.meta["kpc_per_arcsec"] == pytest.approx(scale) and t.meta["cosmology"] == "Planck18"
    assert t["r_outer_kpc"][0].to_value(u.kpc) == pytest.approx(2.0 * scale)
    assert t.meta["lines"] == ["OIII_5007"] and "HI_4861_flux" not in t.colnames
    assert t.meta["center"]["method"] == "continuum_centroid" and t.meta["psf_fwhm_arcsec"] == 2.0
    t.write(tmp_path / "regions.ecsv")
    back = QTable.read(tmp_path / "regions.ecsv")
    np.testing.assert_allclose(back["OIII_5007_flux"].to_value(UNIT), t["OIII_5007_flux"].to_value(UNIT))
    assert "r_mean_kpc" not in aggregate(make_maps(), annuli(make_maps())).colnames


@pytest.mark.example_data
def test_regions_on_a_real_cube():
    maps = read(example_file("manga-9495-3702.Pipe3D.cube.fits.gz"), ["Ha", "Hb"], catalogs=False)
    center = continuum_center(maps)
    assert center.coord.separation(maps.meta.object_coord) < 0.5 * u.arcsec
    regions = annuli(maps)
    table = aggregate(maps, regions)
    assert u.isclose(regions.width[0], maps.meta.psf_fwhm / 3)
    assert table["HI_4861_detected"][0] and (table["HI_6563_n"] <= table["n_spaxels"]).all()
