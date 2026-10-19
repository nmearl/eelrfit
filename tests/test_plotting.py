import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from conftest import AGN_SED, TRUTH, make_grid, make_maps, observe, region_table
from eelrfit import GalaxyProperties
from eelrfit import plotting as P
from eelrfit.fit import FitConfig, fit_regions
from eelrfit.regions import aggregate, annuli
from eelrfit.sed import read_cloudy_sed


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def analysis():
    """Regions table, results and fits for three synthetic regions; the last is in the low-density limit."""
    truths = [TRUTH, {**TRUTH, "log_U": -3.0, "log_nH": 2.0}, {**TRUTH, "log_U": -3.0, "log_nH": 0.0}]
    table = region_table([observe(truth=t, seed=k) for k, t in enumerate(truths)], radii_kpc=[0.5, 1.5, 2.5],
                         detected=[lambda line: True] * 3)
    results, fits = fit_regions(table, make_grid(), FitConfig(band_ev=(13.6, 54.6)), sed=read_cloudy_sed(AGN_SED))
    return table, results, fits


def test_maps_and_bpt():
    lines = ("HI_4861", "OIII_5007", "HI_6563", "NII_6584", "SII_6717", "SII_6731", "OI_6300")
    yy, xx = np.indices((31, 33))
    flux = np.stack([np.exp(-((xx - 16) ** 2 + (yy - 15) ** 2) / 50) * (k + 1) for k in range(len(lines))])
    maps = make_maps(flux=flux, error=np.full(flux.shape, 0.01), lines=lines, properties=GalaxyProperties(redshift=0.03))
    regions = annuli(maps)
    assert len(P.plot_line_map(maps, "OIII_5007", regions=regions).patches) == regions.n_regions
    axes = P.plot_bpt(maps, regions=regions, regions_table=aggregate(maps, regions))
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert len(axes) == 3 and "Kewley+01" in labels and any(l.startswith("spaxels") for l in labels)
    # Kewley+01 [N II]: log([O III]/Hβ) = 0.61 / (x - 0.47) + 1.19
    k01 = P.demarcation_lines("NII")[0]
    np.testing.assert_allclose(k01["y"], 0.61 / (k01["x"] - 0.47) + 1.19)
    # Model tracks, at a chosen value of a further grid parameter.
    grid = make_grid(n_scales=[0.0, 0.5])
    solar = P.plot_bpt(grid=grid, kinds=("NII",))[0].get_lines()[-2].get_xdata()
    enriched = P.plot_bpt(grid=grid, kinds=("NII",), grid_at={"log_N_scale": 0.5})[0].get_lines()[-2].get_xdata()
    np.testing.assert_allclose(enriched - solar, 0.5)


def test_posteriors(analysis):
    table, results, fits = analysis
    assert "χ²_min" in P.plot_posterior(fits[0]).get_title()
    fig = P.plot_posterior_panels(results, fits, n_panels=2)
    shown = [a for a in fig.axes if a.get_visible()]
    assert len(shown) == 2 and all("kpc" in a.get_title() and "L_ion" in a.get_title() for a in shown)
    fixed = fit_regions(table, make_grid(), FitConfig(fixed={"log_nH": 2.5}))[1][0]
    assert P.plot_posterior(fixed).get_xlabel() == "log U"


def test_profiles_and_luminosity_history(analysis):
    _, results, _ = analysis
    axes = P.plot_parameter_profiles({"model A": results, "model B": results})
    assert [t.get_text() for t in axes[0].get_legend().get_texts()] == ["model A", "model B"]
    ax = P.plot_luminosity_history(results, current={"IR": 42.6})
    assert "13.6–54.6 eV" in ax.get_ylabel()
    # The line joins the drawn positions: limits at their quoted bound.
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), np.asarray(results["log_L_band_quoted"], float))
    assert any("light-travel" in c.get_xlabel() for c in ax.get_children() if c.__class__.__name__ == "SecondaryAxis")
    assert "> 13.6 eV" in P.plot_luminosity_history(results, quantity="log_L_ion").get_ylabel()
    results.meta.update(psf_fwhm_arcsec=2.4, kpc_per_arcsec=0.5)
    (bar,) = [a for a in P.plot_luminosity_history(results).artists if isinstance(a, AnchoredSizeBar)]
    assert bar.txt_label.get_text() == "PSF FWHM 1.2 kpc"


def test_line_ratios_and_residuals(analysis):
    table, results, fits = analysis
    assert len(P.plot_line_ratios(table, fits)) == len(fits[0].lines_used) - 1  # every fitted line but H-beta
    z = P.plot_residuals(results, fits, table).collections[0].get_array()
    assert np.size(z) == len(fits) * len(fits[0].lines_used) and np.nanmax(np.abs(z)) < 5
