import math

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from dust_extinction.parameter_averages import O94
from scipy.optimize import minimize_scalar

from conftest import AGN_SED, FIT_LINES, LOG_NH, LOG_U, TRUTH, example_file, make_grid, model_intensities, observe, region_table
from eelrfit import GalaxyProperties, read
from eelrfit.cloudy import CloudyGrid
from eelrfit.fit import FitConfig, fit_region, fit_regions
from eelrfit.lines import get_line
from eelrfit.pipeline import analyze
from eelrfit.sed import read_cloudy_sed

SII = ("SII_6717", "SII_6731")


def test_recovery():
    fit = fit_region(observe(), make_grid())
    assert fit.status == "ok" and fit.best == pytest.approx(TRUTH) and fit.chi2_min == pytest.approx(0, abs=1e-6)
    assert fit.dof == len(FIT_LINES) - 4  # normalization, A_V, log U, log nH
    assert np.isclose(fit.posterior.sum(), 1)
    summaries = [fit_region(observe(seed=k), make_grid()).summary("log_U") for k in range(10)]
    assert sum(s["p16"] - 0.25 <= TRUTH["log_U"] <= s["p84"] + 0.25 for s in summaries) >= 8
    # Reddened data need the dust term.
    obs = observe(truth={**TRUTH, "A_V": 1.5})
    with_dust, without = fit_region(obs, make_grid()), fit_region(obs, make_grid(), FitConfig(attenuation=None))
    assert with_dust.best["A_V"] == pytest.approx(1.5) and without.chi2_min > with_dust.chi2_min + 10


def test_posterior_does_not_depend_on_the_normalization_line():
    obs = observe(seed=3)
    U, N = np.meshgrid(LOG_U, LOG_NH, indexing="ij")
    base = fit_region(obs, make_grid())
    other = fit_region(obs, make_grid(scale=1.0 / model_intensities(U, N)["HI_6563"]))  # models relative to H-alpha
    np.testing.assert_allclose(other.log_posterior, base.log_posterior, atol=1e-8)


def _best_cell_model(fit):
    m = model_intensities(np.array(fit.best["log_U"]), np.array(fit.best["log_nH"]))
    a = O94(Rv=3.1)(np.array([get_line(l).wavelength_aa for l in fit.lines_used]) * u.AA)
    return np.array([m[l] for l in fit.lines_used]) * 10 ** (-0.4 * fit.best["A_V"] * a)


@pytest.mark.parametrize("ratio_floor", [{}, {SII: 0.03}])
def test_normalization_is_the_exact_profile_maximum(ratio_floor):
    # Model mismatch and high S/N, where an approximate profile would differ most.
    rng = np.random.default_rng(7)
    obs = {l: (f * 10 ** rng.normal(0, 0.15), 0.01 * f) for l, (f, _) in observe().items()}
    fit = fit_region(obs, make_grid(), FitConfig(ratio_floor=ratio_floor))
    f = np.array([obs[l][0] for l in fit.lines_used])
    cov = np.diag(np.array([obs[l][1] for l in fit.lines_used]) ** 2 + (0.1 * f) ** 2)
    if ratio_floor:
        i, j = fit.lines_used.index(SII[0]), fit.lines_used.index(SII[1])
        cov[i, j] = cov[j, i] = (0.1**2 - 0.03**2 / 2) * f[i] * f[j]  # the pair's ratio has a 3% model uncertainty
    weight = np.linalg.inv(cov)
    g = _best_cell_model(fit)
    s_fit = fit.best_model_flux[fit.lines_used[0]] / g[0]

    def chi2(s):
        return (f - s * g) @ weight @ (f - s * g)

    ref = minimize_scalar(chi2, bounds=(0, 100 * s_fit), method="bounded", options={"xatol": 1e-10 * s_fit})
    assert s_fit == pytest.approx(ref.x, rel=1e-6) and fit.chi2_min == pytest.approx(chi2(s_fit))


def test_mismatched_model_is_not_inflated_above_the_data():
    # A floor proportional to the model would let the fit brighten the model to inflate its variance.
    rng = np.random.default_rng(11)
    ratios = []
    for seed in range(20):
        obs = {l: (f * 10 ** rng.normal(0, 0.3), 0.005 * f) for l, (f, _) in observe(seed=seed).items()}
        fit = fit_region(obs, make_grid())
        ratios += [obs[l][0] / fit.best_model_flux[l] for l in fit.lines_used]
    assert 0.85 < np.median(ratios) < 1.3


def test_floors():
    obs = observe(seed=5, rel_err=0.005)
    tight = fit_region(obs, make_grid(), FitConfig(model_floor=0.0)).summary("log_U")
    loose = fit_region(obs, make_grid(), FitConfig(model_floor=0.3)).summary("log_U")
    assert loose["p84"] - loose["p16"] > tight["p84"] - tight["p16"]
    # The [S II] ratio floor sharpens the density; at √2 × model_floor (independent errors) it changes nothing.
    obs = observe(rel_err=0.002)
    loose = fit_region(obs, make_grid(), FitConfig(ratio_floor={})).summary("log_nH")
    tight = fit_region(obs, make_grid(), FitConfig(ratio_floor={SII: 0.03})).summary("log_nH")
    assert tight["p84"] - tight["p16"] < 0.6 * (loose["p84"] - loose["p16"]) and tight["p16"] <= 2.5 <= tight["p84"]
    plain = fit_region(observe(seed=3), make_grid(), FitConfig(ratio_floor={}))
    same = fit_region(observe(seed=3), make_grid(), FitConfig(ratio_floor={SII: np.sqrt(2) * 0.1}))
    np.testing.assert_allclose(same.log_posterior, plain.log_posterior, atol=1e-9)
    assert FitConfig().ratio_floor == {SII: 0.05} and FitConfig().to_dict()["ratio_floor"] == {"SII_6717/SII_6731": 0.05}
    assert FitConfig().to_dict()["attenuation"] == "O94(Rv=3.1)"
    with pytest.raises(ValueError, match="one pair"):
        FitConfig(ratio_floor={SII: 0.05, ("SII_6731", "NII_6584"): 0.05})


def test_excluded_models_missing_lines_and_fixed_parameters():
    i, j = list(LOG_U).index(-2.5), list(LOG_NH).index(2.5)
    for flag in ("failed", "warnings"):
        ds = make_grid().dataset.copy(deep=True)
        ds[flag].loc[{"log_U": -2.5, "log_nH": 2.5}] = True
        assert fit_region(observe(), CloudyGrid(ds)).posterior[i, j] == 0
    assert fit_region(observe(), CloudyGrid(ds), FitConfig(allow_warned_models=True)).best["log_nH"] == 2.5
    obs = observe()
    del obs["OI_6300"]
    assert fit_region(obs, make_grid()).lines_missing == {"OI_6300": "no valid measurement"}
    assert fit_region({k: obs[k] for k in ("HI_4861", "OIII_5007")}, make_grid()).status == "too_few_lines"
    fixed = fit_region(observe(seed=1), make_grid(), FitConfig(fixed={"log_nH": 2.5}))
    assert fixed.parameters == ("log_U",) and fixed.fixed["log_nH"] == 2.5
    with pytest.raises(KeyError, match="Cannot fix"):
        fit_region(observe(), make_grid(), FitConfig(fixed={"log_Z": 0}))


def test_extra_grid_dimension_is_marginalized():
    obs = observe()
    obs["NII_6584"] = tuple(v * 10**0.5 for v in obs["NII_6584"])  # the data are at log_N_scale = 0.5
    grid = make_grid(n_scales=[0.0, 0.25, 0.5, 0.75])
    results, fits = fit_regions(region_table([obs], radii_kpc=[1.0]), grid)
    assert fits[0].parameters == ("log_U", "log_nH", "log_N_scale") and results["log_N_scale_best"][0] == 0.5
    assert results["log_N_scale_p16"][0] <= 0.5 <= results["log_N_scale_p84"][0]


def test_luminosities_limits_and_quoted_values():
    truths = [TRUTH, {**TRUTH, "log_nH": 0.0}, TRUTH]
    obs = [observe(truth=t, seed=k + 10) for k, t in enumerate(truths)]
    detected = [lambda line: True, lambda line: True, lambda line: line in ("HI_6563", "OIII_5007")]
    sed = read_cloudy_sed(AGN_SED)
    results, _ = fit_regions(region_table(obs, radii_kpc=[0.5, 1.5, 2.5], detected=detected), make_grid(), FitConfig(band_ev=(13.6, 54.6)), sed=sed)
    assert results["constrained"].tolist() == [True, True, False]  # region 2: H-beta not detected
    # Q = 4π r² n_H c U at the best fit, and L from the SED's luminosity per ionizing photon.
    q = math.log10(4 * math.pi * (0.5 * u.kpc).to_value(u.cm) ** 2 * const.c.cgs.value) + results["log_nH_best"][0] + results["log_U_best"][0]
    assert results["log_Q_best"][0] == pytest.approx(q)
    assert results["log_L_band_best"][0] == pytest.approx(q + math.log10(sed.band_luminosity_per_q((13.6, 54.6)).to_value(u.erg)))
    # Region 1 is in the [S II] low-density limit: n_H, and hence L, is an upper limit, quoted at p84.
    assert "lower" in results["log_nH_edge"][1] and results["log_L_limit"][1] == "upper"
    for name in ("log_Q", "log_L_ion", "log_L_band"):
        bound = {"upper": "p84", "lower": "p16"}
        np.testing.assert_array_equal(results[f"{name}_quoted"], [results[f"{name}_{bound.get(lim, 'p50')}"][k] for k, lim in enumerate(results["log_L_limit"])])
    # Without a band only L_ion is computed; without radii no Q at all.
    assert "log_L_band_p50" not in fit_regions(region_table(obs, radii_kpc=[0.5, 1.5, 2.5]), make_grid(), sed=sed)[0].colnames
    assert "log_Q_p50" not in fit_regions(region_table(obs), make_grid(), sed=sed)[0].colnames


@pytest.mark.example_data
def test_pipeline_on_a_real_cube(tmp_path):
    """Plumbing: a real cube with the toy grid (not a science result)."""
    maps = read(example_file("manga-9495-3702.Pipe3D.cube.fits.gz"), properties=GalaxyProperties(redshift=0.022))
    result = analyze(maps, make_grid(), sed=read_cloudy_sed(AGN_SED))
    ok = result.results["status"] == "ok"
    assert ok.any() and (result.results["log_L_ion_p50"][ok] > 30).all()
    assert result.results.meta["target_id"] == "9495-3702"
    result.write(tmp_path / "out.ecsv")
    assert (tmp_path / "out_regions.ecsv").exists()
