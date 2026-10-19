import math

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from eelrfit.sed import CLOUDY_MAX_RYD, RYDBERG_EV, SED, log_q_from_u, read_cloudy_sed, write_cloudy_sed


def write_power_law(tmp_path, alpha, header="units ev", nufnu=False):
    e = np.geomspace(1e-3, 1e8, 400)
    f = e ** (alpha + 1) if nufnu else e**alpha  # only the shape matters
    rows = [f"{e[0]:.10e} {f[0]:.10e} {header}{' nuFnu' if nufnu else ''}"] + [f"{x:.10e} {y:.10e}" for x, y in zip(e[1:], f[1:])]
    path = tmp_path / "pl.sed"
    path.write_text("# power law\n" + "\n".join(rows) + "\n***\n1 1\n")
    return path


@pytest.mark.parametrize("nufnu", [False, True])
def test_band_luminosity_of_a_power_law(tmp_path, nufnu):
    alpha, (e1, e2) = -1.5, (13.6, 54.6)
    sed = read_cloudy_sed(write_power_law(tmp_path, alpha, nufnu=nufnu))
    # L_band / Q = ∫_band ν^α dν / ∫_{1 Ryd}^{E_max} ν^(α-1) dν × h, energies in eV
    emax = CLOUDY_MAX_RYD * RYDBERG_EV
    expected = ((e2 ** (alpha + 1) - e1 ** (alpha + 1)) / (alpha + 1)) / ((emax**alpha - RYDBERG_EV**alpha) / alpha)
    assert sed.band_luminosity_per_q((e1, e2)).to_value(u.eV) == pytest.approx(expected, rel=1e-3)


def test_cloudy_sed_format(tmp_path):
    # Rydberg unless a units keyword says otherwise; keywords on any line; '#' comments anywhere; stop at ***.
    assert read_cloudy_sed(write_power_law(tmp_path, -1.0, header="")).energy_ev[0] == pytest.approx(1e-3 * RYDBERG_EV)
    path = tmp_path / "late.sed"
    path.write_text("# header\n1 1\n2 4 nuFnu # flux is nu f_nu\n3 9 units ev\n***\n10 10 units hz\n")
    sed = read_cloudy_sed(path)
    np.testing.assert_allclose(sed.energy_ev, [1, 2, 3])
    np.testing.assert_allclose(sed.fnu, np.array([1, 4, 9]) / (sed.energy_ev * u.eV).to_value(u.Hz, u.spectral()))
    path.write_text("1000 1 units angstrom\n100 1\n")
    assert read_cloudy_sed(path).energy_ev.max() == pytest.approx((100 * u.AA).to_value(u.eV, u.spectral()))
    path.write_text("1 1 units parsecs\n2 1\n")
    with pytest.raises(ValueError, match="Unsupported SED units"):
        read_cloudy_sed(path)


def test_write_and_validate(tmp_path):
    path = write_cloudy_sed(tmp_path / "x.sed", np.array([1.0, 2.0, 4.0]), np.array([2.0, 0.0, 1.0]), comments=["made in a test"])
    assert path.read_text().startswith("# made in a test\n")
    sed = read_cloudy_sed(path)
    np.testing.assert_allclose(sed.energy_ev, [1, 2, 4])
    assert sed.fnu[1] == pytest.approx(2e-30)  # zero floored relative to the peak (Cloudy interpolates in log)
    assert np.all(np.isfinite(SED(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 1.0])).fnu_at(np.array([1.5, 2.5]))))
    with pytest.raises(ValueError):
        SED(np.array([1.0, 1.0]), np.array([1.0, 2.0]))


def test_log_q():
    r = 1 * u.kpc
    assert log_q_from_u(-3, 2, r) == pytest.approx(math.log10(4 * math.pi * r.to_value(u.cm) ** 2 * const.c.cgs.value) - 1)
