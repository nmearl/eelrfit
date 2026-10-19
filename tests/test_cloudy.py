import json
import os
import shutil
import stat
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from eelrfit.cloudy import Axis, CloudyGrid, CloudyRunError, GridSpec, example_specs, read_cloudy_grid, run_grid
from eelrfit.cloudy.grid import read_grd, read_lin
from eelrfit.cloudy.run import _sha256, grid_key, resolve_executable
from eelrfit.sed import read_cloudy_sed

DATA = Path(__file__).parent / "data" / "cloudy"
GRID_1D = DATA / "c25.00_1d"  # log U = -3, -2.5, -2 at log nH = 1 (real Cloudy c25.00 output)
GRID_2D = DATA / "c25.00_2d"  # log U = -3, -2 x log nH = 1, 3
GRID_ABUND = DATA / "c25.00_abund"  # log U = -3, log nH = 2; N scale -0.5..1 (0.5 dex) x S scale -0.5, 0
SED = example_specs()["starecheski_agn"].with_name("starecheski_agn.sed")
COMMANDS = ("filling factor -5", "iterate to convergence", "stop column density 21")


def spec(log_U=(-3.0, -2.0, 0.5), log_nH=1.0, **kwargs):
    axes = (Axis("log_U", "ionization parameter {value}", log_U), Axis("log_nH", "hden {value}", log_nH))
    return GridSpec(axes=axes + tuple(kwargs.pop("extra_axes", ())), sed=SED, **{"commands": COMMANDS, **kwargs})


def load(directory, parameters=None, fixed=None, **kwargs):
    return read_cloudy_grid(directory / "gridrun.grd", directory / "gridrun.lin", parameters=parameters, fixed=fixed, **kwargs)


# --- specifications ---------------------------------------------------------------------------


def test_spec_renders_a_cloudy_deck():
    deck = spec(extra_axes=[Axis("log_N_scale", "element nitrogen scale {value} log", (-0.5, 1.0, 0.5))]).render_input(ncpus=4)
    assert deck.splitlines() == [
        'table SED "sed.sed"',
        "ionization parameter -3 vary", "grid -3 -2 0.5 ncpus 4",
        "hden 1",
        "element nitrogen scale -0.5 log vary", "grid -0.5 1 0.5 ncpus 4",
        *COMMANDS,
        'save grid "grid.grd" last no hash',
        'save line list "grid.lin" "linelist.dat" emergent last no hash',
    ]
    s = spec(log_nH=(0.0, 4.0, 0.25))
    assert s.shape == {"log_U": 3, "log_nH": 17} and s.n_models == 51 and s.fixed == {}
    assert spec(lines=["OIII_5007"]).render_line_list() == "H  1 4861.32A\nO  3 5006.84A\n"  # H-beta always added


def test_spec_validation():
    with pytest.raises(ValueError, match="exactly one \\{value\\}"):
        Axis("log_U", "ionization parameter", (-3, -2, 0.5))
    with pytest.raises(ValueError, match="written by EELRFit"):
        spec(commands=["save overview \"x.ovr\""])
    with pytest.raises(ValueError, match="At least one axis must vary"):
        spec(log_U=-3.0)
    with pytest.raises(ValueError, match="exactly one of sed and cloudy_sed"):
        GridSpec(axes=(Axis("log_U", "ionization parameter {value}", (-3, -2, 0.5)),))
    with pytest.raises(ValueError, match="needs a command and exactly one of range and value"):
        GridSpec.from_dict({"sed": {"file": str(SED)}, "axes": {"log_U": {"command": "ionization parameter {value}"}}})


def test_shipped_specs(tmp_path, monkeypatch):
    specs = {name: GridSpec.read(path) for name, path in example_specs().items()}
    assert specs["starecheski_agn_free_no"].shape == {"log_U": 31, "log_nH": 25, "log_N_scale": 16}
    assert specs["starecheski_agn"].sed.is_file()
    # The shipped AGN SED: log L(13.6-54.6 eV) per ionizing photon, as used in the re-analysis
    sed = read_cloudy_sed(specs["starecheski_agn"].sed)
    assert np.log10(sed.band_luminosity_per_q((13.6, 54.6)).to_value(u.erg)) == pytest.approx(-10.486, abs=1e-3)
    # A Cloudy-bundled SED is looked up in $CLOUDY_DATA_PATH/SED or next to the executable.
    jin = specs["jin12_agn"]
    (tmp_path / "c25.00" / "data" / "SED").mkdir(parents=True)
    (tmp_path / "c25.00" / "source").mkdir()
    (tmp_path / "c25.00" / "data" / "SED" / "AGN_Jin12_Eddr_mid.sed").write_text("1 1\n2 1\n")
    monkeypatch.delenv("CLOUDY_DATA_PATH", raising=False)
    assert jin.sed_path(tmp_path / "c25.00" / "source" / "cloudy.exe").parent.name == "SED"
    with pytest.raises(FileNotFoundError):
        jin.sed_path()


# --- reading Cloudy output ------------------------------------------------------------------


def test_read_cloudy_output():
    two = read_grd(GRID_2D / "gridrun.grd")
    assert two["cloudy_names"] == ["IONIZATIO", "HDEN=%f L"] and two["values"].tolist() == [[-3, 1], [-3, 3], [-2, 1], [-2, 3]]
    labels, values = read_lin(GRID_1D / "gridrun.lin")
    assert labels[0] == "H  1 6562.80A" and values.shape == (3, 10)
    np.testing.assert_array_equal(values[:, 1], 1.0)  # relative to H-beta

    grid = load(GRID_1D, ["log_U"], {"log_nH": 1.0}, cloudy_version="25.00")
    assert grid.parameters == ("log_U",) and grid.fixed_parameters == {"log_nH": 1.0}
    np.testing.assert_allclose(grid.intensity("OIII_5007").values, [4.9711, 8.8668, 12.306])
    assert "Ha" in grid and "HeII_4686" not in grid

    grid = load(GRID_2D, ["log_U", "log_nH"])
    assert grid.intensity("OIII_5007").sel(log_U=-2, log_nH=3).item() == pytest.approx(15.325)
    s2 = grid.intensity("SII_6717") / grid.intensity("SII_6731")
    assert (s2.sel(log_nH=1) > s2.sel(log_nH=3)).all()  # the [S II] doublet constrains the density

    # Cloudy truncates column names ("ELEMENT S" is also what silicon gives), so names come from the spec.
    assert load(GRID_ABUND).parameters == ("element_n", "element_s")
    grid = load(GRID_ABUND, ["log_N_scale", "log_S_scale"], {"log_U": -3.0, "log_nH": 2.0})
    nii = grid.intensity("NII_6584").sel(log_S_scale=0.0).values
    assert np.all(np.diff(nii) > 0) and 10 < nii[-1] / nii[0] < 10**1.5  # [N II] rises with N/H, less than proportionally


def test_netcdf_round_trip_and_flags(tmp_path):
    shutil.copytree(GRID_2D, tmp_path / "g")
    grd = tmp_path / "g" / "gridrun.grd"
    rows = grd.read_text().splitlines()
    rows[2] = rows[2].replace("\tF\tF\t", "\tT\tF\t", 1)  # model 1 (log U = -3, log nH = 3) failed
    grd.write_text("\n".join(rows) + "\n")
    grid = load(tmp_path / "g", ["log_U", "log_nH"], cloudy_version="25.00", provenance={"note": "test"}, input_deck="hden 1 vary\n")
    assert not grid.usable.sel(log_U=-3, log_nH=3).item() and int(grid.usable.sum()) == 3
    back = CloudyGrid.read(grid.to_netcdf(tmp_path / "grid.nc"))
    np.testing.assert_array_equal(back.intensity().values, grid.intensity().values)
    assert back.dataset["failed"].dtype == bool and back.provenance == {"note": "test"} and back.cloudy_version == "25.00"
    lin = tmp_path / "g" / "gridrun.lin"
    lin.write_text("\n".join(lin.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="has 3 models but"):
        load(tmp_path / "g", ["log_U", "log_nH"])


# --- running grids --------------------------------------------------------------------------

FAKE_CLOUDY = """#!{python}
import pathlib, re, shutil, sys
calls = pathlib.Path({calls!r})
calls.write_text(calls.read_text() + "run\\n" if calls.exists() else "run\\n")
name = sys.argv[sys.argv.index("-r") + 1]
deck = pathlib.Path(name + ".in").read_text()
shutil.copy(pathlib.Path({fixture!r}) / "gridrun.grd", re.search(r'save grid "([^"]+)"', deck).group(1))
shutil.copy(pathlib.Path({fixture!r}) / "gridrun.lin", re.search(r'save line list "([^"]+)"', deck).group(1))
pathlib.Path(name + ".out").write_text("            Cloudy 25.00\\n [Stop in cdMain, Cloudy exited OK]\\n")
sys.exit({exit_code})
"""


def fake_cloudy(tmp_path, exit_code=0, name="cloudy.exe"):
    """An executable that 'runs' a grid by copying the GRID_1D output."""
    exe, calls = tmp_path / name, tmp_path / f"{name}.calls"
    exe.write_text(FAKE_CLOUDY.format(python=sys.executable, fixture=str(GRID_1D), calls=str(calls), exit_code=exit_code))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe, calls


def test_run_grid_and_cache(tmp_path):
    exe, calls = fake_cloudy(tmp_path)
    grid = run_grid(spec(), executable=exe, cache_dir=tmp_path / "cache")
    assert grid.cloudy_version == "25.00" and grid.shape == {"log_U": 3} and grid.fixed_parameters == {"log_nH": 1.0}
    run_dir = Path(grid.provenance["run_dir"])
    assert {p.name for p in run_dir.iterdir()} >= {"input.in", "linelist.dat", "sed.sed", "grid.nc", "run.json", "cloudy.log"}
    assert json.loads((run_dir / "run.json").read_text())["status"] == "ok"
    assert grid.provenance["sed"]["sha256"] == _sha256(SED)
    run_grid(spec(), executable=exe, cache_dir=tmp_path / "cache")
    assert calls.read_text().count("run") == 1  # cached
    run_grid(spec(), executable=exe, cache_dir=tmp_path / "cache", force=True)
    assert calls.read_text().count("run") == 2


def test_cache_key(tmp_path):
    # The SED enters by its contents, not its name; the CPU count does not enter.
    assert grid_key(spec(), "sed-a", "exe") != grid_key(spec(), "sed-b", "exe") != grid_key(spec(), "sed-a", "other-exe")
    assert "ncpus" not in spec().render_input()


def test_run_failures(tmp_path, monkeypatch):
    exe, _ = fake_cloudy(tmp_path, exit_code=1)
    with pytest.raises(CloudyRunError, match="exited with code 1"):
        run_grid(spec(), executable=exe, cache_dir=tmp_path / "cache")
    assert json.loads(next((tmp_path / "cache").glob("*/run.json")).read_text())["status"] == "failed"
    exe, _ = fake_cloudy(tmp_path, name="ok.exe")
    with pytest.raises(CloudyRunError, match="expected"):  # 5 models specified, 3 "computed"
        run_grid(spec(log_U=(-3.0, -1.0, 0.5)), executable=exe, cache_dir=tmp_path / "cache")
    monkeypatch.delenv("CLOUDY_EXE", raising=False)
    with pytest.raises(CloudyRunError, match="CLOUDY_EXE"):
        resolve_executable()
    monkeypatch.setenv("CLOUDY_EXE", str(exe))
    assert resolve_executable() == exe


@pytest.mark.skipif(not os.environ.get("CLOUDY_EXE"), reason="set CLOUDY_EXE to run real Cloudy")
def test_real_cloudy(tmp_path):
    s = GridSpec(axes=(Axis("log_U", "ionization parameter {value}", (-3.0, -2.0, 1.0)), Axis("log_nH", "hden {value}", 2.0)),
                 sed=SED, commands=("stop zone 3",), lines=("Hb", "OIII_5007", "HeI_5876", "OII_3727"))
    grid = run_grid(s, cache_dir=tmp_path)
    assert grid.shape == {"log_U": 2} and grid.fixed_parameters["log_nH"] == 2.0
    assert set(grid.line_ids) == {"HI_4861", "OIII_5007", "HeI_5876", "OII_3727"} and bool(grid.usable.all())

