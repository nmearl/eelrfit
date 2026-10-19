import pytest
from click.testing import CliRunner

from conftest import AGN_SED, make_grid
from eelrfit.cli import main
from test_cloudy import SED, fake_cloudy


def test_grid_commands(tmp_path):
    runner = CliRunner()
    exe, _ = fake_cloudy(tmp_path)
    spec = tmp_path / "grid.toml"
    spec.write_text(f'[sed]\nfile = "{SED}"\n[axes]\nlog_U = {{ command = "ionization parameter {{value}}", range = [-3.0, -2.0, 0.5] }}\n'
                    'log_nH = { command = "hden {value}", value = 1.0 }\n')
    result = runner.invoke(main, ["grid", "run", str(spec), "--dry-run"])
    assert result.exit_code == 0 and "ionization parameter -3 vary" in result.output
    result = runner.invoke(main, ["grid", "run", str(spec), "--cloudy-exe", str(exe), "--cache-dir", str(tmp_path / "cache"),
                                  "-o", str(tmp_path / "g.nc")])
    assert result.exit_code == 0, result.output
    result = runner.invoke(main, ["grid", "show", str(tmp_path / "g.nc")])
    assert "log_U: -3 to -2 (3 values)" in result.output and f"SED: {tmp_path}" in result.output  # the run directory's copy
    assert "starecheski_agn_free_no  (12400 models)" in runner.invoke(main, ["grid", "examples"]).output
    assert runner.invoke(main, ["grid", "run", "nonexistent"]).exit_code != 0


@pytest.mark.filterwarnings("ignore:constrained_layout not applied")  # the synthetic cube has only a few annuli
def test_analyze(manga_file, tmp_path):
    grid = make_grid().to_netcdf(tmp_path / "toy.nc")
    out = tmp_path / "out" / "galaxy.ecsv"
    args = ["analyze", str(manga_file), "-g", str(grid), "--redshift", "0.03", "--band", "13.6", "54.6", "-o", str(out)]
    result = CliRunner().invoke(main, args + ["--sed", str(AGN_SED), "--figures", str(tmp_path / "figures")])
    assert result.exit_code == 0, result.output
    assert "Redshift  : 0.03000 (user)" in result.output and "Annuli" in result.output
    assert out.exists() and (out.parent / "galaxy_regions.ecsv").exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) >= 5
    # The toy grid records no SED, so there are no luminosities without --sed; fixing an unknown parameter fails.
    assert "not computed (no SED" in CliRunner().invoke(main, args).output
    result = CliRunner().invoke(main, args + ["--fix", "log_Z=0"])
    assert result.exit_code != 0 and "Cannot fix" in result.output
    assert "is not NAME=VALUE" in CliRunner().invoke(main, args + ["--fix", "log_nH"]).output
