from __future__ import annotations

from pathlib import Path

import astropy.units as u
import click
import numpy as np

__all__ = ["main"]


@click.group()
@click.version_option(package_name="eelrfit")
def main():
    """Photoionization modeling of emission-line maps with Cloudy grids."""


# --- analysis ---------------------------------------------------------------------------------


def _fixed(ctx, param, values) -> dict[str, float]:
    out = {}
    for value in values:
        name, sep, number = value.partition("=")
        try:
            out[name.strip()] = float(number)
        except ValueError:
            raise click.BadParameter(f"{value!r} is not NAME=VALUE") from None
        if not sep or not name.strip():
            raise click.BadParameter(f"{value!r} is not NAME=VALUE")
    return out


@main.command()
@click.argument("cube", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-g",
    "--grid",
    "grid_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Cloudy grid (NetCDF, from 'eelrfit grid run').",
)
@click.option(
    "--sed",
    "sed_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="SED for the luminosities (default: the SED the grid was computed with).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Results table (ECSV; default <target>.ecsv). The region table is written alongside.",
)
@click.option(
    "--loader", help="Loader to read CUBE with (default: chosen from the file)."
)
@click.option(
    "--redshift", type=float, help="Use this redshift instead of a catalog lookup."
)
@click.option(
    "--no-catalogs",
    is_flag=True,
    help="Do not look up catalogs (the header redshift, if any, is used).",
)
@click.option(
    "--center",
    default="continuum",
    show_default=True,
    help="'continuum' (centroid of the continuum map) or a candidate center such as 'catalog'.",
)
@click.option(
    "--width",
    type=float,
    help="Annulus width in arcsec (default: a third of the PSF FWHM).",
)
@click.option(
    "--snr-threshold",
    type=float,
    default=3.0,
    show_default=True,
    help="Region S/N for a line to count as detected.",
)
@click.option(
    "--fix",
    "fixed",
    multiple=True,
    callback=_fixed,
    metavar="NAME=VALUE",
    help="Hold a grid parameter at one grid value, e.g. --fix log_nH=2.5 (repeatable).",
)
@click.option(
    "--band",
    nargs=2,
    type=float,
    metavar="LO HI",
    help="Also compute the luminosity in this band (eV).",
)
@click.option(
    "--model-floor",
    type=float,
    default=0.10,
    show_default=True,
    help="Fractional model uncertainty per line.",
)
@click.option(
    "--figures",
    type=click.Path(file_okay=False, path_type=Path),
    help="Write diagnostic figures to this directory.",
)
def analyze(
    cube,
    grid_path,
    sed_path,
    output,
    loader,
    redshift,
    no_catalogs,
    center,
    width,
    snr_threshold,
    fixed,
    band,
    model_floor,
    figures,
):
    """Fit the emission-line maps of CUBE with a Cloudy grid, annulus by annulus.

    Writes per-annulus fits (log U, log nH, A_V, Q, and L_ion, with percentiles
    and limits) and prints a summary.
    """
    from . import read
    from .cloudy import CloudyGrid
    from .fit import FitConfig
    from .maps import GalaxyProperties
    from .pipeline import analyze as run_analysis
    from .sed import read_cloudy_sed

    try:
        properties = (
            GalaxyProperties(redshift=redshift) if redshift is not None else None
        )
        maps = read(
            cube, loader=loader, catalogs=not no_catalogs, properties=properties
        )
        grid = CloudyGrid.read(grid_path)
        sed = read_cloudy_sed(sed_path) if sed_path else grid.sed()
        config = FitConfig(
            fixed=fixed, band_ev=tuple(band) if band else None, model_floor=model_floor
        )
        result = run_analysis(
            maps,
            grid,
            sed=sed,
            center=center,
            snr_threshold=snr_threshold,
            fit_config=config,
            width=width * u.arcsec if width else None,
        )
    except (OSError, ValueError, KeyError, LookupError) as exc:
        raise click.ClickException(str(exc)) from None

    target = maps.meta.target_id
    output = output or Path(f"{target}.ecsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write(output, overwrite=True)
    _summary(maps, result, sed)
    click.echo(
        f"Wrote     : {output} and {output.with_name(output.stem + '_regions.ecsv')}"
    )
    if figures:
        for path in _figures(maps, result, grid, figures):
            click.echo(f"Figure    : {path}")


def _summary(maps, result, sed) -> None:
    props, results = maps.properties, result.results
    z = (
        f"{props.redshift:.5f} ({props.sources.get('redshift')})"
        if props and props.redshift is not None
        else "none (radii in arcsec only)"
    )
    click.echo(f"Target    : {maps.meta.survey} {maps.meta.target_id}")
    click.echo(f"Redshift  : {z}")
    for note in props.notes if props else ():
        click.echo(f"Note      : {note}")
    ok = np.asarray(results["status"] == "ok") & np.asarray(
        results["constrained"], bool
    )
    chi2 = np.asarray(results["chi2_min"], float) / np.maximum(
        np.asarray(results["dof"], float), 1
    )
    click.echo(
        f"Annuli    : {len(results)} ({ok.sum()} constrained), width {result.regions.meta['width_arcsec']:.2f}\""
    )
    if ok.any():
        click.echo(
            f"Fit       : median reduced chi2 {np.median(chi2[ok]):.1f} over constrained annuli"
        )
    quantity = "log_L_band" if "log_L_band_quoted" in results.colnames else "log_L_ion"
    if sed is None:
        click.echo("L_ion     : not computed (no SED: pass --sed)")
    elif f"{quantity}_quoted" in results.colnames and ok.any():
        quoted = np.asarray(results[f"{quantity}_quoted"], float)
        i = int(np.flatnonzero(ok)[np.nanargmax(np.where(ok, quoted, -np.inf)[ok])])
        limit = (
            str(results["log_L_limit"][i]) if "log_L_limit" in results.colnames else ""
        )
        flag = {
            "upper": " (upper limit)",
            "lower": " (lower limit)",
            "unconstrained": " (density unconstrained)",
        }.get(limit, "")
        r = float(u.Quantity(results["r_mean_kpc"][i]).to_value(u.kpc))
        click.echo(
            f"Peak      : {quantity} = {quoted[i]:.2f}{flag} at {r:.2f} kpc (constrained annuli)"
        )


def _figures(maps, result, grid, directory: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from . import plotting as P
    except ImportError:
        raise click.ClickException(
            "Figures need matplotlib: pip install 'eelrfit[plot]'"
        ) from None
    directory.mkdir(parents=True, exist_ok=True)
    stem, results, written = maps.meta.target_id, result.results, []

    def save(fig, name):
        path = directory / f"{stem}_{name}.png"
        fig.savefig(path)
        written.append(path)

    save(P.plot_parameter_profiles(results)[0].figure, "profiles")
    if any(c.endswith("_quoted") and c != "log_Q_quoted" for c in results.colnames):
        save(P.plot_luminosity_history(results).figure, "luminosity")
    save(P.plot_line_ratios(result.regions, result.fits)[0].figure, "line_ratios")
    save(P.plot_residuals(results, result.fits, result.regions).figure, "residuals")
    save(P.plot_posterior_panels(results, result.fits), "posteriors")
    # Model tracks at the median fitted value of any further grid parameter (e.g. N/O).
    ok = np.asarray(results["constrained"], bool)
    grid_at = {
        p: float(np.nanmedian(np.asarray(results[f"{p}_p50"], float)[ok]))
        for p in grid.parameters
        if p not in ("log_U", "log_nH") and f"{p}_p50" in results.colnames and ok.any()
    }
    save(
        P.plot_bpt(
            maps, regions_table=result.regions, grid=grid, max_tracks=4, grid_at=grid_at
        )[0].figure,
        "bpt",
    )
    plt.close("all")
    return written


# --- grids --------------------------------------------------------------------------------------


@main.group()
def grid():
    """Cloudy grids: run TOML specifications and inspect the results."""


def _spec_path(name: str) -> Path:
    from .cloudy import example_specs

    path = Path(name).expanduser()
    if path.is_file():
        return path
    specs = example_specs()
    if name in specs:
        return specs[name]
    raise click.BadParameter(
        f"no such file, nor a shipped specification ({', '.join(sorted(specs))})",
        param_hint="SPEC",
    )


@grid.command("run")
@click.argument("spec")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also write the grid to this NetCDF file.",
)
@click.option(
    "--cloudy-exe",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Cloudy executable (default $CLOUDY_EXE).",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Where run directories live (default $EELRFIT_GRID_CACHE or EELRFit's cache directory).",
)
@click.option(
    "--cpus",
    type=click.IntRange(min=1),
    help="Cloudy processes (default: the job's CPU allocation).",
)
@click.option("--timeout", type=float, help="Seconds before Cloudy is stopped.")
@click.option(
    "--progress-interval",
    type=float,
    default=60.0,
    show_default=True,
    help="Seconds between progress reports.",
)
@click.option("--force", is_flag=True, help="Re-run even if the grid is cached.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the plan and the Cloudy input without running.",
)
def grid_run(
    spec,
    output,
    cloudy_exe,
    cache_dir,
    cpus,
    timeout,
    progress_interval,
    force,
    dry_run,
):
    """Run (or load from the cache) the Cloudy grid of SPEC.

    SPEC is a TOML file or the name of a shipped specification (see 'eelrfit
    grid examples'). Results are cached by their physics and the Cloudy build.
    The CPU count follows the job allocation ($EELRFIT_CPUS, CPU affinity,
    $SLURM_CPUS_PER_TASK, ...) unless --cpus is given.
    """
    from .cloudy import CloudyRunError, GridSpec, run_grid
    from .cloudy.run import (
        _sha256,
        available_cpus,
        default_cache_dir,
        resolve_executable,
    )

    path = _spec_path(spec)
    try:
        grid_spec = GridSpec.read(path)
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(f"{path}: {exc}") from None
    try:
        exe, exe_note = resolve_executable(cloudy_exe), None
    except CloudyRunError as exc:
        if not dry_run:
            raise click.ClickException(str(exc)) from None
        exe, exe_note = None, str(exc)
    try:
        sed = grid_spec.sed_path(exe)
        sed_note = f"{sed} (sha256 {_sha256(sed)[:12]})"
    except FileNotFoundError as exc:
        if not dry_run:
            raise click.ClickException(str(exc)) from None
        sed_note = str(exc)
    cache = cache_dir.expanduser() if cache_dir else default_cache_dir()
    ncpus = min(cpus or available_cpus(), grid_spec.n_models)

    click.echo(f"Grid    : {path}")
    click.echo(
        f"Models  : {grid_spec.n_models} ({', '.join(f'{k} {v}' for k, v in grid_spec.shape.items())})"
    )
    click.echo(f"SED     : {sed_note}")
    click.echo(f"Cloudy  : {exe or exe_note}")
    click.echo(f"CPUs    : {ncpus}")
    click.echo(f"Cache   : {cache}")
    if dry_run:
        click.echo("\n" + grid_spec.render_input(ncpus=ncpus).rstrip())
        return
    try:
        result = run_grid(
            grid_spec,
            executable=exe,
            cache_dir=cache,
            force=force,
            timeout=timeout,
            ncpus=ncpus,
            progress_interval=progress_interval,
            report=click.echo,
        )
    except CloudyRunError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f"Done    : {result!r}")
    click.echo(
        f"Failed  : {int(result.dataset['failed'].sum())} of {grid_spec.n_models} models"
    )
    click.echo(f"Run dir : {result.provenance.get('run_dir')}")
    if output:
        result.to_netcdf(output.expanduser())
        click.echo(f"Wrote   : {output}")


@grid.command("show")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def grid_show(path):
    """Summarize a grid NetCDF file."""
    from .cloudy import CloudyGrid

    result = CloudyGrid.read(path)
    click.echo(repr(result))
    for name in result.parameters:
        values = result.dataset[name].values
        click.echo(
            f"  {name}: {values.min():g} to {values.max():g} ({len(values)} values)"
        )
    click.echo(
        f"  failed models: {int(result.dataset['failed'].sum())}, with warnings: {int(result.dataset['warnings'].sum())}"
    )
    sed = result.sed()
    click.echo(
        f"  SED: {sed.path if sed else 'not available (pass --sed to eelrfit analyze)'}"
    )
    prov = result.provenance
    if prov:
        click.echo(
            f"  run: {prov.get('run_dir', '?')} on {prov.get('host', {}).get('hostname', '?')}, {prov.get('seconds', '?')} s"
        )


@grid.command("examples")
def grid_examples():
    """List the grid specifications shipped with EELRFit."""
    from .cloudy import GridSpec, example_specs

    for name, path in sorted(example_specs().items()):
        spec = GridSpec.read(path)
        click.echo(
            f"{name}  ({spec.n_models} models)\n  {path}\n  "
            + " ".join(spec.description.split())
            + "\n"
        )
