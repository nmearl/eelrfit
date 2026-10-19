# Command line

EELRFit installs an `eelrfit` command with the following subcommands. Each
one describes its options with `--help`.

```console
eelrfit analyze CUBE -g GRID [options]     # fit one cube
eelrfit grid run SPEC [options]            # compute a grid, or load it from the cache
eelrfit grid show GRID                     # summarize a grid file
eelrfit grid examples                      # list the shipped grid specifications
```

## `eelrfit analyze`

`eelrfit analyze` fits the emission-line maps of a cube with a grid, one
annulus at a time. It writes the results and region tables and prints a
summary.

```console
$ eelrfit analyze manga-10845-12701.Pipe3D.cube.fits.gz -g grid.nc --band 13.6 54.6
Target    : MaNGA 10845-12701
Redshift  : 0.06007 (manga_dr17:nsa_redshift)
Annuli    : 23 (6 constrained), width 0.84"
Fit       : median reduced chi2 6.9 over constrained annuli
Peak      : log_L_band = 44.48 (upper limit) at 3.58 kpc (constrained annuli)
Wrote     : 10845-12701.ecsv and 10845-12701_regions.ecsv
```

| Option | Meaning |
|---|---|
| `-g, --grid FILE` | the grid, as a NetCDF file from `eelrfit grid run` (required) |
| `--sed FILE` | SED for the luminosities (default: the SED the grid was computed with) |
| `-o, --output FILE` | results table (default `<target>.ecsv`); the region table is written next to it |
| `--loader NAME` | loader to read the cube with (default: chosen from the file) |
| `--redshift Z` | use this redshift instead of looking it up in a catalog |
| `--no-catalogs` | skip the catalog lookup and use the header redshift, if there is one |
| `--center NAME` | `continuum` (default) or a candidate center, such as `catalog` |
| `--width ARCSEC` | annulus width (default: a third of the PSF FWHM) |
| `--snr-threshold S` | S/N an annulus needs for a line to count as detected (default 3) |
| `--fix NAME=VALUE` | hold a grid parameter at one of its grid values; can be repeated |
| `--band LO HI` | also compute the luminosity in this energy band, in eV |
| `--model-floor F` | fractional model uncertainty of each line (default 0.1) |
| `--figures DIR` | write diagnostic figures (parameter profiles, luminosity history, line ratios, residuals, posteriors, and BPT diagrams) to this directory |

For example, a fixed-density model with solar N/O is fitted with:

```console
eelrfit analyze cube.fits.gz -g grid.nc --fix log_nH=2.25 --fix log_N_scale=0 -o fixed.ecsv
```

## `eelrfit grid run`

`SPEC` is either a TOML file or the name of a shipped specification (see
[Cloudy grids](grids.md)).

| Option | Meaning |
|---|---|
| `-o, --output FILE` | also write the grid to this NetCDF file |
| `--cloudy-exe FILE` | Cloudy executable (default: `$CLOUDY_EXE`) |
| `--cache-dir DIR` | directory for the run directories (default: `$EELRFIT_GRID_CACHE`, or EELRFit's cache directory) |
| `--cpus N` | number of Cloudy processes (default: the job's CPU allocation) |
| `--timeout SECONDS` | stop Cloudy after this many seconds |
| `--progress-interval SECONDS` | time between progress reports (default 60) |
| `--force` | compute the grid again even if it is cached |
| `--dry-run` | show the plan and the Cloudy input without running Cloudy |

## Environment variables

| Variable | Meaning |
|---|---|
| `CLOUDY_EXE` | path to the Cloudy executable |
| `CLOUDY_DATA_PATH` | Cloudy's data directory, used to find SEDs given as `cloudy = "..."` |
| `EELRFIT_GRID_CACHE` | directory for grid run directories |
| `EELRFIT_CPUS` | number of CPUs to use, overriding the detected value |
