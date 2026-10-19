# Quick start

This example computes a model grid, fits a MaNGA galaxy, and produces its
luminosity history. It assumes that EELRFit and Cloudy are
[installed](installation.md).

## Compute a model grid

EELRFit ships with a few grid specifications, which `eelrfit grid examples`
lists. The `starecheski_agn` specification describes 527 single-cloud AGN
models, with $\log U$ from −4 to −1 and $\log n_\mathrm{H}$ from 0 to 4. The
grid takes about 15 CPU-hours to compute, or about 2 hours on 8 cores, since
Cloudy runs the models in parallel. Runs are cached, so running the same
command again returns immediately.

```console
eelrfit grid run starecheski_agn --dry-run      # show the plan and the Cloudy input
eelrfit grid run starecheski_agn -o agn_grid.nc
```

## Download a data cube

This example uses the MaNGA DR17 pyPipe3D cube of 9495-3702.

```console
curl -fLO https://data.sdss.org/sas/dr17/manga/spectro/pipe3d/v3_1_1/3.1.1/9495/manga-9495-3702.Pipe3D.cube.fits.gz
```

## Fit the cube

```console
eelrfit analyze manga-9495-3702.Pipe3D.cube.fits.gz -g agn_grid.nc --figures figures/
```

The command reads the cube and looks up its redshift in the MaNGA DR17
catalog, which is downloaded and cached the first time. It bins the line maps
into annuli a third of the PSF FWHM wide around the nucleus and fits each
annulus with the grid. The fitted parameters ($\log U$, $\log n_\mathrm{H}$,
$A_V$, $Q$, and $L_\mathrm{ion}$, as percentiles with limit flags and fit
statistics) are written to `9495-3702.ecsv`, one row per annulus, and the
measured line fluxes to `9495-3702_regions.ecsv`. The command also prints a
summary and saves the diagnostic [figures](guide/figures.md).

In annuli where the [S II] doublet is at its low-density limit, the density,
and therefore the luminosity, is only an upper limit. The
[fitting and results](guide/fitting.md) page explains why.

## The same analysis in Python

```python
import eelrfit
from eelrfit.cloudy import CloudyGrid
from eelrfit.pipeline import analyze

maps = eelrfit.read("manga-9495-3702.Pipe3D.cube.fits.gz")
grid = CloudyGrid.read("agn_grid.nc")
result = analyze(maps, grid, sed=grid.sed())   # use the SED the grid was computed with
result.results                                 # the table of fitted annuli
result.write("9495-3702.ecsv")
```
