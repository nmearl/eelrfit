# EELRFit

EELRFit models spatially resolved emission-line maps with grids of Cloudy
photoionization models. It was written for extended emission-line regions
(EELRs) around active galactic nuclei. EELRFit bins integral-field line maps
into annuli, fits the line fluxes of each annulus with a model grid, and
converts the fitted ionization parameter and density into the ionizing
luminosity required at each radius. The result is a luminosity history of the
nucleus.

EELRFit is in early development, and its interfaces may still change.

## Installation

EELRFit is not yet on PyPI, so install it from a clone of this repository:

```console
pip install .              # or: uv pip install .
pip install '.[plot]'      # with matplotlib, for figures
```

Computing new model grids also requires Cloudy, which
`scripts/install_cloudy.sh` can build. See the
[installation instructions](https://nmearl.github.io/eelrfit/installation/)
for details.

## Quick start

The following commands build Cloudy, compute a model grid, download a MaNGA
cube, and fit it. The
[quick start](https://nmearl.github.io/eelrfit/quickstart/) explains each step.

```console
scripts/install_cloudy.sh                                   # build Cloudy (once)
export CLOUDY_EXE=~/.local/share/cloudy/c25.00/source/cloudy.exe
eelrfit grid run starecheski_agn -o agn_grid.nc             # 527 AGN models, about 15 CPU-hours
curl -fLO https://data.sdss.org/sas/dr17/manga/spectro/pipe3d/v3_1_1/3.1.1/9495/manga-9495-3702.Pipe3D.cube.fits.gz
eelrfit analyze manga-9495-3702.Pipe3D.cube.fits.gz -g agn_grid.nc --figures figures/
```

The same fit in Python:

```python
import eelrfit
from eelrfit.cloudy import CloudyGrid
from eelrfit.pipeline import analyze

maps = eelrfit.read("manga-9495-3702.Pipe3D.cube.fits.gz")
grid = CloudyGrid.read("agn_grid.nc")
result = analyze(maps, grid, sed=grid.sed())   # use the SED the grid was computed with
result.write("9495-3702.ecsv")
```

## Documentation

The documentation is at https://nmearl.github.io/eelrfit/. It includes
installation instructions, a step-by-step quick start, a user guide, a
description of the methodology, and the API reference. The user guide covers
reading data and writing loaders, Cloudy grid specifications, fitting and
results, figures, and the command line.

## Tests

```console
uv run pytest
EELRFIT_EXAMPLE_DATA=/path/to/cubes uv run pytest   # also run the tests on real pyPipe3D cubes
CLOUDY_EXE=/path/to/cloudy.exe uv run pytest         # also run a small Cloudy grid
```
