# EELRFit

Photoionization modeling of spatially resolved emission-line maps with grids
of Cloudy models, built for extended emission-line regions (EELRs) around
active galactic nuclei. EELRFit bins integral-field line maps into annuli,
fits each annulus's line fluxes with a model grid (with dust and a model
uncertainty), and turns the fitted ionization parameter and density into the
ionizing luminosity needed at each radius: a luminosity history of the
nucleus.

*Status: early development; interfaces may change.*

## Installation

EELRFit is not on PyPI yet; install it from a clone of this repository:

```console
pip install .              # or: uv pip install .
pip install '.[plot]'      # with matplotlib, for figures
```

Computing new model grids also needs Cloudy, which `scripts/install_cloudy.sh`
builds (see the documentation).

## Quick start

From nothing to the luminosity history of one MaNGA galaxy (the documentation
explains each step):

```console
scripts/install_cloudy.sh                                   # once: build Cloudy
export CLOUDY_EXE=~/.local/share/cloudy/c25.00/source/cloudy.exe
eelrfit grid run starecheski_agn -o agn_grid.nc             # 527 AGN models, about 15 CPU-hours
curl -fLO https://data.sdss.org/sas/dr17/manga/spectro/pipe3d/v3_1_1/3.1.1/9495/manga-9495-3702.Pipe3D.cube.fits.gz
eelrfit analyze manga-9495-3702.Pipe3D.cube.fits.gz -g agn_grid.nc --figures figures/
```

Or in Python:

```python
import eelrfit
from eelrfit.cloudy import CloudyGrid
from eelrfit.pipeline import analyze

maps = eelrfit.read("manga-9495-3702.Pipe3D.cube.fits.gz")
grid = CloudyGrid.read("agn_grid.nc")
result = analyze(maps, grid, sed=grid.sed())   # sed: the SED the grid was computed with
result.write("9495-3702.ecsv")
```

## Documentation

Installation, a step-by-step quick start, the user guide (reading data and
writing loaders, Cloudy grid specifications, fitting and results, figures, the
command line), the methodology behind the defaults, and the API reference.
Build and view it with [Zensical](https://zensical.org):

```console
uv run --group docs zensical serve      # http://localhost:8000
uv run --group docs zensical build      # static site in site/
```

## Tests

```console
uv run pytest
EELRFIT_EXAMPLE_DATA=/path/to/cubes uv run pytest   # also the tests on real pyPipe3D cubes
CLOUDY_EXE=/path/to/cloudy.exe uv run pytest         # also a tiny real Cloudy grid
```
