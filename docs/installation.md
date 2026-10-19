# Installation

EELRFit requires Python 3.13 or later. It is not yet on PyPI, so install it
from a clone of the repository:

```console
pip install .              # or: uv pip install .
pip install '.[plot]'      # with matplotlib, for figures
```

## Cloudy

Fitting only needs an existing grid. Computing new grids requires
[Cloudy](https://www.nublado.org), which is not installed with EELRFit. The
repository includes a script that downloads a Cloudy release, verifies its
checksum, builds it, and runs Cloudy's smoke test. It needs curl, make, perl,
and a C++17 compiler, and takes 10 to 30 minutes and about 4 GB of disk.

```console
scripts/install_cloudy.sh                 # c25.00 into ~/.local/share/cloudy
export CLOUDY_EXE=~/.local/share/cloudy/c25.00/source/cloudy.exe
```

EELRFit finds Cloudy through the `CLOUDY_EXE` environment variable, so add the
`export` line to your shell profile. Alternatively, pass the path as the
`executable` argument of `run_grid`. The script takes the following options.

| Option | Meaning |
|---|---|
| `-v VERSION` | Cloudy release (default `c25.00`) |
| `-p PREFIX` | install directory; Cloudy is placed in `PREFIX/VERSION` |
| `-j JOBS` | number of parallel `make` jobs (default: all CPUs) |
| `-a ARCHIVE` | use a release archive that was already downloaded |

!!! note
    Cloudy compiles the location of its data files into the executable, so do
    not move the installation after building it. Its Makefile may also
    optimize for the local CPU (`-march=native`), so build Cloudy on the same
    kind of machine that will run the grids. On cluster nodes without internet
    access, download the archive on a login node and pass it with `-a`.

## Development

```console
uv sync                                  # the package, matplotlib, and pytest
uv run pytest
EELRFIT_EXAMPLE_DATA=/path/to/cubes uv run pytest   # also run the tests on real pyPipe3D cubes
CLOUDY_EXE=/path/to/cloudy.exe uv run pytest         # also run a small Cloudy grid
uv run --group docs zensical serve                   # serve these docs at http://localhost:8000
```
