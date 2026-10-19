# Cloudy grids

A grid is a regular set of Cloudy models, and the fit compares the line
intensities they predict (relative to Hβ) with the data. Each grid is
described by a short TOML specification and computed with `eelrfit grid run`.
Runs are cached, so the same specification gives the same grid on a laptop or
a cluster. The result is stored as a NetCDF file, which
[`CloudyGrid`][eelrfit.cloudy.CloudyGrid] reads.

## Specifications

Each axis of a grid is a Cloudy command with a `{value}` placeholder, together
with either a range or a single value. All other commands are passed to Cloudy
unchanged.

```toml
description = "AGN models with free N/O"

[sed]
file = "agn.sed"                      # relative to this file
# cloudy = "AGN_Jin12_Eddr_mid.sed"   # or a file in Cloudy's data/SED directory

[axes]
log_U = { command = "ionization parameter {value}", range = [-4.0, -1.0, 0.1] }
log_nH = { command = "hden {value}", range = [0.0, 4.0, 0.25] }
log_N_scale = { command = "element nitrogen scale {value} log", range = [-0.5, 1.0, 0.1] }
# for a fixed axis, give value = 2.0 instead of range

[model]
commands = ["filling factor -5", "stop column density 21", "iterate to convergence"]
lines = []                            # default: every line in eelrfit.LINES (H-beta is always added)
```

Ranges are written as `[start, stop, step]` and include both ends, as in
Cloudy's `grid` command. At least one axis must vary.

The axis names become the dimensions of the grid. Ionizing luminosities can
only be computed if the grid has axes named `log_U` and `log_nH`. If one of
these is held fixed, declare it as an axis with `value = ...` rather than
adding the command to `commands`, so that its value is recorded with the grid.

The SED must be a file in Cloudy's `table SED` format. Its contents, rather
than its name, are part of the grid's identity. With `cloudy = "..."`, EELRFit
looks for the file in `$CLOUDY_DATA_PATH/SED` and then in the data directory
next to the Cloudy executable.

The `commands` list may not include `save`, `table SED`, `grid`, or `vary`,
since EELRFit writes those itself. Any physical setting that is not given is
left at Cloudy's default.

### Shipped specifications

EELRFit includes the following specifications, which `eelrfit grid examples`
lists along with their file paths. Any of them can be copied as a starting
point.

| Name | Models | Description |
|---|---|---|
| `starecheski_agn` | 527 | AGN models of Starecheski et al., with a broken power-law SED, a filling factor of 10⁻⁵, a column density of 10²¹ cm⁻², and solar abundances |
| `starecheski_agn_free_no` | 12,400 | the same models, with $\log n_\mathrm{H}$ extended to 6 and N/O free from −0.5 to +1.0 dex |
| `jin12_agn` | 527 | an example using the Jin et al. (2012) AGN SED that is distributed with Cloudy |

## Running a grid

```console
eelrfit grid run my_grid.toml --dry-run      # show the plan and the Cloudy input
eelrfit grid run my_grid.toml -o my_grid.nc
eelrfit grid show my_grid.nc
```

The specification can be a file or the name of a shipped specification. Each
grid runs in its own directory under `$EELRFIT_GRID_CACHE` (by default,
EELRFit's cache directory). The directory is named after a hash of the input
deck, the line list, the SED contents, and the Cloudy executable, so running
an identical grid again returns the cached result immediately. Use `--force`
to compute it again. The directory keeps the Cloudy input and output, the
parsed `grid.nc`, and a `run.json` record, and all of this is also recorded in
the grid's provenance. While Cloudy runs, progress is printed every minute.

The same can be done from Python.

```python
from eelrfit.cloudy import GridSpec, run_grid

grid = run_grid(GridSpec.read("my_grid.toml"))
grid.intensity("NII_6584")   # relative to H-beta, with one dimension per varied axis
grid.sed()                   # the SED the grid was computed with
```

### CPUs and clusters

Cloudy runs the models of a grid in parallel by forking processes. Left to
itself, it uses every CPU on the machine, even in a batch job that was only
allocated part of a node. EELRFit therefore passes it an explicit number of
processes. The number comes from the first of `--cpus`, `$EELRFIT_CPUS`, the
CPU affinity mask (which reflects the job's allocation on Linux),
`$SLURM_CPUS_PER_TASK`, `$NSLOTS`, and `$PBS_NUM_PPN` that is set, and
otherwise from the number of CPUs on the machine. A grid always runs on a
single node.

Each model of the shipped AGN specifications takes 1.5 to 5 CPU-minutes,
depending on the processor. When all models have finished, Cloudy merges their
output files one at a time. On a network file system this can take more than
15 minutes for a grid of about 10,000 models, so allow for it in the job's
time limit. A job that is stopped during the merge loses the grid.

A minimal SLURM job looks like this:

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=128 --mem=200G --time=12:00:00
export CLOUDY_EXE=$HOME/cloudy/c25.00/source/cloudy.exe
export EELRFIT_GRID_CACHE=/scratch/$USER/eelrfit-grids
eelrfit grid run starecheski_agn_free_no -o $SLURM_SUBMIT_DIR/grid.nc
```

## Existing Cloudy output

Grids computed outside EELRFit can be read with
[`read_cloudy_grid`][eelrfit.cloudy.read_cloudy_grid]. It needs the `.grd` and
`.lin` files written by `save grid` and `save line list ... emergent`, the
names of the varied parameters, and the values of any fixed ones.
