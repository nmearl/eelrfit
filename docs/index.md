# EELRFit

EELRFit models spatially resolved emission-line maps with grids of
[Cloudy](https://www.nublado.org) photoionization models. It was written for
extended emission-line regions (EELRs) around active galactic nuclei. The
ionization of the gas at a given radius reflects the luminosity of the nucleus
when the light that reaches it was emitted, so the radial profile of the
ionizing luminosity traces the recent history of the nucleus.

For each integral-field cube, EELRFit reads the emission-line flux maps and
looks up the galaxy's redshift. It then bins the maps into annuli around the
nucleus, with uncertainties that account for the correlated noise of
reconstructed cubes, and fits the line fluxes of each annulus with a model
grid, marginalizing over dust, the ionization parameter, the density, and any
abundance scalings in the grid. Finally, it converts the fitted ionization
parameter and density into the ionizing photon rate and luminosity required at
that radius, and flags the values that are only upper or lower limits.

![Ionizing luminosity against radius for MaNGA 11965-3703](images/luminosity_history.png)

The [installation](installation.md) page covers EELRFit and Cloudy, and the
[quick start](quickstart.md) goes from a data cube to a luminosity history.
The user guide describes [reading data](guide/data.md),
[Cloudy grids](guide/grids.md), [fitting and results](guide/fitting.md),
[figures](guide/figures.md), and the [command line](guide/cli.md). The
[methodology](methodology.md) page explains the defaults, and the
[API reference](api/index.md) documents every public function and class.

EELRFit is in early development, and its interfaces may still change.
