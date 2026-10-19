from __future__ import annotations

import dataclasses
import datetime
from importlib.metadata import version
from pathlib import Path

import astropy.units as u
from astropy.cosmology import Planck18
from astropy.table import QTable

from .cloudy.grid import CloudyGrid
from .fit import FitConfig, RegionFit, fit_regions
from .maps import EmissionLineMaps
from .regions import aggregate, annuli
from .sed import SED

__all__ = ["AnalysisResult", "analyze"]


@dataclasses.dataclass(frozen=True, eq=False)
class AnalysisResult:
    """Outputs of `analyze`.

    ``results`` has one row per region, with the radii, the percentiles and
    best-fit values of the grid parameters, A_V, log Q, and log L_ion, the fit
    statistics, and the lines used; its ``meta`` holds the run metadata (target, grid ranges, SED,
    configuration, software versions). ``regions`` is the aggregated
    measurement table and ``fits`` the full per-region posteriors.
    """

    regions: QTable
    results: QTable
    fits: list[RegionFit]

    def write(self, path, overwrite: bool = False) -> None:
        """Write ``results`` (with metadata) as ECSV; the region table goes alongside."""
        path = Path(path)
        self.results.write(path, format="ascii.ecsv", overwrite=overwrite)
        self.regions.write(
            path.with_name(path.stem + "_regions.ecsv"),
            format="ascii.ecsv",
            overwrite=overwrite,
        )


def analyze(
    maps: EmissionLineMaps,
    grid: CloudyGrid,
    *,
    sed: SED | None = None,
    center="continuum",
    width: u.Quantity | None = None,
    snr_threshold: float = 3.0,
    fit_config: FitConfig | None = None,
    cosmology=Planck18,
) -> AnalysisResult:
    """Run the EELR analysis on one set of maps.

    Steps: annuli around the continuum center → per-region weighted means →
    grid fit with attenuation and model floor → Q(H) and L_ion from the SED.
    The defaults are explained in the Methodology section of the documentation.
    """
    fit_config = fit_config or FitConfig()
    regions = annuli(maps, center=center, width=width)
    table = aggregate(maps, regions, snr_threshold=snr_threshold, cosmology=cosmology)
    results, fits = fit_regions(table, grid, fit_config, sed=sed)
    results.meta.update(
        {
            "target_id": maps.meta.target_id,
            "survey": maps.meta.survey,
            "grid_ranges": results.meta["grid"]["parameters"],
            "grid_fixed": results.meta["grid"]["fixed"],
            "sed_name": sed.path if sed is not None else None,
            "eelrfit_version": version("eelrfit"),
            "created": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            ),
        }
    )
    return AnalysisResult(table, results, fits)
