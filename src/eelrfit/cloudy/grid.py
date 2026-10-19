from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ..lines import EmissionLine, get_line, line_from_cloudy_label, parse_cloudy_label
from ..sed import SED, read_cloudy_sed

__all__ = ["CloudyGrid", "read_cloudy_grid", "read_grd", "read_lin"]


def _parameter_name(cloudy_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", cloudy_name.lower()).strip("_") or "parameter"


def read_grd(path: str | Path) -> dict[str, Any]:
    """Parse a Cloudy ``save grid`` file.

    Returns
    -------
    dict
        ``parameters`` (sanitized column names, in column order), ``cloudy_names``,
        ``values`` (n_models × n_parameters array), ``failed`` and ``warnings``
        (boolean arrays), and ``exit_code`` (list of str), in grid-index order.
    """
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].startswith("#Index"):
        raise ValueError(f"{path} is not a Cloudy grid file (no '#Index' header)")
    header = lines[0].split("\t")
    try:
        first = header.index("#seq") + 1
        last = header.index("grid parameter string")
    except ValueError:
        raise ValueError(f"Unexpected grid file header in {path}: {header}") from None
    cloudy_names = header[first:last]
    names = [_parameter_name(n) for n in cloudy_names]

    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    order = np.argsort([int(r[0]) for r in rows], kind="stable")
    rows = [rows[i] for i in order]
    values = np.array(
        [[float(r[i]) for i in range(first, last)] for r in rows], dtype=float
    ).reshape(len(rows), len(names))
    return {
        "parameters": names,
        "cloudy_names": cloudy_names,
        "values": values,
        "failed": np.array([r[1].strip() == "T" for r in rows]),
        "warnings": np.array([r[2].strip() == "T" for r in rows]),
        "exit_code": [r[3].strip() for r in rows],
    }


def read_lin(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Parse a Cloudy ``save line list`` file into (labels, n_models × n_lines values)."""
    text = Path(path).read_text().splitlines()
    if not text or not text[0].startswith("#lineslist"):
        raise ValueError(
            f"{path} is not a Cloudy line-list file (no '#lineslist' header)"
        )
    labels = [label.strip() for label in text[0].split("\t")[1:] if label.strip()]
    rows = []
    for line in text[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if not fields[0].startswith("iteration"):
            raise ValueError(f"Unexpected row in {path}: {line[:60]!r}")
        values = [float(v) for v in fields[1:] if v.strip()]
        if len(values) != len(labels):
            raise ValueError(
                f"Row in {path} has {len(values)} values for {len(labels)} lines"
            )
        rows.append(values)
    return labels, np.array(rows, dtype=float).reshape(len(rows), len(labels))


class CloudyGrid:
    """A regular grid of Cloudy models with predicted line intensities.

    Wraps an `xarray.Dataset` with:

    * one dimension per varied parameter (``log_U``, ``log_nH``, ...) and a
      ``line`` dimension holding canonical line ids (or the raw Cloudy label
      for lines not in `eelrfit.lines.LINES`);
    * ``intensity`` (parameters × line): emergent intensity relative to H-beta;
    * ``failed`` and ``warnings`` (parameters): Cloudy's per-model flags;
    * scalar coordinates for parameters held fixed (e.g. ``log_nH`` in a
      fixed-density grid);
    * attributes: ``cloudy_version``, ``intensity_units``, ``input_deck``, and a
      JSON ``provenance`` record.
    """

    def __init__(self, dataset: xr.Dataset):
        for name in ("intensity", "failed", "warnings"):
            if name not in dataset:
                raise ValueError(f"A CloudyGrid dataset needs a {name!r} variable")
        if dataset["intensity"].dims[-1] != "line":
            raise ValueError("'intensity' must have 'line' as its last dimension")
        self._ds = dataset

    @property
    def dataset(self) -> xr.Dataset:
        return self._ds

    @property
    def parameters(self) -> tuple[str, ...]:
        """Varied parameters, in dimension order."""
        return tuple(d for d in self._ds["intensity"].dims if d != "line")

    @property
    def fixed_parameters(self) -> dict[str, float]:
        """Parameters held fixed, as recorded in scalar coordinates."""
        return {name: float(c) for name, c in self._ds.coords.items() if c.ndim == 0}

    @property
    def line_ids(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self._ds["line"].values)

    @property
    def shape(self) -> dict[str, int]:
        return {p: self._ds.sizes[p] for p in self.parameters}

    @property
    def cloudy_version(self) -> str | None:
        return self._ds.attrs.get("cloudy_version") or None

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self._ds.attrs.get("provenance", "{}"))

    def __contains__(self, line: str | EmissionLine) -> bool:
        try:
            return get_line(line).id in self.line_ids
        except KeyError:
            return str(line) in self.line_ids

    def intensity(self, line: str | EmissionLine | None = None) -> xr.DataArray:
        """Intensity relative to H-beta, for one line or all lines."""
        if line is None:
            return self._ds["intensity"]
        try:
            key = get_line(line).id
        except KeyError:
            key = str(line)
        if key not in self.line_ids:
            raise KeyError(
                f"{key} is not in this grid; lines: {', '.join(self.line_ids)}"
            )
        return self._ds["intensity"].sel(line=key)

    def sed(self) -> SED | None:
        """The SED the grid was computed with, if its file is still available (else None).

        Looks for the copy in the grid's run directory, then the recorded
        source path; a recorded checksum must match.
        """
        prov = self.provenance
        recorded = prov.get("sed") or {}
        candidates = [Path(prov["run_dir"]) / "sed.sed"] if prov.get("run_dir") else []
        candidates += [Path(recorded["path"])] if recorded.get("path") else []
        for path in candidates:
            if path.is_file():
                sed = read_cloudy_sed(path)
                if not recorded.get("sha256") or sed.sha256 == recorded["sha256"]:
                    return sed
        return None

    @property
    def usable(self) -> xr.DataArray:
        """Models that did not fail and have finite intensities for every line."""
        finite = np.isfinite(self._ds["intensity"]).all("line")
        return finite & ~self._ds["failed"]

    def to_netcdf(self, path: str | Path) -> Path:
        """Write the grid to a NetCDF file (scipy engine; no extra libraries needed)."""
        path = Path(path)
        self._ds.to_netcdf(path, engine="scipy")
        return path

    @classmethod
    def read(cls, path: str | Path) -> CloudyGrid:
        """Read a grid written by `to_netcdf`."""
        with xr.open_dataset(path, engine="scipy") as ds:
            ds = ds.load()
        for name in ("failed", "warnings"):
            ds[name] = ds[name].astype(bool)
        return cls(ds)

    def __repr__(self) -> str:
        shape = ", ".join(f"{k}={v}" for k, v in self.shape.items())
        fixed = ", ".join(f"{k}={v:g}" for k, v in self.fixed_parameters.items())
        return (
            f"<CloudyGrid Cloudy {self.cloudy_version or '?'} ({shape})"
            + (f" fixed: {fixed}" if fixed else "")
            + f" lines=[{', '.join(self.line_ids)}]>"
        )


def read_cloudy_grid(
    grd: str | Path,
    lin: str | Path,
    *,
    parameters: Sequence[str] | None = None,
    fixed: Mapping[str, float] | None = None,
    input_deck: str = "",
    cloudy_version: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> CloudyGrid:
    """Build a `CloudyGrid` from Cloudy ``.grd`` and ``.lin`` output files.

    Parameters
    ----------
    grd, lin : path-like
        Output of ``save grid`` and ``save line list ... emergent``.
    parameters : sequence of str, optional
        Names of the varied parameters, in ``vary`` order (`GridSpec` axis
        names). Default: the ``.grd`` column names, lowercased.
    fixed : mapping, optional
        Parameters held fixed, recorded as scalar coordinates.
    input_deck : str
        Text of the Cloudy input, recorded.
    cloudy_version : str, optional
        Cloudy version that produced the output (e.g. ``"25.00"``).
    provenance : mapping, optional
        Additional JSON-serializable provenance to record.

    Raises
    ------
    ValueError
        If the files are inconsistent or the models do not form a regular grid.

    Notes
    -----
    The deck must contain ``save grid "<name>.grd" last no hash`` and ``save
    line list "<name>.lin" "<line list>" emergent last no hash`` (as decks
    rendered by `GridSpec` do). Formats, verified with Cloudy c25.00:

    * ``.grd`` is tab-separated, with header ``#Index  Failure?  Warnings?
      Exit code  #rank  #seq  <one column per varied parameter>  grid
      parameter string``. Cloudy truncates the parameter column names
      (``IONIZATIO``; ``ELEMENT S`` for both sulfur and silicon), hence
      ``parameters``.
    * ``.lin`` starts with ``#lineslist`` and tab-separated labels that contain
      spaces (``H  1 6562.80A``); each row is ``iteration N`` plus one value
      per label, in grid-index order. With ``emergent``, values are
      intensities relative to H-beta.
    * Fixed parameters are not in the output, hence ``fixed``.
    """
    meta = read_grd(grd)
    labels, values = read_lin(lin)
    n_models = len(meta["values"])
    if values.shape[0] != n_models:
        raise ValueError(f"{lin} has {values.shape[0]} models but {grd} has {n_models}")

    names = list(parameters) if parameters is not None else meta["parameters"]
    if len(names) != len(meta["parameters"]):
        raise ValueError(
            f"{grd} varies {len(meta['parameters'])} parameters ({meta['cloudy_names']}), not {len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(
            f"Ambiguous grid parameters {meta['cloudy_names']} in {grd}; pass parameters="
        )
    axes = [np.unique(np.round(meta["values"][:, j], 6)) for j in range(len(names))]
    if math.prod(len(a) for a in axes) != n_models:
        raise ValueError(
            f"The {n_models} models in {grd} do not form a regular grid over {names}"
        )
    index = tuple(
        np.searchsorted(axes[j], np.round(meta["values"][:, j], 6))
        for j in range(len(names))
    )
    flat = np.ravel_multi_index(index, [len(a) for a in axes])
    if len(np.unique(flat)) != n_models:
        raise ValueError(f"Duplicate parameter combinations in {grd}")

    shape = [len(a) for a in axes]
    intensity = np.full((*shape, len(labels)), np.nan)
    intensity.reshape(n_models, len(labels))[flat] = values
    failed = np.zeros(shape, dtype=bool)
    failed.reshape(-1)[flat] = meta["failed"]
    warnings = np.zeros(shape, dtype=bool)
    warnings.reshape(-1)[flat] = meta["warnings"]

    ids, wavelengths = [], []
    for label in labels:
        line = line_from_cloudy_label(label)
        ids.append(line.id if line else label)
        wavelengths.append(line.wavelength_aa if line else parse_cloudy_label(label)[1])
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate lines in {lin}: {ids}")

    fixed = {k: float(v) for k, v in (fixed or {}).items() if k not in names}

    coords: dict[str, Any] = {name: (name, axis) for name, axis in zip(names, axes)}
    coords.update(
        line=("line", ids),
        cloudy_label=("line", labels),
        wavelength=("line", np.array(wavelengths)),
    )
    coords.update(fixed)
    dims = (*names, "line")
    ds = xr.Dataset(
        {
            "intensity": (dims, intensity),
            "failed": (tuple(names), failed),
            "warnings": (tuple(names), warnings),
        },
        coords=coords,
        attrs={
            "cloudy_version": cloudy_version or "",
            "intensity_units": "emergent intensity relative to H-beta (Cloudy 'save line list ... emergent')",
            "cloudy_parameter_columns": json.dumps(
                dict(zip(names, meta["cloudy_names"]))
            ),
            "input_deck": input_deck,
            "provenance": json.dumps(dict(provenance or {}), default=str),
        },
    )
    ds["wavelength"].attrs["units"] = "Angstrom"
    return CloudyGrid(ds)
