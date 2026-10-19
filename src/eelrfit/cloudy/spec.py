from __future__ import annotations

import dataclasses
import math
import os
import re
import tomllib
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from ..lines import LINES, EmissionLine, get_line

__all__ = ["Axis", "GridSpec", "ParameterRange", "example_specs"]

# Base name of the files a rendered deck tells Cloudy to write.
GRID_BASENAME = "grid"
LINE_LIST_NAME = "linelist.dat"

# Commands EELRFit writes itself; they may not appear in [model] commands.
_RESERVED = re.compile(r"^\s*(save|table\s+sed|grid)\b|\bvary\b", re.IGNORECASE)
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclasses.dataclass(frozen=True)
class ParameterRange:
    """Evenly spaced values, inclusive of both ends (Cloudy ``grid`` syntax)."""

    start: float
    stop: float
    step: float

    def __post_init__(self):
        if self.step <= 0:
            raise ValueError(f"step must be positive; got {self.step}")
        if self.stop < self.start:
            raise ValueError(
                f"stop ({self.stop}) must not be below start ({self.start})"
            )

    @property
    def values(self) -> np.ndarray:
        n = math.floor((self.stop - self.start) / self.step + 1e-6) + 1
        return np.round(self.start + self.step * np.arange(n), 10)

    def __len__(self) -> int:
        return len(self.values)


@dataclasses.dataclass(frozen=True)
class Axis:
    """A grid parameter: a Cloudy command with a ``{value}`` placeholder, and its value or range."""

    name: str
    command: str
    values: float | ParameterRange

    def __post_init__(self):
        if not _NAME.match(self.name):
            raise ValueError(f"Axis name {self.name!r} must be a Python identifier")
        if self.command.count("{value}") != 1:
            raise ValueError(
                f"The command of axis {self.name!r} needs exactly one {{value}}: {self.command!r}"
            )
        if _RESERVED.search(self.command):
            raise ValueError(
                f"The command of axis {self.name!r} may not contain save, table SED, grid or vary"
            )
        values = self.values
        if isinstance(values, (list, tuple)):
            if len(values) != 3:
                raise ValueError(
                    f"The range of axis {self.name!r} is [start, stop, step]; got {values!r}"
                )
            values = ParameterRange(*(float(v) for v in values))
        elif not isinstance(values, ParameterRange):
            values = float(values)
        object.__setattr__(self, "values", values)

    @property
    def varied(self) -> bool:
        return isinstance(self.values, ParameterRange)

    def render(self, ncpus: int | None = None) -> list[str]:
        """Deck lines: the command, and for a range ``vary`` plus a ``grid`` line."""
        if not self.varied:
            return [self.command.format(value=f"{self.values:g}")]
        r = self.values
        cpus = f" ncpus {int(ncpus)}" if ncpus is not None else ""
        return [
            self.command.format(value=f"{r.start:g}") + " vary",
            f"grid {r.start:g} {r.stop:g} {r.step:g}{cpus}",
        ]


@dataclasses.dataclass(frozen=True)
class GridSpec:
    """A Cloudy grid: axes, SED, further commands, and the lines to save.

    Usually read from a TOML file (`GridSpec.read`); the format is described
    in the user guide ("Cloudy grids"), and `example_specs` lists the
    specifications shipped with EELRFit.

    Parameters
    ----------
    axes : sequence of Axis
        In deck order; at least one must vary.
    sed : path-like, optional
        Tabulated SED for Cloudy's ``table SED`` command. Its contents, not
        its name, enter the grid's identity.
    cloudy_sed : str, optional
        Instead of ``sed``: a file in the ``data/SED`` directory of the Cloudy
        installation that runs the grid.
    commands : sequence of str
        Other Cloudy commands, passed through unchanged.
    lines : sequence of str or EmissionLine
        Lines to save; default every line in `eelrfit.lines.LINES`. H-beta is
        always included, since intensities are relative to it.
    description : str
    """

    axes: tuple[Axis, ...]
    sed: Path | None = None
    cloudy_sed: str | None = None
    commands: tuple[str, ...] = ()
    lines: tuple[EmissionLine, ...] = tuple(LINES.values())
    description: str = ""

    def __post_init__(self):
        axes = tuple(self.axes)
        names = [a.name for a in axes]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate axis names: {names}")
        if not any(a.varied for a in axes):
            raise ValueError("At least one axis must vary")
        if (self.sed is None) == (self.cloudy_sed is None):
            raise ValueError("Give exactly one of sed and cloudy_sed")
        commands = tuple(str(c).strip() for c in self.commands)
        for command in commands:
            if _RESERVED.search(command):
                raise ValueError(
                    f"{command!r}: save, table SED, grid and vary commands are written by EELRFit"
                )
        lines = tuple(
            dict.fromkeys(get_line(line) for line in (self.lines or LINES.values()))
        )
        if LINES["HI_4861"] not in lines:
            lines = (LINES["HI_4861"], *lines)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(
            self, "sed", None if self.sed is None else Path(self.sed).expanduser()
        )
        object.__setattr__(self, "commands", commands)
        object.__setattr__(self, "lines", lines)

    @classmethod
    def read(cls, path: str | os.PathLike) -> GridSpec:
        """Read a TOML specification; relative SED paths are relative to the file."""
        path = Path(path)
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh), base_dir=path.parent)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], base_dir: str | os.PathLike = "."
    ) -> GridSpec:
        unknown = set(data) - {"description", "sed", "axes", "model"}
        if unknown:
            raise ValueError(f"Unknown sections or keys: {sorted(unknown)}")
        sed = dict(data.get("sed", {}))
        if set(sed) - {"file", "cloudy"} or len(sed) != 1:
            raise ValueError("[sed] needs exactly one of 'file' and 'cloudy'")
        axes = []
        for name, axis in dict(data.get("axes", {})).items():
            axis = dict(axis)
            if (
                set(axis) - {"command", "range", "value"}
                or ("range" in axis) == ("value" in axis)
                or "command" not in axis
            ):
                raise ValueError(
                    f"Axis {name!r} needs a command and exactly one of range and value"
                )
            axes.append(
                Axis(
                    name,
                    axis["command"],
                    axis["range"] if "range" in axis else axis["value"],
                )
            )
        model = dict(data.get("model", {}))
        if set(model) - {"commands", "lines"}:
            raise ValueError(
                f"Unknown [model] keys: {sorted(set(model) - {'commands', 'lines'})}"
            )
        file = (
            Path(base_dir) / Path(sed["file"]).expanduser() if "file" in sed else None
        )
        return cls(
            axes=tuple(axes),
            sed=file,
            cloudy_sed=sed.get("cloudy"),
            commands=tuple(model.get("commands", ())),
            lines=tuple(model.get("lines", ())),
            description=str(data.get("description", "")),
        )

    @property
    def varied(self) -> tuple[Axis, ...]:
        return tuple(a for a in self.axes if a.varied)

    @property
    def fixed(self) -> dict[str, float]:
        return {a.name: a.values for a in self.axes if not a.varied}

    @property
    def shape(self) -> dict[str, int]:
        return {a.name: len(a.values) for a in self.varied}

    @property
    def n_models(self) -> int:
        return math.prod(self.shape.values())

    def sed_path(self, executable: str | os.PathLike | None = None) -> Path:
        """The SED file; a ``cloudy_sed`` is looked for in ``$CLOUDY_DATA_PATH/SED``, else next to the executable."""
        if self.sed is not None:
            return self.sed
        roots = [
            Path(p) for p in os.environ.get("CLOUDY_DATA_PATH", "").split(":") if p
        ]
        if executable is not None:
            roots.append(
                Path(executable).resolve().parent.parent / "data"
            )  # <cloudy>/source/cloudy.exe
        for root in roots:
            if (root / "SED" / self.cloudy_sed).is_file():
                return root / "SED" / self.cloudy_sed
        raise FileNotFoundError(
            f"{self.cloudy_sed} is not in the SED directory of {[str(r) for r in roots] or 'any known Cloudy data path'}"
        )

    def render_line_list(self) -> str:
        """Contents of the line-list file passed to ``save line list``."""
        return "".join(f"{line.cloudy_label}\n" for line in self.lines)

    def render_input(self, sed_name: str = "sed.sed", ncpus: int | None = None) -> str:
        """The Cloudy input deck. ``ncpus`` limits the processes Cloudy forks (it does not change the physics)."""
        if ncpus is not None and int(ncpus) < 1:
            raise ValueError("ncpus must be at least 1")
        out = [f'table SED "{sed_name}"']
        for axis in self.axes:
            out += axis.render(ncpus)
        out += self.commands
        out.append(f'save grid "{GRID_BASENAME}.grd" last no hash')
        out.append(
            f'save line list "{GRID_BASENAME}.lin" "{LINE_LIST_NAME}" emergent last no hash'
        )
        return "\n".join(out) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable description (as in the TOML file, with resolved paths)."""

        def values(a: Axis):
            return (
                {"range": [a.values.start, a.values.stop, a.values.step]}
                if a.varied
                else {"value": a.values}
            )

        return {
            "description": self.description,
            "sed": (
                {"file": str(self.sed)}
                if self.sed is not None
                else {"cloudy": self.cloudy_sed}
            ),
            "axes": {a.name: {"command": a.command, **values(a)} for a in self.axes},
            "model": {
                "commands": list(self.commands),
                "lines": [line.id for line in self.lines],
            },
        }


def example_specs() -> dict[str, Path]:
    """The specifications shipped with EELRFit, by name (file stem)."""
    root = resources.files("eelrfit.cloudy") / "specs"
    return {
        Path(p.name).stem: Path(str(p))
        for p in root.iterdir()
        if p.name.endswith(".toml")
    }
