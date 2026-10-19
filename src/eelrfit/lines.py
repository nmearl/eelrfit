from __future__ import annotations

import re
from dataclasses import dataclass

import astropy.units as u

__all__ = [
    "LINES",
    "EmissionLine",
    "get_line",
    "line_from_cloudy_label",
    "parse_cloudy_label",
    "species_from_label",
]

_ROMAN = (
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
)


@dataclass(frozen=True)
class EmissionLine:
    """A single emission line (or an unresolved blend).

    Lines are identified by ion and rest-frame air wavelength, as
    ``<element><ion>_<wavelength>`` with the conventional rounded wavelength
    (``OIII_5007``, ``HI_6563``). The free-text labels of pyPipe3D (``Hb``,
    ``[OIII]5007``) and Cloudy (``O  3 5006.84A``) differ between products and
    versions, so they are only used to cross-check a wavelength match.

    Parameters
    ----------
    id : str
        Canonical identifier, e.g. ``"OIII_5007"``.
    element : str
        Element symbol, e.g. ``"O"``.
    ion : int
        Spectroscopic ionization stage (1 = neutral, 2 = singly ionized, ...).
    wavelength_aa : float
        Rest-frame air wavelength in Angstrom.
    forbidden : bool
        Whether the transition is forbidden (written in square brackets).
    blend : bool
        Whether this entry represents an unresolved blend of several lines.
    aliases : tuple of str
        Additional case-insensitive names accepted by `get_line`.
    cloudy_label_override : str or None
        Cloudy line label to use instead of the one generated from the ion and
        wavelength. Needed for multiplets and blends, whose total is stored by
        Cloudy under a ``Blnd`` label (e.g. He I 5876 = ``Blnd 5875.66A``; the
        ``He 1`` label matches only one fine-structure component).
    """

    id: str
    element: str
    ion: int
    wavelength_aa: float
    forbidden: bool = False
    blend: bool = False
    aliases: tuple[str, ...] = ()
    cloudy_label_override: str | None = None

    @property
    def cloudy_label(self) -> str:
        """Label identifying this line in Cloudy output, e.g. ``"O  3 5006.84A"``."""
        if self.cloudy_label_override:
            return self.cloudy_label_override
        return f"{self.element:<2s}{self.ion:>2d} {self.wavelength_aa:.2f}A"

    @property
    def species(self) -> str:
        """Element plus ionization stage in spectroscopic notation, e.g. ``"OIII"``."""
        return f"{self.element}{_ROMAN[self.ion - 1]}"

    @property
    def wavelength(self) -> u.Quantity:
        """Rest-frame air wavelength."""
        return self.wavelength_aa * u.AA

    @property
    def label(self) -> str:
        """Human-readable label, e.g. ``"[OIII] 5007"``."""
        species = f"[{self.species}]" if self.forbidden else self.species
        return f"{species} {self.id.split('_', 1)[1]}"

    def __str__(self) -> str:
        return self.id


def _line(
    id, element, ion, wavelength, forbidden=False, blend=False, aliases=(), cloudy=None
):
    return EmissionLine(
        id, element, ion, wavelength, forbidden, blend, tuple(aliases), cloudy
    )


# Air wavelengths in Angstrom. These are the lines commonly used for
# photoionization modeling of EELRs in optical IFS data; extend as needed.
_CATALOG = (
    _line("HI_6563", "H", 1, 6562.80, aliases=("Ha", "Halpha", "H-alpha")),
    _line("HI_4861", "H", 1, 4861.32, aliases=("Hb", "Hbeta", "H-beta")),
    _line("HI_4340", "H", 1, 4340.46, aliases=("Hg", "Hgamma", "H-gamma")),
    _line("HI_4102", "H", 1, 4101.73, aliases=("Hd", "Hdelta", "H-delta")),
    _line("HeI_5876", "He", 1, 5875.62, blend=True, cloudy="Blnd 5875.66A"),
    _line("HeII_4686", "He", 2, 4685.68),
    _line("OI_6300", "O", 1, 6300.30, forbidden=True),
    _line("OI_6364", "O", 1, 6363.78, forbidden=True),
    _line("OII_3726", "O", 2, 3726.03, forbidden=True),
    _line("OII_3729", "O", 2, 3728.82, forbidden=True),
    _line(
        "OII_3727", "O", 2, 3727.40, forbidden=True, blend=True, cloudy="Blnd 3727.00A"
    ),
    _line("OIII_4363", "O", 3, 4363.21, forbidden=True),
    _line("OIII_4959", "O", 3, 4958.91, forbidden=True),
    _line("OIII_5007", "O", 3, 5006.84, forbidden=True),
    _line("NII_5755", "N", 2, 5754.59, forbidden=True),
    _line("NII_6548", "N", 2, 6548.05, forbidden=True),
    _line("NII_6584", "N", 2, 6583.45, forbidden=True),
    _line("SII_6717", "S", 2, 6716.44, forbidden=True),
    _line("SII_6731", "S", 2, 6730.82, forbidden=True),
    _line("SIII_6312", "S", 3, 6312.06, forbidden=True),
    _line("SIII_9069", "S", 3, 9068.60, forbidden=True),
    _line("SIII_9531", "S", 3, 9530.60, forbidden=True),
    _line("NeIII_3869", "Ne", 3, 3868.75, forbidden=True),
    _line("ArIII_7136", "Ar", 3, 7135.79, forbidden=True),
)

# All known lines, keyed by canonical identifier.
LINES: dict[str, EmissionLine] = {line.id: line for line in _CATALOG}

_LOOKUP: dict[str, EmissionLine] = {}
for _l in _CATALOG:
    for _name in (_l.id, *_l.aliases):
        _LOOKUP[_name.lower()] = _l


def get_line(line: str | EmissionLine) -> EmissionLine:
    """Return the `EmissionLine` for a canonical identifier or alias.

    Raises
    ------
    KeyError
        If the name is not a known identifier or alias.
    """
    if isinstance(line, EmissionLine):
        return line
    try:
        return _LOOKUP[line.lower()]
    except KeyError:
        known = ", ".join(LINES)
        raise KeyError(
            f"Unknown emission line {line!r}. Known lines: {known}"
        ) from None


_HYDROGEN_LABEL = re.compile(
    r"^H(I|\d+|alpha|beta|gamma|delta|epsilon|a|b|g|d)$", re.IGNORECASE
)
_SPECIES_LABEL = re.compile(
    r"^([A-Z][a-z]?)(" + "|".join(sorted(_ROMAN, key=len, reverse=True)) + r")$"
)


def species_from_label(label: str) -> str | None:
    """Infer the species (e.g. ``"OIII"``) from a free-text line label.

    Returns ``None`` when the label cannot be interpreted unambiguously; for
    example pyPipe3D labels H-epsilon as ``"He"``, which is indistinguishable
    from helium.
    """
    text = label.strip()
    if _HYDROGEN_LABEL.match(text):
        return "HI"
    text = re.sub(r"[\[\]]", "", text)
    text = re.sub(r"[\d.]+$", "", text)
    match = _SPECIES_LABEL.match(text)
    return match.group(0) if match else None


_CLOUDY_LABEL = re.compile(r"^\s*([A-Za-z]{1,4})\s*(\d*)\s+([\d.]+)\s*([Am]?)\s*$")
_ELEMENTS = {
    "H",
    "He",
    "C",
    "N",
    "O",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "S",
    "Cl",
    "Ar",
    "Ca",
    "Fe",
    "Ni",
}


def parse_cloudy_label(label: str) -> tuple[str | None, float]:
    """Split a Cloudy line label into species and wavelength in Angstrom.

    ``"O  3 5006.84A"`` gives ``("OIII", 5006.84)``; ``"Blnd 5875.66A"`` gives
    ``(None, 5875.66)``. Wavelengths labeled in microns (``m``) are converted.
    """
    match = _CLOUDY_LABEL.match(label)
    if not match:
        raise ValueError(f"Cannot parse Cloudy line label {label!r}")
    name, stage, wave, unit = match.groups()
    wavelength = float(wave) * (1e4 if unit == "m" else 1.0)
    element = name.capitalize()
    if stage and element in _ELEMENTS:
        return f"{element}{_ROMAN[int(stage) - 1]}", wavelength
    return None, wavelength


def line_from_cloudy_label(
    label: str, tolerance_aa: float = 0.1
) -> EmissionLine | None:
    """The catalog line a Cloudy label refers to, or ``None`` if it is not in the catalog.

    A label matches a line whose `EmissionLine.cloudy_label` is identical (up to
    whitespace), or, for ordinary labels, whose species agrees and whose
    wavelength is within ``tolerance_aa``.
    """
    normalized = " ".join(label.split())
    for line in LINES.values():
        if " ".join(line.cloudy_label.split()) == normalized:
            return line
    species, wavelength = parse_cloudy_label(label)
    if species is None:
        return None
    matches = [
        line
        for line in LINES.values()
        if line.cloudy_label_override is None
        and line.species == species
        and abs(line.wavelength_aa - wavelength) <= tolerance_aa
    ]
    return matches[0] if len(matches) == 1 else None
