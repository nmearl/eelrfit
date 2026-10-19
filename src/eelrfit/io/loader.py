from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from ..catalogs import Catalog, lookup, merge_header
from ..lines import EmissionLine
from ..maps import EmissionLineMaps, GalaxyProperties

__all__ = ["Loader", "loaders", "read", "register_loader"]

_REGISTRY: dict[str, type[Loader]] = {}


class Loader:
    """Reads one kind of data product into `EmissionLineMaps`.

    Subclasses set `name`, implement `identify` and `read`, and list the
    catalogs their targets are in; `register_loader` makes them available to
    `eelrfit.io.read`. Options (e.g. which extension to read) are arguments of
    the subclass's ``__init__``.
    """

    # Short name, used to select the loader explicitly (``read(path, loader=name)``).
    name: ClassVar[str]
    # Catalogs looked up for the target's properties, merged in order (first value wins).
    catalogs: ClassVar[tuple[Catalog, ...]] = ()

    @classmethod
    def identify(cls, path: Path) -> bool:
        """Whether this loader reads ``path``."""
        raise NotImplementedError

    def read(
        self, path: Path, lines: Iterable[str | EmissionLine] | None = None
    ) -> EmissionLineMaps:
        """Read the maps, without catalog properties (`eelrfit.io.read` adds those)."""
        raise NotImplementedError


def register_loader(cls: type[Loader]) -> type[Loader]:
    """Make a loader available to `read` (usable as a class decorator)."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} has no name")
    _REGISTRY[cls.name] = cls
    return cls


def loaders() -> dict[str, type[Loader]]:
    """The registered loaders by name."""
    return dict(_REGISTRY)


def read(
    path: str | Path,
    lines: Iterable[str | EmissionLine] | None = None,
    *,
    loader: str | type[Loader] | Loader | None = None,
    catalogs: bool | Iterable[Catalog] = True,
    properties: GalaxyProperties | None = None,
    cache: bool | str = True,
) -> EmissionLineMaps:
    """Read a data product and attach the target's galaxy properties.

    Parameters
    ----------
    path : path-like
    lines : iterable of str or EmissionLine, optional
        Lines to read; by default every line the product has.
    loader : str, Loader subclass or instance, optional
        By default the one registered loader whose `Loader.identify` accepts
        the file.
    catalogs : bool or iterable of Catalog
        Where to look up the redshift and candidate centers: the loader's
        catalogs (True), none (False), or the catalogs given.
    properties : GalaxyProperties, optional
        Use these instead of a catalog lookup.
    cache : bool or "update"
        For remote catalogs, passed to `astropy.utils.data.download_file`.

    Notes
    -----
    In every case header values fill what is missing (e.g. a header
    redshift) and disagreements are recorded in ``properties.notes``.
    """
    path = Path(path).expanduser()
    if loader is None:
        matches = [cls for cls in _REGISTRY.values() if cls.identify(path)]
        if len(matches) != 1:
            found = ", ".join(cls.name for cls in matches) or "none"
            raise ValueError(
                f"Cannot choose a loader for {path.name} (matching: {found}); pass loader="
            )
        loader = matches[0]
    elif isinstance(loader, str):
        try:
            loader = _REGISTRY[loader]
        except KeyError:
            raise ValueError(
                f"Unknown loader {loader!r}; registered: {', '.join(_REGISTRY)}"
            ) from None
    if isinstance(loader, type):
        loader = loader()
    maps = loader.read(path, lines)
    if properties is None:
        selected = (
            loader.catalogs
            if catalogs is True
            else () if catalogs is False else tuple(catalogs)
        )
        properties = (
            lookup(selected, maps.meta.target_id, cache)
            if selected
            else GalaxyProperties()
        )
    return maps.with_properties(merge_header(properties, maps.meta))
