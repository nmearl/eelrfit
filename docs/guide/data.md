# Reading data

All of the analysis works on a single object,
[`EmissionLineMaps`][eelrfit.maps.EmissionLineMaps], which holds flux and
error maps for a set of emission lines on one spatial grid, together with a
celestial WCS, provenance metadata, optional masks and continuum map, and the
galaxy properties adopted for the analysis. Data from any source can be
analyzed once they are in this form.

## pyPipe3D cubes

[`eelrfit.read`][eelrfit.io.read] reads the pyPipe3D products of MaNGA, SAMI,
and KILOGAS.

```python
import eelrfit

maps = eelrfit.read("manga-9495-3702.Pipe3D.cube.fits.gz")
maps                       # <EmissionLineMaps MaNGA:9495-3702 shape=(42, 42) lines=[HI_6563, ...]>
maps.flux_map("OIII_5007") # one line, as an astropy Quantity
maps.snr("Ha")             # lines can be named by id or alias
maps.properties.redshift   # 0.022224, from the MaNGA DR17 catalog
maps.properties.sources    # where each value came from
```

By default, `read` looks for every line in the line catalog (`eelrfit.LINES`)
and lists the ones it cannot find in `maps.unavailable`. With `lines=[...]`,
it reads only the requested lines and raises an error if one is missing.
Fluxes are stored as delivered. Negative fluxes are kept, since they are
legitimate noisy measurements of faint lines, and no mask is applied. A
measurement is considered valid where both the flux and the error are finite
and the error is positive.

Lines are identified by ion and rest-frame air wavelength, such as `OIII_5007`
or `HI_6563`. The labels in pyPipe3D files are not unique and differ between
its two emission-line extensions, so EELRFit matches the planes by wavelength
and uses the label only as a cross-check.

| Loader | Files | Redshift and centers |
|---|---|---|
| [`manga`][eelrfit.io.MaNGALoader] | `manga-<plate>-<ifu>.Pipe3D.cube.fits.gz` (SDSS DR17) | DR17 Pipe3D summary table |
| [`sami`][eelrfit.io.SAMILoader] | SAMI-based pyPipe3D cubes | file header (`Z_SPEC`) |
| [`kilogas`][eelrfit.io.KILOGASLoader] | `KG-MaNGA-*` and `KG-SAMI-*` | KILOGAS tables |

`read` chooses the loader from the file. To use a specific loader, pass its
name (`loader="manga"`) or an instance. Loader options are arguments of the
loader class. For example,
`read(path, loader=MaNGALoader(extension="FLUX_ELINES"))` reads pyPipe3D's
shorter line list.

## Galaxy properties and catalogs

Each loader declares the catalogs that contain its targets, and `read` looks
the target up in them when it reads the file. The result is stored in
`maps.properties` as a [`GalaxyProperties`][eelrfit.maps.GalaxyProperties]
object, which holds the redshift and a set of candidate centers, along with
the source of each value. Values from the file header fill in anything the
catalogs do not provide, and any disagreement between the two is recorded in
`properties.notes`.

```python
eelrfit.read(path)                                   # use the loader's catalogs
eelrfit.read(path, catalogs=False)                   # skip the lookup and any download
eelrfit.read(path, properties=eelrfit.GalaxyProperties(redshift=0.03))   # supply the values
eelrfit.read(path, catalogs=[my_catalog])            # use other catalogs
```

A [`Catalog`][eelrfit.catalogs.Catalog] describes a FITS table and the columns
that hold the values.

```python
from eelrfit.catalogs import Catalog

my_catalog = Catalog(
    name="my_sample",
    source="https://example.org/sample.fits",        # or a local path
    id_column="target",                              # matched against maps.meta.target_id
    redshift="z",
    centers={"catalog": ("ra", "dec")},              # in degrees
)
```

Remote tables are downloaded once and cached by astropy. Distances are not
taken from catalogs. Physical radii are computed from the redshift and a
cosmology instead, as described in the [methodology](../methodology.md).

## Other data

Products that EELRFit cannot read can still be analyzed by constructing the
maps directly.

```python
import astropy.units as u
from eelrfit import EmissionLineMaps, GalaxyProperties, SourceMetadata

maps = EmissionLineMaps(
    lines=("HI_4861", "OIII_5007", "HI_6563", "NII_6584", "SII_6717", "SII_6731"),
    flux=flux * u.Unit("1e-17 erg s-1 cm-2"),        # shape (n_lines, ny, nx)
    error=error * u.Unit("1e-17 erg s-1 cm-2"),
    wcs=wcs,                                         # 2-D celestial WCS
    meta=SourceMetadata(survey="mine", instrument="MUSE", target_id="NGC 1234", psf_fwhm=0.8 * u.arcsec),
    continuum=continuum_map,                         # optional; used to find the nucleus
    properties=GalaxyProperties(redshift=0.01),
)
```

The PSF FWHM sets the default annulus width, and the continuum map is used to
find the default center. If either is missing, pass `width` or `center` to the
analysis.

To read a product regularly, write a loader. Subclass
[`Loader`][eelrfit.io.Loader], or [`Pipe3DLoader`][eelrfit.io.Pipe3DLoader]
for pyPipe3D cubes (which only requires `identify` and `metadata`), and
register it.

```python
from pathlib import Path
from eelrfit.io import Loader, register_loader

@register_loader
class MyLoader(Loader):
    name = "mine"
    catalogs = (my_catalog,)

    @classmethod
    def identify(cls, path: Path) -> bool:
        return path.name.endswith(".mine.fits")

    def read(self, path, lines=None):
        ...                                          # build and return EmissionLineMaps
```

`read` then uses the new loader for any file it identifies and looks up its
catalogs. If more than one loader accepts a file, `read` raises an error and
asks you to choose one with `loader=`.
