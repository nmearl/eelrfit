# Methodology

This page describes the default choices in EELRFit's analysis and the reasons
for them. Every default can be changed through function arguments or through
`FitConfig`.

## Data and regions

### Center

Radii are measured from the flux-weighted centroid of the cube's own
continuum map, which for pyPipe3D cubes is the pseudo-V-band map. EELRFit
starts at the peak of the lightly smoothed map and then iterates a centroid
within one PSF FWHM. Because the center is measured on the same pixel grid as
the line maps, it is not affected by astrometric or catalog offsets. Catalog
and header positions remain available as alternatives (for example,
`annuli(center="catalog")`), and an offset of more than 0.5″ from the catalog
position is recorded as a note.

### Annuli

Regions are projected circular annuli, a third of the PSF FWHM wide. Each
annulus is fitted independently, which is valid at any width, and narrow
annuli sample the radial profiles finely. Neighboring annuli are not
independent, however. The PSF FWHM is recorded with the results and drawn as
a scale bar on radial profiles, and the noise correlation between annuli is
reported in `meta["region_noise_correlation"]`. For adjacent annuli in MaNGA
cubes, it is about 0.7.

The radii are not deprojected. Extended emission-line gas does not
necessarily lie in the stellar disk, and a projected radius is a lower limit
on the distance from the nucleus regardless of the geometry.

### Combining spaxels

The flux of a line in a region is the inverse-variance weighted mean of all
valid spaxels (those with a finite flux and a positive error), including
spaxels with negative fluxes. EELRFit does not apply a S/N cut to individual
spaxels, since that would bias faint lines high and discard most of their
signal. Instead, the S/N threshold (3 by default) is applied to the region's
value, and regions below it are flagged rather than dropped.

The uncertainty of the mean includes the correlated noise of reconstructed
cubes. For spaxel errors $\sigma_i$ and a Gaussian correlation between spaxels
a distance $d_{ij}$ apart,

$$
\operatorname{Var}(\bar f) = \frac{\sum_{i,j} \rho_{ij} / (\sigma_i \sigma_j)}{\left(\sum_i \sigma_i^{-2}\right)^2},
\qquad \rho_{ij} = \exp\left(-\frac{d_{ij}^2}{2 s^2}\right),
$$

with $s = 1.92$ spaxels, evaluated for the actual spaxels and weights of each
region. This value reproduces the MaNGA calibration of Westfall et al.
(2019), $1 + 1.62 \log_{10} N$ for compact bins of $N$ spaxels, and is the
value used by the MaNGA Data Analysis Pipeline. It is also applied to other
instruments until they have their own calibration. The scatter of the spaxels
about the mean is kept as a measure of real spatial variation.

### Distances

Physical radii use the angular-diameter distance computed from the redshift
with an explicit astropy cosmology, Planck18 by default. Distances listed in
catalogs are not used.

## Models

### Grids

Model grids are regular Cloudy grids described by TOML specifications
(`eelrfit.cloudy.GridSpec`). Physical settings that a specification does not
give are left at Cloudy's own defaults, and the input deck, the SED checksum,
and the Cloudy build are recorded with every grid. Models that Cloudy reports
as failed, or as completed with warnings, are excluded from the fit. Setting
`FitConfig(allow_warned_models=True)` includes the latter.

### Ionizing luminosity

The ionization parameter sets the rate of hydrogen-ionizing photons reaching
the gas, and the SED converts that rate into a luminosity:

$$
Q(\mathrm{H}) = 4\pi r^2 n_\mathrm{H}\, c\, U,
\qquad
L_\mathrm{band} = Q(\mathrm{H}) \,
\frac{\int_\mathrm{band} L_\nu \, d\nu}{\int_{1\,\mathrm{Ryd}}^{\infty} L_\nu / (h\nu) \, d\nu}.
$$

Here $r$ is the mean projected radius of the spaxels in the region, and the
integrals are taken over the SED file the grid was computed with, read the
same way Cloudy reads it. Cloudy counts every photon above 1 Ryd in $U$, so
the lower limit of the second integral is 1 Ryd. The `log_L_ion` column gives
the luminosity above 1 Ryd, and `FitConfig(band_ev=...)` adds the luminosity
in another band.

Anything that scales all lines by the same factor, such as the filling
factor, the covering factor, or the emitting volume, cancels in the fit. The
filling factor also has very little effect on the line ratios at fixed $U$,
$n_\mathrm{H}$, and column density.

## Fitting

### Lines

The default lines are Hα, Hβ, Hδ, [O I] 6300, [O III] 5007, [N II] 6584, and
the [S II] 6717, 6731 doublet. [O III] 4959 and [N II] 6548 are left out
because their ratios to 5007 and 6584 are fixed by atomic physics. Including
them would add no information and would double the weight of those lines.

### Likelihood

At grid point $\theta$ and attenuation $A_V$, the model flux of line $i$ is

$$
\mu_i = s\, g_i, \qquad g_i = m_i(\theta)\, 10^{-0.4 A_V a_i},
$$

where $m_i$ is the grid intensity relative to Hβ, $a_i = A(\lambda_i)/A_V$
comes from the attenuation curve, and $s$ is a free normalization. The
likelihood is Gaussian,

$$
-2 \ln \mathcal{L} = (\mathbf{f} - s\mathbf{g})^\mathsf{T} C^{-1} (\mathbf{f} - s\mathbf{g}) + \ln \det(2\pi C),
\qquad C_{ii} = \sigma_i^2 + (\epsilon f_i)^2,
$$

where $f_i$ and $\sigma_i$ are the measured flux and its error, and
$\epsilon = 0.1$ is a fractional model uncertainty relative to the *observed*
flux. The normalization is profiled, meaning that at every $(\theta, A_V)$ it
takes its best non-negative value,

$$
\hat{s} = \max\left(0, \frac{\mathbf{g}^\mathsf{T} C^{-1} \mathbf{f}}{\mathbf{g}^\mathsf{T} C^{-1} \mathbf{g}}\right).
$$

Fitting fluxes in this way avoids the correlations that a fit of ratios to Hβ
introduces through the Hβ error, and unlike marginalizing over $s$, it does
not depend on which line the model intensities are normalized to. Since $C$
does not depend on the model, the profile is exact. A model uncertainty
proportional to the model flux, $\epsilon \mu_i$, would instead let the fit
brighten a poorly matching model to increase its own variance.

### The [S II] doublet

The ratio of the two [S II] lines depends almost entirely on the density and
on atomic data, so its model uncertainty is set to 5%, instead of the 14% that
two independent 10% uncertainties would give. To achieve this, the model
uncertainties of the two lines are correlated,

$$
C_{ab} = \left(\epsilon^2 - \tfrac{1}{2}\epsilon_r^2\right) f_a f_b, \qquad \epsilon_r = 0.05,
$$

so that each line keeps an uncertainty of $\epsilon$ while their ratio has an
uncertainty of $\epsilon_r$ (`FitConfig.ratio_floor`).

### Dust

$A_V$ is fitted together with the model on a grid from 0 to 4 mag and
marginalized with a uniform prior. The intrinsic Balmer ratios are therefore
the model's own, rather than a fixed value of 2.86, and Hα is not used twice.
The default attenuation curve is that of Cardelli, Clayton & Mathis (1989)
with the optical coefficients of O'Donnell (1994) and $R_V = 3.1$
(`dust_extinction`'s `O94`). Any callable that returns $A(\lambda)/A_V$ can be
used instead.

### Parameters

All grid parameters, including $\log U$, $\log n_\mathrm{H}$, and any
abundance scalings, have uniform priors in their grid coordinates and are
marginalized. The constraint on the density comes from the [S II] doublet as
predicted by the models themselves. Posteriors are summarized by their 16th,
50th, and 84th percentiles and by the best-fit grid point. A parameter is
flagged as being at a grid edge when more than 5% of its probability lies in
the outermost cell.

### Limits

$Q$ and $L$ are proportional to $n_\mathrm{H}$, so when the density posterior
piles up at an edge of the grid, the luminosity is only a limit. Below the
[S II] low-density limit, at about $10^2\,\mathrm{cm^{-3}}$, the line ratios
constrain $U$ but not $n_\mathrm{H}$, and $L$ is an upper limit
(`log_L_limit`). The median of a limit depends on how far the prior extends
beyond the edge of the grid, so limits are quoted at the 84th percentile for
upper limits and the 16th percentile for lower limits (the `*_quoted`
columns).

### Constrained regions

Every region is fitted and reported, but only regions in which Hβ and at
least four of the fitted lines are detected are flagged as `constrained`.
Summaries such as the peak luminosity should only use constrained regions.
Fits to undetected lines are dominated by the prior, and because
$L \propto r^2$, the outermost of these regions would otherwise dominate the
summaries.
