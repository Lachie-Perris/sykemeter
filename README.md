Sykemeter
=========

Sykemeter is an automated surf forecast page for a fixed surf spot somewhere in the GBR. 
It combines offshore NOAA GFS Wave forecast data, a local SWAN-derived
wave transfer matrix, wind conditions, and predicted tide data.

Forecast Inputs
---------------

- NOAA GFS Wave: offshore wave height, peak period, wave direction, wind speed,
  and wind direction at the nearest model grid point.
- SWAN wave transfer matrix: translates offshore wave height to the nearshore
  surf spot using modelled representative wave cases.
- Predicted tide data: matched to the forecast window and plotted at high time
  resolution.

Wave Transfer Matrix
--------------------

The transfer matrix was derived from wave buoy observations from OTI.
Historic wave records were filtered for outliers, then representative offshore
conditions were selected with a maximum-dissimilarity algorithm. These cases were
run through SWAN over a local bathymetry grid. At the surf spot, the
transfer coefficient is:

    spot Hs / offshore Hs

The sparse SWAN case outputs were interpolated across circular wave direction
and log-scaled peak period to fill an operational 24 by 18 matrix used for the forecast.

Main Files
----------

- surf_forecast.py: downloads/processes forecast data and generates the final
  forecast SVG.
- wave_transfer_matrix.txt: operational SWAN-derived transfer matrix.
- generate_site.py: GitHub Actions entry point for generating the public SVG.
- docs/index.html: GitHub Pages webpage.
- docs/final_surf_forecast.svg: current public forecast graphic.
- docs/assets/: supporting figures explaining the wave-transfer method.


