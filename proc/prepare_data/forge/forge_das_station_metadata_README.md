# FORGE DAS station metadata (78A-32 / 78B-32)

## What this is
Station table for GaMMA / LOKI.

- channel -> depth: tap-test calibrated linear map
- (E_m, N_m): **well trajectory offsets** (E-W, N-S vs TVD) from FORGE_Well_Trajectories_GES_April2022.xlsx with interpolation
- lat/lon: converted from EPSG:26912 -> EPSG:4326

## Channel selection
APPLY_WELL_AB_KEEP=True
- 78A: (92, 1062)
- 78B: (1216, 2385)

## Origin for local coordinates
E0=335865.45 m, N0=4262983.53 m
x_km=(E-E0)/1000, y_km=(N-N0)/1000

## Columns
station_id, well, channel, index, E_m, N_m, lon, lat, elev_m, depth_m, x_km, y_km, z_depth_km, z_elev_km
