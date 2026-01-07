# FORGE DAS station metadata (78A-32 / 78B-32)

## What this is
A GaMMA-ready station table created from tap-test calibrated endpoints (Tables 1 & 2).
Assumptions:
- Wells are treated as **vertical**
- Receiver positions are assigned by **linear interpolation** between wellhead and well-bottom endpoints

## Horizontal coordinates (meters)
- 78A-32: E=335780.84, N=4262991.99
- 78B-32: E=335865.45, N=4262983.53
For each well, E/N are constant for all channels (vertical-well approximation).

## Local coordinates (km)
Origin is set to the 78B wellhead:
- E0=335865.45 m, N0=4262983.53 m
x_km = (E - E0) / 1000
y_km = (N - N0) / 1000

## Channel → depth (linear interpolation)

### 78A-32 (deep -> surface)
Endpoints:
- depth=0.00 m at ch=1080, elev=1701.92 m
- depth=989.90 m at ch=70,   elev=712.02 m
Formulas:
- depth_m(ch) = (ch - 1080) * (989.90 / (70 - 1080))
- elev_m(ch)  = 1701.92 - depth_m(ch)

### 78B-32 (surface -> deep)
Endpoints:
- depth=0.00 m at ch=1196, elev=1705.62 m
- depth=1193.42 m at ch=2400, elev=511.58 m
Formulas:
- depth_m(ch) = (ch - 1196) * (1193.42 / (2400 - 1196))
- elev_m(ch)  = 1705.62 - depth_m(ch)

## z conventions included
- z_depth_km = depth_m / 1000   (depth positive downward)
- z_elev_km  = -elev_m / 1000   (negative elevation)

Use one consistently in GaMMA.

## Array index mapping included
Per your mapping:
- 78A: index = channel - 70
- 78B: index = (channel - 1196) + 1011

## Output files
- forge_das_station_metadata.csv
- forge_das_station_metadata_README.md
