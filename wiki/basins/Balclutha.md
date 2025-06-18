# Basin : Balclutha

## Overview
|         |                     |
|---------|---------------------|
| Version | 20p7           |
| Type    | 1        |
| Author  | Cameron Douglas (USER2020)            |
| Created | 2020-07           |


## Images
<a href="../images/maps/SI_se.png"><img src="../images/maps/SI_se.png" width="75%"></a>

*Figure 1 Location*

<a href="../images/regional/Balclutha_basin_map.png"><img src="../images/regional/Balclutha_basin_map.png" width="75%"></a>

*Figure 2 Balclutha Basin Map*

<a href="../images/basins/green_class.png"><img src="../images/basins/green_class.png" width="75%"></a>

*Figure 3 Green Class*

<a href="../images/basins/green_rock.png"><img src="../images/basins/green_rock.png" width="75%"></a>

*Figure 4 Green Rock*


## Notes
- Green area (in green_rock.png) tentatively regarded as a sedimentary rock (soft rock)
- Implemented as two separate basins (Mosgiel/Balclutha) joined by the sedimentary rock
- May need more rigorous classification

## Data
### Boundaries
- Balclutha_outline_WGS84 : [TXT](../../velocity_modelling/data/regional/Balclutha/Balclutha_outline_WGS84.txt) / [GeoJSON](../../velocity_modelling/data/regional/Balclutha/Balclutha_outline_WGS84.geojson)

### Surfaces
- NZ_DEM_HD : [HDF5](../../velocity_modelling/data/global/surface/NZ_DEM_HD.h5) / [TXT](../../velocity_modelling/data/global/surface/NZ_DEM_HD.in) (Submodel: canterbury1d_v2)
- Balclutha_basement_WGS84 : [HDF5](../../velocity_modelling/data/regional/Balclutha/Balclutha_basement_WGS84.h5) / [TXT](../../velocity_modelling/data/regional/Balclutha/Balclutha_basement_WGS84.in) (Submodel: N/A)

### Smoothing Boundaries
- [Balclutha_smoothing.txt](../../velocity_modelling/data/regional/Balclutha/Balclutha_smoothing.txt)

## Data retrieved from
### Boundaries
- [bal_outline_WGS84.txt](https://github.com/ucgmsim/Velocity-Model/tree/main/Data/USER20_BASINS/bal_outline_WGS84.txt)

### Surfaces
- [NZ_DEM_HD.in](https://github.com/ucgmsim/Velocity-Model/tree/main/Data/DEM/NZ_DEM_HD.in)
- [bal_proj_WGS84.in](https://github.com/ucgmsim/Velocity-Model/tree/main/Data/USER20_BASINS/bal_proj_WGS84.in)

---
*Page generated on: June 18, 2025, 17:14 NZST/NZDT*
