# Model Versions

The NZCVM uses a **model version system** as the primary interface between the velocity modeling code and the data repository. Model versions define different configurations of velocity models by specifying which data components to use and how to combine them.

## Overview

Since the code (`velocity_modelling`) and data (`nzcvm_data`) repositories are now separated, the model version YAML files serve as the main **glue** that connects:
- Code-side: Model version YAML files in `model_versions/` folder
- Data-side: `nzcvm_registry.yaml` file that catalogs available datasets

This system is used by multiple tools:
- `generate_3d_model.py` for creating 3D velocity models
- `generate_1d_profiles.py` for extracting 1D velocity profiles
- `extract_cross_section.py` for cross-section analysis
- `generate_threshold_points.py` for threshold point generation

## Model Version Files

Model version files are YAML files located in the `model_versions/` folder. The naming convention replaces dots with 'p' (e.g., `2p03.yaml` for version 2.03).

### File Structure

```yaml
surfaces:
  - path: surface/NZ_DEM_HD.h5
    submodel: ep_tomography_submod_v2010

tomography:
  - name: EP2010
    vs30: nz_with_offshore
    special_offshore_tapering: true
    GTL: true

basin_edge_smoothing: true

basins:
  - Canterbury_v19p1
  - Wellington_v19p6
  - # ... other basins
```

### Component Descriptions

#### Surfaces
Defines global surfaces used in the model:
- **path**: Location of surface data file in the data repository
- **submodel**: Velocity model submodule for assigning velocities below this surface

#### Tomography
Specifies the tomography model configuration:
- **name**: Tomography model identifier (must exist in `nzcvm_registry.yaml`)
- **vs30**: Vs30 model for near-surface velocity adjustments
- **special_offshore_tapering**: Apply specialized offshore velocity model
- **GTL**: Enable Geotechnical Layer model for near-surface adjustments

#### Basin Edge Smoothing
Boolean flag to enable smoothing at basin boundaries.

#### Basins
List of basin models to incorporate (each must be defined in `nzcvm_registry.yaml`).

## Pre-configured Versions

### Version 2.02 (`2p02.yaml`)
- Uses EP2010 tomography model
- Includes 9 basin models (Canterbury, North Canterbury, Banks Peninsula, Kaikoura, Cheviot, Hanmer, Marlborough, Nelson, Wellington v19p6)
- Basic configuration for early model development

### Version 2.03 (`2p03.yaml`)
- Uses EP2010 tomography model
- Adds WaikatoHauraki basin (10 total basins)
- Standard configuration for most applications

### Version 2.07 (`2p07.yaml`)
- **Updated to EP2020 tomography model** (major change from EP2010)
- Extensive basin expansion (34 total basins)
- Includes all major regions: Canterbury, Wellington v21p8, Hawkes Bay, Gisborne, Wairarapa, etc.
- Added South Island basins: Wanaka, Mackenzie, Wakatipu, Alexandra, Ranfurly, Otago regions
- Enhanced regional coverage with Karamea, Collingwood, Springs Junction

### Version 2.08 (`2p08.yaml`)
- Continues with EP2020 tomography model
- Basin model updates and refinements (41 total basins)
- Updated Wellington to v25p5, Kaikoura to v25p5, Nelson to v25p5
- Added more comprehensive coverage: PalmerstonNorth, Southland, TeAnau, Tolaga Bay, Waiapu, West Coast, Westport
- Improved Hanmer basin (v25p3)

### Version 2.09 (`2p09.yaml`)
- Continues with EP2020 tomography model
- Most comprehensive basin coverage (44 total basins)
- Additional basin updates: North Canterbury v25p8, PalmerstonNorth v25p8
- Added specialized regions: Queens Charlotte v25p8, Castle Hill v25p8, Whakatane v25p8
- Most recent model with enhanced basin boundary definitions

### Key Differences Summary

| Version | Tomography | Basin Count | Notable Features |
|---------|------------|-------------|------------------|
| 2.02 | EP2010 | 9 | Basic regional coverage |
| 2.03 | EP2010 | 10 | Adds Waikato-Hauraki |
| 2.07 | **EP2020** | 34 | Major tomography update, extensive basin expansion |
| 2.08 | EP2020 | 39 | Basin version updates, added West Coast regions |
| 2.09 | EP2020 | 42 | Latest comprehensive coverage, specialized regions |

**Recommendation**: Use version 2.09 for new projects requiring the most complete and up-to-date model coverage.

## Creating Custom Versions

### Step 1: Create YAML File
Create a new file in `model_versions/` following the naming convention:
```bash
# For version 1.5
touch model_versions/1p5.yaml
```

### Step 2: Define Components
```yaml
surfaces:
  - path: surface/NZ_DEM_HD.h5
    submodel: ep_tomography_submod_v2020

tomography:
  - name: EP2020
    vs30: nz_with_offshore
    special_offshore_tapering: false
    GTL: true

basin_edge_smoothing: true

basins:
  - Wellington_v19p6
  - Canterbury_v19p1
```

### Step 3: Use Custom Version
Reference in configuration:
```ini
# In nzcvm.cfg
MODEL_VERSION=1.5
```

Or override at runtime:
```bash
python scripts/generate_3d_model.py config.cfg --model-version 1.5
```

## Special Features

### Special Offshore Tapering
When enabled (`special_offshore_tapering: true`):
- Applies specialized 1D velocity model for offshore regions
- Triggered for points with distance from shoreline > 0 and Vs30 < 100 m/s
- Creates smooth transition from land-based to offshore models
- Prevents abrupt velocity changes in shallow waters

### Geotechnical Layer (GTL)
When enabled (`GTL: true`):
- Adjusts near-surface velocities based on Vs30 values
- Uses Ely (2010) methodology
- Provides more accurate shallow velocity structure

## Integration with Data Registry

Model versions work with `nzcvm_registry.yaml` from the data repository:

1. **Model version** specifies *which* components to use
2. **Registry** defines *where* to find those components
3. **Code** combines them according to model version specifications

This separation allows:
- Version control of model configurations independent of data
- Easy switching between different model setups
- Reuse of model versions across different tools

## Best Practices

### Naming Conventions
- Use semantic versioning (e.g., 2.03, 1.5)
- Replace dots with 'p' in filenames (`2p03.yaml`)
- Use descriptive names for custom versions

### Documentation
- Comment complex configurations
- Document rationale for component choices
- Maintain changelog for version updates

### Testing
- Validate custom versions with small test regions
- Compare outputs with established versions
- Test all tools that use the model version

## Troubleshooting

### Common Issues
1. **Missing components**: Ensure all referenced basins/surfaces exist in registry
2. **Version not found**: Check filename follows naming convention
3. **Invalid YAML**: Validate syntax with YAML parser

### Debugging
Use verbose logging to trace model version loading:
```bash
python scripts/generate_3d_model.py config.cfg --log-level DEBUG
```

## Example Usage

```bash
# Use pre-configured version
python scripts/generate_3d_model.py nzcvm.cfg

# Override model version
python scripts/generate_3d_model.py nzcvm.cfg --model-version 2.07

# Generate 1D profiles with specific version
python scripts/generate_1d_profiles.py --model-version 2.03 --location-csv sites.csv

# Extract cross-section using model version
python scripts/extract_cross_section.py model.h5 --model-version 2.07
```
