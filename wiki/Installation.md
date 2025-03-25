# Installation and Usage Guide

This page provides detailed instructions for installing and using the NZCVM software.

## System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL recommended for best performance)
- **Python**: Python 3.8 or later
- **Disk Space**: Approximately 10GB for the full dataset and code

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/ucgmsim/velocity_modelling.git
cd velocity_modelling
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n velocity_modelling python=3.10
conda activate velocity_modelling
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements include:
- Scientific computing: numpy, scipy, h5py, pandas
- Visualization: diffimg, pygmt, pyvista, matplotlib
- Geospatial: shapely, geopandas
- Testing: hypothesis, pytest
- Project dependencies: qcore, pyyaml, tqdm, typer
- Performance: numba, dask

### Step 4: Download Data (If Not Included in Repository)

Some data files may need to be downloaded separately due to their size:

```bash
# Example script to download data (if provided)
python cvm/tools/download_data.py
```

Alternatively, contact the project maintainers for access to the complete dataset.

## Basic Usage

### Generating a Velocity Model

To generate a velocity model, use the `nzvm.py` script with appropriate arguments:

```bash
python cvm/scripts/nzvm.py generate-velocity-model /path/to/config/nzvm.cfg --out-dir /path/to/output
```

#### Required Arguments

- **Configuration File**: Path to the `nzvm.cfg` file that defines the model parameters
- **Output Directory**: Directory where the output files will be saved

#### Optional Arguments

- **--model-version**: Override the model version specified in the configuration file
- **--log-level**: Set the logging level (DEBUG, INFO, WARNING, ERROR)
- **--format**: Specify the output format (emod3d, csv, or both)

### Example Usage Scenarios

#### Scenario 1: Generate a Wellington Velocity Model

```bash
python cvm/scripts/nzvm.py generate-velocity-model tests/scenarios/Wellington_2p07/nzvm.cfg --out-dir OutDir/Wellington_2p07/Python
```

#### Scenario 2: Generate a Canterbury 1D Velocity Model

```bash
python cvm/scripts/nzvm.py generate-velocity-model tests/scenarios/Cant1D_2p07/nzvm.cfg --out-dir OutDir/Cant1D_2p07/Python
```

#### Scenario 3: Generate a Custom Velocity Model

1. Create a custom configuration file:

```ini
CALL_TYPE=GENERATE_VELOCITY_MOD
MODEL_VERSION=2.07
ORIGIN_LAT=-41.2865
ORIGIN_LON=174.7762
ORIGIN_ROT=0.0
EXTENT_X=50
EXTENT_Y=50
EXTENT_ZMAX=50.0
EXTENT_ZMIN=0.0
EXTENT_Z_SPACING=0.1
EXTENT_LATLON_SPACING=0.1
MIN_VS=0.5
TOPO_TYPE=BULLDOZED
OUTPUT_DIR=/path/to/output
```

2. Run the script with the custom configuration:

```bash
python cvm/scripts/nzvm.py generate-velocity-model /path/to/custom/nzvm.cfg --out-dir /path/to/output
```

## Output Files

After successful execution, the output files will be located in the specified output directory. The main output files include:

- **EMOD3D Files**: Binary files containing velocity and density values
  - `vp3dfile.p`: P-wave velocity values
  - `vs3dfile.s`: S-wave velocity values
  - `rho3dfile.d`: Density values
  - `in_basin_mask.b`: Basin membership (ie. ID of the basin the grid point belongs to. -1 indicates not inside any basin)

- **CSV Files** (if requested): Text files containing velocity and density values in a tabular format
  - `<model_name>.csv`: CSV file with grid points and their properties

For more details on the output formats, see the [Output Formats](OutputFormats.md) page.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Error: "ModuleNotFoundError: No module named 'xyz'"
   - Solution: Install the missing package with `pip install xyz`

2. **Data File Not Found**
   - Error: "FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data/file'"
   - Solution: Ensure that all required data files are downloaded and in the correct location

3. **Memory Errors**
   - Error: "MemoryError" or program crashes without error message
   - Solution: Reduce the size of the model grid or use a machine with more memory

### Getting Help

If you encounter issues not covered in this documentation:

1. Check the project's GitHub issues page for similar problems and solutions
2. Contact the project maintainers through GitHub or email
3. Provide detailed information about your system and the error message when seeking help

## Detailed Configuration Guide

### Model Version System

The NZCVM uses a model version system to define different configurations of the velocity model. This allows users to select different combinations of tomography models, basins, and other parameters.

#### Relationship Between nzvm.cfg and Model Version Files

When you specify a `MODEL_VERSION` in your nzvm.cfg file (e.g., `MODEL_VERSION=2.03`), the system looks for a corresponding YAML file in the `model_version` folder. The naming convention replaces dots with 'p', so for `MODEL_VERSION=2.03`, the system looks for `2p03.yaml`.

Example of `nzvm.cfg` with MODEL_VERSION=2.03:
```ini
CALL_TYPE=GENERATE_VELOCITY_MOD
MODEL_VERSION=2.03  # Will use 2p03.yaml from model_version folder
ORIGIN_LAT=-43.4776
ORIGIN_LON=172.6870
ORIGIN_ROT=23.0
EXTENT_X=20
EXTENT_Y=20
EXTENT_ZMAX=45.0
EXTENT_ZMIN=0.0
EXTENT_Z_SPACING=0.2
EXTENT_LATLON_SPACING=0.2
MIN_VS=0.5
TOPO_TYPE=BULLDOZED
OUTPUT_DIR=/tmp
```

You can also override the model version at runtime using the `--model-version` parameter:

```bash
python cvm/scripts/nzvm.py generate-velocity-model /path/to/nzvm.cfg --out-dir /path/to/output --model-version 2.07
```

#### Explaining 2p03.yaml

The `2p03.yaml` file defines the components and parameters for model version 2.03. Here's a detailed explanation of its contents:

```yaml
GTL: true
surfaces:
  - path: global/surface/NZ_DEM_HD.in
    submodel: ep_tomography_submod_v2010

tomography: 2010_NZ_OFFSHORE
basin_edge_smoothing: true
basins:
  - Canterbury_v19p1
  - NorthCanterbury_v19p1
  - BanksPeninsulaVolcanics_v19p1
  - Kaikoura_v19p1
  - Cheviot_v19p1
  - Hanmer_v19p1
  - Marlborough_v19p1
  - Nelson_v19p1
  - Wellington_v19p6
  - WaikatoHauraki_v19p7
```

**Explanation of key components:**

1. **GTL** (Geotechnical Layer): When set to `true`, enables the use of a geotechnical layer that models near-surface velocity structures based on Vs30 values.

2. **surfaces**: Defines the global surfaces used in the model. Each surface has:
   - **path**: The location of the surface data file
   - **submodel**: The velocity model submodule used for assigning velocity values below this surface
   
   In this example, `NZ_DEM_HD.in` is a high-definition digital elevation model for New Zealand, and `ep_tomography_submod_v2010` is the submodel that computes velocities using the 2010 tomography model.

3. **tomography**: Specifies which tomography model to use. `2010_NZ_OFFSHORE` is an extension of the 2010 Eberhart-Phillips tomography model that includes offshore regions.

4. **basin_edge_smoothing**: When set to `true`, applies smoothing at basin edges to avoid sharp transitions between basin and non-basin regions.

5. **basins**: Lists all the basin models incorporated into this version. Each basin is defined in the `nzcvm_registry.yaml` file with its own surfaces, boundaries, and submodels.

### Creating Custom Model Versions

You can create your own model version by placing a new YAML file in the `model_version` folder. Follow the naming convention of using 'p' in place of dots (e.g., `custom_1p0.yaml` for version 1.0).

Steps to create a custom model version:

1. Create a new YAML file in the `model_version` folder (e.g., `custom_1p0.yaml`)
2. Define the required components (GTL, surfaces, tomography, and basins)
3. Run the model with your custom version by either:
   - Setting `MODEL_VERSION=custom_1.0` in your nzvm.cfg file
   - Using the `--model-version custom_1.0` argument when running nzvm.py

Example of a minimal custom model version:

```yaml
GTL: true
surfaces:
  - path: global/surface/NZ_DEM_HD.in
    submodel: ep_tomography_submod_v2020

tomography: 2020_NZ_OFFSHORE
basin_edge_smoothing: true
basins:
  - Wellington_v19p6
  - GreaterWellington_v19p6
```

This custom version uses the 2020 tomography model and only includes the Wellington and Greater Wellington basins.

## Running the Model

To generate a velocity model using a specific configuration:

1. Prepare your nzvm.cfg file with the desired parameters, including MODEL_VERSION
2. Run the nzvm.py script with the necessary arguments

### Complete Example: Wellington Model with Version 2.07

1. Create or use an existing nzvm.cfg file:

```ini
CALL_TYPE=GENERATE_VELOCITY_MOD
MODEL_VERSION=2.07
ORIGIN_LAT=-41.2865
ORIGIN_LON=174.7762
ORIGIN_ROT=0.0
EXTENT_X=50
EXTENT_Y=50
EXTENT_ZMAX=50.0
EXTENT_ZMIN=0.0
EXTENT_Z_SPACING=0.1
EXTENT_LATLON_SPACING=0.1
MIN_VS=0.5
TOPO_TYPE=BULLDOZED
OUTPUT_DIR=/path/to/output
```

2. Run the script:

```bash
python cvm/scripts/nzvm.py generate-velocity-model /path/to/nzvm.cfg --out-dir /path/to/output
```

3. The script will:
   - Read the configuration from nzvm.cfg
   - Load the model version 2.07 from 2p07.yaml
   - Set up the model grid based on the specified origin and extent
   - Apply the tomography model specified in 2p07.yaml
   - Incorporate the basins listed in 2p07.yaml
   - Generate the velocity model files in the output directory

4. Check the output directory for the generated files:
   - EMOD3D binary files (model.v_p, model.v_s, model.rho)
   - Model info file (model.info)
   - CSV files (if requested)
   - Log files documenting the generation process
