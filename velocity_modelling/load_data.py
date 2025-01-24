import numpy as np
from typing import List, Dict
from pathlib import Path
from data_format import BasinData, BasinSurfaceRead, VeloMod1DData, GlobalMesh, PartialGlobalMesh, GlobalModelParameters
def load_1d_velo_sub_model(v1d_path: Path) -> VeloMod1DData:
    """
    Load a 1D velocity submodel into memory.

    Parameters
    ----------
    file_name : str
        The filename to open and read.

    Returns
    -------
    VeloMod1DData
        Struct containing a 1D velocity model.
    """

    velo_mod_1d_data = VeloMod1DData()
    try:
        with open(v1d_path, "r") as file:
            # Discard header line
            next(file)
            data = np.loadtxt(file)
            velo_mod_1d_data.Vp = data[:, 0].tolist()
            velo_mod_1d_data.Vs = data[:, 1].tolist()
            velo_mod_1d_data.Rho = data[:, 2].tolist()
            velo_mod_1d_data.Dep = data[:, 5].tolist()
            velo_mod_1d_data.nDep = len(velo_mod_1d_data.Dep)
    except FileNotFoundError:
        print(f"Error 1D velocity model file {v1d_path} not found.")
        exit(1)

    return velo_mod_1d_data


def load_basin_data(basin_names: List[str]):
    """
    Purpose: load all basin data into the basin_data structure

    Input variables:
    basin_data - dict containing basin data (surfaces submodels etc)
    global_model_parameters - dict containing all model parameters (surface names, submodel names, basin names etc)

    Output variables:
    n.a.
    """
    basins = [nzvm_registry_get_basin(name) for name in basin_names]

    basin_data = BasinData(len(basins))

    # loop over nBasins and load in surfaces, boundaries and sub-models
    for i, basin in enumerate(basins):
        basin_data.surf[i] = [load_basin_surface(DATA_ROOT / surface['path']) for surface in basin['surfaces']]
        print("All basin surfaces loaded.")
        basin_data.boundary[i] = [load_basin_boundary(DATA_ROOT / boundary['path']) for boundary in basin['boundaries']]
        print("All basin boundaries loaded.")
        basin_data.basin_submodel_data[i] = [load_basin_submodel(surface) for surface in basin['surfaces']]
        print("All basin sub model data loaded.")

    return basin_data


def load_basin_boundary(basin_boundary_path: Path):
    try:
        data = np.loadtxt(DATA_ROOT / basin_boundary_path)
        lon = data[:, 0]
        lat = data[:, 1]
        boundary_data = np.column_stack((lon, lat))

        assert lon[-1] == lon[0]
        assert lat[-1] == lat[0]
    except FileNotFoundError:
        print(f"Error basin boundary file {basin_boundary_path} not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading basin boundary file {basin_boundary_path}: {e}")
        exit(1)
    return boundary_data


def load_basin_submodel(basin_surface: dict):
    """
    Purpose: load a basin sub-model into the basin_data structure

    Input variables:
    basin_surface - dict containing basin surface data

    Output variables:
    n.a.
    """
    submodel_name = basin_surface['submodel']
    if submodel_name == 'null':
        return None
    submodel = nzvm_registry_get_submodel(submodel_name)
    if submodel is None:
        print(f"Error: submodel {submodel_name} not found.")
        exit(1)

    if submodel['type'] == 'vm1d':
        vm1d = nzvm_registry_get_vm1d(submodel['name'])
        if vm1d is None:
            print(f"Error: vm1d {submodel['name']} not found.")
            exit(1)
        return load_1d_velo_sub_model(DATA_ROOT / vm1d['path'])

    elif submodel['type'] == 'relation':
        # TODO: implement relation submodel loading
        return VeloMod1DData()
    elif submodel['type'] == 'perturbation':
        # TODO: implement perturbation submodel loading
        return VeloMod1DData()


def load_basin_surface(basin_surface_path: Path):
    print(f"Loading basin surface file {basin_surface_path}")

    try:
        with open(basin_surface_path, 'r') as f:
            nLat, nLon = map(int, f.readline().split())
            basin_surf_read = BasinSurfaceRead(nLat, nLon)

            # Reading latitudes and longitudes in one go
            latitudes = np.fromfile(f, dtype=float, count=nLat, sep=' ')
            longitudes = np.fromfile(f, dtype=float, count=nLon, sep=' ')

            basin_surf_read.lati = latitudes
            basin_surf_read.loni = longitudes

            # Reading raster data efficiently
            raster_data = np.fromfile(f, dtype=float, count=nLat * nLon, sep=' ')
            assert len(
                raster_data) == nLat * nLon, f"Error: in {basin_surface_path} raster data length mismatch: {len(raster_data)} != {nLat * nLon}"
            basin_surf_read.raster = raster_data.reshape((nLon, nLat)).T

            firstLat = basin_surf_read.lati[0]
            lastLat = basin_surf_read.lati[nLat - 1]
            basin_surf_read.maxLat = max(firstLat, lastLat)
            basin_surf_read.minLat = min(firstLat, lastLat)

            firstLon = basin_surf_read.loni[0]
            lastLon = basin_surf_read.loni[nLon - 1]
            basin_surf_read.maxLon = max(firstLon, lastLon)
            basin_surf_read.minLon = min(firstLon, lastLon)

            return basin_surf_read

    except FileNotFoundError:
        print(f"Error basin surface file {basin_surface_path} not found.")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def load_tomo_surface_data(tomo_name: str, offshore_surface_name: str = DEFAULT_OFFSHORE_DISTANCE,
                           offshore_v1d_name: str = DEFAULT_OFFSHORE_1D_MODEL) -> VeloMod1DData:
    tomo = nzvm_registry_get_tomography(tomo_name)
    offshore_surface = nzvm_registry_get_surface(offshore_surface_name)
    offshore_v1d = nzvm_registry_get_vm1d(offshore_v1d_name)

    return TomographyData(tomo['elev'], DATA_ROOT / tomo['vs30_path'], tomo['special_offshore_tapering'],
                          DATA_ROOT / tomo['path'], DATA_ROOT / offshore_surface['path'],
                          DATA_ROOT / offshore_v1d['path'])


def load_all_global_data(global_model_params: dict,
                         logger: Logger):  # -> (VeloMod1DData, NZTomographyData, GlobalSurfaces, BasinData):

    """
    Load all data required to generate the velocity model, global surfaces, sub velocity models, and all basin data.

    Parameters
    ----------
    global_model_params : dict
        Struct containing all model parameters (surface names, submodel names, basin names, etc).
    calculation_log : CalculationLog
        Struct containing calculation data and output directory (tracks various parameters for error reporting, etc).
    velo_mod_1d_data : VeloMod1DData
        Struct containing a 1D velocity model.
    nz_tomography_data : NZTomographyData
        Struct containing tomography sub velocity model data (tomography surfaces depths, etc).
    global_surfaces : GlobalSurfaces
        Struct containing pointers to global surfaces (whole domain surfaces which sub velocity models apply between).
    basin_data : BasinData
        Struct containing basin data (surfaces, submodels, etc).

    Returns
    -------
    VeloMod1DData

    NZTomographyData

    GlobalSurfaces

    BasinData
    """
    velo_mod_1d_data = None
    nz_tomography_data = None

    # read in sub velocity models
    print("Loading global velocity submodel data.")
    for i in range(len(global_model_params['velo_submodels'])):
        if global_model_params['velo_submodels'][i] == "v1DsubMod":
            velo_mod_1d_data = load_1d_velo_sub_model(global_model_params['velo_submodels'][i])
            print("Loaded 1D velocity model data.")
        elif global_model_params['velo_submodels'][i] == "NaNsubMod":
            # no data required for NaN velocity sub model, leave as placeholder
            pass
        else:  # tomography sub model EPtomo2010subMod eg. 2020_NZ_OFFSHORE

            nz_tomography_data = load_tomo_surface_data(global_model_params['tomography'])
            print("Loaded tomography data.")

    pass

    # load in vector containing basin 'wall-type' boundaries to apply smoothing near
    if nz_tomography_data is not None:
        nz_tomography_data.smooth_boundary = SmoothingBoundary()
        load_smooth_boundaries(nz_tomography_data, global_model_params['basins'])

    print("Completed loading of global velocity submodel data.")

    # read in global surfaces
    global_surfaces = load_global_surface_data(global_model_params['surfaces'])
    print("Completed loading of global surfaces.")
    #
    # read in basin surfaces and boundaries
    print("Loading basin data.")
    basin_data = load_basin_data(global_model_params['basins'])
    print("Completed loading basin data.")
    print("All global data loaded.")
    return velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data


def load_global_surface(surface_file: Path):
    try:
        with open(surface_file, 'r') as f:
            nLat, nLon = map(int, f.readline().split())
            global_surf_read = GlobalSurfaceRead(nLat, nLon)

            # Reading latitudes and longitudes in one go
            latitudes = np.fromfile(f, dtype=float, count=nLat, sep=' ')
            longitudes = np.fromfile(f, dtype=float, count=nLon, sep=' ')

            global_surf_read.lati = latitudes
            global_surf_read.loni = longitudes

            # Reading raster data efficiently
            raster_data = np.fromfile(f, dtype=float, count=nLat * nLon, sep=' ')
            try:
                global_surf_read.raster = raster_data.reshape((nLon, nLat)).T
            except:
                sys.exit(
                    f"Error: f{surface_file}  data shape {raster_data.shape} does not match lat/lon shape {nLat}x{nLon}={nLat * nLon}")

            firstLat = global_surf_read.lati[0]
            lastLat = global_surf_read.lati[nLat - 1]
            global_surf_read.maxLat = max(firstLat, lastLat)
            global_surf_read.minLat = min(firstLat, lastLat)

            firstLon = global_surf_read.loni[0]
            lastLon = global_surf_read.loni[nLon - 1]
            global_surf_read.maxLon = max(firstLon, lastLon)
            global_surf_read.minLon = min(firstLon, lastLon)

            return global_surf_read

    except FileNotFoundError:
        print(f"Error surface file {surface_file} not found.")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def load_global_surface_data(global_surface_names: List[str]):
    surfaces = [nzvm_registry_get_surface(name) for name in global_surface_names]
    global_surfaces = GlobalSurfaces()

    for surface in surfaces:
        global_surfaces.surf.append(load_global_surface(DATA_ROOT / surface['path']))

    return global_surfaces


# def generate_velocity_model(global_mesh, model_extent, global_model_parameters, velo_mod_1d_data, nz_tomography_data,
#                             global_surfaces, basin_data, calculation_log, gen_extract_velo_mod_call, output_dir,
#                             smoothing_required, n_pts_smooth):


def load_smooth_boundaries(nz_tomography_data: TomographyData, basin_names: List[str]):
    smooth_bound = nz_tomography_data.smooth_boundary
    count = 0

    for basin_name in basin_names:
        basin = nzvm_registry_get_basin(basin_name)
        if "smoothing" in basin:
            boundary_vec_filename = DATA_ROOT / basin["smoothing"]["path"]

            if boundary_vec_filename.exists():
                print(f"Loading offshore smoothing file: {boundary_vec_filename}.")
                try:
                    data = np.fromfile(boundary_vec_filename, dtype=float, sep=' ')
                    x_pts = data[0::2]
                    y_pts = data[1::2]
                    smooth_bound.xPts.extend(x_pts)
                    smooth_bound.yPts.extend(y_pts)
                    count += len(x_pts)
                except Exception as e:
                    print(f"Error reading smoothing boundary vector file {boundary_vec_filename}: {e}")
            else:
                print(f"Error smoothing boundary vector file {boundary_vec_filename} not found.")
        else:
            print(f"Smoothing not required for basin {basin_name}.")
    # print(count)
    # assert count <= MAX_NUM_POINTS_SMOOTH_VEC
    smooth_bound.n = count