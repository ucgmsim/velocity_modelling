from logging import Logger
from typing import List

import numpy as np
from copy import deepcopy

from velocity_modelling.cvm.constants import MAX_DIST_SMOOTH
from velocity_modelling.cvm.global_model import (
    GlobalSurfaces,
    PartialGlobalSurfaceDepths,
    TomographyData,
)  # noqa: F401
from velocity_modelling.cvm.basin_model import (
    BasinData,
    InBasin,
    PartialBasinSurfaceDepths,
    determine_if_within_basin_lat_lon,
    InBasinGlobalMesh,
)
from velocity_modelling.cvm.geometry import MeshVector
from velocity_modelling.cvm.registry import CVMRegistry
from velocity_modelling.cvm.velocity1d import VelocityModel1D


class PartialGlobalQualities:
    def __init__(self, n_lon: int, n_depth: int):
        self.vp = np.zeros((n_lon, n_depth), dtype=np.float64)
        self.vs = np.zeros((n_lon, n_depth), dtype=np.float64)
        self.rho = np.zeros((n_lon, n_depth), dtype=np.float64)
        self.inbasin = np.zeros((n_lon, n_depth), dtype=np.int8)


class QualitiesVector:
    def __init__(self, n_depth: int):
        self.vp = np.zeros(n_depth, dtype=np.float64)
        self.vs = np.zeros(n_depth, dtype=np.float64)
        self.rho = np.zeros(n_depth, dtype=np.float64)
        self.inbasin = np.zeros(n_depth, dtype=np.int8)

    def prescribe_velocities(
        self,
        global_model_parameters: dict,
        velo_mod_1d_data: VelocityModel1D,
        nz_tomography_data: TomographyData,
        global_surfaces: GlobalSurfaces,
        basin_data_list: List[BasinData],
        mesh_vector: MeshVector,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        partial_basin_surface_depths_list: List[PartialBasinSurfaceDepths],
        in_basin_list: List[InBasin],
        topo_type: str,
        on_boundary: bool,
        calculation_log: Logger,
    ):

        partial_global_surface_depths.interpolate_global_surface_depths(
            global_surfaces, mesh_vector, calculation_log
        )

        dZ = 0
        depth_change = 0

        shifted_mesh_vector = None

        # in_any_basin_lat_lon = any(
        #     [
        #         basin_data.determine_if_within_basin_lat_lon(mesh_vector)
        #         for basin_data in basin_data_list
        #     ]
        # )
        in_any_basin_lat_lon = any(
            in_basin.in_basin_lat_lon for in_basin in in_basin_list
        )
        # TODO: test this
        if topo_type == "SQUASHED":
            shifted_mesh_vector = deepcopy(mesh_vector)

            depth_change = -mesh_vector.z  # is this correct????
            shifted_mesh_vector.z = (
                partial_global_surface_depths.depths[1] - depth_change
            )
        # TODO: test this
        elif topo_type == "SQUASHED_TAPERED":
            dZ = mesh_vector.z[0] - mesh_vector.z[1]
            TAPER_DIST = 1.0
            shifted_mesh_vector = deepcopy(mesh_vector)

            depth_change = -mesh_vector.z
            TAPER_VAL = np.where(
                (depth_change == 0)
                | (partial_global_surface_depths.depths[1] == 0)
                | (partial_global_surface_depths.depths[1] < 0),
                1.0,
                1.0
                - (
                    depth_change
                    / (partial_global_surface_depths.depths[1] * TAPER_DIST)
                ),
            )
            TAPER_VAL = np.clip(TAPER_VAL, 0.0, None)
            shifted_mesh_vector.z = (
                partial_global_surface_depths.depths[1] * TAPER_VAL - depth_change
            )

        elif topo_type in ["BULLDOZED", "TRUE"]:
            shifted_mesh_vector = mesh_vector

        else:
            raise ValueError("User specified TOPO_TYPE not recognised, see readme.")

        for basin_ind, basin_data in enumerate(basin_data_list):
            partial_basin_surface_depths_list[
                basin_ind
            ].interpolate_basin_surface_depths(
                in_basin_list[basin_ind],
                shifted_mesh_vector,
            )

        basin_flag = False
        z = 0

        # TODO: maybe a place to do some parallelization
        for k in range(mesh_vector.nz):
            if topo_type in ["BULLDOZED", "TRUE"]:
                z = mesh_vector.z[k]
            elif topo_type in ["SQUASHED", "SQUASHED_TAPERED"]:
                z = shifted_mesh_vector.z[k]

            for i, basin_data in enumerate(
                basin_data_list
            ):  # TODO: we already know the basin with valid basin surface depth..no need to loop through all basins
                in_basin = in_basin_list[i]
                if in_basin.in_basin_depth[k]:
                    basin_flag = True
                    self.inbasin[k] = i  # basin number

                    self.assign_basin_qualities(
                        partial_basin_surface_depths_list[i],
                        partial_global_surface_depths,
                        nz_tomography_data,
                        mesh_vector,
                        in_any_basin_lat_lon,
                        on_boundary,
                        z,
                        i,  # basin_num
                        k,  # z_ind
                    )

            if not basin_flag:
                self.inbasin[k] = (
                    0  # This is incorrect in the original code. It should be -1. 0 indicates this is inside the 0th basin
                )
                velo_mod_ind = partial_global_surface_depths.find_global_submodel_ind(z)

                velo_mod_name = global_model_parameters["submodels"][velo_mod_ind]
                self.call_global_submodel(
                    velo_mod_name,
                    z,
                    k,
                    global_model_parameters,
                    partial_global_surface_depths,
                    velo_mod_1d_data,
                    nz_tomography_data,
                    mesh_vector,
                    in_any_basin_lat_lon,
                    on_boundary,
                )

            if z > partial_global_surface_depths.depths[1]:
                self.rho[k] = np.nan
                self.vp[k] = np.nan
                self.vs[k] = np.nan

            basin_flag = False

        if topo_type == "BULLDOZED":
            # for k in range(mesh_vector.nz):
            #     if mesh_vector.z[k] > 0:
            #         self.nan_sub_mod(k)
            mask = mesh_vector.z > 0

            self.rho[mask] = np.nan
            self.vp[mask] = np.nan
            self.vs[mask] = np.nan

    def assign_qualities(
        self,
        cvm_registry: CVMRegistry,
        velo_mod_1d_data: VelocityModel1D,
        nz_tomography_data: TomographyData,
        global_surfaces: GlobalSurfaces,
        basin_data_list: List[BasinData],
        mesh_vector: MeshVector,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        partial_basin_surface_depths_list: List[PartialBasinSurfaceDepths],
        in_basin_list: List[InBasin],
        in_basin_mesh: InBasinGlobalMesh,  # New argument
        topo_type: str,
        logger: Logger,
    ):
        """
        Determine if lat-lon point lies within the smoothing zone and prescribe velocities accordingly.
        """
        smooth_bound = nz_tomography_data.smooth_boundary

        closest_ind = None

        if smooth_bound.n == 0:
            distance = (
                1e6  # if there are no points in the smoothing boundary, then skip
            )
        else:
            closest_ind, distance = (
                smooth_bound.determine_if_lat_lon_within_smoothing_region(mesh_vector)
            )

        # calculate vs30 (used as a proxy to determine if point is on- or off-shore, only if using tomography)
        if (
            nz_tomography_data.tomography_loaded
            and cvm_registry.vm_global_params["GTL"]
        ):
            nz_tomography_data.calculate_vs30_from_tomo_vs30_surface(
                mesh_vector
            )  # mesh_vector.vs30 updated
            nz_tomography_data.calculate_distance_from_shoreline(
                mesh_vector
            )  # mesh_vector.distance_from_shoreline updated

        # in_any_basin = any(
        #     [
        #         basin_data.determine_if_within_basin_lat_lon(
        #             mesh_vector
        #         )  # this can be preprocessed in bulk
        #         for basin_data in basin_data_list
        #     ]
        # )

        in_any_basin = any(in_basin.in_basin_lat_lon for in_basin in in_basin_list)

        # point lies within smoothing zone, is offshore, and is not in any basin (i.e., outside any boundaries)
        if (
            distance <= MAX_DIST_SMOOTH
            and not in_any_basin
            and cvm_registry.vm_global_params[
                "GTL"
            ]  # Ely et al. 2010: Geotechnical Layer. If True, then apply the offshore basin depth
            and mesh_vector.vs30 < 100
        ):
            # point lies within smoothing zone and is not in any basin (i.e., outside any boundaries)
            qualities_vector_a = QualitiesVector(mesh_vector.nz)
            qualities_vector_b = QualitiesVector(mesh_vector.nz)
            in_basin_b_list = [
                InBasin(basin_data, mesh_vector.nz) for basin_data in basin_data_list
            ]
            partial_global_surface_depths_b = PartialGlobalSurfaceDepths(
                len(global_surfaces.surfaces)
            )
            partial_basin_surface_depths_list_b = [
                PartialBasinSurfaceDepths(basin_data) for basin_data in basin_data_list
            ]

            original_lat = mesh_vector.lat
            original_lon = mesh_vector.lon

            # overwrite the lat-lon with the location on the boundary
            assert (
                closest_ind is not None
            )  # closest_ind should not be None if distance < MAX_DIST_SMOOTH
            mesh_vector.lat = smooth_bound.y[closest_ind]
            mesh_vector.lon = smooth_bound.x[closest_ind]

            # determine if the point is in any basin
            # Use preprocessed smooth_basin_membership from in_basin_mesh
            # Check and handle if smooth_basin_membership is None
            if in_basin_mesh.smooth_basin_membership is None:
                logger.error(
                    "smooth_basin_membership is None, falling back to manual calculation"
                )
                smooth_indices = []
                for basin_idx, basin_data in enumerate(basin_data_list):
                    if determine_if_within_basin_lat_lon(
                        basin_data, mesh_vector.lat, mesh_vector.lon
                    ):
                        smooth_indices.append(basin_idx)
            else:
                smooth_indices = in_basin_mesh.smooth_basin_membership[closest_ind]

            for i, in_basin in enumerate(in_basin_b_list):
                in_basin.in_basin_lat_lon = i in smooth_indices

            # #TODO: need to update in_basin_b_list with the new lat-lon
            # for i, basin_data in enumerate(basin_data_list):
            #     in_basin_b_list[i].in_basin_lat_lon = any(
            #         [
            #             determine_if_within_basin_lat_lon(basin_data,
            #                 mesh_vector.lat, mesh_vector.lon
            #             )  # this can be preprocessed in bulk
            #         ]
            #     )

            # velocity vector just inside the boundary
            on_boundary = True
            qualities_vector_b.prescribe_velocities(
                cvm_registry.vm_global_params,
                velo_mod_1d_data,
                nz_tomography_data,
                global_surfaces,
                basin_data_list,
                mesh_vector,
                partial_global_surface_depths_b,
                partial_basin_surface_depths_list_b,
                in_basin_b_list,
                topo_type,
                on_boundary,
                logger,
            )

            # overwrite the lat-lon with the original lat-lon point
            mesh_vector.lat = original_lat
            mesh_vector.lon = original_lon

            # velocity vector at the point in question
            on_boundary = False
            qualities_vector_a.prescribe_velocities(
                cvm_registry.vm_global_params,
                velo_mod_1d_data,
                nz_tomography_data,
                global_surfaces,
                basin_data_list,
                mesh_vector,
                partial_global_surface_depths,
                partial_basin_surface_depths_list,
                in_basin_list,
                topo_type,
                on_boundary,
                logger,
            )

            # apply smoothing between the two generated velocity vectors
            smooth_dist_ratio = distance / MAX_DIST_SMOOTH
            inverse_ratio = 1 - smooth_dist_ratio

            valid_indices = ~np.isnan(qualities_vector_a.vp)
            self.vp[valid_indices] = (
                smooth_dist_ratio * qualities_vector_a.vp[valid_indices]
                + inverse_ratio * qualities_vector_b.vp[valid_indices]
            )
            self.vs[valid_indices] = (
                smooth_dist_ratio * qualities_vector_a.vs[valid_indices]
                + inverse_ratio * qualities_vector_b.vs[valid_indices]
            )
            self.rho[valid_indices] = (
                smooth_dist_ratio * qualities_vector_a.rho[valid_indices]
                + inverse_ratio * qualities_vector_b.rho[valid_indices]
            )

            invalid_indices = np.isnan(qualities_vector_a.vp)
            self.vp[invalid_indices] = np.nan
            self.vs[invalid_indices] = np.nan
            self.rho[invalid_indices] = np.nan
        else:
            on_boundary = False
            self.prescribe_velocities(
                cvm_registry.vm_global_params,
                velo_mod_1d_data,
                nz_tomography_data,
                global_surfaces,
                basin_data_list,
                mesh_vector,
                partial_global_surface_depths,
                partial_basin_surface_depths_list,
                in_basin_list,
                topo_type,
                on_boundary,
                logger,
            )

    def assign_basin_qualities(
        self,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        nz_tomography_data: TomographyData,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon: bool,
        on_boundary: bool,
        depth: float,
        basin_num: int,
        z_ind: int,
    ):
        """
        Assign Vp, Vs, and Rho to the individual grid point.

        Parameters
        ----------
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Struct containing depths for all applicable basin surfaces at one lat-lon location.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Struct containing global surface depths.
        nz_tomography_data : TomographyData
            Struct containing tomography data.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        in_any_basin_lat_lon : bool
            Flag indicating if the point is in any basin.
        on_boundary : bool
            Flag indicating if the point is on a boundary.
        depth : float
            The depth of the grid point to determine the properties at.
        basin_num : int
            The basin number pertaining to the basin of interest.
        z_ind : int
            The depth index of the single grid point.

        Returns
        -------
        None
        """
        # Determine the indices of the basin surfaces above and below the given depth
        ind_above = partial_basin_surface_depths.determine_basin_surface_above(depth)
        # ind_below = partial_basin_surface_depths.determine_basin_surface_below(depth)

        # Call the sub-velocity models to assign the qualities
        self.call_basin_submodel(
            partial_basin_surface_depths,
            partial_global_surface_depths,
            nz_tomography_data,
            mesh_vector,
            in_any_basin_lat_lon,
            on_boundary,
            depth,
            ind_above,
            basin_num,
            z_ind,
        )

    def call_global_submodel(
        self,
        submodel_name: str,
        z: float,
        k: int,
        global_model_parameters: dict,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        velo_mod_1d_data: VelocityModel1D,
        nz_tomography_data: TomographyData,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon: bool,
        on_boundary: bool,
    ):
        """
        Call the appropriate sub-velocity model based on the global sub-velocity model index.

        Parameters
        ----------
        submodel_name : str
            The name of the global submodel.
        z : float
            The depth of the grid point to determine the properties at.
        k : int
            The depth index of the single grid point.
        global_model_parameters : dict
            Struct containing all model parameters (surface names, submodel names, basin names, etc.)
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Struct containing global surface depths.
        velo_mod_1d_data : velocity_modelling.cvm.velocity.VelocityModel1D
            Struct containing the 1D velocity model data.
        nz_tomography_data : TomographyData
            Struct containing tomography data.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        in_any_basin_lat_lon : bool
            Flag indicating if the point is in any basin.
        on_boundary : bool
            Flag indicating if the point is on a boundary.

        Returns
        -------
        None
        """

        if submodel_name == "NaNsubMod":
            self.nan_sub_mod(k)

        elif submodel_name == "EPtomo2010subMod":
            from velocity_modelling.cvm.submodel import EPtomo2010 as eptomo2010

            eptomo2010.main(
                k,
                z,
                self,
                mesh_vector,
                nz_tomography_data,
                partial_global_surface_depths,
                global_model_parameters["GTL"],
                in_any_basin_lat_lon,
                on_boundary,
            )

        elif submodel_name == "Cant1D_v1":
            from velocity_modelling.cvm.submodel import Cant1D_v1 as Cant1D_v1

            Cant1D_v1.main(k, z, self, velo_mod_1d_data)
        else:
            raise ValueError(f"Error: Submodel {submodel_name} not found in registry.")

    def call_basin_submodel(
        self,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        nz_tomography_data,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon,
        on_boundary,
        depth,
        ind_above,
        basin_num,
        z_ind,
    ):
        """
        Call the appropriate sub-velocity model based on the basin submodel name.

        Parameters
        ----------
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Struct containing depths for all applicable basin surfaces at one lat-lon location.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Struct containing global surface depths.
        nz_tomography_data : TomographyData
            Struct containing tomography data.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        in_any_basin_lat_lon : bool
            Flag indicating if the point is in any basin.
        on_boundary : bool
            Flag indicating if the point is on a boundary.
        depth : float
            The depth of the grid point to determine the properties at.
        ind_above : int
            Index of the surface directly above the grid point.
        basin_num : int
            The basin number pertaining to the basin of interest.
        z_ind : int
            The depth index of the single grid point.

        Returns
        -------
        None
        """

        basin_data = partial_basin_surface_depths.basin
        self.inbasin[z_ind] = basin_num  # basin number that the point is in

        submodel_name, submodel_data = basin_data.submodels[ind_above]

        if submodel_name == "NaNsubMod":
            self.nan_sub_mod(z_ind)

        elif submodel_name in ["Cant1D_v1", "Cant1D_v2", "Cant1D_v2_Pliocene_Enforced"]:
            from velocity_modelling.cvm.submodel import Cant1D_v1 as Cant1D_v1

            Cant1D_v1.main(z_ind, depth, self, submodel_data)
        elif submodel_name == "PaleogeneSubMod_v1":
            from velocity_modelling.cvm.submodel import (
                PaleogeneSubMod_v1 as PaleogeneSubMod_v1,
            )

            PaleogeneSubMod_v1.main(z_ind, self)
        elif submodel_name == "PlioceneSubMod_v1":
            from velocity_modelling.cvm.submodel import (
                PlioceneSubMod_v1 as PlioceneSubMod_v1,
            )

            PlioceneSubMod_v1.main(z_ind, self)
        elif submodel_name == "MioceneSubMod_v1":
            from velocity_modelling.cvm.submodel import (
                MioceneSubMod_v1 as MioceneSubMod_v1,
            )

            MioceneSubMod_v1.main(z_ind, self)
        elif submodel_name == "BPVSubMod_v1":
            from velocity_modelling.cvm.submodel import BPVSubMod_v1 as BPVSubMod_v1

            BPVSubMod_v1.main(z_ind, self)
        elif submodel_name == "BPVSubMod_v2":
            from velocity_modelling.cvm.submodel import BPVSubMod_v2 as BPVSubMod_v2

            BPVSubMod_v2.main(z_ind, self)
        elif submodel_name == "BPVSubMod_v3":
            from velocity_modelling.cvm.submodel import BPVSubMod_v3 as BPVSubMod_v3

            BPVSubMod_v3.main(z_ind, depth, self, partial_basin_surface_depths)
        elif submodel_name == "BPVSubMod_v4":
            from velocity_modelling.cvm.submodel import BPVSubMod_v4 as BPVSubMod_v4

            BPVSubMod_v4.main(
                z_ind,
                depth,
                self,
                partial_basin_surface_depths,
                partial_global_surface_depths,
            )
        else:
            raise ValueError(f"Error: Submodel {submodel_name} not found in registry.")

    def nan_sub_mod(self, k: int):
        """
        Assign NaN values to the velocities and density at the given depth index.

        Parameters
        ----------
        k : int
            The depth index to assign NaN values to.
        """
        self.rho[k] = np.nan
        self.vp[k] = np.nan
        self.vs[k] = np.nan
