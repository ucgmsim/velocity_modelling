from logging import Logger
from typing import List

import numpy as np

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

        shifted_mesh_vector = None

        in_any_basin_lat_lon = any(
            in_basin.in_basin_lat_lon for in_basin in in_basin_list
        )
        # TODO: test this
        if topo_type == "SQUASHED":
            depth_change = -mesh_vector.z
            shifted_mesh_vector = MeshVector(
                lat=mesh_vector.lat,
                lon=mesh_vector.lon,
                z=partial_global_surface_depths.depths[1] - depth_change,
                nz=mesh_vector.nz,  # Include other necessary attributes
            )
        # TODO: test this
        elif topo_type == "SQUASHED_TAPERED":
            dZ = mesh_vector.z[0] - mesh_vector.z[1]
            TAPER_DIST = 1.0
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
            shifted_mesh_vector = MeshVector(
                lat=mesh_vector.lat,
                lon=mesh_vector.lon,
                z=partial_global_surface_depths.depths[1] * TAPER_VAL - depth_change,
                nz=mesh_vector.nz,
            )

        elif topo_type in ["BULLDOZED", "TRUE"]:
            shifted_mesh_vector = mesh_vector

        else:
            raise ValueError("User specified TOPO_TYPE not recognised, see readme.")

        if in_any_basin_lat_lon:
            for i, in_basin in enumerate(in_basin_list):
                if (
                    in_basin.in_basin_lat_lon
                ):  # Only interpolate if the point is in the basin
                    partial_basin_surface_depths_list[
                        i
                    ].interpolate_basin_surface_depths(
                        in_basin,
                        shifted_mesh_vector,
                    )

        # Initialize arrays
        self.inbasin = np.full(mesh_vector.nz, -1, dtype=int)
        self.vp = np.zeros(mesh_vector.nz)
        self.vs = np.zeros(mesh_vector.nz)
        self.rho = np.zeros(mesh_vector.nz)

        # Compute z values for all depths
        z_values = (
            mesh_vector.z
            if topo_type in ["BULLDOZED", "TRUE"]
            else shifted_mesh_vector.z
        )
        k_indices = np.arange(mesh_vector.nz)

        # Precompute basin membership for all depths
        in_basin_depths = np.array(
            [in_basin.in_basin_depth for in_basin in in_basin_list]
        )  # Shape: (34, 225)
        basin_mask = in_basin_depths  # Shape: (34, 225)
        # Find the first basin index for each depth (reverse axis to match sequential order)
        # basin_indices = np.argmax(basin_mask, axis=0)  # This finds the last True index
        # To find the first True index, reverse the basin order and use argmax, then adjust
        basin_indices = np.argmax(basin_mask[::-1], axis=0)
        basin_per_k = np.where(
            np.any(basin_mask, axis=0), len(basin_mask) - 1 - basin_indices, -1
        )

        # Identify depths in basins vs. not in basins
        in_basin_mask = basin_per_k >= 0
        out_basin_mask = ~in_basin_mask

        # Process depths in basins
        if np.any(in_basin_mask):
            k_in_basin = k_indices[in_basin_mask]
            z_in_basin = z_values[in_basin_mask]
            basins_in_basin = basin_per_k[in_basin_mask]

            # Debug: Log basin assignments
            # print(f"Points in basins: {len(k_in_basin)} depths")
            # print(f"Basins assigned: {np.unique(basins_in_basin)}")

            # Group by basin to process each basin's depths together
            for basin_ind in np.unique(basins_in_basin):
                basin_subset_mask = basins_in_basin == basin_ind
                k_subset = k_in_basin[basin_subset_mask]
                z_subset = z_in_basin[basin_subset_mask]

                # Vectorized call to assign_basin_qualities
                self.assign_basin_qualities_vectorized(
                    partial_basin_surface_depths_list[basin_ind],
                    partial_global_surface_depths,
                    nz_tomography_data,
                    mesh_vector,
                    in_any_basin_lat_lon,
                    on_boundary,
                    z_subset,
                    basin_ind,
                    k_subset,
                )

            # Debug: Log intermediate vp, vs, rho values
            # print(f"Basin {basin_ind}, k_subset={k_subset}")
            # print(f"vp[{k_subset}]:", self.vp[k_subset])
            # print(f"vs[{k_subset}]:", self.vs[k_subset])
            # print(f"rho[{k_subset}]:", self.rho[k_subset])

        # Process depths not in any basin
        if np.any(out_basin_mask):
            k_out_basin = k_indices[out_basin_mask]
            z_out_basin = z_values[out_basin_mask]

            # Vectorized call to find submodel indices
            velo_mod_indices = (
                partial_global_surface_depths.find_global_submodel_ind_vectorized(
                    z_out_basin
                )
            )
            velo_mod_names = np.array(global_model_parameters["submodels"])[
                velo_mod_indices
            ]
            # print(f"Points not in basins: {len(k_out_basin)} depths")
            # print(f"Submodel indices: {np.unique(velo_mod_indices)}")
            # print(f"Submodel names: {np.unique(velo_mod_names)}")

            # Vectorized call to call_global_submodel
            self.call_global_submodel_vectorized(
                velo_mod_names,
                z_out_basin,
                k_out_basin,
                global_model_parameters,
                partial_global_surface_depths,
                velo_mod_1d_data,
                nz_tomography_data,
                mesh_vector,
                in_any_basin_lat_lon,
                on_boundary,
            )

            # Debug: Log intermediate vp, vs, rho values
            # print(f"Non-basin points, k_out_basin={k_out_basin}")
            # print(f"vp[{k_out_basin}]:", self.vp[k_out_basin])
            # print(f"vs[{k_out_basin}]:", self.vs[k_out_basin])
            # print(f"rho[{k_out_basin}]:", self.rho[k_out_basin])

        # Apply NaN masking for depths above the surface
        mask_above_surface = z_values > partial_global_surface_depths.depths[1]
        self.rho[mask_above_surface] = np.nan
        self.vp[mask_above_surface] = np.nan
        self.vs[mask_above_surface] = np.nan

        # Debug: Check NaN masking
        # print(f"Points above surface: {np.sum(mask_above_surface)}")

        # Apply NaN masking for bulldozed topography
        if topo_type == "BULLDOZED":
            mask_above_zero = mesh_vector.z > 0
            self.rho[mask_above_zero] = np.nan
            self.vp[mask_above_zero] = np.nan
            self.vs[mask_above_zero] = np.nan

            # Debug: Check bulldozed masking
            # print(f"Points above zero (BULLDOZED): {np.sum(mask_above_zero)}")

        # Debug: Final output
        # print(f"j={j}, k={k}, final vp:", self.vp)
        # print(f"j={j}, k={k}, final vs:", self.vs)
        # print(f"j={j}, k={k}, final rho:", self.rho)
        # print(f"j={j}, k={k}, final inbasin:", self.inbasin)

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

    # TODO: I need to fix this function to automatically trigger the right main function of the submodel
    def call_global_submodel_vectorized(
        self,
        submodel_names: np.ndarray,
        z_values: np.ndarray,
        k_indices: np.ndarray,
        global_model_parameters: dict,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        velo_mod_1d_data: VelocityModel1D,
        nz_tomography_data: TomographyData,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon: bool,
        on_boundary: bool,
    ):
        for name in np.unique(submodel_names):
            mask = submodel_names == name
            z_subset = z_values[mask]
            k_subset = k_indices[mask]

            if name == "NaNsubMod":
                self.nan_sub_mod_vectorized(k_subset)
            elif name == "EPtomo2010subMod":
                from velocity_modelling.cvm.submodel import EPtomo2010

                EPtomo2010.main_vectorized(
                    k_subset,
                    z_subset,
                    self,
                    mesh_vector,
                    nz_tomography_data,
                    partial_global_surface_depths,
                    global_model_parameters["GTL"],
                    in_any_basin_lat_lon,
                    on_boundary,
                )
            elif name == "Cant1D_v1":
                from velocity_modelling.cvm.submodel import Cant1D_v1

                Cant1D_v1.main_vectorized(k_subset, z_subset, self, velo_mod_1d_data)
            else:
                raise ValueError(f"Error: Submodel {name} not found in registry.")

    # TODO: I need to fix this function to automatically trigger the right main function of the submodel
    # some unused positional arguments are here for future use. will think of a better way to handle this
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
            from velocity_modelling.cvm.submodel import Cant1D_v1

            Cant1D_v1.main(z_ind, depth, self, submodel_data)
        elif submodel_name == "PaleogeneSubMod_v1":
            from velocity_modelling.cvm.submodel import PaleogeneSubMod_v1

            PaleogeneSubMod_v1.main(z_ind, self)
        elif submodel_name == "PlioceneSubMod_v1":
            from velocity_modelling.cvm.submodel import PlioceneSubMod_v1

            PlioceneSubMod_v1.main(z_ind, self)
        elif submodel_name == "MioceneSubMod_v1":
            from velocity_modelling.cvm.submodel import MioceneSubMod_v1

            MioceneSubMod_v1.main(z_ind, self)
        elif submodel_name == "BPVSubMod_v1":
            from velocity_modelling.cvm.submodel import BPVSubMod_v1

            BPVSubMod_v1.main(z_ind, self)
        elif submodel_name == "BPVSubMod_v2":
            from velocity_modelling.cvm.submodel import BPVSubMod_v2

            BPVSubMod_v2.main(z_ind, self)
        elif submodel_name == "BPVSubMod_v3":
            from velocity_modelling.cvm.submodel import BPVSubMod_v3

            BPVSubMod_v3.main(z_ind, depth, self, partial_basin_surface_depths)
        elif submodel_name == "BPVSubMod_v4":
            from velocity_modelling.cvm.submodel import BPVSubMod_v4

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

    def call_basin_submodel_vectorized(
        self,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        nz_tomography_data,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon,
        on_boundary,
        depths,
        ind_above,
        basin_num,
        z_indices,
    ):
        basin_data = partial_basin_surface_depths.basin
        submodel_name, submodel_data = basin_data.submodels[ind_above]

        if submodel_name == "NaNsubMod":
            self.nan_sub_mod_vectorized(z_indices)
        elif submodel_name in ["Cant1D_v1", "Cant1D_v2", "Cant1D_v2_Pliocene_Enforced"]:
            from velocity_modelling.cvm.submodel import Cant1D_v1 as Cant1D_v1

            Cant1D_v1.main_vectorized(z_indices, depths, self, submodel_data)
        elif submodel_name == "PaleogeneSubMod_v1":
            from velocity_modelling.cvm.submodel import (
                PaleogeneSubMod_v1 as PaleogeneSubMod_v1,
            )

            PaleogeneSubMod_v1.main_vectorized(z_indices, self)
        elif submodel_name == "PlioceneSubMod_v1":
            from velocity_modelling.cvm.submodel import (
                PlioceneSubMod_v1 as PlioceneSubMod_v1,
            )

            PlioceneSubMod_v1.main_vectorized(z_indices, self)
        elif submodel_name == "MioceneSubMod_v1":
            from velocity_modelling.cvm.submodel import (
                MioceneSubMod_v1 as MioceneSubMod_v1,
            )

            MioceneSubMod_v1.main_vectorized(z_indices, self)
        elif submodel_name == "BPVSubMod_v1":
            from velocity_modelling.cvm.submodel import BPVSubMod_v1 as BPVSubMod_v1

            BPVSubMod_v1.main_vectorized(z_indices, self)
        elif submodel_name == "BPVSubMod_v2":
            from velocity_modelling.cvm.submodel import BPVSubMod_v2 as BPVSubMod_v2

            BPVSubMod_v2.main_vectorized(z_indices, self)
        elif submodel_name == "BPVSubMod_v3":
            from velocity_modelling.cvm.submodel import BPVSubMod_v3 as BPVSubMod_v3

            BPVSubMod_v3.main_vectorized(
                z_indices, depths, self, partial_basin_surface_depths
            )
        elif submodel_name == "BPVSubMod_v4":
            from velocity_modelling.cvm.submodel import BPVSubMod_v4 as BPVSubMod_v4

            BPVSubMod_v4.main_vectorized(
                z_indices,
                depths,
                self,
                partial_basin_surface_depths,
                partial_global_surface_depths,
            )
        else:
            raise ValueError(f"Error: Submodel {submodel_name} not found in registry.")

    def assign_basin_qualities_vectorized(
        self,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        nz_tomography_data: TomographyData,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon: bool,
        on_boundary: bool,
        depths: np.ndarray,  # Array of depths
        basin_num: int,
        z_indices: np.ndarray,  # Array of z indices
    ):
        # Vectorized determination of surfaces above
        ind_above = (
            partial_basin_surface_depths.determine_basin_surface_above_vectorized(
                depths
            )
        )

        # Group depths by ind_above to handle different submodels
        for idx in np.unique(ind_above):
            mask = ind_above == idx
            depths_subset = depths[mask]
            z_indices_subset = z_indices[mask]

            self.inbasin[z_indices_subset] = basin_num
            self.call_basin_submodel_vectorized(
                partial_basin_surface_depths,
                partial_global_surface_depths,
                nz_tomography_data,
                mesh_vector,
                in_any_basin_lat_lon,
                on_boundary,
                depths_subset,
                idx,
                basin_num,
                z_indices_subset,
            )

    def nan_sub_mod_vectorized(self, z_indices):
        self.vp[z_indices] = np.nan
        self.vs[z_indices] = np.nan
        self.rho[z_indices] = np.nan
