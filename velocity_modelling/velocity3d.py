"""
3D Velocity Model Module.

This module provides classes and functions for calculating 3D velocity models
by combining global velocity models, basin models, and tomographic data.
It includes vectorized implementations for efficiently handling meshes with
multiple depth points at the same latitude/longitude.

The module's core functionality processes geology layers to determine seismic
velocities (P-wave, S-wave) and density at specified coordinates, handling
complex scenarios like basin boundaries and smooth transitions between models.
"""

import logging
from logging import Logger
from typing import Optional

import numpy as np

from velocity_modelling.basin_model import (
    BasinData,
    BasinMembership,
    InBasin,
    PartialBasinSurfaceDepths,
)
from velocity_modelling.constants import MAX_DIST_SMOOTH, TopoTypes, VelocityTypes
from velocity_modelling.geometry import (
    AdjacentPoints,
    MeshVector,
)
from velocity_modelling.global_model import (
    GlobalSurfaceRead,
    PartialGlobalSurfaceDepths,
    TomographyData,
    interpolate_global_surface,
)
from velocity_modelling.registry import CVMRegistry
from velocity_modelling.velocity1d import (
    VelocityModel1D,
)


class PartialGlobalQualities:
    """
    Container for storing velocity and density data for a partial global mesh.

    Parameters
    ----------
    n_lon : int
        Number of longitude points.
    n_depth : int
        Number of depth points.

    Attributes
    ----------
    vp : np.ndarray
        P-wave velocities (km/s).
    vs : np.ndarray
        S-wave velocities (km/s).
    rho : np.ndarray
        Densities (g/cm^3).
    inbasin : np.ndarray
        Basin membership indicators.
    """

    def __init__(self, n_lon: int, n_depth: int):
        """
        Initialize arrays for velocity and density data.
        """
        self.vp = np.zeros((n_lon, n_depth), dtype=np.float64)
        self.vs = np.zeros((n_lon, n_depth), dtype=np.float64)
        self.rho = np.zeros((n_lon, n_depth), dtype=np.float64)
        self.inbasin = np.zeros((n_lon, n_depth), dtype=np.int8)


class QualitiesVector:
    """
    Container for velocity and density data at a single lat-lon point with multiple depths.

    Parameters
    ----------
    n_depth : int
        Number of depth points.
    logger : Logger, optional
        Logger instance for logging messages.

    Attributes
    ----------
    vp : np.ndarray
        P-wave velocities (km/s).
    vs : np.ndarray
        S-wave velocities (km/s).
    rho : np.ndarray
        Densities (g/cm^3).
    inbasin : np.ndarray
        Basin membership indicators.
    logger : Logger
        Logger instance for logging messages.
    """

    def __init__(self, n_depth: int, logger: Optional[Logger] = None):
        """
        Initialize arrays for velocity and density data.


        """
        if logger is None:
            self.logger = Logger(name="velocity_model.qualities_vector")
        else:
            self.logger = logger

        self.vp = np.zeros(n_depth, dtype=np.float64)
        self.vs = np.zeros(n_depth, dtype=np.float64)
        self.rho = np.zeros(n_depth, dtype=np.float64)
        self.inbasin = np.zeros(n_depth, dtype=np.int8)

    def prescribe_velocities(
        self,
        global_params: dict,
        vm1d_data: VelocityModel1D,
        nz_tomography_data: TomographyData,
        global_surfaces: list[GlobalSurfaceRead],
        mesh_vector: MeshVector,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        partial_basin_surface_depths_list: list[PartialBasinSurfaceDepths],
        in_basin_list: list[InBasin],
        topo_type: TopoTypes,
        on_boundary: bool = False,
    ):
        """
        Calculate velocities and densities for all depths at a given lat-lon point.
        This is where the actual velocity/density assignment happens based on
        basin membership, global surfaces, and 1D models.

        Parameters
        ----------
        global_params : dict
            Parameters for the global velocity model.
        vm1d_data : VelocityModel1D
            1D velocity model data.
        nz_tomography_data : TomographyData
            Tomography data for velocity adjustments.
        global_surfaces : list[GlobalSurfaceRead]
            List of GlobalSurfaceRead objects.
        mesh_vector : MeshVector
            Lat-lon point with multiple depths.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Global surface depth data at this location.
        partial_basin_surface_depths_list : list[PartialBasinSurfaceDepths]
            Basin surface depth data at this location.
        in_basin_list : list[InBasin]
            Basin membership indicators.
        topo_type : TopoTypes
            Topography handling method ("TRUE", "BULLDOZED", "SQUASHED", "SQUASHED_TAPERED").
        on_boundary : bool, optional
            True if this point is on a basin (including smoothing) boundary. Default is False.
        """

        partial_global_surface_depths.interpolate_global_surface_depths(
            global_surfaces, mesh_vector
        )

        in_any_basin_lat_lon = any(
            in_basin.in_basin_lat_lon for in_basin in in_basin_list
        )

        if topo_type == TopoTypes.SQUASHED:
            depth_change = -mesh_vector.z
            shifted_mesh_vector = mesh_vector.copy()
            shifted_mesh_vector.z = (
                partial_global_surface_depths.depths[1] - depth_change
            )

        elif topo_type == TopoTypes.SQUASHED_TAPERED:
            taper_dist = 1.0
            depth_change = -mesh_vector.z

            # Calculate denominator - check if it's a scalar or array
            denominator = partial_global_surface_depths.depths[1] * taper_dist

            # Create mask for safe division (avoiding zeros)
            valid_mask = depth_change != 0

            # Initialize taper_val array
            taper_val = np.ones_like(depth_change)

            # Check if denominator is a scalar or array
            if np.isscalar(denominator):
                if (
                    np.isfinite(denominator) and denominator > 0
                ):  # Scalar, finite, and positive
                    taper_val[valid_mask] = 1.0 - (
                        depth_change[valid_mask] / denominator
                    )
                # If denominator is nan, inf, or <= 0, keep taper_val as 1.0 (no action needed)
            else:
                # Handle array denominator
                array_valid_mask = valid_mask & (denominator > 0)
                taper_val[array_valid_mask] = 1.0 - (
                    depth_change[array_valid_mask] / denominator[array_valid_mask]
                )
            # Ensure taper_val is within bounds
            taper_val = np.clip(taper_val, 0.0, None)
            shifted_mesh_vector = mesh_vector.copy()
            shifted_mesh_vector.z = (
                partial_global_surface_depths.depths[1] * taper_val - depth_change
            )

        elif topo_type in [TopoTypes.BULLDOZED, TopoTypes.TRUE]:
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
            if topo_type in [TopoTypes.BULLDOZED, TopoTypes.TRUE]
            else shifted_mesh_vector.z
        )
        k_indices = np.arange(mesh_vector.nz)

        # Precompute basin membership for all depths (robust to no basins)
        n_basins = len(in_basin_list)
        if n_basins == 0:
            # No basins - all depths are outside basins
            basin_per_k = np.full(mesh_vector.nz, -1, dtype=int)
            in_basin_mask = np.zeros(mesh_vector.nz, dtype=bool)
            out_basin_mask = np.ones(mesh_vector.nz, dtype=bool)
        else:
            basin_mask = np.asarray(
                [in_basin.in_basin_depth for in_basin in in_basin_list], dtype=bool
            )  # Shape: (n_basins, nz)
            # Find the index of the last basin containing each depth point.
            # The last basin in the list has the highest priority.
            # np.argmax finds the first 'True' in the reversed array, which corresponds to the last 'True' in the original.
            basin_indices = np.argmax(basin_mask[::-1], axis=0)
            basin_per_k = np.where(
                np.any(basin_mask, axis=0), n_basins - 1 - basin_indices, -1
            )
            # Identify depths inside basins and outside basins
            in_basin_mask = basin_per_k >= 0
            out_basin_mask = ~in_basin_mask

        # Process depths in basins
        if np.any(in_basin_mask):
            k_in_basin = k_indices[in_basin_mask]
            z_in_basin = z_values[in_basin_mask]
            basins_in_basin = basin_per_k[in_basin_mask]

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

        # Process depths not in any basin
        if np.any(out_basin_mask):
            k_out_basin = k_indices[out_basin_mask]
            z_out_basin = z_values[out_basin_mask]

            # Vectorized call to find submodel indices
            submodel_indices = (
                partial_global_surface_depths.find_global_submodel_ind_vectorized(
                    z_out_basin
                )
            )
            submodel_names = np.array(global_params["submodels"])[submodel_indices]

            # Vectorized call to call_global_submodel
            self.call_global_submodel_vectorized(
                submodel_names,
                z_out_basin,
                k_out_basin,
                global_params,
                partial_global_surface_depths,
                vm1d_data,
                nz_tomography_data,
                mesh_vector,
                in_any_basin_lat_lon,
                on_boundary,
            )

        # Apply NaN masking for depths above the surface
        mask_above_surface = z_values > partial_global_surface_depths.depths[1]
        self.rho[mask_above_surface] = np.nan
        self.vp[mask_above_surface] = np.nan
        self.vs[mask_above_surface] = np.nan

        # Apply NaN masking for bulldozed topography
        if topo_type == TopoTypes.BULLDOZED:
            mask_above_zero = mesh_vector.z > 0
            self.rho[mask_above_zero] = np.nan
            self.vp[mask_above_zero] = np.nan
            self.vs[mask_above_zero] = np.nan

    def assign_qualities(
        self,
        cvm_registry: CVMRegistry,
        vm1d_data: VelocityModel1D,
        nz_tomography_data: TomographyData,
        global_surfaces: list[GlobalSurfaceRead],
        basin_data_list: list[BasinData],
        mesh_vector: MeshVector,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        partial_basin_surface_depths_list: list[PartialBasinSurfaceDepths],
        in_basin_list: list[InBasin],
        basin_membership: BasinMembership | None,
        topo_type: TopoTypes,
    ):
        """
        Determine if lat-lon point lies within the smoothing zone and prescribe velocities accordingly.

        This method handles smoothing between velocity models at boundaries. Smoothing is applied
        only when the point is within the smoothing zone and mesh_basin_membership is provided. If
        mesh_basin_membership is None (e.g., isolated stations, thresholds, 1D profiles), smoothing is skipped.

        Parameters
        ----------
        cvm_registry : CVMRegistry
            Registry containing model parameters.
        vm1d_data : VelocityModel1D
            1D velocity model data.
        nz_tomography_data : TomographyData
            Tomography data providing the background velocity values, including optional smoothing boundary.
        global_surfaces : list[GlobalSurfaceRead]
            List of GlobalSurfaceRead objects.
        basin_data_list : list[BasinData]
            List of basin data objects.
        mesh_vector : MeshVector
            Mesh vector containing lat, lon, and depth values.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Global surface depth data at this location.
        partial_basin_surface_depths_list : list[PartialBasinSurfaceDepths]
            Basin surface depth data at this location.
        in_basin_list : list[InBasin]
            Basin membership flags (pre-populated for isolated station workflows).
        basin_membership : BasinMembership or None
            Basin membership handler with optional smoothing boundary preprocessing.
            Can be None to skip smoothing (though this should be rare - usually you want
            to provide it even for isolated points to handle offshore smoothing correctly).
        topo_type : TopoTypes
            Topography handling method.
        """

        smooth_bound = nz_tomography_data.smooth_boundary

        closest_ind = None

        if smooth_bound.n_points == 0:
            distance = (
                1e6  # if there are no points in the smoothing boundary, then skip
            )

        else:
            closest_ind, distance = (
                smooth_bound.determine_if_lat_lon_within_smoothing_region(mesh_vector)
            )

        # calculate vs30 (used as a proxy to determine if point is on- or off-shore, only if using tomography)
        if nz_tomography_data.gtl:
            nz_tomography_data.calculate_vs30_from_tomo_vs30_surface(
                mesh_vector
            )  # mesh_vector.vs30 updated
            nz_tomography_data.calculate_distance_from_shoreline(
                mesh_vector
            )  # mesh_vector.distance_from_shoreline updated

        in_any_basin = any(in_basin.in_basin_lat_lon for in_basin in in_basin_list)

        in_smoothing_zone = (
            distance <= MAX_DIST_SMOOTH
            and not in_any_basin
            and nz_tomography_data.gtl  # Ely et al. 2010: Geotechnical Layer. If True, then apply the offshore basin depth
            and mesh_vector.vs30 < 100  # offshore if vs30 < 100 m/s
        )

        # Apply smoothing only if basin_membership is provided; otherwise skip smoothing gracefully.
        if in_smoothing_zone and basin_membership is not None:
            # point lies within smoothing zone and is not in any basin (i.e., outside any boundaries)
            qualities_vector_a = QualitiesVector(mesh_vector.nz)
            qualities_vector_b = QualitiesVector(mesh_vector.nz)
            in_basin_b_list = [
                InBasin(basin_data, mesh_vector.nz) for basin_data in basin_data_list
            ]
            partial_global_surface_depths_b = PartialGlobalSurfaceDepths(
                len(global_surfaces)
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
            mesh_vector.lat = smooth_bound.lats[closest_ind]
            mesh_vector.lon = smooth_bound.lons[closest_ind]

            # determine if the point is in any basin using precomputed boundary membership
            if basin_membership.smoothing_boundary_basin_indices is None:
                self.logger.log(
                    logging.ERROR,
                    "smoothing_boundary_basin_indices is None, falling back to manual calculation",
                )
                smooth_indices = basin_membership.check_one_station(
                    mesh_vector.lat, mesh_vector.lon
                )
            else:
                smooth_indices = basin_membership.smoothing_boundary_basin_indices[
                    closest_ind
                ]

            for i, in_basin in enumerate(in_basin_b_list):
                in_basin.in_basin_lat_lon = i in smooth_indices

            # velocity vector just inside the boundary
            on_boundary = True
            qualities_vector_b.prescribe_velocities(
                cvm_registry.global_params,
                vm1d_data,
                nz_tomography_data,
                global_surfaces,
                mesh_vector,
                partial_global_surface_depths_b,
                partial_basin_surface_depths_list_b,
                in_basin_b_list,
                topo_type,
                on_boundary,
            )

            # overwrite the lat-lon with the original lat-lon point
            mesh_vector.lat = original_lat
            mesh_vector.lon = original_lon

            # velocity vector at the point in question
            on_boundary = False
            qualities_vector_a.prescribe_velocities(
                cvm_registry.global_params,
                vm1d_data,
                nz_tomography_data,
                global_surfaces,
                mesh_vector,
                partial_global_surface_depths,
                partial_basin_surface_depths_list,
                in_basin_list,
                topo_type,
                on_boundary,
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
            # No smoothing needed or no basin_membership provided ; prescribe velocities directly
            on_boundary = False
            self.prescribe_velocities(
                cvm_registry.global_params,
                vm1d_data,
                nz_tomography_data,
                global_surfaces,
                mesh_vector,
                partial_global_surface_depths,
                partial_basin_surface_depths_list,
                in_basin_list,
                topo_type,
                on_boundary,
            )

    # TODO: I need to fix this function to automatically trigger the right main function of the submodel.
    # Keeping this way for now, but will think of a better way to handle this
    def call_global_submodel_vectorized(
        self,
        submodel_names: np.ndarray,
        depths: np.ndarray,
        z_indices: np.ndarray,
        global_params: dict,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        vm1d_data: VelocityModel1D,
        nz_tomography_data: TomographyData,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon: bool,
        on_boundary: bool,
    ):
        """
        Call the appropriate global sub-velocity models for multiple depths.

        This vectorized implementation processes multiple depths efficiently.

        Parameters
        ----------
        submodel_names : np.ndarray
            Names of global submodels for each depth.
        depths : np.ndarray
            Depths (metres) to calculate properties for.
        z_indices : np.ndarray
            Indices of depths in the mesh.
        global_params : dict
            Parameters for the global velocity model.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Global surface depth data.
        vm1d_data : VelocityModel1D
            1D velocity model data.
        nz_tomography_data : TomographyData
            Tomography data.
        mesh_vector : MeshVector
            Lat-lon point with multiple depths.
        in_any_basin_lat_lon : bool
            Whether this lat-lon point is in any basin.
        on_boundary : bool
            Whether this point is on a model boundary.
        """
        # Sort by z_indices to preserve depth order
        order = np.argsort(z_indices)
        z_indices = z_indices[order]
        depths = depths[order]
        submodel_names = submodel_names[order]

        for name in np.unique(submodel_names):
            mask = submodel_names == name
            depth_subset = depths[mask]
            index_subset = z_indices[mask]

            if name == "nan_submod":
                from velocity_modelling.submodel import nan_submod

                nan_submod.main_vectorized(index_subset, self)
            elif name == "ep_tomography_submod_v2010":
                from velocity_modelling.submodel import ep_tomography_submod_v2010

                # Precompute interpolated values for all surfaces and vtype at this (lat, lon)
                global_surf_read = nz_tomography_data.surfaces[0]["vp"]
                adjacent_points = AdjacentPoints.find_global_adjacent_points(
                    global_surf_read.lats,
                    global_surf_read.lons,
                    mesh_vector.lat,
                    mesh_vector.lon,
                )

                # Precompute interpolated values for all surfaces and vtype
                num_surfaces = len(nz_tomography_data.surfaces)
                interpolated_global_surface_values = {
                    vtype.name: np.zeros(num_surfaces) for vtype in VelocityTypes
                }
                for idx in range(num_surfaces):
                    for vtype in VelocityTypes:
                        surface = nz_tomography_data.surfaces[idx][vtype.name]
                        val = interpolate_global_surface(
                            surface, mesh_vector.lat, mesh_vector.lon, adjacent_points
                        )
                        interpolated_global_surface_values[vtype.name][idx] = val

                ep_tomography_submod_v2010.main_vectorized(
                    index_subset,
                    depth_subset,
                    self,
                    mesh_vector,
                    nz_tomography_data,
                    partial_global_surface_depths,
                    in_any_basin_lat_lon,
                    on_boundary,
                    interpolated_global_surface_values,
                )
            elif name == "canterbury1d_v2":
                from velocity_modelling.submodel import canterbury1d_submod

                canterbury1d_submod.main_vectorized(
                    index_subset, depth_subset, self, vm1d_data
                )
            else:
                raise ValueError(f"Error: Submodel {name} not found in registry.")

    # TODO: I need to fix this function to automatically trigger the right main function of the submodel
    # some unused positional arguments are here for future use. will think of a better way to handle this
    def call_basin_submodel_vectorized(
        self,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        nz_tomography_data: TomographyData,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon: bool,
        on_boundary: bool,
        depths: np.ndarray,
        ind_above: int,
        basin_num: int,
        z_indices: np.ndarray,
    ):
        """
        Call the appropriate basin sub-velocity models for multiple depths.

        Parameters
        ----------
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Basin surface depths at this location.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Global surface depths at this location.
        nz_tomography_data : TomographyData
            Tomography data.
        mesh_vector : MeshVector
            Lat-lon point with multiple depths.
        in_any_basin_lat_lon : bool
            Whether this lat-lon point is in any basin.
        on_boundary : bool
            Whether this point is on a model boundary.
        depths : np.ndarray
            Depths (metres) to calculate properties for.
        ind_above : int
            Index of surface above these depths.
        basin_num : int
            Basin identifier.
        z_indices : np.ndarray
            Indices of depths in the mesh.
        """

        basin_data = partial_basin_surface_depths.basin
        submodel_name, submodel_data = basin_data.submodels[ind_above]

        if submodel_name == "nan_submod":
            from velocity_modelling.submodel import nan_submod

            nan_submod.main_vectorized(z_indices, self)

        elif submodel_name in [
            "canterbury1d_v1",
            "canterbury1d_v2",
            "canterbury1d_v2_pliocene_enforced",
            "canterbury1d_v3_pliocene_enforced",
            "nelson_v1",
            "palmerstonnorth_v1",
        ]:
            from velocity_modelling.submodel import canterbury1d_submod

            canterbury1d_submod.main_vectorized(z_indices, depths, self, submodel_data)
        elif submodel_name == "paleogene_submod_v1":
            from velocity_modelling.submodel import paleogene_submod_v1

            paleogene_submod_v1.main_vectorized(z_indices, self)
        elif submodel_name == "pliocene_submod_v1":
            from velocity_modelling.submodel import pliocene_submod_v1

            pliocene_submod_v1.main_vectorized(z_indices, self)
        elif submodel_name == "miocene_submod_v1":
            from velocity_modelling.submodel import miocene_submod_v1

            miocene_submod_v1.main_vectorized(z_indices, self)
        elif submodel_name == "paleogene_submod_v2":
            from velocity_modelling.submodel import paleogene_submod_v2

            paleogene_submod_v2.main_vectorized(z_indices, self)
        elif submodel_name == "pliocene_submod_v2":
            from velocity_modelling.submodel import pliocene_submod_v2

            pliocene_submod_v2.main_vectorized(z_indices, self)
        elif submodel_name == "miocene_submod_v2":
            from velocity_modelling.submodel import miocene_submod_v2

            miocene_submod_v2.main_vectorized(z_indices, self)
        elif submodel_name == "pn_pliocene_submod_v1":
            from velocity_modelling.submodel import pn_pliocene_submod_v1

            pn_pliocene_submod_v1.main_vectorized(z_indices, self)
        elif submodel_name == "bpv_submod_v1":
            from velocity_modelling.submodel import bpv_submod_v1

            bpv_submod_v1.main_vectorized(z_indices, self)
        elif submodel_name == "bpv_submod_v2":
            from velocity_modelling.submodel import bpv_submod_v2

            bpv_submod_v2.main_vectorized(z_indices, self)
        elif submodel_name == "bpv_submod_v3":
            from velocity_modelling.submodel import bpv_submod_v3

            bpv_submod_v3.main_vectorized(
                z_indices, depths, self, partial_basin_surface_depths
            )
        elif submodel_name == "bpv_submod_v4":
            from velocity_modelling.submodel import bpv_submod_v4

            bpv_submod_v4.main_vectorized(
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
        """
        Assign velocities and densities for points within a basin.

        Parameters
        ----------
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Basin surface depths at this location.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Global surface depths at this location.
        nz_tomography_data : TomographyData
            Tomography data.
        mesh_vector : MeshVector
            Lat-lon point with multiple depths.
        in_any_basin_lat_lon : bool
            Whether this lat-lon point is in any basin.
        on_boundary : bool
            Whether this point is on a model boundary.
        depths : np.ndarray
            Depths (metres) to calculate properties for.
        basin_num : int
            Basin identifier.
        z_indices : np.ndarray
            Indices of depths in the mesh.
        """

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

    def nan_sub_mod_vectorized(self, z_indices: np.ndarray):
        """
        Assign NaN values to velocities and density at specified depth indices.

        Parameters
        ----------
        z_indices : np.ndarray
            Depth indices to assign NaN values to.
        """
        self.vp[z_indices] = np.nan
        self.vs[z_indices] = np.nan
        self.rho[z_indices] = np.nan
