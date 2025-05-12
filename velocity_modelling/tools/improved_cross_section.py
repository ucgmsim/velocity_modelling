#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from pathlib import Path

from velocity_modelling.constants import EARTH_RADIUS_MEAN
from numba import njit

#!/usr/bin/env python3
import numpy as np

def point_in_rotated_rectangle(point, origin, width, height, angle_degrees):
    """
    Check if a point is inside a rotated rectangle.
    
    Args:
        point: Tuple (x, y) representing the point to check
        origin: Tuple (x, y) representing the center of the rectangle
        width: Width of the rectangle
        height: Height of the rectangle
        angle_degrees: Rotation angle in degrees (clockwise)
    
    Returns:
        bool: True if the point is inside the rotated rectangle
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Translate point relative to rectangle center
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    
    # Rotate point in the opposite direction of rectangle rotation
    x_rot = dx * cos_angle + dy * sin_angle
    y_rot = -dx * sin_angle + dy * cos_angle
    
    # Check if the rotated point is within the rectangle
    half_width = width / 2
    half_height = height / 2
    
    return (x_rot >= -half_width and x_rot <= half_width and
            y_rot >= -half_height and y_rot <= half_height)

def get_domain_bounds(origin_lat, origin_lon, origin_rot, extent_x, extent_y):
    """
    Calculate the corners of a rotated domain.
    
    Args:
        origin_lat: Latitude of domain center
        origin_lon: Longitude of domain center
        origin_rot: Rotation angle in degrees (clockwise)
        extent_x: Width of domain in km
        extent_y: Height of domain in km
    
    Returns:
        tuple: (corners, min_lat, max_lat, min_lon, max_lon)
               corners is a list of (lon, lat) tuples for the domain corners
               min_lat, max_lat, min_lon, max_lon are the bounding box coordinates
    """
    # Define corners relative to center (in km)
    half_x = extent_x / 2
    half_y = extent_y / 2
    corners_local = [
        (-half_x, -half_y),  # Bottom-left
        (half_x, -half_y),   # Bottom-right
        (half_x, half_y),    # Top-right
        (-half_x, half_y)    # Top-left
    ]
    
    # Convert angle to radians
    angle_rad = np.radians(origin_rot)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Rotate corners
    corners_rotated = []
    for x, y in corners_local:
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle
        corners_rotated.append((x_rot, y_rot))
    
    # Convert from km to degrees (approximate)
    # 1 degree of latitude ≈ 111 km
    # 1 degree of longitude ≈ 111 km * cos(latitude) at the equator
    lat_scale = 111.0
    lon_scale = 111.0 * np.cos(np.radians(origin_lat))
    
    # Apply conversion and translate to absolute coordinates
    corners_geo = []
    for x_rot, y_rot in corners_rotated:
        lon = origin_lon + x_rot / lon_scale
        lat = origin_lat + y_rot / lat_scale
        corners_geo.append((lon, lat))
    
    # Calculate bounding box
    lons = [lon for lon, lat in corners_geo]
    lats = [lat for lon, lat in corners_geo]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    return corners_geo, min_lat, max_lat, min_lon, max_lon

def extract_domain_info(h5file):
    """
    Extract domain information from HDF5 file.
    
    Args:
        h5file: Path to HDF5 file
        
    Returns:
        dict: Dictionary containing domain parameters
    """
    import h5py
    
    domain_info = {}
    try:
        with h5py.File(h5file, 'r') as f:
            # First check for domain info in the config group (NZ Velocity Model format)
            if '/config' in f:
                print("Found config group, checking for domain parameters...")
                config_group = f['/config']
                
                # Map config attributes to standard domain parameter names
                mapping = {
                    'origin_lat': 'ORIGIN_LAT',
                    'origin_lon': 'ORIGIN_LON',
                    'origin_rot': 'ORIGIN_ROT',
                    'extent_x': 'EXTENT_X',
                    'extent_y': 'EXTENT_Y',
                    'extent_zmin': 'EXTENT_ZMIN',
                    'extent_zmax': 'EXTENT_ZMAX',
                    'h_depth': 'EXTENT_Z_SPACING',
                    'h_lat_lon': 'EXTENT_LATLON_SPACING'
                }
                
                # Copy attributes from config group to domain_info
                for config_key, domain_key in mapping.items():
                    if config_key in config_group.attrs:
                        domain_info[domain_key] = config_group.attrs[config_key]
                        print(f"  Found {domain_key}: {domain_info[domain_key]}")
                
                # If we got all essential parameters, we can return
                if all(key in domain_info for key in ['ORIGIN_LAT', 'ORIGIN_LON', 'ORIGIN_ROT', 'EXTENT_X', 'EXTENT_Y']):
                    print("Successfully extracted domain parameters from config group")
                    return domain_info
            
            # If we didn't find all parameters in the config group, or there is no config group,
            # try to parse the config_string attribute which might contain the parameters
            if '/config' in f and 'config_string' in f['/config'].attrs:
                print("Checking config_string attribute...")
                config_string = f['/config'].attrs['config_string']
                for line in config_string.split('\n'):
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key in ['ORIGIN_LAT', 'ORIGIN_LON', 'ORIGIN_ROT', 'EXTENT_X', 'EXTENT_Y',
                                  'EXTENT_ZMIN', 'EXTENT_ZMAX', 'H_DEPTH', 'H_LAT_LON']:
                            try:
                                # Convert string to float
                                domain_info[key] = float(value)
                                print(f"  Found {key}: {domain_info[key]}")
                            except ValueError:
                                print(f"  Could not convert {key}={value} to float")
                
                # Map H_DEPTH and H_LAT_LON to their standard names if needed
                if 'H_DEPTH' in domain_info and 'EXTENT_Z_SPACING' not in domain_info:
                    domain_info['EXTENT_Z_SPACING'] = domain_info['H_DEPTH']
                if 'H_LAT_LON' in domain_info and 'EXTENT_LATLON_SPACING' not in domain_info:
                    domain_info['EXTENT_LATLON_SPACING'] = domain_info['H_LAT_LON']
                
                # If we now have all essential parameters, we can return
                if all(key in domain_info for key in ['ORIGIN_LAT', 'ORIGIN_LON', 'ORIGIN_ROT', 'EXTENT_X', 'EXTENT_Y']):
                    print("Successfully extracted domain parameters from config_string")
                    return domain_info
            
            # If still not found, check at root level
            print("Checking root attributes...")
            for key in ['ORIGIN_LAT', 'ORIGIN_LON', 'ORIGIN_ROT', 
                      'EXTENT_X', 'EXTENT_Y', 'EXTENT_ZMIN', 
                      'EXTENT_ZMAX', 'EXTENT_Z_SPACING', 'EXTENT_LATLON_SPACING']:
                if key in f.attrs and key not in domain_info:
                    domain_info[key] = f.attrs[key]
                    print(f"  Found {key}: {domain_info[key]}")
            
            # Check other locations and variations
            if not all(key in domain_info for key in ['ORIGIN_LAT', 'ORIGIN_LON', 'ORIGIN_ROT']):
                print("Checking for variations of domain parameters...")
                for prefix in ['', 'domain_', 'DOMAIN_', 'model_']:
                    for key_base, key_std in [
                        ('ORIGIN_LAT', 'ORIGIN_LAT'), 
                        ('ORIGIN_LON', 'ORIGIN_LON'),
                        ('ORIGIN_ROT', 'ORIGIN_ROT'),
                        ('EXTENT_X', 'EXTENT_X'),
                        ('EXTENT_Y', 'EXTENT_Y'),
                        ('orig_lat', 'ORIGIN_LAT'),
                        ('orig_lon', 'ORIGIN_LON'),
                        ('rot', 'ORIGIN_ROT'),
                        ('ext_x', 'EXTENT_X'),
                        ('ext_y', 'EXTENT_Y'),
                        ('origin_latitude', 'ORIGIN_LAT'),
                        ('origin_longitude', 'ORIGIN_LON'),
                        ('rotation', 'ORIGIN_ROT'),
                        ('width', 'EXTENT_X'),
                        ('height', 'EXTENT_Y')
                    ]:
                        attr_key = f"{prefix}{key_base}"
                        if attr_key in f.attrs and key_std not in domain_info:
                            domain_info[key_std] = f.attrs[attr_key]
                            print(f"  Found {attr_key} -> {key_std}: {domain_info[key_std]}")
                            
                # Check domain group if needed
                if not all(key in domain_info for key in ['ORIGIN_LAT', 'ORIGIN_LON']):
                    if '/domain' in f:
                        domain_group = f['/domain']
                        for key in domain_group.attrs:
                            # Try to map to standard key names
                            std_key = key.upper()
                            if std_key in ['ORIGIN_LAT', 'ORIGIN_LON', 'ORIGIN_ROT', 
                                         'EXTENT_X', 'EXTENT_Y'] and std_key not in domain_info:
                                domain_info[std_key] = domain_group.attrs[key]
                                print(f"  Found domain/{key} -> {std_key}: {domain_info[std_key]}")
            
            # If we have the essential parameters, print them
            if all(key in domain_info for key in ['ORIGIN_LAT', 'ORIGIN_LON', 'ORIGIN_ROT', 'EXTENT_X', 'EXTENT_Y']):
                print("Final domain parameters:")
                for key, value in domain_info.items():
                    print(f"  {key}: {value}")
            else:
                print("Warning: Missing some essential domain parameters")
                print("Available parameters:", list(domain_info.keys()))
                
    except Exception as e:
        print(f"Error extracting domain info: {e}")
        import traceback
        traceback.print_exc()
    
    return domain_info

# Add a function to check if a point is within the rotated domain
def point_in_domain(lat, lon, domain_info):
    """
    Check if a point is inside the rotated domain.
    
    Args:
        lat: Latitude to check
        lon: Longitude to check
        domain_info: Dictionary with domain parameters
        
    Returns:
        bool: True if the point is inside the domain
    """
    # Get domain parameters
    origin_lat = domain_info.get('ORIGIN_LAT', 0)
    origin_lon = domain_info.get('ORIGIN_LON', 0)
    origin_rot = domain_info.get('ORIGIN_ROT', 0)
    extent_x = domain_info.get('EXTENT_X', 100)
    extent_y = domain_info.get('EXTENT_Y', 100)
    
    # Convert to km from center (approximate)
    lat_scale = 111.0  # km per degree of latitude
    lon_scale = 111.0 * np.cos(np.radians(origin_lat))  # km per degree of longitude
    
    dx = (lon - origin_lon) * lon_scale  # km east from origin
    dy = (lat - origin_lat) * lat_scale  # km north from origin
    
    # Check if point is in the rotated rectangle
    return point_in_rotated_rectangle((dx, dy), (0, 0), extent_x, extent_y, origin_rot)

# Modify the clip_to_domain function to respect rotation
def clip_to_domain(lat, lon, domain_info):
    """
    Clip coordinates to the closest point in the domain.
    
    Args:
        lat: Latitude to clip
        lon: Longitude to clip
        domain_info: Dictionary with domain parameters
        
    Returns:
        tuple: (clipped_lat, clipped_lon)
    """
    # Handle the case when lat or lon is None
    if lat is None or lon is None:
        print("Warning: Cannot clip None coordinates to domain")
        return lat, lon
    
    # Get domain parameters
    origin_lat = domain_info.get('ORIGIN_LAT', 0)
    origin_lon = domain_info.get('ORIGIN_LON', 0)
    origin_rot = domain_info.get('ORIGIN_ROT', 0)
    extent_x = domain_info.get('EXTENT_X', 100)
    extent_y = domain_info.get('EXTENT_Y', 100)
    
    # Calculate corners of the domain
    corners, min_lat, max_lat, min_lon, max_lon = get_domain_bounds(
        origin_lat, origin_lon, origin_rot, extent_x, extent_y
    )
    
    # If point is already in domain, return as is
    if point_in_domain(lat, lon, domain_info):
        return lat, lon
    
    # Convert point to km from center
    lat_scale = 111.0
    lon_scale = 111.0 * np.cos(np.radians(origin_lat))
    
    dx = (lon - origin_lon) * lon_scale
    dy = (lat - origin_lat) * lat_scale
    
    # Convert to rotated coordinates
    angle_rad = np.radians(origin_rot)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    x_rot = dx * cos_angle + dy * sin_angle
    y_rot = -dx * sin_angle + dy * cos_angle
    
    # Clip in rotated coordinates
    half_x = extent_x / 2
    half_y = extent_y / 2
    
    x_clipped = np.clip(x_rot, -half_x, half_x)
    y_clipped = np.clip(y_rot, -half_y, half_y)
    
    # Convert back to geographic coordinates
    dx_clipped = x_clipped * cos_angle - y_clipped * sin_angle
    dy_clipped = x_clipped * sin_angle + y_clipped * cos_angle
    
    clipped_lon = origin_lon + dx_clipped / lon_scale
    clipped_lat = origin_lat + dy_clipped / lat_scale
    
    return clipped_lat, clipped_lon

def get_lat_lon_ranges(h5file):
    """Extract and return the ranges of latitude and longitude from the HDF5 file."""
    import h5py
    try:
        with h5py.File(h5file, 'r') as f:
            lat = f['/mesh/lat'][()]
            lon = f['/mesh/lon'][()]
        return min(lat), max(lat), min(lon), max(lon)
    except Exception as e:
        print(f"Error reading coordinate ranges: {e}")
        # Return some default values that will cause the error to be noticed
        return -90, 90, -180, 180

def create_domain_polygon(h5file):
    """
    Create a polygon representing the domain boundary.
    
    Args:
        h5file: Path to HDF5 file
        
    Returns:
        tuple: (corners, min_lat, max_lat, min_lon, max_lon, origin_lat, origin_lon, rotation)
               corners is a list of (lon, lat) tuples for the domain corners
    """
    domain_info = extract_domain_info(h5file)
    
    # Get domain parameters (with defaults if missing)
    origin_lat = domain_info.get('ORIGIN_LAT', 0)
    origin_lon = domain_info.get('ORIGIN_LON', 0)
    origin_rot = domain_info.get('ORIGIN_ROT', 0)
    extent_x = domain_info.get('EXTENT_X', 100)  # default 100km 
    extent_y = domain_info.get('EXTENT_Y', 100)  # default 100km
    
    # Calculate domain corners
    corners, min_lat, max_lat, min_lon, max_lon = get_domain_bounds(
        origin_lat, origin_lon, origin_rot, extent_x, extent_y
    )
    
    return corners, min_lat, max_lat, min_lon, max_lon, origin_lat, origin_lon, origin_rot


@njit
def compute_rotation_matrix(
    origin_lat: float, origin_lon: float, origin_rot: float
) -> np.ndarray:
    """
    Compute the rotation matrix to transform geographic coordinates to a rotated system.
    
    Parameters:
    - origin_lat, origin_lon: Origin coordinates (degrees).
    - origin_rot: Rotation angle (degrees, counterclockwise).
    
    Returns:
    - rot_matrix: 3x3 rotation matrix.
    """
    RPERD = np.pi / 180.0
    lat_rad = origin_lat * RPERD
    lon_rad = origin_lon * RPERD
    rot_rad = origin_rot * RPERD

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    sin_rot = np.sin(rot_rad)
    cos_rot = np.cos(rot_rad)

    rot_matrix = np.zeros((3, 3))
    rot_matrix[0, 0] = cos_rot * cos_lon - sin_rot * sin_lon * cos_lat
    rot_matrix[0, 1] = -sin_rot * cos_lon - cos_rot * sin_lon * cos_lat
    rot_matrix[0, 2] = sin_lon * sin_lat
    rot_matrix[1, 0] = cos_rot * sin_lon + sin_rot * cos_lon * cos_lat
    rot_matrix[1, 1] = -sin_rot * sin_lon + cos_rot * cos_lon * cos_lat
    rot_matrix[1, 2] = -cos_lon * sin_lat
    rot_matrix[2, 0] = sin_rot * sin_lat
    rot_matrix[2, 1] = cos_rot * sin_lat
    rot_matrix[2, 2] = cos_lat

    return rot_matrix

@njit
def geographic_to_rotated_coords(
    lat: float, lon: float, origin_lat: float, origin_lon: float, origin_rot: float
) -> tuple[float, float]:
    """
    Transform geographic coordinates to rotated coordinates (x, y) in km using spherical geometry.
    """
    RPERD = np.pi / 180.0
    lat_rad = lat * RPERD
    lon_rad = lon * RPERD

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    cart = np.array([
        cos_lat * cos_lon,
        cos_lat * sin_lon,
        sin_lat
    ])

    rot_matrix = compute_rotation_matrix(origin_lat, origin_lon, origin_rot)
    cart_rot = np.dot(rot_matrix, cart)
    theta_rot = np.arccos(cart_rot[2])
    phi_rot = np.arctan2(cart_rot[1], cart_rot[0])
    x = EARTH_RADIUS_MEAN * theta_rot * np.cos(phi_rot)
    y = EARTH_RADIUS_MEAN * theta_rot * np.sin(phi_rot)

    return x, y

def improved_cross_section(h5file, lat1, lon1, lat2=None, lon2=None, n_points=200, 
                         property_name='vs', output_png=None, vmin=None, vmax=None, cmap='plasma', max_depth=None):
    print(f"Creating improved cross-section from {h5file}")
    
    # 1) Load data and mesh
    try:
        with h5py.File(h5file, 'r') as f:
            print("HDF5 file opened successfully")
            if '/mesh' not in f:
                raise KeyError("'/mesh' group not found in HDF5 file")
            if '/properties' not in f:
                raise KeyError("'/properties' group not found in HDF5 file")
            if f'/properties/{property_name}' not in f:
                raise KeyError(f"'/properties/{property_name}' not found in HDF5 file")
            
            lat = f['/mesh/lat'][()]
            lon = f['/mesh/lon'][()]
            z = f['/mesh/z'][()]
            data = f[f'/properties/{property_name}'][()]
            
            # Load rotation parameters, extent_zmax, and h_depth (extent_z_spacing)
            config_group = f.get('/config')
            if config_group is None:
                print("Warning: '/config' group not found")
                origin_lat, origin_lon, origin_rot, extent_zmax, extent_z_spacing = None, None, None, None, None
            else:
                print("Config group found. Available attributes:", list(config_group.attrs.keys()))
                origin_lat = config_group.attrs.get('origin_lat')
                origin_lon = config_group.attrs.get('origin_lon')
                origin_rot = config_group.attrs.get('origin_rot')
                extent_zmax = config_group.attrs.get('extent_zmax')
                extent_z_spacing = config_group.attrs.get('h_depth')  # Updated to h_depth
            
            print(f"Data shape: {data.shape}")
            print(f"Dimensions: z={len(z)}, lat={len(lat)}, lon={len(lon)}")
            print(f"Mesh lat range: {lat.min():.4f} to {lat.max():.4f}")
            print(f"Mesh lon range: {lon.min():.4f} to {lon.max():.4f}")
            print(f"Raw z range: {z.min():.4f} to {z.max():.4f}")
            print(f"Data min: {np.nanmin(data):.2f}, max: {np.nanmax(data):.2f}")
            print(f"Data contains NaN: {np.any(np.isnan(data))}")
            if all(x is not None for x in [origin_lat, origin_lon, origin_rot]):
                print(f"Rotation parameters: origin_lat={origin_lat}, origin_lon={origin_lon}, origin_rot={origin_rot}")
            else:
                print("Warning: Rotation parameters (origin_lat, origin_lon, origin_rot) not found or incomplete. Assuming no rotation.")
            if extent_zmax is not None:
                print(f"extent_zmax from config: {extent_zmax}")
            else:
                print("Warning: extent_zmax not found in config.")
            if extent_z_spacing is not None:
                print(f"h_depth (extent_z_spacing) from config: {extent_z_spacing}")
            else:
                print("Warning: h_depth (extent_z_spacing) not found in config.")
    
    except Exception as e:
        print(f"Error accessing HDF5 file: {e}")
        raise
    
    # Debug: Print sample data values at the top and bottom of the original data array
    print(f"Sample {property_name} values at z[0] = {z[0]} (z_km = {z[0] * extent_z_spacing:.4f} km):")
    print(data[0, :5, :5])
    print(f"Sample {property_name} values at z[-1] = {z[-1]} (z_km = {z[-1] * extent_z_spacing:.4f} km):")
    print(data[-1, :5, :5])
    
    # 2) Convert z values to kilometers using extent_z_spacing (h_depth)
    if extent_z_spacing is None:
        raise ValueError("h_depth (extent_z_spacing) not found in HDF5 config; required to convert z values to kilometers.")
    
    z_km = z * extent_z_spacing  # Convert raw z to kilometers (positive depths: 0 to 45.8 km)
    print(f"z range in km (after conversion): {z_km.min():.4f} to {z_km.max():.4f}")
    
    # 3) Check ordering of z and data
    # z should be surface-to-deep (0 to 229), matching data (surface-to-deep)
    if z[0] < z[-1]:
        print("z is in ascending order (surface to deep), which matches data ordering.")
    else:
        print("z is in descending order (deep to surface). Reversing z and data to match surface-to-deep convention.")
        z = z[::-1]
        z_km = z_km[::-1]
        data = data[::-1, :, :]
    
    # 4) Set maximum depth
    if max_depth is None:
        if extent_zmax is None:
            raise ValueError("extent_zmax not found in HDF5 config and --max-depth not provided.")
        max_depth = extent_zmax
        print(f"Using default max_depth from extent_zmax: {max_depth} km")
    else:
        print(f"Using user-specified max_depth: {max_depth} km")
    
    # Clip the z array and data to the top max_depth km from the surface
    # z_km values are in surface-to-deep order (0 to 45.8 km), surface is at z_km.min() (0 km)
    surface_z_km = z_km.min()  # Should be 0 km
    max_depth_z_km = surface_z_km + max_depth  # e.g., 0 + 5 = 5 km
    z_mask = (z_km >= surface_z_km) & (z_km <= max_depth_z_km)
    if not np.any(z_mask):
        raise ValueError(f"No depths found within max_depth={max_depth} km from the surface. Mesh z range (km): {z_km.min()} to {z_km.max()}")
    z = z[z_mask]  # Keep raw z for interpolation
    z_km = z_km[z_mask]  # Keep z in km for plotting
    data = data[z_mask, :, :]  # Shape: (new_nz, lat_dim, lon_dim)
    print(f"Depths clipped to top {max_depth} km from the surface. New z range (km): {z_km.min():.4f} to {z_km.max():.4f}")
    
    # Debug: Print sample data values near the surface after clipping
    surface_index = np.argmin(z_km)  # Index of the surface (z_km = 0)
    print(f"Sample {property_name} values near the surface (z_km = {z_km[surface_index]:.4f} km):")
    print(data[surface_index, :5, :5])  # Print a 5x5 subset of the data at the surface
    
    # 5) Extract domain info for visualization
    domain_info = extract_domain_info(h5file)
    print(f"Domain info: {domain_info}")
    
    # 6) Transform mesh to rotated coordinates if rotation parameters are available
    lat1_orig, lon1_orig = lat1, lon1
    lat2_orig, lon2_orig = lat2, lon2
    
    print(f"Original cross-section coordinates: ({lat1}, {lon1}) to ({lat2}, {lon2})")
    
    if all(x is not None for x in [origin_lat, origin_lon, origin_rot]):
        # Transform the mesh coordinates to rotated system
        x_mesh = np.zeros((len(lat), len(lon)))
        y_mesh = np.zeros((len(lat), len(lon)))
        for i in range(len(lat)):
            for j in range(len(lon)):
                x_mesh[i, j], y_mesh[i, j] = geographic_to_rotated_coords(
                    lat[i], lon[j], origin_lat, origin_lon, origin_rot
                )
        
        # Transform the cross-section coordinates to rotated system
        x1_rot, y1_rot = geographic_to_rotated_coords(lat1, lon1, origin_lat, origin_lon, origin_rot)
        if lat2 is not None and lon2 is not None:
            x2_rot, y2_rot = geographic_to_rotated_coords(lat2, lon2, origin_lat, origin_lon, origin_rot)
        else:
            x2_rot, y2_rot = None, None
        
        print("Rotated cross-section coordinates: ({:.2f}, {:.2f}) to ({:.2f}, {:.2f})".format(
            x1_rot, y1_rot, x2_rot if x2_rot is not None else 0.0, y2_rot if y2_rot is not None else 0.0
        ))
        print("Rotated mesh x range: {:.2f} to {:.2f}".format(x_mesh.min(), x_mesh.max()))
        print("Rotated mesh y range: {:.2f} to {:.2f}".format(y_mesh.min(), y_mesh.max()))
    else:
        print("Using geographic coordinates (no rotation applied)")
        x_mesh, y_mesh = np.meshgrid(lon, lat)
        x1_rot, y1_rot = lon1, lat1
        x2_rot, y2_rot = lon2, lat2 if lat2 is not None else None
    
    # 7) Clip coordinates to the mesh domain (in rotated coordinates)
    is_diagonal = x2_rot is not None and y2_rot is not None and (x2_rot != x1_rot or y2_rot != y1_rot)
    x1_rot_orig, y1_rot_orig = x1_rot, y1_rot
    x2_rot_orig, y2_rot_orig = x2_rot, y2_rot
    
    if is_diagonal:
        print("Using simple mesh-extent clipping for diagonal slice in rotated coordinates")
        x1_rot = np.clip(x1_rot, x_mesh.min(), x_mesh.max())
        y1_rot = np.clip(y1_rot, y_mesh.min(), y_mesh.max())
        x2_rot = np.clip(x2_rot, x_mesh.min(), x_mesh.max())
        y2_rot = np.clip(y2_rot, y_mesh.min(), y_mesh.max())
    else:
        print("Using simple mesh-extent clipping in rotated coordinates")
        x1_rot = np.clip(x1_rot, x_mesh.min(), x_mesh.max())
        y1_rot = np.clip(y1_rot, y_mesh.min(), y_mesh.max())
        if x2_rot is not None:
            x2_rot = np.clip(x2_rot, x_mesh.min(), x_mesh.max())
        if y2_rot is not None:
            y2_rot = np.clip(y2_rot, y_mesh.min(), y_mesh.max())
    
    if (x1_rot, y1_rot, x2_rot, y2_rot) != (x1_rot_orig, y1_rot_orig, x2_rot_orig, y2_rot_orig):
        print("NOTE: Rotated coordinates clipped:")
        print("  Original rotated: ({:.4f}, {:.4f}) to ({:.4f}, {:.4f})".format(x1_rot_orig, y1_rot_orig, x2_rot_orig, y2_rot_orig))
        print("  Clipped rotated : ({:.4f}, {:.4f}) to ({:.4f}, {:.4f})".format(x1_rot, y1_rot, x2_rot, y2_rot))
    
    # 8) Sort axes if needed
    if len(lat) > 1 and lat[1] < lat[0]:
        print("Sorting latitude axis (was decreasing)")
        lat = lat[::-1]
        data = data[:, ::-1, :] if data.shape[1] == len(lat) else data
        x_mesh = x_mesh[::-1, :]
        y_mesh = y_mesh[::-1, :]
    
    if len(lon) > 1 and lon[1] < lon[0]:
        print("Sorting longitude axis (was decreasing)")
        lon = lon[::-1]
        data = data[:, :, ::-1] if data.shape[2] == len(lon) else data
        x_mesh = x_mesh[:, ::-1]
        y_mesh = y_mesh[:, ::-1]
    
    # 9) Set up cross-section path in rotated coordinates
    if x2_rot is not None and y2_rot is not None:
        x_pts = np.linspace(x1_rot, x2_rot, n_points)
        y_pts = np.linspace(y1_rot, y2_rot, n_points)
        distance = np.sqrt(((lat2-lat1)*111)**2 + ((lon2-lon1)*111*np.cos(np.radians((lat1+lat2)/2)))**2)
        x_label = f"Distance along profile (approx. {distance:.1f} km)"
    elif y2_rot is not None:
        y_pts = np.linspace(y1_rot, y2_rot, n_points)
        x_pts = np.full(n_points, x1_rot)
        x_label = "Y-coordinate (km)"
    elif x2_rot is not None:
        x_pts = np.linspace(x1_rot, x2_rot, n_points)
        y_pts = np.full(n_points, y1_rot)
        x_label = "X-coordinate (km)"
    else:
        raise ValueError("At least one of lat2 or lon2 must be provided to define a cross-section")
    
    print("Cross-section path x range: {:.2f} to {:.2f}".format(x_pts.min(), x_pts.max()))
    print("Cross-section path y range: {:.2f} to {:.2f}".format(y_pts.min(), y_pts.max()))
    
    # 10) Handle lat/lon dimension mismatch and set up interpolator grid
    lat_dim, lon_dim = data.shape[1], data.shape[2]
    if lat_dim != len(lat) or lon_dim != len(lon):
        print(f"Detected lat/lon mismatch: data dims ({lat_dim},{lon_dim}) vs mesh ({len(lat)},{len(lon)})")
        new_x = np.linspace(x_mesh.min(), x_mesh.max(), lat_dim)
        new_y = np.linspace(y_mesh.min(), y_mesh.max(), lon_dim)
    else:
        new_x = np.array([x_mesh[i, 0] for i in range(len(lat))])
        new_y = np.array([y_mesh[0, j] for j in range(len(lon))])
    
    # 11) Build interpolator on (z, new_x, new_y)
    # Use raw z values for interpolation (as they match the data array)
    print("Creating interpolator with grid sizes: {} {} {}".format(len(z), len(new_x), len(new_y)))
    interp = RegularGridInterpolator(
        (z, new_x, new_y),
        data,
        bounds_error=False,
        fill_value=np.nan,
        method='linear'
    )

    # 12) Sample cross-section
    print(f"Sampling cross-section along {n_points} points...")
    results = np.zeros((len(z), n_points))
    for i, depth in enumerate(z):
        if i % 20 == 0:
            print(f"Processing layer {i+1}/{len(z)}")
        pts = np.column_stack((np.full(n_points, depth), x_pts, y_pts))
        results[i, :] = interp(pts)

    print(f"Results shape: {results.shape}")
    print(f"Results min: {np.nanmin(results):.2f}, max: {np.nanmax(results):.2f}")
    print(f"Results contains NaN: {np.any(np.isnan(results))}")
    print(f"Results sample (first 5 points at first depth): {results[0, :5]}")
    
    nan_count = np.isnan(results).sum()
    total = results.size
    if nan_count > 0:
        print(f"Warning: {nan_count}/{total} ({nan_count/total:.1%}) NaN after interpolation")
    if nan_count == total:
        print("Error: all interpolated values are NaN. Coordinates likely outside mesh.")
        print("Try known-good coords: --lat1 -44.3 --lon1 170.3 --lat2 -44.3 --lon2 171.3")
        return
    if nan_count < total:
        fill = np.nanmin(results)
        results = np.nan_to_num(results, nan=fill)

    # 13) Plot cross-section
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Flip z_plot to have 0 at the top and max depth at the bottom
    z_plot = z_km[::-1]  # Reverse the z_km array
    results_plot = results[::-1, :]  # Reverse the results array along the depth axis
    
    print(f"Plotted depth range: {z_plot.min():.2f} to {z_plot.max():.2f} km")
    
    if lat2 is not None and lon2 is not None:
        x_vals = np.linspace(0, distance, n_points)
    elif lat2 is not None:
        x_vals = y_pts  # Since y corresponds to latitude in rotated coords
    else:
        x_vals = x_pts  # Since x corresponds to longitude in rotated coords
    
    pcm = ax.pcolormesh(x_vals, z_plot, results_plot, shading='auto', 
                      cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel("Depth (km)")
    ax.set_ylim([z_plot.max(), z_plot.min()])  # Flip the y-axis by setting max depth at the bottom
    
    unit = "(g/cm³)" if property_name == "rho" else "(km/s)"
    cbar = fig.colorbar(pcm)
    cbar.set_label(f"{property_name} {unit}")
    
    if lat2 is not None and lon2 is not None:
        title = f"{property_name} cross-section from ({lat1:.4f}, {lon1:.4f}) to ({lat2:.4f}, {lon2:.4f})"
    elif lat2 is not None:
        title = f"{property_name} cross-section at longitude {lon1:.4f}"
    else:
        title = f"{property_name} cross-section at latitude {lat1:.4f}"
    
    ax.set_title(title)
   
    if domain_info and all(key in domain_info for key in ['ORIGIN_LAT', 'ORIGIN_LON', 'ORIGIN_ROT']):
        corners, min_lat, max_lat, min_lon, max_lon, origin_lat, origin_lon, rotation = create_domain_polygon(h5file)
        
        import cartopy
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.geoaxes import GeoAxes
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        axins = inset_axes(ax, width="30%", height="30%", loc='lower left', 
                        bbox_to_anchor=(0.0, 0.0, 1, 1), bbox_transform=ax.transAxes,
                        axes_class=GeoAxes, 
                        axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
        
        axins.add_feature(cfeature.COASTLINE, linewidth=0.5)
        axins.add_feature(cfeature.LAND, facecolor='lightgray')
        axins.add_feature(cfeature.OCEAN, facecolor='lightblue')
        
        domain_lons = [corner[0] for corner in corners] + [corners[0][0]]
        domain_lats = [corner[1] for corner in corners] + [corners[0][1]]
        axins.plot(domain_lons, domain_lats, 'r--', linewidth=1.5, label='Domain', transform=ccrs.PlateCarree())
        
        axins.plot([lon1_orig, lon2_orig], [lat1_orig, lat2_orig], 'b-', linewidth=1.5, label='Cross-section', transform=ccrs.PlateCarree())
        axins.plot([lon1_orig], [lat1_orig], 'bo', markersize=4, transform=ccrs.PlateCarree())
        axins.plot([lon2_orig], [lat2_orig], 'bo', markersize=4, transform=ccrs.PlateCarree())
        
        axins.plot([origin_lon], [origin_lat], 'ro', markersize=4, transform=ccrs.PlateCarree())
        arrow_length = 0.05
        dx = arrow_length * np.cos(np.radians(rotation))
        dy = arrow_length * np.sin(np.radians(rotation))
        axins.arrow(origin_lon, origin_lat, dx, dy, 
                    head_width=0.02, head_length=0.02, fc='r', ec='r', transform=ccrs.PlateCarree())
        axins.text(origin_lon + dx*1.2, origin_lat + dy*1.2, f"{rotation:.1f}°", 
                ha='left', va='center', color='red', fontsize=8, transform=ccrs.PlateCarree())
        
        buffer = 0.5
        axins.set_extent([min_lon - buffer, max_lon + buffer, min_lat - buffer, max_lat + buffer], crs=ccrs.PlateCarree())
        
        axins.gridlines(draw_labels={"top": "x", "right": "y"}, linestyle='--', alpha=0.5, linewidth=0.5)
        axins.legend(loc='upper right', fontsize=4)
    
    plt.tight_layout()
    
    if output_png:
        if not output_png.lower().endswith('.png'):
            output_png += '.png'
        # Append max_depth to the filename
        base, ext = output_png.rsplit('.', 1)
        output_png = f"{base}_maxdepth-{max_depth}.{ext}"
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_png}")
    
    plt.show()
    print("Cross-section created successfully")
    return

def main():
    parser = argparse.ArgumentParser(description="Create improved cross-section plots")
    parser.add_argument("h5file", help="Path to HDF5 velocity model")
    parser.add_argument("--lat1", type=float, required=True, help="Start latitude")
    parser.add_argument("--lon1", type=float, required=True, help="Start longitude")
    parser.add_argument("--lat2", type=float, help="End latitude (if doing a lat cross-section or diagonal)")
    parser.add_argument("--lon2", type=float, help="End longitude (if doing a lon cross-section or diagonal)")
    parser.add_argument("-p", "--property", choices=["vs", "vp", "rho"],
                      default="vp", help="Property to plot")
    parser.add_argument("--png", help="Save plot as PNG image with specified filename")
    parser.add_argument("--vmin", type=float, help="Minimum value for colorbar")
    parser.add_argument("--vmax", type=float, help="Maximum value for colorbar")
    parser.add_argument("--max-depth", type=float, help="Maximum depth for the cross-section (km). Defaults to extent_zmax from HDF5 config.")
    parser.add_argument("--cmap", default='inferno', help="Colormap name (viridis, plasma, etc.)")
    parser.add_argument("-n", "--n_points", type=int, default=200, 
                      help="Number of points along cross-section")
    
    args = parser.parse_args()
    
    if args.lat2 is None and args.lon2 is None:
        parser.error("At least one of --lat2 or --lon2 must be provided")
    
    print("Note: If this cross-section fails, try these known-good test coordinates:")
    print("  --lat1 -44.3 --lon1 170.3 --lat2 -44.3 --lon2 171.3")
    
    improved_cross_section(
        args.h5file,
        args.lat1,
        args.lon1,
        args.lat2,
        args.lon2,
        args.n_points,
        args.property,
        args.png,
        args.vmin,
        args.vmax,
        args.cmap,
        args.max_depth
    )

if __name__ == "__main__":
    main()