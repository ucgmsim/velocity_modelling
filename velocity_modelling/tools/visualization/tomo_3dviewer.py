"""
3D Tomography Viewer for HDF5-based velocity models.

This script provides an interactive 3D visualization of tomography slices
stored in an HDF5 file. It uses PyVista and PyQt for the user interface
and rendering. Users can toggle the visibility of different elevation layers,
overlay original data points from a text file, and inspect the model from
various angles.

The script is run from the command line and accepts various options to
customize the visualization, such as selecting the scalar field, defining
latitude/longitude ranges, and setting color map limits.
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

import h5py
import numpy as np
import pandas as pd
import pyvista as pv
import typer
from pyvista.plotting.camera import Camera
from pyvistaqt import QtInteractor
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QLabel,
    QMainWindow,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)

DEFAULT_CAMERA_ANGLE = (-0.018774705983602088, 0.8058896333404005, 0.5917680367252902)


def is_number(s: str) -> bool:
    """Check if a string can be converted to a float.

    Parameters
    ----------
    s : str
        The input string.

    Returns
    -------
    bool
        True if the string can be converted to a float, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_ep_txt(txt_path: Path) -> pd.DataFrame:
    """Read an EP-style tomography text file into a pandas DataFrame.

    The file is expected to have a specific format with columns for velocity
    parameters and coordinates. This function handles file parsing, column
    naming, and coordinate adjustments (elevation sign and longitude range).

    Parameters
    ----------
    txt_path : Path
        Path to the input text file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the tomography data with standardized column names.
    """
    col_names = [
        "vp",
        "vp_o_vs",
        "vs",
        "rho",
        "sf_vp",
        "sf_vp_o_vs",
        "x",
        "y",
        "elevation",
        "lat",
        "lon",
    ]
    df = pd.read_csv(txt_path, sep=r"\s+", skiprows=2, names=col_names, engine="python")
    df["elevation"] = -1 * df["elevation"]
    df["lon"] = np.where(df["lon"] < 0, df["lon"] + 360, df["lon"])
    return df


def load_stacked_slices(
    h5_path: Path,
    scalar: str = "vp",
    lat_range: Optional[tuple[float, float]] = None,
    lon_range: Optional[tuple[float, float]] = None,
) -> tuple[dict[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load and stack tomography slices from an HDF5 file.

    Reads data for specified elevation layers from an HDF5 file, optionally
    cropping it to a given latitude and longitude range.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 file.
    scalar : str, optional
        The scalar field to load (e.g., "vp", "vs"), by default "vp".
    lat_range : tuple of float, optional
        The latitude range (min, max) to crop the data. If None, all latitudes
        are used. By default None.
    lon_range : tuple of float, optional
        The longitude range (min, max) to crop the data. If None, all latitudes
        are used. By default None.

    Returns
    -------
    tuple
        A tuple containing:
        - keys (list[str]): Sorted list of elevation keys.
        - lon (np.ndarray): Array of longitudes within the specified range.
        - lat (np.ndarray): Array of latitudes within the specified range.
        - scalar_values np.ndarray: A 3d array containing scalar data of shape (n_elevations, n_latitudes, n_longitudes).
    """
    with h5py.File(h5_path, "r") as f:
        keys = sorted((k for k in f.keys() if is_number(k)), key=float, reverse=True)
        sample = f[keys[0]]
        lat_full = sample["latitudes"][:]
        lon_full = sample["longitudes"][:]

        lat_mask = (
            (lat_full >= lat_range[0]) & (lat_full <= lat_range[1])
            if lat_range
            else np.ones_like(lat_full, bool)
        )
        lon_mask = (
            (lon_full >= lon_range[0]) & (lon_full <= lon_range[1])
            if lon_range
            else np.ones_like(lon_full, bool)
        )

        lat = lat_full[lat_mask]
        lon = lon_full[lon_mask]

        scalar_values = []
        for k in keys:
            data = f[k][scalar][:][np.ix_(lat_mask, lon_mask)]
            scalar_values.append(data)
        return keys, lon, lat, np.array(scalar_values)


def make_flat_surfaces(
    elevations: list[str],
    lon: np.ndarray,
    lat: np.ndarray,
    scalar_values: np.ndarray,
    gap: float = 0.1,
) -> list[tuple[pv.StructuredGrid, float, float, float]]:
    """Create a list of PyVista surfaces from stacked 2D data.

    Each slice in the data cube is converted into a flat surface mesh,
    vertically stacked with a specified gap.

    Parameters
    ----------
    elevations : list[str]
        List of elevation values for each slice.
    lon : np.ndarray
        1D array of longitudes.
    lat : np.ndarray
        1D array of latitudes.
    scalar_values : np.ndarray
        3D data array with shape (elevation, lat, lon).
    gap : float, optional
        Vertical gap between the stacked surfaces, by default 0.1.


    Returns
    -------
    tuple
        A tuple containing:
        - grids (dict): A dictionary mapping elevations to their corresponding PyVista surface meshes.
        Each key is a float elevation value, and the value is a PyVista StructuredGrid object.
        - global_min (float): The minimum value in the entire data cube, used for consistent color mapping.
        - global_max (float): The maximum value in the entire data cube, used for consistent color mapping.

    """

    grid_dict = {}
    global_min = np.min(scalar_values)
    global_max = np.max(scalar_values)

    for iz in range(len(elevations)):
        z = -gap * iz
        xs, ys = np.meshgrid(lon, lat, indexing="xy")
        zs = np.full_like(xs, z)
        surf = pv.StructuredGrid(xs, ys, zs).extract_surface()
        surf["values"] = scalar_values[iz].ravel(order="C")
        grid_dict[float(elevations[iz])] = surf

    return grid_dict, global_min, global_max


class TomoApp(QMainWindow):
    """Main application window for the 3D tomography viewer.

    This class encapsulates the Qt-based GUI, including the PyVista plotter
    and layer visibility controls.


    Parameters
    ----------
    title : str
        Title for the window.
    scalar_name : str
        Name of the scalar field being displayed.
    grid_dict : dict[pv.StructuredGrid]
        Dictionary mapping elevations to PyVista StructuredGrid objects.
    clim : tuple, optional
        Color map limits (min, max) to override the defaults. By default None.
    points_by_elevation : dict, optional
        Dictionary mapping elevations to point data to be overlaid. By default None.
    debug : bool, optional
        Flag to enable debug printing. By default False.

    """

    def __init__(
        self,
        title: str,
        scalar_name: str,
        grid_dict: dict[pv.StructuredGrid],
        clim: tuple[float, float],
        points_by_elevation: Optional[dict[float, np.ndarray]] = None,
        debug: bool = False,
    ):
        """Initialize the Tomography 3D Viewer application."""

        super().__init__()
        self.debug = debug

        # Find the topmost grid (highest elevation)
        top_grid = grid_dict[max(grid_dict.keys())]
        # Dynamically determine focal point from the center of the top grid
        self.focal_point = top_grid.center

        # Store camera settings relative to the focal point
        self.camera_position = (
            self.focal_point[0] - 1.63,
            self.focal_point[1] - 46.57,
            self.focal_point[2] + 63.37,
        )
        self.view_up = DEFAULT_CAMERA_ANGLE

        self.plotter_widget = QtInteractor(self)

        self.setWindowTitle(f"Tomography 3D Viewer - {title}")

        title_label = QLabel(f"{title}- {scalar_name}")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.plotter_widget)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.actors = []
        self.slice_actor = None
        self.point_actors = []
        self.points_by_elevation = points_by_elevation

        self.toggle_panel = QWidget()
        layout = QVBoxLayout(self.toggle_panel)
        #        layout.addWidget(QLabel(self.windowTitle()))
        layout.addWidget(QLabel("Layer Visibility"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.toggle_panel)

        dock = QDockWidget("Layers", self)
        dock.setWidget(scroll)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        for i, elevation in enumerate(
            sorted(grid_dict.keys())[::-1]
        ):  # reverse order for top-down view
            grid = grid_dict[elevation]
            clim = clim
            actor = self.plotter_widget.add_mesh(
                grid,
                scalars="values",
                cmap="hot_r",
                opacity=1.0,
                clim=clim,
                show_scalar_bar=False,
                name=f"{elevation}km",
                lighting=False,
            )
            self.actors.append(actor)

            cb = QCheckBox(f"{elevation:.0f} km")
            cb.setChecked(True)

            # Add optional point overlay for this elevation if provided
            if self.points_by_elevation and elevation in self.points_by_elevation:
                z = -i * 0.1 + 0.01  # slightly above the surface
                pts = self.points_by_elevation[elevation]
                pdata = pv.PolyData(
                    np.column_stack((pts[:, 1], pts[:, 0], np.full(len(pts), z)))
                )
                pdata["values"] = pts[:, 2] if pts.shape[1] > 2 else np.zeros(len(pts))
                if self.debug:
                    print(pdata["values"])

                p_actor = self.plotter_widget.add_mesh(
                    pdata,
                    scalars="values",
                    cmap="hot_r",
                    point_size=4,
                    render_points_as_spheres=True,
                    show_scalar_bar=False,
                    clim=clim,
                )
                self.point_actors.append((elevation, p_actor))
                if elevation == max(self.points_by_elevation):
                    p_actor.SetVisibility(True)
                else:
                    p_actor.SetVisibility(False)

            cb.stateChanged.connect(
                lambda state, a=actor: self.set_visibility(a, state)
            )
            layout.addWidget(cb)

        self.plotter_widget.add_scalar_bar(title=scalar_name.upper(), n_labels=5)
        self.plotter_widget.add_axes(
            xlabel="Longitude (°E)",
            ylabel="Latitude (°N)",
            zlabel="Elevation",
            label_size=(0.4, 0.4),
        )
        self.plotter_widget.camera_position = "iso"
        self.plotter_widget.set_scale(xscale=1, yscale=1, zscale=0.05)
        self.plotter_widget.renderer.interpolate_before_map = True

        # 🧭 Set custom camera view
        self.reset_camera()

        self.plotter_widget.add_key_event("v", self.toggle_all)
        self.plotter_widget.add_key_event("s", self.toggle_slice)
        self.plotter_widget.add_key_event("r", self.reset_camera)

        self.plotter_widget.renderer.GetActiveCamera().AddObserver(
            "ModifiedEvent", self.on_camera_modified
        )

    def set_topmost_point_visibility(self) -> None:
        """Control visibility of point overlays.

        Ensures that only the points corresponding to the topmost visible
        elevation layer are displayed.
        """
        if not self.points_by_elevation:
            return
        checkboxes = self.toggle_panel.findChildren(QCheckBox)
        visible_elevations = [
            float(cb.text().split()[0]) for cb in checkboxes if cb.isChecked()
        ]
        print(visible_elevations)
        if visible_elevations:
            top = max(visible_elevations)  #
            for d, a in self.point_actors:
                is_visible = d == top
                a.SetVisibility(is_visible)
                if is_visible:
                    print(f"👁️ Top visible elevation: {top} km")
                    pdata = a.GetMapper().GetInputAsDataSet()
                    if "values" in pdata.point_data:
                        print("📊 Scalar values:", pdata["values"])
                    else:
                        print("⚠️ 'values' not found in point data")

    def set_visibility(self, actor: pv.Actor, state: int) -> None:
        """Set the visibility of a layer and update point overlays.

        This is a slot connected to the layer visibility checkboxes.

        Parameters
        ----------
        actor : pv.Actor
            The main surface actor for the layer.

        state : int
            The state of the checkbox (Qt.Checked or Qt.Unchecked).

        """
        actor.SetVisibility(state == Qt.Checked)
        self.set_topmost_point_visibility()
        self.plotter_widget.render()

    def toggle_all(self) -> None:
        """Toggle the visibility of all layers."""
        for actor in self.actors:
            actor.SetVisibility(not actor.GetVisibility())
        self.plotter_widget.render()

    def toggle_slice(self) -> None:
        """Toggle a cross-section slice view."""
        if self.slice_actor:
            self.plotter_widget.remove_actor(self.slice_actor)
            self.slice_actor = None
        else:
            grid = self.actors[len(self.actors) // 2].GetMapper().GetInputAsDataSet()
            bounds = grid.bounds
            x_center = 0.5 * (bounds[0] + bounds[1])
            sliced = grid.slice(normal="x", origin=(x_center, 0, 0))
            self.slice_actor = self.plotter_widget.add_mesh(
                sliced, color="white", line_width=2
            )
        self.plotter_widget.render()

    def reset_camera(self) -> None:
        """Reset the camera to a predefined position and orientation."""
        camera = self.plotter_widget.renderer.GetActiveCamera()
        camera.SetPosition(self.camera_position)
        camera.SetFocalPoint(self.focal_point)
        camera.SetViewUp(self.view_up)
        self.plotter_widget.renderer.ResetCameraClippingRange()
        self.plotter_widget.render()

    def on_camera_modified(self, caller: Camera, event: str) -> None:
        """Print camera parameters when modified (for debugging).

        Parameters
        ----------
        caller : Camera
            The camera object that triggered the event.
        event : str
            The event name (e.g., "ModifiedEvent").
        """
        if not self.debug:
            return

        cam = self.plotter_widget.renderer.GetActiveCamera()
        print("📸 Camera changed:")
        print("  Position   :", cam.GetPosition())
        print("  FocalPoint :", cam.GetFocalPoint())
        print("  ViewUp     :", cam.GetViewUp())


@cli.from_docstring(app)
def launch_viewer(
    h5file: Annotated[Path, typer.Argument(help="HDF5 file with tomography slices")],
    scalar: Annotated[
        str, typer.Option(help="Scalar field to show", show_default=True)
    ] = "vp",
    gap: Annotated[float, typer.Option(help="Vertical gap between slices")] = 0.2,
    lat_range: Annotated[
        Optional[tuple[float, float]], typer.Option(help="Latitude range [min max]")
    ] = None,
    lon_range: Annotated[
        Optional[tuple[float, float]], typer.Option(help="Longitude range [min max]")
    ] = None,
    clim: Annotated[
        Optional[tuple[float, float]], typer.Option(help="Color range [min max]")
    ] = None,
    txt: Annotated[
        Optional[Path],
        typer.Option(help="Optional EP-style TXT input file for original points"),
    ] = None,
    debug: Annotated[bool, typer.Option(help="Enable debug print statements")] = False,
) -> None:
    """Launch the interactive tomography viewer with optional overlays and controls.

    This function serves as the main entry point for the command-line application.
    It parses arguments, loads data, and initializes the Qt application and
    the TomoApp viewer window.

    Parameters
    ----------
    h5file : Path
        Path to the HDF5 file with tomography slices.
    scalar : str, optional
        Scalar field to show (e.g., "vp", "vs"), by default "vp".
    gap : float, optional
        Vertical gap between slices, by default 0.2.
    lat_range : tuple of float, optional
        Latitude range (min, max) to display, by default None.
    lon_range : tuple of float, optional
        Longitude range (min, max) to display, by default None.
    clim : tuple of float, optional
        Color range (min, max) for the scalar bar, by default None.
    txt : Path, optional
        Optional EP-style TXT input file for original points overlay, by default None.
    debug : bool, optional
        Enable debug print statements, by default False.
    """
    elevations, lon, lat, scalar_values = load_stacked_slices(
        h5file, scalar=scalar, lat_range=lat_range, lon_range=lon_range
    )
    grid_dict, vmin, vmax = make_flat_surfaces(
        elevations, lon, lat, scalar_values, gap=gap
    )

    points_by_elevation = None
    if txt:
        df = read_ep_txt(txt)
        df_filtered = df[df["elevation"].isin([float(d) for d in elevations])]
        if debug:
            print(df_filtered.head())
        points_by_elevation = {
            float(d): df_filtered[
                np.isclose(df_filtered["elevation"], float(d), atol=1e-3)
            ][["lat", "lon", scalar]].values
            for d in elevations
        }

    app_qt = QApplication.instance() or QApplication(sys.argv)

    # validate clim input
    if clim is None:
        clim = (vmin, vmax)
    else:
        if len(clim) != 2 or not all(isinstance(c, (int, float)) for c in clim):
            raise ValueError("clim must be a tuple of two numeric values (min, max).")
        clim = tuple(clim)

    window = TomoApp(
        h5file.stem,
        scalar,
        grid_dict,
        clim,
        points_by_elevation=points_by_elevation,
        debug=debug,
    )
    window.show()
    # Ensure clean shutdown
    app_qt.lastWindowClosed.connect(app_qt.quit)

    sys.exit(app_qt.exec_())


if __name__ == "__main__":
    app()
