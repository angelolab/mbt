from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property, wraps
from os import PathLike
from typing import Any, Literal

try:
    from typing import TypedDict, Unpack
except ImportError:
    from typing_extensions import TypedDict, Unpack

from pathlib import Path

import numpy as np
import numpydantic.dtype as ndt
import patito as pt
import polars as pl
import xarray as xr
from loguru import logger
from numpydantic import NDArray
from polars._typing import ParquetCompression
from polars.io.cloud.credential_provider._providers import CredentialProviderFunction
from upath import UPath

from mbt._core import MibiReader
from mbt.core.models import (
    C,
    ImageType,
    MassCalibrationModel,
    MibiDataModel,
    MibiFilePanelModel,
    UserPanelModel,
    X,
    Y,
)
from mbt.core.utils import (
    _set_tof_ranges,
    format_image_name,
    format_run_name,
    resolve_image_types,
)
from mbt.im import to_xarray
from mbt.io import write_dir_tiffs


# Helper decorator to handle potential errors during property loading
def _property_loader(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # Check if already computed by cached_property mechanism
            return func(self, *args, **kwargs)
        except (pt.DataFrameValidationError, Exception) as e:
            prop_name = func.__name__
            logger.error(f"Error loading property '{prop_name}' for file {self.file_path}: {e}")
            return None

    return wrapper


class WriteParquetKwargs(TypedDict):
    """Parameters for writing Parquet files."""

    compression: ParquetCompression
    compression_level: int | None
    statistics: bool | str | dict[str, bool]
    row_group_size: int | None
    data_page_size: int | None
    use_pyarrow: bool
    pyarrow_options: dict[str, Any]
    partition_by: str | Sequence[str]
    partition_chunk_size_bytes: int
    storage_options: dict[str, Any]
    credential_provider: CredentialProviderFunction | Literal["auto"] | None
    retries: int


class MibiFile:
    """
    Context manager and data loader for MIBI binary files.

    Loads pixel data and associated metadata (panel, mass calibration) lazily
    using cached properties or standard properties. Validates data structures
    using Patito/Pydantic models. A user-supplied panel can be provided for
    validation against UserPanelModel; otherwise, the panel embedded in the
    MIBI file is loaded and validated against MibiFilePanelModel.
    """

    # --- Initialization & Context Management --- #

    def __init__(
        self,
        file_path: PathLike,
        panel: pl.DataFrame | None = None,
        time_resolution: float | None = 500e-6,
        num_chunks: int = 32,
        sparsity: bool = False,
    ):
        """Initialize the MibiFile context manager.

        Parameters
        ----------
        file_path
            Path to the MIBI binary file.
        panel
            Optional user-supplied Polars DataFrame representing the panel.
            If provided, it will be validated against `UserPanelModel`.
            If None, the panel embedded in the MIBI file will be loaded and
            validated against `MibiFilePanelModel`.
        time_resolution
            Optional time resolution override in microseconds (e.g., 500e-6).
            If None, the value from the MIBI file's mass calibration will be used.
        num_chunks
            Number of chunks to divide the MIBI file into for parallel processing
            by the Rust backend. (more is better, but after 64 there are diminishing returns).
        sparsity
            If True, indicates that the resulting xarray images should use
            sparse arrays (e.g., `sparse.COO`). Defaults to False.

        Raises
        ------
        FileNotFoundError
            If the specified `file_path` does not exist or is not a file.
        pt.DataFrameValidationError
            If the user-supplied `panel` fails validation against `UserPanelModel`.
        """
        if (fp := UPath(file_path)).is_file():
            self.file_path = fp
        else:
            raise FileNotFoundError(f"MIBI file not found: {fp}")

        self.reader = MibiReader(self.file_path, num_chunks)
        self.num_chunks = num_chunks
        self._time_resolution_override = time_resolution
        self.sparsity = sparsity
        self._panel: pt.DataFrame[MibiFilePanelModel] | pt.DataFrame[UserPanelModel] | None = None

        self._user_panel: pt.DataFrame[UserPanelModel] | None = None
        if panel is not None:
            try:
                # Validate and store the user-provided panel immediately
                self._user_panel = UserPanelModel.validate(panel)
                logger.success("User-supplied panel validated successfully.")
            except (pt.DataFrameValidationError, Exception) as e:
                logger.error(f"User-supplied panel validation failed for file: {self.file_path}: {e}")
                raise e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    # --- Core Data Properties --- #

    @cached_property
    @_property_loader
    def dimensions(self) -> tuple[int, int]:
        """(Cached) Returns the (n_x_pixels, n_y_pixels) dimensions from the file."""
        # Note: Rust returns width, height. MibiFile typically uses y, x internally.
        width, height = self.reader.dimensions
        return width, height

    @property
    def n_x_pixels(self) -> int:
        """Returns the number of pixels in the x-dimension."""
        return self.dimensions[0]

    @property
    def n_y_pixels(self) -> int:
        """Returns the number of pixels in the y-dimension."""
        return self.dimensions[1]

    @cached_property
    @_property_loader
    def mass_calibration(self) -> MassCalibrationModel:
        """(Cached) Loads and validates the mass calibration data."""
        mass_calibration_json = self.reader.mass_calibration_json
        return MassCalibrationModel.model_validate_json(mass_calibration_json)

    @property
    def time_resolution(self) -> float:
        """Returns the time resolution (override or from mass calibration)."""
        return self._time_resolution_override or self.mass_calibration.time_resolution

    @cached_property
    @_property_loader
    def fov_id(self) -> str:
        """(Cached) Returns the FOV ID."""
        return self.reader.fov_id

    @cached_property
    @_property_loader
    def fov_name(self) -> str:
        """(Cached) Returns the FOV name."""
        return self.reader.fov_name or self.file_path.stem

    @cached_property
    @_property_loader
    def run_name(self) -> str:
        """(Cached) Returns the run name."""
        return self.reader.run_name

    @cached_property
    @_property_loader
    def run_uuid(self) -> str:
        """(Cached) Returns the run UUID."""
        return self.reader.run_uuid

    @cached_property
    @_property_loader
    def fov_size_microns(self) -> float:
        """(Cached) Returns the FOV size in microns."""
        return self.reader.fov_size_microns

    # --- Instrument/Acquisition Properties --- #

    @cached_property
    @_property_loader
    def instrument_identifier(self) -> str:
        """(Cached) Returns the instrument identifier string."""
        return self.reader.instrument_identifier

    @cached_property
    @_property_loader
    def instrument_control_version(self) -> str:
        """(Cached) Returns the instrument control software version string."""
        return self.reader.instrument_control_version

    @cached_property
    @_property_loader
    def tof_app_version(self) -> str:
        """(Cached) Returns the TOF application version string."""
        return self.reader.tof_app_version

    @cached_property
    @_property_loader
    def dwell_time_millis(self) -> float:
        """(Cached) Returns the dwell time per pixel in milliseconds."""
        return self.reader.dwell_time_millis

    @cached_property
    @_property_loader
    def acquisition_status(self) -> str:
        """(Cached) Returns the acquisition status string."""
        return self.reader.acquisition_status

    @cached_property
    @_property_loader
    def scan_count(self) -> int:
        """(Cached) Returns the number of scans performed for the FOV."""
        return self.reader.scan_count

    @cached_property
    @_property_loader
    def imaging_preset_name(self) -> str:
        """(Cached) Returns the name of the imaging preset used."""
        return self.reader.imaging_preset_name

    @cached_property
    @_property_loader
    def imaging_aperture(self) -> str:
        """(Cached) Returns the imaging aperture setting used."""
        return self.reader.imaging_aperture

    @cached_property
    @_property_loader
    def acquisition_start_timestamp(self) -> int:
        """(Cached) Returns the acquisition start timestamp (likely Unix epoch milliseconds)."""
        return self.reader.acquisition_start_timestamp

    @cached_property
    @_property_loader
    def acquisition_end_timestamp(self) -> int:
        """(Cached) Returns the acquisition end timestamp (likely Unix epoch milliseconds)."""
        return self.reader.acquisition_end_timestamp

    # --- Formatting Properties --- #

    @cached_property
    def formatted_fov_name(self) -> str:
        """(Cached) Returns the formatted FOV name (e.g., 'fov-1-R13C3')."""
        return format_image_name(self.fov_id, self.fov_name)

    @cached_property
    def formatted_run_name(self) -> str:
        """(Cached) Returns the formatted run name (e.g., '2022-07-04_NBL_TMA2')."""
        return format_run_name(self.run_name)

    @property
    def panel(self) -> pt.DataFrame[MibiFilePanelModel] | pt.DataFrame[UserPanelModel]:
        """
        Returns the panel DataFrame.

        Loads from file and validates on first access if not provided by the user.
        Returns the panel with TOF ranges if `add_tof_ranges_to_panel` has been called.
        """
        if self._panel is None:
            if self._user_panel is not None:
                self._panel = self._user_panel
            else:
                try:
                    file_panel_df = self.reader.panel
                    if file_panel_df is None:
                        raise ValueError("Could not load panel from MIBI file.")
                    self._panel = MibiFilePanelModel.validate(file_panel_df)
                except (pt.DataFrameValidationError, Exception) as e:
                    logger.error(f"Error loading/validating panel from file {self.file_path}: {e}")
                    raise e

        # If _panel somehow still None, raise error (shouldn't happen with checks above)
        if self._panel is None:
            raise AttributeError("Panel could not be loaded or validated.")

        return self._panel

    @property
    def n_masses(self) -> int:
        """(Cached) Returns the number of masses in the panel."""
        return len(self.panel.get_column("mass").unique())

    @property
    def n_targets(self) -> int:
        """(Cached) Returns the number of targets in the panel."""
        return len(self.panel.get_column("target").unique())

    @cached_property
    @_property_loader
    def data(self) -> pt.LazyFrame[MibiDataModel]:
        """(Cached) Lazily loads and returns the validated pulse data LazyFrame."""
        raw_data_df: pl.DataFrame = self.reader.get_dataframe()
        # Validate the raw data against the MibiDataModel
        validated_data_df = MibiDataModel.validate(raw_data_df)
        return validated_data_df.lazy()

    # --- Panel Operations --- #

    def add_tof_ranges_to_panel(self) -> None:
        """
        Adds lower and upper time-of-flight (TOF) range columns to the panel.

        Calculates TOF ranges based on the 'mass_start' and 'mass_stop'
        columns using the file's mass calibration. Modifies the internal panel
        state, so subsequent accesses to `self.panel` will include these columns.

        Raises
        ------
            ValueError: If the panel DataFrame lacks 'mass_start' or 'mass_stop'.
            AttributeError: If mass calibration data cannot be loaded.
        """
        # Access properties to ensure they are loaded
        current_panel = self.panel

        panel_with_ranges = _set_tof_ranges(
            panel=current_panel,  # Use the currently loaded panel
            mass_offset=self.mass_calibration.mass_offset,
            mass_gain=self.mass_calibration.mass_gain,
            time_resolution=self.time_resolution,
        )

        validation_model = UserPanelModel if self._user_panel is not None else MibiFilePanelModel
        try:
            self._panel = validation_model.validate(panel_with_ranges, allow_superfluous_columns=True)
        except (pt.DataFrameValidationError, Exception) as e:
            logger.error(f"Error validating panel after adding TOF ranges: {e}")
            raise e

    def get_channel_info(self, channel_name: str) -> pl.DataFrame | None:
        """Get panel info for a specific channel using the 'target' column."""
        try:
            return self.panel.filter(pl.col("target") == channel_name)
        except pl.exceptions.PolarsError as e:
            logger.error(f"Error filtering panel for channel '{channel_name}': {e}")
            return None

    # --- Image Generation --- #

    def _generate_channel_images(
        self,
        aggregation_expr: pl.Expr,
        output_dtype: type,
        required_data_cols: list[str],
        result_col_alias: str,
        image_type: ImageType,
    ) -> xr.DataArray:
        """Internal helper to generate multichannel images based on an aggregation."""
        # Access properties to trigger loading if needed
        current_panel = self.panel
        pulse_data = self.data
        fov_microns = self.fov_size_microns
        nx = self.n_x_pixels
        ny = self.n_y_pixels

        # Check if TOF ranges are present, add them if not
        if "lower_tof_range" not in current_panel.columns or "upper_tof_range" not in current_panel.columns:
            self.add_tof_ranges_to_panel()
            current_panel = self.panel  # Re-access panel to get the updated version

        channel_col = "target" if "target" in current_panel.columns else "channel_name"
        if channel_col not in current_panel.columns:
            raise AttributeError("Panel missing channel identifier column ('target' or 'channel_name').")

        # Group panel by TOF ranges and aggregate channel names
        grouped_panel = (
            current_panel.select(channel_col, "lower_tof_range", "upper_tof_range")
            .group_by(["lower_tof_range", "upper_tof_range"])
            .agg(pl.col(channel_col).sort().str.join("_").alias("grouped_channel_name"))
            .sort("lower_tof_range")
        )

        channel_names: list[str] = grouped_panel.get_column("grouped_channel_name").to_list()
        output_images: dict[str, NDArray] = {name: np.zeros((ny, nx), dtype=output_dtype) for name in channel_names}

        for row in grouped_panel.iter_rows(named=True):
            grouped_channel_name = row["grouped_channel_name"]
            lower_tof = row["lower_tof_range"]
            upper_tof = row["upper_tof_range"]

            # Filter data for the current TOF range group
            # Use the 'pulse_data' property which returns the LazyFrame
            pixel_aggregated_channel = (
                pulse_data.filter(pl.col("pulse_time").is_between(lower_tof, upper_tof, closed="both"))
                .select(["pixel_y", "pixel_x"] + required_data_cols)
                .group_by(["pixel_y", "pixel_x"])
                .agg(aggregation_expr)
                .collect()
            )

            # Populate the NumPy array for this channel group
            y_coords: NDArray = pixel_aggregated_channel.get_column("pixel_y").to_numpy(allow_copy=False)
            x_coords: NDArray = pixel_aggregated_channel.get_column("pixel_x").to_numpy(allow_copy=False)
            values: NDArray = pixel_aggregated_channel.get_column(result_col_alias).to_numpy(allow_copy=False)
            output_images[grouped_channel_name][y_coords, x_coords] = values

        img_xr = to_xarray(
            output_images,
            run_name=self.formatted_run_name,
            fov_name=self.formatted_fov_name,
            image_type=image_type,
            sparsity=self.sparsity,
            scale={
                C: 1,
                X: fov_microns / nx,
                Y: fov_microns / ny,
            },
        )
        return img_xr

    # --- Image Properties (remain cached) --- #

    @cached_property
    def counts_image(self) -> xr.DataArray:
        """(Cached) Returns the counts image."""
        return self._generate_channel_images(
            aggregation_expr=pl.count().alias("counts"),
            output_dtype=ndt.UInt32,
            required_data_cols=[],
            result_col_alias="counts",
            image_type=ImageType.COUNTS,
        )

    @cached_property
    def intensity_image(self) -> xr.DataArray:
        """(Cached) Returns the intensity image."""
        return self._generate_channel_images(
            aggregation_expr=pl.sum("pulse_intensity").alias("sum_intensity"),
            output_dtype=ndt.Int64,
            required_data_cols=["pulse_intensity"],
            result_col_alias="sum_intensity",
            image_type=ImageType.INTENSITY,
        )

    @cached_property
    def intensity_width_image(self) -> xr.DataArray:
        """(Cached) Returns the intensity*width image."""
        return self._generate_channel_images(
            aggregation_expr=(pl.col("pulse_intensity") * pl.col("pulse_width")).sum().alias("sum_intensity_width"),
            output_dtype=ndt.Int64,
            required_data_cols=["pulse_intensity", "pulse_width"],
            result_col_alias="sum_intensity_width",
            image_type=ImageType.INTENSITY_WIDTH,
        )

    # --- Data Export --- #

    def get_selected_image_data(self, image_types: list[ImageType]) -> dict[str, xr.DataArray]:
        """Get a dictionary of selected image data arrays based on the provided image types."""
        selected_data: dict[str, xr.DataArray] = {}
        if ImageType.COUNTS in image_types:
            selected_data[ImageType.COUNTS.value] = self.counts_image
        if ImageType.INTENSITY in image_types:
            selected_data[ImageType.INTENSITY.value] = self.intensity_image
        if ImageType.INTENSITY_WIDTH in image_types:
            selected_data[ImageType.INTENSITY_WIDTH.value] = self.intensity_width_image
        return selected_data

    def write_data(self, file_path: PathLike, **kwargs: Unpack[WriteParquetKwargs]) -> None:
        """Write the raw pulse data (LazyFrame) to a Parquet file.

        Parameters
        ----------
        file_path
            Path to the output file.
        kwargs
            Additional keyword arguments to pass to `pl.DataFrame.write_parquet`.
        """
        output_path = UPath(file_path)
        self.data.collect().write_parquet(output_path, **kwargs)

    def write_dataset_to_zarr(
        self,
        output_directory: PathLike,
        image_types: ImageType | str | list[ImageType] | list[str] | list[ImageType | str] | None = None,
        shard_shape: dict[str, int] | Literal["auto"] | None = "auto",
    ) -> None:
        """Write the generated image DataArrays to an Xarray Dataset Zarr store.

        Parameters
        ----------
        output_directory
            Path to the output directory.
        image_types
            List of image types to write. If None, all image types will be written.
        shard_shape
            Shape of the shards to write. If "auto", the shard shape will be determined automatically.
        """
        from mbt.io.zarr import write_dataset_to_zarr

        resolved_types = resolve_image_types(image_types)
        if not resolved_types:
            logger.warning(
                f"No image types resolved for Zarr dataset writing for {self.file_path}. Nothing will be written."
            )
            return

        selected_images_dict = self.get_selected_image_data(resolved_types)
        if not selected_images_dict:
            logger.warning(
                f"Dataset for Zarr is empty after filtering by image_types for {self.file_path}. Nothing will be written."
            )
            return

        dataset_to_write = xr.Dataset(selected_images_dict)

        default_storage_chunks = {C: 1, Y: min(self.n_y_pixels, 256), X: min(self.n_x_pixels, 256)}

        write_dataset_to_zarr(
            dataset=dataset_to_write,
            output_directory=output_directory,
            run_fmt_name=self.formatted_run_name,
            fov_fmt_name=self.formatted_fov_name,
            n_masses=len(dataset_to_write.get(C, [] if C not in dataset_to_write else dataset_to_write[C])),
            ny=self.n_y_pixels,
            nx=self.n_x_pixels,
            storage_chunks=default_storage_chunks,
            shard_shape=shard_shape,
        )
        logger.success(f"Successfully wrote Xarray Zarr dataset to {output_directory} for {self.file_path}")

    def write_ome_zarr(
        self,
        output_directory: PathLike,
        image_types: ImageType | str | list[ImageType] | list[str] | list[ImageType | str] | None = None,
    ) -> None:
        """Write the generated image DataArrays to an OME-Zarr store."""
        from mbt.io.zarr import write_ome_zarr

        resolved_types = resolve_image_types(image_types)
        if not resolved_types:
            logger.warning(
                f"No image types resolved for OME-Zarr writing for {self.file_path}. Nothing will be written."
            )
            return

        image_data_dict = self.get_selected_image_data(resolved_types)
        if not image_data_dict:
            logger.warning(
                f"Image data for OME-Zarr is empty after filtering for {self.file_path}. Nothing will be written."
            )
            return

        ome_zarr_image_type_strings = list(image_data_dict.keys())  # Get string keys from the dict

        write_ome_zarr(
            image_data=image_data_dict,
            output_directory=output_directory,
            run_fmt_name=self.formatted_run_name,
            fov_fmt_name=self.formatted_fov_name,
            ny=self.n_y_pixels,
            nx=self.n_x_pixels,
            n_masses=len(image_data_dict),
            fov_microns=self.fov_size_microns,
            image_type=ome_zarr_image_type_strings,
        )
        logger.success(f"Successfully wrote OME-Zarr to {output_directory} for {self.file_path}")

    def write_tifffile(
        self,
        output_directory: PathLike,
        image_types: ImageType | str | list[ImageType] | list[str] | list[ImageType | str] | None = None,
    ):
        """Write the generated image DataArrays to TIFF files (one per channel per image type)."""
        resolved_types = resolve_image_types(image_types)
        if not resolved_types:
            logger.warning(f"No image types resolved for TIFF writing for {self.file_path}. Nothing will be written.")
            return

        selected_images_dict = self.get_selected_image_data(resolved_types)
        if not selected_images_dict:
            logger.warning(f"No images selected for TIFF writing for {self.file_path}. Nothing will be written.")
            return

        for image_data_array in selected_images_dict.values():
            write_dir_tiffs(image_data_array, output_directory=Path(output_directory))
        logger.success(
            f"Successfully wrote TIFF files to {output_directory} for {self.file_path} (types: {list(selected_images_dict.keys())})"
        )

    def to_spatialdata(self):
        """Convert the MibiFile Images to a SpatialData object."""
        raise NotImplementedError("SpatialData export is not yet implemented.")
