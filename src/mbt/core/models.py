from datetime import date
from enum import StrEnum

import patito as pt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

C = "c"
X = "x"
Y = "y"


class MibiDataModel(pt.Model):
    """Patito model for validating the core MIBI pixel data DataFrame."""

    pixel_x: int = pt.Field(dtype=pl.UInt32, ge=0)
    pixel_y: int = pt.Field(dtype=pl.UInt32, ge=0)
    trigger_index: int = pt.Field(dtype=pl.UInt16, ge=0)
    pulse_time: int = pt.Field(dtype=pl.UInt16, ge=0)
    pulse_width: int = pt.Field(dtype=pl.UInt8, ge=0)
    pulse_intensity: int = pt.Field(dtype=pl.UInt16, gt=0)

    model_config = ConfigDict(strict=True, extra="allow")


class MibiFilePanelModel(pt.Model):
    """Patito model for validating the panel DataFrame extracted from the MIBI file."""

    target: str = pt.Field(dtype=pl.Utf8)
    mass: float = pt.Field(dtype=pl.Float64, gt=0.0)
    element: str = pt.Field(dtype=pl.Utf8)
    clone: str = pt.Field(dtype=pl.Utf8)
    mass_start: float = pt.Field(dtype=pl.Float64, gt=0.0)
    mass_stop: float = pt.Field(dtype=pl.Float64, gt=0.0)
    id: int = pt.Field(dtype=pl.Int32)
    external_id: str = pt.Field(dtype=pl.Utf8)
    concentration: float = pt.Field(dtype=pl.Float64, gt=0.0)
    lot: str = pt.Field(dtype=pl.Utf8)
    manufacture_date: date = pt.Field(dtype=pl.Date)
    conjugate_id: int = pt.Field(dtype=pl.Int32)

    model_config = ConfigDict(strict=True, extra="allow")


class UserPanelModel(pt.Model):
    """Patito model for validating an optional user-supplied panel."""

    channel_name: str = pt.Field(dtype=pl.Utf8)
    mass: float = pt.Field(dtype=pl.Float64, gt=0.0)
    mass_start: float = pt.Field(dtype=pl.Float64, gt=0.0)
    mass_stop: float = pt.Field(dtype=pl.Float64, gt=0.0)

    model_config = ConfigDict(strict=True, extra="allow")


class MassCalibrationModel(BaseModel):
    """Pydantic model for validating the mass calibration data."""

    masses: list[float] = Field(..., description="List of masses for the calibration")
    bins: list[int] = Field(..., description="List of bins for the calibration")
    mass_offset: float = Field(..., alias="massOffset", description="Mass offset for the calibration")
    mass_gain: float = Field(..., alias="massGain", description="Mass gain for the calibration")
    time_resolution: float = Field(..., alias="timeResolution", description="Time resolution for the calibration")
    time_resolution_nano_s: float = Field(
        ..., alias="timeResolutionNanoS", description="Time resolution in nano seconds for the calibration"
    )

    model_config = ConfigDict(strict=True, populate_by_name=True, extra="allow")


class ImageType(StrEnum):
    """Enum for the different image types."""

    ALL = "all"
    COUNTS = "counts"
    INTENSITY = "intensity"
    INTENSITY_WIDTH = "intensity_width"


class Format(StrEnum):
    """Enum for the different formats."""

    TIFFS = "tiffs"
    OME_ZARR = "ome-zarr"
    XRDATSET = "xrdataset"


# Helper constant for actual, non-aggregate image types
_CONCRETE_IMAGE_TYPES = tuple(it for it in ImageType if it != ImageType.ALL)
