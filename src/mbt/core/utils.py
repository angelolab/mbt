import re
from collections.abc import Sequence

import natsort as ns
import polars as pl

from mbt.core.models import _CONCRETE_IMAGE_TYPES, ImageType


def _mass_to_tof(mass_expr: pl.Expr, mass_offset: float, mass_gain: float, time_resolution: float) -> pl.Expr:
    """Converts a mass Polars expression to a time-of-flight expression.

    Parameters
    ----------
    mass_expr
        Polars expression representing m/z values.
    mass_offset
        Mass offset for parabolic transformation.
    mass_gain
        Mass gain for parabolic transformation.
    time_resolution
        Time resolution for scaling.

    Returns
    -------
        Polars expression representing time-of-flight values (UInt16).
    """
    tof_expr = (mass_gain * mass_expr.sqrt() + mass_offset) / time_resolution
    return tof_expr


def _tof_to_mass(tof_expr: pl.Expr, mass_offset: float, mass_gain: float, time_resolution: float) -> pl.Expr:
    """Converts a time-of-flight Polars expression to a mass expression.

    Parameters
    ----------
    tof_expr
        Polars expression representing time-of-flight values.
    mass_offset
        Mass offset for parabolic transformation.
    mass_gain
        Mass gain for parabolic transformation.
    time_resolution
        Time resolution for scaling.

    Returns
    -------
        Polars expression representing m/z values.
    """
    return (((time_resolution * tof_expr) - mass_offset) / mass_gain).pow(2)


def _set_tof_ranges(panel: pl.DataFrame, mass_offset: float, mass_gain: float, time_resolution: float) -> pl.DataFrame:
    """Calulates the lower and upper time-of-flight (TOF) range columns.

    Calculates and adds lower/upper Time-of-Flight (TOF) range columns
    to the panel DataFrame based on mass_start/mass_stop and calibration parameters.

    Parameters
    ----------
    panel
        Polars DataFrame with 'mass_start' and 'mass_stop' columns.
    mass_offset
        Mass offset for parabolic transformation.
    mass_gain
        Mass gain for parabolic transformation.
    time_resolution
        The time resolution used in TOF calculations.

    Returns
    -------
        A new Polars DataFrame with 'lower_tof_range' (UInt16) and
        'upper_tof_range' (UInt16) added.

    Raises
    ------
        ValueError: If the panel DataFrame lacks 'mass_start' or 'mass_stop'.
    """
    required_cols = ["mass_start", "mass_stop"]
    if not all(col in panel.columns for col in required_cols):
        missing = [col for col in required_cols if col not in panel.columns]
        raise ValueError(f"Panel DataFrame must contain columns: {missing}")

    updated_panel = panel.with_columns(
        lower_tof_range=_mass_to_tof(
            pl.col("mass_start"),
            mass_offset=mass_offset,
            mass_gain=mass_gain,
            time_resolution=time_resolution,
        ),
        upper_tof_range=_mass_to_tof(
            pl.col("mass_stop"),
            mass_offset=mass_offset,
            mass_gain=mass_gain,
            time_resolution=time_resolution,
        ),
    )
    return updated_panel


def format_image_name(fov_id: str, fov_name: str) -> str:
    """Formats an image name string from FOV ID and FOV name.

    Transforms FOV IDs like 'FOV1', 'fov23' into 'fov-1', 'fov-23'.
    Combines the formatted FOV ID with the FOV name.

    Parameters
    ----------
    fov_id
        The field of view identifier (e.g., 'FOV1').
    fov_name
        The field of view name (e.g., 'R13C3').

    Returns
    -------
        A formatted string (e.g., 'fov-1-R13C3').
    """
    match = re.match(r"fov(\d+)", fov_id, re.IGNORECASE)
    formatted_fov_id = f"fov-{match.group(1)}" if match else fov_id.lower()
    return f"{formatted_fov_id}-{fov_name}"


def format_run_name(run_name: str) -> str:
    """Formats a run name string by removing any '_part<number>' suffix.

    Parameters
    ----------
    run_name
        The original run name (e.g., '2022-07-04_NBL_TMA2_part1').

    Returns
    -------
        The formatted run name (e.g., '2022-07-04_NBL_TMA2').
    """
    formatted_name = re.sub(r"_part\d+$", "", run_name)
    return formatted_name


def resolve_image_types(
    selection: ImageType | str | list[ImageType] | list[str] | list[ImageType | str] | None,
) -> list[ImageType]:
    """
    Resolves the input image type selection into a definitive list of concrete ImageType enums.

    Uses natsorted for the final sort.

    Parameters
    ----------
    selection
        The image type selection to resolve.

    Returns
    -------
        A list of ImageType enums.
    """
    if selection is None:
        # natsorted works on the enum values directly if they are strings or comparable
        return ns.natsorted(list(_CONCRETE_IMAGE_TYPES), key=lambda x: x.value)

    final_types: set[ImageType] = set()
    items_to_process: Sequence[ImageType | str]

    if isinstance(selection, str | ImageType):
        items_to_process = [selection]
    elif isinstance(selection, list):
        items_to_process = selection
    else:
        raise TypeError(
            f"Invalid selection type for image_types: {type(selection)}. "
            "Expected str, ImageType, list[str | ImageType], or None."
        )

    if not items_to_process and isinstance(selection, list):  # Handles empty list input
        return []

    for item in items_to_process:
        if isinstance(item, str):
            try:
                item_standardized = item.lower()
                if item_standardized == ImageType.ALL.value:
                    final_types.update(_CONCRETE_IMAGE_TYPES)
                else:
                    # Validate and convert string to ImageType enum
                    final_types.add(ImageType(item_standardized))
            except ValueError as e:
                # Re-raise with a more informative message if it's due to invalid enum value
                valid_enums = [e.value for e in ImageType]
                raise ValueError(
                    f"Invalid image type string: '{item}'. Must be one of {valid_enums} (case-insensitive)."
                ) from e
        elif isinstance(item, ImageType):
            if item == ImageType.ALL:
                final_types.update(_CONCRETE_IMAGE_TYPES)
            else:
                final_types.add(item)
        else:
            raise TypeError(
                f"Invalid item type in image_types selection list: {type(item)}. Elements must be str or ImageType."
            )

    return ns.natsorted(list(final_types), key=lambda x: x.value)
