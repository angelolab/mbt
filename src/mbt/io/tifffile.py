from os import PathLike

import dask.array as da
import imageio.v3 as iio
import xarray as xr
from dask import compute, delayed  # type: ignore
from upath import UPath


@delayed
def _save_channel(slice_data: xr.DataArray, file_path: UPath, compression: str = "zlib"):
    match slice_data.data:
        case da.Array():
            image = slice_data.compute()
        case _:
            image = slice_data

    iio.imwrite(file_path, image, plugin="tifffile", photometric="minisblack", compression=compression)


def write_dir_tiffs(
    image: xr.DataArray,
    output_directory: PathLike,
    compression: str = "zlib",
):
    """Write each 'c' slice of an xarray.DataArray to a separate TIFF file.

    Parameters
    ----------
    image : xr.DataArray
        The DataArray to save. Must have 'c', 'y', and 'x' dimensions.
    output_directory : PathLike
        The directory where the TIFF files will be saved.
    filename_prefix : str, optional
        A prefix for the output filenames. If None, uses data_array.name.
        Defaults to None.

    """
    run_name = image.attrs["run_name"]
    fov_name = image.attrs["fov_name"]
    image_type = image.attrs["image_type"]
    out_dir = UPath(output_directory) / run_name / fov_name / image_type
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_names = image["c"].values.tolist()

    delayed_saves = []

    for channel_name in channel_names:
        channel_slice = image.sel(c=channel_name)
        safe_channel_name = str(channel_name).replace(" ", "_").replace("/", "-")

        file_path = out_dir / f"{safe_channel_name}.tiff"

        # Create delayed task
        task = _save_channel(channel_slice, file_path, compression)
        delayed_saves.append(task)
    compute(*delayed_saves)
