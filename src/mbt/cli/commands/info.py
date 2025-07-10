from pathlib import Path

import polars as pl
import typer

# Rich imports grouped
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from mbt.cli.utils import error_text
from mbt.core.mibifile import MibiFile
from mbt.core.models import MassCalibrationModel

app = typer.Typer()


def polars_to_rich_table(df: pl.DataFrame, title="Panel"):
    """Converts a Polars DataFrame to a Rich Table."""
    table = Table(title=title, show_header=True, header_style="bold blue")
    for column in df.columns:
        table.add_column(str(column))
    for row in df.iter_rows():
        table.add_row(*[str(value) for value in row])
    return table


# --- Helper Functions for Panel Creation ---


def _create_dimension_panel_group(f: MibiFile) -> Group:
    """Creates the vertically grouped dimension panels."""
    panel: pl.DataFrame = f.panel
    shape_y: int = f.n_y_pixels
    shape_x: int = f.n_x_pixels
    fov_size_microns: float = f.fov_size_microns

    n_unique_masses = panel["mass_start"].n_unique()
    n_unique_targets = panel["target"].n_unique()

    # Dimension Info Panel
    dim_str = (
        f"Targets     : {n_unique_targets}\n"
        f"Masses      : {n_unique_masses}\n"
        f"Y           : {shape_y}px\n"
        f"X           : {shape_x}px"
    )
    dim_panel = Panel(dim_str, title="Image Dimensions", border_style="magenta", expand=False)

    # Physical Dimensions Panel
    pixel_scale_x = fov_size_microns / shape_x if shape_x > 0 else 0.0
    pixel_scale_y = fov_size_microns / shape_y if shape_y > 0 else 0.0
    physical_str = (
        f"FOV Size : {fov_size_microns:.2f} µm x {fov_size_microns:.2f} µm\n"
        f"Scale X  : {pixel_scale_x:.3f} µm/px\n"
        f"Scale Y  : {pixel_scale_y:.3f} µm/px"
    )
    physical_panel = Panel(physical_str, title="Physical Dimensions", border_style="blue", expand=False)

    return Group(dim_panel, physical_panel)


def _create_details_panel(f: MibiFile) -> Panel:
    """Creates the File & FOV Details panel."""
    run_name: str = f.run_name
    run_uuid: str = f.run_uuid
    fov_id: str = f.fov_id
    fov_name: str = f.fov_name
    acq_status: str = f.acquisition_status

    details_str = (
        f"Run Name : {run_name}\n"
        f"Run UUID : {run_uuid}\n"
        f"FOV ID   : {fov_id}\n"
        f"FOV Name : {fov_name}\n"
        f"Acq Stat : {acq_status}"
    )
    return Panel(details_str, title="File & FOV Details", border_style="green", expand=False)


def _create_mass_cal_panel(f: MibiFile) -> Panel:
    """Creates the Mass Calibration details panel."""
    mass_calibration: MassCalibrationModel | None = f.mass_calibration
    time_resolution: float = f.time_resolution

    mc_offset_str = "N/A"
    mc_gain_str = "N/A"
    if mass_calibration:
        mc_offset_str = f"{mass_calibration.mass_offset:.4f}"
        mc_gain_str = f"{mass_calibration.mass_gain:.4f}"

    mass_cal_str = f"Offset   : {mc_offset_str}\nGain     : {mc_gain_str}\nTime Res.: {time_resolution:.4e} s"
    return Panel(mass_cal_str, title="Mass Calibration", border_style="yellow", expand=False)


def _create_instr_acq_panel(f: MibiFile) -> Panel:
    """Creates the Instrument & Acquisition panel."""
    instr_id: str = f.instrument_identifier
    ctrl_ver: str = f.instrument_control_version
    tof_ver: str = f.tof_app_version
    preset: str = f.imaging_preset_name
    aperture: str = f.imaging_aperture
    dwell: float = f.dwell_time_millis
    scans: int = f.scan_count

    instr_acq_str = (
        f"Instrument  : {instr_id}\n"
        f"Control Ver.: {ctrl_ver}\n"
        f"TOF Ver.    : {tof_ver}\n"
        f"Preset      : {preset}\n"
        f"Aperture    : {aperture}\n"
        f"Dwell Time  : {dwell:.3f} ms\n"
        f"Scan Count  : {scans}"
    )
    return Panel(instr_acq_str, title="Instrument & Acquisition", border_style="cyan", expand=False)


def _create_panel_details_renderable(f: MibiFile) -> Align:
    """Creates the centered Panel Details table or an empty message."""
    panel = f.panel
    if not panel.is_empty():
        panel_table = polars_to_rich_table(panel, title="Panel Details")
        return Align.center(panel_table)
    else:
        return Align.center("[yellow]Panel data not found or empty.[/yellow]")


# --- Main Command ---


@app.command(no_args_is_help=True)
def info(path: Path = typer.Argument(..., help="Path to the Mibi file")):
    """Display information about a MIBI file."""
    console = Console()  # Create a Console instance

    try:
        with MibiFile(path) as f:
            # Create panel components using helpers
            dimension_group = _create_dimension_panel_group(f)
            details_panel = _create_details_panel(f)
            instr_acq_panel = _create_instr_acq_panel(f)
            panel_details_renderable = _create_panel_details_renderable(f)
            mass_cal_panel = _create_mass_cal_panel(f)

            # --- Render Info Panels Side-by-Side and Centered ---
            details_mass_cal_group = Group(details_panel, mass_cal_panel)
            info_columns = Columns([dimension_group, details_mass_cal_group, instr_acq_panel])
            centered_columns = Align.center(info_columns)
            console.print(centered_columns)

            # --- Render Panel Table (Centered) ---
            console.print(panel_details_renderable)

    except FileNotFoundError as e:
        console.print(error_text(f"Error: {e}"))
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(error_text(f"An unexpected error occurred: {e}"))
        raise typer.Exit(code=1) from e
