import typer

from .commands import extract_app, info_app

app = typer.Typer()


app.add_typer(info_app, name="info")


app = typer.Typer()


def version_callback(value: bool) -> None:
    """Show the mbt version."""
    if value:
        from mbt import __version__

        typer.echo(f"mbt version: {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        help="Show the mbt version.",
        is_eager=True,
    ),
) -> None:
    """A CLI for working with MIBI files."""
    if ctx.invoked_subcommand is None and not version:
        typer.echo(ctx.get_help())
        raise typer.Exit()


app.add_typer(info_app)
app.add_typer(extract_app)
