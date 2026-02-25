import click

from mosaic.cli.analyze import analyze
from mosaic.cli.plot import plot
from mosaic.cli.results import results
from mosaic.cli.simulate import simulate


@click.group()
def cli() -> None:
    """Mosaic: Arbitration Graphs for Composable Motion Planning with Safety Bounds."""


cli.add_command(analyze)
cli.add_command(plot)
cli.add_command(results)
cli.add_command(simulate)
