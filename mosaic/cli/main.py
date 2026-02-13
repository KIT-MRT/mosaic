import click

from mosaic.cli.simulate import simulate
from mosaic.cli.results import results


@click.group()
def cli():
    """Mosaic: Arbitration Graphs for Composable Motion Planning with Safety Bounds."""


cli.add_command(simulate)
cli.add_command(results)
