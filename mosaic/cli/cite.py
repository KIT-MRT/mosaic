import click

CITATION = """\
@misc{large2026mosaic,
  title={Mosaic: An Extensible Framework for Composing Rule-Based and Learned Motion Planners},
  author={Le Large, Nick and Steiner, Marlon and Wang, Lingguang and Poh, Willi and Pauls, Jan-Hendrik and Ta\\c{s}, {\\"{O}}mer {\\c{S}}ahin and Stiller, Christoph},
  year={2026},
  eprint={2604.13853},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2604.13853},
}"""


@click.command()
def cite() -> None:
    """Print the BibTeX citation for Mosaic."""
    click.echo(CITATION)
