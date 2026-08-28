"""Scoring function typeset with Typst. Requires `typst`."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final

import poster_style

REQUIRED_COMMANDS: Final = ("typst",)

TYPST_DOCUMENT: Final = """\
#let gate = rgb("%(gate)s")
#let ink = rgb("%(ink)s")

#set page(width: auto, height: auto, margin: 0pt, fill: none)
#set text(fill: ink)

$ S_"total" thick = thick
  text(fill: gate, S_"coll" dot.op S_"driv" dot.op S_"dir")
  thick dot.op thick G_"prog" dot.op S_"perf" $
"""


def render(output_path: Path) -> None:
    missing = [name for name in REQUIRED_COMMANDS if shutil.which(name) is None]
    if missing:
        message = f"missing {', '.join(missing)}; see https://typst.app"
        raise RuntimeError(message)

    document = TYPST_DOCUMENT % {"gate": poster_style.TEAL, "ink": poster_style.INK}

    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        source = workspace / "equation.typ"
        _ = source.write_text(document, encoding="utf-8")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _ = subprocess.run(
            ["typst", "compile", "--format", "svg", str(source), str(output_path)],
            check=True,
            capture_output=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=poster_style.FIGURES_DIR / "scoring-equation.svg",
        help="where to write the SVG (default: %(default)s)",
    )
    arguments = parser.parse_args()
    render(arguments.output)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
