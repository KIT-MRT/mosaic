"""QR code linking to the repository."""

import argparse
from pathlib import Path
from typing import Final

import segno

import poster_style

REPOSITORY_URL: Final = "https://github.com/KIT-MRT/mosaic"

ERROR_CORRECTION: Final = "m"
MODULE_SCALE: Final = 8
# Without a quiet zone the code sits flush against its badge and scanners struggle.
QUIET_ZONE_MODULES: Final = 3


def render(output_path: Path, url: str = REPOSITORY_URL) -> None:
    code = segno.make(url, error=ERROR_CORRECTION)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    code.save(
        str(output_path),
        kind="svg",
        scale=MODULE_SCALE,
        border=QUIET_ZONE_MODULES,
        dark=poster_style.INK,
        light=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=REPOSITORY_URL,
        help="URL to encode (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=poster_style.FIGURES_DIR / "qr-mosaic.svg",
        help="where to write the SVG (default: %(default)s)",
    )
    arguments = parser.parse_args()
    render(arguments.output, arguments.url)
    print(f"wrote {arguments.output} encoding {arguments.url}")


if __name__ == "__main__":
    main()
