"""Export the poster PDF and its preview thumbnail from poster.svg.

Requires a Chromium build (`chromium`, `chromium-browser` or `google-chrome`).

Chromium is used rather than Inkscape because Inkscape rasterises an SVG pulled
in through an `<image>` element (at the linked file's intrinsic pixel size) on
its way to PDF, which is what pixelated the arbitration graph and the ablation
plot. Chromium keeps those references vector.
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final

from PIL import Image

import poster_style

CHROMIUM_COMMANDS: Final = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
)

POSTER_DIR: Final = poster_style.SCRIPTS_DIR.parent / "poster"
POSTER_SVG: Final = POSTER_DIR / "poster.svg"
POSTER_PDF: Final = POSTER_DIR / "poster.pdf"
# Rasterised for the invitation slide, which links it by this path.
PREVIEW_PNG: Final = poster_style.FIGURES_DIR / "poster_preview.png"
PREVIEW_WIDTH_PX: Final = 1325

# ISO A0, matching poster.svg's own page dimensions.
PAGE_WIDTH_MM: Final = 841
PAGE_HEIGHT_MM: Final = 1189

_XML_DECLARATION: Final = re.compile(r"^\s*<\?xml[^>]*\?>")
_RELATIVE_HREF: Final = re.compile(
    r'((?:xlink:)?href=")(?!#|/|data:|https?:|file:)([^"]+)(")'
)


def _find_chromium() -> str:
    for command in CHROMIUM_COMMANDS:
        executable = shutil.which(command)
        if executable is not None:
            return executable
    raise RuntimeError(
        f"need one of {', '.join(CHROMIUM_COMMANDS)}; see https://www.chromium.org"
    )


def _wrap_in_html(source_svg: Path, page_style: str) -> str:
    """Inline the poster SVG into an HTML page sized by ``page_style``.

    Asset links are rewritten to absolute paths so the page renders correctly
    from the temporary directory it is handed to Chromium in.
    """
    markup = _XML_DECLARATION.sub("", source_svg.read_text(encoding="utf-8"), count=1)
    base_dir = source_svg.resolve().parent
    markup = _RELATIVE_HREF.sub(
        lambda match: f"{match.group(1)}{base_dir / match.group(2)}{match.group(3)}",
        markup,
    )
    return f'<!doctype html><meta charset="utf-8"><style>{page_style}</style>{markup}'


def _run_chromium(chromium: str, page_html: str, *arguments: str) -> None:
    with tempfile.TemporaryDirectory() as work_dir:
        page = Path(work_dir) / "poster.html"
        page.write_text(page_html, encoding="utf-8")
        _ = subprocess.run(
            [
                chromium,
                "--headless",
                # Trusted local input; --no-sandbox keeps this working in the
                # unprivileged containers CI runs in.
                "--no-sandbox",
                "--disable-gpu",
                "--force-color-profile=srgb",
                *arguments,
                str(page),
            ],
            check=True,
            capture_output=True,
        )


def build_pdf(chromium: str, source_svg: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    page_style = (
        f"@page{{size:{PAGE_WIDTH_MM}mm {PAGE_HEIGHT_MM}mm;margin:0}}"
        "html,body{margin:0;padding:0}"
        f"svg{{display:block;width:{PAGE_WIDTH_MM}mm;height:{PAGE_HEIGHT_MM}mm}}"
    )
    _run_chromium(
        chromium,
        _wrap_in_html(source_svg, page_style),
        "--no-pdf-header-footer",
        f"--print-to-pdf={target}",
    )


def build_preview(chromium: str, source_svg: Path, target: Path, width_px: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    height_px = round(width_px * PAGE_HEIGHT_MM / PAGE_WIDTH_MM)
    page_style = (
        "html,body{margin:0;padding:0;background:#fff}"
        f"svg{{display:block;width:{width_px}px;height:{height_px}px}}"
    )
    _run_chromium(
        chromium,
        _wrap_in_html(source_svg, page_style),
        "--hide-scrollbars",
        f"--window-size={width_px},{height_px}",
        "--default-background-color=FFFFFFFF",
        f"--screenshot={target}",
    )
    # Guarantee a plain RGB PNG for the invitation slide, whatever Chromium wrote.
    with Image.open(target) as raster:
        raster.convert("RGB").save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg",
        type=Path,
        default=POSTER_SVG,
        help="source poster SVG (default: %(default)s)",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=POSTER_PDF,
        help="where to write the PDF (default: %(default)s)",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=PREVIEW_PNG,
        help="where to write the preview PNG (default: %(default)s)",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=PREVIEW_WIDTH_PX,
        help="preview width in pixels (default: %(default)s)",
    )
    arguments = parser.parse_args()

    chromium = _find_chromium()

    build_pdf(chromium, arguments.svg, arguments.pdf)
    print(f"wrote {arguments.pdf}")
    build_preview(chromium, arguments.svg, arguments.preview, arguments.preview_width)
    print(f"wrote {arguments.preview}")


if __name__ == "__main__":
    main()
