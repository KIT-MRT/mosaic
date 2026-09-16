"""Export each poster variant's print PDF and preview thumbnail from its SVG.

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
from typing import Final, NamedTuple

from PIL import Image

import poster_style

CHROMIUM_COMMANDS: Final = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
)

POSTER_DIR: Final = poster_style.SCRIPTS_DIR.parent / "poster"
PREVIEW_WIDTH_PX: Final = 1325


class PosterVariant(NamedTuple):
    svg: Path
    pdf: Path
    preview: Path


# Rasterised previews are checked in; the landscape one is also used by the
# invitation slide, which links it by this path.
VARIANTS: Final = (
    PosterVariant(
        svg=POSTER_DIR / "poster_portrait.svg",
        pdf=POSTER_DIR / "poster_portrait.pdf",
        preview=poster_style.FIGURES_DIR / "poster_portrait_preview.png",
    ),
    PosterVariant(
        svg=POSTER_DIR / "poster_landscape.svg",
        pdf=POSTER_DIR / "poster_landscape.pdf",
        preview=poster_style.FIGURES_DIR / "poster_landscape_preview.png",
    ),
)

_XML_DECLARATION: Final = re.compile(r"^\s*<\?xml[^>]*\?>")
_RELATIVE_HREF: Final = re.compile(
    r'((?:xlink:)?href=")(?!#|/|data:|https?:|file:)([^"]+)(")'
)
_SVG_WIDTH_MM: Final = re.compile(r'<svg\b[^>]*\swidth="([\d.]+)mm"')
_SVG_HEIGHT_MM: Final = re.compile(r'<svg\b[^>]*\sheight="([\d.]+)mm"')


def _read_page_size_mm(source_svg: Path) -> tuple[float, float]:
    """Read the page's physical size from its own width/height attributes.

    Every poster variant (portrait, landscape, ...) declares its own page size
    on the root element, so the page size is read from there instead of being
    duplicated as a constant per variant.
    """
    markup = source_svg.read_text(encoding="utf-8")
    width_match = _SVG_WIDTH_MM.search(markup)
    height_match = _SVG_HEIGHT_MM.search(markup)
    if width_match is None or height_match is None:
        raise ValueError(f"{source_svg}: root <svg> has no width/height in mm")
    return float(width_match.group(1)), float(height_match.group(1))


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


def export_variant(chromium: str, variant: PosterVariant, preview_width: int) -> None:
    export_pdf(chromium, variant.svg, variant.pdf)
    print(f"wrote {variant.pdf}")
    export_preview(chromium, variant.svg, variant.preview, preview_width)
    print(f"wrote {variant.preview}")


def export_pdf(chromium: str, source_svg: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    width_mm, height_mm = _read_page_size_mm(source_svg)
    page_style = (
        f"@page{{size:{width_mm}mm {height_mm}mm;margin:0}}"
        "html,body{margin:0;padding:0}"
        f"svg{{display:block;width:{width_mm}mm;height:{height_mm}mm}}"
    )
    _run_chromium(
        chromium,
        _wrap_in_html(source_svg, page_style),
        "--no-pdf-header-footer",
        f"--print-to-pdf={target}",
    )


def export_preview(
    chromium: str, source_svg: Path, target: Path, width_px: int
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    width_mm, height_mm = _read_page_size_mm(source_svg)
    height_px = round(width_px * height_mm / width_mm)
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
        help="export only this SVG, not every variant (needs --pdf/--preview)",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="where to write the PDF (with --svg)",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="where to write the preview PNG (with --svg)",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=PREVIEW_WIDTH_PX,
        help="preview width in pixels (default: %(default)s)",
    )
    arguments = parser.parse_args()

    if arguments.svg is not None:
        if arguments.pdf is None or arguments.preview is None:
            parser.error("--svg requires --pdf and --preview")
        variants = (PosterVariant(arguments.svg, arguments.pdf, arguments.preview),)
    elif arguments.pdf is not None or arguments.preview is not None:
        parser.error("--pdf/--preview require --svg")
    else:
        variants = VARIANTS

    chromium = _find_chromium()
    for variant in variants:
        export_variant(chromium, variant, arguments.preview_width)


if __name__ == "__main__":
    main()
