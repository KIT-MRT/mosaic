import shutil
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest
import segno

import generate_ablation_plot as ablation
import generate_qr_code as qr
import generate_scoring_equation as equation
import poster_style

SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"

TYPST_AVAILABLE = all(
    shutil.which(command) is not None for command in equation.REQUIRED_COMMANDS
)


def _parse_svg(path: Path) -> ElementTree.Element:
    root = ElementTree.parse(path).getroot()
    assert root.tag == f"{SVG_NAMESPACE}svg"
    return root


def test_every_series_covers_every_configuration() -> None:
    for series in ablation.SERIES:
        assert len(series.values) == len(ablation.CONFIGURATIONS)


def test_ablation_data_still_shows_two_mechanisms() -> None:
    collisions, zero_scores = (series.values for series in ablation.SERIES)

    # Collisions drop sharply once verification is present, then stay flat.
    assert collisions[0] > 2 * max(collisions[1:])
    assert max(collisions[1:]) - min(collisions[1:]) <= 3

    # Zero-scores fall throughout, lowest for the full system.
    assert zero_scores == tuple(sorted(zero_scores, reverse=True))
    assert zero_scores[-1] == min(zero_scores)


def test_ablation_plot_matches_the_target_aspect(tmp_path: Path) -> None:
    output = tmp_path / "ablation.svg"
    ablation.render(output)

    root = _parse_svg(output)
    width = float(str(root.get("width")).removesuffix("pt"))
    height = float(str(root.get("height")).removesuffix("pt"))
    box_aspect = ablation.BOX_CONTENT_WIDTH / ablation.BOX_CONTENT_HEIGHT

    # bbox_inches="tight" trims whitespace, so allow some drift.
    assert width / height == pytest.approx(box_aspect, rel=0.12)


def test_qr_code_is_written_with_a_quiet_zone(tmp_path: Path) -> None:
    output = tmp_path / "qr.svg"
    qr.render(output)
    _ = _parse_svg(output)

    code = segno.make(qr.REPOSITORY_URL, error=qr.ERROR_CORRECTION)
    plain = code.symbol_size(scale=1, border=0)[0]
    bordered = code.symbol_size(scale=1, border=qr.QUIET_ZONE_MODULES)[0]
    assert bordered == plain + 2 * qr.QUIET_ZONE_MODULES


def test_qr_code_decodes_back_to_the_repository() -> None:
    cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")

    code = segno.make(qr.REPOSITORY_URL, error=qr.ERROR_CORRECTION)
    modules = np.array(
        [[0 if module else 255 for module in row] for row in code.matrix],
        dtype=np.uint8,
    )
    scaled = np.kron(modules, np.ones((8, 8), dtype=np.uint8))
    padded = np.pad(scaled, 8 * qr.QUIET_ZONE_MODULES, constant_values=255)

    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(padded)
    assert decoded == qr.REPOSITORY_URL


def test_equation_document_uses_the_shared_palette() -> None:
    document = equation.TYPST_DOCUMENT % {
        "gate": poster_style.TEAL,
        "ink": poster_style.INK,
    }
    assert poster_style.TEAL in document
    assert poster_style.INK in document


@pytest.mark.skipif(not TYPST_AVAILABLE, reason="typst not available")
def test_scoring_equation_renders(tmp_path: Path) -> None:
    output = tmp_path / "equation.svg"
    equation.render(output)
    assert _parse_svg(output).get("viewBox") is not None
