"""IROS 2026 lightning talk for Mosaic."""

from inkflow import Deck, Slide, animations, transitions


def main() -> Deck:
    morph = transitions.Morph(1.0)

    return Deck(
        title="Mosaic",
        transition=transitions.Crossfade(),
        slides=[
            Slide("theory"),
            Slide("practice", transition=morph),
            Slide(
                "four-jobs",
                transition=morph,
                animations=[
                    animations.FadeIn(el)
                    for el in ["g-generate", "g-select", "g-verify", "g-fallback"]
                ],
            ),
            Slide(
                "arbitration-graph",
                transition=morph,
            ),
            Slide(
                "composition",
                animations=[
                    animations.FadeIn(el)
                    for el in ["pdm-t2", "col-fd", "fd-t2", "col-mosaic", "mosaic-t2"]
                ],
            ),
            Slide("results"),
            Slide("invitation"),
        ],
    )
