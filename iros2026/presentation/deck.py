"""IROS 2026 lightning talk for Mosaic."""

from inkflow import Deck, Slide, animations, transitions


def main() -> Deck:
    morph = transitions.Morph(1.0)

    return Deck(
        title="Mosaic",
        transition=transitions.Crossfade(),
        slides=[
            Slide("theory", notes="notes/theory.md"),
            Slide("practice", notes="notes/practice.md", transition=morph),
            Slide(
                "four-jobs",
                notes="notes/four-jobs.md",
                transition=morph,
                animations=[
                    animations.FadeIn(el)
                    for el in ["g-generate", "g-select", "g-verify", "g-fallback"]
                ],
            ),
            Slide(
                "arbitration-graph",
                notes="notes/arbitration-graph.md",
                transition=morph,
            ),
            Slide(
                "composition",
                notes="notes/composition.md",
                animations=[
                    animations.FadeIn(el)
                    for el in ["pdm-t2", "col-fd", "fd-t2", "col-mosaic", "mosaic-t2"]
                ],
            ),
            Slide("results", notes="notes/results.md"),
            Slide("invitation", notes="notes/invitation.md"),
        ],
    )
