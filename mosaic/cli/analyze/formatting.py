"""Shared formatting utilities for the analyze command output."""

from typing import List

import click

WIDTH = 60


def title_box(label: str) -> None:
    """Print a prominent title box for the experiment name."""
    inner = WIDTH - 2
    click.echo(f"\n╭{'─' * inner}╮")
    click.echo(f"│  {label:<{inner - 2}}│")
    click.echo(f"╰{'─' * inner}╯")


def section(title: str) -> None:
    """Print a section header with extra spacing for visual separation."""
    bar = "─" * (WIDTH - len(title) - 4)
    click.echo(f"\n\n── {click.style(title, bold=True)} {bar}")


def subsection(title: str) -> None:
    """Print a lighter sub-header within a section."""
    click.echo(f"\n  {click.style(title, bold=True)}")


def kv(key: str, value: str, key_width: int = 22) -> None:
    """Print a key-value pair with consistent alignment."""
    click.echo(f"  {key:<{key_width}}{value}")


def truncate(text: str, width: int) -> str:
    """Truncate a string to width, appending '…' if needed."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def fail_marker(count: int) -> str:
    """Return a red marker if count > 0, empty otherwise."""
    if count > 0:
        return click.style(" ◆", fg="red")
    return ""


def colored_diff(value: float, fmt: str = "+.4f", invert: bool = False) -> str:
    """Color a diff value: green for better, red for worse.

    By default positive = better. Set invert=True if positive = worse.
    """
    text = f"{value:{fmt}}"
    if value > 0:
        color = "red" if invert else "green"
    elif value < 0:
        color = "green" if invert else "red"
    else:
        color = None
    if color:
        return click.style(text, fg=color)
    return text


class Table:
    """Bordered table with box-drawing characters.

    Usage:
        t = Table(["Name", "Score"], [30, 6], ["<", ">"])
        t.row(["foo", "0.95"])
        t.row(["bar", "0.87"])
        t.render()

    Renders:
        ┌────────────────────────────────┬────────┐
        │ Name                           │  Score │
        ├────────────────────────────────┼────────┤
        │ foo                            │   0.95 │
        │ bar                            │   0.87 │
        └────────────────────────────────┴────────┘
    """

    def __init__(
        self, columns: List[str], widths: List[int], aligns: List[str]
    ) -> None:
        self.columns = columns
        self.widths = widths
        self.aligns = aligns
        self._rows: List = []

    def row(self, values: List[str], suffix: str = "") -> "Table":
        """Add a data row. Optional suffix is printed outside the right border."""
        self._rows.append(("data", values, suffix))
        return self

    def separator(self) -> "Table":
        """Add a horizontal separator between data rows."""
        self._rows.append(("sep",))
        return self

    def render(self, indent: int = 2) -> None:
        """Print the complete bordered table."""
        pad = " " * indent

        def _border(left: str, mid: str, right: str) -> str:
            cells = [f"{'─' * (w + 2)}" for w in self.widths]
            return pad + left + mid.join(cells) + right

        def _data(values: List[str], suffix: str = "") -> str:
            cells = []
            for val, w, a in zip(values, self.widths, self.aligns):
                cells.append(f" {val:{a}{w}} ")
            return pad + "│" + "│".join(cells) + "│" + suffix

        click.echo(_border("╭", "┬", "╮"))
        click.echo(_data(self.columns))
        click.echo(_border("├", "┼", "┤"))

        for entry in self._rows:
            if entry[0] == "sep":
                click.echo(_border("├", "┼", "┤"))
            else:
                _, values, suffix = entry
                click.echo(_data(values, suffix))

        click.echo(_border("╰", "┴", "╯"))
