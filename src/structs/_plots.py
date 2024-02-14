from dataclasses import dataclass


@dataclass(frozen=True)
class PlotOptions:
    """Utility class to gather together the options for a plot."""

    filename: str
    title: str
    x_axis: str
    y_axis: str
    labels: list[str]
    save: bool
