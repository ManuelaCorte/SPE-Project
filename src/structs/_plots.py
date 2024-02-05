from dataclasses import dataclass


@dataclass(frozen=True)
class PlotOptions:
    filename: str
    title: str
    x_axis: str
    y_axis: str
    labels: list[str]
    save: bool
