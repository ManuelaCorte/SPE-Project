from dataclasses import dataclass
from pprint import pformat
from typing import Any


@dataclass
class SignificanceResult:
    name: str
    statistic: float
    pvalue: float


@dataclass
class StationarityTest:
    result: SignificanceResult
    lags: int
    crit: dict[str, float]

    def __init__(self, name: str, value: Any, pvalue: Any, lags: Any, crit: Any):
        self.result = SignificanceResult(name, value, pvalue)
        self.lags = lags
        self.crit = crit

    def __repr__(self) -> str:
        return (
            f"{self.result.name} test statistic: {self.result.statistic}\n"
            f"p-value: {self.result.pvalue}\n"
            f"Number of lags: {self.lags}\n"
            f"Critical values: {pformat(self.crit, compact=True, sort_dicts=False)}"
        )
