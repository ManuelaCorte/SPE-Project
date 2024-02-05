from typing import Any

import numpy as np
import numpy.typing as npt


def remove_nans(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    mask = ~np.isnan(x)
    y = x[mask].copy()
    return y


def normalize(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return (x - x.mean()) / x.std()


def unnormalize(x: npt.NDArray[Any], mean: float, std: float) -> npt.NDArray[Any]:
    return (x * std) + mean
