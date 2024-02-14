from typing import Any

import numpy as np
import numpy.typing as npt


def remove_nans(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Removes NaNs from an array."""
    mask = ~np.isnan(x)
    y = x[mask].copy()
    return y


def normalize(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Normalizes an array by subtracting the mean and dividing by the standard deviation.
    The result has a mean of 0 and a standard deviation of 1."""
    return (x - x.mean()) / x.std()


def unnormalize(x: npt.NDArray[Any], mean: float, std: float) -> npt.NDArray[Any]:
    """Unnormalizes an array by multiplying by the standard deviation and adding the mean."""
    return (x * std) + mean
