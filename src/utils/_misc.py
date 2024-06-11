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


def train_test_split(
    x: npt.NDArray[Any], train_size: float = 0.8
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Splits an array into training and testing sets. Since we are working with time series data,
    the test set is equivalent to the most recent data points.

    Parameters:
        x: The array to split
        train_size: The proportion of the data to use for training

    Returns:
        The training and testing sets"""
    n = len(x)
    n_train = int(n * train_size)
    return x[:n_train], x[n_train:]
