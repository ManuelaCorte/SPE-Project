from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", np.float_, np.int_)


def remove_nans(x: npt.NDArray[T]) -> npt.NDArray[T]:
    """Removes NaNs from an array."""
    mask = ~np.isnan(x)
    y = x[mask].copy()
    return y


def normalize(x: npt.NDArray[T]) -> npt.NDArray[T]:
    """Normalizes an array between 0 and 1."""
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


def unnormalize(x: npt.NDArray[T], x_min: Any, x_max: Any) -> npt.NDArray[T]:
    """Unnormalizes an array."""
    return x * (x_max - x_min) + x_min


def train_test_split(
    x: npt.NDArray[T], train_size: float = 0.8
) -> tuple[npt.NDArray[T], npt.NDArray[T]]:
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
