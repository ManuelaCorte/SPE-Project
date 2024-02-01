from typing import Literal

from src.utils import Float, Matrix


def seasonal_differencing(
    ts: Matrix[Literal["N"], Float], m: int
) -> Matrix[Literal["N"], Float]:
    """
    Seasonal differencing a time series. Works similarly to normal differencing but the values
    to subtract are taken m values apert.

    Params:
        s: time series
        m: seasonal period

    Returns:
        differenced time series
    """
    if m < 0:
        raise ValueError("period must be greater than or equal to 0")

    if len(ts.shape) > 1:
        raise ValueError("the time series must be a 1D array")

    if m == 0:
        return ts

    return seasonal_differencing(ts[m:] - ts[:-m], m)


def remove_seasonal_differencing(
    ts: Matrix[Literal["N"], Float], m: int
) -> Matrix[Literal["N"], Float]:
    """
    Remove seasonal differencing from a time series.

    Params:
        s: time series
        m: seasonal period

    Returns:
        undifferenced time series
    """
    if m < 0:
        raise ValueError("period must be greater than or equal to 0")

    if len(ts.shape) > 1:
        raise ValueError("the time series must be a 1D array")

    if m == 0:
        return ts

    return remove_seasonal_differencing(ts[m:] + ts[:-m], m)


def differencing(
    ts: Matrix[Literal["N"], Float], d: int
) -> Matrix[Literal["N"], Float]:
    """
    Differencing a time series. If d=1 then this is equivalent to
    computing y_t' = y_t - y_{t-1}. Higher orders of differencing
    are computed recursively on the lower order differencing, for example
    for d=2 we have y_t'' = y_t' - y_{t-1}' = y_t - 2y_{t-1} + y_{t-2}.

    Params:
        s: time series
        d: order of differencing

    Returns:
        differenced time series
    """
    if d < 0:
        raise ValueError("d must be greater than or equal to 0")

    if len(ts.shape) > 1:
        raise ValueError("the time series must be a 1D array")

    if d == 0:
        return ts

    return differencing(ts[1:] - ts[:-1], d - 1)


def remove_differencing(
    ts: Matrix[Literal["N"], Float], d: int
) -> Matrix[Literal["N"], Float]:
    """
    Remove differencing from a time series.

    Params:
        s: time series
        d: order of differencing

    Returns:
        undifferenced time series
    """
    if d < 0:
        raise ValueError("d must be greater than or equal to 0")

    if len(ts.shape) > 1:
        raise ValueError("the time series must be a 1D array")

    if d == 0:
        return ts

    return remove_differencing(ts[1:] + ts[:-1], d - 1)
