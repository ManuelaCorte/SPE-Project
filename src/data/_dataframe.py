from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from src.structs import Country, Indicator
from src.utils import Float, Matrix

GDP, IR, CPI = Indicator


def get_indicators_columns(columns: Any | list[str]) -> list[str]:
    """
    Get the columns containing the economic indicators

    Parameters:
        columns: The columns of the dataframe

    Returns:
        The columns containing the economic indicators
    """
    if isinstance(columns, pd.Index):
        columns = columns.values.tolist()

    indicators: list[str] = []
    for column in columns:
        year: str = column[:4]
        if not year.isnumeric():
            indicators.append(column)

    return indicators


def get_time_periods_colums(columns: Any | list[str]) -> list[str]:
    """
    Get the columns containing the time periods. If those columns represent years
    then they are simply the years while for quarters (months) they are equal to
    YYYYQ1 (YYYYMM).

    Parameters:
        columns: The columns of the dataframe

    Returns:
        The columns containing the time periods
    """
    if isinstance(columns, pd.Index):
        columns = columns.values.tolist()

    time_periods: list[str] = []
    for column in columns:
        year = column[:4]
        if year.isnumeric():
            time_periods.append(column)

    return time_periods


def convert_to_matrix(
    df: pd.DataFrame,
    indicator: Indicator,
    country: Optional[Country] = None,
    pct: bool = False,
) -> Matrix[Literal["N"], Float]:
    """
    Convert the dataframe to a matrix containing the values of the given indicator

    Parameters:
        df: The dataframe
        indicator: The indicator to be extracted
        country: The country to be extracted
        pct: Whether to return the values as is or as percentages changes from the previous period

    Returns:
        The matrix containing the values of the given indicator
    """
    # Get row corresponding to the given indicator
    dataframe = df[df["Indicator Name"] == indicator.value]

    if country is not None:
        # Get only the row corresponding to the country
        dataframe = dataframe[dataframe["Country Code"] == country.value]

    if pct:
        return dataframe["Value_pct"].to_numpy()
    else:
        return dataframe["Value"].to_numpy()


def convert_to_structured_matrix(
    df: pd.DataFrame,
    indicator: Indicator,
    country: Optional[Country] = None,
    pct: bool = False,
) -> Matrix[Literal["N"], Float]:
    """
    Convert the dataframe to a matrix containing the values of the given indicator. The
    matrix is structured and contains the date associated as well: each row is a tuple
    (value, date).

    Parameters:
        df: The dataframe
        indicator: The indicator
        country: The country to be extracted. If None, all the countries are considered
        pct: Whether to return the values as is or as percentages changes from the previous period

    Returns:
        The matrix containing the values of the given indicator
    """
    # Get row corresponding to the given indicator
    dataframe = df[df["Indicator Name"] == indicator.value]

    if country is not None:
        # Get only the row corresponding to the country
        dataframe = dataframe[dataframe["Country Code"] == country.value]

    data = dataframe[["Value", "Date"]].to_records(index=False)

    if pct:
        return np.array(data, dtype=[("Value_pct", "f8"), ("Date", "O")])
    else:
        return np.array(data, dtype=[("Value", "f8"), ("Date", "O")])


def serialize_country_data(
    df: pd.DataFrame,
    country: Country,
    pct: bool = False,
) -> tuple[dict[Indicator, Matrix[Literal["N"], Float]], Matrix[Literal["N"], np.str_]]:
    """
    Serialize the data of a country taking only the time periods in which all the indicators
    are available and converting them to matrices. Also returns the shared time periods.

    Parameters:
        df: The dataframe
        country: The country
        pct: Whether to return the values as is or as percentages changes from the previous period

    Returns:
        Country's data, divided by indicator and the months common to all the indicators
    """
    country_df = df[df["Country Code"] == country.value]
    indicators_series: dict[Indicator, pd.DataFrame] = {}
    for indicator in Indicator:
        indicators_series[indicator] = country_df[
            country_df["Indicator Name"] == indicator.value
        ]

    # * make sure that all series start from the same date
    first_common_year = max(
        [int(indicators_series[indicator]["Year"].iloc[0]) for indicator in Indicator]  # type: ignore
    )
    for indicator in Indicator:
        indicators_series[indicator] = indicators_series[indicator][
            indicators_series[indicator]["Year"] >= first_common_year
        ]
    first_common_month = max(
        [int(indicators_series[indicator]["Month"].iloc[0]) for indicator in Indicator]  # type: ignore
    )
    for indicator in Indicator:
        first_series_month = int(indicators_series[indicator]["Month"].iloc[0])  # type: ignore
        indicators_series[indicator] = indicators_series[indicator][
            abs(first_series_month - first_common_month) :
        ]

    # * make sure that all series end to the same date
    last_common_year = min(
        [int(indicators_series[indicator]["Year"].iloc[-1]) for indicator in Indicator]  # type: ignore
    )
    for indicator in Indicator:
        indicators_series[indicator] = indicators_series[indicator][
            indicators_series[indicator]["Year"] <= last_common_year
        ]
    last_common_month = min(
        [int(indicators_series[indicator]["Month"].iloc[-1]) for indicator in Indicator]  # type: ignore
    )
    for indicator in Indicator:
        last_series_month = int(indicators_series[indicator]["Month"].iloc[-1])  # type: ignore
        if last_common_month != last_series_month:
            indicators_series[indicator] = indicators_series[indicator][
                : -abs(last_common_month - last_series_month)
            ]

    # * check we have the same number of points for each indicator
    lengths = [len(indicators_series[indicator]) for indicator in Indicator]
    if len(set(lengths)) != 1:
        raise Exception(
            f"{country} has different lengths of indicator even if same start and end date: {lengths}"
        )

    # * convert to matrix
    country_data: dict[Indicator, Matrix[Literal["N"], Float]] = {}
    for indicator in Indicator:
        country_data[indicator] = convert_to_matrix(
            indicators_series[indicator], indicator, pct=pct
        )

    return country_data, indicators_series[GDP]["Date"].to_numpy()
