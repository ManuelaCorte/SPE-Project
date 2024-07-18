import os
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
        length = len(indicators_series[indicator])
        if length == 0:
            raise Exception(f"{country} does not have data for {indicator}")

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


def create_countries_data(
    country: Country, all_countries: bool, pct: bool
) -> tuple[
    list[Country],
    dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]],
    dict[Country, Matrix[Literal["N"], np.str_]],
]:
    """Serilize the data for the given country so that only the time periods in which all the
    indicators are available are considered and converted to matrices.

    Parameters:
        country: The country to be considered
        all_countries: Whether to consider all the countries or only the given one
        pct: Whether to return the values as is or as percentages changes from the previous period or as raw values

    Returns:
        The countries, the data for the countries and the dates
    """
    countries = Country.get_all_countries() if all_countries else [country]
    countries_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]] = {}
    dates: dict[Country, Matrix[Literal["N"], np.str_]] = {}
    for country in countries:
        if os.path.exists("data/cleaned/dataset.csv"):
            df = pd.read_csv("data/cleaned/dataset.csv")
        else:
            raise FileNotFoundError(
                "The cleaned dataset is not available. Run the data generation script first."
            )

        try:
            country_data: dict[Indicator, Matrix[Literal["N"], Float]] = {}

            country_data, date = serialize_country_data(df, country, pct=pct)
            dates[country] = date
            countries_data[country] = country_data
        except Exception as e:
            print(f"Error while processing {country.name}: {e}")
    return countries, countries_data, dates


def divide_training_test_covid_data(
    countries: list[Country],
    countries_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]],
    dates: dict[Country, Matrix[Literal["N"], np.str_]],
):
    """Divide the data between training, testing and flawed due to covid. The training data
    is the data up to the end of 2016, the testing data is the data after that year and the covid
    data is the data after 2020.

    Parameters:
        countries: The countries to be considered
        countries_data: The data for the countries
        dates: The dates for the countries

    Returns:
        The training, testing and covid data
    """
    TESTING_YEAR = "2017"
    COVID_YEAR = "2020"
    print(
        f"Data is divided between training, testing (after {TESTING_YEAR}) and flawed due to covid (after {COVID_YEAR})"
    )

    training_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]] = {}
    test_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]] = {}
    covid_data: dict[Country, dict[Indicator, Matrix[Literal["N"], Float]]] = {}

    print()
    for country in countries:
        if country not in countries_data:
            continue
        country_data = countries_data[country]

        training_data[country] = {}
        test_data[country] = {}
        covid_data[country] = {}

        this_date = dates[country]
        test_date_index = -1
        covid_date_index = -1
        for i in range(len(this_date)):
            if test_date_index < 0 and TESTING_YEAR in this_date[i]:
                test_date_index = i
            if covid_date_index < 0 and COVID_YEAR in this_date[i]:
                covid_date_index = i

        for indicator in country_data:
            data = country_data[indicator]
            training_data[country][indicator] = data[:test_date_index:]
            test_data[country][indicator] = data[test_date_index:covid_date_index:]
            covid_data[country][indicator] = data[covid_date_index::]

        training_points = len(training_data[country][Indicator.GDP])
        testing_points = len(test_data[country][Indicator.GDP])
        covid_points = len(covid_data[country][Indicator.GDP])

        country_name = "{:<13}".format(country.name)
        print(
            f"Country {country_name} starts from {this_date[0]} and ends at {this_date[-1]} ({training_points} training, {testing_points} test, {covid_points} covid)"
        )
    print()

    return training_data, test_data, covid_data
