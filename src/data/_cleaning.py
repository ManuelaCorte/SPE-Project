import os
from typing import LiteralString

import pandas as pd
from pandas import DataFrame, Index
from pandas.io.parsers import TextFileReader

from src.structs.constants import Country, Indicator, TimePeriod


def clean_dataset(
    output_file: str,
    country: Country,
    time_granularity: TimePeriod,
    save_intermediate: bool = False,
    force: bool = False,
) -> None:
    chunksize: int = 10**4
    output_file_path = "dataset/cleaned"
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    reader: TextFileReader = pd.read_csv(
        "dataset/raw/IFS_09-26-2023 00-50-38-77_timeSeries.csv",
        chunksize=chunksize,
        dtype=str,
    )

    if os.path.exists(output_file_path + "/removed_indicators.csv") and not force:
        df = pd.read_csv(output_file_path + "/removed_indicators.csv", dtype=str)
    else:
        df = _remove_unused_indicators(reader)
        if save_intermediate:
            df.to_csv(output_file_path + "/removed_indicators.csv", index=False)

    country_path: LiteralString = output_file_path + f"/{country.name.lower()}"
    if not os.path.exists(output_file_path + f"/{country.name.lower()}"):
        os.makedirs(country_path)

    if os.path.exists(country_path + "/filtered_by_country.csv") and not force:
        df = pd.read_csv(country_path + f"/{country.name.lower()}_data.csv", dtype=str)
    else:
        df = _filter_dataframe_by_country(df, country)
        if save_intermediate:
            df.to_csv(
                country_path + f"/{country.name.lower()}_data.csv",
                index=False,
            )

    if (
        os.path.exists(country_path + f"/filtered_by_{time_granularity.value}.csv")
        and not force
    ):
        df = pd.read_csv(
            country_path + f"/filtered_by_{time_granularity.value}.csv",
            dtype=str,
        )
    else:
        df = _get_different_time_granularities(df, time_granularity)
        if save_intermediate:
            df.to_csv(
                country_path + f"/filtered_by_{time_granularity}.csv",
                index=False,
            )

    df.to_csv(output_file, index=False)


def remove_empty_columns(
    input_file: str,
    output_file: str,
    force: bool = False,
) -> None:
    """
    Remove all columns that have only NaN values.

    Args:
        input_file: The path to the csv file containing the dataset.
        output_file: The path to the csv file where the dataset without empty columns
            will be saved.
        force: If True, the function will not check if the output file already exists.
    """
    if os.path.exists(output_file) and not force:
        return

    df: DataFrame = pd.read_csv(input_file, dtype=str)
    df.dropna(axis=1, how="all", inplace=True)
    df.to_csv(output_file, index=False)


###################################################################################################
### Private methods
###################################################################################################


def _remove_unused_indicators(reader: TextFileReader) -> DataFrame:
    """
    Remove all indicators that are not in the list of indicators we are interested in.

    Args:
        reader: The reader to read the csv file.

    Returns:
        The dataframe containing only the rows corresponding to the indicators we are interested in.
    """

    chunks: list[DataFrame] = []

    for chunk in reader:
        assert isinstance(chunk, DataFrame)
        # Filter all lines not containing the following indicators
        chunk = chunk[
            chunk["Indicator Name"].str.contains(Indicator.GDP.value)
            | chunk["Indicator Name"].str.contains(Indicator.CPI.value)
            | chunk["Indicator Name"].str.contains(Indicator.IR.value)
        ]

        chunk = chunk[chunk["Attribute"] != "Status"]
        chunk.drop(columns=["Attribute", "Indicator Code"], inplace=True)

        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def _filter_dataframe_by_country(dataframe: DataFrame, country: Country) -> DataFrame:
    """
    Filter the given dataframe by the given country.

    Args:
        dataframe: The dataframe to filter.
        country: The country to filter by. If the country is None, only the
            rows corresponding to all g20 countries will be returned.

    Returns:
        The dataframe containing only the rows corresponding to the given country.
    """
    if country.name == Country.G20.name:
        return dataframe[dataframe["Country Code"].isin(Country.get_all_codes())]

    filtered_df: DataFrame = dataframe[dataframe["Country Code"] == country.value]
    return filtered_df


def _get_different_time_granularities(
    dataframe: DataFrame,
    time_period: TimePeriod,
) -> DataFrame:
    indicators_columns: list[str] = []
    # Add all colums that don't represent a time period
    for column in dataframe.columns:
        if not column.isnumeric():
            indicators_columns.append(column)
        else:
            break

    time_periods_columns: list[str] = []
    match time_period.value:
        case "year":
            for column in dataframe.columns.difference(indicators_columns):
                if column.isnumeric():
                    time_periods_columns.append(column)
        case "quarter":
            for column in dataframe.columns.difference(indicators_columns):
                year, quarter = column[:4], column[4:]
                if year.isnumeric() and quarter.startswith("Q"):
                    time_periods_columns.append(column)
        case "month":
            for column in dataframe.columns.difference(indicators_columns):
                year, month = column[:4], column[4:]
                if year.isnumeric() and month.startswith("M"):
                    time_periods_columns.append(column)
        case _:
            raise ValueError("Invalid time period")

    # Create a new dataframe with the columns corresponding to the given time period
    columns: Index[str] = dataframe.columns.difference(
        indicators_columns + time_periods_columns
    )
    dataframe.drop(
        columns=columns,
        inplace=True,
        axis=1,
    )
    return dataframe
