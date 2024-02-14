import os
from typing import LiteralString, Optional

import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from src.structs import Country, Indicator, TimePeriod

from ._dataframe import get_indicators_columns, get_time_periods_colums


def clean_gdp_dataset(
    country: Country,
    time_granularity: TimePeriod,
    save_intermediate: bool = False,
    force: bool = False,
) -> DataFrame:
    """
    Read the raw dataset and clean it by removing unused indicators and filtering by country and by specific
    time rate (i.e. year, quarter, month).

    Parameters:
        country: The country to filter by. If the country is None, only the
            rows corresponding to all g20 countries will be returned.
        time_granularity: The time granularity to filter by.
        save_intermediate: If True, the intermediate dataframes will be saved to csv files.
        force: If True, the function will not check if the output file already exists.

    Returns:
        The dataframe containing the cleaned data.
    """
    chunksize: int = 10**4
    output_file_path = "data/cleaned"
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    reader: TextFileReader = pd.read_csv(
        "data/raw/IFS_09-26-2023 00-50-38-77_timeSeries.csv",
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

    return df


def clean_inflation_dataset(
    country: Country,
    time_granularity: TimePeriod,
    save_intermediate: bool = False,
    force: bool = False,
) -> DataFrame:
    """
    Read the raw dataset and clean it by removing unused indicators and filtering by country and by specific
    time rate (i.e. year, quarter, month).

    Parameters:
        country: The country to filter by. If the country is None, only the
            rows corresponding to all g20 countries will be returned.
        time_granularity: The time granularity to filter by.
        save_intermediate: If True, the intermediate dataframes will be saved to csv files.
        force: If True, the function will not check if the output file already exists.

    Returns:
        The dataframe containing the cleaned data."""
    output_file_path = "data/cleaned"
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    match time_granularity.value:
        case "year":
            df = pd.read_csv(
                "data/raw/Inflation data - hcpi_a.csv",
                dtype=str,
            )
        case "quarter":
            df = pd.read_csv(
                "data/raw/Inflation data - hcpi_q.csv",
                dtype=str,
            )
        case "month":
            df = pd.read_csv(
                "data/raw/Inflation data - hcpi_m.csv",
                dtype=str,
            )

    # Rename column to match the other datasets
    df.drop(columns=["Country Code", "Indicator Type"], inplace=True)
    df.rename(
        columns={"IMF Country Code": "Country Code", "Series Name": "Indicator Name"},
        inplace=True,
    )
    _rename_time_period_columns(df, time_granularity)

    country_path: LiteralString = output_file_path + f"/{country.name.lower()}"
    if not os.path.exists(output_file_path + f"/{country.name.lower()}"):
        os.makedirs(country_path)

    if (
        os.path.exists(country_path + f"/inflation_{time_granularity.value}.csv")
        and not force
    ):
        df = pd.read_csv(
            country_path + f"/inflation_{time_granularity.value}.csv", dtype=str
        )
    else:
        df = _filter_dataframe_by_country(df, country)
        if save_intermediate:
            df.to_csv(
                country_path + f"/inflation_{time_granularity.value}.csv",
                index=False,
            )

    return df


def remove_empty_values(
    input: str | DataFrame,
    output: Optional[str] = None,
) -> DataFrame:
    """
    Remove all rows and columns that have only NaN values.

    Parameters:
        input_file: The path to the csv file containing the dataset.
        output_file: If specified, the path to the csv file where the dataset without nan values.
        force: If True, the function will not check if the output file already exists.

    Returns:
        The dataframe containing the cleaned data.

    """
    if isinstance(input, str):
        df: DataFrame = pd.read_csv(input, dtype=str)
    else:
        df = input

    # Remove all rows that have only NaN values
    df = df.dropna(axis=0, how="all")

    # Remove all columns that have any NaN value
    df = df.dropna(
        axis=1,
        how="any",
    )

    if output is not None:
        df.to_csv(output, index=False)

    return df


###################################################################################################
### Private methods
###################################################################################################


def _remove_unused_indicators(reader: TextFileReader) -> DataFrame:
    """
    Remove all indicators that are not in the list of indicators we are interested in.

    Parameters:
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
            | chunk["Indicator Name"].str.contains(Indicator.CPIDX.value)
            | chunk["Indicator Name"].str.contains(Indicator.IR.value)
        ]

        chunk = chunk[chunk["Attribute"] != "Status"]
        chunk.drop(columns=["Attribute", "Indicator Code"], inplace=True)

        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def _filter_dataframe_by_country(dataframe: DataFrame, country: Country) -> DataFrame:
    """
    Filter the given dataframe by the given country.

    Parameters:
        dataframe: The dataframe to filter.
        country: The country to filter by. If the country is None, only the
            rows corresponding to all g20 countries will be returned.

    Returns:
        The dataframe containing only the rows corresponding to the given country.
    """
    if country.name == Country.G20.name:
        return dataframe[dataframe["Country Code"].isin(Country.get_all_codes())]

    filtered_df: DataFrame = dataframe[dataframe["Country Code"] == str(country.value)]
    return filtered_df


def _get_different_time_granularities(
    dataframe: DataFrame,
    time_period: TimePeriod,
) -> DataFrame:
    """
    Removes the columns not corresponding to the given time period.

    Parameters:
        dataframe: The dataframe to filter.
        time_period: The time period to filter by (year/quarter/month).

    Returns:
        The dataframe containing only the columns corresponding to the given time period.
    """
    columns: list[str] = get_time_periods_colums(dataframe.columns)

    indicators_columns = get_indicators_columns(dataframe.columns)
    time_periods_columns: list[str] = []
    match time_period.value:
        case "year":
            for column in columns:
                if column.isnumeric():
                    time_periods_columns.append(column)
        case "quarter":
            for column in columns:
                year, quarter = column[:4], column[4:]
                if year.isnumeric() and quarter.startswith("Q"):
                    time_periods_columns.append(column)
        case "month":
            for column in columns:
                year, month = column[:4], column[4:]
                if year.isnumeric() and month.startswith("M"):
                    time_periods_columns.append(column)

    # Create a new dataframe with the columns corresponding to the given time period
    columns_to_remove = dataframe.columns.difference(
        indicators_columns + time_periods_columns
    )
    dataframe.drop(columns=columns_to_remove, inplace=True)

    return dataframe


def _rename_time_period_columns(
    dataframe: DataFrame, time_period: TimePeriod
) -> DataFrame:
    """
    Rename to time periods columns to match the other datasets.

    Parameters:
        dataframe: The dataframe to filter.
        time_period: The time period to filter by (year/quarter/month).

    Returns:
        The dataframe containing only the columns corresponding to the given time period.
    """
    columns: list[str] = get_time_periods_colums(dataframe.columns)
    old_columns_names: list[str] = []
    new_columns_names: list[str] = []

    match time_period.value:
        case "year":
            for column in columns:
                if column.isnumeric():
                    old_columns_names.append(column)
                    new_columns_names.append(column)
        case "quarter":
            for column in columns:
                year, quarter = column[:4], column[4:]
                if year.isnumeric() and quarter.isnumeric():
                    old_columns_names.append(column)
                    new_columns_names.append(f"{year}Q{int(quarter)}")
        case "month":
            for column in columns:
                year, month = column[:4], column[4:]
                if year.isnumeric() and month.isnumeric():
                    old_columns_names.append(column)
                    new_columns_names.append(f"{year}M{int(month)}")

    dataframe.rename(
        columns=dict(zip(old_columns_names, new_columns_names)), inplace=True
    )

    return dataframe
