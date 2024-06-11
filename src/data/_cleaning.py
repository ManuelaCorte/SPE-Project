import os

import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from src.structs import Country, Indicator, TimePeriod

from ._dataframe import get_indicators_columns, get_time_periods_colums


def clean_dataset(save_intermediate: bool = False, force: bool = False) -> DataFrame:
    """
    Read the raw dataset and clean it by removing unused indicators, filtering only G20 countries and taking monthly data. For
    GDP, the quarterly data is interpolated to get monthly data. The dataframe is then reshaped to have the form
    Country | Indicator | Year | Month | Value

    Parameters:
        save_intermediate: If True, the intermediate dataframes (filtered indicators and country spcific datasets)
        will be saved to csv files.
        force: If True, all intermediate and final files will be removed and the function will start from scratch.
    Returns:
        The dataframe containing the cleaned data.
    """
    chunksize: int = 10**4
    intermediate_file_path = "data/intermediate/"
    if not os.path.exists(intermediate_file_path):
        os.makedirs(intermediate_file_path)

    if force:
        for file in os.listdir(intermediate_file_path):
            os.remove(intermediate_file_path + file)
        for file in os.listdir("data/cleaned/"):
            os.remove("data/cleaned/" + file)

    reader: TextFileReader = pd.read_csv(
        "data/raw/IFS_09-26-2023 00-50-38-77_timeSeries.csv",
        chunksize=chunksize,
        dtype=str,
    )

    if os.path.exists(intermediate_file_path + "removed_indicators.csv"):
        df = pd.read_csv(intermediate_file_path + "removed_indicators.csv", dtype=str)
    else:
        df = _remove_unused_information(reader)
        if save_intermediate:
            df.to_csv(intermediate_file_path + "removed_indicators.csv", index=False)

    countries: list[DataFrame] = []
    for country in Country:
        # if country == Country.G20:
        #     continue
        print("Processing", country.name.capitalize())
        gdp_interest = _get_monthly_data(df, country)
        inflation = _clean_inflation_dataset(country)
        data = pd.concat([gdp_interest, inflation], ignore_index=True)
        data["Date"] = pd.to_datetime(
            data["Year"].astype(str) + "-" + data["Month"].astype(str).str.zfill(2),
            format="%Y-%m",
        )
        data = _convert_to_percentage(data)
        countries.append(data)

        if save_intermediate:
            data.to_csv(
                intermediate_file_path + f"/{country.name.lower()}.csv", index=False
            )

    df = pd.concat(countries, ignore_index=True)

    df.sort_values(by=["Country Name", "Indicator Name", "Year", "Month"], inplace=True)
    # Reorder columns
    df = df[
        [
            "Country Code",
            "Country Name",
            "Indicator Name",
            "Date",
            "Year",
            "Month",
            "Value",
            "Value_pct",
        ]
    ]

    df.to_csv("data/cleaned/dataset.csv", index=False)
    return df


###################################################################################################
### Private methods
###################################################################################################
def _clean_inflation_dataset(
    country: Country,
) -> DataFrame:
    """
    Read the raw dataset and clean it by removing unused indicators taking quarterly data and filtering by country.

    Parameters:
        country: The country to filter by. If the country is None

    Returns:
        The dataframe containing the cleaned data.
    """
    df = pd.read_csv(
        "data/raw/Inflation data - hcpi_m.csv",
        dtype=str,
    )

    # Filter by country
    df = df[df["IMF Country Code"] == str(country.value)]

    # Remove final notes columns
    df.drop(columns=["Unnamed: 644", "Data source", "Base date", "Note"], inplace=True)

    # Rename column to match the other datasets
    df.drop(columns=["Country Code", "Indicator Type"], inplace=True)
    df.rename(
        columns={
            "IMF Country Code": "Country Code",
            "Country": "Country Name",
            "Series Name": "Indicator Name",
        },
        inplace=True,
    )

    # Replace 0 with NaN
    df = df.replace("0.0", float("nan"))

    df = _rename_time_period_columns(df, TimePeriod.MONTH)

    df = _reshape_dataframe(df)

    return df


def _remove_unused_information(reader: TextFileReader) -> DataFrame:
    """
    Remove all indicators that are not in the list of indicators we are interested in and all
    countries not in the G20.

    Parameters:
        reader: The reader to read the csv file.

    Returns:
        The filtered dataframe.
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

        # Remove all lines not corresponding to the G20 countries
        chunk = chunk[chunk["Country Code"].isin(Country.get_all_codes())]

        # Remove final columns used for notes
        chunk.drop(columns=["Base Year", "Unnamed: 1770"], inplace=True)

        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


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
        case _:
            raise ValueError("Invalid time period")

    # Create a new dataframe with the columns corresponding to the given time period
    columns_to_remove = dataframe.columns.difference(
        indicators_columns + time_periods_columns
    )
    dataframe = dataframe.drop(columns=columns_to_remove)

    return dataframe


def _get_monthly_data(dataframe: DataFrame, country: Country) -> DataFrame:
    """Returns dataframe with monthly data and interpolation of quarter data for gdp.
    The dataframe is then reshaped to have the form Country | Indicator | Year | Month | Value

    Parameters:
        dataframe: The dataframe to filter.

     Returns:
         cleaned dataframe
    """
    country_dataframe = dataframe[dataframe["Country Code"] == str(country.value)]
    gdp_quarter = _get_different_time_granularities(
        country_dataframe, TimePeriod.QUARTER
    )
    gdp_quarter = gdp_quarter[
        gdp_quarter["Indicator Name"].str.contains(Indicator.GDP.value)
    ]
    gdp_quarter = _reshape_dataframe(gdp_quarter)

    # Convert the date to yyyy-mm-dd format
    gdp_quarter["Date"] = pd.to_datetime(
        gdp_quarter["Year"].astype(str)
        + "-"
        + gdp_quarter["Month"].astype(str).str.zfill(2),
        format="%Y-%m",
    )
    gdp_quarter["Date"] = pd.to_datetime(gdp_quarter["Date"]).dt.to_period("M")
    gdp_quarter = gdp_quarter.set_index("Date")

    # Interpolate the quarterly data to get monthly data
    gdp_month = gdp_quarter.resample("M").interpolate()
    gdp_month["Date"] = gdp_month.index
    gdp_month["Year"] = gdp_month["Date"].dt.year
    gdp_month["Month"] = gdp_month["Date"].dt.month
    gdp_month = gdp_month.drop(columns=["Date"])
    gdp_month = gdp_month.reset_index(drop=True)

    # Repeat indicator name and country code for each month
    gdp_month["Indicator Name"] = Indicator.GDP.value
    gdp_month["Country Code"] = country.value
    gdp_month["Country Name"] = country.name.capitalize()

    # Get monthly data for all other indicators
    monthly_data = _get_different_time_granularities(
        country_dataframe, TimePeriod.MONTH
    )
    interest_rates = monthly_data[monthly_data["Indicator Name"] == Indicator.IR.value]
    interest_rates = _reshape_dataframe(interest_rates)

    df = pd.concat([gdp_month, interest_rates], ignore_index=True)

    return df


def _reshape_dataframe(df: DataFrame) -> DataFrame:
    """
    Reshape the dataframe to have the form Country | Indicator | Year | Month | Value

    Parameters:
        dataframe: The dataframe to filter.

    Returns:
        The dataframe containing the cleaned data.
    """
    # Reshape the dataframe to have the form Country | Indicator | Year | Month | Value
    dataframe = df.melt(
        id_vars=("Country Name", "Country Code", "Indicator Name"),
        var_name="Time",
        value_name="Value",
    )
    # Convert format 1990M2 to 1990-02 and 1990Q2 to 1990-06
    dataframe["Year"] = dataframe["Time"].str.extract(r"(\d{4})")
    dataframe["Month"] = dataframe["Time"].str.extract(r"(\d{4}[M|Q])(\d{1,2})")[1]
    dataframe["Month"] = dataframe["Month"].str.replace("M", "")
    dataframe["Month"] = dataframe["Month"].str.replace("Q", "")
    if dataframe.empty:
        return dataframe
    if "Q" in dataframe["Time"].values[0]:
        dataframe["Month"] = dataframe["Month"].astype(int) * 3

    dataframe["Year"] = dataframe["Year"].astype(int)
    dataframe["Month"] = dataframe["Month"].astype(int)
    dataframe.drop(columns=["Time"], inplace=True)

    dataframe = dataframe.dropna(axis=0, how="all", subset=["Value"])

    # Remove quotes from the values
    dataframe["Value"] = dataframe["Value"].str.replace(",", ".")
    dataframe["Value"] = dataframe["Value"].str.replace('"', "")
    dataframe["Value"] = dataframe["Value"].str.replace("'", "")
    dataframe["Value"] = dataframe["Value"].str.strip()

    dataframe["Value"] = dataframe["Value"].astype(float)

    return dataframe


def _rename_time_period_columns(
    dataframe: DataFrame, time_period: TimePeriod
) -> DataFrame:
    """
    Rename to time periods columns to match the other datasets. The columns are renamed to the format
    YYYY for year, YYYYQX for quarter and YYYYMX for month.

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
        case _:
            raise ValueError("Invalid time period")

    dataframe.rename(
        columns=dict(zip(old_columns_names, new_columns_names)), inplace=True
    )

    return dataframe


def _convert_to_percentage(df: DataFrame) -> DataFrame:
    """
    Add columns to the dataframe containing the percentage change of the indicators values.

    Returns:
        The dataframe with the GDP values added.
    """
    gdp = df[df["Indicator Name"].str.contains(Indicator.GDP.value)].copy()
    ir = df[df["Indicator Name"].str.contains(Indicator.IR.value)].copy()
    cpi = df[df["Indicator Name"].str.contains(Indicator.CPI.value)].copy()
    gdp["Value_pct"] = df["Value"].pct_change()
    ir["Value_pct"] = df["Value"].pct_change()
    cpi["Value_pct"] = df["Value"].pct_change()

    # Remove the first row
    gdp = gdp.iloc[1:]
    ir = ir.iloc[1:]
    cpi = cpi.iloc[1:]

    return pd.concat([gdp, ir, cpi], ignore_index=True)
