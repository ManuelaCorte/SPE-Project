from enum import Enum, StrEnum

# from typing import Literal


class Country(Enum):
    """
    Enum containing the correspondence between the countries we are interested in and their
    corresponding international code. The list of codes is taken from
    https://www.imf.org/external/pubs/ft/weo/2022/02/weodata/co.pdf
    """

    ARGENTINA = 213
    AUSTRALIA = 193
    BRAZIL = 223
    CANADA = 156
    FRANCE = 132
    GERMANY = 134
    INDIA = 534
    INDONESIA = 536
    ITALY = 136
    CHINA = 924
    JAPAN = 158
    MEXICO = 273
    RUSSIA = 922
    SAUDI_ARABIA = 456
    SOUTH_AFRICA = 199
    SOUTH_KOREA = 542
    TURKEY = 186
    UNITED_KINGDOM = 122
    UNITED_STATES = 111
    EUROPEAN_UNION = 163
    G20 = 0

    @staticmethod
    def get_all_codes() -> list[str]:
        return [str(country.value) for country in Country]


class Indicator(StrEnum):
    """
    Enum containing the indicators we are interested in.
    """

    GDP = "Gross Domestic Product"
    IR = "Interest Rates"
    CPIDX = "Consumer Price Index"
    CPI = "Consumer Price Inflation"

    @staticmethod
    def get_all_names() -> list[str]:
        return [indicator.value for indicator in Indicator]


class TimePeriod(StrEnum):
    """
    Enum containing the time periods we are interested in.
    """

    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
