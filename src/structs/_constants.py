from enum import Enum, StrEnum

# from typing import Literal


class Country(Enum):
    """
    Enum containing the correspondence between the countries we are interested in and their
    corresponding international code. The list of codes is taken from
    https://www.imf.org/external/pubs/ft/weo/2022/02/weodata/co.pdf
    """

    # ARGENTINA = 213
    # AUSTRALIA = 193
    BRAZIL = 223
    CANADA = 156
    # FRANCE = 132
    # GERMANY = 134
    INDIA = 534
    INDONESIA = 536
    ITALY = 136
    # CHINA = 924
    JAPAN = 158
    MEXICO = 273
    RUSSIA = 922
    # SAUDI_ARABIA = 456
    SOUTH_AFRICA = 199
    SOUTH_KOREA = 542
    # TURKEY = 186
    # UNITED_KINGDOM = 122
    UNITED_STATES = 111
    # EUROPEAN_UNION = 163
    # G20 = 0

    @staticmethod
    def get_all_codes() -> list[str]:
        return [str(country.value) for country in Country]


class Indicator(StrEnum):
    """
    Enum containing the indicators we are interested in.
    """

    GDP = "Gross Domestic Product, Real, Seasonally Adjusted, Domestic Currency"
    IR = "Financial, Interest Rates, Lending Rate, Percent per annum"
    # CPI = "Prices, Consumer Price Index, All items, Index"
    CPI = "Headline Consumer Price Index"

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

class HiddenState(Enum):
    """
    The hidden state of the markov chain. Each state tells if the Interest Rate and the CPI are increasing or decreasing
    """

    I_IR_I_CPI = 0
    I_IR_D_CPI = 1
    D_IR_I_CPI = 2
    D_IR_D_CPI = 3

    @staticmethod
    def get_all_states() -> list[int]:
        return [state.value for state in HiddenState]

    @staticmethod
    def get_state(ir: float, cpi: float):
        if ir > 0 and cpi > 0:
            return HiddenState.I_IR_I_CPI
        elif ir > 0 and cpi < 0:
            return HiddenState.I_IR_D_CPI
        elif ir < 0 and cpi > 0:
            return HiddenState.D_IR_I_CPI
        else:
            return HiddenState.D_IR_D_CPI


class KnownVariables(Enum):
    """
    The known variables of the markov chain. Each state tells if the GDP is increasing or decreasing
    """

    I_GDP = 0
    D_GDP = 1

    @staticmethod
    def get_all_variables() -> list[int]:
        return [var.value for var in KnownVariables]

    @staticmethod
    def get_variable(gdp: float):
        return KnownVariables.I_GDP if gdp > 0 else KnownVariables.D_GDP
