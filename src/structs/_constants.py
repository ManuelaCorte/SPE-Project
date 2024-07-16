from enum import Enum, StrEnum
from typing import Self

# from typing import Literal


class Country(Enum):
    """
    Enum containing the correspondence between the countries we are interested in and their
    corresponding international code. The list of codes is taken from
    https://www.imf.org/external/pubs/ft/weo/2022/02/weodata/co.pdf
    """

    ARGENTINA = 213
    AUSTRALIA = 193
    AUSTRIA = 122
    AZERBAIJAN = 912
    BELGIUM = 124
    BANGLADESH = 513
    BRAZIL = 223
    CANADA = 156
    CHILE = 228
    DENMARK = 128
    EGYPT = 469
    FINLAND = 172
    FRANCE = 132
    GERMANY = 134
    GREECE = 174
    IRELAND = 178
    INDIA = 534
    INDONESIA = 536
    ITALY = 136
    ISRAEL = 436
    CHINA = 924
    JAPAN = 158
    MEXICO = 273
    MOROCCO = 686
    NETHERLANDS = 138
    NEW_ZEALAND = 196
    NORWAY = 142
    PAKISTAN = 564
    RUSSIA = 922
    SAUDI_ARABIA = 456
    SOUTH_AFRICA = 199
    SPAIN = 184
    SWEDEN = 144
    SWITZERLAND = 146
    SOUTH_KOREA = 542
    TURKEY = 186
    UKRAINE = 926
    UNITED_KINGDOM = 122
    UNITED_STATES = 111

    @staticmethod
    def get_all_codes() -> list[str]:
        return [str(country.value) for country in Country]

    @staticmethod
    def get_all_countries():
        return [country for country in Country]


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
