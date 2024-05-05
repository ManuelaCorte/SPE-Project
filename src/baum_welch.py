import os
import pandas as pd

from src.data import clean_dataset, convert_to_structured_matrix
from src.structs import Country, Indicator

GDP, IR, CPI = Indicator

def serialize_country_data(country: Country):
    country_indicators: dict[Indicator, dict[str, list[float]]] = {}
    for indicator in Indicator:
      matrix = convert_to_structured_matrix(df, indicator, country)
      values = [m[0] for m in matrix]
      years: list[float] = []
      for m in matrix:
          year_s = m[1].split("-")
          year_n = int(year_s[0]) * 100 + int(year_s[1])
          years.append(year_n)
      country_indicators[indicator] = {'years': years, 'values': values}

    def getlist(indicator: Indicator):
        return country_indicators[indicator]['years']
    
    while getlist(CPI)[0] != getlist(GDP)[0] or getlist(CPI)[0] != getlist(IR)[0]:
        if getlist(CPI)[0] < getlist(GDP)[0] and getlist(CPI)[0] < getlist(IR)[0]:
            getlist(CPI).pop(0)
            country_indicators[CPI]['values'].pop(0)
        elif getlist(IR)[0] < getlist(GDP)[0]:
            getlist(IR).pop(0)
            country_indicators[IR]['values'].pop(0)
        else: 
            getlist(GDP).pop(0)
            country_indicators[GDP]['values'].pop(0)

    while getlist(CPI)[-1] != getlist(GDP)[-1] or getlist(CPI)[-1] != getlist(IR)[-1]:
        if getlist(CPI)[-1] > getlist(GDP)[-1] and getlist(CPI)[-1] > getlist(IR)[-1]:
            getlist(CPI).pop()
            country_indicators[CPI]['values'].pop()
        elif getlist(IR)[-1] > getlist(GDP)[-1]:
            getlist(IR).pop()
            country_indicators[IR]['values'].pop()
        else: 
            getlist(GDP).pop()
            country_indicators[GDP]['values'].pop()
      
    length = len(country_indicators[GDP]['years'])
    country_data: dict[Indicator, list[float]] = {}
    for indicator in Indicator:
        country_data[indicator] = country_indicators[indicator]['values']
        if len(country_data[indicator]) != length:
            raise Exception(f"{country} has different lengths of indicator even if same start and end date")
    
    for i in range(length):
        if i == length - 1: continue
        country_data[CPI][length - i - 1] = country_data[CPI][length - i - 1] - country_data[CPI][length - i - 2]
        country_data[IR][length - i - 1] = country_data[IR][length - i - 1] - country_data[IR][length - i - 2]

    for indicator in Indicator:
        country_data[indicator].pop(0)
    return country_data
    
      

if __name__ == "__main__":
    if os.path.exists("data/cleaned/dataset.csv"):
        df = pd.read_csv("data/cleaned/dataset.csv")
    else:
        df = clean_dataset(save_intermediate=True)

    countries_data: dict[Country, dict[Indicator, list[float]]] = {}

    for country in Country:
      countries_data[country] = serialize_country_data(country)
    print(countries_data[Country.BRAZIL])

    

      
