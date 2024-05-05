import os
import pandas as pd

from src.data import clean_dataset, convert_to_structured_matrix
from src.structs import Country, Indicator

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
    
    while getlist(Indicator.CPI)[0] != getlist(Indicator.GDP)[0] or getlist(Indicator.CPI)[0] != getlist(Indicator.IR)[0]:
        if getlist(Indicator.CPI)[0] < getlist(Indicator.GDP)[0] and getlist(Indicator.CPI)[0] < getlist(Indicator.IR)[0]:
            getlist(Indicator.CPI).pop(0)
        elif getlist(Indicator.IR)[0] < getlist(Indicator.GDP)[0]:
            getlist(Indicator.IR).pop(0)
        else: 
            getlist(Indicator.GDP).pop(0)

    while getlist(Indicator.CPI)[-1] != getlist(Indicator.GDP)[-1] or getlist(Indicator.CPI)[-1] != getlist(Indicator.IR)[-1]:
        if getlist(Indicator.CPI)[-1] > getlist(Indicator.GDP)[-1] and getlist(Indicator.CPI)[-1] > getlist(Indicator.IR)[-1]:
            getlist(Indicator.CPI).pop()
        elif getlist(Indicator.IR)[-1] > getlist(Indicator.GDP)[-1]:
            getlist(Indicator.IR).pop()
        else: 
            getlist(Indicator.GDP).pop()
    return country_indicators
    
      

if __name__ == "__main__":
    if os.path.exists("data/cleaned/dataset.csv"):
        df = pd.read_csv("data/cleaned/dataset.csv")
    else:
        df = clean_dataset(save_intermediate=True)

    country_data: dict[Country, dict[Indicator, dict[str, list[float]]]] = {}

    for country in Country:
      country_data[country] = serialize_country_data(country)
    print(country_data)

    

      
