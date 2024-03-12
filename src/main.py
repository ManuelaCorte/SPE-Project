import os
import warnings

import pandas as pd

from src.data import clean_dataset, convert_to_structured_matrix
from src.structs import Country, Indicator

warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":
    if os.path.exists("data/cleaned/dataset.csv"):
        df = pd.read_csv("data/cleaned/dataset.csv")
    else:
        df = clean_dataset(save_intermediate=True)

    matrix = convert_to_structured_matrix(df, Indicator.IR, Country.UNITED_STATES)
    print(matrix)
