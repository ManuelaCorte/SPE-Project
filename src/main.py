from src.data import clean_dataset
from src.structs.constants import Country, TimePeriod

if __name__ == "__main__":
    clean_dataset("test.csv", Country.ITALY, TimePeriod.QUARTER, save_intermediate=True)
