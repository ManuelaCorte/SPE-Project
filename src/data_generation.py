import warnings

from src.data import clean_dataset

warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":
    clean_dataset(save_intermediate=True)
    print("Data generation completed.")
