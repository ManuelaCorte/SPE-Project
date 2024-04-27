import argparse
import warnings

from src.data import clean_dataset

warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        default=True,
        help="Save intermediate data files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force data generation",
    )
    args = parser.parse_args()

    clean_dataset(save_intermediate=args.save_intermediate, force=args.force)
    print("Data generation completed.")
