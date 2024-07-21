# Simulation and Perfomance Evaluation Project

## Setting up the project

To get the project up and running, you need to clone the repository and navigate to the project directory.

```
git clone https://github.com/ManuelaCorte/SPE-Project.git
cd SPE-Project
```

The required dependencies can be installed using [Poetry](https://python-poetry.org/docs/). Once Poetry is installed, run the following command to install the dependencies:

```
poetry install --no-root
```

Alternatively, you can install the dependencies system-wise using pip (not recommended):

```
pip install -r requirements.txt
```

## Running the project

To run any file, you can either use the `poetry run` command or activate the virtual environment created by Poetry by running `poetry shell`.

### Data generation

The raw datasets can be retrieved at [this link](https://drive.google.com/drive/folders/1ClauAPLzxeDO2zt1UJ6nhdN8vMoptdKP?usp=sharing) and then saved in the `data/raw` directory. Using this method, if _git_ asks you to commit the changes, you can ignore them using this command

```
git update-index --assume-unchanged data/cleaned/* data/raw/*
```

To then generate the final dataset, run the following command:

```
poetry run python -m src.data_generation
```

The final dataset will be saved in the `data/processed` directory, while the specific countries datasets will be saved in the `data/intermediate` directory.

## Running the different models

### Hidden Markov Model

To run the Baum-Welch algorithm for the HMM model, run the following command:

```
poetry run python -m src.hmm [--epochs EPOCHS] [--country COUNTRY] [--multiple-series] [--help]
```

where `EPOCHS` is the number of iterations for the Baum-Welch algorithm, `COUNTRY` is the country for which the model will be trained, and `--multiple-series` is a flag that indicates whether to train the model on all available countries.

The list of available countries as well as additional information can be obtained by running the following command:

```
poetry run python -m src.hmm --help
```

### Correlation Estimation

The Pearson, Kendall and Spearman correlation coefficients can be estimated with bootstrapping by running the following command:

```
poetry run python -m src.correlation [--repetitions REPETITIONS] [--alpha ALPHA] [--country COUNTRY] [--all] [--help]
```

where `REPETITIONS` is the number of bootstrap repetitions, `ALPHA` is the significance level for the confidence interval, `COUNTRY` is the country for which the correlation coefficients will be estimated, and `--all` is a flag that indicates whether to estimate the correlation coefficients for all available countries.

### Regression

The regression model is implemented as a linear regression model where the residuals are modeled as a first-order autoregressive process following the Prais-Winsten method. To run the regression model, run the following command:

```
poetry run python -m src.regression [--country COUNTRY] [--tolerance TOLERANCE] [--add_constant] [--help]
```

where `COUNTRY` is the country for which the regression model will be trained, `TOLERANCE` is the tolerance level / p-value for the Ljung-Box test to check for autocorrelation in the residuals, and `--add_constant` is a flag that indicates whether to add a constant term to the regression model.

## Contributing

In order to keep the project organized, pre-commit hooks are used. To install them, run the following command:

```
poetry run pre-commit install
poetry run pre-commit install-hooks
```

The pre-commit hooks will run every time you commit changes to the repository. If any of the hooks fail, the commit will be aborted. If you want to run the hooks manually, you can do so by running the following command:

```
poetry run pre-commit run --all-files
```

These hooks are also run by GitHub Actions on every push.

Moreover, the project is fully typed using [Pyright](https://microsoft.github.io/pyright/#/). Pyright configurations are specified inside the `pyproject.toml` file. To check the correctness of type hints, run the following command:

```
poetry run pyright src/
```
