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

The raw datasets can be retrieved by running the following command

```
git lfs pull
```

and wiil be saved in the `data/raw` directory.

If you don't have Git LFS installed, you can download the raw datasets from [this link](https://drive.google.com/drive/folders/1ClauAPLzxeDO2zt1UJ6nhdN8vMoptdKP?usp=sharing) and save them in the `data/raw` directory. Using this method, if _git_ asks you to commit the changes, you can ignore them using this command
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
poetry run python -m src.hmm --countries <list of countries separates by a space>
```

The list of available countries can be obtained by running the following command:

```
poetry run python -m src.hmm --help
```

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
