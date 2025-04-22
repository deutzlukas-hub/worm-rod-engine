# Worm-rod-engine

## Installation

Follow these steps to set up your developement environment.

1. Use to conda create python environment from `environment.yml` file
```
conda env create -f environment.yml
```
2. Active conda environment with
```
conda activate worm-rod-engine
```
3. Use pip to add the worm-rod-engine package defined in `setup.py` to the active environment 
```
pip install -e .
```
3. Open a Python shell and try to import worm-rod-engine to very installation
```
>>> import worm_rod_engine
```

# Usage

The main functionality is provided through the `worm-rod-engine.worm.Worm` class. For example use cases refer to the `worm_rod_engine/examples` directory.

## Running Tests

To ensure everything is working correctly, run all unit-tests with
```
python tests/run_tests.py
``` 

