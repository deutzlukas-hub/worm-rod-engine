# Worm-rod-engine

## Installation

Follow these steps to set up your developement environment.

1. Create conda/mamba environment from `environment.yml` file
```
conda env create -f environment.yml
```
2. Active conda environment with
```
conda activate worm-rod-engine
```
3. Open a Python shell and try importing the package to very installation
```
>>> import worm_rod_engine
```

# Usage

The main functionality is provided through the `worm-rod-engine.worm.Worm` class. For example use cases refer to the `worm_rod_engine/examples` directory.

## Running Tests

To ensure everything is working correctly, run the tests with
```
python tests/run_tests.py
``` 
This commond excutes all tests defined in the `tests` directory. 
