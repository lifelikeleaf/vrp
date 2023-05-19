# Instructions

Note: this project requires Python 3.9+.

Install all the dependencies:
```
pip install -r requirements.txt
```

Do a quick test run:
```
python main.py
```

`main.py` is the main entry point for configuring experiments.

The experiment runner in `experiments.py` is responsible for actually running experiments and writing output data to files.

The core package written for this project is under the `vrp/decomp/` directory.

The directory `vrp/third_party/` contains third party open source code that is no longer installable via pip, e.g. cvrplib reader that reads CVRPLIB benchmark instances.

The `archived/` folder contains outdated code that's no longer needed. It's only kept for reference and historical reaons.
