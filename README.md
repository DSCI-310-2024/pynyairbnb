# pynyairbnb

A package for DSCI 310 Group's Airbnb Analysis.

[![codecov](https://codecov.io/gh/DSCI-310-2024/pynyairbnb/graph/badge.svg?token=duRYHnZQ12)](https://codecov.io/gh/DSCI-310-2024/pynyairbnb)

## Installation

```bash
$ pip install pynyairbnb
```

## Usage

`pynyairbnb` can be used to load data from insideairbnb.com's NYC Airbnb Open Data, perform all necessary preprocessing necessary, generate visualizations, build a knn-classification model and conduct a hyperparameter optimization on the model as follows:

```python
from pynyairbnb.data_preprocessing import data_preprocessing
from pynyairbnb.plotting import plot_pynyairbnb
from pynyairbnb.pynyairbnb import nyairbnb_analysis

data_preprocessing("example-link-data.csv", "documents/data_files")  # url to your data and path to save your data
plot_pynyairbnb("documents/data_files/train_df.csv", "documents/data_figures", "documents/data_tables") # path to data files and output paths to save figures and tables
nyairbnb_analysis("documents/data_files", "documents/data_tables") # path to data files and output path to save tables
```

All outputs get saved as .csv or .jpg files that you can read in using the `pandas.read_csv()` function or the `matplotlib.pyplot.imread()` function respectively.

## Contributing

Interested in contributing? Check out the [contributing guidelines](./CONTRIBUTING.md). Please note that this project is released with a [Code of Conduct](./CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## License

`pynyairbnb` was created by by the members of Group 9 for DSCI 310. It is licensed under the terms of the [MIT license](./LICENSE).

_Please refer to `LICENSE.md` for detailed licensing information._

## Credits

`pynyairbnb` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

This package also utilizes New York City Airbnb Open Data from [insideairbnb.com](http://insideairbnb.com/get-the-data/) for example demonstrations.

## Contributors

Rashi Selarka

Riddhi Battu

Oliver Gullery

Prithvi Sureka
