# pynyairbnb

A package for DSCI 310 Group's Airbnb Analysis.

[![codecov](https://codecov.io/gh/DSCI-310-2024/pynyairbnb/graph/badge.svg?token=duRYHnZQ12)](https://codecov.io/gh/DSCI-310-2024/pynyairbnb)


Due to the amount of functions we created for our project, we are not successfully able to achieve a high code coverage score. For our main repository, we have tested all functions in our repository. However, for our package repository, we have only provided test functions for functions that we believe to be the most crucial for our package. We have also spoken to the professor about our issue, and she has approved our request of submitting our project with the given code coverage score. Below, we have listed the functions that we have specifically worked on for milestone 4:

### Functions tested in - DSCI-310-2024/DSCI_310_Group_9_NY-airbnb-analysis/src:

- `add_price_category(data)`
- `build_preprocessor(numerical_data, text_data, categorical_data)`
- `build_clf_model(model, preprocessor, tbl_out_dir, X_train, y_train, X_test, y_test, replacement_dict, clf_report_file_name)`
- `knn_param_optimization(knn_model, tbl_out_dir, X_train, y_train, X_test, y_test, replacement_dict, output_file_name, param_dist)`
- `main(input_dir, tbl_out_dir)`

### Functions tested in DSCI-310-2024\pynyairbnb/src:

**data_preprocessing.py** 
- `create_dir_if_not_exists(directory)`
- `read_data(data_url, out_dir)`
- `convert_missing_values(data)`
- `split_data(data)`
- `save_dataframes(out_dir, train_df, test_df)`
- `add_price_category(data)`
- `data_preprocessing(input_path, out_dir)`

**plotting.py**
- `sns_plotting(plot_type, data, x='number_of_reviews', y='price', figlength=14, figheight=7)`
- `build_preprocessor(numerical_data, text_data, categorical_data)`
- `build_clf_model(model, preprocessor, tbl_out_dir, X_train, y_train, X_test, y_test, replacement_dict, clf_report_file_name)`
- `knn_param_optimization(knn_model, tbl_out_dir, X_train, y_train, X_test, y_test, replacement_dict)`


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


