import os
import pandas as pd
import sys
import pytest
import warnings
from click.testing import CliRunner
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pynyairbnb.data_preprocessing import create_dir_if_not_exists, read_data, convert_missing_values, split_data, save_dataframes, add_price_category, data_preprocessing

def test_create_dir_if_not_exists(tmpdir):
    """
    Test that a directory is created if it doesn't exist.
    """
    # tmpdir is a pytest fixture that provides a temporary directory unique to the test invocation
    new_dir = tmpdir.mkdir("sub").join("newdir")
    create_dir_if_not_exists(str(new_dir))
    assert os.path.exists(new_dir)

def test_read_data(tmpdir):
    """
    Test that data is correctly read and saved.
    """
    # Prepare
    out_dir = tmpdir
    data_url = 'http://data.insideairbnb.com/united-states/ny/new-york-city/2023-12-04/visualisations/listings.csv'  
    # Execute
    read_data(data_url, str(out_dir))
    # Validate
    saved_file = os.path.join(out_dir, 'airbnb_data_2023.csv')
    assert os.path.exists(saved_file)  # More checks can be added here

@pytest.fixture
def mock_data():
    return pd.DataFrame({
        'id': range(10),  # 10 samples
        'host_id': range(10, 20),
        'price': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'reviews_per_month': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'price_category': ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+', '350+', '350+']
    })

# def test_preprocess_data(mock_data, tmpdir):
#     csv_path = os.path.join(tmpdir, "mock_data.csv")
#     mock_data.to_csv(csv_path, index=False)
#     processed_data = data_preprocessing(csv_path, tmpdir)
#     assert all(processed_data['id'].apply(lambda x: isinstance(x, str)))
#     assert all(processed_data['host_id'].apply(lambda x: isinstance(x, str)))
#     assert 'reviews_per_month' in processed_data.columns

def test_convert_missing_values():
    """
    Test that missing values are correctly converted.
    """
    df = pd.DataFrame({'id': [1, None], 'host_id': [None, 2], 'reviews_per_month': [3.0, None]})
    converted_df = convert_missing_values(df)
    assert converted_df.isnull().sum().sum() == 0  # No NaN values
    assert all(isinstance(x, str) for x in converted_df['id'])
    assert all(isinstance(x, str) for x in converted_df['host_id'])
    assert converted_df['reviews_per_month'].dtype == float

def test_split_data():
    """
    Test data splitting functionality.
    """
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    train, test = split_data(df)
    assert len(train) > 0 and len(test) > 0  # Ensure non-empty splits
    assert len(train) + len(test) == len(df)  # Ensure complete data coverage

def test_save_dataframes(tmpdir):
    """
    Test that dataframes are correctly saved to specified directory.
    """
    # Prepare
    out_dir = tmpdir
    train_df = pd.DataFrame({'price': [100, 150], 'price_category': ['100-150', '150-200']})
    test_df = pd.DataFrame({'price': [200], 'price_category': ['200-250']})

    # Execute
    save_dataframes(str(out_dir), train_df, test_df)

    # Validate
    for filename in ['train_df.csv', 'test_df.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']:
        saved_file = os.path.join(str(out_dir), filename)
        assert os.path.exists(saved_file), f"{filename} was not saved correctly"

def test_add_price_category():
    """
    Test that price categories are correctly added to dataframe.
    """
    df = pd.DataFrame({'price': [25, 75, 125, 175, 225, 275, 325, 375]})
    expected_categories = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+']
    result_df = add_price_category(df)
    
    assert 'price_category' in result_df.columns, "price_category column not added"
    assert all(result_df['price_category'] == expected_categories), "Price categories do not match expected values"
def test_add_price_category_spanning_all_ranges():
    data = pd.DataFrame({'price': [-10, 25, 75, 125, 175, 225, 275, 325, 375]})
    result = add_price_category(data)
    expected_categories = ['0-50', '0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+']
    assert all(result['price_category'] == expected_categories), "Test failed: Prices spanning all ranges are not categorized correctly."

def test_add_price_category_single_category():
    data = pd.DataFrame({'price': [100, 105, 110]})
    result = add_price_category(data)
    # Ensure the expected_series has the same categories and order as the result
    expected_series = pd.Series(['50-100', '100-150', '100-150'], name='price_category')
    expected_dtype = pd.CategoricalDtype(categories=['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+'], ordered=True)
    expected_series = expected_series.astype(expected_dtype)
    pd.testing.assert_series_equal(result['price_category'], expected_series, check_categorical=True)

def test_add_price_category_empty_dataframe():
    data = pd.DataFrame({'price': []})
    result = add_price_category(data)
    assert result.empty, "Test failed: The function should return an empty DataFrame when provided with one."


def test_add_price_category_with_negative_prices():
    data = pd.DataFrame({'price': [-1, -20]})
    result = add_price_category(data)
    assert all(result['price_category'] == ['0-50', '0-50']), "Test failed: Negative prices are not categorized correctly."

def test_add_price_category_with_floats():
    data = pd.DataFrame({'price': [49.99, 100.01]})
    result = add_price_category(data)
    assert all(result['price_category'] == ['0-50', '100-150']), "Test failed: Float prices are not categorized correctly."

def test_add_price_category_with_boundary_prices():
    data = pd.DataFrame({'price': [50, 100, 150, 200, 250, 300, 350]})
    result = add_price_category(data)
    expected_categories = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350']
    assert all(result['price_category'] == expected_categories), "Test failed: Boundary prices are not categorized correctly."

def test_add_price_category_preserves_dtype():
    data = pd.DataFrame({'price': [25, 75], 'other_column': [1, 2]})
    original_dtype = data.dtypes
    result = add_price_category(data)
    assert data.drop(columns=['price_category']).dtypes.equals(original_dtype), "Test failed: Original data types are altered."

def test_invalid_data_formats(mock_data, tmp_path):
    with warnings.catch_warnings():
        # Filter out the specific UserWarning
        warnings.filterwarnings("ignore", message="Pandas doesn't allow columns to be created via a new attribute name", category=UserWarning)

        mock_data.return_value = pd.DataFrame({
            'id': ['one', 'two'],
            'host_id': [3, 4],
            'price': ['a hundred', 'two hundred'],
            'reviews_per_month': [1.0, 'two']
        })
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(data_preprocessing, ['--input_path', 'dummy.csv', '--out_dir', str(tmp_path)], prog_name="data_preprocessing")
            assert result.exit_code != 0


def test_price_categorization_logic():
    mock_df = pd.DataFrame({
        'price': [75, 150, 225, 300]
    })
    expected_categories = ['50-100', '100-150', '200-250', '250-300']
    categorized_df = add_price_category(mock_df)
    assert all(categorized_df['price_category'] == expected_categories)

    
def test_data_splitting_proportions(mock_data):
    train_df, test_df = split_data(mock_data)
    total_len = len(mock_data)
    train_len = len(train_df)
    test_len = len(test_df)
    # Check if the proportions approximately match the expected 80-20 split
    assert train_len / total_len == pytest.approx(0.8, 0.05)
    assert test_len / total_len == pytest.approx(0.2, 0.05)

# def test_data_preprocessing(tmpdir):
#     """
#     Test the end-to-end data preprocessing functionality.
#     """
#     # Setup - Create a sample CSV file in tmpdir
#     sample_data = pd.DataFrame({'price': [25, 75, 375], 'reviews_per_month': [1, 2, None]})
#     sample_data_path = tmpdir.join("sample_data.csv")
#     sample_data.to_csv(str(sample_data_path), index=False)
    
#     processed_data_dir = tmpdir
#     input_path = str(sample_data_path)
#     out_dir = str(processed_data_dir)

#     # Execute
#     data_preprocessing(input_path, out_dir)

#     # Validate - Check if processed files exist
#     for filename in ['train_df.csv', 'test_df.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']:
#         saved_file = os.path.join(out_dir, filename)
#         assert os.path.exists(saved_file), f"{filename} was not processed or saved correctly"
