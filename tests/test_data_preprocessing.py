import os
import pandas as pd
import pytest
import warnings
from click.testing import CliRunner
from pynyairbnb.data_preprocessing import create_dir_if_not_exists, read_data, convert_missing_values, split_data, save_dataframes, add_price_category, data_preprocessing

def test_create_dir_if_not_exists(tmpdir):
    """
    Test whether a new directory is created when it does not already exist.
    
    Args:
    tmpdir (py.path.local): A pytest fixture that provides a temporary directory unique to the test invocation.
    
    Asserts:
    That the directory is actually created and exists on the filesystem.
    """
    # tmpdir is a pytest fixture that provides a temporary directory unique to the test invocation
    new_dir = tmpdir.mkdir("sub").join("newdir")
    create_dir_if_not_exists(str(new_dir))
    assert os.path.exists(new_dir)

def test_read_data(tmpdir):
    """
    Ensure that data is correctly downloaded and saved to a local directory from a specified URL.
    
    Args:
    tmpdir (py.path.local): Temporary directory provided by pytest for file operations.
    
    Asserts:
    That the file is saved to the expected path.
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

def test_convert_missing_values():
    """
    Test the conversion of missing values in a DataFrame, including the transformation of numerical IDs to string type and the handling of NaNs.
    
    Asserts:
    That no NaN values remain in the DataFrame and that ID columns are correctly converted to string type.
    """
    df = pd.DataFrame({'id': [1, None], 'host_id': [None, 2], 'reviews_per_month': [3.0, None]})
    converted_df = convert_missing_values(df)
    assert converted_df.isnull().sum().sum() == 0  # No NaN values
    assert all(isinstance(x, str) for x in converted_df['id'])
    assert all(isinstance(x, str) for x in converted_df['host_id'])
    assert converted_df['reviews_per_month'].dtype == float

def test_split_data():
    """
    Validate the functionality of splitting a DataFrame into training and testing subsets.
    
    Asserts:
    That both train and test DataFrames are non-empty and together account for all the data in the original DataFrame.
    """
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    train, test = split_data(df)
    assert len(train) > 0 and len(test) > 0  # Ensure non-empty splits
    assert len(train) + len(test) == len(df)  # Ensure complete data coverage

def test_save_dataframes(tmpdir):
    """
    Test the functionality to save multiple DataFrame splits to specified files within a directory.
    
    Args:
    tmpdir (py.path.local): Temporary directory for saving files.
    
    Asserts:
    That all specified files are correctly saved in the directory.
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
    Test the addition of a 'price_category' column to a DataFrame based on predefined price ranges.
    
    Asserts:
    That the 'price_category' column is added and contains correct categorical labels based on the price values.
    """
    df = pd.DataFrame({'price': [25, 75, 125, 175, 225, 275, 325, 375]})
    expected_categories = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+']
    result_df = add_price_category(df)
    
    assert 'price_category' in result_df.columns, "price_category column not added"
    assert all(result_df['price_category'] == expected_categories), "Price categories do not match expected values"
    
def test_add_price_category_spanning_all_ranges():
    """
    Test the functionality of the add_price_category function to ensure it correctly assigns price categories 
    that span across all predefined ranges from negative values to above the highest specified bin.

    Asserts:
    That the 'price_category' column is accurately categorized for a range of values covering the entire spectrum 
    from negative to above the maximum set bin.
    """
    data = pd.DataFrame({'price': [-10, 25, 75, 125, 175, 225, 275, 325, 375]})
    result = add_price_category(data)
    expected_categories = ['0-50', '0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+']
    assert all(result['price_category'] == expected_categories), "Test failed: Prices spanning all ranges are not categorized correctly."

def test_add_price_category_single_category():
    """
    Verify that the add_price_category function correctly categorizes multiple entries within the same price range.

    Asserts:
    That all prices within a narrow range are consistently categorized and match the expected categorical data type.
    """
    data = pd.DataFrame({'price': [100, 105, 110]})
    result = add_price_category(data)
    # Ensure the expected_series has the same categories and order as the result
    expected_series = pd.Series(['50-100', '100-150', '100-150'], name='price_category')
    expected_dtype = pd.CategoricalDtype(categories=['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+'], ordered=True)
    expected_series = expected_series.astype(expected_dtype)
    pd.testing.assert_series_equal(result['price_category'], expected_series, check_categorical=True)

def test_add_price_category_empty_dataframe():
    """
    Ensure that the add_price_category function handles an empty DataFrame without errors and returns an empty DataFrame.

    Asserts:
    That an empty input DataFrame results in an empty output DataFrame.
    """
    data = pd.DataFrame({'price': []})
    result = add_price_category(data)
    assert result.empty, "Test failed: The function should return an empty DataFrame when provided with one."


def test_add_price_category_with_negative_prices():
    """
    Test the add_price_category function with negative price values to ensure they are categorized in the lowest price range.

    Asserts:
    That negative prices are treated as the lowest category, ensuring robustness in categorization logic.
    """
    data = pd.DataFrame({'price': [-1, -20]})
    result = add_price_category(data)
    assert all(result['price_category'] == ['0-50', '0-50']), "Test failed: Negative prices are not categorized correctly."

def test_add_price_category_with_floats():
    """
    Verify that the add_price_category function accurately categorizes prices expressed as floats into the correct price ranges.

    Asserts:
    That prices just below and just above price range boundaries are categorized correctly, demonstrating precision in boundary conditions.
    """
    data = pd.DataFrame({'price': [49.99, 100.01]})
    result = add_price_category(data)
    assert all(result['price_category'] == ['0-50', '100-150']), "Test failed: Float prices are not categorized correctly."

def test_add_price_category_with_boundary_prices():
    """
    Confirm that the add_price_category function correctly categorizes prices at the exact boundary points of the predefined price ranges.

    Asserts:
    That boundary prices are included in the correct categories, important for ensuring accuracy at range transitions.
    """
    data = pd.DataFrame({'price': [50, 100, 150, 200, 250, 300, 350]})
    result = add_price_category(data)
    expected_categories = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350']
    assert all(result['price_category'] == expected_categories), "Test failed: Boundary prices are not categorized correctly."

def test_add_price_category_preserves_dtype():
    """
    Test that the add_price_category function preserves the original data types of existing columns when adding a new category column.

    Asserts:
    That no data type alterations occur in the original DataFrame columns after adding the category, ensuring data integrity.
    """
    data = pd.DataFrame({'price': [25, 75], 'other_column': [1, 2]})
    original_dtype = data.dtypes
    result = add_price_category(data)
    assert data.drop(columns=['price_category']).dtypes.equals(original_dtype), "Test failed: Original data types are altered."

def test_invalid_data_formats(mock_data, tmp_path):
    """
    Test the data preprocessing tool's ability to handle data frames with invalid data formats, ensuring robust error handling.

    Asserts:
    That the system appropriately fails when encountering data that cannot be properly formatted or processed.
    """
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
    """
    Test the logical consistency of the price categorization in the add_price_category function with a predefined set of price inputs.

    Asserts:
    That the price categories assigned are consistent with expected output for a given range of test prices.
    """
    mock_df = pd.DataFrame({
        'price': [75, 150, 225, 300]
    })
    expected_categories = ['50-100', '100-150', '200-250', '250-300']
    categorized_df = add_price_category(mock_df)
    assert all(categorized_df['price_category'] == expected_categories)

    
def test_data_splitting_proportions(mock_data):
    """
    Verify that the data splitting function accurately splits data into training and testing datasets according to specified proportions.

    Asserts:
    That the proportions of the split approximately match the expected 80-20 training-testing ratio, validating the splitting logic.
    """
    train_df, test_df = split_data(mock_data)
    total_len = len(mock_data)
    train_len = len(train_df)
    test_len = len(test_df)
    # Check if the proportions approximately match the expected 80-20 split
    assert train_len / total_len == pytest.approx(0.8, 0.05)
    assert test_len / total_len == pytest.approx(0.2, 0.05)


def test_convert_missing_values_with_none():
    """Test convert_missing_values with None as input."""
    result = convert_missing_values(None)
    assert result is None, "Function should return None for None input."

def test_convert_missing_values_with_empty_dataframe():
    """Test convert_missing_values with an empty DataFrame as input."""
    empty_df = pd.DataFrame()
    result = convert_missing_values(empty_df)
    assert result.empty, "Function should return an empty DataFrame for empty DataFrame input."

@pytest.fixture
def setup_mocks(mocker, mock_data):
    """
    Fixture function that sets up the necessary mocks for data preprocessing tests.

    Parameters:
    - mocker: The pytest-mock mocker fixture.
    - mock_data: The mock data to be used for testing.

    Returns:
    - A dictionary containing the mock objects for various functions involved in data preprocessing.
    """
    mock_create_dir = mocker.patch('pynyairbnb.data_preprocessing.create_dir_if_not_exists')
    mock_read_data = mocker.patch('pynyairbnb.data_preprocessing.read_data', return_value=mock_data)
    mock_convert_missing = mocker.patch('pynyairbnb.data_preprocessing.convert_missing_values', return_value=mock_data)
    mock_split_data = mocker.patch('pynyairbnb.data_preprocessing.split_data', return_value=(mock_data, mock_data))
    mock_add_price_category = mocker.patch('pynyairbnb.data_preprocessing.add_price_category', side_effect=lambda x: x)
    mock_save_dataframes = mocker.patch('pynyairbnb.data_preprocessing.save_dataframes')
    
    return {
        'mock_create_dir': mock_create_dir,
        'mock_read_data': mock_read_data,
        'mock_convert_missing': mock_convert_missing,
        'mock_split_data': mock_split_data,
        'mock_add_price_category': mock_add_price_category,
        'mock_save_dataframes': mock_save_dataframes
    }

def test_data_preprocessing(setup_mocks):
    """Tests the orchestration of the data preprocessing pipeline."""
    # Execute the function
    data_preprocessing('dummy/path/to/data.csv', 'dummy/path/to/output', 'dummy/path/to/raw')
    # Verify all steps are called correctly
    setup_mocks['mock_create_dir'].assert_called_once_with('dummy/path/to/output')
    setup_mocks['mock_read_data'].assert_called_once_with('dummy/path/to/data.csv', 'dummy/path/to/raw')
    setup_mocks['mock_convert_missing'].assert_called_once()
    setup_mocks['mock_split_data'].assert_called_once()
    setup_mocks['mock_save_dataframes'].assert_called_once()