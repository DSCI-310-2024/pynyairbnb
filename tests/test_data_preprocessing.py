import os
import pandas as pd
import sys
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
