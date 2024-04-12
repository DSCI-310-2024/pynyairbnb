#import click
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_dir_if_not_exists(directory):
    """Create the directory if it doesn't exist.

    Checks if a directory exists at the specified path, and if not, it creates the directory along with any necessary parent directories.

    Parameters
    ----------
    directory : str
        The path of the directory to create.

    Returns
    -------
    None

    Examples
    --------
    >>> create_dir_if_not_exists('./data/processed')
    # This will create the 'processed' directory within the 'data' directory if it does not already exist.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

def read_data(data_url, out_dir):
    """Loads and saves the data for this project from a specified URL or file path.

    The function reads a CSV file from the given URL or file path, then saves it to the specified output directory after creating the directory if it does not exist.

    Parameters
    ----------
    data_url : str
        URL or path to the dataset file.
    out_dir : str
        Output directory to save the downloaded data.

    Returns
    -------
    pd.DataFrame
        The DataFrame of the raw data.

    Examples
    --------
    >>> read_data('https://example.com/airbnb_data.csv', './data/raw')
    # Downloads the dataset from the specified URL and saves it as 'airbnb_data_2023.csv' in './data/raw'.
    """
    
    create_dir_if_not_exists(out_dir)

    data = pd.read_csv(data_url)
    data.to_csv(os.path.join(out_dir, 'airbnb_data_2023.csv'), index=False)
    return data

def convert_missing_values(data):
    """Fill missing values in specific columns and convert certain columns to string type.

    Specifically, fills missing values in the 'reviews_per_month' column with 0 and converts 'id' and 'host_id' columns to strings.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame containing the data.

    Returns
    -------
    pd.DataFrame
        The DataFrame after filling missing values and converting column types.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, None],
    ...     'host_id': [None, 2, 3],
    ...     'reviews_per_month': [1.5, None, 2.0]
    ... })
    >>> convert_missing_values(df)
        id host_id  reviews_per_month
    0   1    None                1.5
    1   2       2                0.0
    2  NaN       3                2.0
    """
    
    if data is None or data.empty:
        pass
    else:
        data['id'] = data['id'].astype(str)
        data['host_id'] = data['host_id'].astype(str)
        data['reviews_per_month'] = data['reviews_per_month'].fillna(0)
    return data

def split_data(data):
    """Split the dataset into training and testing subsets.

    The function splits the data into training and testing datasets with a default size of 20% for testing.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be split.

    Returns
    -------
    tuple
        A tuple containing the training and testing DataFrames.

    Examples
    --------
    >>> df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
    >>> train, test = split_data(df)
    >>> print(train.shape, test.shape)
    # Output example: (2, 3) (1, 3), depending on the random split.
    """
    return train_test_split(data, test_size=0.2, shuffle=True)

def save_dataframes(out_dir, train_df, test_df):
    """Save the processed DataFrames to specified directory.

    Saves the training and testing DataFrames to CSV files in the given directory. Additionally, splits the features and labels ('price_category') and saves them separately.

    Parameters
    ----------
    out_dir : str
        The output directory to save the dataframes.
    train_df : pd.DataFrame
        The training dataset DataFrame.
    test_df : pd.DataFrame
        The testing dataset DataFrame.

    Returns
    -------
    None

    Examples
    --------
    >>> train_df = pd.DataFrame({'price': [100, 150], 'price_category': ['100-150', '150-200']})
    >>> test_df = pd.DataFrame({'price': [200], 'price_category': ['200-250']})
    >>> save_dataframes('./data/processed', train_df, test_df)
    # Saves 'train_df.csv', 'test_df.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', and 'y_test.csv' in './data/processed'.
    """
    train_df.to_csv(os.path.join(out_dir, 'train_df.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test_df.csv'), index=False)

    X_train = train_df.drop(['price'], axis=1)
    y_train = train_df['price_category']
    X_test = test_df.drop(['price'], axis=1)
    y_test = test_df['price_category']

    X_train.to_csv(os.path.join(out_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(out_dir, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(out_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)

def add_price_category(data):
    """
    Adds a 'price_category' column to the DataFrame based on predefined price ranges, facilitating easier analysis of data by price segments.
    
    This function categorizes each entry into one of seven price ranges, thereby enabling quick insights into the distribution of prices. The categories are defined as follows: 
    '0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350'. 
    Each range captures the minimum inclusive and maximum exclusive price points for that category, except for the '0-50' range, which is inclusive of 50, and '300-350', which is inclusive of 350.
    
    Parameters:
    -----------
    - data (pd.DataFrame): A DataFrame containing at least a 'price' column with numeric values. The 'price' values should be integers or floats.
    
    Returns:
    --------
    - pd.DataFrame: The modified DataFrame with an additional column 'price_category'. This column contains categorical labels indicating the price range for each row.
    
    Examples:
    ---------
    >>> import pandas as pd
    >>> data = pd.DataFrame({"price": [25, 75, 125, 175, 225, 275, 325, 375]})
    >>> add_price_category(data)
        price price_category
    0     25           0-50
    1     75         50-100
    2    125       100-150
    3    175       150-200
    4    225       200-250
    5    275       250-300
    6    325       300-350
    7    375           350+
    
    Notes:
    ------
    - The function directly modifies the input DataFrame by adding a new column 'price_category'.
    - Prices are categorized based on predefined bins, which are set to be inclusive of the lower bound and exclusive of the upper bound for all categories except for the first ('0-50') and the last ('350+'), which includes all prices above 350.
    - Negative price values are treated as errors in input data and will be categorized into the lowest bin ('0-50'), implying the need for data cleaning if such values exist.
    """
    categories = pd.cut(
        data['price'],
        bins=[-float('inf'), 50, 100, 150, 200, 250, 300, 350, float('inf')],
        labels=['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+'],
        include_lowest=True
    )
    data['price_category'] = categories
    return data

def data_preprocessing(input_path, out_dir):
    """Main function orchestrating the data cleaning and preprocessing.

    Reads data from a specified input path, performs cleaning operations including filling missing values and converting data types, splits the data into training and testing datasets, adds a 'price_category' column based on predefined price ranges, and saves the processed datasets to the specified output directory.

    Parameters
    ----------
    input_path : str
        Path to input data file.
    out_dir : str
        Path to directory where processed files will be saved.

    Returns
    -------
    None

    Examples
    --------
    >>> data_preprocessing('data/raw/airbnb_data_2023.csv', 'data/processed')
    # Reads the raw data, processes it, and saves the processed data into the 'data/processed' directory.
    """
    create_dir_if_not_exists(out_dir)

    data = read_data(input_path, out_dir)
    data = convert_missing_values(data)
    train_df, test_df = split_data(data)
    train_df = add_price_category(train_df)
    test_df = add_price_category(test_df)
    save_dataframes(out_dir, train_df, test_df)