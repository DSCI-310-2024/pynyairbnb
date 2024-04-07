#import click
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_dir_if_not_exists(directory):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_data(data_url, out_dir):
    """Loads and saves the data for this project."""
    
    create_dir_if_not_exists(out_dir)

    data = pd.read_csv(data_url)
    data.to_csv(os.path.join(out_dir, 'airbnb_data_2023.csv'), index=False)

def convert_missing_values(data):
    """Fill missing values and convert columns to string."""
    data['id'] = data['id'].astype(str)
    data['host_id'] = data['host_id'].astype(str)
    data['reviews_per_month'] = data['reviews_per_month'].fillna(0)
    return data

def split_data(data):
    """Split the data into training and testing datasets."""
    return train_test_split(data, test_size=0.2, shuffle=True)

def save_dataframes(out_dir, train_df, test_df):
    """Save the processed dataframes to the output directory."""
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

# @click.command()
# @click.option('--input_path', type=str, default="data/raw/airbnb_data_2023.csv", help='Path to input data')
# @click.option('--out_dir', type=str, default="data/cleaned", help='Path to write the file')
def data_preprocessing(input_path, out_dir):
    """Main function orchestrating the data cleaning and preprocessing."""
    create_dir_if_not_exists(out_dir)

    data = read_data(input_path)
    data = convert_missing_values(data)
    train_df, test_df = split_data(data)
    train_df = add_price_category(train_df)
    test_df = add_price_category(test_df)
    save_dataframes(out_dir, train_df, test_df)