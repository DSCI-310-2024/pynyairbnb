import pandas as pd
import matplotlib
import pytest
from pynyairbnb.plotting import sns_plotting

# Same Data to be used for all tests
data = pd.DataFrame({"price": [25, 75, 125, 175, 225, 275, 325, 375],
                    "number_of_reviews": [2, 0, 1, 15, 22, 7, 5, 3],
                    "reviews_per_month": [0, 3, 4, 1, 2, 0, 3, 4],
                    "room_type": ["cat2", "cat1", "cat5", "cat4", "cat3", "cat6", "cat8", "cat7"]})

# Check to see if output type is correct given correct inputs
def test_sns_plotting_output_type():
    """_summary_
    Ensures that the output of the sns plotting function is a matplotlib figure 
    """
    result = sns_plotting('scatterplot', data, 'number_of_reviews', 'price', 20, 10)
    assert type(result) == matplotlib.figure.Figure, "Test failed: Output type is incorrect."

# Check to see if exception raised for n/a plot type
def test_sns_plotting_plottype_error():
    """_summary_
    Ensures that running the sns plotting function with the incorrect inputs leads to an error
    """
    with pytest.raises(Exception):
        result = sns_plotting('barplot', data, 'number_of_reviews', 'price', 20, 10)

# Check to see if value error raised for x-variable not in data
def test_sns_plotting_x_error():
    """_summary_
    Ensures that running the sns plotting function with the incorrect inputs leads to an error
    """
    with pytest.raises(ValueError):
        sns_plotting('scatterplot', data, 'random_x', 'price', 20, 10)

# Check to see if value error raised for y-variable not in data
def test_sns_plotting_y_error():
    """_summary_
    Ensures that running the sns plotting function with the incorrect inputs leads to an error
    """
    with pytest.raises(ValueError):
        sns_plotting('scatterplot', data, 'number_of_reviews', 'random_y', 20, 10)

# Check to see the figlength and figheight are both <= 25 to avoid being too large
def test_sns_plotting_figsize_check():
    """_summary_
    Ensures that the figure size of the sns plot is of a reasonable size (<= 25 inches)
    """
    result = sns_plotting('scatterplot', data, 'number_of_reviews', 'price')
    assert result.get_size_inches()[0] <= 25 and result.get_size_inches()[1] <= 25, "Test failed: Plot size is too large."