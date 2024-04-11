import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.image as mpimg
import os
#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pynyairbnb.data_preprocessing import create_dir_if_not_exists

def rank_correlations(corr_matrix):
    """Rank the correlations present in the given correlation matrix, excluding self-correlations.

    This function takes a correlation matrix, flattens it to include only unique pairings of variables, and then sorts these pairs by the absolute value of their correlation in descending order. Finally, it filters out repeated correlations to provide a concise list of the top correlations.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        A DataFrame representing a correlation matrix where both rows and columns are variables.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the top 10 unique variable pairings sorted by their absolute correlation value, excluding self-correlations.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1], "C": [1, 3, 2]})
    >>> corr_matrix = data.corr()
    >>> print(rank_correlations(corr_matrix))
    """
    # flattening matrix
    flattened_matrix = corr_matrix.stack().reset_index()
    #renaming columns
    flattened_matrix.columns = ['Variable_1', 'Variable_2', 'Correlation']
    # removing duplicate variable names
    flattened_matrix = flattened_matrix.loc[flattened_matrix['Variable_1'] != flattened_matrix['Variable_2']]
    corr_column = flattened_matrix['Correlation']
    flattened_matrix = flattened_matrix.iloc[abs(corr_column).argsort()[::-1]]
    flattened_matrix = flattened_matrix.loc[flattened_matrix['Correlation'].duplicated()]
    #print(f'Top 10 Variable Correlations: \n{flattened_matrix.head(10)}')
    return flattened_matrix.head(10)

def sns_plotting(plot_type, data, x='number_of_reviews', y='price', figlength=14, figheight=7):
    """
    Generates a plot from the seaborn library with the provided plot type, the x-variable, and the y-variable from provided data.

    This function does so by using matplotlib pyplot subplots and setting the figsize to the provided figlength and figheight.
    
    Parameters:
    -----------
    - plot_type (str): Type of plot to generate - one of 'scatterplot', 'boxplot', 'histplot' or 'heatmap'.
    - data (pd.DataFrame): A DataFrame containing at least 2 columns.
    - x (str): x-variable present in data. (Not applicable for 'heatmap')
    - y (str): y-variable present in data. (Not applicable for 'heatmap')
    - figlength (int): The length of the plotting area in inches.
    - figheight (int): The height of the plotting area in inches.
    
    Returns:
    --------
    - matplotlib.figure.Figure: The matplotlib figure containing the plot.
    
    Examples:
    ---------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> data = pd.DataFrame({"price": [25, 75, 125, 175, 225, 275, 325, 375],
                            "number_of_reviews": [2, 0, 1, 15, 22, 7, 5, 3]})
    >>> figure = sns_plotting('scatterplot', data, x='number_of_reviews', y='price', figlength=12, figheight=6)
    >>> figure.tight_layout
    
    Notes:
    ------
    - The function directly modifies the input DataFrame to what is necessary for the plot.
    """
    if plot_type == "heatmap":

        corr_matrix = data.select_dtypes(include=["int64", "float64"]).corr()
        train_corr = corr_matrix.corr(method = 'pearson')
        mask = np.zeros_like(train_corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(train_corr, mask=mask, vmin=-1, vmax=1, center=0, linewidths=.5, cmap="vlag")
        fig.suptitle('Correlation Heat Map of Numeric Predictors', fontsize=12)

    elif plot_type == "scatterplot":

        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(figlength, figheight))

        sns.scatterplot(data=data, x=x, y=y, hue='room_type', ax=ax[0])
        if x == 'number_of_reviews':
            ax[0].set_title("Price vs Number of Reviews Coloured by Room Type")
            ax[0].set(xlabel="# of Reviews", ylabel="Price")
        elif x == 'reviews_per_month':
            ax[0].set_title("Price vs Reviews Per Month Coloured by Room Type")
            ax[0].set(xlabel="# of Reviews Per Month", ylabel="Price")

        sns.scatterplot(data=data, x=x, y=y, hue='room_type', ax=ax[1])
        ax[1].set_ylim(-100, 5000)
        if x == 'number_of_reviews':
            ax[1].set_title("Price (< 5000) vs Number of Reviews For Clarity")
            ax[1].set(xlabel="# of Reviews", ylabel="Price")
        elif x == 'reviews_per_month':
            ax[1].set_title("Price (< 5000) vs Reviews Per Month For Clarity")
            ax[1].set(xlabel="# of Reviews Per Month", ylabel="Price")

    elif plot_type == "boxplot":

        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(figlength, figheight))
        
        sns.boxplot(data=data, x=x, y=np.log(data[y]), ax=ax[0])
        if x == 'neighbourhood_group':
            ax[0].set_title("Log Price by Neighbourhood Group")
            ax[0].set(xlabel="Neighbourhood Group", ylabel="Log Price")
        elif x == 'room_type':
            ax[0].set_title("Log Price by Room Type")
            ax[0].set(xlabel="Room Type", ylabel="Log Price")

        sns.boxplot(data=data, x=x, y=np.log(data[y]), showfliers=False, ax=ax[1])
        if x == 'neighbourhood_group':
            ax[1].set_title("Price by Neighbourhood Group (Outliers Removed)")
            ax[1].set(xlabel="Neighbourhood Group", ylabel="Price")
        elif x == 'room_type':
            ax[1].set_title("Price by Room Type (Outliers Removed)")
            ax[1].set(xlabel="Room Type", ylabel="Price")
    
    elif plot_type == "histplot":
        Q1 = data[y].quantile(0.25)
        Q3 = data[y].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_train_df = data[(data[y] >= lower_bound) & (data[y] <= upper_bound)]

        fig, ax = plt.subplots(figsize=(figlength, figheight))

        sns.histplot(data=filtered_train_df, x=y)
        fig.suptitle('Price Distribution without Outliers')
        plt.xlabel('Price')
        plt.ylabel('Frequency')

    else: raise Exception("plot_type must be one of scatterplot, boxplot, histplot or heatmap")
    
    return fig

def plot_pynyairbnb(input_file, viz_out_dir, tbl_out_dir):
    """Creates and saves visualizations and tables for pynyairbnb data analysis.

    This function orchestrates the creation of various plots and tables as part of the data analysis process for the pynyairbnb project. It reads the dataset from the given input file, generates specified visualizations (e.g., heatmaps, scatter plots, box plots), and then saves these figures to the specified output directory for visualizations. It also ranks correlations among variables and saves this information as a table in the specified output directory for tables.

    Parameters
    ----------
    input_file : str
        Path to the dataset CSV file.
    viz_out_dir : str
        Output directory to save the generated figures.
    tbl_out_dir : str
        Output directory to save the generated tables.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_pynyairbnb('data/raw/airbnb_data_nyc.csv', 'data/figures', 'data/tables')
    
    Notes
    -----
    - This function assumes that the input file is a CSV containing the required columns for the specified visualizations and tables.
    - The directories specified by `viz_out_dir` and `tbl_out_dir` will be created if they do not already exist, using the `create_dir_if_not_exists` function.
    """
    
    create_dir_if_not_exists(viz_out_dir)
    create_dir_if_not_exists(tbl_out_dir)
    
    data = pd.read_csv(input_file)

    # Fig. 1 Correlation Heat Map of Numerical Predictors

    fig1 = sns_plotting('heatmap', data, 7, 5)
    fig1.tight_layout()
    fig1.savefig(os.path.join(viz_out_dir, 'corr_heat_map.jpg'))

    # Fig. 2 Map with Distribution of Listings by Location and Price

    fig2, ax2 = plt.subplots(nrows=1,ncols=2,figsize=(14, 6))

    vmin, vmax = 0, 400 # Setting color limits to more typical range, e.g., 0 to 400

    ax2[0].scatter(data['longitude'], data['latitude'], c=data['price'], #cmap='viridis',
                s=10, alpha=0.6, vmin=vmin, vmax=vmax)
    fig2.colorbar(plt.scatter(data['longitude'], data['latitude'], c=data['price'], #cmap='viridis',
                s=10, alpha=0.6, vmin=vmin, vmax=vmax), label='Price', ax=ax2[0])
    ax2[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2[0].set_title('Distribution of Listings by Location and Price')
    ax2[0].set(xlabel='Longitude', ylabel='Latitude')

    img_path = 'src/images/New_York_City_Map.jpg'
    try:
        img = mpimg.imread(img_path)

        ax2[1].imshow(img)
        ax2[1].set_title('Map of New York City')
        ax2[1].axis("off")

        fig2.savefig(os.path.join(viz_out_dir, 'listing_locations.jpg'))
    
    except FileNotFoundError:
        print("Image file not found. Please check the file path.")

    # Fig. 3 Price vs Number of Reviews Coloured by Room Type Scatterplot
    
    fig3 = sns_plotting('scatterplot', data, x='number_of_reviews', y='price', figlength=20, figheight=6)
    fig3.tight_layout
    fig3.savefig(os.path.join(viz_out_dir, 'price_vs_reviews.jpg'))

    # Fig. 4 Price vs Reviews Per Month Coloured by Room Type Scatterplot

    fig4 = sns_plotting('scatterplot', data, x='reviews_per_month', y='price', figlength=20, figheight=6)
    fig4.tight_layout
    fig4.savefig(os.path.join(viz_out_dir, 'price_vs_reviews_per_month.jpg'))
    
    # Fig. 5 Log Price and Price by Neighbourhood Group Boxplot

    fig5 = sns_plotting('boxplot', data, x='neighbourhood_group', y='price', figlength=12, figheight=6)
    fig5.tight_layout
    fig5.savefig(os.path.join(viz_out_dir, 'neighbourhood_groups_boxplots.jpg'))

    # Fig. 6 Log Price and Price by Room Type Boxplot

    fig6 = sns_plotting('boxplot', data, x='room_type', y='price', figlength=12, figheight=6)
    fig6.tight_layout
    fig6.savefig(os.path.join(viz_out_dir, 'room_type_boxplots.jpg'))

    # Fig. 7 Price Histogram (Outliers Removed)

    fig7 = sns_plotting('histplot', data, y='price', figlength=7, figheight=5)
    fig7.tight_layout
    fig7.savefig(os.path.join(viz_out_dir, 'price_histogram.jpg'))
    
    # Table 1: Correlations Ranked

    corr_matrix = data.select_dtypes(include=["int64", "float64"]).corr()
    table1 = rank_correlations(corr_matrix)
    table1.to_csv(os.path.join(tbl_out_dir, 'correlations_ranked.csv'), index=False)