import pandas as pd
import os
import sys
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from src.function_build_preprocessor import build_preprocessor
from pynyairbnb.data_preprocessing import create_dir_if_not_exists

def build_preprocessor(numerical_data, text_data, categorical_data):
    """_summary_
    Builds a preprocessor for numerical, text, and categorical data with the following transformations: Standard scaler, One hot encoding, and count Vectorizer
    Args:
        numerical_data (_type_): numeric data
        text_data (_type_): text data 
        categorical_data (_type_): cateogorical data 

    Returns:
        _type_: _description_
    """
    # Numerical Transformer
    numerical_transformer = StandardScaler()

    # Categorical Transformer
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Text Data Transformer
    text_transformer = CountVectorizer()
    
    # Making Our Preprocessor
    preprocessor = make_column_transformer(
        (numerical_transformer, numerical_data),
        (categorical_transformer, categorical_data),
        (text_transformer, text_data),
        remainder='drop'
    )
    
    return preprocessor

def build_clf_model(model, preprocessor, tbl_out_dir, X_train, y_train, X_test, y_test, replacement_dict, clf_report_file_name):
    """_summary_
    Builds a classification model with X_train, y_train, X_test, y_test and saves the classification report to clf_saved_fp 
    Args:
        model (_type_): classification model specified 
        preprocessor (_type_): preprocessor with data transformations
        tbl_out_dir (_type_): path to save our classification report tables
        X_train (_type_): training data input features
        y_train (_type_): training data target variable
        X_test (_type_): testing data input features
        y_test (_type_): testing data target variable
        replacement_dict (_type_): Dictionary with proper Formatting for classification report 
        clf_saved_fp (_type_): File name to save classification report
    
    Returns:
        Model that has been trained on our training data
        Classification report that is saved to Csv file (output)
    """
    model_pipe = make_pipeline(preprocessor, model)
    model = model_pipe.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    clf_report = classification_report(y_test, predictions, output_dict=True)
    clf_report = {replacement_dict.get(key, key): value for (key, value) in clf_report.items()}
    clf_report = pd.DataFrame(clf_report).transpose()
    
    # Saving table
    clf_report.to_csv(os.path.join(tbl_out_dir, clf_report_file_name))
    
    return model 

    
def knn_param_optimization(knn_model, tbl_out_dir, X_train, y_train, X_test, y_test,replacement_dict):
    """_summary_
    Performs Hyperparameter optimization for KNN model 
    Args:
        knn_model (_type_): KNN Model Built with Sklearn Library 
        tbl_out_dir (_type_): Data Input
        X_train (_type_): training data input features
        y_train (_type_): training data target variable
        X_test (_type_): testing data input features
        y_test (_type_): testing data target variable
        replacement_dict (_type_): Dictionary with proper Formatting for classification report 
        output_file_name (str): The name to save the output as, will be a .csv file 
        
        
    Output:
    Saves output of model to csv file defined in output_file_name
    """
    # KNN Hyperparameter Optimization
    param_dist = {
        'n_neighbors': randint(1, 30),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  
    }

    try:
        rand_search = RandomizedSearchCV(knn_model, param_distributions=param_dist, n_iter=5,  
                                        n_jobs=1,  
                                        scoring='accuracy', cv=3, 
                                        verbose=1, random_state=42, pre_dispatch='2*n_jobs')  
    except Exception as e:
        print(f"Error with rand_search: {e}")
        rand_search = RandomizedSearchCV(knn_model, param_distributions=param_dist, n_iter=5,
                                        scoring='accuracy', cv=3, verbose=1, random_state=42)

    rand_search.fit(X_train, y_train)

    # Classification Report After Hyperparameter Optimization
    rand_search_predictions = rand_search.predict(X_test)
    hyperparam_clf_report = classification_report(y_test, rand_search_predictions, output_dict=True)
    hyperparam_clf_report = dict((replacement_dict[key], value) for (key, value) in hyperparam_clf_report.items() if key in replacement_dict)
    hyperparam_clf_report = pd.DataFrame(hyperparam_clf_report).transpose()

    # Saving table
    hyperparam_clf_report.to_csv(os.path.join(tbl_out_dir, 'hyperparam_classification_report.csv'))


def nyairbnb_analysis(input_dir, tbl_out_dir):
    """Creates model and saves tables to src/tables."""
    
    create_dir_if_not_exists(tbl_out_dir)
    
    replacement_dict = {'0.0':'0-50', '1.0':'50-100', '2.0':'100-150', '3.0':'150-200', '4.0':'200-250','5.0':'250-300',
                '6.0':'300-350', '7.0':'350+', 'accuracy':'accuracy', 'macro avg':'macro avg', 'weighted avg':'weighted avg'}
    
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))

    # numeric data
    numerical_data = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                    'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm']

    # text data
    text_data = "name"

    # Categorical Data
    categorical_data = ['neighbourhood_group', 'neighbourhood', 'room_type']

    # Encoding our y_train & y_test with ordinal encoder
    categories = [['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350+']]
    ordinal_encoder = OrdinalEncoder(categories=categories)
    y_train_encoded = ordinal_encoder.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_encoded = ordinal_encoder.transform(y_test.values.reshape(-1, 1)).ravel()
    
    # Making Our Preprocessor
    preprocessor = build_preprocessor(numerical_data=numerical_data, text_data=text_data, categorical_data=categorical_data)
    

    # Implementing a Dummy Regressor model as a baseline to assess our model with and saving results to csv file
    try:
        build_clf_model(model=DummyClassifier(), preprocessor=preprocessor, tbl_out_dir=tbl_out_dir, X_train=X_train,
                    y_train=y_train_encoded, X_test=X_test, y_test=y_test_encoded, replacement_dict=replacement_dict, clf_report_file_name='dummy_classification_report.csv')
    except Exception as e:
        print(f"Error creating dummy model: {e}")

    # implementing knn model and saving results to a csv file
    try:
        knn_model = build_clf_model(model = KNeighborsClassifier(), preprocessor=preprocessor, tbl_out_dir=tbl_out_dir, X_train=X_train, 
                    y_train=y_train_encoded, X_test=X_test, y_test=y_test_encoded,replacement_dict=replacement_dict, clf_report_file_name='knn_classification_report.csv')
    except Exception as e:
        print(f"Error creating KNN model: {e}")


    # performing hyperparameter optimization with our knn model and saving classification report results to csv
    try:
        knn_param_optimization(knn_model=knn_model, tbl_out_dir=tbl_out_dir, X_train=X_train, y_train=y_train_encoded, X_test=X_test, 
                            y_test=y_test_encoded,replacement_dict=replacement_dict)
    except Exception as e: 
        print(f'Error performing hyperparameter optimization: {e}') 