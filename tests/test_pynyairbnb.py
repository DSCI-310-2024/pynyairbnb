import sys
import os
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from pynyairbnb.pynyairbnb import build_preprocessor, build_clf_model, knn_param_optimization, nyairbnb_analysis


def test_preprocessor_type():
    """Test that the function returns a ColumnTransformer object."""
    numerical_data = ['numerical']
    text_data = 'text'
    categorical_data = ['category']
    
    preprocessor = build_preprocessor(numerical_data, text_data, categorical_data)
    assert isinstance(preprocessor, ColumnTransformer)

def test_transformer_assignment():
    """Test that the correct transformers are assigned to the specified types of data."""
    preprocessor = build_preprocessor(['num'], ['text'], ['cat'])
    transformers = {name: type(trans) for name, trans, cols in preprocessor.transformers}
    
    assert transformers.get('standardscaler') == StandardScaler
    assert transformers.get('onehotencoder') == OneHotEncoder
    assert transformers.get('countvectorizer') == CountVectorizer

def test_preprocessor():
    """_summary_
    Tests the functionality of build preprocessor method
    
    """
    # Create an artificial dataset
    np.random.seed(0)  # For reproducibility
    data = pd.DataFrame({
        'numerical': np.random.randn(100),
        'text': np.random.choice(['First text', 'Text number 2', 'Third sentence of text'], size=100),
        'category': np.random.choice(['A', 'B', 'C'], size=100),
        'target': np.array([0]*90 + [1]*10)  # target variable made with 90 zeros and 10 1s, meaning dummy classifier should predict 0 everytime
    })
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('target', axis=1), data['target'], test_size=0.25, random_state=42)
    
    # Define columns
    numerical_data = ['numerical']
    text_data = 'text'
    categorical_data = ['category']
    
    # Build the preprocessor
    preprocessor = build_preprocessor(numerical_data, text_data, categorical_data)
    
    # combine preprocessor with a dummymodel with strategy most frequent
    model = make_pipeline(preprocessor, DummyClassifier(strategy='most_frequent'))
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Dummy Classifier Accuracy: {accuracy:.4f}")
    
    assert accuracy >= 0.5, "Accuracy should be at least 0.5"
    
X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def test_build_clf_model():
    """_summary_
    Tests the build clf model and ensures output file is correctly created
    """
    preprocessor = StandardScaler()
    model = KNeighborsClassifier()
    tbl_out_dir = "./test_outputs"
    os.makedirs(tbl_out_dir, exist_ok=True)
    replacement_dict = {'0': 'Class 0', '1': 'Class 1'}
    clf_report_file_name = "test_clf_report.csv"
    

    clf_model = build_clf_model(model, preprocessor, tbl_out_dir, X_train, y_train, X_test, y_test, replacement_dict, clf_report_file_name)
    

    assert clf_model is not None

    assert os.path.isfile(os.path.join(tbl_out_dir, clf_report_file_name))
    

    os.remove(os.path.join(tbl_out_dir, clf_report_file_name))
    os.rmdir(tbl_out_dir)