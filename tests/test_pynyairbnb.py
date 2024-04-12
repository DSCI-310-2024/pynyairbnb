import sys
import os
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from tempfile import TemporaryDirectory
import numpy as np
from sklearn.datasets import load_iris
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
    
def test_clf_model_inputs():
    """_summary_
    Ensures that running the build_clf function function with the incorrect inputs leads to a type error
    """
    with pytest.raises(TypeError):
        clf_model = build_clf_model("faulty_value", "faulty_input", "faulty_input", X_train, y_train, X_test, y_test, "test", "test_name")
        
def test_empty_datasets():
    """_summary_
    Ensures that if model is run with empty data sets, it raises a value error
    """
    with TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            build_clf_model(KNeighborsClassifier(), StandardScaler(), temp_dir, 
                            pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series(),
                            {}, 'test_report.csv')

def test_invalid_data_types():
    """_summary_
    Tests that model runs into value error if invalid datatypes are entered
    """
    with TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            build_clf_model(KNeighborsClassifier(), StandardScaler(), temp_dir, 
                            [0, 1], [0, 1], [1, 0], [1, 0],
                            {}, 'test_report.csv')

def test_data_with_missing_values():
    """_summary_
    Ensures function raises value error for missing values 
    """
    X_train = pd.DataFrame({'feature1': [np.nan, 1], 'feature2': [1, 0]})
    y_train = pd.Series([0, 1])
    X_test = pd.DataFrame({'feature1': [1, 0], 'feature2': [0, 1]})
    y_test = pd.Series([1, 0])
    with TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            build_clf_model(KNeighborsClassifier(), StandardScaler(), temp_dir, 
                            X_train, y_train, X_test, y_test,
                            {}, 'test_report.csv')

def test_invalid_path():
    """_summary_
    Ensures value error is raised if an invalid file path is input 
    """
    with pytest.raises(ValueError):
        build_clf_model(KNeighborsClassifier(), StandardScaler(), '/invalid/path', 
                        pd.DataFrame({'feature1': [0, 1], 'feature2': [1, 0]}), pd.Series([0, 1]),
                        pd.DataFrame({'feature1': [1, 0], 'feature2': [0, 1]}), pd.Series([1, 0]),
                        {}, 'test_report.csv')

@pytest.fixture
def iris_data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_knn_param_optimization(iris_data, tmpdir):
    X_train, X_test, y_train, y_test = iris_data
    knn_model = KNeighborsClassifier()
    output_dir = str(tmpdir.mkdir("output"))  # Converting to string because tmpdir is a py.path
    try:
        knn_param_optimization(knn_model, output_dir, X_train, y_train, X_test, y_test, {})
        assert os.path.isfile(os.path.join(output_dir, "hyperparam_classification_report.csv"))
    except Exception as e:
        pytest.fail(f"knn_param_optimization raised an unexpected exception: {e}")

# Define test data directory
TEST_DATA_DIR = 'docs/sample_data'

@pytest.fixture
def input_data(tmpdir):
    """Fixture to provide test input data."""
    # Create temporary directory
    tmp_input_dir = tmpdir.mkdir('input')
    
    # Copy test data files to temporary directory
    test_data_files = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    for file in test_data_files:
        src_path = os.path.join(TEST_DATA_DIR, file)
        dst_path = os.path.join(tmp_input_dir, file)
        os.system(f'cp {src_path} {dst_path}')
    
    return tmp_input_dir

@pytest.fixture
def input_data(tmpdir):
    """Fixture to provide test input data."""
    # Create temporary directory
    tmp_input_dir = tmpdir.mkdir('input')
    
    # Copy test data files to temporary directory
    test_data_files = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    for file in test_data_files:
        src_path = os.path.join(TEST_DATA_DIR, file)
        dst_path = os.path.join(tmp_input_dir, file)
        os.system(f'cp {src_path} {dst_path}')
    
    return tmp_input_dir