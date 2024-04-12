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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from pynyairbnb.pynyairbnb import build_preprocessor, build_clf_model, knn_param_optimization, nyairbnb_analysis, create_dir_if_not_exists


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
def setup_knn():
    """Fixture to setup the KNN model and test data."""
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[1.5], [2.5]])
    y_test = np.array([0, 1])
    knn_model = KNeighborsClassifier()
    tbl_out_dir = "./"
    replacement_dict = {'0': 'Class 0', '1': 'Class 1'}
    return knn_model, tbl_out_dir, X_train, y_train, X_test, y_test, replacement_dict

def test_knn_param_optimization(mocker, setup_knn):
    knn_model, tbl_out_dir, X_train, y_train, X_test, y_test, replacement_dict = setup_knn

    # Use real RandomizedSearchCV to ensure proper behavior
    rand_search_instance = RandomizedSearchCV(knn_model, {})
    mocker.patch.object(RandomizedSearchCV, 'fit', return_value=None)
    mocker.patch.object(RandomizedSearchCV, 'predict', return_value=y_test)

    # Use a real DataFrame for outputs
    classification_report_dict = {'0': {'precision': 1.0}, '1': {'precision': 1.0}}
    classification_report_df = pd.DataFrame(classification_report_dict).transpose()
    mocker.patch('sklearn.metrics.classification_report', return_value=classification_report_dict)

    # Execute the function
    knn_param_optimization(knn_model, tbl_out_dir, X_train, y_train, X_test, y_test, replacement_dict)

    # No exception should occur; ensure file writing occurs as expected
    assert os.path.isfile(os.path.join(tbl_out_dir, 'hyperparam_classification_report.csv'))

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

def test_classification_report_save_failure(mocker):
    """
    Test case to verify that an exception is raised when saving the classification report fails due to a permission error.
    """
    mocker.patch('pandas.DataFrame.to_csv', side_effect=PermissionError("Permission denied"))
    
    with pytest.raises(PermissionError) as excinfo:
        build_clf_model(KNeighborsClassifier(), StandardScaler(), '/protected/path',
                        X_train, y_train, X_test, y_test, {}, 'test_report.csv')
    assert "Permission denied" in str(excinfo.value)

def test_with_corrupted_csv_data(mocker):
    """
    Test case to check the behavior of nyairbnb_analysis function when the CSV data is corrupted.

    Args:
        mocker: The mocker object used for patching the pandas.read_csv function.

    Raises:
        pd.errors.ParserError: If the CSV data is corrupted.

    Returns:
        None
    """
    mocker.patch('pandas.read_csv', side_effect=pd.errors.ParserError("Corrupted data"))

    with pytest.raises(pd.errors.ParserError) as excinfo:
        nyairbnb_analysis('input_dir', 'output_dir')
    assert "Corrupted data" in str(excinfo.value)

def test_directory_creation_failure(mocker):
    """
    Test case to verify the handling of directory creation failure.

    This test mocks the 'os.makedirs' function to raise a 'PermissionError' exception
    with the message 'Permission denied'. It then calls the 'create_dir_if_not_exists'
    function with a protected path and asserts that the raised exception contains the
    expected error message.

    Args:
        mocker: The mocker object from the pytest-mock library.

    Raises:
        AssertionError: If the directory creation permission error is not handled correctly.

    """
    mocker.patch('os.makedirs', side_effect=PermissionError("Permission denied"))
    with pytest.raises(PermissionError) as excinfo:
        create_dir_if_not_exists('/protected/path')
    assert "Permission denied" in str(excinfo.value), "Did not handle directory creation permission error correctly"

def test_model_training_failure(mocker):
    """
    Test case to verify that model training failure is handled correctly.

    Args:
        mocker: The mocker object used for patching the 'fit' method of sklearn.pipeline.Pipeline.

    Raises:
        Exception: If the model training fails.

    Returns:
        None
    """
    mocker.patch('sklearn.pipeline.Pipeline.fit', side_effect=Exception("Training failed"))
    with pytest.raises(Exception) as excinfo:
        build_clf_model(KNeighborsClassifier(), make_pipeline(StandardScaler()), 'output_dir',
                        X_train, y_train, X_test, y_test, {}, 'test_report.csv')
    assert "Training failed" in str(excinfo.value), "Model training failure not handled"

def test_invalid_input_path(mocker):
    """
    Test case to verify the behavior when an invalid input path is provided.
    It mocks the 'pandas.read_csv' function to raise a FileNotFoundError and checks if the correct exception is raised.

    Args:
        mocker: The mocker object used for mocking the 'pandas.read_csv' function.

    Raises:
        FileNotFoundError: If the input file path does not exist.

    Returns:
        None
    """
    mocker.patch('pandas.read_csv', side_effect=FileNotFoundError("File not found"))
    with pytest.raises(FileNotFoundError) as excinfo:
        nyairbnb_analysis('/non/existent/path', 'output_dir')
    assert "File not found" in str(excinfo.value), "Did not handle non-existent file path correctly"

def test_knn_param_optimization_success(mocker):
    """
    Test case for successful parameter optimization of KNN model.

    Args:
        mocker: The mocker object for patching methods.

    Raises:
        AssertionError: If an unexpected failure occurs during parameter optimization.

    """
    random_predictions = np.random.randint(0, 2, size=len(y_test))
    mocker.patch('sklearn.model_selection.RandomizedSearchCV.fit', return_value=None)
    mocker.patch('sklearn.model_selection.RandomizedSearchCV.predict', return_value=random_predictions)

    try:
        knn_model = KNeighborsClassifier()
        knn_param_optimization(knn_model, 'output_dir', X_train, y_train, X_test, y_test, {})
    except Exception as e:
        pytest.fail(f"Unexpected failure during parameter optimization: {e}")

def test_classification_report_output(mocker):
    """
    Test function to verify if the classification report is outputted correctly.

    Args:
        mocker: The mocker object used for patching the 'pandas.DataFrame.to_csv' method.

    Returns:
        None
    """
    mock_to_csv = mocker.patch('pandas.DataFrame.to_csv')
    build_clf_model(KNeighborsClassifier(), StandardScaler(), 'output_dir', X_train, y_train, X_test, y_test, {}, 'test_report.csv')
    mock_to_csv.assert_called_once_with(os.path.join('output_dir', 'test_report.csv'))
