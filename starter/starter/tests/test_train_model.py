import pytest
from .. import train_model
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from joblib import load
import os

path_model = os.path.abspath(os.path.join(__file__, os.path.pardir, "model.pkl"))
path_scores = os.path.abspath(os.path.join(__file__, os.path.pardir,  "scores.csv"))

# Arrange
@pytest.fixture
def data():
    path_data = os.path.abspath(os.path.join(__file__, *[os.path.pardir]*3, "data/census_clean.csv"))
    data = pd.read_csv(path_data)
    data_processed, y_data, encoder = train_model.process_data(data, categorical_features=train_model.cat_features, label=train_model.label, training=True)
    return data_processed, y_data


def test_process_data(data):
    data_processed, y_data = data
    assert all([col not in data_processed.columns.values for col in train_model.cat_features])
    assert train_model.label not in data_processed.columns.values
    assert y_data.dtype == int


def test_train_model(data):
    data_processed, y_data = data
    clf = LogisticRegression()
    train_model.train_model(clf, data_processed, y_data, path_model)
    check_is_fitted(clf)
    assert os.path.exists(path_model)


def test_evaluate_model(data):
    data_processed, y_data = data
    clf = load(path_model)
    score = train_model.evaluate_model(clf, data_processed, y_data, path_scores)
    assert score >= 0. and score <= 1.
    assert os.path.exists(path_scores)

