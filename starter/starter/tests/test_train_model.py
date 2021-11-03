import pytest
from .. import train_model
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from joblib import load
import os
import boto3

path_model = os.path.abspath(os.path.join(__file__, os.path.pardir, "model.pkl"))
path_scores = os.path.abspath(os.path.join(__file__, os.path.pardir,  "scores.csv"))


# Arrange
@pytest.fixture(scope="session")
def data():
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )
    bucket = "udacity-exercice-3"
    prefix = 'data/12/c208530a5680c15ae19b34152286dd'
    response = s3.get_object(Bucket=bucket, Key=prefix)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    assert status == 200, f"Unsuccessful S3 get_object response. Status - {status}"
    data = pd.read_csv(response.get("Body"))
    # path_data = os.path.abspath(os.path.join(__file__, *[os.path.pardir] * 3, "data/census_clean.csv"))
    # data = pd.read_csv(path_data)
    data_processed, y_data, encoder = train_model.process_data(data, categorical_features=train_model.cat_features, label=train_model.label, training=True)
    return data_processed, y_data


@pytest.fixture(scope="session", autouse=True)
def remove():
    yield
    os.remove(path_model)
    os.remove(path_scores)


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







