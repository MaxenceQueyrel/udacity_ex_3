from fastapi.testclient import TestClient
import json
from starter.main import app

client = TestClient(app)


def test_hello_world():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["Hello World!"]


def test_inference_sample_1():
    sample = {"age": "32", "workclass": "Private", "fnlgt": "116138",
              "education": "Masters", "education-num": "14",
              "marital-status": "Never-married", "occupation": "Tech-support",
              "relationship": "Not-in-family", "race": "Asian-Pac-Islander",
              "sex": "Male", "capital-gain": "0", "capital-loss": "0",
              "hours-per-week": "11", "native-country": "Taiwan"}
    r = client.post("/inference/", data=json.dumps(sample))
    assert r.status_code == 200
    assert r.json() == ['Salary predicted is <=50K'] or r.json() == ['Salary predicted is >50K']


def test_inference_sample_2():
    sample = {"age": "53", "workclass": "Private", "fnlgt": "234721",
              "education": "11th", "education-num": "7",
              "marital-status": "Married-civ-spouse", "occupation": "Handlers-cleaners",
              "relationship": "Husband", "race": "Black",
              "sex": "Male", "capital-gain": "0", "capital-loss": "0",
              "hours-per-week": "40", "native-country": "United-States"}
    r = client.post("/inference/", data=json.dumps(sample))
    assert r.status_code == 200
    assert r.json() == ['Salary predicted is <=50K'] or r.json() == ['Salary predicted is >50K']
