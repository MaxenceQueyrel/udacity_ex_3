# Put the code for your API here.
import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from starter.starter import train_model
from joblib import load
import pandas as pd


# setup DVC on Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# Load the model
assert os.path.exists(train_model.path_model), "The model {} should exist".format(train_model.path_model)
model = load(train_model.path_model)

# Instantiate the app.
app = FastAPI()


# Define a GET on the specified endpoint.
@app.get("/")
async def hello():
    return {"Hello World!"}


class CensusSample(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.post("/inference/")
async def model_inference(sample: CensusSample):
    sample = pd.DataFrame(jsonable_encoder([sample]))
    sample, _, _ = train_model.process_data(sample, train_model.cat_features, train_model.label, training=False,
                                            encoder=load(train_model.path_encoder), with_label=False)
    prediction = model.predict(sample)
    salary = train_model.classes[prediction[0]]
    return {"Salary predicted is {}".format(salary)}


