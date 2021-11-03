# Put the code for your API here.
import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from starter import train_model
from joblib import load
import pandas as pd
import boto3
import tempfile

# Instantiate the app.
app = FastAPI()

# Load the model
s3 = boto3.client('s3',
                    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

bucket = "udacity-exercice-3"
object_model = 'model/a6/5606716c2aed7b51571bdb300cf12c'
object_encoder = 'model/c7/ab6a0a97911ac2d594b7827af87060'

try:
    with tempfile.TemporaryFile() as fp:
        s3.download_fileobj(Fileobj=fp, Bucket=bucket, Key=object_model)
        fp.seek(0)
        model = load(fp)
except Exception as e:
    print("The model should exists in the S3 bucket")
    raise e

try:
    with tempfile.TemporaryFile() as fp:
        s3.download_fileobj(Fileobj=fp, Bucket=bucket, Key=object_encoder)
        fp.seek(0)
        encoder = load(fp)
except Exception as e:
    print("The encoder should exists in the S3 bucket")
    raise e


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
                                            encoder=encoder, with_label=False)
    prediction = model.predict(sample)
    salary = train_model.classes[prediction[0]]
    return {"Salary predicted is {}".format(salary)}


