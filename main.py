from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel
from typing import Union 
from starter.ml import model as model_lib
from starter. ml import data as data_lib

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

#####Import process_data
process_data=data_lib.process_data

####Catergorical Variable
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

###Dict to convert names
Convert_Dict={
"education_num":"education-num",
"marital_status":"marital-status",
"capital_gain":"capital-gain",
"capital_loss":"capital-loss",
"hours_per_week":"hours-per-week",
"native_country":"native-country"
}


###Load Encoder, Model, 1b
with open("model/model.pickle", "rb") as input_file:
    model=pickle.load(input_file)
with open("model/encoder.pickle", "rb") as input_file:
    encoder=pickle.load(input_file)
with open("model/1b.pickle", "rb") as input_file:
    Lb=pickle.load(input_file)


# Declare the data object with its components and their type.
class Feature_List(BaseModel):
    age: Union[int, list] 
    workclass: Union[str, list]  
    fnlgt: Union[int, list]  
    education: Union[str, list]  
    education_num: Union[int, list] 
    marital_status: Union[str, list] 
    occupation: Union[str, list] 
    relationship: Union[str, list] 
    race: Union[str, list] 
    sex: Union[str, list] 
    capital_gain: Union[int, list] 
    capital_loss: Union[int, list] 
    hours_per_week: Union[int, list] 
    native_country: Union[str, list] 


def Creat_Dict(item):
    Dict={
    "age":item.age,
    "workclass":item.workclass,
    "fnlgt":item.fnlgt,
    "education":item.education,
    "education-num":item.education_num,
    "marital-status":item.marital_status,
    "occupation":item.occupation,
    "relationship":item.relationship,
    "race":item.race,
    "sex":item.sex,
    "capital-gain":item.capital_gain,
    "capital-loss":item.capital_loss,
    "hours-per-week":item.hours_per_week,
    "native-country":item.native_country,
    }
    return Dict


def Create_Panda_DF(Input_data):
    data=pd.DataFrame(Input_data) 
    return data  

def Prediction_Pipe(data_download,encoder_download,lb_download,cat_features):
    X, y_data, encoder, lb_test = process_data(
    data_download, categorical_features=cat_features, label=None, training=False, encoder=encoder_download)
    res=model_lib.inference(model, X)
    return list(lb_download.inverse_transform(res))

######test input#######
Input=Input={'age': [52],
 'workclass': ['Self-emp-inc'],
 'fnlgt': [287927],
 'education': ['HS-grad'],
 'education-num': [9],
 'marital-status': ['Married-civ-spouse'],
 'occupation': ['Exec-managerial'],
 'relationship': ['Wife'],
 'race': ['White'],
 'sex': ['Female'],
 'capital-gain': [15024],
 'capital-loss': [0],
 'hours-per-week': [40],
 'native-country': ['United-States']}


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"Welcome to my Machine Learning API"}

######Post 
@app.post("/items/")
async def post_items(item: Feature_List):
    _dict=Creat_Dict(item)
    data=Create_Panda_DF(_dict)
    Out=Prediction_Pipe(data,encoder,Lb,cat_features)
    return Out

######Post 
@app.get("/items/")
async def run_model():
    data=Create_Panda_DF(Input)
    Out=Prediction_Pipe(data,encoder,Lb,cat_features)
    return Out



