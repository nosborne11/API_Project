# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle
# Add the necessary imports for the starter code.
import pandas as pd 
import numpy as np
from starter.ml.data import process_data
from starter.ml.model import  train_model,compute_model_metrics,inference
from starter.ml import model as model_lib

Input_1={'age': [52],
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

Input_2={'age': [50],
 'workclass': ['Self-emp-not-inc'],
 'fnlgt': [83311],
 'education': ['Bachelors'],
 'education-num': [13],
 'marital-status': ['Married-civ-spouse'],
 'occupation': ['Exec-managerial'],
 'relationship': ['Husband'],
 'race': ['White'],
 'sex': ['Male'],
 'capital-gain': [0],
 'capital-loss': [0],
 'hours-per-week': [13],
 'native-country': ['United-States']}

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


###Load Encoder, Model, 1b
with open("model/model.pickle", "rb") as input_file:
    model=pickle.load(input_file)
with open("model/encoder.pickle", "rb") as input_file:
    encoder=pickle.load(input_file)
with open("model/1b.pickle", "rb") as input_file:
    Lb=pickle.load(input_file)

def Create_Panda_DF(Input_data):
    data=pd.DataFrame(Input_data) 
    return data  

def Prediction_Pipe(data_download,encoder_download,lb_download,cat_features):
    X, y_data, encoder, lb_test = process_data(
    data_download, categorical_features=cat_features, label=None, training=False, encoder=encoder_download)
    res=model_lib.inference(model, X)
    return list(lb_download.inverse_transform(res))

def test_data():
  data=Create_Panda_DF(Input_1)
  assert len(data)>0

def test_model():
  data=Create_Panda_DF(Input_2)
  Out=Prediction_Pipe(data,encoder,Lb,cat_features)
  assert model!=None


def test_encoder():
  data=Create_Panda_DF(Input_2)
  Out=Prediction_Pipe(data,encoder,Lb,cat_features)
  assert encoder!=None

def test_Lb():
  data=Create_Panda_DF(Input_2)
  Out=Prediction_Pipe(data,encoder,Lb,cat_features)
  assert Lb!=None

def test_one():
  data=Create_Panda_DF(Input_1)
  Out=Prediction_Pipe(data,encoder,Lb,cat_features)
  assert Out==[">50K"]

def test_two():
  data=Create_Panda_DF(Input_2)
  Out=Prediction_Pipe(data,encoder,Lb,cat_features)
  assert Out==["<=50K"]

