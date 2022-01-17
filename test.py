# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle
# Add the necessary imports for the starter code.
import pandas as pd 
import numpy as np
from starter.ml.data import process_data
from starter.ml.model import  train_model,compute_model_metrics,inference
from starter.ml import model as model_lib

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

data_all=pd.read_csv('data/census_modified.csv')


def Prediction_Pipe(data_download,encoder_download,lb_download,cat_features):
    train, test = train_test_split(data_all, test_size=0.20)
    X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder_download)
    res=model_lib.inference(model, X_test)
    return res,y_test


def Get_Results(data_all):
  Out,y_test=Prediction_Pipe(data_all,encoder,Lb,cat_features)
  res=compute_model_metrics(Lb.transform(y_test).ravel(),Out)
  return res

def test_Encoder():
  train, test = train_test_split(data_all, test_size=0.20)
  X_train, y_train, encoder_train, lb_train = process_data(
    train, categorical_features=cat_features, label="salary", training=True)
  assert encoder_train!=None

def test_Lb():
  train, test = train_test_split(data_all, test_size=0.20)
  X_train, y_train, encoder_train, lb_train = process_data(
    train, categorical_features=cat_features, label="salary", training=True)
  assert lb_train!=None

def test_precision():
  res=Get_Results(data_all)
  precision=res[0]
  assert precision>=0.5

def test_recall():
  res=Get_Results(data_all)
  recall=res[1]
  assert recall>=0.5

def test_fbeta():
  res=Get_Results(data_all)
  fbeta=res[1]
  assert fbeta>=0.5

