import requests
import json as js

Input_1={'age': [52],
 'workclass': ['Self-emp-inc'],
 'fnlgt': [287927],
 'education': ['HS-grad'],
 'education_num': [9],
 'marital_status': ['Married-civ-spouse'],
 'occupation': ['Exec-managerial'],
 'relationship': ['Wife'],
 'race': ['White'],
 'sex': ['Female'],
 'capital_gain': [15024],
 'capital_loss': [0],
 'hours_per_week': [40],
 'native_country': ['United-States']}

Input_2={'age': [50],
 'workclass': ['Self-emp-not-inc'],
 'fnlgt': [83311],
 'education': ['Bachelors'],
 'education_num': [13],
 'marital_status': ['Married-civ-spouse'],
 'occupation': ['Exec-managerial'],
 'relationship': ['Husband'],
 'race': ['White'],
 'sex': ['Male'],
 'capital_gain': [0],
 'capital_loss': [0],
 'hours_per_week': [13],
 'native_country': ['United-States']}


def test_get():
    url = 'http://127.0.0.1:8000/'
    x = requests.get(url)
    assert x.status_code == 200
    assert x.json()==["Welcome to my Machine Learning API"]


def test_post_1():
    url = 'http://127.0.0.1:8000/items/'
    x = requests.post(url,data=js.dumps(Input_1))
    assert x.status_code == 200
    assert x.json()==[">50K"]

def test_post_2():
    url = 'http://127.0.0.1:8000/items/'
    x = requests.post(url,data=js.dumps(Input_2))
    assert x.status_code == 200
    assert x.json()==["<=50K"]