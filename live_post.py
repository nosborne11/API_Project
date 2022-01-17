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

def Run():
    url = 'http://salary-api-nosborne11.herokuapp.com/items/'
    x = requests.post(url,data=js.dumps(Input_1))
    print(f'''Status Code: {x.status_code}''')
    print(f'''Result: {x.json()[0]}''')

if __name__=='__main__':
    Run()