# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle
# Add the necessary imports for the starter code.
import pandas as pd 
import numpy as np
from ml.data import process_data
from ml.model import  train_model,compute_model_metrics,inference

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
###Save Data
def Save_Data(model,encoder,lb_train):
      ###Save Model Encoder 1b
    with open(r"../model/model.pickle", "wb") as output_file:
        pickle.dump(model,file=output_file)
    with open(r"../model/encoder.pickle", "wb") as output_file:
        pickle.dump(encoder,file=output_file)
    with open(r"../model/1b.pickle", "wb") as output_file:
        pickle.dump(lb_train,file=output_file)


####Slice all data for analysis

def Slice_Data_all(data,encoder,model,lb, features):
    Data_Slice=pd.DataFrame(columns=['Class','Mean','STD_Dev'])
    Res=[]
    for fet in features: 
        for cls in data[fet].unique():
            df_temp = data[data[fet] == cls]
            _type=fet+' : '+cls
            res=Get_Metrics(df_temp,encoder,model,lb,_type)
            Res.append(res)
    Save_Slice_Data(Res)

#####Saves data to slice_output.txt
def Save_Slice_Data(Res):
    DD=pd.DataFrame(Res,columns=['Type','precision','recall','fbeta'])
    DD.to_csv("../slice_output.txt")
    

def Get_Metrics(test,encoder,model,lb,_type):
     # Proces the test data with the process_data function.
    X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder)
    #####Prediction Results
    preds=inference(model, X_test)
    res=compute_model_metrics(lb.transform(y_test).ravel(),preds)
    if _type=='All':
        print(f'''Type: {_type}''')
        print(f'''precision: {res[0]}''')
        print(f'''recall: {res[1]}''')
        print(f'''fbeta: {res[2]}''')
    return [_type,res[0],res[1],res[2]]



###Run Training  
def Run():
    # Add code to load in the data.
    data=pd.read_csv('../data/census_modified.csv')

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb_train = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

    # Train and save a model.
    model=train_model(X_train,y_train)

    #####Save Data
    Save_Data(model,encoder,lb_train)
    
    ####Run Test
    res_list=Get_Metrics(test,encoder,model,lb_train,'All')

    ####Slice Data and saves output
    Slice_Data_all(test, encoder,model,lb_train,cat_features)

   
if __name__ == "__main__":
    Run()
    












    