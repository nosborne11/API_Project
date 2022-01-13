# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle
# Add the necessary imports for the starter code.
import pandas as pd 
import numpy as np
from ml.data import process_data
from ml.model import  train_model,compute_model_metrics,inference



# Add code to load in the data.
data=pd.read_csv('../data/census_modified.csv')



# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb_train = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder
)


# Train and save a model.
model=train_model(X_train,y_train)

###Save Model
with open(r"model.pickle", "wb") as output_file:
    pickle.dump(model,file=output_file)
    
#####Prediction Results
preds=inference(model, X_test)
res=compute_model_metrics(lb_train.transform(y_test).ravel(),preds)
print(f'''precision: {res[0]}''')
print(f'''recall: {res[1]}''')
print(f'''fbeta: {res[2]}''')

###Save Prediction Results
with open(r"result.pickle", "wb") as output_file:
    pickle.dump(res,file=output_file)
    
    
    