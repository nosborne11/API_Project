import pickle

with open("result.pickle", "rb") as input_file:
    result=pickle.load(input_file)

#####Unit Tests#####
def test_precision():
    assert result['precision']>0.5
    
def test_recall():
    assert result['recall']>0.5
    
def test_fbeta():
    assert result['fbeta']>0.5