from tkinter.messagebox import RETRY
from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np 
import pandas as pd
import pickle as pkl 


# one-hot encoding categorical features
CATEGORICALS = ['sex']
DUMMIES = {
    'sex':['M','F','I']
}

# rescaling numerical features
NUMERICS = ['length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight']
BOUNDARIES = {
    'length': (0.075000, 0.815000),
    'diameter': (0.055000, 0.650000),
    'height': (0.000000, 1.130000),
    'whole_weight': (0.002000, 2.825500),
    'shucked_weight': (0.001000, 1.488000),
    'viscera_weight': (0.000500, 0.760000),
    'shell_weight': (0.001500, 1.005000)
}


def dummy_encode(in_df, dummies):
    out_df = in_df.copy()
    
    for feature, values in dummies.items():
        for value in values:
            dummy_name = '{}__{}'.format(feature, value)
            out_df[dummy_name] = (out_df[feature] == value).astype(int)
            
        del out_df[feature]
        # print('Dummy-encoded feature\t\t{}'.format(feature))
    return out_df


def minmax_scale(in_df, boundaries):
    out_df = in_df.copy()
    
    for feature, (min_val, max_val) in boundaries.items():      
        col_name = '{}__norm'.format(feature)
        
        out_df[col_name] = round((out_df[feature] - min_val)/(max_val - min_val),3)
        out_df.loc[out_df[col_name] < 0, col_name] = 0
        out_df.loc[out_df[col_name] > 1, col_name] = 1
            
        del out_df[feature]
        # print('MinMax Scaled feature\t\t{}'.format(feature))
    return out_df


def prepare_data(inputs):
    #Arg : inputs is a table of json representing the query data
    #
    #Prepare the data for the prediction
    #Convert the json inputs into a pandas dataframe
    #Apply the same features pre-processing like done 
    #during the training step

    inputs_frame = pd.DataFrame(inputs)
    X_predict = dummy_encode(in_df=inputs_frame, dummies=DUMMIES)
    X_predict = minmax_scale(in_df=X_predict, boundaries=BOUNDARIES)

    return X_predict


app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def make_infer():
    #model loading 
    model_path = "pickles/model_v1.pkl"
    model = pkl.load(open(model_path, 'rb'))

    #Getting the inputs
    data = request.get_json()
    inputs = data["inputs"]
    X_predict = prepare_data(inputs)

    predictions = model.predict(X_predict)
    proba_predicts = model.predict_proba(X_predict)
    probabilities = []

    #Probabilities of predicted classes
    for i, classes in enumerate(proba_predicts):
        classe = predictions[i]
        probabilities.append(classes[classe])

    class_probas = zip(predictions, probabilities)
    outputs = {}
    outputs["outputs"] = [{"label": int(l), "probability":p} \
        for (l, p) in class_probas]

    return outputs

@app.route('/', methods=['GET'])

def print_hello():
    return "Welcome to the coolest API ;) !!!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

