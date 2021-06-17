import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import wr_nlp_train_linear_svc

pd.set_option('display.max_colwidth', 300)

app = Flask(__name__)
model = pickle.load(open('nlp_WR20.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print('---------------- request value -------------------')
    print(type(request.form.get('experience')))
    final_features = request.form.get('experience')

    #final_features = [np.array(int_features)]
    print(final_features)
    print('----------------- request --------------------')
    prediction = wr_nlp_train_linear_svc.wr_model(final_features)
    print(prediction)
        
    print('------------------ request 2 ----------------------')

    return render_template('index.html', prediction_text='==> Suggested category is {}'.format(prediction))
    #return redirect(prediction)
    #render_template('type.html', prediction_text='Is your request is for {}'.format(prediction))

@app.route('/building',methods=['POST'])
def building():
    '''
    For rendering results on HTML GUI
    '''
    print('---------------- request value -------------------')
    print(type(request.form.get('building')))
    building_name = request.form.get('building')

    #return redirect(prediction)
    render_template('building.html', prediction_text='Is your request is for {}'.format(building_name))

if __name__ == "__main__":
    app.run(debug=True)