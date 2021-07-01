from operator import index
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import wr_nlp_train_linear_svc
import json

pd.set_option('display.max_colwidth', 300)
#pd.show_versions(as_json=False)

app = Flask(__name__)
model = pickle.load(open('nlp_WR20.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

""" @app.route('/status')
def status():
    wr = {"7009677":"Complete","7007651":"On Work Order","6979886":"Complete"}
    query_input = request.args.get('workrequest')
    for x, y in wr.items():
        if x == query_input:
            return y """
 
@app.route('/status')
def status():
    df = pd.DataFrame(
        {'work_request' : ["7009677","7007651","6979886","658997","7145890","7019410"],
        'status' : ['Complete','On Work Order','Complete','New','Complete','Complete'],
        'resource' : ['John','Lonan','Revathi','Christi','Angel','Albie'],
        'scheduled_end_date' : ['02/21/2022','08/21/2021','02/21/2022','10/03/2021','02/11/2022','08/15/2021']
        }
    )
    query_input1 = request.args.get('workrequest')
    query_input2 = request.args.get('field')
    wr = df.loc[df['work_request'] == query_input1][query_input2].values[0]
    #df.index[df['work_request'] == query_input1]
    return wr

@app.route("/search")
def get_input():
    query_input = request.args.get("loc")
    prediction = wr_nlp_train_linear_svc.wr_model(query_input) 
    output = np.array_str(prediction)
    return output

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