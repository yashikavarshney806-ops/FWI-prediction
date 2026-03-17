import os

from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

# Resolve model paths relative to this file so the app works no matter the current working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# import ridge regressor and standard scaler pickled files
ridge_model = pickle.load(open(os.path.join(MODEL_DIR, 'ridge.pkl'), 'rb'))
standard_scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        DC = float(request.form['DC'])
        ISI = float(request.form['ISI'])
        BUI = float(request.form['BUI'])
    
        new_scaled_data = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI]])
        result = ridge_model.predict(new_scaled_data)
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':  
    app.run(host="0.0.0.0",port=5000,debug=True)            