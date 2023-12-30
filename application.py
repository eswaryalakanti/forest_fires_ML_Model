from flask import Flask,jsonify,request,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as ss
import pickle

model=pickle.load(open('models/ridge.pkl','rb'))
sc=pickle.load(open('models/scaler.pkl','rb'))

application = Flask(__name__)
app=application

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/prediction",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        #new_data_scaled=sc.transform([[12,1,1,1,1,1,1,1,1]])
        new_data_scaled=sc.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        print(new_data_scaled)
        result=model.predict(new_data_scaled)
        print(result)

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")

