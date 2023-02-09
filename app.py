import pickle
from flask import Flask, request, app, jsonify,url_for,render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

# loading the columns
labelVars = pickle.load(open("./models/labelVars.pkl","rb"))
oneHotVars = pickle.load(open("./models/oneHotVars.pkl","rb"))
numVars = pickle.load(open("./models/numVars.pkl","rb"))
totalCols = pickle.load(open("./models/totalCols.pkl","rb"))

# loading the models
labelGender = pickle.load(open("./models/labelGender.pkl","rb"))
oneHotEncoder = pickle.load(open("./models/oneHotEncoder.pkl","rb"))
MinMaxScaler = pickle.load(open("./models/MinMaxScaler.pkl","rb"))
best_model = pickle.load(open("./models/best_model.pkl","rb"))
labelRest = pickle.load(open("./models/labelRest.pkl","rb"))

@app.route("/")
def home():
  return render_template("home.html")

if __name__=="__main__":
  app.run(debug=True)