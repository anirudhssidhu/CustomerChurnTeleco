import pickle
from flask import Flask, request, app, jsonify,url_for,render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

# loading the columns
numVariables = pickle.load(open("./models/ScalingVar.pkl", "rb"))
labelVariables = pickle.load(open("./models/LabelVar.pkl","rb"))
oneHotVariables = pickle.load(open("./models/OneHotVar.pkl","rb"))

# loading the models
scaler = pickle.load(open("./models/Scaler.pkl", "rb"))
model = pickle.load(open("./models/BestModel.pkl", "rb"))

@app.route("/")
def home():
  return render_template("home.html")

if __name__=="__main__":
  app.run(debug=True)