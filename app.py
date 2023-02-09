import pickle
from flask import Flask, request, app, jsonify,url_for,render_template
import sklearn

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
ohe = pickle.load(open("./models/ohe.pkl","rb"))
minMax = pickle.load(open("./models/minMax.pkl","rb"))
best_model = pickle.load(open("./models/best_model.pkl","rb"))
labelRest = pickle.load(open("./models/labelRest.pkl","rb"))

@app.route("/")
def home():
  return render_template("home.html")

@app.route("/predictApi",methods=["POST"])
def predict_api():
  data = [x for x in request.form.values()]

  # converting the received input as dataFrame
  input=np.array(list(data)).reshape(1,-1)
  ipDf = pd.DataFrame(input,columns=totalCols)

  # convert the tenure, monthlyCharges, totalCharges to numeric
  ipDf['tenure'] = pd.to_numeric(ipDf['tenure'],errors='coerce')
  ipDf['MonthlyCharges'] = pd.to_numeric(ipDf['MonthlyCharges'],errors='coerce')
  ipDf['TotalCharges'] = pd.to_numeric(ipDf['TotalCharges'],errors='coerce')
  ipDf['SeniorCitizen'] = pd.to_numeric(ipDf['SeniorCitizen'],errors='coerce')


  # gender labelling
  ipDf['gender']=ipDf[['gender']].apply(labelGender.transform)

  # label encoding the other variables
  ipDf[labelVars] =ipDf[labelVars].apply(labelRest.transform)

  # one hot encoding the labels
  encoded = ohe.transform(ipDf[oneHotVars]).toarray()
  encoded_df = pd.DataFrame(encoded,columns=ohe.get_feature_names_out())
  ipDf = pd.concat([ipDf,encoded_df], axis=1)
  ipDf.drop(columns=ohe.feature_names_in_,inplace=True)

  # scaling of the numeric variables
  ipDf[numVars] = minMax.transform(ipDf[numVars])
  
  #import the model best
  output=best_model.predict(ipDf)
  if output[0] == 0:
    prediction_text = f" NO--The Customer will not Churn"
  else: 
    prediction_text = f" YES--The Customer will Churn "

  return render_template("home.html",prediction_text=prediction_text)


if __name__=="__main__":
  app.run(debug=True)