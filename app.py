from flask import Flask,render_template,request
import joblib
import numpy as np
model = joblib.load('models/model.h5')
scaler = joblib.load('models/scaler.h5')

app = Flask(__name__)
@app.route('/',methods=['Get'])
def Home():
    return render_template ('index.html')
@app.route('/Predict',methods=['Get'])
def Predict():
    input_data=[
    request.args.get('Temprature'),
    request.args.get('Humidity'),
    request.args.get('Hour'),
    request.args.get('Is_Rush_Hour',0),
    request.args.get('month')
    ]
    season=[int (n)for n in (request.args.get('Season').split(','))]
    Weather=[int (n)for n in (request.args.get('Weather').split(','))]
    Week_Day_Name=[int (n)for n in (request.args.get('Day').split(','))]
    Peroid_of_day=[int (n)for n in (request.args.get('Peroid').split(','))]
    input_data+=season
    input_data+=Weather
    input_data+=Week_Day_Name
    input_data+=Peroid_of_day
    input_data=[int(n)for n in input_data]
    profit= model.predict(scaler.transform([input_data]))
    profit=round(float(profit),0)
    return render_template ('index.html',profit=profit)


if __name__ == "__main__":
    app.run(debug=True)