import os
import pickle
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

app = Flask(__name__)

# Load the trained model
model_file = 'models/PCASSS_model.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/inspect")
def inspect():
    return render_template("inspect.html")

@app.route("/inspect", methods=["POST"])
def inspect_data():
    if request.method == "POST":
        GlobalReactivePower = float(request.form['input1'])
        Global_intensity = float(request.form['input2'])
        Sub_metering_1 = float(request.form['input3'])
        Sub_metering_2 = float(request.form['input4'])
        Sub_metering_3 = float(request.form['input5'])
        X = [[GlobalReactivePower, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3]]
        output = round(model.predict(X)[0], 3)
        return render_template('output.html', output=output)
    else:
        return render_template('inspect.html')

#if __name__ == "__main__":
    #app.run(debug=True)
