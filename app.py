from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read user input
    values = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age'])
    ]

    # Predict
    prediction = model.predict([values])[0]
    result = "Diabetes Detected ✅" if prediction == 1 else "No Diabetes ❌"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
