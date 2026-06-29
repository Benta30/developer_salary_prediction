"""
    deploying developer salary prediction model using Flask

    To run: 
        python app/flask_api.py

    then open browser :
        http://127.0.0.1:5000
render_template helps us send that data to ur browser
"""
from pathlib import Path
import pandas as pd
import numpy as np
from  flask import Flask, request, render_template
import joblib

# app setup
app = Flask(__name__, template_folder='templates')

# model_path (in the file explorer hadi kwa folder ya project)
root_dir = Path(__file__).resolve().parent 
MODEL_PATH = root_dir.parent/"models"/"salary_pipeline.pkl"
print(f"Loading model from: {MODEL_PATH}")

#picking the model
pipeline = joblib.load(MODEL_PATH)
print(f":Model loaded from {MODEL_PATH}")

# html form 




# the route is like the next button on a web page moving to different pages say predict or the home  route
@app.route('/') # sets the root path, ie when one visits http://127.0.0.1:5000
def home():
    return render_template('index.html') # calling the html file from templates

@app.route('/predict', methods=['POST']) # this is when one visits http://127.0.0.1:5000/predict
def predict():
    input_df = pd.DataFrame([{
        'Country': request.form.get('country'),
        'YearsCode': float(request.form.get('years')),
        'EdLevel': request.form.get('education'),
        'Employment': request.form.get('employment'),
        'LanguageCount': request.form.get('languages')

    }])

    prediction = pipeline.predict(input_df)[0] #(we save our prediction and we return our prediction) we pick the zeroth since zero always hold the target variable
    salary=f"{int(np.clip(prediction, 10000, 500000)):,}" # should be in that range 10k to 500k

    return render_template('index.html', salary=salary)


# helps us run our file
if __name__== '__main__':
    app.run(debug=True)