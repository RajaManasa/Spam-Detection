from flask import current_app as app
from flask import render_template, request
from joblib import load
import numpy as np

@app.route('/', methods=['GET', 'POST'])
def spam_detection():
    preds_as_str = ""
    if request.method == 'POST':
        test_input = request.form.get("message")
        
        if test_input:
            # Load the model
            model = load('model_spam_detection.joblib')
            
            # Transform the input as needed by the model
            test_input_transformed = [test_input]  # Assuming the model expects a list of strings
            
            # Make the prediction
            preds = model.predict(test_input_transformed)
            
            # Convert the predictions to a string for display
            preds_as_str = str(preds[0])  # Assuming preds is an array and you want the first result
    
    return render_template('index.html', preds=preds_as_str)

