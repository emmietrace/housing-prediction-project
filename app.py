import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
# Ensure 'house_price_model.pkl' is in the 'model' folder or same directory
model = joblib.load('model/house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [
            float(request.form['OverallQual']),
            float(request.form['GrLivArea']),
            float(request.form['TotalBsmtSF']),
            float(request.form['GarageCars']),
            float(request.form['FullBath']),
            float(request.form['YearBuilt'])
        ]

        # Convert to numpy array for prediction
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        
        # Format the output (e.g., $150,000)
        output = f"${prediction[0]:,.2f}"

        return render_template('index.html', prediction_text=output)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)