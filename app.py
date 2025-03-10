from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs (only 2 features)
        form_values = list(request.form.values())
        if len(form_values) != 2:
            return render_template('index.html', prediction_text="Error: Please provide both inputs.")

        # Convert inputs to floats
        features = [float(x) if x.replace('.', '', 1).isdigit() else 0 for x in form_values]
        final_features = [np.array(features)]

        # Make prediction
        prediction = model.predict(final_features)
        result = "Diabetes" if prediction[0] == 1 else "No Diabetes"

        return render_template('index.html', prediction_text=f'Predicted Disease: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
