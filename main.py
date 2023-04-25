from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"": {"origins": ""}})

@app.route("/")
def index():
    return "This is the intro page"


@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    # Get the data from the request
    data = request.get_json(force=True)

    # Load the XGBClassifier model
    model = joblib.load('customer_churn.pkl')

    # Extract the features from the data
    features = pd.DataFrame(data)
    print(features)

    # Make the prediction
    prediction = model.predict(features).tolist()

    if prediction[0] == 0:
        prediction = 'No'
    if prediction[0] == 1:
        prediction = 'Yes'

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run(debug=True)
