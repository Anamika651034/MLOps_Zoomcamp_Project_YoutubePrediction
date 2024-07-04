# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import json

app = Flask(__name__)

# Load the model
model = pickle.load(open('/workspaces/MLOps_Zoomcamp_Project_YoutubePrediction/models/model.pkl', 'rb'))
print("Model Loaded")

@app.route('/api', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    print("Data is :" , data)

    # # Extract features from JSON data
    # views = data['VIEWS']
    # total_number_of_videos = data['TOTAL_NUMBER_OF_VIDEOS']
    # category = data['CATEGORY']

    # Prepare input for prediction
    #input_features = [['VIEWS, TOTAL_NUMBER_OF_VIDEOS, CATEGORY]]

    # Make prediction using model loaded from disk
    prediction = model.predict([[data['VIEWS'],data['TOTAL_NUMBER_OF_VIDEOS'], data['CATEGORY']]])

    # Take the first value of prediction (assuming it's a single output)
    output = prediction[0]
    print(prediction[0])

    #return json.dump({'prediction': output})
    return json.dumps(output, default=str)


if __name__ == '__main__':
    app.run(port=5001, debug=True)
    #app.run()
