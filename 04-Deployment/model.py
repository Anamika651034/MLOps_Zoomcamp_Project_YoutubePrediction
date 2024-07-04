# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
#For Evaluating the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
#preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
#Combine preprocessing
from sklearn.compose import ColumnTransformer
import re

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mlops_zoomcamp_youtube_pred_experiment")

# Importing the dataset
data = pd.read_csv('/workspaces/MLOps_Zoomcamp_Project_YoutubePrediction/dataset/youtube.csv') 

# Clean numeric columns
def clean_numeric_column(column):
    cleaned_column = []
    for value in column:
        if isinstance(value, str):
            if 'K' in value:
                cleaned_column.append(float(re.sub(r'[^0-9.]', '', value)) * 1000)
            elif 'M' in value:
                cleaned_column.append(float(re.sub(r'[^0-9.]', '', value)) * 1000000)
            elif 'B' in value:
                cleaned_column.append(float(re.sub(r'[^0-9.]', '', value)) * 1000000000)
            else:
                cleaned_column.append(float(re.sub(r'[^0-9.]', '', value)))
        else:
            cleaned_column.append(value)
    return cleaned_column

data['VIEWS'] = clean_numeric_column(data['VIEWS'])
data['TOTAL_NUMBER_OF_VIDEOS'] = clean_numeric_column(data['TOTAL_NUMBER_OF_VIDEOS'])
data['SUBSCRIBERS'] = clean_numeric_column(data['SUBSCRIBERS'])

# Features and target
X = data[['VIEWS', 'TOTAL_NUMBER_OF_VIDEOS', 'CATEGORY']]
y = data['SUBSCRIBERS']

#Define preprocessing steps for numerical and categorial features
numeric_features = ['VIEWS', 'TOTAL_NUMBER_OF_VIDEOS']
numeric_transformer = Pipeline(steps=[
   ('scaler', StandardScaler())
])
categorical_features = ['CATEGORY']
categorical_transformer = Pipeline(steps=[
   ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
   transformers=[
       ('num', numeric_transformer, numeric_features),
       ('cat', categorical_transformer, categorical_features)
   ])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print(X_train.head(1))

# Model pipeline
regressor = Pipeline(steps=[
   ('preprocessor', preprocessor),
   ('regressor', LinearRegression())  # Change to LinearRegression or any other regressor you want to use
])
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')

# Predict the number of subscribers for each channel in the dataset
data['Predicted Subscribers'] = regressor.predict(X)

# Save the predictions to a new CSV file
data.to_csv('/workspaces/MLOps_Zoomcamp_Project_YoutubePrediction/resultset/Result_youtube_channels_with_predictions.csv', index=False)
print("Predictions saved to 'youtube_channels_with_predictions.csv'")


# Saving model to disk
pickle.dump(regressor, open('/workspaces/MLOps_Zoomcamp_Project_YoutubePrediction/models/model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('/workspaces/MLOps_Zoomcamp_Project_YoutubePrediction/models/model.pkl','rb'))

# Example prediction with proper input structure
new_data = pd.DataFrame({'VIEWS': [1000000], 'TOTAL_NUMBER_OF_VIDEOS': [500], 'CATEGORY': ['Entertainment']})
predicted_subscribers = model.predict(new_data)
print(predicted_subscribers)

with mlflow.start_run():
        
        mlflow.set_tag("developer", "Anamika Vishwakarma")
        mlflow.log_param("data", "/workspaces/MLOps_Zoomcamp_Project_YoutubePrediction/dataset/youtube.csv")
        mlflow.log_param("model", "Youtube_Subscribers_Prediction")

        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.log_metric("mean_squared_error", mse)

        mlflow.log_artifact(local_path="/workspaces/MLOps_Zoomcamp_Project_YoutubePrediction/artifacts", artifact_path="models_pickle")