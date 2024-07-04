#preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import DictVectorizer

#Combine preprocessing
from sklearn.compose import ColumnTransformer

#For Evaluating the model
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Algorithm used - LinearRegression
from sklearn.linear_model import LinearRegression

#Split the data into random train and test subsets
from sklearn.model_selection import train_test_split

#To convert million , billion string into float
import re


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer



@transformer
def transform(data):
  # Drop rows with NaN values
 data.dropna(inplace=True)
 data['VIEWS'] = clean_numeric_column(data['VIEWS'])
 data['TOTAL_NUMBER_OF_VIDEOS'] = clean_numeric_column(data['TOTAL_NUMBER_OF_VIDEOS'])
 data['SUBSCRIBERS'] = clean_numeric_column(data['SUBSCRIBERS'])

 
 # Define features and target
 numeric_features = ['VIEWS', 'TOTAL_NUMBER_OF_VIDEOS']
 categorical_features = ['CATEGORY']

 # Prepare data for training
 train_dicts = data[categorical_features + numeric_features].to_dict(orient='records')

 # Initialize DictVectorizer
 dv = DictVectorizer()

 # Transform categorical features into numerical representation
 X_train = dv.fit_transform(train_dicts)
 print("X_train shape:", X_train.shape)

 # Extract target variable
 y_train = data['SUBSCRIBERS'].values
 print("y_train shape:", y_train.shape)

 # Initialize Linear Regression model
 model = LinearRegression()

 # Train the model
 model.fit(X_train, y_train)

 # Predict on the test set
 y_pred = model.predict(X_train)

 # Evaluate the model
 mse = mean_squared_error(y_train, y_pred, squared=False)
 print(f'MSE: {mse}')

 return data

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


