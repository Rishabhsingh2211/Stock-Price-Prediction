import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load the preprocessed data into a pandas dataframe
df = pd.read_csv('preprocessed_data.csv')

# Split the data into training and testing sets
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build a linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Build a random forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate the performance of the models
print('Linear Regression Model:')
print('MSE:', mean_squared_error(y_test, y_pred_lr))
print('R^2 score:', r2_score(y_test, y_pred_lr))

print('Random Forest Regression Model:')
print('MSE:', mean_squared_error(y_test, y_pred_rf))
print('R^2 score:', r2_score(y_test, y_pred_rf))
