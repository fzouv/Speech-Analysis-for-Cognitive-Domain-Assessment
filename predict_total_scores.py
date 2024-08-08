import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


file_path = '/Users/fotianazouvani/Downloads/SLP/Diss/both_withmoca.csv'
data = pd.read_csv(file_path)

# excluding irrelevant columns
exclude_cols = ['name','Label']
features = data.drop(columns=exclude_cols + ['moca_score'])

# setting moca scores as target variable
target = data['moca_score']

# train/test set split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# training Random Forest
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Mean Absolute Error (MAE): {mae}")
print(f"Random Forest Mean Squared Error (MSE): {mse}")
print(f"Random Forest R^2 Score: {r2}")

# 5-fold cross-validation for Random Forest
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def mae_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return mean_absolute_error(y, y_pred)

def mse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return mean_squared_error(y, y_pred)

cv_mae = cross_val_score(regressor, features, target, cv=kf, scoring=mae_scorer)
cv_mse = cross_val_score(regressor, features, target, cv=kf, scoring=mse_scorer)
cv_r2 = cross_val_score(regressor, features, target, cv=kf, scoring='r2')

mean_cv_mae = np.mean(cv_mae)
mean_cv_mse = np.mean(cv_mse)
mean_cv_r2 = np.mean(cv_r2)

print(f"5-Fold Cross-Validated Random Forest Mean Absolute Error (MAE): {mean_cv_mae}")
print(f"5-Fold Cross-Validated Random Forest Mean Squared Error (MSE): {mean_cv_mse}")
print(f"5-Fold Cross-Validated Random Forest R^2 Score: {mean_cv_r2}")

#Dummy Regressor for baseline
dummy_regressor = DummyRegressor(strategy='mean')
dummy_regressor.fit(X_train, y_train)

y_dummy_pred = dummy_regressor.predict(X_test)

# baseline model evaluation
dummy_mae = mean_absolute_error(y_test, y_dummy_pred)
dummy_mse = mean_squared_error(y_test, y_dummy_pred)
dummy_r2 = r2_score(y_test, y_dummy_pred)

print(f"Baseline Mean Absolute Error (MAE): {dummy_mae}")
print(f"Baseline Mean Squared Error (MSE): {dummy_mse}")
print(f"Baseline R^2 Score: {dummy_r2}")