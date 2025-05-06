import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

# Load dataset
housing = pd.read_csv("E:\\phy\\minort\\.ipynb_checkpoints\\housing.csv")

# Function to remove outliers based on IQR
def getOutliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"(IQR = {IQR}) Outliers are values outside the range: ({lower_bound}, {upper_bound})")
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Outliers found: {len(outliers)} out of total {df[column].size}")
    return df[~df[column].isin(outliers[column])]

housing = getOutliers(housing, "total_rooms")

# Fill missing values
housing["total_bedrooms"] = housing["total_bedrooms"].fillna(housing["total_bedrooms"].median())

# Encode categorical feature
labelEncoder = LabelEncoder()
housing["ocean_proximity"] = labelEncoder.fit_transform(housing["ocean_proximity"])

# Split features and target
X = housing.drop("median_house_value", axis=1)
y = housing["median_house_value"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost with hyperparameter tuning
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150],
    'reg_alpha': [0.1, 0.5],
    'reg_lambda': [1, 2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5]
}

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)

grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)

# Final model with best parameters
best_model = grid.best_estimator_

# Evaluate performance
train_rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
cv_rmse = np.mean(np.sqrt(-cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)))

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test  RMSE: {test_rmse:.2f}")
print(f"CV    RMSE: {cv_rmse:.2f}")

# Save model and scaler
joblib.dump(best_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
