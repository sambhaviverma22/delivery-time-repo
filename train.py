import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

orders = pd.read_csv("data/delivery_times.csv")
FEATURES = ["distance_km", "prep_time_min", "traffic_level", "rain"]

X = orders[FEATURES]
y = orders["delivery_min"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Baseline: always guess the average
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_mae = mean_absolute_error(y_test, lr.predict(X_test_scaled))

rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

print(f"Baseline MAE          : {baseline_mae:.2f} minutes")
print(f"Linear Regression MAE : {lr_mae:.2f} minutes")
print(f"Random Forest MAE     : {rf_mae:.2f} minutes")
