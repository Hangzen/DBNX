import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

train_df = pd.read_excel("training.xlsx")
test_df = pd.read_excel("test.xlsx")
X_train = train_df.drop(columns=["UKBB"])
y_train = train_df["UKBB"]
X_test = test_df.drop(columns=["UKBB"])
y_test = test_df["UKBB"]
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
result_df.to_excel("Result.xlsx", index=False)

mse = mean_squared_error(y_test, y_pred)
print(f"{mse}")
