import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

x,y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8, shuffle=True, random_state=66
)

model = XGBRegressor(n_estimators=100, learning_rate=0.1)

model.fit(x_train, y_train, verbose=True, eval_metric=["logloss","rmse"], eval_set=[(x_train, y_train), (x_test,y_test)]
          ,early_stopping_rounds=200)
# rmse, mae, logloss, error, auc

result = model.evals_result()
print(result)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 :", r2)

import matplotlib.pyplot as plt

epochs = len(result['validation_0']['logloss'])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label='Train')
ax.plot(x_axis, result['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label='Train')
ax.plot(x_axis, result['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()