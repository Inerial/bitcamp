import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

x,y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8, shuffle=True, random_state=66
)

model = XGBRegressor(n_estimators=2000, learning_rate=0.01)

model.fit(x_train, y_train, verbose=True, eval_metric="rmse", eval_set=[(x_train, y_train), (x_test,y_test)])
# rmse, mae, logloss, error, auc

result = model.evals_result()
print(result)