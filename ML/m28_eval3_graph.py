import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score

x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8, shuffle=True, random_state=66
)

model = XGBClassifier(n_estimators=1000, learning_rate=0.1)

model.fit(x_train, y_train, verbose=True, eval_metric=["mlogloss","merror"], eval_set=[(x_train, y_train), (x_test,y_test)]
          ,early_stopping_rounds=20)
# rmse, mae, logloss, error, auc

result = model.evals_result()
print(result)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("acc :", acc)

import matplotlib.pyplot as plt

epochs = len(result['validation_0']['mlogloss'])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, result['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['merror'], label='Train')
ax.plot(x_axis, result['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('Error')
plt.title('XGBoost Error')
plt.show()

# acc : 1.0
# PS D:\Study>