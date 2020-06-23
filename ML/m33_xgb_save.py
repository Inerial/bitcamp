import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score

x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8, shuffle=True, random_state=66
)

model = XGBClassifier(n_estimators=500, learning_rate=0.01)

model.fit(x_train, y_train, verbose=True, eval_metric="error", eval_set=[(x_train, y_train), (x_test,y_test)])
# rmse, mae, logloss, error, auc

result = model.evals_result()
# print(result)

y_pred = model.predict(x_test)
acc=accuracy_score(y_pred, y_test)
print("acc :", acc)

# import pickle

# pickle.dump(model, open("./model/xgb_save/cancer.pickle.dat", "wb"))

# print("저장")

# model2 = pickle.load(open("./model/xgb_save/cancer.pickle.dat"))
# print("불러오기")

# y_pred = model2.predict(x_test)
# acc = accuracy_score(y_pred,y_test)
# print("acc :", acc)

# import joblib
# joblib.dump(model, "./model/xgb_save/cancer.joblib.dat")
# print("저장")

# model2 = joblib.load("./model/xgb_save/cancer.joblib.dat")
# print("불리다.")

# y_pred = model2.predict(x_test)
# acc = accuracy_score(y_test,y_pred)
# print("acc :", acc)

model.save_model("./model/xgb_save/cancer.load_model.dat")
print("저장")

model2 = XGBClassifier()
model2.load_model("./model/xgb_save/cancer.load_model.dat")
print("불리다.")

y_pred = model2.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("acc :", acc)