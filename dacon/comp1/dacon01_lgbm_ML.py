import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error as mae
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)


x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')
print(y_train)
def plot_feature_importacnes_cancer(model, title):
    n_features = x_train.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), range(0, x_train.shape[1]))
    plt.title(title)
    plt.xlabel("feature_importace")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

# 2. model

parameter = [
    {'n_estimators': [3000],
    'learning_rate': [0.05],
    'max_depth': [6], 
    'boosting': ['dart'], 
    'drop_rate' : [0.3],
    'objective': ['regression'], 
    'metric': ['logloss','mae'], 
    'is_training_metric': [True], 
    'num_leaves': [144], 
    'feature_fraction': [0.7], 
    'bagging_fraction': [0.7], 
    'bagging_freq': [5], 
    'seed': [66]
    }
]
fit_params = {
    'verbose': True,
    'eval_set' : [(x_train,y_train),(x_test,y_test)],
    # 'early_stopping_rounds' : 5
}
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
y_test_pred = []
y_pred = []
search = RandomizedSearchCV(LGBMRegressor(), parameter, cv = kfold, n_iter=1)

features = []
for i in range(4):
    fit_params['eval_set'] = [(x_train,y_train[:,i]),(x_test,y_test[:,i])]
    search.fit(x_train, y_train[:,i],**fit_params)
    y_pred.append(search.predict(x_pred))
    y_test_pred.append(search.predict(x_test))
    print(search.best_score_)
    plt.subplot(2,2,i+1)
    features.append(search.best_estimator_.feature_importances_)
    plot_feature_importacnes_cancer(search.best_estimator_, i)


y_pred = np.array(y_pred).T
y_test_pred = np.array(y_test_pred).T

print(y_pred.shape)
r2 = r2_score(y_test,y_test_pred)
mae = mae(y_test,y_test_pred)
print('r2 :', r2)
print('mae :', mae)

print(features)
plt.show()

submissions = pd.DataFrame({
    "id": test.index,
    "hhb": y_pred[:,0],
    "hbo2": y_pred[:,1],
    "ca": y_pred[:,2],
    "na": y_pred[:,3]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)

