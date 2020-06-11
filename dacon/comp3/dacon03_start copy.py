import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, MaxPool1D, LSTM
from keras import backend

def kaeri_metric(y_true, y_pred):
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)
def E1(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)
def E2(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


x = np.load('./dacon/comp3/x.npy')
y = np.load('./dacon/comp3/y.npy')
x_pred = np.load('./dacon/comp3/x_pred.npy')

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state = 66
)

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

train1, train2, train3 = x_train.shape
test1, test2, test3 = x_test.shape
pred1, pred2, pred3 = x_pred.shape

# x_train = scaler.fit_transform(x_train.reshape(train1, train2*train3)).reshape(train1, train2, train3)
# x_test = scaler.fit_transform(x_test.reshape(test1, test2* test3)).reshape(test1, test2, test3)
# x_pred = scaler.fit_transform(x_pred.reshape(pred1, pred2* pred3)).reshape(pred1, pred2, pred3)




# 2. 모델
def build_model():
    inputs = Input(shape=(x.shape[1], x.shape[2]))
    lstms = LSTM(200,)(inputs)
    lstms = Dropout(0.3)(lstms)

    denses = Dense(25)(lstms)
    denses = Dropout(0.3)(denses)
    denses = Dense(25)(denses)
    denses = Dropout(0.3)(denses)
    denses = Dense(25)(denses)
    denses = Dropout(0.3)(denses)
    denses = Dense(25)(denses)
    denses = Dropout(0.3)(denses)
    denses = Dense(25)(denses)
    denses = Dropout(0.3)(denses)
    outputs = Dense(y.shape[1])(denses)

    model = Model(inputs = inputs, outputs=outputs)

    model.compile(optimizer = 'adam', loss='mse', metrics=['mse'])
    return model

model = KerasRegressor(build_fn=build_model())

pipe = Pipeline([
                ('scaler', RobustScaler()), ## 스케일러
                ('models', model)  ## 모델 순
                # 이름   함수 
                # 이름은 파라미터 지정용, 각각 함수에 지정한 파라미터들을 넣어준다
])

parameters = {
    "models__epochs":[150],
    "models__batch_size":[500]
}
model = RandomizedSearchCV(pipe, parameters, cv=5)

model.fit(x_train, y_train)

print("best_model :", model.best_params_)

y_pred = model.predict(x_test)
mspe = kaeri_metric(y_test, y_pred)
print('mspe : ', mspe)

y_pred = model.predict(x_pred)

print(y_pred)

submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./dacon/comp3/comp3_sub.csv', index = False)

# mspe :  3.3595243892423294