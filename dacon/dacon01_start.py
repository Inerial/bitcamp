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
from keras.layers import Input, Dense, Dropout

train = pd.read_csv('./data/dacon/comp1/train.csv', sep=',', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', index_col = 0, header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', sep=',', index_col = -4, header = 0)

print(train.shape) # x_train, x_test
print(test.shape) # x_predict
print(submission.shape) # y_predict


train = train.interpolate() # 선형보간법
test = test.interpolate()

train = train.fillna(method ='ffill')
train = train.fillna(method ='bfill')
test = test.fillna(method ='ffill')
test = test.fillna(method ='bfill')


x_train ,y_train = train.iloc[:,:-4].values, train.iloc[:,-4:].values
x_pred = test.values


# 2. model
def build_model(hidden_layers = 1, nodes = 128, activation = 'relu', optimizers= 'adam', drop = 0.5):
    inputs = Input(shape=(x_train.shape[1], ))
    
    denses = Dense(nodes, activation= activation)(inputs)
    denses = Dropout(drop)(denses)
    for i in range(hidden_layers-1):
        denses = Dense(nodes, activation= activation)(denses)
        denses = Dropout(drop)(denses)
    outputs = Dense(y_train.shape[1], activation=activation)(denses)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizers, loss = 'mae', metrics = ['mean_absolute_error'])
    return model

def create_hyperparameters():
    batches = [100,200,300,400,500,600,700,800,900,1000]
    nodes = [64,128,256,512,1024]
    layers = range(3,20)
    optimizers = ['rmsprop', 'adam', 'adadelta','sgd','adamax','nadam','adagrad']
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
    activation = ['relu', 'elu', 'selu', 'exponential', 'linear']
    epochs = [30,50,80,100,150]
    return {"models__batch_size" : batches, "models__epochs": epochs, "models__optimizers": optimizers, "models__drop" : dropout , 
            "models__activation" : activation, "models__nodes" :nodes, "models__hidden_layers" : layers}
model = KerasRegressor(build_fn=build_model)
parameters = create_hyperparameters()

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('models', model)
])
search = RandomizedSearchCV(pipe, parameters, cv=5, n_iter=10)

search.fit(x_train, y_train)

y_pred = search.predict(x_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": y_pred[:,0],
    "hbo2": y_pred[:,1],
    "ca": y_pred[:,2],
    "na": y_pred[:,3]
})

submissions.to_csv('./dacon/comp1_sub'+i+'.csv', index = False)