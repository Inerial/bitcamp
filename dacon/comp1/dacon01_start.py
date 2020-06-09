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
from keras import backend

test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)

# 2. model
def build_model(hidden_layers = 6, nodes = 128, activation = 'relu', optimizers= 'adam', drop = 0.5):
    backend.clear_session()
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
    batches = [500,1000]
    # nodes = [64,128,256]
    # layers = range(3,20,2)
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.3, 0.5]
    activation = ['relu', 'elu', 'linear']
    epochs = [150]
    return {"models__batch_size" : batches, "models__epochs": epochs, "models__optimizers": optimizers, "models__drop" : dropout , 
            "models__activation" : activation}# , "models__nodes" :nodes, "models__hidden_layers" : layers}
model = KerasRegressor(build_fn=build_model)
parameters = create_hyperparameters()

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('models', model)
])
search = RandomizedSearchCV(pipe, parameters, cv=5, n_iter=20)

search.fit(x_train, y_train)

print(search.best_params_)
print("MAE :", search.score(x_test,y_test))
y_pred = search.predict(x_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": y_pred[:,0],
    "hbo2": y_pred[:,1],
    "ca": y_pred[:,2],
    "na": y_pred[:,3]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)