from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Activation,Embedding,LSTM
import numpy as np

x_train = np.array([-2,-1,0,1,2,3,4,5,6,7])
y_train = np.array([90, 98, 106, 114, 122,130,138,146,154,162])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([914,922,930,938,946,954,962,970,978,986])

model = Sequential()
model.add(Dense(1000, input_dim=1, activation='relu'))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data = (x_test, y_test))
loss,acc = model.evaluate(x_test ,y_test ,batch_size=1)

print("loss : ", loss)
print("acc : ",acc)

output = model.predict(x_test)
print("결과물 : \n", output)