import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.data



#2. 모델
model = Sequential()
model.add(LSTM(800, input_shape=(5-1,1)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))

model.summary()

##model.save(".//model//save_keras44.h5")
model.save("./model/save_keras44.h5")
##model.save("./model/save_keras44.h5")
'''
#3. 훈련
model.compile(optimizer="adam", loss = 'mse',metrics=['mse'])

from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='val_loss', patience = 20, mode = "auto")

model.fit(x_train, y_train , epochs = 1000, callbacks=[early], validation_split= 0.25, shuffle = True)

#4. 평가,예측
loss, mse = model.evaluate(x_test,y_test)
print('loss :', loss)
print('mse :', mse)

y_pred = model.predict(x_pred)
print(y_pred)

'''