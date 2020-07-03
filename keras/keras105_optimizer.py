import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam, Adamax

model = Sequential()

model.add(Dense(10,input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1010101))
model.add(Dense(1))

# optimizer = Adam(lr=0.001) # loss : 4.91854962092475e-06
# optimizer = RMSprop(lr=0.001) # loss : 0.006224682554602623
# optimizer = SGD(lr=0.001) # loss : 0.003473188728094101
# optimizer = Adadelta(lr=0.001) # loss : 4.1884589195251465
# optimizer = Adagrad(lr=0.001) # loss : 4.360927505331347e-06
# optimizer = Nadam(lr=0.001) # loss : 0.25854504108428955
optimizer = Adam(lr=0.001) # loss : 4.91854962092475e-06


model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

model.fit(x,y,epochs=100)

loss, mse = model.evaluate(x,y)
print('loss :', loss)

pred1 = model.predict([3.5])
print(pred1)