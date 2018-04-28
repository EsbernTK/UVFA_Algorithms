import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from numpy import genfromtxt


file = open('TrainingData.txt','r')
text = file.read().split("\\n")
samples =[[float(i) for i in text[j].split(",")] for j in range(0,len(text)-1)]
print(samples)
print(len(samples))
data = np.array(samples)
X = [data[i][1:] for i in range(0,len(data))]
Y = [[data[i][0]]for i in range(0,len(data))]

for i in range(0,len(X)):
    print(len(X[i]))
trainingSamplesX = X[0:len(samples)-int(len(samples)/5)]
trainingSamplesY = Y[0:len(samples)-int(len(samples)/5)]
testSamplesX = X[len(samples)-int(len(samples)/5):]
testSamplesY = Y[len(samples)-int(len(samples)/5):]


model = Sequential()
model.add(Dense(52,input_dim=30,activation='relu'))
model.add(Dense(52,activation='relu'))
model.add(Dense(52,activation='relu'))
model.add(Dense(52,activation='relu'))
model.add(Dense(1,activation='relu'))

optimizer = Adam(lr=0.003, decay=0.002)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

model.fit(trainingSamplesX,trainingSamplesY, batch_size=32, epochs=100,verbose=1)