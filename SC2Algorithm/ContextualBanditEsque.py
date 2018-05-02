import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from numpy import genfromtxt

normalizers = [600,1,30,50,1,1,1,1,5,5,600,600,30,50,1,1,600,1,30,50,1,1,1,1,5,5,600,600,30,50,1,1]
trainingFile = open('TrainingData.txt','r')
trainingText = trainingFile.read().split("\\n")
trainingData =[[float(i) for i in trainingText[j].split(",")] for j in range(0,len(trainingText)-1)]

testFile = open('TestData.txt','r')
testText = testFile.read().split("\\n")
testData =[[float(i) for i in testText[j].split(",")] for j in range(0,len(testText)-1)]
#print(samples)
print(len(trainingData[0]))
print(len(testData))

#print(data)
trainingSamplesX = [[trainingData[i][j]/normalizers[j-1] for j in range(1,len(trainingData[i]))] for i in range(32,len(trainingData))]
trainingSamplesY = [[trainingData[i][0]/100.0+1]for i in range(32,len(trainingData))]

testSamplesX = [[testData[i][j]/normalizers[j-1] for j in range(1,len(testData[i]))] for i in range(32,len(testData))]
testSamplesY = [[testData[i][0]/100.0+1]for i in range(32,len(testData))]
#for i in range(0,len(X)):
#    print(len(X[i]))
#trainingSamplesX = X[0:len(X)-int(len(X)/5)]
#trainingSamplesY = Y[0:len(Y)-int(len(Y)/5)]
#"#print(trainingSamplesY)
#testSamplesX = X[len(X)-int(len(X)/5):]
#testSamplesY = Y[len(Y)-int(len(Y)/5):]
print(np.max(testData,0))

model = Sequential()
model.add(Dense(30,input_dim=32,activation='relu')) #,activation='relu'
model.add(Dropout(0.1))
model.add(Dense(52,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(40,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(5,activation='relu'))
model.add(Dense(1,activation='relu'))

optimizer = Adam(lr=0.002, decay=0.00002)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

model.fit(trainingSamplesX,trainingSamplesY, batch_size=32, epochs=10,verbose=1)
acc = model.evaluate(testSamplesX,testSamplesY,batch_size=32,verbose=1)
model.save("model.h5")
print(acc)
predTrain = model.predict(trainingSamplesX[0:100])
for i in range(0,len(predTrain)):
    print(predTrain[i][0],trainingSamplesY[i][0])


print()
print()
print()
predTest = model.predict(testSamplesX[0:100])
for i in range(0,len(predTest)):
    print(predTest[i][0],testSamplesY[i][0])