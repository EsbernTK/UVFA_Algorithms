import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from numpy import genfromtxt

normalizers = [600,1,30,50,1,1,1,1,5,5,600,600,30,50,1,1,600,1,30,50,1,1,1,1,5,5,600,600,30,50,1,1]

testFile = open('TestData.txt','r')
testText = testFile.read().split("\\n")
testData =[[float(i) for i in testText[j].split(",")] for j in range(0,len(testText)-1)]

testSamplesX = [[testData[i][j]/normalizers[j-1] for j in range(1,len(testData[i]))] for i in range(32,len(testData))]
testSamplesY = [[testData[i][0]/100.0+1]for i in range(32,len(testData))]

model = load_model("model4GOODWith36.h5")

acc = model.evaluate(testSamplesX,testSamplesY,batch_size=32,verbose=1)
predTest = model.predict(testSamplesX[0:100])
for i in range(0,len(predTest)):
    print(predTest[i][0],testSamplesY[i][0])