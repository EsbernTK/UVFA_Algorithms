import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from OptSpace import OptSpace
import matplotlib.pyplot as plt
import time
import random
import Labyrinth

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

n = 9
m = 9
d = 0.8
env = Labyrinth.Environment(n,m,1)
Visual = np.zeros((n,m))
env = Labyrinth.Environment(n,m,len(env.oneDStates))
for idx,state in enumerate(env.oneDStates):
    env.djikstra(state,idx,d,1)
    for visIdx,visList in enumerate(env.states):
        for visIdx1, visVal in enumerate(visList):
            if(type(visVal) == Labyrinth.State):
                Visual[visIdx][visIdx1] = visVal.Values[idx]
    print(np.array(Visual))
    print()
    print()

ValueMatrix = np.zeros((len(env.oneDStates),len(env.oneDStates)))
print(np.shape(ValueMatrix))
for idx,state in enumerate(env.oneDStates):
    ValueMatrix[idx][:] = state.Values

ValueMatrix = (ValueMatrix - np.mean(np.mean(ValueMatrix))) * 2


sparseIndexes = env.findLowerRightSquareStateIndexes()
sparseValueMatrix = [[val if idxI not in sparseIndexes else 0 for idxI,val in enumerate(ValueMatrix[idxJ])] if idxJ not in sparseIndexes else [0 for i in range(len(env.oneDStates))] for idxJ,list in enumerate(ValueMatrix)]

#X,S,Y,dist = OptSpace(sparseValueMatrix,7,10,1e-8)
#print('Distance between real matrix and predicted matrix is',np.linalg.norm(np.matmul(X,np.matmul(S,Y.T)) - ValueMatrix,'fro')/np.sqrt(len(env.oneDStates)*len(env.oneDStates)))

trainingMatrixX = []
trainingMatrixY = []
for idx,val in enumerate(env.oneDStates):
    if (idx not in sparseIndexes):
        for i in range(len(env.oneDStates)):
            if(i not in sparseIndexes):

                trainingMatrixX.append(val.index.copy())
                trainingMatrixX[len(trainingMatrixX)-1].append(env.oneDStates[i].index[0])

                trainingMatrixX[len(trainingMatrixX)-1].append(env.oneDStates[i].index[1])
                trainingMatrixX[len(trainingMatrixX) - 1].append(d)
                trainingMatrixY.append(val.Values[i])
print(trainingMatrixX)
print(trainingMatrixY)

testMatrixX = []
testMatrixY = []
print(sparseIndexes)
for idx,val in enumerate(sparseIndexes):
    for i in range(len(env.oneDStates)):
        testMatrixX.append(env.oneDStates[i].index.copy())
        testMatrixX[len(testMatrixX)-1].append(env.oneDStates[val].index[0])

        testMatrixX[len(testMatrixX)-1].append(env.oneDStates[val].index[1])
        testMatrixX[len(testMatrixX) - 1].append(d)
        testMatrixY.append(env.oneDStates[i].Values[val])

print(np.shape(testMatrixX))
print(testMatrixY)

model = Sequential()
model.add(Dense(20,input_dim=5,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu'))

optimizer = Adam(lr=0.003, decay=0.002)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

model.fit(trainingMatrixX,trainingMatrixY, batch_size=32, epochs=100,verbose=1)
score = model.evaluate(testMatrixX,testMatrixY,verbose=0)
result = model.predict(testMatrixX,batch_size=32,verbose=0)
print(result)
print(testMatrixY)
result = np.reshape(result,(len(result)))
result = np.reshape(result,(len(sparseIndexes),len(env.oneDStates)))
print(result)
visualization = np.zeros((len(sparseIndexes),n,m))
for i in range(len(sparseIndexes)):
    for idx,val in enumerate(env.oneDStates):
        visualization[i][val.index[0]][val.index[1]] = result[i][idx] #- val.Values[sparseIndexes[i]]
    print()
    print(visualization[i])

#X,S,Y,dist = OptSpace(sparseValueMatrix,20,1000,1e-8)
#print('Distance between real matrix and predicted matrix is',np.linalg.norm(np.matmul(X,np.matmul(S,Y.T)) - ValueMatrix,'fro')/np.sqrt(len(env.oneDStates)*len(env.oneDStates)))


#newValues = np.matmul(X,np.matmul(S,Y.T))
#
#for idx,val in enumerate(newValues[:][len(newValues)-1]):
#    state = env.oneDStates[idx]
#    Visual[state.index[0]][state.index[1]] = val
#print(Visual)

#for idx,list in enumerate(env.states):
#    for idx1, val in enumerate(list):
#        if(type(val) == Labyrinth.State):
#            print(idx,idx1,val.NeighbourStates)

