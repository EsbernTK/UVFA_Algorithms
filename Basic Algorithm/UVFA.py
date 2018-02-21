import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from OptSpace import OptSpace
import matplotlib.pyplot as plt
import time
import random
import Labyrinth

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

n = 9
m = 9

env = Labyrinth.Environment(n,m,1)
Visual = np.zeros((n,m))
env = Labyrinth.Environment(n,m,len(env.oneDStates))
for idx,state in enumerate(env.oneDStates):
    env.djikstra(state,idx,0.9)
    for visIdx,visList in enumerate(env.states):
        for visIdx1, visVal in enumerate(visList):
            if(type(visVal) == Labyrinth.State):
                Visual[visIdx][visIdx1] = visVal.Values[idx]
    print(np.array(Visual))
    print()
    print()
ValueMatrix = np.zeros((len(env.oneDStates),len(env.oneDStates)))
for idx,state in enumerate(env.oneDStates):
    ValueMatrix[idx][:] = state.Values
sparseIndexes = env.findLowerRightSquareStateIndexes()
sparseValueMatrix = [[val if idxI not in sparseIndexes else 0 for idxI,val in enumerate(ValueMatrix[idxJ])] if idxJ not in sparseIndexes else [0 for i in range(len(env.oneDStates))] for idxJ,list in enumerate(ValueMatrix)]
print(sparseValueMatrix)
X,S,Y,dist = OptSpace(np.array(sparseValueMatrix),7,10,1e-8)
print(np.linalg.norm(np.matmul(X,np.matmul(S,Y.T)) - ValueMatrix,'fro')/np.sqrt(len(env.oneDStates)*len(env.oneDStates)))


#for idx,list in enumerate(env.states):
#    for idx1, val in enumerate(list):
#        if(type(val) == Labyrinth.State):
#            print(idx,idx1,val.NeighbourStates)

