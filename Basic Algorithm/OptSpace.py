from scipy.sparse import *
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import norm

def OptSpace(M,rank,num_iter,tol):
    #tol stops the algorithm if the distance is lower than tol


    nxm = np.shape(M)
    E = M.copy(M).tocsr().data.fill(1)
    eps = np.nonzero(E)/np.sqrt(nxm[0]*nxm[1])
    print(eps)




indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
bsr_matrix((data,indices,indptr), shape=(6, 6)).toarray()


M = [[0,0,0,1],[0,1,2,3],[1,4,0,0],[0,5,2,1]]
OptSpace(M,3,10,1)