from scipy.sparse import *
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import norm

from functools import reduce
import operator

def getoptS(X,Y,M_E,E):
    nxr = np.shape(X)
    print(np.shape(np.matrix.transpose(X)))
    print(np.shape(M_E.todense()))
    print(np.shape(Y))
    print(np.shape(np.matmul(X.T,M_E.todense())))


    C = np.matmul(np.matmul(X.T,M_E.todense()),Y.T)
    Cnxm = np.shape(C)
    CnxmN = Cnxm[0]*Cnxm[1]
    C = np.array(C.flatten())[0]
    A = np.array([[0 for i in range(CnxmN)]for j in range(CnxmN)])

    for i in range(nxr[1]):
        for j in range(nxr[1]):
            ind = j*nxr[1] +i;
            temp = np.matmul(np.matmul(X.T,np.multiply(np.matmul(np.array([X[:,i]]).T,np.array([Y[j,:]])),E.todense())),Y.T)
            A[:,ind] = np.array(temp.flatten())[0]
    S = np.linalg.solve(A, C)
    return np.reshape(S,(nxr[1],nxr[1]))



def OptSpace(M,rank,num_iter,tol):
    #tol stops the algorithm if the distance is lower than tol
    M = csr_matrix(M)

    nxm = np.shape(M)
    E = M.copy()


    for idx, val in enumerate(E.data):
        E.data[idx] = 1

    eps = E.nnz/np.sqrt(nxm[0]*nxm[1])

    m0 = 10000;
    rho = 0;

    rescal_param = np.sqrt(E.nnz * rank / np.power(norm(M, 'fro'),2));

    M = M * rescal_param;


    M_t = M;
    d = sum(E);
    #print(d)
    d_ = np.mean(d.todense());
    #print(d_,2*d_)
    for idx,val in enumerate(d.data):
        if(val > 2*d_):
            list1 = M.getcol(idx).nonzero()[0]
            p = np.random.permutation(len(list1))

            for i in range((int)(np.ceil(2*d_)), len(p)):
                M_t[list1[p[i]],idx] = 0
    d = E.sum(1);
    #print(M_t)
    #print(d)
    d_ = np.mean(d);
    #print(d_,2*d_)
    for idx,val in enumerate(d):
        if(val[0] > 2*d_):
            list1 = M.getrow(idx).nonzero()[1]
            p = np.random.permutation(len(list1))
            for i in range((int)(np.ceil(2*d_)), len(p)):
                M_t[idx,list1[p[i]]] = 0
    #print(M_t)

    X0, S0, Y0 = scipy.sparse.linalg.svds(M_t, rank);
    X0 = X0 * np.sqrt(nxm[0])
    Y0 = Y0 * np.sqrt(nxm[1])

    S0 = S0 / eps;

    X = X0
    Y = Y0
    S = getoptS(X,Y,M,E)

indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
bsr_matrix((data,indices,indptr), shape=(6, 6)).toarray()


M = [[0,0,0,1,0,0,0,1],[0,0,0,3,0,0,0,1],[1,0,0,0,0,0,0,1],[0,0,2,1,0,0,0,1],[7,5,2,1,2,3,5,6],[0,5,0,0,0,0,5,6],[0,5,2,1,0,0,0,0],[0,5,2,1,0,0,0,0],[0,5,2,1,0,0,0,0]]

OptSpace(M,3,10,1)