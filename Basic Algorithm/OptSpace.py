from scipy.sparse import *
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import norm

def getoptS(X,Y,M_E,E):
    nxr = np.shape(X)
    print(np.shape(np.matrix.transpose(X)))
    print(np.shape(M_E))
    print(np.shape(Y))
    print(np.shape(np.dot(M_E,Y.T)))

    C = np.dot(X,np.dot(M_E,Y.T))
    print(np.shape(C))
    #print(X[:,0])
    #print(np.array([Y[:,0]]).T)
    #print(np.shape(np.matmul(X[:,0],np.array([Y[0,:]]).T)))
    for i in range(nxr[1]):
        for j in range(nxr[1]):
            ind = j*nxr[1] +i;
            #temp = (X[:,i] *np.matrix.transpose(Y[:,j]))
            #print(temp)


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
    np.dot(np.dot(S0,X0),Y0.T)
    S0 = S0 / eps;

    X = X0
    Y = Y0
    getoptS(X,Y,M,E)

indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
bsr_matrix((data,indices,indptr), shape=(6, 6)).toarray()


M = [[0,0,0,1,0,0,0,1],[0,0,0,3,0,0,0,1],[1,0,0,0,0,0,0,1],[0,0,2,1,0,0,0,1],[7,5,2,1,2,3,5,6],[0,5,0,0,0,0,5,6],[0,5,2,1,0,0,0,0],[0,5,2,1,0,0,0,0],[0,5,2,1,0,0,0,0]]

OptSpace(M,3,10,1)