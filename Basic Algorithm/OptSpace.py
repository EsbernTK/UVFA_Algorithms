from scipy.sparse import *
import numpy as np
from numpy import matlib
import scipy.sparse
from scipy.sparse.linalg import norm

import time
from functools import reduce
import operator
def G(X,m0,r):

    z = np.sum(X**2,1)/(2*m0*r)
    y = np.exp((z - 1)** 2) - 1;

    for ind,val in enumerate(np.nonzero(z<1)[0]):
        y[val] = 0

    return sum(y)
def F_t(X,Y,S,M,E,m0,rho):
    nxr = np.shape(X)
    out1 = sum(sum(((np.array((np.matmul(X, np.matmul(S, Y.T))) - M) * np.array(E.todense())) ** 2))) / 2
    out2 = rho*G(Y,m0,nxr[1])

    out3 = rho*G(X,m0,nxr[1])

    return out1+out2+out3
def getoptT(X,W,Y,Z,S,M,E,m0,rho):

    norm2WZ = np.power(np.linalg.norm(W, 'fro'),2) + np.power(np.linalg.norm(Z, 'fro'),2);
    #print(norm2WZ)
    f = [F_t(X,Y,S,M,E,m0,rho)]
    t = -0.1
    for i in range(20):

        f.append(F_t(np.array(X +t*W),np.array(Y + t*Z),S,M,E,m0,rho))
        if(f[i+1] - f[0] <= 0.5*(t)*norm2WZ):
            #print(t)
            return t;
        t /= 2
    return t



def getoptS(X,Y,M_E,E):
    nxr = np.shape(X)
    C = np.matmul(X.T,np.matmul(M_E.todense(),Y))
    Cnxm = np.shape(C)
    CnxmN = Cnxm[0]*Cnxm[1]
    C = np.array(C.flatten())[0]
    A = np.array([[0 for i in range(CnxmN)]for j in range(CnxmN)])

    for i in range(nxr[1]):
        for j in range(nxr[1]):
            ind = j*nxr[1] +i;

            temp = np.matmul(X.T,np.matmul(np.multiply(np.matmul(np.array([X[:,i]]).T,np.array([Y[:,j]])),E.todense()),Y))
            A[:,ind] = np.array(temp.flatten())[0]
    S = np.linalg.solve(A, C)
    return np.reshape(S,(nxr[1],nxr[1]))

def Gp(X,m0,r):
    z = np.sum(X**2,1)/(2*m0*r)
    z = 2 * np.exp((z-1)**2) * (z-1)
    for ind,val in enumerate(np.nonzero(z)[0]):
        if(z[val] < 0):
            z[val] = 0
    return X * np.matlib.repmat(z,r,1).T / (m0*r)

def gradF_t(X,Y,S,M,E,m0,rho):
    nxr = np.shape(X)
    mxr = np.shape(Y)


    XS = np.matmul(X,S)
    YS = np.matmul(Y,S.T)
    XSY = np.matmul(XS,Y.T)

    Qx = np.matmul(X.T,np.matmul(np.multiply(M.todense()-XSY,E.todense()),YS))/nxr[0]
    Qy = np.matmul(Y.T, np.matmul(np.multiply(M.todense() - XSY, E.todense()).T, XS)) / mxr[0]

    W = np.matmul(np.multiply(XSY-M.todense(),E.todense()),YS) + np.matmul(X,Qx) + rho*Gp(X,m0,nxr[1])
    Z = np.matmul(np.multiply(XSY-M.todense(),E.todense()).T,XS) + np.matmul(Y,Qy) + rho*Gp(Y,m0,nxr[1])

    return W, Z

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
    Y0 = Y0.T
    X0 = X0 * np.sqrt(nxm[0])
    Y0 = Y0 * np.sqrt(nxm[1])

    S0 = S0 / eps;

    X = X0
    Y = Y0
    S = getoptS(X,Y,M,E)
    dist = [np.linalg.norm(np.multiply((M - np.matmul(X,np.matmul(S,Y.T))),E.todense()),'fro')/np.sqrt(E.nnz)]
    print("Initial Dist:", dist)

    for i in range(num_iter):
        W,Z = gradF_t(X,Y,S,M,E,m0,rho)
        t = getoptT(X, W, Y, Z, S, M, E, m0, rho)
        X = np.array(X + t * W);
        Y = np.array(Y + t * Z);
        S = getoptS(X, Y, M, E);
        dist.append(np.linalg.norm(np.multiply((M - np.matmul(X,np.matmul(S,Y.T))),E.todense()),'fro')/np.sqrt(E.nnz))
        print("At iteration",i,"the error is: ",dist[i+1])
        if (dist[i + 1] < tol):
            break;
    S = S/rescal_param
    return X,S,Y,dist

def TestOptspace():
    n = 1001;
    m = 1000;
    r = 7;
    tol = 1e-8 ;

    np.random.seed(2)
    eps = 10*r*np.log10(n);
    U = np.random.randn(n,r);
    V = np.random.randn(m,r);
    sig = np.eye(r)
    M0 = np.matmul(U,np.matmul(sig,V.T))
    thefile = open('testFull.txt', 'w')
    for idxY, list in enumerate(M0):
        for idxX, value in enumerate(list):
            if (idxX < len(list) - 1):
                thefile.write("%s," % value)
            else:
                thefile.write("%s" % value)
        thefile.write("\n")
    thefile.close()
    E = 1 - np.ceil(np.random.rand(n,m) - (eps/np.sqrt(n*m)))

    M = M0 * E

    thefile = open('testSparse.txt', 'w')
    for idxY, list in enumerate(M0):
        for idxX, value in enumerate(list):
            if (idxX < len(list) - 1):
                thefile.write("%s," % value)
            else:
                thefile.write("%s" % value)
        thefile.write("\n")
    thefile.close()

    np.random.seed((int)(time.time()))

    print(M)

    X,S,Y,dist = OptSpace(M,r,10,tol)

    print(np.linalg.norm(np.matmul(X,np.matmul(S,Y.T)) - M0,'fro')/np.sqrt(m*n))
    print(np.matmul(X,np.matmul(S,Y.T)) - M0)
