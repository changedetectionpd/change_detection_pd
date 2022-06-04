import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def psi(x,mu,sigma):
    if np.linalg.det(sigma)>0:
        a = 1/(2*np.pi*pow(np.linalg.det(sigma),1/2))
        b = np.dot(np.dot(x-mu,np.linalg.inv(sigma)),x-mu)
        if (-b/2)<=500:
            return a*np.exp(-b/2)
        else:
            return 1
    else:
        return 0

def estep(n,m,x,w,mu,sigma):
    eta = pd.DataFrame(np.zeros(n*m).reshape(n,m))
    alert1 = 0
    for i in range(n):
        a = 0
        for k in range(m):
            a += w[k]*psi(x[i],mu[k],sigma[k])
        if a!=0:
            for j in range(m):
                eta.iloc[i][j] = w[j]*psi(x[i],mu[j],sigma[j])/a
        else:
            alert1 = 1
    if alert1 == 0:
        return eta
    else:
        return pd.DataFrame(np.zeros(n*m).reshape(n,m))

def mstep(n,m,x,w,mu,sigma,eta):
    w1 = w.copy()
    mu1 = mu.copy()
    sigma1 = sigma.copy()
    alert2 = 0
    for j in range(m):
        a = 0
        for i in range(n):
            a += eta.iloc[i][j]
        w[j] = a/n
        
    for j in range(m):
        b = 0
        c = 0
        for i in range(n):
            b += eta.iloc[i][j]*x[i]
        for i in range(n):
            c += eta.iloc[i][j]
        if c!=0:
            mu[j] = b/c
        else:
            alert2 = 1
    
    for j in range(m):
        e = 0
        f = 0
        for i in range(n):
            e += eta.iloc[i][j]*np.dot(np.array([x[i]-mu[j]]).T,np.array([x[i]-mu[j]]))
        for i in range(n):
            f += eta.iloc[i][j]
        if (f!=0)&(np.linalg.det(e)>0):
            sigma[j] = e/f

        else:
            alert2 = 1
    if alert2 ==0:
        return w,mu,sigma
    else:
        return w1,mu1,sigma1

def EM_Algo(data,m):
    n = len(data)
    w = [1/m]*m
    mu = np.array([np.random.rand(2)])*np.mean(data,axis=0)
    A = np.random.rand(2,2)
    sigma = np.array([np.dot(A,A.T)])*np.std(data)
    for i in range(m-1):
        mu = np.append(mu,[np.random.rand(2)*np.mean(data,axis=0)],axis=0)
        A = np.random.rand(2,2)
        sigma = np.append(sigma,[np.dot(A,A.T)],axis = 0)*np.std(data)
    num_iter = 8
    eta1 = estep(n,m,data,w,mu,sigma)
    if np.all(eta1==0)==True:
        for i in range(n):
            for j in range(m):
                eta1.iloc[i][j]=psi(data[i],mu[j],sigma[j])
        z = eta1.idxmax(axis=1)
        for k in range(m):
            w[k] = sum(z==k)/n
        
    for i in range(num_iter):
        eta = estep(n,m,data,w,mu,sigma)
        if np.all(eta==0)==False:
            w,mu,sigma = mstep(n,m,data,w,mu,sigma,eta)
            eta1 = eta
    
    return w,mu,sigma,eta1

def calculate_CMN(n,K):
    C = [0,1]
    C_2 = 0
    for t in range(n+1):
        C_2 += (math.factorial(n)/(math.factorial(t)*math.factorial(n-t)))*pow(t/n,t)*pow(1-(t/n),n-t)
    if K>=2:
        C.append(C_2)
    for j in range(3,K+1):
        C.append(0)
        C[j] = C[j-1]+(n/(K-2))*C[j-2]
    
    return C[-1]

def get_DNML(K,ns,n,mu,sigma):
    S = 0
    T = 0
    U = 0
    
    R1,R2 = get_R(mu,K)
    l1,l2 = get_l(sigma,K)

    for k in range(K):
        n_k = ns[k]

        if n_k!=0:
            l,v = np.linalg.eig(sigma[k])
            a_1 = n_k*np.log(2*np.pi) + n_k
            a_2 = n_k*np.log(n_k/(2*math.e)) - np.log(np.pi)/2
            a_3 = -math.lgamma(2)
            a_4 = 0
            a_5 = 0
            a_6 = 0
            if np.linalg.norm(mu[k],ord=2)**2>0:
                a_4 += np.log(np.linalg.norm(mu[k],ord=2)**2)
            else:
                a_4 = -np.inf
                
            if np.linalg.det(sigma[k])>0:
                a_1 += n_k*np.log(np.linalg.det(sigma[k]))/2
            else:
                a_1 = -np.inf
                
            for i in range(len(l)):
                if l[i]>0:
                    a_4 += -np.log(l[i])
            if (R1!=0):
                if (R2/R1>0):
                    if np.log(R2/R1)>0:
                        a_5 += np.log(np.log(R2/R1))
            if (l1!=0):
                if (l2/l1>0):
                    if np.log(l2/l1)>0:
                        a_5 += 2*np.log(np.log(l2/l1))
            if n_k>=2:
                a_6 += -math.lgamma((n_k-1)/2)
            if n_k>=3:
                a_6 += -math.lgamma((n_k-2)/2)
            
            S += a_1+a_2+a_3+a_4+a_5+a_6

    for k in range(K):
        n_k = ns[k]
        if n_k!=0:
            T += n_k * (np.log(n)-np.log(n_k))

    U = np.log(calculate_CMN(n,K))

    return S+T+U

def get_l(sigma,K):
    L = []
    for k in range(K):
        l,v = np.linalg.eig(sigma[k])
        L.append(l[0])
        L.append(l[1])
    return min(L),max(L)

def get_R(mu,K):
    R = []
    for k in range(K):
        R.append(np.linalg.norm(mu[k],ord=2)**2)
    return min(R),max(R)

def get_copy_number(x, b):
    return int(x*b)

def get_K_mu_sigma(A, max_K, b):
    C = np.zeros((1,2))
    for s in range(len(A)):
        distance = A.T[1][s]-A.T[0][s]
        if np.all(C==0):
            if get_copy_number(distance, b)>0:
                for t in range(get_copy_number(distance, b)):
                    if t==0:
                        C = np.array([A[s]+np.random.normal(0,0.01)])
                    else:
                        C = np.append(C,[A[s]+np.random.normal(0,0.01)],axis=0)
        else:
            if get_copy_number(distance, b)>0:
                for t in range(get_copy_number(distance, b)):
                    C = np.append(C,[A[s]+np.random.normal(0,0.01)],axis=0)        
                        
    if len(C)<=1:
        num_comp = 0
        mu = None
        sigma = None
    else:
        x = np.append(np.array([C.T[0]]),[C.T[1]-C.T[0]],axis=0).T
        DNMLs = []
        mus = []
        sigmas = []
        for K in range(1,min(len(x)+1,max_K)):
            w, mu, sigma, eta = EM_Algo(x,K)
            mus.append(mu)
            sigmas.append(sigma)
            z = eta.idxmax(axis=1)
            n = len(x)
            ns = []
            for j in range(K):
                if sum(z==j)!=0:
                    ns.append(sum(z==j))        
            if len(ns)!=K:
                DNML = np.inf
            else:
                DNML = get_DNML(K, ns, n, mu, sigma)
            DNMLs.append(DNML)
        num_comp = np.argmin(DNMLs)+1
        mu = mus[num_comp-1]
        sigma = sigmas[num_comp-1]
    return num_comp, mu, sigma
