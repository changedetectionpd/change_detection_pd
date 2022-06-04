import numpy as np
import math
from scipy import integrate

def get_weight(x):
    return x[1]

def AB_valuate(data, gamma, D):
    target_sum = 0
    b_count = 0
    log_n = 0
    log_w = 0
    total_w = 0
    d_counts = []
    for i in range(len(data)):
        count = 0
        sum_norm = 0
        sum_w = 0
        to_w = 0
        d_count = []
        for j in range(len(data)):
            norm_s = np.linalg.norm(data[i]-data[j],ord=2)**2
            if (norm_s <= (1+gamma)*D):
                sum_norm += norm_s
                count += 1
                sum_w += np.log(get_weight(data[j]))
                to_w += get_weight(data[j])
                d_count.append(j)
        if count!=0:
            if (sum_norm/count) <= D:
                target_sum += sum_norm/count
                b_count += 1
                log_n += np.log(count)
                log_w += sum_w/count
                total_w += np.log(to_w)
                d_counts += d_count
    return target_sum, b_count, log_n, log_w, total_w, len(d_counts)

def get_NML(data,n,m,gamma,D,epsilon):
    target_sum, b_count, a_7, a_8, a_9, _ = AB_valuate(data,gamma,D)
    if target_sum>pow(0.1,100):
        a_1 = (n*m/2)*np.log(target_sum)
        a_2 = -b_count*((m/(m+4))*np.log(n)-m*np.log(epsilon))
        a_3 = n*m*np.log(pow(n,1/(m+4))/epsilon)
        a_5 = n*m*np.log(np.pi)/2
        a_6 = -math.lgamma(n*m/2)
        if 2*np.pi*D*pow(n,2/(m+4))/(m*pow(epsilon,2))>pow(0.1,100):
            if np.log(2*np.pi*D*pow(n,2/(m+4))/(m*pow(epsilon,2)))>pow(0.1,100):
                a_4 = np.log(np.log(2*np.pi*D*pow(n,2/(m+4))/(m*pow(epsilon,2))))
                return a_1+a_2+a_3+a_4+a_5+a_6-a_7-a_8+a_9
            else:
                return np.inf
        else:
            return np.inf
    else:
        return -np.inf
    
def get_idx_ds(x, n, m, gamma, epsilon, param):
    idx = []
    time = 0
    Ds = []
    D = 0
    i = 0
    while time!=2:
        D += param*pow(1.2,i)
        Ds.append(D)
        D = Ds[-1]
        idx.append(get_NML(x, n, m, gamma, D, epsilon))
        value = AB_valuate(x, gamma, D)[-1]
        if i == 0:
            if value >= 0.5 * pow(len(x),2):
                time = 2           
        if (value == pow(len(x),2))&(time!=2):
            time  += 1
            D = Ds[-2]
            Ds = Ds[:-1]
            idx = idx[:-1]
            i = 1  
        i+=1
    return idx, Ds

def get_WKC(x, n, m, gamma, epsilon, param):
    a, Ds = get_idx_ds(x, n, m, gamma, epsilon, param)
    while (np.inf in a)|(-np.inf in a)|(len(a)<20):
        if (np.inf in a):
            epsilon = epsilon/10
            a, Ds = get_idx_ds(x,n,m,gamma,epsilon,param)
        elif (-np.inf in a):
            gamma += 0.1
            param = 1.0
            a,Ds = get_idx_ds(x,n,m,gamma,epsilon,param)
        else:
            if (len(a)<20):
                param = param/10
                a,Ds = get_idx_ds(x,n,m,gamma,epsilon,param)
    if min(a)<0:
        b = a+(-min(a))
        L0 = 0
    else:
        b = a
        L0 = min(b)
    delta_D = max(Ds)-min(Ds)
    delta_L = max(b)-min(b)
    KC = 2-(integrate.cumtrapz(b, Ds, initial=0)[-1]-L0*delta_D)/((delta_D)*(delta_L)/2)
    return KC