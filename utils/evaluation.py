import numpy as np
from sklearn.metrics import auc

def calc_falarms_benefit(scores, change_points, h=100, T=100, N_thr=100, eps=1e-2):
    scores_max, scores_min = np.nanmax(scores), np.nanmin(scores)
    threshold_list = np.linspace(scores_min - eps, scores_max + eps, N_thr)

    falarms = []
    benefits = []
    
    N = len(scores)
    
    for threshold in threshold_list:
        binary_alarm = (np.array(scores) >= threshold).astype(np.int)
        
        benefit = np.zeros(N)
        for cp in change_points:
            benefit[cp:cp+T+1] = 1.0 - np.arange(T+1)/T
            # benefit[cp-T:cp+T+1] = 1.0 - np.hstack((np.arange(T, 0, -1), np.arange(T+1)))/T
        
        total_benefit = np.sum(binary_alarm * benefit)
        n_falarm = np.sum(binary_alarm * (benefit == 0.0).astype(np.int))
        
        benefits.append(total_benefit/np.sum(np.ones(N) * benefit))
        falarms.append(n_falarm/np.sum(np.ones(N) * (benefit == 0.0).astype(np.int)))

    benefits = np.array(benefits)    
    falarms = np.array(falarms)
    
    return falarms, benefits

def calc_auc_average(scores_list, 
                     cps_true=np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]) - 1,
                     T=100):
    auc_list = []
    
    N_trial = scores_list.shape[0]
    
    for i in range(N_trial):
        falarms, benefits = calc_falarms_benefit(scores_list[i, :], cps_true, T=T)
        auc_ = auc(falarms, benefits)
        auc_list.append(auc_)
        
    return np.array(auc_list)


def InvRunLen(size):
    n = len(size)
    inv_run_len = np.zeros(n)
    for i in range(0, n):
        if round(size[i] - (1.0 + i)) == 0.0:
            inv_run_len[i] = 0
        else:
            inv_run_len[i] = 1/(size[i] + 1)
    return inv_run_len

def get_right_wrong_diff(detection,change_points,T):
    right = 0
    wrong = 0
    for i in range(len(detection)):
        a = 0
        for j in range(len(change_points)):
            if (detection[i]<=change_points[j]+T)&(detection[i]>=change_points[j])&(a==0):
                right += 1
                a = 1
        if a!=1:
            wrong += 1
    return right, wrong


def get_threshold(scores,change_points,T):
    N_thr=100
    eps=1e-2
    scores_max, scores_min = np.nanmax(scores), np.nanmin(scores)
    threshold_list = np.linspace(scores_min - eps, scores_max + eps, N_thr)
    goodness = []
    for threshold in threshold_list:
        detection = []
        for i in range(len(scores)):
            if scores[i]>threshold:
                detection.append(i)
        right,wrong = get_right_wrong_diff(detection,change_points,T)
        if (wrong!=0)&(right+wrong>=40):
            goodness.append(right/(right+wrong))
        else:
            goodness.append(0)
    
    return threshold_list[np.argmax(goodness)]


def get_evaluation(detection,change_points,T):
    right_cnt = 0
    wrong = 0
    number = []
    benefits = []
    benefit = 1.0 - np.arange(T+1)/T
    for i in range(len(detection)):
        a = 0
        for j in range(len(change_points)):
            if (detection[i]<change_points[j]+T)&(detection[i]>=change_points[j])&(a==0):
                right_cnt += 1
                a = 1
                if j not in number:
                    benefits.append(benefit[detection[i]-change_points[j]])
                    number.append(j)
    return np.sum(benefits)/len(change_points), right_cnt/len(detection)
