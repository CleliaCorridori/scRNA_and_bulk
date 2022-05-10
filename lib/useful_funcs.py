from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

from lib.import_funcs import *

def scalingNorm(matx):
    scaler = MinMaxScaler()
    # transform data
    return(np.transpose(scaler.fit_transform(np.transpose(matx))))

def Pearson_info(matx, thr=10**(-5)):
    #matx[abs(matx)<=thr]=float("NAN")
    
    corr_matx = np.zeros((np.shape(matx)[0], np.shape(matx)[0]))
    for ii in range(np.shape(matx)[0]):
        for jj in range(np.shape(matx)[0]):
            #idx_ii = np.argwhere(np.isnan(matx[ii, :]))
            #idx_jj = np.argwhere(np.isnan(matx[jj, :]))
            idx_ii = np.argwhere(matx[ii, :]<=thr)
            idx_jj = np.argwhere(matx[jj, :]<=thr)
            # idx = np.unique(np.concatenate((idx_ii, idx_jj)))
            idx = np.intersect1d(idx_ii, idx_jj)
            #print(ii, jj, (len(matx[ii, :])- len(idx))/len(matx[ii, :]))
            #print(idx_ii, "\n", idx_jj, "\n", idx)
    
            vect_ii = np.delete(matx[ii, :], idx)
            vect_jj = np.delete(matx[jj, :], idx)
            #print(np.where(vect_ii!=0), len(vect_jj))
            #print(len(vect_ii), len(vect_jj))
            corr_matx[ii,jj] = np.corrcoef(vect_ii, vect_jj)[0,1]
            #print(ii,jj, np.corrcoef(vect_ii, vect_jj)[0,1])
    return(corr_matx)

def remove_zero(data, thr=10**(-5)):
    out = np.copy(data)
    out[abs(out)<=thr]=float("NAN")
    return(out)