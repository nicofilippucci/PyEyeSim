
import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import copy 

import hmmlearn.hmm  as hmm


def DiffCompsHMM(datobj,stim=0,ncomps=np.arange(2,6),NRep=10,NTest=3,covar='full'):
    ''' fit and cross validate HMM for a number of different hidden state numbers, as defined by ncomps'''
    if len(ncomps)<7:
        fig,ax=plt.subplots(ncols=3,nrows=2,figsize=(13,6))
    elif len(ncomps)<12:
        fig,ax=plt.subplots(ncols=4,nrows=3,figsize=(14,8))
  
    fig,ax2=plt.subplots()
    
    scoretrain,scoretest=np.zeros((NRep,len(ncomps))),np.zeros((NRep,len(ncomps)))
    for cc,nc in enumerate(ax.flat):
        if cc<len(ncomps):
            print('num comps: ',ncomps[cc],' num:', cc+1,'/', len(ncomps))
            for rep in range(NRep):
                if rep==NRep-1:
                    vis=True
                else:
                    vis=False
                hmm,scoretrain[rep,cc],scoretest[rep,cc]=datobj.FitVisHMM(datobj.stimuli[stim],ncomps[cc],covar=covar,ax=nc,ax2=ax2,vis=vis,NTest=NTest,verb=False)

    plt.legend(['train','test'])
    plt.tight_layout()
    
    fig,ax=plt.subplots()
    ax.errorbar(ncomps,np.mean(scoretrain,0),stats.sem(scoretrain,0),color='g',label='train',marker='o')
    ax.errorbar(ncomps,np.mean(scoretest,0),stats.sem(scoretest,0),color='r',label='test',marker='o')
    ax.set_xlabel('num of components')
    ax.set_ylabel('log(likelihood)')
    ax.legend()
    return 


    

def FitScoreHMMGauss(ncomp,xx,xxt,lenx,lenxxt,covar='full', n_iter=100, iter=1, bic=False):
    bestHMM = None
    if bic:
        best_sctr, best_scte = np.inf, np.inf
    else:
        best_sctr, best_scte = -np.inf, -np.inf
    if isinstance(ncomp,int):
        ncomp = [ncomp]

    for c in ncomp:
        for _ in range(iter):
            t = False
            HMM=hmm.GaussianHMM(n_components=c, covariance_type=covar, n_iter=n_iter)
            HMM.fit(xx,lenx)
            if bic:
                sctr=HMM.bic(xx,lenx)/np.sum(lenx)
                scte=HMM.bic(xxt,lenxxt)/np.sum(lenxxt)
                t = sctr<best_sctr 
            else:
                sctr=HMM.score(xx,lenx)/np.sum(lenx)
                scte=HMM.score(xxt,lenxxt)/np.sum(lenxxt)
                t = sctr>best_sctr
            if t:
                best_scte=scte
                best_sctr=sctr
                bestHMM=copy.deepcopy(HMM)
    return bestHMM,best_sctr,best_scte

def FitScoreHMMGauss_ind(ncomp,dat,lengths,trainsubj_idx,covar='full'):
    ''' for fitting on one observer and testing on all other '''
    IdxEnd=np.cumsum(lengths)
    IdxStart=np.append(0,np.cumsum(lengths)[:-1 ]) 
    HMM=hmm.GaussianHMM(n_components=ncomp, covariance_type=covar)
    
    datrain=dat[IdxStart[trainsubj_idx]:IdxEnd[trainsubj_idx]]    
    HMM.fit(datrain)
    scores=np.zeros(len(lengths))
    for s in range(len(lengths)):  
        dattest=dat[IdxStart[s]:IdxEnd[s]]           
        scores[s]=HMM.score(dattest)/(IdxEnd[s]-IdxStart[s])
    return HMM,scores
