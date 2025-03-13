# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:54:38 2024

@author: aratoj87
"""

from math import e
from re import S
import numpy as np
from numpy import cross, matlib, std
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt
import time

from sympy import Si
from .scanpathsimhelper import AOIbounds,CreatAoiRects,Rect,SaccadeLine,CalcSim, CheckCoor,CalcSimAlt,angle_difference_power, angle_difference_peak180, KuiperStat, CosineSim

def AOIFix(self,p,FixTrialX,FixTrialY,nDivH,nDivV,InferS=1):
    """ given a sequence of X,Y fixation data and AOI divisions, calculate static N and p matrix) """ 
    nAOI=nDivH*nDivV
    AOInums=np.arange(nAOI).reshape(nDivV,nDivH)
    NFix=len(FixTrialX)  # num of data points
    # set AOI bounds
   # print(p,SizeGendX)
    if InferS==0:
        AOIboundsH=AOIbounds(0, self.x_size,nDivH)       
        AOIboundsV=AOIbounds(0, self.y_size,nDivV)  
    elif InferS==1:
        AOIboundsH=AOIbounds(self.boundsX[p,0], self.boundsX[p,1],nDivH)       
        AOIboundsV=AOIbounds(self.boundsY[p,0], self.boundsY[p,1],nDivV)   
    elif InferS==2:
        if hasattr(self,'images'):
            ims=np.shape(self.images[self.stimuli[p]])
        else:
            ims=np.array([self.y_size,self.x_size])
        AOIboundsH=AOIbounds(0, ims[1],nDivH)       
        AOIboundsV=AOIbounds(0,ims[0],nDivV)  

    # set parameters & arrays to store data
    StatPtrial=np.zeros(nAOI) # static probabilities.
    StatNtrial=np.zeros(nAOI) # static counts.

   
    WhichAOIH=np.zeros(NFix)
    WhichAOIV=np.zeros(NFix)
    for x in range(NFix):
        WhichAOIH[x]=CheckCoor(AOIboundsH,FixTrialX[x]) # store which horizontal AOI each fixation is
        WhichAOIV[x]=CheckCoor(AOIboundsV,FixTrialY[x]) # store which vertical AOI each fixation is

    WhichAOI=np.zeros(NFix)
    WhichAOI[:]=np.nan
    for x in range(NFix):
        if WhichAOIV[x]>-1 and WhichAOIH[x]>-1:   # only use valid idx
            WhichAOI[x]=AOInums[np.intp(WhichAOIV[x]),np.intp(WhichAOIH[x])]  # get combined vertival and horizontal
    for st in range(nAOI): # gaze transition start
        StatNtrial[st]=np.sum(WhichAOI==st)  # get count in AOI
        StatPtrial[st]=np.sum(WhichAOI==st)/np.sum(np.isfinite(WhichAOI)) # calculate stationary P for each AOI    
    return NFix,StatPtrial,StatNtrial

    
def SaccadeSel(self,SaccadeObj,nHor,nVer=0,InferS=False): 
    ''' select saccades for angle comparison method'''
    if nVer==0:
        nVer=nHor  # if number of vertical divisions not provided -- use same as the number of horizontal
    SaccadeAOIAngles=[]
    SaccadeAOIAnglesCross=[]
    if InferS:
        if hasattr(self,'boundsX')==False:
            print('runnnig descriptives to get bounds')
            self.RunDescriptiveFix()  
        AOIRects=CreatAoiRects(nHor,nVer,self.boundsX,self.boundsY)
    else:
        AOIRects=CreatAoiRects(nHor,nVer,self.x_size,self.y_size,allsame=self.np)

    Saccades=np.zeros((((self.ns,self.np,nVer,nHor))),dtype=np.ndarray)  # store an array of saccades that cross the cell, for each AOI rectangle of each trial for each partiicpant
    for s in np.arange(self.ns):
        SaccadeAOIAngles.append([])
        SaccadeAOIAnglesCross.append([])
        for p in range(self.np):
            SaccadeAOIAngles[s].append(np.zeros(((int(self.nsac[s,p]),nVer,nHor))))
           # print(s,p,NSac[s,p])
            SaccadeAOIAngles[s][p][:]=np.nan
            SaccadeAOIAnglesCross[s].append(np.zeros(((int(self.nsac[s,p]),nVer,nHor))))
            SaccadeAOIAnglesCross[s][p][:]=np.nan
            for sac in range(len(SaccadeObj[s][p])):
                SaccadeDots=SaccadeObj[s][p][sac].LinePoints()
                for h in range(nHor):
                    for v in range(nVer):
                       # print(h,v)
                        if AOIRects[p][h][v].Cross(SaccadeDots)==True:
                          #  print(h,v,SaccadeObj[s][p][sac].Angle())
                            SaccadeAOIAngles[s][p][sac,v,h]=SaccadeObj[s][p][sac].Angle()  # get the angle of the sacccade

                if np.sum(SaccadeAOIAngles[s][p][sac,:,:]>0)>1:  # select saccaded that use multiple cells
                    #print('CrossSel',SaccadeAOIAngles[s][p][sac,:,:])
                    SaccadeAOIAnglesCross[s][p][sac,:,:]=SaccadeAOIAngles[s][p][sac,:,:]
                else:
                    # search index of all the not nan values of SaccadeAOIAngles[s][p][sac,:,:]
                    idxs = np.argwhere(~np.isnan(SaccadeAOIAngles[s][p][sac, :, :]))
                    if len(idxs) > 0:
                        for idx in idxs:
                            v, h = idx
                            if isinstance(Saccades[s,p,v,h], np.ndarray):
                                Saccades[s,p,v,h]=np.append(Saccades[s,p,v,h],SaccadeAOIAngles[s][p][sac,v,h])
                            else:
                                Saccades[s,p,v,h]=np.array([SaccadeAOIAngles[s][p][sac,v,h]])
                
            # store saccades that cross multiple AOI rectangles
            for h in range(nHor):
                for v in range(nVer):
                    if np.sum(np.isfinite(SaccadeAOIAnglesCross[s][p][:,v,h]))>0:
                        if isinstance(Saccades[s,p,v,h], np.ndarray):
                            Saccades[s,p,v,h]=np.append(Saccades[s,p,v,h],SaccadeAOIAnglesCross[s][p][~np.isnan(SaccadeAOIAnglesCross[s][p][:,v,h]),v,h])
                        else:
                            Saccades[s,p,v,h]=np.array(SaccadeAOIAnglesCross[s][p][~np.isnan(SaccadeAOIAnglesCross[s][p][:,v,h]),v,h])
                    elif not isinstance(Saccades[s,p,v,h], np.ndarray):
                        Saccades[s,p,v,h]=np.array([])
                    
    return Saccades

def SaccadeSingleSel(self, SaccadeObj, nHor, stim, nVer=0, InferS=False): 
    ''' 
    Overriden method to select saccades for angle comparison method for a specific stimulus
    
    Select saccades for angle comparison method for a specific stimulus
    '''
    
    if nVer == 0:
        nVer = nHor  # if number of vertical divisions not provided -- use same as the number of horizontal
    
    SaccadeAOIAngles = []
    SaccadeAOIAnglesCross = []
    
    if InferS:
        if not hasattr(self, 'boundsX'):
            print('Running descriptives to get bounds')
            self.RunDescriptiveFix()  
        AOIRects = CreatAoiRects(nHor, nVer, self.boundsX, self.boundsY)
    else:
        AOIRects = CreatAoiRects(nHor, nVer, self.x_size, self.y_size, allsame=self.np)
        
    Saccades = np.zeros(((self.ns, nVer, nHor)), dtype=np.ndarray)  # Array of saccades crossing each AOI rectangle for each trial and participant
    for s in np.arange(self.ns):
        SaccadeAOIAngles.append(np.zeros((int(self.nsac[s, stim]), nVer, nHor)))
        SaccadeAOIAngles[s][:] = np.nan
        SaccadeAOIAnglesCross.append(np.zeros((int(self.nsac[s, stim]), nVer, nHor)))
        SaccadeAOIAnglesCross[s][:] = np.nan
        
        for sac in range(len(SaccadeObj[s][stim])):
            SaccadeDots = SaccadeObj[s][stim][sac].LinePoints()
            
            for h in range(nHor):
                for v in range(nVer):
                    if AOIRects[stim][h][v].Cross(SaccadeDots):
                        SaccadeAOIAngles[s][sac, v, h] = SaccadeObj[s][stim][sac].Angle()  # Get angle of the saccade

            # Select saccades that cross multiple cells
            if np.sum(SaccadeAOIAngles[s][sac, :, :] > 0) > 1:
                SaccadeAOIAnglesCross[s][sac, :, :] = SaccadeAOIAngles[s][sac, :, :]
            else:
                # search index of all the not nan values of SaccadeAOIAngles[s][p][sac,:,:]
                idxs = np.argwhere(~np.isnan(SaccadeAOIAngles[s][sac, :, :]))
                if len(idxs) > 0:
                    for idx in idxs:
                        v, h = idx
                        if isinstance(Saccades[s,v,h], np.ndarray):
                            Saccades[s,v,h]=np.append(Saccades[s,v,h],SaccadeAOIAngles[s][sac,v,h])
                        else:
                            Saccades[s,v,h]=np.array([SaccadeAOIAngles[s][sac,v,h]])

        # store saccades that cross multiple AOI rectangles
        for h in range(nHor):
            for v in range(nVer):
                if np.sum(np.isfinite(SaccadeAOIAnglesCross[s][:,v,h]))>0:
                    if isinstance(Saccades[s,v,h], np.ndarray):
                        Saccades[s,v,h]=np.append(Saccades[s,v,h],SaccadeAOIAnglesCross[s][~np.isnan(SaccadeAOIAnglesCross[s][:,v,h]),v,h])
                    else:
                        Saccades[s,v,h]=np.array(SaccadeAOIAnglesCross[s][~np.isnan(SaccadeAOIAnglesCross[s][:,v,h]),v,h])
                elif not isinstance(Saccades[s,v,h], np.ndarray):
                    Saccades[s,v,h]=np.array([])
        
    
    return Saccades


def SacSim1Group(self,Saccades,Thr=5,p='all',normalize='add',method='default',power=1, match=False):
    ''' calculate saccade similarity for each stimulus, between each pair of participants ,
    needs saccades stored as PyEyeSim saccade objects stored in AOIs as input,
    vertical and horizontal dimensions are inferred from the input
    Thr=5: threshold for similarity in degree
    !! if Thr is 0, use power function for difference in angle, for now this is a difference score, not a similarity
    normalize, if provided must be add or mult 
    simcalc: True all angles transformed to below 180 before calculating similarity'''
    
    nVer=np.shape(Saccades)[2]
    nHor=np.shape(Saccades)[3]
        
    SimSacP=np.zeros((self.ns,self.ns,self.np,nVer,nHor))  
    SimSacP[:]=np.nan
    for s1 in range(self.ns):
        for s2 in range(self.ns):
            if s1!=s2:
                for p1 in range(self.np):
                    if self.nsac[s1,p1]>5 and self.nsac[s2,p1]>5:                    
                        for h in range(nHor):
                            for v in range(nVer):
                                if len(Saccades[s1,p1,v,h])>0 and len(Saccades[s2,p1,v,h])>0:
                                    
                                    if Thr==0:
                                        if method == 'peak180':
                                            SimSacP[s1,s2,p1,v,h]=angle_difference_peak180(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h],power=power, match=match)
                                        elif method == 'power':
                                            SimSacP[s1,s2,p1,v,h]=angle_difference_power(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h],power=power)
                                        elif method == 'Kuiper':
                                            SimSacP[s1,s2,p1,v,h]=KuiperStat(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h])
                                        else:
                                            raise ValueError('Invalid method')
                                    else:
                                        if method == 'default':  
                                            simsacn=CalcSimAlt(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h],Thr=Thr)                       
                                            if normalize=='add':
                                                SimSacP[s1,s2,p1,v,h]=simsacn/(len(Saccades[s1,p1,v,h])+len(Saccades[s2,p1,v,h]))
                                            elif normalize=='mult':
                                                SimSacP[s1,s2,p1,v,h]=simsacn/(len(Saccades[s1,p1,v,h])*len(Saccades[s2,p1,v,h]))
                                        elif method == 'cosine':
                                            SimSacP[s1,s2,p1,v,h]=CosineSim(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h], Thr=Thr)
    return SimSacP

  
def SacSim1GroupAll2All(self,Saccades,Thr=5,p='all',normalize='add',method='true',power=1, match=False):
    ''' calculate saccade similarity for each stimulus, and across all stimuli, between each pair of participants ,
    needs saccades stored as PyEyeSim saccade objects stored in AOIs as input,
    vertical and horizontal dimensions are inferred from the input
    Thr=5: threshold for similarity    
    !! if Thr is 0, use power function for difference in angle, for now this is a difference score, not a similarity

    normalize, if provided must be add or mult '''
    
    nVer=np.shape(Saccades)[2]
    nHor=np.shape(Saccades)[3]
        
    SimSacP=np.zeros((self.ns,self.ns,self.np,self.np,nVer,nHor))  
    SimSacP[:]=np.nan
    for s1 in range(self.ns):
        for s2 in range(self.ns):
            if s1!=s2:
                for p1 in range(self.np):
                    for p2 in range(self.np):
                        if self.nsac[s1,p1]>5 and self.nsac[s2,p2]>5:                    
                            for h in range(nHor):
                                for v in range(nVer):
                                    if len(Saccades[s1,p1,v,h])>0 and len(Saccades[s2,p2,v,h])>0:
                                        
                                        if Thr==0:
                                            if method == 'peak180':
                                                SimSacP[s1,s2,p1,p2,v,h]=angle_difference_peak180(Saccades[s1,p1,v,h],Saccades[s2,p2,v,h],power=power, match=match)
                                            elif method == 'power':
                                                SimSacP[s1,s2,p1,p2,v,h]=angle_difference_power(Saccades[s1,p1,v,h],Saccades[s2,p2,v,h],power=power)
                                            elif method == 'Kuiper':
                                                SimSacP[s1,s2,p1,p2,v,h]=KuiperStat(Saccades[s1,p1,v,h],Saccades[s2,p2,v,h])
                                            else:
                                                raise ValueError('Invalid method')
                                        else:
                                            if method == 'default':  
                                                simsacn=CalcSimAlt(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h],Thr=Thr)                       
                                                if normalize=='add':
                                                    SimSacP[s1,s2,p1,v,h]=simsacn/(len(Saccades[s1,p1,v,h])+len(Saccades[s2,p1,v,h]))
                                                elif normalize=='mult':
                                                    SimSacP[s1,s2,p1,v,h]=simsacn/(len(Saccades[s1,p1,v,h])*len(Saccades[s2,p1,v,h]))
                                            elif method == 'cosine':
                                                SimSacP[s1,s2,p1,v,h]=CosineSim(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h], Thr=Thr)
     
    return SimSacP

def SacSim2GroupAll2All(self,Saccades1,Saccades2,Thr=5,p='all',normalize='add',power=1):
    ''' calculate saccade similarity for each stimulus, from two different observations,  across all stimuli, between each pair of participants ,
    needs saccades stored as PyEyeSim saccade objects divided into grid AOIs as input,
    vertical and horizontal dimensions are inferred from the input, have to match between the two datasets
    Thr=5: threshold for similarity    
    !! if Thr is 0, use power function for difference in angle, for now this is a difference score, not a similarity

    normalize, if provided must be add or mult '''
    
    nVer1=np.shape(Saccades1)[2]
    nHor1=np.shape(Saccades1)[3]
    nVer2=np.shape(Saccades2)[2]
    nHor2=np.shape(Saccades2)[3]
    assert nVer1==nVer2,'vertical grid division mismatch'
    assert nHor1==nHor2,'horizontal grid division mismatch'

    SimSacP=np.zeros((self.ns,self.ns,self.np,self.np,nVer1,nHor1))  
    SimSacP[:]=np.nan
    for s1 in range(self.ns):
        for s2 in range(self.ns):
            for p1 in range(self.np):
                for p2 in range(self.np):
                    for h in range(nHor1):
                        for v in range(nVer1):
                            if len(Saccades1[s1,p1,v,h])>0 and len(Saccades2[s2,p2,v,h])>0:                                 
                                if Thr==0:
                                    SimSacP[s1,s2,p1,p2,v,h]=angle_difference_power(Saccades1[s1,p1,v,h],Saccades2[s2,p2,v,h],power=power)

                                else:
                                    simsacn=CalcSimAlt(Saccades1[s1,p1,v,h],Saccades2[s2,p2,v,h],Thr=Thr)
                                    if normalize=='add':
                                        SimSacP[s1,s2,p1,p2,v,h]=simsacn/(len(Saccades1[s1,p1,v,h])+len(Saccades2[s2,p2,v,h]))
                                    elif normalize=='mult':
                                        SimSacP[s1,s2,p1,p2,v,h]=simsacn/(len(Saccades1[s1,p1,v,h])*len(Saccades2[s2,p2,v,h]))
 
    return SimSacP


def SacSimPipeline(self,divs=[4,5,7,9],Thr=5,InferS=True,normalize='add',power=1):
    ''' if Thr>0, threshold based similarity ratio,
    if Thr=0, average saccadic angle difference 
    if Thr=0 and power>1, average saccadic angle difference on the value defined by power
    this pipeline compares observers within each stimulus
    '''
    SaccadeObj=self.GetSaccades()
    StimSims=np.zeros((len(divs),self.np))
    StimSimsInd=np.zeros((len(divs),self.ns,self.np))
    SimsAll=[]
    for cd,ndiv in enumerate(divs):
        print(cd,ndiv)
        sacDivSel=self.SaccadeSel(SaccadeObj,ndiv,InferS=InferS)
        SimSacP=self.SacSim1Group(sacDivSel,Thr=Thr,normalize=normalize,power=power)
        StimSimsInd[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(SimSacP,4),3),0)
        StimSims[cd,:]=np.nanmean(np.nanmean(np.nanmean(np.nanmean(SimSacP,4),3),0),0)
        SimsAll.append(SimSacP)
    return StimSims,np.nanmean(StimSimsInd,0),SimsAll

def SacSimPipelineAll2All(self,divs=[4,5,7,9],Thr=5,InferS=True,normalize='add',method='default',power=1, match=False):
    ''' if Thr>0, threshold based similarity ratio,
    if Thr=0, average saccadic angle difference 
    if Thr=0 and power>1, average saccadic angle difference on the value defined by power
    the all to all pipeline compares observers both within and also between stimuli, therefore has a longer runtime
    
    '''
    SaccadeObj=self.GetSaccades()
    StimSims=np.zeros((len(divs),self.np,self.np))
    StimSimsInd=np.zeros((len(divs),self.ns,self.np,self.np))
    SimsAll=[]
    for cd,ndiv in enumerate(divs):
        start_time = time.time()
        print(cd,ndiv)
        sacDivSel=self.SaccadeSel(SaccadeObj,ndiv,InferS=InferS)
        SimSacP=self.SacSim1GroupAll2All(sacDivSel,Thr=Thr,normalize=normalize,method=method,power=power, match=match)
        StimSimsInd[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(SimSacP,5),4),0)
        StimSims[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(np.nanmean(SimSacP,5),4),0),0)
        SimsAll.append(SimSacP)
        end_time=time.time()
        print(f"calculating all to all similarity with div {ndiv}*{ndiv} took {end_time - start_time:.3f} sec")
    return StimSims,np.nanmean(StimSimsInd,0),SimsAll

def ScanpathSim2Groups(self,stim,betwcond,nHor=5,nVer=0,inferS=False,Thr=0,normalize='add', method='default', power=1, match=False):
    if hasattr(self,'subjects')==0:
        self.GetParams()  
    SaccadeObj=self.GetSaccades()
    if type(stim)==str:
        if stim=='all':
            stimn=np.arange(self.ns)  # all stimuli
        else:
            stimn=np.nonzero(self.stimuli==stim)[0]

    else:    
        stimn=np.nonzero(self.stimuli==stim)[0] 
    if nVer==0:
        nVer=nHor  #
    
    SaccadeDiv=self.SaccadeSel(SaccadeObj,nHor=nHor,nVer=nVer,InferS=inferS)    
    SimSacP=self.SacSim1Group(SaccadeDiv,Thr=Thr,normalize=normalize, method=method, power=power, match=match)
    WhichC,WhichCN=self.GetGroups(betwcond)
    Idxs=[]
   
    #Cols=['darkred','cornflowerblue']
    fig,ax=plt.subplots(ncols=2,nrows=2,figsize=(10,8))
                        
    for cc,cond in enumerate(self.Conds):
        Idxs.append(np.nonzero(WhichCN==cond)[0])
    SimVals=np.zeros((2,2))
    SimValsSD=np.zeros((2,2))

    for cgr1,gr1 in enumerate(self.Conds):
        for cgr2,gr2 in enumerate(self.Conds):
            Vals=np.nanmean(np.nanmean(SimSacP[Idxs[cgr1],:,stimn,:,:][:,Idxs[cgr2],:,:],0),0)  
            SimVals[cgr1,cgr2]=np.nanmean(Vals)
            SimValsSD[cgr1,cgr2]=np.nanstd(Vals)
            self.VisGrid(Vals,stim,ax=ax[cgr1,cgr2],cbar=True,inferS=inferS,alpha=.8)
            ax[cgr1,cgr2].set_title(str(gr1)+' '+str(gr2)+' mean= '+str(np.round(SimVals[cgr1,cgr2],3)))
    
    return SimVals,SimValsSD

def ScanpathSimSubject2Subject(self, stim, nHor=5, nVer=0, inferS=False, Thr=5, normalize='add', method='default', match=False):
    '''
    Calculate saccade similarity for a specific stimulus, between each pair of participants.

    Parameters
    ----------
    stim : int
        Index of the stimulus.
    nHor : int, optional
        Number of horizontal divisions. The default is 5.
    nVer : int, optional
        Number of vertical divisions.
        If not provided, nVer=nHor.
    inferS : bool, optional
        Whether to infer the bounds of the AOI rectangles.
        The default is False.
    Thr : int, optional
        Threshold for similarity. The default is 5.
    normalize : str, optional
        If provided, must be 'add' or 'mult'. The default is 'add'.

    Returns
    -------
    SimSacP : np.array
        Saccade similarity matrix.
        Shape: (self.ns, self.ns, nVer, nHor)
    SimVals : np.array
        Mean similarity values.
    SimValsSD : np.array
        Standard deviation of similarity values.

    '''
    if nVer==0:
        nVer=nHor

    SaccadeObj=self.GetSaccades()
    SimVals=np.zeros((self.ns,self.ns))
    SimValsSD=np.zeros((self.ns,self.ns))
    Saccades=self.SaccadeSingleSel(SaccadeObj,nHor=nHor,stim=stim,nVer=nVer,InferS=inferS)

    SimSacP=np.zeros((self.ns,self.ns,nVer,nHor))  
    SimSacP[:]=np.nan
    for s1 in range(self.ns):
        if self.nsac[s1,stim]<=5:
            SimVals[s1,s1]=-np.inf
            SimValsSD[s1,s1]=-np.inf
        for s2 in range(self.ns):
            if s1!=s2:
                if self.nsac[s1,stim]>5 and self.nsac[s2,stim]>5:          
                    for h in range(nHor):
                        for v in range(nVer):
                            if len(Saccades[s1,v,h])>0 and len(Saccades[s2,v,h])>0:                     
                                simsacn=CalcSim(Saccades[s1,v,h],Saccades[s2,v,h],Thr=Thr,method=method, match=match)
                                if method == 'default':
                                    if normalize=='add':
                                        SimSacP[s1,s2,v,h]=simsacn/(len(Saccades[s1,v,h])+len(Saccades[s2,v,h]))
                                    elif normalize=='mult':
                                        SimSacP[s1,s2,v,h]=simsacn/(len(Saccades[s1,v,h])*len(Saccades[s2,v,h]))
                                else:
                                    SimSacP[s1,s2,v,h]=simsacn
                    Vals=SimSacP[s1,s2,:,:]
                    SimVals[s1,s2]=np.nanmean(Vals)
                    SimValsSD[s1,s2]=np.nanstd(Vals)
                else:
                    SimVals[s1,s2]=-np.inf
                    SimValsSD[s1,s2]=-np.inf

    return SimSacP,SimVals,SimValsSD

def ScanpathSimSubj2Groups(self, stim, betwcond, subjects, nHor=5, nVer=0, inferS=False, Thr=0, normalize='add', method='default', power=1, match=False):
    if not hasattr(self, 'subjects'):
        self.GetParams()
    if nVer == 0:
        nVer = nHor
    if not isinstance(subjects, list):
        if isinstance(subjects, int):
            subjects = [subjects]
        else:
            raise ValueError('Invalid subjects')
    
    SaccadeObj = self.GetSaccades()
    Saccades = self.SaccadeSingleSel(SaccadeObj, nHor=nHor, stim=stim, nVer=nVer, InferS=inferS)
    _, WhichCN = self.GetGroups(betwcond)
    
    # Get unique conditions
    condition = np.unique(WhichCN)
    # Initialize SimVals as a list of lists to hold concatenated values
    SimVals = [[[] for _ in range(len(condition))] for _ in range(len(subjects))]

    # If we give all subjects as input, we dont need to check that s2 is not in subjects (ie we need to test all subjects)
    # Otherwise we need to check that s2 is not in the (test) subjects
    # In this way we ensure to split the data in training and test set
    if len(subjects) == self.ns:
        s = []
    else:
        s = subjects
    
    # Calculate similarity between subjects of different groups and the selected subject
    for s1_idx, s1 in enumerate(subjects):
        for s2 in range(self.ns):
            if s1 != s2 and s2 not in s:
                tot_val = []
                for h in range(nHor):
                    for v in range(nVer):
                        if len(Saccades[s1, v, h]) > 0 and len(Saccades[s2, v, h]) > 0:
                            group = WhichCN[s2]
                            if Thr == 0:
                                if method == 'peak180':
                                    val = angle_difference_peak180(Saccades[s1, v, h], Saccades[s2, v, h], power=power, match=match)
                                elif method == 'power':
                                    val = angle_difference_power(Saccades[s1, v, h], Saccades[s2, v, h], power=power)
                                elif method == 'Kuiper':
                                    val = KuiperStat(Saccades[s1, v, h], Saccades[s2, v, h])
                                else:
                                    raise ValueError('Invalid method')
                            else:
                                if method == 'default':
                                    simsacn = CalcSimAlt(Saccades[s1, v, h], Saccades[s2, v, h], Thr=Thr)
                                    if normalize == 'add':
                                        val = simsacn / (len(Saccades[s1, v, h]) + len(Saccades[s2, v, h]))
                                    elif normalize == 'mult':
                                        val = simsacn / (len(Saccades[s1, v, h]) * len(Saccades[s2, v, h]))
                                elif method == 'cosine':
                                    val = CosineSim(Saccades[s1, v, h], Saccades[s2, v, h], Thr=Thr)
                            tot_val.append(val)
                    if len(tot_val) > 0:
                        SimVals[s1_idx][group].append(np.nanmean(tot_val))
    return SimVals

def SacSimSubj2Group(self, Saccades, WhichCN, subjects, Thr=0, normalize='add', method='default', power=1, match=False, nosubj=[], SingleROI=False):
    if not hasattr(self, 'subjects'):
        self.GetParams()
    if not isinstance(subjects, list):
        if isinstance(subjects, int):
            subjects = [subjects]
        else:
            raise ValueError('Invalid subjects')

    nVer=np.shape(Saccades)[1]
    nHor=np.shape(Saccades)[2]
    
    # Get unique conditions
    condition = np.unique(WhichCN)
    # Initialize SimVals as a list of lists to hold concatenated values
    SimVals = [[[] for _ in range(len(condition))] for _ in range(len(subjects))]
    if SingleROI:
        # Initialize SimValsROI for storing similarity values for each ROI (nVer x nHor)
        SimValsROI = [[[[np.nan for _ in range(nHor)] for _ in range(nVer)] for _ in range(len(condition))] for _ in range(len(subjects))]

    # If we give all subjects as input, we dont need to check that s2 is not in subjects (ie we need to test all subjects)
    # Otherwise we need to check that s2 is not in the (test) subjects
    # In this way we ensure to split the data in training and test set
    if len(subjects) == self.ns and len(nosubj) == 0:
        s = []
    elif len(nosubj) > 0:
        s = nosubj
    else:
        s = subjects

    # Calculate similarity between subjects of different groups and the selected subject
    for s1_idx, s1 in enumerate(subjects):
        for s2 in range(self.ns):
            if s1 != s2 and s2 not in s:
                tot_val = []
                for h in range(nHor):
                    for v in range(nVer):
                        if len(Saccades[s1, v, h]) > 0 and len(Saccades[s2, v, h]) > 0:
                            group = WhichCN[s2]
                            if Thr == 0:
                                if method == 'peak180':
                                    val = angle_difference_peak180(Saccades[s1, v, h], Saccades[s2, v, h], power=power, match=match)
                                elif method == 'power':
                                    val = angle_difference_power(Saccades[s1, v, h], Saccades[s2, v, h], power=power)
                                elif method == 'Kuiper':
                                    val = KuiperStat(Saccades[s1, v, h], Saccades[s2, v, h])
                                else:
                                    raise ValueError('Invalid method')
                            else:
                                if method == 'default':
                                    simsacn = CalcSimAlt(Saccades[s1, v, h], Saccades[s2, v, h], Thr=Thr)
                                    if normalize == 'add':
                                        val = simsacn / (len(Saccades[s1, v, h]) + len(Saccades[s2, v, h]))
                                    elif normalize == 'mult':
                                        val = simsacn / (len(Saccades[s1, v, h]) * len(Saccades[s2, v, h]))
                                elif method == 'cosine':
                                    val = CosineSim(Saccades[s1, v, h], Saccades[s2, v, h], Thr=Thr)
                            if SingleROI:
                                if isinstance(SimValsROI[s1_idx][group][v][h], np.ndarray):
                                    SimValsROI[s1_idx][group][v][h] = np.append(SimValsROI[s1_idx][group][v][h], val)
                                else:
                                    SimValsROI[s1_idx][group][v][h] = np.array([val])
                            tot_val.append(val)
                SimVals[s1_idx][group].append(np.nanmean(tot_val))

    if SingleROI:
        return SimVals, SimValsROI
    else:
        return SimVals
    

def SacSimSubj2GroupPlusFeature(self, stim, WhichCN, subjects, nHor=5, nVer=0, nHor_f=0, nVer_f=0, inferS=False, Thr=0, normalize='add', method='default', power=1, match=False, nosubj=[]):
    if not hasattr(self, 'subjects'):
        self.GetParams()
    if nVer == 0:
        nVer = nHor
    if not isinstance(subjects, list):
        if isinstance(subjects, int):
            subjects = [subjects]
        else:
            raise ValueError('Invalid subjects')
        
    def SaccadeAndFeatures(self, SaccadeObj, nHor, stim, nVer=0, InferS=False):
        if nVer == 0:
            nVer = nHor  # if number of vertical divisions not provided, use same as horizontal

        SaccadeAOIAngles = []   # store angles per saccade per ROI cell (even if the saccade crosses multiple cells)
        SaccadeAOIAnglesCross = []  # for saccades that cross >1 ROI

        # Create AOI rectangles. Here we assume CreatAoiRects returns a nested structure,
        # e.g., AOIRects[stim][h][v] is an AOI rectangle object that has a .Cross(LinePoints) method.
        if InferS:
            if not hasattr(self, 'boundsX'):
                print('Running descriptives to get bounds')
                self.RunDescriptiveFix()  
            AOIRects = CreatAoiRects(nHor, nVer, self.boundsX, self.boundsY)
        else:
            AOIRects = CreatAoiRects(nHor, nVer, self.x_size, self.y_size, allsame=self.np)

        ns = self.ns  

        Saccades = np.zeros(((ns, nVer, nHor)), dtype=np.ndarray)  # Array of saccades crossing each AOI rectangle for each trial and participant
        for s in np.arange(self.ns):
            SaccadeAOIAngles.append(np.zeros((int(self.nsac[s, stim]), nVer, nHor)))
            SaccadeAOIAngles[s][:] = np.nan
            SaccadeAOIAnglesCross.append(np.zeros((int(self.nsac[s, stim]), nVer, nHor)))
            SaccadeAOIAnglesCross[s][:] = np.nan
            
            for sac in range(len(SaccadeObj[s][stim])):
                SaccadeDots = SaccadeObj[s][stim][sac].LinePoints()
                
                for h in range(nHor):
                    for v in range(nVer):
                        if AOIRects[stim][h][v].Cross(SaccadeDots):
                            SaccadeAOIAngles[s][sac, v, h] = SaccadeObj[s][stim][sac].Angle()  # Get angle of the saccade
                                            

                # Select saccades that cross multiple cells
                if np.sum(SaccadeAOIAngles[s][sac, :, :] > 0) > 1:
                    SaccadeAOIAnglesCross[s][sac, :, :] = SaccadeAOIAngles[s][sac, :, :]
                else:
                    # search index of all the not nan values of SaccadeAOIAngles[s][p][sac,:,:]
                    idxs = np.argwhere(~np.isnan(SaccadeAOIAngles[s][sac, :, :]))
                    if len(idxs) > 0:
                        for idx in idxs:
                            v, h = idx
                            if isinstance(Saccades[s,v,h], np.ndarray):
                                Saccades[s,v,h]=np.append(Saccades[s,v,h],SaccadeAOIAngles[s][sac,v,h])
                            else:
                                Saccades[s,v,h]=np.array([SaccadeAOIAngles[s][sac,v,h]])

            # store saccades that cross multiple AOI rectangles
            for h in range(nHor):
                for v in range(nVer):
                    if np.sum(np.isfinite(SaccadeAOIAnglesCross[s][:,v,h]))>0:
                        if isinstance(Saccades[s,v,h], np.ndarray):
                            Saccades[s,v,h]=np.append(Saccades[s,v,h],SaccadeAOIAnglesCross[s][~np.isnan(SaccadeAOIAnglesCross[s][:,v,h]),v,h])
                        else:
                            Saccades[s,v,h]=np.array(SaccadeAOIAnglesCross[s][~np.isnan(SaccadeAOIAnglesCross[s][:,v,h]),v,h])
                    elif not isinstance(Saccades[s,v,h], np.ndarray):
                        Saccades[s,v,h]=np.array([])

        Features = np.empty((ns, nVer, nHor), dtype=object)
        for s in range(ns):
            cords = []
            for sac in range(len(SaccadeObj[s][stim])):
                x1,y1,x2,y2 = SaccadeObj[s][stim][sac].Coords()
                if cords == []:
                    cords.append((x1,y1))
                else:
                    cords.append((x1,y1))
            cords.append((x2,y2))
            for h in range(nVer):
                for v in range(nHor):
                    angles = np.array(Saccades[s, v, h])
                    
                    # Mean and STD for angles (if there is any data)
                    if angles.size > 0:
                        mean_angle = np.nanmean(angles)
                        std_angle = np.nanstd(angles)
                    else:
                        mean_angle = np.nan
                        std_angle = np.nan

                    #Count of fixations in the cell
                    landing_first_fix = [np.nan, np.nan]
                    fixation_count = 0
                    duration = np.array([])
                    for x,y in cords:
                        if AOIRects[stim][h][v].Contains(x,y):
                            if fixation_count == 0:
                                landing_first_fix = [x,y]
                            fixation_count += 1
                            # self.data[['subjectID','duration','mean_x','mean_y']]
                            # check fixation duration where subjectID = DyslexiaDat.subjects[s] and mean_x, mean_y are within the AOI
                            duration = np.append(duration, self.data.loc[(self.data['subjectID'] == self.subjects[s]) & (self.data['mean_x'] == x) & (self.data['mean_y'] == y)]['duration'].values)

                    # Number of revisits – incoming saccades hitting this ROI from outside
                    revisits = 0
                    exits = False
                    for sac in range(len(SaccadeObj[s][stim])):
                        x1,y1,x2,y2 = SaccadeObj[s][stim][sac].Coords()
                        if AOIRects[stim][h][v].Contains(x1,y1) and not AOIRects[stim][h][v].Contains(x2,y2):
                            exits = True
                        if not AOIRects[stim][h][v].Contains(x1,y1) and AOIRects[stim][h][v].Contains(x2,y2) and exits:
                            revisits += 1

                    Features[s, v, h] = {'mean_angle': mean_angle,
                                            'std_angle': std_angle,
                                            'fixation_count': fixation_count,
                                            'mean_fixation_duration': np.nanmean(duration),
                                            'number_of_revisits': revisits,
                                            'landing_first_coord':  (landing_first_fix[0] + landing_first_fix[1]) * (landing_first_fix[0] + landing_first_fix[1] + 1) / 2 + landing_first_fix[1] # Normalize and Interleave (Cantor Pairing Function)
                                            }
                    
        return Saccades, Features
    
    if nHor_f == 0:
        nHor_f = nHor

    if nVer_f == 0:
        nVer_f = nHor_f
    
    SaccadeObj = self.GetSaccades()
    Saccades, Features = SaccadeAndFeatures(self, SaccadeObj, nHor=nHor_f, stim=stim, nVer=nVer_f, InferS=inferS)

    if not (nHor_f == nHor and nVer_f == nVer):
        Saccades = self.SaccadeSingleSel(SaccadeObj, nHor=nHor, stim=stim, nVer=nVer, InferS=inferS)
    
    # Get unique conditions
    condition = np.unique(WhichCN)
    # Initialize SimVals as a list of lists to hold concatenated values
    SimVals = [[[] for _ in range(len(condition))] for _ in range(len(subjects))]
    SimValsROI = [[[[np.nan for _ in range(nHor)] for _ in range(nVer)] for _ in range(len(condition))] for _ in range(len(subjects))]


    # If we give all subjects as input, we dont need to check that s2 is not in subjects (ie we need to test all subjects)
    # Otherwise we need to check that s2 is not in the (test) subjects
    # In this way we ensure to split the data in training and test set
    if len(subjects) == self.ns and len(nosubj) == 0:
        s = []
    elif len(nosubj) > 0:
        s = nosubj
    else:
        s = subjects
        
    # Calculate similarity between subjects of different groups and the selected subject
    for s1_idx, s1 in enumerate(subjects):
        for s2 in range(self.ns):
            if s1 != s2 and s2 not in s:
                tot_val = []
                for h in range(nHor):
                    for v in range(nVer):
                        if len(Saccades[s1, v, h]) > 0 and len(Saccades[s2, v, h]) > 0:
                            group = WhichCN[s2]
                            if Thr == 0:
                                if method == 'peak180':
                                    val = angle_difference_peak180(Saccades[s1, v, h], Saccades[s2, v, h], power=power, match=match)
                                elif method == 'power':
                                    val = angle_difference_power(Saccades[s1, v, h], Saccades[s2, v, h], power=power)
                                elif method == 'Kuiper':
                                    val = KuiperStat(Saccades[s1, v, h], Saccades[s2, v, h])
                                else:
                                    raise ValueError('Invalid method')
                            else:
                                if method == 'default':
                                    simsacn = CalcSimAlt(Saccades[s1, v, h], Saccades[s2, v, h], Thr=Thr)
                                    if normalize == 'add':
                                        val = simsacn / (len(Saccades[s1, v, h]) + len(Saccades[s2, v, h]))
                                    elif normalize == 'mult':
                                        val = simsacn / (len(Saccades[s1, v, h]) * len(Saccades[s2, v, h]))
                                elif method == 'cosine':
                                    val = CosineSim(Saccades[s1, v, h], Saccades[s2, v, h], Thr=Thr)
                            if isinstance(SimValsROI[s1_idx][group][v][h], np.ndarray):
                                SimValsROI[s1_idx][group][v][h] = np.append(SimValsROI[s1_idx][group][v][h], val)
                            else:
                                SimValsROI[s1_idx][group][v][h] = np.array([val])
                            tot_val.append(val)
                if len(tot_val) > 0:
                    SimVals[s1_idx][group].append(np.nanmean(tot_val))

    if nHor_f == nHor and nVer_f == nVer:
        SimAndFeatureVals = np.empty((len(subjects), nVer, nHor), dtype=object)
        for s1_idx, s1 in enumerate(subjects):
            for v in range(nVer):
                for h in range(nHor):
                    feature_val = Features[s1, v, h]
                    mean_g0 = np.nanmean(SimValsROI[s1_idx][0][v][h])
                    std_g0 = np.nanstd(SimValsROI[s1_idx][0][v][h])
                    mean_g1 = np.nanmean(SimValsROI[s1_idx][1][v][h])
                    std_g1 = np.nanstd(SimValsROI[s1_idx][1][v][h])
                    SimAndFeatureVals[s1_idx][v][h] = [feature_val['fixation_count'] if not np.isnan(feature_val['fixation_count']) else 0,
                                                        feature_val['mean_fixation_duration'] if not np.isnan(feature_val['mean_fixation_duration']) else 0,
                                                        feature_val['number_of_revisits'],
                                                        feature_val['landing_first_coord'] if not np.isnan(feature_val['landing_first_coord']) else -1,
                                                        mean_g0 if not np.isnan(mean_g0) else 0,
                                                        mean_g1 if not np.isnan(mean_g1) else 0,
                                                        std_g0 if not np.isnan(std_g0) else 0,
                                                        std_g1 if not np.isnan(std_g1) else 0
                                                    ]
        return SimAndFeatureVals
    else:
        FeatureVals = np.empty((len(subjects), nVer_f, nHor_f), dtype=object)
        for s1_idx, s1 in enumerate(subjects):
            for v in range(nVer_f):
                for h in range(nHor_f):
                    feature_val = Features[s1, v, h]
                    FeatureVals[s1_idx][v][h] = [feature_val['fixation_count'] if not np.isnan(feature_val['fixation_count']) else 0,
                                                        feature_val['mean_fixation_duration'] if not np.isnan(feature_val['mean_fixation_duration']) else 0,
                                                        feature_val['number_of_revisits'],
                                                        feature_val['landing_first_coord'] if not np.isnan(feature_val['landing_first_coord']) else -1
                                                        ]
        
        SimsVals = np.empty((len(subjects), nVer, nHor), dtype=object)
        for s1_idx, s1 in enumerate(subjects):
            for v in range(nVer):
                for h in range(nHor):
                    mean_g0 = np.nanmean(SimValsROI[s1_idx][0][v][h])
                    std_g0 = np.nanstd(SimValsROI[s1_idx][0][v][h])
                    mean_g1 = np.nanmean(SimValsROI[s1_idx][1][v][h])
                    std_g1 = np.nanstd(SimValsROI[s1_idx][1][v][h])
                    SimsVals[s1_idx][v][h] = [mean_g0 if not np.isnan(mean_g0) else 0,
                                                mean_g1 if not np.isnan(mean_g1) else 0,
                                                std_g0 if not np.isnan(std_g0) else 0,
                                                std_g1 if not np.isnan(std_g1) else 0
                                            ]
        
        return FeatureVals, SimsVals