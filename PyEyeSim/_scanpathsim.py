# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:54:38 2024

@author: aratoj87
"""

import numpy as np
from numpy import matlib
from scipy import stats,ndimage
import pandas as pd
import matplotlib.pyplot as plt

from .scanpathsimhelper import AOIbounds,CreatAoiRects,Rect,SaccadeLine,CalcSim, CheckCoor







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

            for h in range(nHor):
                for v in range(nVer):
                    if np.sum(np.isfinite(SaccadeAOIAnglesCross[s][p][:,v,h]))>0:
                        Saccades[s,p,v,h]=np.array(SaccadeAOIAnglesCross[s][p][~np.isnan(SaccadeAOIAnglesCross[s][p][:,v,h]),v,h])
                    else:
                        Saccades[s,p,v,h]=np.array([])
    return Saccades

def SaccadeSel(self, stim, SaccadeObj, nHor, nVer=0, InferS=False): 
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

        # Store saccades that cross multiple AOI rectangles
        for h in range(nHor):
            for v in range(nVer):
                if np.sum(np.isfinite(SaccadeAOIAnglesCross[s][:, v, h])) > 0:
                    Saccades[s, v, h] = np.array(SaccadeAOIAnglesCross[s][~np.isnan(SaccadeAOIAnglesCross[s][:, v, h]), v, h])
                else:
                    Saccades[s, v, h] = np.array([])
    
    return Saccades


def SacSim1Group(self,Saccades,Thr=5,p='all',normalize='add'):
    ''' calculate saccade similarity for each stimulus, between each pair of participants ,
    needs saccades stored as PyEyeSim saccade objects stored in AOIs as input,
    vertical and horizontal dimensions are inferred from the input
    Thr=5: threshold for similarity
    normalize, if provided must be add or mult '''
    
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
                                        
                                    simsacn=CalcSim(Saccades[s1,p1,v,h],Saccades[s2,p1,v,h],Thr=Thr)
                                    if normalize=='add':
                                        SimSacP[s1,s2,p1,v,h]=simsacn/(len(Saccades[s1,p1,v,h])+len(Saccades[s2,p1,v,h]))
                                    elif normalize=='mult':
                                        SimSacP[s1,s2,p1,v,h]=simsacn/(len(Saccades[s1,p1,v,h])*len(Saccades[s2,p1,v,h]))
 
    return SimSacP

  
def SacSim1GroupAll2All(self,Saccades,Thr=5,p='all',normalize='add'):
    ''' calculate saccade similarity for each stimulus, and across all stimuli, between each pair of participants ,
    needs saccades stored as PyEyeSim saccade objects stored in AOIs as input,
    vertical and horizontal dimensions are inferred from the input
    Thr=5: threshold for similarity
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
                                            
                                        simsacn=CalcSim(Saccades[s1,p1,v,h],Saccades[s2,p2,v,h],Thr=Thr)
                                        if normalize=='add':
                                            SimSacP[s1,s2,p1,p2,v,h]=simsacn/(len(Saccades[s1,p1,v,h])+len(Saccades[s2,p2,v,h]))
                                        elif normalize=='mult':
                                            SimSacP[s1,s2,p1,p2,v,h]=simsacn/(len(Saccades[s1,p1,v,h])*len(Saccades[s2,p2,v,h]))
 
    return SimSacP




def SacSimPipeline(self,divs=[4,5,7,9],Thr=5,InferS=True,normalize='add'):
    SaccadeObj=self.GetSaccades()
    StimSims=np.zeros((len(divs),self.np))
    StimSimsInd=np.zeros((len(divs),self.ns,self.np))
    SimsAll=[]
    for cd,ndiv in enumerate(divs):
        print(cd,ndiv)
        sacDivSel=self.SaccadeSel(SaccadeObj,ndiv,InferS=InferS)
        SimSacP=self.SacSim1Group(sacDivSel,Thr=Thr,normalize=normalize)
        StimSimsInd[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(SimSacP,4),3),0)
        StimSims[cd,:]=np.nanmean(np.nanmean(np.nanmean(np.nanmean(SimSacP,4),3),0),0)
        SimsAll.append(SimSacP)
    return StimSims,np.nanmean(StimSimsInd,0),SimsAll

def SacSimPipelineAll2All(self,divs=[4,5,7,9],Thr=5,InferS=True,normalize='add'):
    SaccadeObj=self.GetSaccades()
    StimSims=np.zeros((len(divs),self.np,self.np))
    StimSimsInd=np.zeros((len(divs),self.ns,self.np,self.np))
    SimsAll=[]
    for cd,ndiv in enumerate(divs):
        print(cd,ndiv)
        sacDivSel=self.SaccadeSel(SaccadeObj,ndiv,InferS=InferS)
        SimSacP=self.SacSim1GroupAll2All(sacDivSel,Thr=Thr,normalize=normalize)
        StimSimsInd[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(SimSacP,5),4),0)
        StimSims[cd,:,:]=np.nanmean(np.nanmean(np.nanmean(np.nanmean(SimSacP,5),4),0),0)
        SimsAll.append(SimSacP)
    return StimSims,np.nanmean(StimSimsInd,0),SimsAll

def ScanpathSim2Groups(self,stim,betwcond,nHor=5,nVer=0,inferS=False,Thr=5,normalize='add'):
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
    SimSacP=self.SacSim1Group(SaccadeDiv,Thr=Thr,normalize=normalize)
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

def ScanpathSimSubject2Subject(self, stim, nHor=5, nVer=0, inferS=False, Thr=5, normalize='add'):
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
    Saccades=self.SaccadeSel(stim,SaccadeObj,nHor=nHor,nVer=nVer,InferS=inferS)

    SimSacP=np.zeros((self.ns,self.ns,nVer,nHor))  
    SimSacP[:]=np.nan
    for s1 in range(self.ns):
        for s2 in range(self.ns):
            if s1!=s2:
                if self.nsac[s1,stim]>5 and self.nsac[s2,stim]>5:                    
                    for h in range(nHor):
                        for v in range(nVer):
                            if len(Saccades[s1,v,h])>0 and len(Saccades[s2,v,h])>0:                     
                                simsacn=CalcSim(Saccades[s1,v,h],Saccades[s2,v,h],Thr=Thr)
                                if normalize=='add':
                                    SimSacP[s1,s2,v,h]=simsacn/(len(Saccades[s1,v,h])+len(Saccades[s2,v,h]))
                                elif normalize=='mult':
                                    SimSacP[s1,s2,v,h]=simsacn/(len(Saccades[s1,v,h])*len(Saccades[s2,v,h]))
                    Vals=SimSacP[s1,s2,:,:]
                    SimVals[s1,s2]=np.nanmean(Vals)
                    SimValsSD[s1,s2]=np.nanstd(Vals)

    return SimSacP,SimVals,SimValsSD
