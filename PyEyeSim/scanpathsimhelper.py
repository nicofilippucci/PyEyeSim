
import numpy as np
from numpy import mat, matlib
#from scipy import stats,ndimage
#import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine



def CheckCoor(AOIs,FixLoc):
    ''' to check if fixation coordinates are within AOI'''  
    for coor in range(len(AOIs)-1):
        if (FixLoc>AOIs[coor]) and (FixLoc<=AOIs[coor+1]):
            AOI=coor
            break
        else: # if gaze out of screen
            AOI=np.nan                      
    return AOI 


def AOIbounds(start,end,nDiv):  
    ''' calcuale AOI bounds, linearly spaced from: start to: end, for nDiv number of divisions'''
    return np.linspace(start,end,nDiv+1)  


def CreatAoiRects(nHorD,nVerD,BoundsX,BoundsY,allsame=0):
    AOIRects=[]
    if type(BoundsX)==int:
        for a in range(allsame):
            AOIRects.append([])
            AOIboundsH=AOIbounds(0,BoundsX,nHorD)
            AOIboundsV=AOIbounds(0,BoundsY,nVerD)
            for h in range(nHorD):
                AOIRects[a].append([])
                for v in range(nVerD):
                    AOIRects[a][h].append(Rect(AOIboundsH[h],AOIboundsV[v],AOIboundsH[h+1],AOIboundsV[v+1]))  # store AOIs as objects
    else:
        for p in range(np.shape(BoundsX)[0]):
            AOIboundsH=AOIbounds(BoundsX[p,0],BoundsX[p,1],nHorD)
            AOIboundsV=AOIbounds(BoundsY[p,0],BoundsY[p,1],nVerD)
          #  print(AOIboundsH)
           # print(AOIboundsV)
            AOIRects.append([])
            for h in range(nHorD):
                AOIRects[p].append([])
                for v in range(nVerD):
                    AOIRects[p][h].append(Rect(AOIboundsH[h],AOIboundsV[v],AOIboundsH[h+1],AOIboundsV[v+1]))  # store AOIs as objects
    return AOIRects



class Rect:
    def __init__(self, x1,y1,x2,y2):
        if x1 > x2: 
            x1,x2=x2,x1 # to start with smaller x
        if y1 > y2:
            y1,y2,=y2,y1 # to start with smaller y
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2    
    def Vis(self,alp=.2,Col='b'):
        plt.plot([self.x1,self.x1],[self.y1,self.y2],alpha=alp,color=Col)
        plt.plot([self.x2,self.x2],[self.y1,self.y2],alpha=alp,color=Col)
        plt.plot([self.x1,self.x2],[self.y1,self.y1],alpha=alp,color=Col)
        plt.plot([self.x1,self.x2],[self.y2,self.y2],alpha=alp,color=Col)
        
    def Contains(self,x,y):
        if x>=self.x1 and x<self.x2 and y>=self.y1 and y<self.y2:              
            return True
        else:
            return False
        
    def Cross(self,LinePoints):
        CrossXidx=np.nonzero((LinePoints[0]>=self.x1)&(LinePoints[0]<self.x2))
        if len(CrossXidx)>0:
            if any(LinePoints[1][CrossXidx]>=self.y1)==True and any(LinePoints[1][CrossXidx]<self.y2)==True:
                return True
            else:
                return False
        else:
            return False
                     
    def Contains2(self,LinePoints):
      
         CrossXidx=np.nonzero((LinePoints[:,0]>self.x1)&(LinePoints[:,0]<self.x2))
         CrossYidx=np.nonzero((LinePoints[:,1]>self.y1)&(LinePoints[:,1]<self.y2))
         return  np.intersect1d(CrossXidx,CrossYidx)    
     
        
class SaccadeLine:
    def __init__(self, coordvect):  # create a saccade object with start and end pixels
        self.x1=coordvect[0]
        self.y1=coordvect[1]
        self.x2=coordvect[2]
        self.y2=coordvect[3]
    def Coords(self):   
        return self.x1,self.y1,self.x2,self.y2
    def length(self):   # length of saccade
        return int(np.sqrt((self.x2-self.x1)**2+(self.y2-self.y1)**2))
    
    def lengthHor(self):  # horizontal length of saccade
        return  np.abs(self.x2-self.x1)
    
    def lengthVer(self):  # vertical length of saccade
        return  np.abs(self.y2-self.y1)
    
    def Angle(self):   # angle of saccade (0-360 deg)
        Ang=np.degrees(np.arccos((self.x2-self.x1)/self.length()))  #calculate angel of saccades
        if self.y2 < self.y1:  # if downward saccade
            Ang=360-Ang  
        return Ang
    def Vis(self,alp=.2,Col='k'):  # plot saccade
        plt.plot([self.x1,self.x2],[self.y1,self.y2],alpha=alp,color=Col)
        return
    
    def LinePoints(self):  # use dots with density of 1dot/1pixel to approximate line.
        LineX=np.linspace(self.x1,self.x2,self.length())
        LineY=np.linspace(self.y1,self.y2,self.length())
        return LineX,LineY
    
    

def CalcSim(saccades1,saccades2,Thr=5, method='default', power=1, match=False):
    ''' calculcate angle based similarity for two arrays of saccade objects (for each cell)'''
    if method=='default':
        saccades1[saccades1>180]-=180
        saccades2[saccades2>180]-=180
        A=saccades1[:,np.newaxis]
        B=saccades2[np.newaxis,:]
        simsacn=np.sum(np.abs(A-B)<Thr)
        return simsacn
    elif method=='power':
        return angle_difference_power(saccades1, saccades2, power=power)
    elif method=='peak180':
        return angle_difference_peak180(saccades1, saccades2, power=power, match=match)
    elif method=='Kuiper':
        return KuiperStat(saccades1, saccades2)
    else:
        raise ValueError('Invalid method: {}'.format(method))

def CalcSimAlt(saccades1,saccades2,Thr=5):
    ''' calculcate angle based similarity for two arrays of saccade objects (for each cell)
    all angles are transformed to below 180 degrees before comparison'''
    saccades1[saccades1>180]-=180
    saccades2[saccades2>180]-=180
    A=saccades1[:,np.newaxis]
    B=saccades2[np.newaxis,:]
    simsacn=np.sum(np.abs(A-B)<Thr)
    return simsacn


def KuiperStat(saccades1,saccades2):
    sample1 = np.sort(saccades1)
    sample2 = np.sort(saccades2)
    all_data = np.sort(np.concatenate((sample1, sample2)))
    ecdf1 = np.searchsorted(sample1, all_data, side='right') / len(sample1)
    ecdf2 = np.searchsorted(sample2, all_data, side='right') / len(sample2)
    d_plus = np.max(ecdf1 - ecdf2)
    d_minus = np.max(ecdf2 - ecdf1)
    v = d_plus + d_minus
    return 1 - v

def angle_difference_power(saccades1,saccades2,power=1):
    ''' this methods calculates differences between 0 and 90 degrees, between all pairs of saccades, than normalizes to the range 0-1, than averages
    by default it is just the mean absolute difference, but can be used for different exponentials by changing power from the default of 1'''
    diffs = np.abs(saccades1[:, np.newaxis] - saccades2) % 360
    mask = diffs > 180
    diffs[mask] = 360 - diffs[mask]
    return np.mean(np.abs((np.minimum(diffs, 180 - diffs)/90))**power)


def angle_difference_peak180(saccades1, saccades2, power=1, match=False):
    """
    Calculates differences between 0 and 180 degrees, with a peak of 1 at 180 degrees.
    Normalizes to the range 0–1, with a decrease back to 0 after 180 degrees.
    Allows for custom exponentials via the `power` parameter.
    
    If `matching` is True, calculates the difference between the two saccade arrays
    based on the array order, penalizing extra saccades.
    
    Parameters:
        saccades1: np.ndarray
            A 1D array of saccade angles in degrees.
        saccades2: np.ndarray
            A 1D array of saccade angles in degrees.
        power: int or float
            The exponent applied to the similarity values. Defaults to 1.
        matching: bool
            If True, matches saccades based on minimum differences, penalizing extras.
    
    Returns:
        float: The mean similarity score, normalized between 0 and 1.
    """
    if match:
        s1 = np.array(saccades1)
        s2 = np.array(saccades2)

        if len(s1) == 0 and len(s2) == 0:
            similarity_score = 0.0 
        elif len(s1) == 0 or len(s2) == 0:
            similarity_score = 1.0
        else:
            if len(s1) > len(s2):
                s1, s2 = s2, s1

            max_shift = len(s2) - len(s1)
            best_score = np.inf
            for shift in range(max_shift + 1):
                aligned_s2 = s2[shift:shift + len(s1)]
                diffs = np.abs(s1 - aligned_s2) % 360
                diffs = np.minimum(diffs, 360 - diffs)
                normalized = diffs / 180  # 0 = perfect match, 1 = opposite direction
                penalty = (len(s2) - len(s1)) / len(s2)
                score = np.mean(normalized ** power)
                score += penalty

                if score < best_score:
                    best_score = score

            similarity_score = best_score
    else:
        diffs = np.abs(saccades1[:, np.newaxis] - saccades2) % 360
        diffs = np.minimum(diffs, 360 - diffs) 
        normalized_diffs = (np.abs(diffs) / 180) 
        symmetric_diffs = 1 - np.abs(1 - normalized_diffs) 
        similarity_score = np.mean(symmetric_diffs**power)
    
    return similarity_score


def CosineSim(saccades1,saccades2,Thr):
   
    bin_edges = np.linspace(0, 360, int(360/Thr)+1)  # 36 bins of 10° each + endpoint
    
    # Compute histograms (normalize to get probability distributions)
    hist1, _ = np.histogram(saccades1, bins=bin_edges, density=True)
    hist2, _ = np.histogram(saccades2, bins=bin_edges, density=True)
        
    # Compute cosine similarity
    return  1 - cosine(hist1, hist2)
    
