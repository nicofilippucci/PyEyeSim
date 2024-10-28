import numpy as np
import matplotlib.pyplot as plt
from .hmmhelper import DiffCompsHMM,FitScoreHMMGauss
from .visualhelper import draw_ellipse
import hmmlearn.hmm  as hmm
from scipy.spatial.distance import cdist
from IPython.utils import io
 # hmm related functions start here
def DataArrayHmm(self,stim,group=-1,tolerance=20,verb=True):
    ''' HMM data arrangement, for the format required by hmmlearn
    tolarance control the numbers of pixels, where out of stimulus fixations are still accepted, currently disabled as not yet adapted for changing bounds
    therefore, participants with invalid fixations are not yet removed
    
    verb-- verbose-- print missing participants, too much printing for leave one out cross validation'''
    
    XX=np.array([])
    YY=np.array([])
    Lengths=np.array([],dtype=int)
    self.suseHMM=np.array([],dtype=int)
    #print('org data for stim')
    for cs,s in enumerate(self.subjects):
        if group!=-1:
            if self.whichC[cs]==group:
                useS=True
            else:
                useS=False
        else:
            useS=True
        if useS:
            fixX,fixY=self.GetFixationData(s,stim)
          #  print(cs,s,fixX)
            if any(fixX<-tolerance) or any(fixX>self.x_size+tolerance) or any(fixY<-tolerance)or any(fixY>self.y_size+tolerance):
                if verb:
                    print('invalid fixation location for subj', s)
           # else:
            if len(fixX)>2:
                XX=np.append(XX,fixX)
                YY=np.append(YY,fixY)
                Lengths=np.append(Lengths,len(fixX))
                self.suseHMM=np.append(self.suseHMM,s)
            elif verb:
                print('not enough fixations for subj', s)

    return XX,YY,Lengths


def MyTrainTest(self,Dat,Lengths,ntest,vis=0,rand=1,totest=0):
    ''' separate hidden markov model dataset, into training and test set'''
    if rand:
        totest=np.random.choice(np.arange(len(Lengths)),size=ntest,replace=False)
    else:
        totest=np.array([totest],dtype=int)
    Idxs=np.cumsum(Lengths)
    lenTrain=np.array([],dtype=int)
    lenTest=np.array([],dtype=int)
    DatTest=np.zeros((0,2))
    DatTr=np.zeros((0,2)) 
    for ci in range(len(Lengths)):
        if ci==0:
            start=0
        else:
            start=Idxs[ci-1]
        if ci in totest:
            DatTest=np.vstack((DatTest,Dat[start:Idxs[ci],:]))
            lenTest=np.append(lenTest,Lengths[ci])
        else:
            DatTr=np.vstack((DatTr,Dat[start:Idxs[ci],:]))
            lenTrain=np.append(lenTrain,Lengths[ci])
    if vis:
        self.MyTrainTestVis(DatTr,DatTest,lenTrain,lenTest,totest)
    return DatTr,DatTest,lenTrain,lenTest   



def FitLOOHMM(self,ncomp,stim,covar='full',verb=False):
    ''' fit HMM, N subject times, leaving out once each time
    ncomp: number of components
    stim: stimulus code 
    covar: covariance type 'full' or  'tied' '''
    NTest=1
    xx,yy,lengths=self.DataArrayHmm(stim,tolerance=80,verb=verb)
    Dat=np.column_stack((xx,yy))
    ScoresLOO=np.zeros(len(self.suseHMM))
    if verb:
        print('num valid observers',len(ScoresLOO))
    for cs,s in enumerate(self.suseHMM):
        DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,NTest,vis=0,rand=0,totest=cs)
        HMMfitted,sctr,scte=FitScoreHMMGauss(ncomp,DatTr,DatTest,lenTrain,lenTest,covar=covar)
        ScoresLOO[cs]=scte
    return Dat,lengths,ScoresLOO
def FitVisHMM(self,stim,ncomp=3,covar='full',ax=0,ax2=0,NTest=5,showim=True,verb=False,incol=False,vis=True):
    ''' fit and visualize HMM -- beta version
    different random train - test split for each iteration-- noisy results
    stim: stimulus name
    ncomp: number of HMM components
    covar: covariance structure full','tied','spherical' ,'diag'
    Ntest: number of participants to test'''
    xx,yy,lengths=self.DataArrayHmm(stim,tolerance=80,verb=verb)
    Dat=np.column_stack((xx,yy))
    
    DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,NTest,vis=0,rand=1)


    HMMfitted,meanscore,meanscoreTe=FitScoreHMMGauss(ncomp,DatTr,DatTest,lenTrain,lenTest,covar=covar)


    if vis:
        if type(ax)==int:
            fig,ax=plt.subplots()
        if type(ax2)==int:
            fig,ax2=plt.subplots()
        self.VisHMM(DatTr,HMMfitted,ax=ax,showim=showim,stim=stim,lengths=lenTrain,incol=incol)
        ax.set_title('n: '+str(ncomp)+' train ll: '+str(np.round(meanscore,2))+' test ll: '+str(np.round(meanscoreTe,2)),fontsize=9)
        ax2.scatter(ncomp,meanscore,color='g',label='training')
        ax2.scatter(ncomp,meanscoreTe,color='r',label='test')
        handles, labels = ax2.get_legend_handles_labels()

        ax2.set_xlabel('num components')
        ax2.set_ylabel('log likelihood')
        ax2.legend(handles[:2], labels[:2])

  
    return HMMfitted,meanscore,meanscoreTe
    
def FitVisHMMGroups(self,stim,betwcond,ncomp=3,covar='full',ax=0,ax2=0,NTest=3,showim=False,Rep=1):
    ''' fit and visualize HMM
    stim: stimulus name
    betwcond: between group condition
    ncomp: number of HMM components
    covar: HMM gaussian covariance type , must be one of 'full','tied','spherical' ,'diag'
    ax: figure to show fitted hmms and fixations
    ax2: confusion matrix
    NTest: number of test participants (randomly selected) 
    showim: =True show image-- throws error if image has not been loaded previously
    Rep=nNum times to repeat the whole process
    
    note that due to the inherent randomness of hmm-s,and the different random train - test split for each iteration, the resutls are quite noisy for a single iteration.'''
    
    self.GetGroups(betwcond)
    Grs=np.unique(self.data[betwcond])
    
    fig,ax=plt.subplots(ncols=len(Grs),figsize=(12,5))
    fig2,ax2=plt.subplots(ncols=2) 

    # data arrangement for groups
    ScoresTrain=np.zeros((Rep,len(Grs),len(Grs)))
    ScoresTest=np.zeros((Rep,len(Grs),len(Grs)))
   
   
    for rep in range(Rep):  
        XXTrain=[]
        LengthsTrain=[]
        XXTest=[]
        LengthsTest=[]
        for cgr,gr in enumerate(Grs):
            xx,yy,Lengths=self.DataArrayHmm(stim,group=cgr,tolerance=50,verb=False)
            if np.sum(np.shape(xx))==0:
                print('data not found')
            Dat=np.column_stack((xx,yy))
            
            DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,Lengths,ntest=NTest,vis=0,rand=1)
            XXTrain.append(DatTr)
            XXTest.append(DatTest)
            LengthsTrain.append(lenTrain)
            LengthsTest.append(lenTest)
        for cgr,gr in enumerate(Grs):
            HMMfitted,meanscore,meanscoreTe=FitScoreHMMGauss(ncomp,XXTrain[cgr],XXTest[cgr],LengthsTrain[cgr],LengthsTest[cgr],covar=covar)
            if rep==0:
                self.VisHMM(XXTrain[cgr],HMMfitted,ax=ax[cgr],showim=showim,stim=stim,lengths=LengthsTrain[cgr])
                
                ax[cgr].set_title(str(gr))
            for cgr2,gr2 in enumerate(Grs):
                ScoresTrain[rep,cgr2,cgr]=HMMfitted.score(XXTrain[cgr2],LengthsTrain[cgr2])/np.sum(LengthsTrain[cgr2])
                ScoresTest[rep,cgr2,cgr]=HMMfitted.score(XXTest[cgr2],LengthsTest[cgr2])/np.sum(LengthsTest[cgr2])

    im=ax2[0].pcolor(np.mean(ScoresTrain,0))
    ax2[0].scatter(np.arange(len(Grs))+.5,np.argmax(np.mean(ScoresTrain,0),0)+.5,color='k')  # mark most likely for each group
    ax2[0].set_title('training')
#       plt.colorbar(im1)
    im=ax2[1].pcolor(np.mean(ScoresTest,0))
    ax2[1].scatter(np.arange(len(Grs))+.5,np.argmax(np.mean(ScoresTest,0),0)+.5,color='k')  # mark most likely for each group

#        plt.colorbar(im2)
    ax2[1].set_title('test')
    ax2[0].set_ylabel('tested')

    for pl in range(2):
        ax2[pl].set_xlabel('fitted')
        ax2[pl].set_xticks(np.arange(len(Grs))+.5)
        ax2[pl].set_xticklabels(Grs)
        ax2[pl].set_yticklabels(Grs,rotation=90)
        
        ax2[pl].set_yticks(np.arange(len(Grs))+.5)
    fig2.subplots_adjust(right=0.8)
    cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
    fig2.colorbar(im, cax=cbar_ax)
#    plt.tight_layout()
    plt.show()


    return ScoresTrain, ScoresTest


def HMMSimPipeline(self,ncomps=[4,6],verb=False,covar='full'):
    ''' fit l hidden markov model to data, with different number of components, each participants likelihood with leave-one-out cross validation
    can have a long run time with longer viewing time/lot of data 
    return the individual loo log likelihoods from the best model (highest log likelihood) for each stimulus 
    verb=True: print line for subjects with not enough fixations. - too much printing for many subjects wiht low number of fixations 
    ncomp: list of integers with the number of components to fit 
    covar: HMM gaussian covariance type , must be one of 'full','tied','spherical' ,'diag'
    '''
    StimSimsHMM=np.zeros((len(ncomps),self.np))
    
    print(np.shape(StimSimsHMM))
    StimSimsHMMall=np.zeros((len(ncomps),self.ns,self.np))
    StimSimsHMMall[:]=np.nan
    for cncomp, ncomp in enumerate(ncomps):
        print(f'fitting HMM with {ncomp} components')
        for cp in range(self.np):
            print(f'for stimulus {self.stimuli[cp]}')
            Dat,lengths,ScoresLOO=self.FitLOOHMM(ncomp,self.stimuli[cp],covar=covar,verb=verb)
            missS=np.setdiff1d(self.subjects,self.suseHMM)
            if len(missS)>0:
                idxs=np.array([],dtype=int)
                for cs,s in enumerate(self.subjects):
                    if s not in missS:
                        idxs=np.append(idxs,cs)            
                StimSimsHMMall[cncomp,idxs,cp]=ScoresLOO
            else:
                StimSimsHMMall[cncomp,:,cp]=ScoresLOO
            StimSimsHMM[cncomp,cp]=np.mean(ScoresLOO)
    return StimSimsHMM,np.nanmean(StimSimsHMMall,0), StimSimsHMMall


def HMMSimPipelineAll2All(self,ncomp=4,verb=False,covar='full',ntest=3, n_iter=100, iter=1, stimuli=None):
    ''' all2all across compariosn evaluation of hidden markov model to data,
    with different number of components, each participants likelihood with leave-one-out cross validation
    can have a long run time with longer viewing time/lot of data 
    
    return the individual loo log likelihoods from the best model (highest log likelihood) for each stimulus 
    verb=True: print line for subjects with not enough fixations. - too much printing for many subjects wiht low number of fixations 
    ncomp: list of integers with the number of components to fit 
    covar: HMM gaussian covariance type , must be one of 'full','tied','spherical' ,'diag'
    '''
    if stimuli is None:
        StimSimsHMMTrain=np.zeros((self.np,self.np))
        StimSimsHMMTest=np.zeros((self.np,self.np))
        stimuli=self.stimuli       
    else:
        StimSimsHMMTrain=np.zeros((len(stimuli),len(stimuli)))
        StimSimsHMMTest=np.zeros((len(stimuli),len(stimuli)))

    DatsTrain={}
    DatsTest={}
    DatsTrainL={}
    DatsTestL={}
    
    for cp,stim in enumerate(stimuli):
        xx,yy,lengths=self.DataArrayHmm(stim,tolerance=80,verb=verb)
        Dat=np.column_stack((xx,yy))
        DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,ntest=ntest,vis=0,rand=0)
        DatsTrain[stim]=DatTr
        DatsTrainL[stim]=lenTrain
        DatsTest[stim]=DatTest
        DatsTestL[stim]=lenTest
        
    for cp1,stim1 in enumerate(stimuli):
        HMMfitted,sctr,scte=FitScoreHMMGauss(ncomp,DatsTrain[stim1],DatsTest[stim1], DatsTrainL[stim1],DatsTestL[stim1],covar=covar, n_iter=n_iter, iter=iter)
        for cp2,stim2 in enumerate(stimuli):
            StimSimsHMMTrain[cp2,cp1]=HMMfitted.score(DatsTrain[stim2],DatsTrainL[stim2])/np.sum(DatsTrainL[stim2])
            StimSimsHMMTest[cp2,cp1]=HMMfitted.score(DatsTest[stim2],DatsTestL[stim2])/np.sum(DatsTestL[stim2])
    
    self.VisSimmat(StimSimsHMMTrain,'Train', stimuli)
    self.VisSimmat(StimSimsHMMTest,'Test', stimuli)
    
    return StimSimsHMMTrain,StimSimsHMMTest


def norm_diff(matrix1, matrix2):
    """
    Computes the normalized norm of the difference between two matrices, ensuring the score is between 0 and 1.
    
    Parameters:
    matrix1, matrix2: The matrices to compare.
    
    Returns:
    float: The normalized norm of the difference.
    """
    diff_norm = np.linalg.norm(matrix1 - matrix2)
    normalization_factor = np.linalg.norm(matrix1) + np.linalg.norm(matrix2)
    
    if normalization_factor == 0:
        return 0
    
    return diff_norm / normalization_factor

def euclidean_distance(v1, v2):
    """
    Computes the Euclidean distance between two 2D vectors.
    
    Parameters:
    v1, v2: The vectors to compare.
    
    Returns:
    float: The Euclidean distance.
    """
    return np.linalg.norm(v1 - v2)

def covariance_shape_and_orientation_diff(cov1, cov2):
    """
    Computes a normalized score that represents the difference in shape and orientation
    between two covariance matrices (2x2), ensuring the score is between 0 and 1.
    
    Parameters:
    cov1, cov2: The 2x2 covariance matrices to compare.
    
    Returns:
    float: A normalized score representing the difference between the shapes and orientations of the ellipses.
    """
    # If covariances are identical, return 0
    if np.allclose(cov1, cov2):
        return 0 

    # Get the eigenvalues and eigenvectors (shape and orientation) for each covariance matrix
    eigvals1, eigvecs1 = np.linalg.eigh(cov1)
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    
    # 1. Shape difference: Compare the eigenvalues (semi-axes lengths of the ellipses)
    # Normalize by the sum of the eigenvalues
    shape_diff = np.abs(eigvals1 - eigvals2) / (np.abs(eigvals1) + np.abs(eigvals2))
    shape_score = np.sum(shape_diff)  # Aggregate the normalized differences
    
    # 2. Orientation difference: Compare the eigenvectors (directions of the semi-axes)
    # Compute the cosine of the angle between the two corresponding eigenvectors
    orientation_diff = np.abs(np.dot(eigvecs1[:, 0], eigvecs2[:, 0]))  # Cosine of the angle between principal axes
    orientation_score = 1 - orientation_diff  # Normalize to be in range [0, 1]
    
    # Combine shape and orientation scores
    total_score = (shape_score + orientation_score) / (shape_score + orientation_score + 1)
    
    return total_score

def reorder_model_states(model1, model2):
    """
    Reorders the states of model2 to best match the states of model1 based on the means.
    
    Parameters:
    model1, model2: The HMM models to reorder and compare.
    
    Returns:
    reordered_model2: model2 with reordered states to match model1.
    """

    # Step 1: Find the best correspondence between states by comparing the means
    mean_distances = cdist(model1.means_, model2.means_, metric='euclidean')
    best_match = np.argmin(mean_distances, axis=1)
    
    # Reorder means
    model2.means_ = model2.means_[best_match]
    
    # Reorder covariances
    model2.covars_ = model2.covars_[best_match]


def compare_hmm_models_with_scores(hmm_models):
    """
    Compares the key matrices (transition matrix, means, covariances) of a list of GaussianHMM models,
    and adds a score indicating the similarity of the matrices.
    
    Parameters:
    hmm_models (list): A list of GaussianHMM models to compare.
    
    Returns:
    dict: A dictionary containing the pairwise differences and similarity scores for each matrix type.
    """
    n_models = len(hmm_models)
    results = {
        'transition_diff': [],
        'means_diff': [],
        'covariances_diff': [],
        'transition_scores': [],
        'means_scores': [],
        'covariances_scores': []
    }
    
    # Compare each pair of models
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1 = hmm_models[i]
            model2 = hmm_models[j]

            # Reorder model2 to match the states of model1
            reorder_model_states(model1, model2)
            
            # Compare transition matrices
            transition_score = norm_diff(model1.transmat_, model2.transmat_)
            results['transition_scores'].append((i, j, transition_score))
            
            # Compare means (Euclidean distance for 2D means)   
            for i in range(len(hmm_models)):
                for j in range(i + 1, len(hmm_models)):
                    means_score = 0
                    for state in range(len(hmm_models[i].means_)):
                        # Compute the Euclidean distance between the means of the two models
                        mean_diff = euclidean_distance(hmm_models[i].means_[state], hmm_models[j].means_[state])
                        # Normalize by the magnitude of the means
                        means_score += mean_diff / max(np.linalg.norm(hmm_models[i].means_[state]), np.linalg.norm(hmm_models[j].means_[state]))
                    
                    # Average the normalized differences across all states
                    means_score /= len(hmm_models[i].means_)

                    results['means_scores'].append((i, j, means_score))
            
                    # Compare covariances (shape and orientation of the ellipses)
                    covariances_score = 0
                    for state in range(len(model1.covars_)):
                        covariances_score += covariance_shape_and_orientation_diff(model1.covars_[state], model2.covars_[state])
                    
                    covariances_score /= len(model1.covars_)
                    results['covariances_scores'].append((i, j, covariances_score))

            results['final_scores'] = (transition_score + means_score + covariances_score) / 3
    
    return results


def HMMSimPiepelineModel2Model(self,ncomp=4,verb=False,covar='full', n_iter=100, iter=1, stimuli=None):
    if stimuli is None:
        StimSimsHMM=np.zeros((self.np,self.np))
        stimuli=self.stimuli       
    else:
        StimSimsHMM=np.zeros((len(stimuli),len(stimuli)))


    DatsTrain={}
    DatsTest={}
    DatsTrainL={}
    DatsTestL={}

    for cp,stim in enumerate(stimuli):
        xx,yy,lengths=self.DataArrayHmm(stim,tolerance=80,verb=verb)
        Dat=np.column_stack((xx,yy))
        DatTr,DatTest,lenTrain,lenTest=self.MyTrainTest(Dat,lengths,ntest=3,vis=0,rand=0)
        DatsTrain[stim]=DatTr
        DatsTrainL[stim]=lenTrain
        DatsTest[stim]=DatTest
        DatsTestL[stim]=lenTest
    

    for cp1,stim1 in enumerate(stimuli):
        with io.capture_output() as captured:
            HMMfittedM1,sctr,scte=FitScoreHMMGauss(ncomp,DatsTrain[stim1],DatsTest[stim1], DatsTrainL[stim1],DatsTestL[stim1],covar=covar, n_iter=n_iter, iter=iter)
        for cp2,stim2 in enumerate(stimuli):
            with io.capture_output() as captured:
                HMMfittedM2,sctr,scte=FitScoreHMMGauss(HMMfittedM1.n_components,DatsTrain[stim2],DatsTest[stim2], DatsTrainL[stim2],DatsTestL[stim2],covar=covar, n_iter=n_iter, iter=iter)
            StimSimsHMM[cp2,cp1] = compare_hmm_models_with_scores([HMMfittedM1, HMMfittedM2])['final_scores']

    self.VisHMMSimmat(StimSimsHMM,'Model Comp', stimuli)

    return StimSimsHMM
