from CXIL.Benchmarking.Evaluation import Evaluation
from quantus import RelevanceRankAccuracy, RelevanceMassAccuracy,AUC
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc

def auc_self(a,s):
     # Prepare shapes.
    score=[]
    for a_i, s_i in zip(a,s):
        a_i = a_i.flatten()
        s_i = s_i.flatten().astype(bool)

        fpr, tpr, _ = roc_curve(y_true=s_i, y_score=a_i)
        score_o = auc(x=fpr, y=tpr)
        score.append(score_o)
    return score

def quantus_metrics(mlmodel, data, labels, exp, masks):

    #exp_list=np.array(exp).reshape(data.shape[0],1,1,data.shape[2]*data.shape[1])
    #masks_list=np.array(masks).reshape(data.shape[0],1,1,data.shape[2]*data.shape[1])
    #data_list=np.array(data).reshape(data.shape[0],1,1,data.shape[2]*data.shape[1])
    #print('exp ',exp.shape)
    #print('data ',data.shape)
    #print('masks ',masks.shape)
    exp=np.absolute(np.array(exp)).reshape(data.shape[0],1,np.prod(data.shape[1:]))
    masks=np.array(masks).reshape(data.shape[0],1,np.prod(data.shape[1:]))

    data=np.array(data).reshape(data.shape[0],1,np.prod(data.shape[1:]))
    rank=RelevanceRankAccuracy(disable_warnings=True)(model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU',channel_first=True)
    rank_mass=RelevanceMassAccuracy(disable_warnings=True)(model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU',channel_first=True)
    exp= np.nan_to_num(exp,posinf=0.0,neginf=0.0, nan=0.0)
    #exp= np.nan_to_num(exp, neginf=0) 
    exp=exp.astype(np.float32)

    masks= np.nan_to_num(masks,posinf=0.0,neginf=0.0, nan=0.0)
    #exp= np.nan_to_num(exp, neginf=0) 
    masks=masks.astype(np.float32)
    #print('exp ',np.any(np.isnan(exp)))
    #print('masks ',np.any(np.isnan(masks)))
    #print(masks)
    #print('NAN','NaN' in exp)
    #print('NAN1',np.nan in exp)
    #print('NAN2',np.inf in exp)
    #print('NAN3',xp.any(xp.isnan(X)))
    #print(exp)
    #Auc=auc_self(exp,masks)
    #import sys 
    #sys.exit(1)
    try:
        Auc=AUC(disable_warnings=True)(model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU',channel_first=True)
    except: 
        Auc=auc_self(exp,masks)
    rank=RelevanceRankAccuracy(disable_warnings=True)(model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU')
    rank_mass=RelevanceMassAccuracy(disable_warnings=True)(model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU')

    #test = True
    #i=0
    #for  a in auc_:
    #    if a != Auc[i]:
    #        test = False
    #    i+=1
    
    #print('AUC IDENTICAL',test )
    #import sys 
    #sys.exit(1)
    return rank, rank_mass, Auc#,auc



def _get_interpretation_metrics(mlmodel, data, labels, exp,masks):
    return quantus_metrics(mlmodel, data, labels, exp, masks)

class Interpretation_metrics(Evaluation):
    """
    Calculates Metrics regarding the Interpretability
    Attributes: 
        mlmodel: Classification / Regression Model 
        explainer: explanation function / or importance grid
        ground_truth: Ground Truth (if it exists)
    """

    def __init__(self, mlmodel,explainer,ground_truth=None):
        super().__init__(mlmodel)
        self.explainer=explainer(mlmodel)
        self.ground_truth=ground_truth
        self.columns = ["rank", "rank_mass", "Auc"]#, "auc"]



    def get_evaluation(self, x,y):
       
        y=y.astype(int)
        if  isinstance(y, np.ndarray):
            y=y.tolist()
        if not isinstance(x, torch.Tensor):
            x_new=torch.from_numpy(x).float()
        exp= self.explainer.attribute(inputs=x_new,target=y).detach().numpy()
        interpretation= _get_interpretation_metrics(self.mlmodel,x,y,exp,self.ground_truth)
        #print(len(interpretation))
        #print(len(interpretation[0]))
        df =pd.DataFrame([])
        df['rank']= interpretation[0]
        df['rank_mass']= interpretation[1]
        df['auc']= interpretation[2]
        #import sys 
        #sys.exit(1)
        #df= pd.DataFrame([])
        #i= 0
        #for c in interpretation:
        #    df[f'{self.columns[i]}']=c
        #    i+=1
        return df