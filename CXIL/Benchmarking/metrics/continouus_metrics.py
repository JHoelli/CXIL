#TODO ThIS IS OVERALL TODO 
from XIL.Benchmarking.Evaluation import Evaluation
from sklearn.metrics import  accuracy_score
import numpy as np 
import torch
import pandas as pd 
from torch.utils.data import Dataset, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def accuracy(original,pred):
    '''Calculates Task Wise Accuracy.'''
    if len(original.shape)>1:
        original=np.argmax(original, axis = 1)
    if len(pred.shape)>1:
        pred = np.argmax(pred, axis=1) 
    return accuracy_score(original , pred)

def catastophic_forgetting(): 
    '''https://github.com/ContinualAI/avalanche/blob/master/avalanche/evaluation/metrics/forgetting_bwt.py'''

    
    pass


class Continous_metrics(Evaluation):
    """
    Calculates Metrics to evaluate Catastophic Forgetting / Inference  
    """

    def __init__(self, mlmodel):
        super().__init__(mlmodel)
        #self.columns = ["accuracy"]


    def get_evaluation(self, x,y, taskwise_x, taskwise_y):
        x =torch.from_numpy(x).float()
        pred= self.mlmodel(x).detach().numpy()
        # TASK 1, 2, 3, 4, 5
        # Overall
        df = pd.DataFrame([])
        acc_overall=accuracy(y,pred)
        df['overall']=[acc_overall]
        for i in range(0, len(taskwise_x)):
            #TODO This is currently incorrect ! 
            #x=TensorDataset((torch.tensor(taskwise_x[i][i].reshape(-1,1,28,28)).float()),transforms.Compose([ transforms.Resize(64)]))
            x =torch.from_numpy(taskwise_x[i]).float()
            pred= self.mlmodel(x).detach().numpy()
            acc_t=accuracy(taskwise_y[i],pred)
            df[f'task_{i}']=[acc_t]
        print('From Continous Metrics', df)
        return df