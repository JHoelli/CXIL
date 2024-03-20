import torch
import numpy as np 
from CXIL.Benchmarking.metrics.classification_metrics import  Classification_metrics
from CXIL.Benchmarking.metrics.interpretation import  Interpretation_metrics
from CXIL.caipi import utils, Transformer
from CXIL.Models.SimpleModel import SimpleMLP
import torch.nn as nn
from captum.attr import InputXGradient
#from avalanche.models import SimpleMLP 
from CXIL.Learning import LearnerStep, Replay
import warnings
import matplotlib.pyplot as plt
import pandas as pd 
warnings.filterwarnings('ignore')
from CXIL.Data.DataLoader import load_data_and_sim
from CXIL.caipi import CAipi_without_TKadd as caipi

dataset='toy_classification'
'''Load Data '''
X,y,X_train,y_train,X_test,y_test, simulation= load_data_and_sim(dataset)
X_with_finetuning=np.concatenate((X,X_train))[0:10]
y_with_finetuning=np.concatenate((y,y_train))[0:10]

model_empty_train = SimpleMLP(num_classes=2,input_size=5*5*3)

trans= Transformer.MeanAndStdTransformer(model_empty_train,num_samples=1)
#learn=LearnerStep.BasicRRRLearner(model)
learn = LearnerStep.Basic(model_empty_train)
learn_replay=Replay.ReplayStrategyLearner(learn, num_classes=2,mem_size = 1000,batch_size= 64)
int_train= caipi.caipi(model_empty_train,learn_replay,transformer=trans, evaluate_data= (X_test,y_test),silent=0)


int_train.iterate(X_with_finetuning)
