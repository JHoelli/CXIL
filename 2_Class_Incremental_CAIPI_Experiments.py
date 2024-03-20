import numpy as np 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Model import SmallModel, train_small
from captum.attr import InputXGradient
import warnings
from CXIL.caipi.Transformer import MeanAndStdTransformer, randomTransformer
from CXIL.Benchmarking.metrics.Helper import get_mask
from CXIL.Data.DataLoader import load_data_and_sim,timeseries
from Evaluation_Script import evaluation
from CXIL.Learning import LearnerStep, Replay
#from avalanche.models import SimpleMLP, SimpleCNN, MobilenetV1
from CXIL.Models.SimpleCNN import SimpleCNN,CNN
from CXIL.Models.SimpleModel import SimpleMLP
from CXIL.Models.Simple1DResNet import ResNetBaseline
from CXIL.Learning import LearnerStep, Replay, EWC, Synaptic_Intelligence, MAS
#THIS IS Server Settigng ! 
from CXIL.caipi.CAipi_without_TKadd import caipi
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from CXIL.Benchmarking.metrics.continouus_metrics import Continous_metrics
from CXIL.caipi.Transformer import MeanAndStdTransformer, randomTransformer
import argparse
import random

'''
CAIPI

'''
import torch.nn.functional as F


warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
''' FLAG '''
rerun_all = False
''' END FLAG '''

#dataset='./CXIL/Data/data/TimeSeries/Testing/data/SimulatedTestingData_Middle_Harmonic_F_1_TS_50.npy' #'toy_classification'
#data_name= 'SimulatedTestingData_Middle_Harmonic_F_1_TS_50'
'''START PARAMETER SECTION'''

parser = argparse.ArgumentParser()
parser.add_argument('dataset')           # positional argument
parser.add_argument('data_name')      # option that takes a value
parser.add_argument('model')
parser.add_argument('transformer')
parser.add_argument('seed')
args = parser.parse_args()
print(f'PARAMETES {args.dataset} {args.data_name} {args.model} {args.transformer}')


'''SET SEEED'''
seed=int(args.seed)
print(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
'''Finsihed Seed'''

dataset= args.dataset #'./CXIL/Data/data/TimeSeries/Testing/data/SimulatedTestingData_Middle_Harmonic_F_50_TS_50.npy' #'toy_classification'
data_name= args.data_name #'SimulatedTestingData_Middle_Harmonic_F_50_TS_50'
model = locals()[f"{args.model}"]#args.model  #SimpleMLP
#dataset= 'tabular'#'iris_cancer' #'toy_classification'
#data_name=dataset
str_model=args.model
'''END PARAMETER SECTION'''



parameter = args.transformer # 'MeanandstdTrans'
trans_function=locals()[f"{ args.transformer}"]#

#if 'mnist' in dataset: 
#    parameter = 'caipi_correction'
if not os.path.isdir(f'./Results/Continous'):
    os.mkdir(f'./Results/Continous')
if not os.path.isdir(f'./Results/Continous/{data_name}'):
    os.mkdir(f'./Results/Continous/{data_name}')
#if not os.path.isdir(f'./Results/Continous/{data_name}/caipi'):
#    os.mkdir(f'./Results/Continous/{data_name}/caipi')
if not os.path.isdir(f'./Results/Continous/{data_name}/CAIPI'):
    os.mkdir(f'./Results/Continous/{data_name}/CAIPI')
if not os.path.isdir(f'./Results/Continous/{data_name}/CAIPI/{parameter}'):
    os.mkdir(f'./Results/Continous/{data_name}/CAIPI/{parameter}')
if not os.path.isdir(f'./Results/Continous/{data_name}/CAIPI/{parameter}/{str_model}'):
    os.mkdir(f'./Results/Continous/{data_name}/CAIPI/{parameter}/{str_model}')

if not rerun_all: 
    # Try to load data 
    if os.path.exists(f'./Results/Continous/{data_name}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv'):
        full_classification_df=pd.read_csv(f'./Results/Continous/{data_name}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')  
        full_classification_df.drop(labels=['Unnamed: 0'],axis=1,inplace=True)
    if os.path.exists(f'./Results/Continous/{data_name}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv'):
        full_interpretability_df= pd.read_csv(f'./Results/Continous/{data_name}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')

'''Load Data '''
if 'Sim' in dataset:#os.path.isfile(dataset):
    print('----TimeSeries----')
    X,y,X_train,y_train,X_test,y_test,  meta_train, meta_test, test_meta,simulation= load_data_and_sim(dataset)
    #simulation= timeseries(meta_train)
    print('test_meta', test_meta.shape)
    ground_truth= simulation.get_ground_truth(X_test) #get_mask(X_test,start=test_meta[:,1:3],end=test_meta[:,3:])
    #print(np.nonzero(ground_truth))
    print('X ', X.shape)
    print('y ', y.shape)
    print('X_train ', X_train.shape)
    print('Y_train ', y_train.shape)
    print('X_test ', X_test.shape)
    print('y_test ', y_test.shape)

else:
    print(f'----{dataset}----')
    X_tasks,y_tasks,X_test_task,y_test_task,X_test,y_test, simulation= load_data_and_sim(dataset)
    meta_tasks=[0,1,2,3,4]
    #ground_truth = simulation.get_ground_truth(X_test)

n_tasks= len(X_tasks)
print('Number Tasks',n_tasks)
n_classes= len(np.unique(y_test))
print('Number Classes',n_classes)

input_size=1
for i in X_tasks[0][0].shape:
    input_size = input_size*i

print('Input Size',input_size)


if 'Simulated' in dataset:#os.path.isfile(dataset):
    print('THIS IS A TODO')
    pass

print('INPUT', input_size)

'''Basic Iterate Baseline'''
if 'CNN' in str(model):
    model_our = model(num_classes=n_classes) 

else:
    if 'ResNet' in str(model):
        input_size=X_train.shape[1]
    model_our = model(num_classes=n_classes,input_size=input_size) 



'''
Upper Bound Full Training --> Sanity
'''


'''
Continous Case without anything(CAIPI Check) - ADAM
'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_ADAM_{seed}.csv'):
    print('Model Continous')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 
    #m_wrap_empty=mod_wrapper(model_empty)

    learn = LearnerStep.Basic(model_empty)

    trans= trans_function(model_empty,num_samples=1)

    cai=caipi(model_empty,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)

    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        #update
    
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,_,_=cai.iterate(xi, taskid= 0)
        #print(len(y_test_task))
        #print(len(y_test))
        if i == 0:
            print(len(y_test_task))
            print(y_test_task)
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_ADAM_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_ADAM_{seed}',save=True,interpretability_measure=False)

    full_classification_df= clas    
    full_interpretability_df= interpret

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')

'''ADAM & Label Trick'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_ADAM_LT_{seed}.csv'):
    print('ADAM Label Trick')
    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 
   
 
    learn = LearnerStep.Basic(model_empty)

    trans= trans_function(model_empty,num_samples=1)

    cai=caipi(model_empty,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)
    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,_,_=cai.iterate(xi)
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_ADAM_LT_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_ADAM_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')




'''
Continous Case without anything(CAIPI Check) - SGD
'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_SGD_{seed}.csv'):
    print('Model Continous')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 
   
    #model_basic_continous = model_empty
    trans= trans_function(model_empty,num_samples=1)

    learn = LearnerStep.Basic(model_empty,optimizer=torch.optim.SGD)
    cai=caipi(model_empty,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)

    for i in range(0, n_tasks):
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
            #training_data=TensorDataset(torch.tensor(xi.reshape(1,-1)).float(),torch.tensor([yi]))
            #print('End Tensor')
            #train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            #model_basic_continous=learn.fit(train_dataloader)
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,_,_=cai.iterate(xi,taskid=i)
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_SGD_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_SGD_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')


#TODO LESEZEICHEN

'''
Continous Case without anything(CAIPI Check) -SGD - LT
'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_SGD_LT_{seed}.csv'):
    print('Model Continous')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 
    
    #model_basic_continous = model_empty
    trans= trans_function(model_empty,num_samples=1)
    learn = LearnerStep.Basic(model_empty,optimizer=torch.optim.SGD)

    cai=caipi(model_empty,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)
    #
    for i in range(0, n_tasks):
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        #    training_data=TensorDataset(torch.tensor(xi.reshape(1,-1)).float(),torch.tensor([yi]))
            #print('End Tensor')
        #    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
        #    model_basic_continous=learn.fit(train_dataloader,i)
        xi=X_tasks[i]
        yi=y_tasks[i]
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,_,_=cai.iterate(xi)
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_SGD_LT_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_SGD_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')



'''EWC '''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_ewc_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_ewc  = model(num_classes=n_classes) 
    else:
        model_basic_ewc  = model(num_classes=n_classes,input_size=input_size) 
    
    trans= trans_function(model_basic_ewc,num_samples=1)
    #learn=Synaptic_Intelligence.Synaptic_Intelligence(model_basic_lwf)
    #learn=BGD.BGD_Learner(model_basic_lwf)
    learn =EWC.Elastic_Weight_Regularizer(model_basic_ewc)
    #LearnerStep.Basic(model_basic_synaptic)
    cai=caipi(model_basic_ewc,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)
    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)
        #TODO does input need to befalttend        s
        print(f'LABEL OF TASKS {np.unique(yi)} {yi.shape}')
        print(f'Correct Label',yi[0])
        acc,f1, precision, recall,_,_=cai.iterate(xi,taskid=i)

        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_ewc_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_ewc_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')


'''EWC & LT '''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_ewc_LT_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_ewc  = model(num_classes=n_classes) 
    else:
        model_basic_ewc  = model(num_classes=n_classes,input_size=input_size) 

 
    trans= trans_function(model_basic_ewc,num_samples=1)
    learn =EWC.Elastic_Weight_Regularizer(model_basic_ewc)
    cai=caipi(model_basic_ewc,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)
    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,_,_=cai.iterate(xi)
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_ewc_LT_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_ewc_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')



''' Synaptic '''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_SI_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 
    
    trans= trans_function(model_basic_synaptic,num_samples=1)
    learn=Synaptic_Intelligence.Synaptic_Intelligence(model_basic_synaptic)
    cai=caipi(model_basic_synaptic,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,_,_=cai.iterate(xi, taskid=i)
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_SI_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_SI_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')



''' Synaptic & LT'''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_SI_LT_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 

    
    trans= trans_function(model_basic_synaptic,num_samples=1)
    learn=Synaptic_Intelligence.Synaptic_Intelligence(model_basic_synaptic)
    cai=caipi(model_basic_synaptic,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,_,_=cai.iterate(xi)
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_SI_LT_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_SI_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')

'''MAS'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_MAS_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 

    learn=MAS.MAS(model_basic_synaptic)

    trans= trans_function(model_basic_synaptic,num_samples=1)

    cai=caipi(model_basic_synaptic,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)
    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,_,_=cai.iterate(xi,taskid=i)
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_MAS_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_MAS_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')

'''MAS LT '''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_MAS_LT_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 
    learn=MAS.MAS(model_basic_synaptic)
   
    trans= trans_function(model_basic_synaptic,num_samples=1)
    cai=caipi(model_basic_synaptic,  learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)
    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,_,_=cai.iterate(xi)
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_MAS_LT_{seed}.csv')
    clas,interpret= evaluation(cai.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_MAS_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')


'''REPLAY'''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_replay_{seed}.csv'):
    print('REPLAY')

    if 'CNN' in str(model):
        model_basic_replay  = model(num_classes=n_classes) 
    else:
        model_basic_replay  = model(num_classes=n_classes,input_size=input_size) 

    learn = LearnerStep.Basic(model_basic_replay)
   
    trans= trans_function(model_basic_replay,num_samples=1)
    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 100,
            batch_size= 32)
    cai=caipi(model_basic_replay, learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)
    #torch.save(cai.model.state_dict(), f'./Results/{data_name}/{parameter}/model_full_trained')
    
    #TODO is here a Issue
    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        acc,f1, precision, recall,_,_=cai.iterate(X_tasks[i],taskid=i)
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_replay_{seed}.csv')
    clas,interpret= evaluation(cai.model , InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_replay_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')

'''Replay & LT '''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_replay_LT_{seed}.csv'):
    print('REPLAY')

    if 'CNN' in str(model):
        model_basic_replay  = model(num_classes=n_classes) 
    else:
        model_basic_replay  = model(num_classes=n_classes,input_size=input_size) 
    
    learn = LearnerStep.Basic(model_basic_replay)
    trans= trans_function(model_basic_replay,num_samples=1)
    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 100,
            batch_size= 32)
    cai=caipi(model_basic_replay, learn,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)
    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        acc,f1, precision, recall,_,_=cai.iterate(X_tasks[i])
        if i == 0:
            val = Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(cai.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/model_replay_LT_{seed}.csv')
    clas,interpret= evaluation(cai.model , InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}',f'model_replay_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/CAIPI/{parameter}/{str_model}/interpretability_metrics_{seed}.csv')

