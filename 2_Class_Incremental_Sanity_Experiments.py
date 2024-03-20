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
from CXIL.Learning import LearnerStep, Replay, EWC, Synaptic_Intelligence, MAS, GEM, LwF, EEIL
#THIS IS Server Settigng ! 
from CXIL.caipi.CAipi_without_TKadd import caipi
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from CXIL.Benchmarking.metrics.continouus_metrics import Continous_metrics
import argparse
import random

warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
''' FLAG '''
rerun_all = True
''' END FLAG '''

#dataset='./CXIL/Data/data/TimeSeries/Testing/data/SimulatedTestingData_Middle_Harmonic_F_1_TS_50.npy' #'toy_classification'
#data_name= 'SimulatedTestingData_Middle_Harmonic_F_1_TS_50'
'''START PARAMETER SECTION'''

parser = argparse.ArgumentParser()
parser.add_argument('dataset')           # positional argument
parser.add_argument('data_name')      # option that takes a value
parser.add_argument('model')
parser.add_argument('seed')
#parser.add_argument('transformer')
args = parser.parse_args()

print(f'PARAMETES {args.dataset} {args.data_name} {args.model}')

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




#parameter = args.transformer # 'MeanandstdTrans'
#trans_function=locals()[f"{ args.transformer}"]#

#if 'mnist' in dataset: 
#    parameter = 'caipi_correction'
if not os.path.isdir(f'./Results/Continous'):
    os.mkdir(f'./Results/Continous')
if not os.path.isdir(f'./Results/Continous/{data_name}'):
    os.mkdir(f'./Results/Continous/{data_name}')
#if not os.path.isdir(f'./Results/Continous/{data_name}/caipi'):
#    os.mkdir(f'./Results/Continous/{data_name}/caipi')
if not os.path.isdir(f'./Results/Continous/{data_name}/Sanity'):
    os.mkdir(f'./Results/Continous/{data_name}/Sanity')
if not os.path.isdir(f'./Results/Continous/{data_name}/Sanity/{str_model}'):
    os.mkdir(f'./Results/Continous/{data_name}/Sanity/{str_model}')

if not rerun_all: 
    # Try to load data 
    if os.path.exists(f'./Results/Continous/{data_name}/Sanity/{str_model}/classification_metrics_{seed}.csv'):
        full_classification_df=pd.read_csv(f'./Results/Continous/{data_name}/Sanity/{str_model}/classification_metrics_{seed}.csv')  
        full_classification_df.drop(labels=['Unnamed: 0'],axis=1,inplace=True)
    if os.path.exists(f'./Results/Continous/{data_name}/Sanity/{str_model}/interpretability_metrics_{seed}.csv'):
        full_interpretability_df= pd.read_csv(f'./Results/Continous/{data_name}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')

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
Upper Bound Full Training
'''
print(os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/Upper_Bound_{seed}'))

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/Upper_Bound_{seed}.csv'):
    print('Upper Bound')
    data= np.concatenate(X_tasks)
    data_y= np.concatenate(y_tasks)
    model_full = model_our
    optimizer = torch.optim.Adam(model_full.parameters(), lr=0.001)
    loss_fn   = nn.CrossEntropyLoss()
    epochs =100
    acc_full= train_small(model_full,data,data_y,X_test,y_test,optimizer,loss_fn, epochs)
    for i in range(0, n_tasks):
        #acc_full=train_small(model_full,X_tasks[i],y_tasks[i],X_test,y_test,optimizer,loss_fn, epochs)
        
        if i == 0:
            #TODO LOGG ALL Instances
            val = Continous_metrics(model_full).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_full).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
        print(val)
    #FINAL EVALUATION
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/Upper_Bound_{seed}.csv')
    clas,interpret= evaluation(model_full, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'Upper_Bound_{seed}',save=True,interpretability_measure=False)
    full_classification_df= clas    
    full_interpretability_df= interpret


'''
Continous Case without anything(Sanity Check) - ADAM
'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_ADAM_{seed}'):
    print('Model Continous')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 

    model_basic_continous = model_empty

    learn = LearnerStep.Basic(model_basic_continous)


    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(xi.reshape(len(xi),-1)).float(),torch.tensor(yi))
        #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_continous=learn.fit(train_dataloader, taskid=i)
        if i == 0:
            val = Continous_metrics(model_basic_continous).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_continous).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_ADAM_{seed}.csv')
    clas,interpret= evaluation(model_basic_continous, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_ADAM_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')

'''ADAM & Label Trick'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_ADAM_LT_{seed}'):
    print('ADAM Label Trick')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 

    model_basic_continous = model_empty

    learn = LearnerStep.Basic(model_basic_continous)


    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(xi.reshape(len(xi),-1)).float(),torch.tensor(yi))
        #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_continous=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_continous).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_continous).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_ADAM_LT_{seed}.csv')
    clas,interpret= evaluation(model_basic_continous, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_ADAM_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')




'''
Continous Case without anything(Sanity Check) - SGD
'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_SGD_{seed}'):
    print('Model Continous')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 

    model_basic_continous = model_empty

    learn = LearnerStep.Basic(model_basic_continous,optimizer=torch.optim.SGD)


    for i in range(0, n_tasks):
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
            #training_data=TensorDataset(torch.tensor(xi.reshape(1,-1)).float(),torch.tensor([yi]))
            #print('End Tensor')
            #train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            #model_basic_continous=learn.fit(train_dataloader)
        xi=X_tasks[i]
        yi=y_tasks[i]
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(xi.reshape(len(xi),-1)).float(),torch.tensor(yi))
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_continous=learn.fit(train_dataloader, taskid=i)
        if i == 0:
            val = Continous_metrics(model_basic_continous).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_continous).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_SGD_{seed}.csv')
    clas,interpret= evaluation(model_basic_continous, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_SGD_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')


#TODO LESEZEICHEN

'''
Continous Case without anything(Sanity Check) -SGD - LT
'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_SGD_LT_{seed}'):
    print('Model Continous')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 

    model_basic_continous = model_empty

    learn = LearnerStep.Basic(model_basic_continous,optimizer=torch.optim.SGD)


    for i in range(0, n_tasks):
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        #    training_data=TensorDataset(torch.tensor(xi.reshape(1,-1)).float(),torch.tensor([yi]))
            #print('End Tensor')
        #    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
        #    model_basic_continous=learn.fit(train_dataloader,i)
        xi=X_tasks[i]
        yi=y_tasks[i]
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(xi.reshape(len(xi),-1)).float(),torch.tensor(yi))
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_continous=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_continous).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_continous).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_SGD_LT_{seed}.csv')
    clas,interpret= evaluation(model_basic_continous, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_SGD_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')



'''EWC '''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_ewc_{seed}'):

    if 'CNN' in str(model):
        model_basic_ewc  = model(num_classes=n_classes) 
    else:
        model_basic_ewc  = model(num_classes=n_classes,input_size=input_size) 

    #learn=Synaptic_Intelligence.Synaptic_Intelligence(model_basic_lwf)
    #learn=BGD.BGD_Learner(model_basic_lwf)
    learn =EWC.Elastic_Weight_Regularizer(model_basic_ewc)
    #LearnerStep.Basic(model_basic_synaptic)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(X_tasks[i]).float(),torch.tensor(y_tasks[i]))
            #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_ewc=learn.fit(train_dataloader, taskid=i)
        if i == 0:
            val = Continous_metrics(model_basic_ewc).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_ewc).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_ewc_{seed}.csv')
    clas,interpret= evaluation(model_basic_ewc, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_ewc_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')


'''EWC & LT '''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_ewc_LT_{seed}'):

    if 'CNN' in str(model):
        model_basic_ewc  = model(num_classes=n_classes) 
    else:
        model_basic_ewc  = model(num_classes=n_classes,input_size=input_size) 

   
    learn =EWC.Elastic_Weight_Regularizer(model_basic_ewc)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(X_tasks[i]).float(),torch.tensor(y_tasks[i]))
            #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_ewc=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_ewc).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_ewc).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_ewc_LT_{seed}.csv')
    clas,interpret= evaluation(model_basic_ewc, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_ewc_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')



''' Synaptic '''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_SI_{seed}'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 

    learn=Synaptic_Intelligence.Synaptic_Intelligence(model_basic_synaptic)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(X_tasks[i]).float(),torch.tensor(y_tasks[i]))
            #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_synaptic=learn.fit(train_dataloader, taskid=i)
        if i == 0:
            val = Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_SI_{seed}.csv')
    clas,interpret= evaluation(model_basic_synaptic, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_SI_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')



''' Synaptic & LT'''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_SI_LT_{seed}'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 

    learn=Synaptic_Intelligence.Synaptic_Intelligence(model_basic_synaptic)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(X_tasks[i]).float(),torch.tensor(y_tasks[i]))
            #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_synaptic=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_SI_LT_{seed}.csv')
    clas,interpret= evaluation(model_basic_synaptic, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_SI_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')

'''MAS'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_MAS_{seed}'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 

    learn=MAS.MAS(model_basic_synaptic)
    for i in range(0, n_tasks):
        print(f'TASK {i}')
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(X_tasks[i]).float(),torch.tensor(y_tasks[i]))
            #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_synaptic=learn.fit(train_dataloader, taskid=i)
        if i == 0:
            val = Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_MAS_{seed}.csv')
    clas,interpret= evaluation(model_basic_synaptic, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_MAS_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')

'''MAS LT '''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_MAS_LT_{seed}'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 

    learn=MAS.MAS(model_basic_synaptic)
    for i in range(0, n_tasks):
        print(f'TASK {i}')
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(X_tasks[i]).float(),torch.tensor(y_tasks[i]))
            #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_synaptic=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_MAS_LT_{seed}.csv')
    clas,interpret= evaluation(model_basic_synaptic, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_MAS_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')

'''REPLAY'''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_replay_{seed}'):
    print('REPLAY')

    if 'CNN' in str(model):
        model_basic_replay  = model(num_classes=n_classes) 
    else:
        model_basic_replay  = model(num_classes=n_classes,input_size=input_size) 

    learn = LearnerStep.Basic(model_basic_replay)

    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 100,
            batch_size= 32)

    for i in range(0, n_tasks):
        for xi,yi in zip(X_tasks[i],y_tasks[i]):
        
            training_data=TensorDataset(torch.tensor(xi[np.newaxis,]).float(),torch.tensor([yi]))
            #print('End Tensor')
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            model_basic_replay=learn.fit(train_dataloader, taskid=i)
        if i == 0:
            val = Continous_metrics(model_basic_replay).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_replay).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_replay_{seed}.csv')
    clas,interpret= evaluation(model_basic_replay , InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_replay_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')

'''Replay & LT '''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_replay_LT_{seed}'):
    print('REPLAY')

    if 'CNN' in str(model):
        model_basic_replay  = model(num_classes=n_classes) 
    else:
        model_basic_replay  = model(num_classes=n_classes,input_size=input_size) 

    learn = LearnerStep.Basic(model_basic_replay)

    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 100,
            batch_size= 32)

    for i in range(0, n_tasks):
        for xi,yi in zip(X_tasks[i],y_tasks[i]):
            training_data=TensorDataset(torch.tensor(xi[np.newaxis,]).float(),torch.tensor([yi]))
            #print('End Tensor')
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            model_basic_replay=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_replay).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_replay).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_replay_LT_{seed}.csv')
    clas,interpret= evaluation(model_basic_replay , InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_replay_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')


'''REPLAY'''
if  False:# rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_replay.csv'):
    print('GEM')

    if 'CNN' in str(model):
        model_basic_gem  = model(num_classes=n_classes) 
    else:
        model_basic_gem = model(num_classes=n_classes,input_size=input_size) 



    learn= GEM.GEM( model_basic_gem, num_classes=n_classes,mem_size = 500,
            batch_size= 64)

    for i in range(0, n_tasks):
        for xi,yi in zip(X_tasks[i],y_tasks[i]):
        
            training_data=TensorDataset(torch.tensor(xi[np.newaxis,]).float(),torch.tensor([yi]))
            #print('End Tensor')
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            model_basic_replay=learn.fit(train_dataloader, taskid=i)
        if i == 0:
            val = Continous_metrics(model_basic_gem).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_gem).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_gem.csv')
    clas,interpret= evaluation(model_basic_gem, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}','model_gem',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')

'''Replay & LT '''

if False:# rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_GEM_LT.csv'):
    print('GEM')

    if 'CNN' in str(model):   
        model_basic_gem  = model(num_classes=n_classes) 
    else:
        model_basic_gem = model(num_classes=n_classes,input_size=input_size) 



    learn= GEM.GEM( model_basic_gem, num_classes=n_classes,mem_size = 500,
            batch_size= 64)

    for i in range(0, n_tasks):
        for xi,yi in zip(X_tasks[i],y_tasks[i]):
            training_data=TensorDataset(torch.tensor(xi[np.newaxis,]).float(),torch.tensor([yi]))
            #print('End Tensor')
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            model_basic_replay=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_gem).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_gem).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_gem_LT.csv')
    clas,interpret= evaluation(model_basic_replay , InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}','model_gem_LT',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')






'''Replay without CAIPI (Sanity Check) '''

if False:# rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_replay_{seed}'):
    print('REPLAY')

    if 'CNN' in str(model):
        model_basic_replay  = model(num_classes=n_classes) 
    else:
        model_basic_replay  = model(num_classes=n_classes,input_size=input_size) 

    learn = LearnerStep.Basic(model_basic_replay)

    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 500,
            batch_size= 32)

    for i in range(0, n_tasks):
        for xi,yi in zip(X_tasks[i],y_tasks[i]):
            training_data=TensorDataset(torch.tensor(xi.reshape(1,-1)).float(),torch.tensor([yi]))
            #print('End Tensor')
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            model_basic_replay=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_replay).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_replay).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_replay_{seed}.csv')
    clas,interpret= evaluation(model_basic_replay , InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_basic_replay_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')

'''LwF Implementation'''

if False:#rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_lwf_{seed}'):
    print('LWF')

    if 'CNN' in str(model):
        model_basic_lwf  = model(num_classes=n_classes) 
    else:
        model_basic_lwf  = model(num_classes=n_classes,input_size=input_size) 


    learn =LwF.LwF(model_basic_lwf, class_incremental=True)
    #LearnerStep.Basic(model_basic_synaptic)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(X_tasks[i].reshape(len(X_tasks[i]),-1)).float(),torch.tensor(y_tasks[i]))
            #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
        model_basic_lwf=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_lwf).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_lwf).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_lwf_{seed}.csv')
    clas,interpret= evaluation(model_basic_lwf, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_basic_lwf_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)


    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')
    
'''Synaptic Intelligence'''
if False:# rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_synaptic_{seed}'):
    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 


    learn =Synaptic_Intelligence.Synaptic_Intelligence(model_basic_synaptic)
    #LearnerStep.Basic(model_basic_synaptic)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        #for xi,yi in zip(X_tasks[i],y_tasks[i]):
        training_data=TensorDataset(torch.tensor(X_tasks[i].reshape(len(X_tasks[i]),-1)).float(),torch.tensor(y_tasks[i]))
            #print('End Tensor')
        train_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)
        model_basic_synaptic=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_synaptic).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_synaptic_{seed}.csv')
    clas,interpret= evaluation(model_basic_synaptic, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_basic_synaptic_{seed}',save=True,interpretability_measure=False)
    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')





'''EWC'''
if False:#rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_ewc1_{seed}'):
    if 'CNN' in str(model):
        model_basic_ewc  = model(num_classes=n_classes) 
    else:
        model_basic_ewc  = model(num_classes=n_classes,input_size=input_size) 

    learn = EWC.Elastic_Weight_Regularizer(model_basic_ewc)

    for i in range(0, n_tasks):
        for xi,yi in zip(X_tasks[i],y_tasks[i]):
            training_data=TensorDataset(torch.tensor(xi.reshape(1,-1)).float(),torch.tensor([yi]))
            #print('End Tensor')
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            model_basic_ewc=learn.fit(train_dataloader)
        if i == 0:
            val = Continous_metrics(model_basic_ewc).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_ewc).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_ewc1_{seed}.csv')
    clas,interpret= evaluation(model_basic_ewc, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}',f'model_basic_ewc1_{seed}',save=True,interpretability_measure=False)
    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')


    '''EEIL'''
if False:#rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_eeil.csv'):
    if 'CNN' in str(model):
        model_basic_eeil = model(num_classes=n_classes) 
    else:
        model_basic_eeil  = model(num_classes=n_classes,input_size=input_size) 

    learn =EEIL.EEIL(model_basic_eeil)

    for i in range(0, n_tasks):
        for xi,yi in zip(X_tasks[i],y_tasks[i]):
            training_data=TensorDataset(torch.tensor(xi.reshape(1,-1)).float(),torch.tensor([yi]))
            #print('End Tensor')
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            model_basic_ewc=learn.fit(train_dataloader,t=0)
        if i == 0:
            val = Continous_metrics(model_basic_ewc).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(model_basic_ewc).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/model_basic_eeil.csv')
    clas,interpret= evaluation(model_basic_ewc, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/Sanity/{str_model}','model_basic_eeil',save=True,interpretability_measure=False)
    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    full_classification_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/Sanity/{str_model}/interpretability_metrics_{seed}.csv')