import numpy as np 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Model import SmallModel, train_small
from captum.attr import InputXGradient
import warnings
from CXIL.Benchmarking.metrics.Helper import get_mask
from CXIL.Data.DataLoader import load_data_and_sim,timeseries
from Evaluation_Script import evaluation
from CXIL.Learning import LearnerStep, Replay
#from avalanche.models import SimpleMLP, SimpleCNN, MobilenetV1
from CXIL.Models.SimpleCNN import SimpleCNN,CNN
from CXIL.Models.SimpleModel import SimpleMLP
from CXIL.Models.Simple1DResNet import ResNetBaseline
from CXIL.Learning import LearnerStep, Replay, EWC, Synaptic_Intelligence,LwF, EEIL, MAS
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from CXIL.Benchmarking.metrics.continouus_metrics import Continous_metrics
from CXIL.rrr import RightRightReasons_no_TK as RightRightReasons
import argparse
import random 

'''
RRR

'''
import torch.nn.functional as F
class mod_wrapper():
    def __init__(self, model) -> None:
        self.model = model
        self.model.eval()
        pass
    def predict(self,item):
        item = torch.from_numpy(item).float()
        pred= F.softmax(self.model(item))
        return pred.detach().numpy()



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
parser.add_argument('seed')
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

loss_rrr= LearnerStep.RRR_Loss

#if 'mnist' in dataset: 
#    parameter = 'int_trainpi_correction'
if not os.path.isdir(f'./Results/Continous'):
    os.mkdir(f'./Results/Continous')
if not os.path.isdir(f'./Results/Continous/{data_name}'):
    os.mkdir(f'./Results/Continous/{data_name}')
#if not os.path.isdir(f'./Results/Continous/{data_name}/int_trainpi'):
#    os.mkdir(f'./Results/Continous/{data_name}/int_trainpi')
if not os.path.isdir(f'./Results/Continous/{data_name}/RRR'):
    os.mkdir(f'./Results/Continous/{data_name}/RRR')
if not os.path.isdir(f'./Results/Continous/{data_name}/RRR'):
    os.mkdir(f'./Results/Continous/{data_name}/RRR')
if not os.path.isdir(f'./Results/Continous/{data_name}/RRR/{str_model}'):
    os.mkdir(f'./Results/Continous/{data_name}/RRR/{str_model}')

if not rerun_all: 
    # Try to load data 
    if os.path.exists(f'./Results/Continous/{data_name}/RRR/{str_model}/classification_metrics_{seed}.csv'):
        full_classification_df=pd.read_csv(f'./Results/Continous/{data_name}/RRR/{str_model}/classification_metrics_{seed}.csv')  
        full_classification_df.drop(labels=['Unnamed: 0'],axis=1,inplace=True)
    if os.path.exists(f'./Results/Continous/{data_name}/RRR/{str_model}/classification_metrics_{seed}.csv'):
        full_interpretability_df= pd.read_csv(f'./Results/Continous/{data_name}/RRR/{str_model}/classification_metrics_{seed}.csv')

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

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_ADAM_{seed}.csv'):
    print('Model Continous')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 
    m_wrap_empty=mod_wrapper(model_empty)
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_empty)#
    learn = LearnerStep.Interactive(model_empty,loss_fn=loss)
    int_train= RightRightReasons.RRR(model_empty,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)
    # trans_function(m_wrap_empty.predict,num_samples=1)
    #int_train=int_trainpi(model_empty,  learn,predict_func=m_wrap_empty.predict,transformer=trans, evaluate_data= (X_test,y_test),silent=0,simulation_logic=simulation)

    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        
        
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi, taskid= 0)
        
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_ADAM_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_ADAM_{seed}',save=True,interpretability_measure=False)

    full_classification_df= clas    
    full_interpretability_df= interpret


    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')


'''ADAM & Label Trick'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_ADAM_LT_{seed}.csv'):
    print('ADAM Label Trick')
    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 
    m_wrap_empty=mod_wrapper(model_empty)
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_empty)
    learn = LearnerStep.Interactive(model_empty,loss_fn=loss)
    int_train= RightRightReasons.RRR(model_empty,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)
   
    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi)
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_ADAM_LT_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_ADAM_LT_{seed}',save=True,interpretability_measure=False)
    #print(clas)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')



'''
Continous Case without anything(CAIPI Check) - SGD
'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_SGD_{seed}.csv'):
    print('Model Continous')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 
    m_wrap_empty=mod_wrapper(model_empty)
    #model_basic_continous = model_empty
    # trans_function(m_wrap_empty.predict,num_samples=1)
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_empty)#
    learn = LearnerStep.Interactive(model_empty,optimizer=torch.optim.SGD,loss_fn=loss)
    int_train= RightRightReasons.RRR(model_empty,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)


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

        
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi,taskid=i)
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_SGD_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_SGD_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')


#TODO LESEZEICHEN

'''
Continous Case without anything(CAIPI Check) -SGD - LT
'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_SGD_LT.csv'):
    print('Model Continous')

    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 
    m_wrap_empty=mod_wrapper(model_empty)
    #model_basic_continous = model_empty
    # trans_function(m_wrap_empty.predict,num_samples=1)
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_empty)
    learn = LearnerStep.Interactive(model_empty,optimizer=torch.optim.SGD,loss_fn=loss)
    int_train= RightRightReasons.RRR(model_empty,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)
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

        
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi)
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_SGD_LT_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_SGD_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')



'''EWC '''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_ewc.csv'):

    if 'CNN' in str(model):
        model_basic_ewc  = model(num_classes=n_classes) 
    else:
        model_basic_ewc  = model(num_classes=n_classes,input_size=input_size) 
 
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_basic_ewc)#.full_RRR_loss
    learn =EWC.Elastic_Weight_Regularizer(model_basic_ewc, loss=loss)
    int_train= RightRightReasons.RRR(model_basic_ewc,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)


    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
     
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi,taskid=i)

        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_ewc_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_ewc_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')


'''EWC & LT '''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_ewc_LT_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_ewc  = model(num_classes=n_classes) 
    else:
        model_basic_ewc  = model(num_classes=n_classes,input_size=input_size) 

  
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_basic_ewc)
    learn =EWC.Elastic_Weight_Regularizer(model_basic_ewc, loss=loss)
    int_train= RightRightReasons.RRR(model_basic_ewc,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)
    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi)
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_ewc_LT_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_ewc_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')



''' Synaptic '''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_SI_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_basic_synaptic)
    learn=Synaptic_Intelligence.Synaptic_Intelligence(model_basic_synaptic, loss=loss)
    int_train= RightRightReasons.RRR(model_basic_synaptic,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi, taskid=i)
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_SI_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_SI_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')



''' Synaptic & LT'''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_SI_LT_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 

    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_basic_synaptic)
    learn=Synaptic_Intelligence.Synaptic_Intelligence(model_basic_synaptic, loss=loss)
    int_train= RightRightReasons.RRR(model_basic_synaptic,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)

    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi)
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_SI_LT_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_SI_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')

'''MAS'''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_MAS_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 

    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_basic_synaptic)
    learn=MAS.MAS(model_basic_synaptic, loss=loss)


    int_train= RightRightReasons.RRR(model_basic_synaptic,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)
    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        #if 'meta_with_finetuning' in locals():
        #    meta=meta_tasks[i]
        #    simulation.update_meta(meta)

        
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi,taskid=i)
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_MAS_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_MAS_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')

'''MAS LT '''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_MAS_LT_{seed}.csv'):

    if 'CNN' in str(model):
        model_basic_synaptic  = model(num_classes=n_classes) 
    else:
        model_basic_synaptic  = model(num_classes=n_classes,input_size=input_size) 

    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_basic_synaptic)
    learn=MAS.MAS(model_basic_synaptic, loss=loss)

    int_train= RightRightReasons.RRR(model_basic_synaptic,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)
    for i in range(0, n_tasks):
        print(f'TASK {i}')
        xi=X_tasks[i]
        yi=y_tasks[i]

        
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(xi,yi)
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)

    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_MAS_LT_{seed}.csv')
    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_MAS_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')


'''REPLAY'''
if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_replay_{seed}.csv'):
    print('REPLAY')

    if 'CNN' in str(model):
        model_basic_replay  = model(num_classes=n_classes) 
    else:
        model_basic_replay  = model(num_classes=n_classes,input_size=input_size)

    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_basic_replay)
    #learn=LearnerStep.Basic(model_basic_replay,loss=loss)
    learn = LearnerStep.Interactive(model_basic_replay, loss_fn=loss)
    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 100,
            batch_size= 32, interact_only_on_curr=True)
    int_train= RightRightReasons.RRR(model_basic_replay,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)

    
    #TODO is here a Issue
    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(X_tasks[i],y_tasks[i],taskid=i)
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_replay_{seed}.csv')
    clas,interpret= evaluation(int_train.model , InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_replay_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')

'''Replay & LT '''

if rerun_all or not os.path.exists(f'./Results/Continous/{dataset}/RRR/{str_model}/model_replay_LT_{seed}.csv'):
    print('REPLAY')

    if 'CNN' in str(model):
        model_basic_replay  = model(num_classes=n_classes) 
    else:
        model_basic_replay  = model(num_classes=n_classes,input_size=input_size) 
    m_wrap_empty=mod_wrapper(model_basic_replay)
    learn = LearnerStep.Basic(model_basic_replay)
    # trans_function(m_wrap_empty.predict,num_samples=1)
    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 100,
            batch_size= 32, interact_only_on_curr=True)
    
    int_train= RightRightReasons.RRR(model_basic_replay,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=0)

    for i in range(0, n_tasks):
        xi=X_tasks[i]
        yi=y_tasks[i]
        #simulation.update_original_labels(yi)
        acc,f1, precision, recall,time_list,_,_=int_train.iterate(X_tasks[i],y_tasks[i])
        if i == 0:
            val = Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)
        else:
            val=pd.concat([val,Continous_metrics(int_train.model).get_evaluation(X_test,y_test, X_test_task,y_test_task)], ignore_index=True)
    val.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/model_replay_LT_{seed}.csv')
    clas,interpret= evaluation(int_train.model , InputXGradient, (X_test,y_test),f'./Results/Continous/{dataset}/RRR/{str_model}',f'model_replay_LT_{seed}',save=True,interpretability_measure=False)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

    '''Saves the metrics & Plots'''
    full_classification_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/Continous/{dataset}/RRR/{str_model}/interpretability_metrics_{seed}.csv')

