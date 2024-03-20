import numpy as np 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Model import SmallModel, train_small
from captum.attr import InputXGradient
import warnings
from CXIL.caipi import utils, Transformer
from CXIL.Benchmarking.metrics.Helper import get_mask
from CXIL.Data.DataLoader import load_data_and_sim,timeseries
from Evaluation_Script import evaluation
from CXIL.Learning import LearnerStep, Replay
#from avalanche.models import SimpleMLP, SimpleCNN, MobilenetV1
from CXIL.Models.SimpleCNN import SimpleCNN,CNN
from CXIL.Models.SimpleModel import SimpleMLP
from CXIL.Models.Simple1DResNet import ResNetBaseline
from CXIL.rrr import RightRightReasons_no_TK as RightRightReasons
#THIS IS Server Settigng ! 
from CXIL.caipi.CAipi_without_TKadd import caipi
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import argparse
import random

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
dataset= args.dataset #'./CXIL/Data/data/TimeSeries/Testing/data/SimulatedTestingData_Middle_Harmonic_F_50_TS_50.npy' #'toy_classification'
data_name= args.data_name #'SimulatedTestingData_Middle_Harmonic_F_50_TS_50'
model = locals()[f"{args.model}"]#args.model  #SimpleMLP
#dataset= 'tabular'#'iris_cancer' #'toy_classification'
#data_name=dataset
'''END PARAMETER SECTION'''

'''SET SEEED'''
seed=int(args.seed)
print(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
'''Finsihed Seed'''
silent= 0

str_model=str(model).split('.')[-1].replace('\'>','')



parameter = 'MeanandstdTrans'
if not os.path.isdir(f'./Results/{data_name}'):
    os.mkdir(f'./Results/{data_name}')
if not os.path.isdir(f'./Results/{data_name}/RRR'):
    os.mkdir(f'./Results/{data_name}/RRR')
if not os.path.isdir(f'./Results/{data_name}/RRR'):
    os.mkdir(f'./Results/{data_name}/RRR')
if not os.path.isdir(f'./Results/{data_name}/RRR/{str_model}'):
    os.mkdir(f'./Results/{data_name}/RRR/{str_model}')



if not rerun_all: 
    # Try to load data 
    if os.path.exists(f'./Results/{data_name}/RRR/{str_model}/classification_metrics_{seed}.csv'):
        full_classification_df=pd.read_csv(f'./Results/{data_name}/RRR/{str_model}/classification_metrics_{seed}.csv')  
        full_classification_df.drop(labels=['Unnamed: 0'],axis=1,inplace=True)
    elif os.path.exists(f'./Results/{data_name}/classification_metrics_{seed}.csv'):
        full_classification_df=pd.read_csv(f'./Results/{data_name}/classification_metrics_{seed}.csv')  
        full_classification_df.drop(labels=['Unnamed: 0'],axis=1,inplace=True)
    if os.path.exists(f'./Results/{data_name}/RRR/{str_model}/interpretability_metrics_{seed}.csv'):
        full_interpretability_df= pd.read_csv(f'./Results/{data_name}/RRR/{str_model}/interpretability_metrics_{seed}.csv')
        full_interpretability_df.drop(labels=['Unnamed: 0'],axis=1,inplace=True)
    elif os.path.exists(f'./Results/{data_name}/interpretability_metrics_{seed}.csv'):
        full_interpretability_df=pd.read_csv(f'./Results/{data_name}/interpretability_metrics_{seed}.csv')  
        full_interpretability_df.drop(labels=['Unnamed: 0'],axis=1,inplace=True)


'''Load Data '''
if 'Simulated' in dataset or 'time' in dataset:#os.path.isfile(dataset):
    print('----TimeSeries----')
    X,y,X_train,y_train,X_test,y_test,  meta_train, meta_test, test_meta,simulation= load_data_and_sim(dataset)
    #simulation= timeseries(meta_train)
    #print('test_meta', test_meta.shape)
    ground_truth= simulation.get_ground_truth(X_test) #get_mask(X_test,start=test_meta[:,1:3],end=test_meta[:,3:])
    print('X ', X.shape)
    print('y ', y.shape)
    print('X_train ', X_train.shape)
    print('Y_train ', y_train.shape)
    print('X_test ', X_test.shape)
    print('y_test ', y_test.shape)

else:
    print(f'----{dataset}----')
    X,y,X_train,y_train,X_test,y_test, simulation= load_data_and_sim(dataset)
    print(str(simulation))
    ground_truth = simulation.get_ground_truth(X_test)
    #print('X ', X.shape)
    #print('y ', y.shape)
    #print('X_train ', X_train.shape)
    #print('Y_train ', y_train.shape)
    #print('X_test ', X_test.shape)
    #print('y_test ', y_test.shape)

n_classes= len(np.unique(y))
print('Number Classes',n_classes)

input_size=1
for i in X.shape[1:]:
    input_size = input_size*i

X_with_finetuning=X
y_with_finetuning=y

if X_train is not None:
    X_with_finetuning=np.concatenate((X,X_train))#[0:10]
    y_with_finetuning=np.concatenate((y,y_train))#[0:10]
if 'Simulated' in dataset:#os.path.isfile(dataset):
    meta_with_finetuning=np.concatenate((meta_train,meta_test))#[0:10]

print('INPUT', input_size)

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

'''Basic Iterate Baseline'''
if 'CNN' in str(model):
    model_our = model(num_classes=n_classes) 

else:
    if 'ResNet' in str(model):
        input_size=X_train.shape[1]
    model_our = model(num_classes=n_classes,input_size=input_size) 
if rerun_all or not os.path.exists(f'./Results/{data_name}/iter_{seed}'):
    learn = LearnerStep.Basic(model_our,optimizer=torch.optim.SGD)


    for xi,yi in zip(X_with_finetuning,y_with_finetuning):
            training_data=TensorDataset(torch.tensor(xi[np.newaxis,:]).float(),torch.tensor([yi]))
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            model_our=learn.fit(train_dataloader)

    clas,interpret= evaluation(model_our, InputXGradient, (X_test,y_test),f'./Results/{data_name}',f'iter_{seed}',save=True,ground_truth=ground_truth)

    full_classification_df= clas
    full_interpretability_df= interpret
    full_classification_df.to_csv(f'./Results/{data_name}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/{data_name}/interpretability_metrics_{seed}.csv')
#clas,interpret= evaluation(model_our, InputXGradient, (X,y))

#'''Basic Iterate Baseline'''#

#model_our = SimpleMLP(num_classes=n_classes,input_size=input_size) 

#learn = LearnerStep.Basic(model_our)


#for xi,yi in zip(X,y):
#        training_data=TensorDataset(torch.tensor(xi.reshape(1,-1)).float(),torch.tensor([yi]))
#        train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
#        model_our=learn.fit(train_dataloader)

#clas,interpret= evaluation(model_our, InputXGradient, (X_test,y_test),f'./Results/{data_name}/RRR','iter half',save=True,ground_truth=ground_truth)

#full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
#full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)

if rerun_all or not os.path.exists(f'./Results/{data_name}/model_full_{seed}'):
    '''Klassisches Training'''
    if 'CNN' in str(model):
        model_full = model(num_classes=n_classes) 
    else: 
        model_full = model(num_classes=n_classes,input_size=input_size)  #SmallModel(X_train.shape[1], len(np.unique(y_train)))
    optimizer = torch.optim.Adam(model_full.parameters(), lr=0.01)
    loss_fn   = nn.CrossEntropyLoss()
    epochs =100

    acc_full=train_small(model_full, X_with_finetuning,y_with_finetuning,X_test,y_test,optimizer,loss_fn, epochs)

    clas,interpret= evaluation(model_full, InputXGradient, (X_test,y_test),f'./Results/{data_name}/',f'model_full_{seed}',save=True,ground_truth=ground_truth)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_interpretability_df.to_csv(f'./Results/{data_name}/interpretability_metrics_{seed}.csv')
    full_classification_df.to_csv(f'./Results/{data_name}/classification_metrics_{seed}.csv')
'''
if rerun_all or not os.path.exists(f'./Results/{data_name}/model_half'):

    if 'CNN' in str(model):
        model_half = model(num_classes=n_classes) 
    else: 
        model_half = model(num_classes=n_classes,input_size=input_size)#SmallModel(X_train.shape[1], len(np.unique(y_train)))
    optimizer = torch.optim.Adam(model_half.parameters(), lr=0.01)
    loss_fn   = nn.CrossEntropyLoss()
    epochs =10
    acc_half=train_small(model_half, X,y,X_test,y_test,optimizer,loss_fn, epochs)
    clas,interpret= evaluation(model_half, InputXGradient, (X_test,y_test),f'./Results/{data_name}/','model_half',save=True,ground_truth=ground_truth)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_interpretability_df.to_csv(f'./Results/{data_name}/interpretability_metrics.csv')
    full_classification_df.to_csv(f'./Results/{data_name}/classification_metrics.csv')
'''
if rerun_all or not os.path.exists(f'./Results/{data_name}/model_empty_{seed}'):
    '''No Training'''
    # No Train Only Model instantiation
    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else: 
        model_empty = model(num_classes=n_classes,input_size=input_size) #SmallModel(X_train.shape[1], len(np.unique(y_train)))
    clas,interpret= evaluation(model_empty, InputXGradient, (X_test,y_test),f'./Results/{data_name}/',f'model_empty_{seed}',save=True,ground_truth=ground_truth)
    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_interpretability_df.to_csv(f'./Results/{data_name}/interpretability_metrics_{seed}.csv')
    full_classification_df.to_csv(f'./Results/{data_name}/classification_metrics_{seed}.csv')
    

if rerun_all or not os.path.exists(f'./Results/{data_name}/RRR/{str_model}/empty_trained_{seed}'):
    '''No Training & RRR- iterativly'''
    # No Train Only Model instantiation
    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else: 
        model_empty = model(num_classes=n_classes,input_size=input_size) 
    m_wrap_empty=mod_wrapper(model_empty)

    #trans= Transformer.DataBasedTransformer(X_train, y_train,m_wrap_empty.predict)
    #learn = LearnerStep.Interactive(model_empty)
    #int_train= RightRightReasons.RRR(model_empty,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=silent)
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_empty)
    learn = LearnerStep.Interactive(model_empty,loss_fn=loss)
    int_train= RightRightReasons.RRR(model_empty,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=silent)

    #update
    #simulation.update_original_labels(y_with_finetuning)
    #if 'meta_with_finetuning' in locals():
    #    simulation.update_meta(meta_with_finetuning)

    #cai=caipi(model_empty,  learn,predict_func=m_wrap_empty.predict,transformer=trans, evaluate_data= (X_test,y_test),silent=silent,simulation_logic=simulation)
    acc,f1, precision, recall,time_list,loss_calc_test,loss_calc_train=int_train.iterate(X_with_finetuning, y_with_finetuning, taskid=None)
    h=f'emptytrained_{seed}'
    pd.DataFrame(acc).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_acc_{h}.csv')
    pd.DataFrame(f1).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_f1_{h}.csv')
    pd.DataFrame(precision).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_precision_{h}.csv')
    pd.DataFrame(recall).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_recall_{h}.csv')
    pd.DataFrame(time_list).to_csv(f'./Results/{data_name}/RRR/{str_model}/time_{h}.csv')
    pd.DataFrame(loss_calc_test).to_csv(f'./Results/{data_name}/RRR/{str_model}/loss_test_{h}.csv')
    pd.DataFrame(loss_calc_train).to_csv(f'./Results/{data_name}/RRR/{str_model}/losss_train_{h}.csv')


    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/{data_name}/RRR/{str_model}',f'empty_trained_{seed}',save=True,ground_truth=ground_truth)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_classification_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/interpretability_metrics_{seed}.csv')
'''
if rerun_all or not os.path.exists(f'./Results/{data_name}/RRR/{str_model}/model_half_trained'):
    if 'CNN' in str(model):
        model_half_n = model(num_classes=n_classes) 
    else: 
        model_half_n = model(num_classes=n_classes,input_size=input_size)
    if os.path.exists(f'./Results/{data_name}/model_half'):
        model_half= torch.load(f'./Results/{data_name}/model_half')
        model_half_n.load_state_dict(model_half)
    else:
       model_half_n.load_state_dict(model_half.state_dict())
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_half_n)
    learn = LearnerStep.Interactive(model_half_n,loss_fn=loss)
    int_train= RightRightReasons.RRR(model_half_n,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=silent)

    #simulation.update_original_labels(y_train)

    #if 'meta_train' in locals():
    #    simulation.update_meta(meta_train)

    acc,f1, precision, recall,time_list,loss_calc_test,loss_calc_train=int_train.iterate(X_train,y_train, taskid=None)
    h='modelhalftrained'
    pd.DataFrame(acc).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_acc_{h}.csv')
    pd.DataFrame(f1).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_f1_{h}.csv')
    pd.DataFrame(precision).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_precision_{h}.csv')
    pd.DataFrame(recall).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_recall_{h}.csv')
    pd.DataFrame(time_list).to_csv(f'./Results/{data_name}/RRR/{str_model}/time_{h}.csv')
    pd.DataFrame(loss_calc_test).to_csv(f'./Results/{data_name}/RRR/{str_model}/loss_test_{h}.csv')
    pd.DataFrame(loss_calc_train).to_csv(f'./Results/{data_name}/RRR/{str_model}/losss_train_{h}.csv')


    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/{data_name}/RRR/{str_model}','model_half_trained',save=True,ground_truth=ground_truth)


    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_classification_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/classification_metrics.csv')
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_interpretability_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/interpretability_metrics.csv')

'''
if rerun_all or not os.path.exists(f'./Results/{data_name}/RRR/{str_model}/model_full_trained_{seed}'):
    '''Full Training & CAIPI'''
    if 'CNN' in str(model):
        model_full_n = model(num_classes=n_classes) 
    else:
        model_full_n=model(num_classes=n_classes,input_size=input_size)
    if os.path.exists(f'./Results/{data_name}/model_full_{seed}'):
        model_full= torch.load(f'./Results/{data_name}/model_full_{seed}')
        model_full_n.load_state_dict(model_full)
    else:

       model_full_n.load_state_dict(model_full.state_dict())
    
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_full_n)
    learn = LearnerStep.Interactive(model_full_n,loss_fn=loss)
    int_train= RightRightReasons.RRR(model_full_n,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=silent)

    acc,f1, precision, recall,time_list,loss_calc_test,loss_calc_train=int_train.iterate(X_with_finetuning,y_with_finetuning, taskid=None)
    h=f'modelfulltrained_{seed}'
    pd.DataFrame(acc).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_acc_{h}.csv')
    pd.DataFrame(f1).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_f1_{h}.csv')
    pd.DataFrame(precision).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_precision_{h}.csv')
    pd.DataFrame(recall).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_recall_{h}.csv')
    pd.DataFrame(time_list).to_csv(f'./Results/{data_name}/RRR/{str_model}/time_{h}.csv')
    pd.DataFrame(loss_calc_test).to_csv(f'./Results/{data_name}/RRR/{str_model}/loss_test_{h}.csv')
    pd.DataFrame(loss_calc_train).to_csv(f'./Results/{data_name}/RRR/{str_model}/losss_train_{h}.csv')

    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/{data_name}/RRR/{str_model}',f'model_full_trained_{seed}',save=True,ground_truth=ground_truth)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)

    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_interpretability_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/interpretability_metrics_{seed}.csv')

if rerun_all or not os.path.exists(f'./Results/{data_name}/RRR/{str_model}/empty_trained_replay_{seed}'):
    print('Replay')
    '''Everything Again in Batch Mode'''
    #TODO INSTEAD OF INTERACTIE -- BASIC ?= 

    '''No Training & CAIPI'''
    if 'CNN' in str(model):
        model_empty = model(num_classes=n_classes) 
    else:
        model_empty = model(num_classes=n_classes,input_size=input_size) 
    m_wrap_empty=mod_wrapper(model_empty)

    #simulation.update_original_labels(y_with_finetuning)
    #print('Update ', len(y_with_finetuning))
    #if 'meta_train' in locals():
    #    print('meta_traibn')
    #    simulation.update_meta(meta_with_finetuning, X_with_finetuning)
    #import sys 
    #sys.exit(1)
    #learn = LearnerStep.Basic(model_empty)#LearnerStep.BasicLearner(model_empty)
    #learn_replay=Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 1000,batch_size= 64)
    #simulation= timeseries(dataset,meta_with_finetuning).simulate
    #cai=caipi(model_empty,  learn_replay,predict_func=m_wrap_empty.predict,transformer=trans, evaluate_data= (X_test,y_test),silent=silent,simulation_logic=simulation)
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation,model_empty)
    #learn=LearnerStep.Basic(model_basic_replay,loss=loss)
    learn = LearnerStep.Interactive(model_empty, loss_fn=loss)
    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 100,
            batch_size= 32, interact_only_on_curr=True)
    int_train= RightRightReasons.RRR(model_empty,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=silent)



    acc,f1, precision, recall,time_list,loss_calc_test,loss_calc_train=int_train.iterate(X_with_finetuning,y_with_finetuning, taskid=None)
    h=f'empty_trained_replay_{seed}'
    pd.DataFrame(acc).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_acc_{h}.csv')
    pd.DataFrame(f1).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_f1_{h}.csv')
    pd.DataFrame(precision).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_precision_{h}.csv')
    pd.DataFrame(recall).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_recall_{h}.csv')
    pd.DataFrame(time_list).to_csv(f'./Results/{data_name}/RRR/{str_model}/time_{h}.csv')
    pd.DataFrame(loss_calc_test).to_csv(f'./Results/{data_name}/RRR/{str_model}/loss_test_{h}.csv')
    pd.DataFrame(loss_calc_train).to_csv(f'./Results/{data_name}/RRR/{str_model}/losss_train_{h}.csv')


    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/{data_name}/RRR/{str_model}',f'empty_trained_replay_{seed}',save=True,ground_truth=ground_truth)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_classification_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/interpretability_metrics_{seed}.csv')
'''
if rerun_all or not os.path.exists(f'./Results/{data_name}/RRR/{str_model}/model_half_trained_replay'):
    
    if 'CNN' in str(model):
        model_half_n = model(num_classes=n_classes) 
    else:
        model_half_n = model(num_classes=n_classes,input_size=input_size)
    if os.path.exists( f'./Results/{data_name}/model_half'):
        model_half=torch.load(f'./Results/{data_name}/model_half')
        model_half_n.load_state_dict(model_half)
    else:
        model_half_n.load_state_dict(model_half.state_dict())
    m_wrap_empty=mod_wrapper(model_half_n)
    #trans= Transformer.MeanAndStdTransformer(m_wrap_empty.predict,num_samples=1)
    #exp=explainer.explain_instance
    #learn = LearnerStep.Basic(model_half_n)#LearnerStep.BasicLearner(model_empty)
    #simulation= timeseries(dataset,meta_with_finetuning).simulate
    #simulation.update_original_labels(y_train)
    #if 'meta_train' in locals():
    #    simulation.update_meta(meta_train,X_train)
    #learn_replay=Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 1000,batch_size= 64)
    #cai=caipi(model_half_n,  learn,predict_func=m_wrap_empty.predict,transformer=trans, evaluate_data= (X_test,y_test),silent=silent,simulation_logic=simulation)

    loss= LearnerStep.RRR_Learner(InputXGradient,simulation, model_half_n)
    #learn=LearnerStep.Basic(model_basic_replay,loss=loss)
    learn = LearnerStep.Interactive( model_half_n, loss_fn=loss)

    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 100,
            batch_size= 64, interact_only_on_curr=True)
    int_train= RightRightReasons.RRR(model_half_n,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=silent)

    acc,f1, precision, recall,time_list,loss_calc_test,loss_calc_train=int_train.iterate(X_train,y_train, taskid=None)
    h='modelhalftrainedreplay'
    pd.DataFrame(acc).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_acc_{h}.csv')
    pd.DataFrame(f1).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_f1_{h}.csv')
    pd.DataFrame(precision).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_precision_{h}.csv')
    pd.DataFrame(recall).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_recall_{h}.csv')
    pd.DataFrame(time_list).to_csv(f'./Results/{data_name}/RRR/{str_model}/time_{h}.csv')
    pd.DataFrame(loss_calc_test).to_csv(f'./Results/{data_name}/RRR/{str_model}/loss_test_{h}.csv')
    pd.DataFrame(loss_calc_train).to_csv(f'./Results/{data_name}/RRR/{str_model}/losss_train_{h}.csv')

    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/{data_name}/RRR/{str_model}','model_half_trained_replay',save=True,ground_truth=ground_truth)


    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_classification_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/classification_metrics.csv')

    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_interpretability_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/interpretability_metrics.csv')
'''
if rerun_all or not os.path.exists(f'./Results/{data_name}/RRR/{str_model}/model_full_trained_replay_{seed}'):
    '''Full Training & CAIPI'''
    if 'CNN' in str(model):
        model_full_n = model(num_classes=n_classes) 
    else:
        model_full_n=model(num_classes=n_classes,input_size=input_size)
    if os.path.exists( f'./Results/{data_name}/model_full_{seed}'):
        model_full=torch.load(f'./Results/{data_name}/model_full_{seed}')
        model_full_n.load_state_dict(model_full)
    else:
        model_full_n.load_state_dict(model_full.state_dict())
    m_wrap_empty=mod_wrapper(model_full_n)
    #simulation= timeseries(dataset,meta_train).simulate
    #exp=explainer.explain_instance
    #ginal_labels(y_train)
    #if 'meta_train' in locals():
    #    simulation.update_meta(meta_train,X_train)
    #learn = LearnerStep.Basic(model_full_n)#LearnerStep.BasicLearner(model_empty)
    #learn_replay=Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 1000,batch_size= 64)
    #cai=caipi(model_full_n, learn_replay,predict_func=m_wrap_empty.predict,transformer=trans, evaluate_data= (X_test,y_test),silent=silent,simulation_logic=simulation)
    loss= LearnerStep.RRR_Learner(InputXGradient,simulation, model_full_n)
    #learn=LearnerStep.Basic(model_basic_replay,loss=loss)
    learn = LearnerStep.Interactive( model_full_n, loss_fn=loss)
    learn= Replay.ReplayStrategyLearner(learn, num_classes=n_classes,mem_size = 100,
            batch_size= 32, interact_only_on_curr=True)
    int_train= RightRightReasons.RRR(model_full_n,learn,test_data=(X_test,y_test),simulation_logic=simulation,silent=silent)

    acc,f1, precision, recall, time_list,loss_calc_test,loss_calc_train=int_train.iterate(X_with_finetuning,y_with_finetuning, taskid=None)

    h=f'modelfulltrainedreplay_{seed}'
    pd.DataFrame(acc).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_acc_{h}.csv')
    pd.DataFrame(f1).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_f1_{h}.csv')
    pd.DataFrame(precision).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_precision_{h}.csv')
    pd.DataFrame(recall).to_csv(f'./Results/{data_name}/RRR/{str_model}/running_recall_{h}.csv')
    pd.DataFrame(time_list).to_csv(f'./Results/{data_name}/RRR/{str_model}/time_{h}.csv')
    pd.DataFrame(loss_calc_test).to_csv(f'./Results/{data_name}/RRR/{str_model}/loss_test_{h}.csv')
    pd.DataFrame(loss_calc_train).to_csv(f'./Results/{data_name}/RRR/{str_model}/losss_train_{h}.csv')

    clas,interpret= evaluation(int_train.model, InputXGradient, (X_test,y_test),f'./Results/{data_name}/RRR/{str_model}',f'model_full_trained_replay_{seed}',save=True,ground_truth=ground_truth)

    full_classification_df=pd.concat([full_classification_df,clas], ignore_index=True)
    full_classification_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/classification_metrics_{seed}.csv')
    full_interpretability_df=pd.concat([full_interpretability_df,interpret], ignore_index=True)
    full_interpretability_df.to_csv(f'./Results/{data_name}/RRR/{str_model}/interpretability_metrics_{seed}.csv')
