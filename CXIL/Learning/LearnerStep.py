import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import numpy as np 
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from XIL.Learning.StoragePolicies import ClassBalancedBuffer

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from XIL.Learning.StoragePolicies import ReservoirSamplingBuffer
import time
#global x_curr


class Learner():
    def __init__(self) -> None:
        pass
    def fit(self)->None:
        pass

#class HedgeBackPropagation(Learner):
#    '''Implementation according to : '''
#    def __init__(self, fit_func,layer_dims=None, X_test=None, y_test=None) -> None:
#        self.fit_func=fit_func
#        self.optimizer = torch.optim.SGD(fit_func.parameters(), lr=0.001)#Adam(fit_func.parameters(), lr=0.001)  
#        self.loss_fn   = nn.CrossEntropyLoss()
#        self.test=False
#        self.batch = []
#        self.layer_dims=layer_dims
#        self.y=[]
#        self.attribution=[] 

#        if X_test is not None: 
#            self.X_test  = Variable(torch.from_numpy(X_test)).float()
#            self.Y_test  = Variable(torch.from_numpy(y_test)).float()
#            self.test=True
#
#    def fit(self,x,y, batch_size=1)->None:
#        loss_function = nn.BCELoss()
#        alpha = [1/(len(self.layer_dims)-2)] * (len(self.layer_dims)-2)
#        self.fit_func.train()
#        EPOCHS  = 1
#        if isinstance(x,np.ndarray):
#            x=x.tolist()
#        if isinstance(y,np.ndarray):
#            y=y.tolist()
#        if len(self.batch) <batch_size-1 and len(x)<batch_size-1:
#            #print('Case 1')
#           
#            if len(self.batch)==0:
#                #print('Case 2')
#                self.batch=x
#                self.y=y
#            else:
#                #print('Case 3')
#                self.batch.extend(x)
#                self.y.extend(y)
#            return self.fit_func
#        else:
#            #print('Case 4')
#            if len(self.batch)==0:
#                self.batch=x
#                self.y=y
#            else:
#                self.batch.extend(x)
#                self.y.extend(y)
#           
#            X_train= Variable(torch.from_numpy(np.array(self.batch))).float()
#            self.batch=[]
#            #TODO make this flexible 
#            y=Variable(torch.from_numpy(np.array(self.y).reshape(-1,2))).float()
#            self.y=[]
#        #get prediction
#        y_pred = self.fit_func(X_train)
#        loss = self.loss_fn(y_pred.float(),y.float())
#        #Calculatare Loss #

        #out1, out2, out3 = self.fit_func(X_train)

#        losses = []

#        loss1 = loss_function(out1, y) 
#        loss2 = loss_function(out2, y) 
#        loss3 = loss_function(out3,y) 


#        loss = loss1 * alpha[0] + loss2 * alpha[1] + loss3 * alpha[2] 

#        losses.append(loss1.item());losses.append(loss2.item());losses.append(loss3.item());
        #losses.append(loss5.item());losses.append(loss6.item());losses.append(loss7.item());losses.append(loss8.item());
#        M = sum(losses)
#        losses = [loss / M for loss in losses]
#        min_loss = np.amin(losses)
#        max_loss = np.amax(losses)
#        range_loss = max_loss - min_loss
#
#        losses = [(loss-min_loss)/range_loss for loss in losses]
#        alpha_u = [0.99 ** loss for loss in losses]
#        alpha = [a * w for a, w in zip(alpha_u, alpha)]#

#        alpha = [ max(0.01, a) for a in alpha]
#        M = sum(alpha)
#        alpha = [a / M for a in alpha]

#        return self.fit_func


def RRR_Loss(inputs,y_pred, labels,  annotation_matrix, input_gradients, fit_func, loss_fn=nn.CrossEntropyLoss(reduction='sum'), lamda1=1000,lamda2=0.0001, taskid= None):
    #start=time.time()
    def _dot_product(X,Y,batch_size,d):
        X = X.reshape(batch_size, 1, -1)
        Y = Y.reshape(batch_size, -1, 1)
        product = torch.matmul(X, Y).squeeze(1)
        return product
    # TODO Only Update relevant heads 

    #if taskid is not None:
    #    l_groups=labels.unique().sort()[0]
    #    print(l_groups)
    #    for lbl_idx, lbl in enumerate(l_groups):
    #            labels[labels == lbl] = lbl_idx
    #            loss = loss_fn(y_pred[:, l_groups.detach().numpy().tolist()].float(), labels.long()) 
    #else: 
    loss = loss_fn(y_pred.float(), labels.long())
    #bce =loss #loss_fn(y_pred.float(), labels.long()) # Used to be axis =1      
    #loss=bce

    params = []
    for param in fit_func.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    
    explanation_penalty= lamda1* torch.sum(_dot_product(torch.Tensor(annotation_matrix)*input_gradients,torch.Tensor(annotation_matrix)*input_gradients,len(inputs),input_gradients.shape[-1]))#,ord=2)#self.input_gradient.attribute(X_train, target=np.argmax(y_pred.detach().numpy(),axis=1))
    smallparams=lamda2*torch.dot(params,params)
    loss= loss+explanation_penalty
    loss=loss+smallparams
    #end=time.time()
    #print('DURATION RRR_Loss', end- start)
    #import sys 
    #sys.exit(1)
    return loss 

class RRR_Learner():

    def __init__(self,input_gradients, annotation_matrix,fit_func):
        self.input_gradients=input_gradients(fit_func)
        self.annotation_matrix=annotation_matrix
        self.fit_func = fit_func
    def __call__(self,inputs,labels, taskid=None,eval=False):
        # Make predictions for this batch

        outputs = self.fit_func(inputs)

        # Compute the loss and its gradients
        importance=[]
        annotation=[]
        igs=self.input_gradients.attribute(inputs=inputs,target=np.argmax(outputs.detach().numpy(),axis=1).tolist()).detach().numpy()
        j=0
       

        for item in inputs:
            print(item.shape)
            # TODO Make this Batchable for SpeedUp
            
            #print(globals()['__builtins__'].keys())
            #print('x_curr' in globals())
        

            if 'x_curr' in globals()['__builtins__']:
                #print('x_curr exists')
                x_curr=globals()['__builtins__']['x_curr']
                #print(item.float().detach().numpy().reshape(-1))
                #print(x_curr.float().detach().numpy().reshape(-1))
                if not np.all(item.float().detach().numpy().reshape(-1) ==  x_curr.float().detach().numpy().reshape(-1)):
                    annotation.append(np.zeros_like(igs[0]))
                    importance.append(np.zeros_like(igs[0]))
                    continue
                else:
                    importance.append(igs[j])
                    print('Item FOUND')
                    #import sys 
                    #sys.exit(1)
            ig=igs[j]#self.input_gradients.attribute(inputs=item.reshape(1,*shape[1:]),target=int(np.argmax(outputs[j].detach().numpy()))).detach().numpy()

            print('Only after ITEM Found')
            print('ig ',ig.shape)
            print('item ', item.shape)
            l, anno=self.annotation_matrix(item.float().detach().numpy(),int(np.argmax(outputs[j].detach().numpy())),ig)
                        
            #importance.append(ig[0])
            if len(anno)== 0: 
                anno= np.zeros_like(ig)
            print('anno ',anno.shape)
            print('ig ',ig.shape)
            annotation.append(anno.reshape(*ig.shape))
            j=j+1
        if len(importance)<1:
            importance=igs
        loss = RRR_Loss(inputs,outputs, labels,np.array(annotation), np.array(importance),self.fit_func, taskid=taskid)
        return loss

class Interactive(Learner):
    '''Basic Version of Basic Learner (without batcch )'''
    def __init__(self, fit_func, loss_fn = RRR_Loss, lr=0.00001, optimizer=torch.optim.SGD,X_test=None, y_test=None) -> None:
        self.fit_func=fit_func
        self.optimizer = optimizer(fit_func.parameters(), lr=lr)#Adam(fit_func.parameters(), lr=0.001)  
        self.loss_fn   = loss_fn
        self.test=False
        self.y=[]
        self.attribution=[] 
        self.mode=None
        self.shape= None
        if X_test is not None: 
            self.X_test  = Variable(torch.from_numpy(X_test)).float()
            self.Y_test  = Variable(torch.from_numpy(y_test)).float()
            self.test=True


    def fit(self,X_train, taskid=None, EPOCHS=1)->None:
        '''
        X_train : 
        input_gradients: Function to Calculate Explanations
        annotation_matrix: Function to get Feedback
        
        '''
        start = time.time()

        self.fit_func.train()

        for epoch in tqdm.trange(EPOCHS):
            for i, data in enumerate(X_train):
                # Iterate Batches
                inputs, labels = data
                shape=inputs.shape
                print('SHAPE',shape)
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()

  
                loss = self.loss_fn(inputs, labels,taskid)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()
        end=time.time()
        print('Interactive ', end-start)

        return self.fit_func

class Basic(Learner):
    '''Basic Version of Basic Learner (without batcch )'''
    def __init__(self, fit_func, loss = nn.CrossEntropyLoss(), lr=0.001, optimizer=torch.optim.Adam,X_test=None, y_test=None, **kwargs_optimizer) -> None:
        self.fit_func=fit_func
        self.optimizer = optimizer(fit_func.parameters(), lr=lr, **kwargs_optimizer)#Adam(fit_func.parameters(), lr=0.001)  
        self.loss_fn   = loss
        self.test=False
        self.y=[]
        self.attribution=[] 
        if X_test is not None: 
            self.X_test  = Variable(torch.from_numpy(X_test)).float()
            self.Y_test  = Variable(torch.from_numpy(y_test)).float()
            self.test=True


    def fit(self,X_train,taskid=None, EPOCHS=1)->None:
        '''
        Attributes: 
            X_train torch.DataLoader: Data to be fit. Labels as INT !
        '''
        self.fit_func.train()
        running_loss = 0.
        loss_list     = np.zeros((EPOCHS,))

        for epoch in tqdm.trange(EPOCHS):

            for i, data in enumerate(X_train):
                inputs, labels = data
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                # Make predictions for this batch
                outputs = self.fit_func(inputs)

                # Compute the loss and its gradients
                if taskid is not None:
                    l_groups=labels.unique().sort()[0]
                    for lbl_idx, lbl in enumerate(l_groups):
                        labels[labels == lbl] = lbl_idx
                    loss = self.loss_fn(outputs[:, l_groups.detach().numpy().tolist()].float(), labels.long()) 
                else: 
                    loss = self.loss_fn(outputs.float(), labels.long())
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if i % 1000 == 999:
                    last_loss = running_loss / 1000 # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    running_loss = 0.
        return self.fit_func



