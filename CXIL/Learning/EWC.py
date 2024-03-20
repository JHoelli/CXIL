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
from XIL.Learning.LearnerStep import Basic

class Elastic_Weight_Regularizer(Basic):

    '''
    Attributes: 
        fit_func: model 
        loss func: loss function
        lr float: learning rate
        optimizer func: optimizer to be used 
        X_test np.array: Testing data
        y_test np.array: Testing data 
        lamb float: Importance of Regularizer
        max_size int: number of saved items

    # Inspired by: https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/ewc.py 
    '''
    def __init__(self, fit_func, loss = nn.CrossEntropyLoss(), lr=0.001, optimizer=torch.optim.Adam,X_test=None, y_test=None, lamb =0.01, max_size=100) -> None:
        super(Elastic_Weight_Regularizer, self).__init__( fit_func, loss, lr, optimizer,X_test, y_test)
        self.old_weights=  {n: p for n, p in self.fit_func.named_parameters() if p.requires_grad}
        self.importance={}
        self.lamb=lamb
        self.task_id= None
        
        self._means={}
        self.exp_counter = 0
        self.max_size=max_size
        #Save last X Values
        self.buffer_x= ReservoirSamplingBuffer(max_size) #TODO Is supposed to be the old dataset 
        self.buffer_y= ReservoirSamplingBuffer(max_size)
        self._precision_matrices = None
        for n, p in deepcopy(self.old_weights).items():
            #TODO is it necessary to update this regulary ?
            self._means[n] = Variable(p.data)
    

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.old_weights).items():
            p.data.zero_()
            precision_matrices[n] = Variable(p.data)

        self.fit_func.eval()
        for input,label in zip(self.buffer_x.buffer, self.buffer_y.buffer):
            self.fit_func.zero_grad()
            try:
                input = Variable(input[np.newaxis,])
            except:
                input = Variable(torch.from_numpy(input[np.newaxis,]))
                label= torch.from_numpy(np.array([label]))
            print(type(label))
            #print(input.shape)
            #TODO USE Ground Truth Here ? See https://github.com/moskomule/ewc.pytorch/issues/4 
            output = self.fit_func(input).view(1, -1)
            #print(label.reshape(-1))
            #print(output)
            #print(F.log_softmax(output, dim=1))
            #label = output.max(1)[1].view(-1)
            loss = F.nll_loss(output, label.long().reshape(-1))
            loss.backward()

            for n, p in self.fit_func.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.buffer_x.buffer)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices
    

    def fit(self,X_train, taskid=None, EPOCHS=1)->None:
        '''
        Attributes: 
            X_train torch.DataLoader: Data to be fit. Labels as INT !
            taskid int: If None LabelTrick is used 
            EPOCHS int: # Runs
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
                # TODO  Is the calculation correct ? 
                #https://github.com/moskomule/ewc.pytorch/blob/master/utils.py

                fin_pen = 0.0
                if self._precision_matrices is not None:
                    for n, p in self.fit_func.named_parameters():
                        #print('p',penalty)
                        penalty = self._precision_matrices[n] * (p - self._means[n]) ** 2
                        #print('p2',penalty.sum())
                        fin_pen += penalty.sum()
                
                #print(fin_pen)
                #print(self.loss_fn(outputs.float(), labels.long()))
                if taskid is None: 
                    l_groups=labels.unique().sort()[0]
                    for lbl_idx, lbl in enumerate(l_groups):
                        labels[labels == lbl] = lbl_idx
                    if 'RRR' in str(type(self.loss_fn)):
                        loss = self.loss_fn(inputs, labels.long())
                    else:
                        loss = self.loss_fn(outputs[:, l_groups.detach().numpy().tolist()].float(), labels.long()) 
                    loss=loss + self.lamb * fin_pen
                else:
                    if 'RRR' in str(type(self.loss_fn)):
                        if taskid>0:
                            loss = self.loss_fn(inputs, labels.long(),taskid) #+ self.lamb * fin_pen
                        else:
                            loss = self.loss_fn(inputs, labels.long(),taskid) + self.lamb * fin_pen
                    else:
                        if taskid>0:
                            loss = self.loss_fn(outputs.float(), labels.long()) #+ self.lamb * fin_pen
                        else:
                            loss = self.loss_fn(outputs.float(), labels.long()) + self.lamb * fin_pen
                #(imp *(self.fit_func.parameters() - self.old_weights).pow(2)).sum() #torch.sum(self.lamb/2 *fisher(self.fit_func.parameters -self.old_weights))
                
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if i % 1000 == 999:
                    last_loss = running_loss / 1000 # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    running_loss = 0.
                if self.task_id != taskid or taskid is None:
                    self.old_weights = {n: p for n, p in self.fit_func.named_parameters() if p.requires_grad}
                                    #TODO IS this correct ? 
                
                    self.buffer_x.update_from_dataset(inputs)
                    self.buffer_y.update_from_dataset(labels)
                    self.buffer_x.buffer=self.buffer_x.buffer
                    self.buffer_y.buffer=self.buffer_y.buffer
                    self.buffer_x.resize('',self.max_size)
                    self.buffer_y.resize('',self.max_size)
                    self._precision_matrices = self._diag_fisher()
                    self.task_id = taskid


               
        return self.fit_func


