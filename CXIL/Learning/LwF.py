# https://github.com/MasLiang/Learning-without-Forgetting-using-Pytorch/blob/main/main.py
# Compare With :  https://github.com/G-U-N/PyCIL/blob/master/models/lwf.py
#TODO RECHECK Behavior with Paper

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
import numpy as np
from collections import defaultdict
#alpha = 0.01

#def kaiming_normal_init(m):#
#	if isinstance(m, nn.Conv2d):
#		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')#
#	elif isinstance(m, nn.Linear):
#		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class LwF():
    def __init__(self, fit_func, loss = nn.CrossEntropyLoss(), lr=0.001, optimizer= optim.SGD,X_test=None, y_test=None, alpha=[0,0.5,1.33,2.25,3.2],temperature=1, class_incremental= False) -> None:
        self.fit_func=fit_func
        #self.fit_func11=copy.deepcopy(fit_func) #TODO Make a deep copy
        self.optimizer = optimizer(fit_func.parameters(), lr=lr) 
        self.loss_fn   = loss
        self.test=False
        self.y=[]
        self.alpha = alpha
        self.temperature = temperature
        self.class_incremental=class_incremental
        self.prev_model = None #dict()
        if class_incremental:
            self.prev_model=dict()
        self.prev_task= 0
        self.task = 0 
        self.n_exp_so_far=1
        self.device='cpu'
        # Experimental setting
        #self.known_classes = []
        #self.known_classes=fit_func.classifier.outfeatures
    def _distillation_loss(self, out, prev_out):
        """Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        #TODO ELIMINATED AU 
        log_p = torch.log_softmax(out / self.temperature, dim=1)
        q = torch.softmax(prev_out / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return res


    def lwf_penalty(self,inputs):
        loss= 0 
        if self.prev_model is not None : 
            y_n = F.softmax(self.fit_func(inputs))
            #print('Current Model', self.fit_func)
            if self.class_incremental:
                for task_id in self.prev_model.keys():
                    if task_id == self.task:
                        break
                    with torch.no_grad():
                        #print(f'Previous Model {task_id}', self.prev_model[task_id])
                        self.prev_model[task_id].eval()
                        y_o=F.softmax(self.prev_model[task_id](inputs))
                        loss+= self._distillation_loss( y_n,y_o)
                return loss
            #        break
            with torch.no_grad():
                #print(f'Previous Model {task_id}', self.prev_model[task_id])
                self.prev_model.eval()
                y_o=F.softmax(self.prev_model(inputs))
            loss+= self._distillation_loss( y_n,y_o)
        return loss
    
    def fit(self,Xtrain,taskid=None,EPOCHS=10):
        self.fit_func.train()
        loss_running=0.
        reg_running=0. 
        BCE_runing = 0. 
        for e in range (0, EPOCHS):
            loss_running=0.
            reg_running=0. 
            BCE_runing = 0. 
            for batch_idx, (inputs, targets) in enumerate(Xtrain):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.fit_func(inputs)
                if taskid is None: 
                    l_groups=targets.unique().sort()[0]
                    for lbl_idx, lbl in enumerate(l_groups):
                        targets[targets == lbl] = lbl_idx
                    loss1 = self.loss_fn(outputs[:, l_groups.detach().numpy().tolist()].float(), targets.long()) 
                else:
                    if taskid>-1:
                        #TODO
                        loss1 = self.loss_fn(outputs,targets)
                #Regularizer Scaling term frim: https://github.com/ContinualAI/continual-learning-baselines/blob/05f36a72a39577f7876ccc4a80d8ff494f20d922/experiments/split_mnist/lwf.py#L6
                loss2=self.lwf_penalty(inputs) * (1- 1/self.n_exp_so_far)
                
                alpha = (  self.alpha[self.task]
                if isinstance(self.alpha, (list, tuple))
                else self.alpha
                 )
                loss= loss1+alpha*loss2
                loss.backward()
                self.optimizer.step()
                self.n_exp_so_far+=1
                loss_running+=loss
                reg_running+=loss2
                BCE_runing+=loss1
                if not self.class_incremental:
                    self.prev_model=copy.deepcopy(self.fit_func).requires_grad_(False)
            #print(f'{e} - {loss_running} - {reg_running} - {BCE_runing}')
        
        if self.class_incremental:
            self.prev_model[self.task]=copy.deepcopy(self.fit_func).requires_grad_(False)
        self.task += 1
        return self.fit_func
    