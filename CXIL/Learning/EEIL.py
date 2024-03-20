import torch
import warnings
from copy import deepcopy
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from CXIL.Learning.Datasets import MemoryDataset
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import StepLR
class EEIL():
    """Class implementing the End-to-end Incremental Learning (EEIL) approach described in
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf
    Original code available at https://github.com/fmcp/EndToEndIncrementalLearning
    Helpful code from https://github.com/arthurdouillard/incremental_learning.pytorch
    """

    def __init__(self, fit_func, device='cpu', nepochs=40, lr=0.1,optimizer= optim.SGD, lr_min=1e-6, lr_factor=10, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=MemoryDataset(), lamb=1.0, T=2, lr_finetuning_factor=0.1,
                 nepochs_finetuning=30, noise_grad=True,remove_existing_head=False):
        
        self.fit_func=fit_func
        self.model_old = None
        self.model_array=[]
        self.lamb = lamb
        self.T = T
        self.lr_finetuning_factor = lr_finetuning_factor
        self.nepochs_finetuning = nepochs_finetuning
        self.noise_grad = noise_grad
        self.exemplars_dataset= exemplars_dataset
        self._train_epoch = 0
        self._finetuning_balanced = None
        self.device=device
        self.clipgrad=clipgrad
        self.fix_bn=fix_bn
        self.nepochs=nepochs
        self.lr= lr
        self.wd=wd
        self.momentum=momentum
        self.optimizer=optimizer(self.fit_func.parameters(), lr=self.lr, weight_decay=wd, momentum=self.momentum)
        self.optimizer_type=optimizer
        self.scheduler=StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.task_cls = []
        self.task_cls_prev = []
        head_var = list(fit_func.children())[-1]
        last_layer = list(fit_func.children())[-1]
        # What is the first Head 

        
        self.heads = nn.ModuleList()

        #if remove_existing_head:
        #    if type(last_layer) == nn.Sequential:
        #        self.out_size = last_layer[-1].in_features
        #        # strips off last linear layer of classifier
        #        del last_layer[-1]
        #    elif type(last_layer) == nn.Linear:
        #        self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                #setattr(self.fit_func, head_var, nn.Sequential())
        #else:
        #    self.out_size = last_layer.out_features


    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def _train_unbalanced(self, trn_loader,t):
        """Unbalanced training"""
        self._finetuning_balanced = False
        self._train_epoch = 0
        loader = self._get_train_loader(trn_loader, False)
        for e in range(self.nepochs):
            self.train_epoch(loader,t,e)
        #super().train_loop(t, loader, val_loader)
        return loader

    def _train_balanced(self, trn_loader,t):
        """Balanced finetuning"""
        self._finetuning_balanced = True
        self._train_epoch = 0
        #orig_lr = self.lr
        #self.lr *= self.lr_finetuning_factor
        #orig_nepochs = self.nepochs
        #self.nepochs = self.nepochs_finetuning
        #print('1',trn_loader)
        loader = self._get_train_loader(trn_loader, True)
        #super().train_loop(t, loader, val_loader)
        self.optimizer=self.optimizer_type(self.fit_func.parameters(), lr=0.01,weight_decay=self.wd,momentum=self.momentum)
        self.scheduler=StepLR(self.optimizer, step_size=10, gamma=0.1)
        for e in range(self.nepochs_finetuning):
            self.train_epoch(loader,t,e)
        #self.lr = orig_lr
        #self.nepochs = orig_nepochs

    def _get_train_loader(self, trn_loader, balanced=False):
        """Modify loader to be balanced or unbalanced"""
        print(len(self.exemplars_dataset))
        exemplars_ds = self.exemplars_dataset
        trn_dataset = trn_loader.dataset
        if balanced:
            indices = torch.randperm(len(trn_dataset))
            trn_dataset = torch.utils.data.Subset(trn_dataset, indices[:len(exemplars_ds)])
        ds = exemplars_ds + trn_dataset
        return DataLoader(ds, batch_size=trn_loader.batch_size,
                              shuffle=True,
                              num_workers=trn_loader.num_workers,
                              pin_memory=trn_loader.pin_memory)

    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        """Add noise to the gradients"""
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(p.grad.shape, device=p.grad.device) * variance)

    def fit(self, trn_loader,t=None):
        """Contains the epochs loop"""
        #print(t)
        self.optimizer=self.optimizer_type(self.fit_func.parameters(), lr=self.lr,weight_decay=self.wd, momentum=self.momentum)
        self.scheduler=StepLR(self.optimizer, step_size=10, gamma=0.1)
        #if t not in 
        #self.task_cls=[]
        if len(self.task_cls) == 0:  # First task is simple training
            #print('a')
            #self.pre_train_process(trn_loader,t)
            for e in range(self.nepochs):
                self.train_epoch(trn_loader,t,e)
            #self.post_train_process(trn_loader,t)
            #super().train_loop(t, trn_loader, val_loader)
            loader = trn_loader
        else:
            #print('b')
            # Page 4: "4. Incremental Learning" -- Only modification is that instead of preparing examplars before
            # training, we do it online using the stored old model.

            # Training process (new + old) - unbalanced training
            loader = self._train_unbalanced( trn_loader,t)
            # Balanced fine-tunning (new + old)
            self._train_balanced(trn_loader,t)
        self.task_cls=self.task_cls_prev
        self.post_train_process()
        # After task training： update exemplars
        print('TASK CLS',self.task_cls)
        print('TASK CLS',self.task_cls_prev)
        self.exemplars_dataset.collect_exemplars(self.task_cls, loader)
        return self.fit_func

    def post_train_process(self):
        """Runs after training all the epochs of the task (after the train session)"""
        # Save old model to extract features late 
        self.model_old = deepcopy(self.fit_func)
        self.model_old.eval()
        for param in self.model_old.parameters():
            param.requires_grad = False
        self.model_array.append(deepcopy(self.model_old))
        print('FITFUC',id(self.fit_func))
        print('OLDFUC',id(self.model_old))

    def train_epoch(self, trn_loader,t,e):
        """Runs a single epoch"""
        loss_full=0
        self.fit_func.train()
        if self.fix_bn and t > 0:
            self.fit_func.freeze_bn()
        for images, targets in trn_loader:
            images = images.to(self.device)
            # Forward old model
            outputs_old = None
            if t > 0:
                with torch.no_grad():
                    #soutputs_old = self.model_old(images)
                    outputs_old=[]
                    for old_model in self.model_array:
                        outputs_old.append(old_model(images))

            # TODO THIS IS A FIX  Forward current model
            outputs= self.fit_func(images)
            

            #print('images shape', images)
            #print('outputs shape', outputs)
            loss = self.criterion(e, outputs, targets.to(self.device), outputs_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # Page 8: "We apply L2-regularization and random noise [21] (with parameters eta = 0.3, gamma = 0.55)
            # on the gradients to minimize overfitting"
            # https://github.com/fmcp/EndToEndIncrementalLearning/blob/master/cnn_train_dag_exemplars.m#L367
            torch.nn.utils.clip_grad_norm_(self.fit_func.parameters(), self.clipgrad)
            if self.noise_grad:
                self._noise_grad(self.fit_func.parameters(), self._train_epoch)
            self.optimizer.step()
            #print(f'LOSS {self.task_cls} {np.unique(targets)}')
            self.task_cls_prev=np.append(self.task_cls_prev,np.unique(targets))
            self.task_cls_prev= np.unique(self.task_cls_prev)
            loss_full+=loss
        #print(loss_full)
        self._train_epoch += 1
        self.scheduler.step()
        

    def criterion(self, t, outputs, targets, outputs_old=None):
        """
        Returns the loss value        
        """
        if outputs_old is not None:
            loss = torch.nn.functional.cross_entropy(torch.cat((outputs,*outputs_old), dim=1), targets)
        else: 
            loss = torch.nn.functional.cross_entropy(outputs, targets) 
        if t > 0 and outputs_old is not None:
            # take into account current head when doing balanced finetuning
            last_head_idx = t if self._finetuning_balanced else (t - 1)
            #print(last_head_idx)
            for i in range(0, len(outputs_old)): #range(last_head_idx):
                #print(i)
                #USed zo have an i 
                loss += self.lamb * F.binary_cross_entropy(F.softmax(outputs / self.T, dim=1),
                                                           F.softmax(outputs_old[i] / self.T, dim=1))
        return loss
