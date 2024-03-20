
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import numpy as np 
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from CXIL.Learning.StoragePolicies import ClassBalancedBuffer

from copy import deepcopy
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from CXIL.Learning.StoragePolicies import ReservoirSamplingBuffer
from CXIL.Learning.LearnerStep import Basic
import sys

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
#global x_curr

class RandomExemplarsSelectionStrategy():
    """Select the exemplars at random in the dataset"""

    def make_sorted_indices(
        self, strategy, data
    ):
        indices = list(range(len(data)))
        random.shuffle(indices)
        return indices


class ReplayStrategyLearner():
    """
    Inspired by https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/replay.py.

    TODO check implementation 
    TODO Litreture 
    """

    def __init__(
        self,
        learner,
        num_classes,
        mem_size = 200,
        batch_size= None,
        batch_size_mem = None,
        task_balanced_dataloader= False,
        storage_policy =None,
        interact_only_on_curr=False
    ):
        self.learner=learner#(fit_func=fit_func, loss = loss, lr=lr, optimizer=optimizer)
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        self.train_dataloader=None
   
        self.interact_only_on_curr=interact_only_on_curr

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ClassBalancedBuffer( max_size=self.mem_size,
                adaptive_size=False, total_num_classes=num_classes
            )

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups  # a Dict<task_id, Dataset>

    def before_training_exp(
        self
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """

        #print('Seen Classes',self.storage_policy.seen_classes )
        if len(self.storage_policy.seen_classes) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            #print('No Classes have been seen')
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size =self.batch_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = self.batch_size

        y= None
        #print('Biffer Groups', self.storage_policy.buffer_groups)
        combined=None
        #print(self.storage_policy.buffer_groups.keys())
        for k in self.storage_policy.buffer_groups.keys(): 
            #print('k',k)
            if combined is None:
                #print('Combined is none ')
                #print(self.storage_policy.buffer_groups[k].buffer)
                combined = self.storage_policy.buffer_groups[k].buffer
                #print('Combined one', combined.shape)
                y=np.repeat(k, len(self.storage_policy.buffer_groups[k].buffer))
            else: 
                #print('Add Other')
                combined = np.concatenate([combined,self.storage_policy.buffer_groups[k].buffer ])
                #print('Combined one', combined.shape)
                y= np.concatenate([y,np.repeat(k, len(self.storage_policy.buffer_groups[k].buffer)) ])

        training_data=TensorDataset(torch.tensor(combined).float(),torch.tensor(y).long())
        #print('End Tensor')
        self.train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        #print('DataLoader')



    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy.update(strategy, **kwargs)
    
    
    def fit(self,data,input_gradients=None, annotation_matrix=None, taskid=None):
        
            #print(globals().keys())
            #print(globals()['__builtins__'])
            #print('x_curr' in globals())

        #start = time.time()
        #start_1 = time.time()
        self.before_training_exp()
        #end_1 = time.time()
        #print('Replay before', end_1-start_1)
        #print('Train DAta Loader', self.train_dataloader)
        if self.train_dataloader is not None:
            #if input_gradients is None:
            
            self.learner.fit_func= self.learner.fit(self.train_dataloader,taskid=taskid)
                #self.after_training_exp(data) 
            #else: 
            #    self.learner.fit_func= self.learner.fit(self.train_dataloader,input_gradients,annotation_matrix,taskid=taskid)
            #    import sys 
            #    sys.exit(1)
                #self.after_training_exp(data)
        #start2 = time.time()
        #TODO For CAIPI Only Add original Data ? 
        self.after_training_exp(data)
        if self.interact_only_on_curr: 
            #global x_curr
            #print(data.shape)
            for a,_ in data:
                globals()['__builtins__']['x_curr']=a
        #end2=time.time()
        #print('Replay AFter', end2-start2)
        #end=time.time()
        #print('Replay Strategy', end-start)
        return self.learner.fit_func