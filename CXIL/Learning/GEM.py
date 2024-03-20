from XIL.Learning.LearnerStep import Basic
import torch
import numpy as np
from torch import nn
import tqdm
from XIL.Learning.StoragePolicies import ReservoirSamplingBuffer
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
import quadprog

class GEM(Basic):
    """
    @inproceedings{GradientEpisodicMemory,
        title={Gradient Episodic Memory for Continual Learning},
        author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
        booktitle={NIPS},
        year={2017},
        url={https://arxiv.org/abs/1706.08840}

    #https://github.com/ruinanzhang/Rotated_MNIST_Continual_Learning/blob/master/Rotated_MNIST_GEM.ipynb
    }
    """

    def __init__(self, fit_func,num_classes, loss = nn.CrossEntropyLoss(), lr=0.001,batch_size=32, optimizer=torch.optim.Adam,X_test=None, y_test=None, max_size=1000, storage_policy=None) -> None:
        super(GEM, self).__init__(fit_func, loss, lr,optimizer, X_test,y_test)
        self.params = {n: p for n, p in self.fit_func.named_parameters() if p.requires_grad}  # For convenience
        self.task_grads = {}
        self.buffer_x= ReservoirSamplingBuffer(max_size) #TODO Is supposed to be the old dataset 
        self.buffer_y= ReservoirSamplingBuffer(max_size)    
        self.task_mem_cache = {}
        self.train_dataloader=None
        self.task_count=0
        self.task_id=None
        self.batch_size=batch_size
        self.batch_size_mem=None
        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == max_size
        else:  # Default
            self.storage_policy = ClassBalancedBuffer( max_size=max_size,
                adaptive_size=False, total_num_classes=num_classes
            )

    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def project2cone2(self, gradient, memories, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector

            Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
        """
        memories_np = memories.cpu().contiguous().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        #print(memories_np.shape, gradient_np.shape)
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        P = P + G * 0.001
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        new_grad = torch.Tensor(x).view(-1)
        if self.gpu:
            new_grad = new_grad.cuda()
        return new_grad

    #def learn_batch(self, train_loader, val_loader=None):
        #TODO GET DATA FROM BUFFER

        # Update model as normal
        #super(GEM, self).learn_batch(train_loader, val_loader)

        # Cache the data for faster processing
        #for t, mem in self.task_memory.items():
            # Concatenate all data in each task
        #    mem_loader = torch.utils.data.DataLoader(mem,
        #                                             batch_size=len(mem),
        #                                             shuffle=False,
        #                                             num_workers=2)
        #    assert len(mem_loader)==1,'The length of mem_loader should be 1'
        #    for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
        #        if self.gpu:
        #            mem_input = mem_input.cuda()
        #            mem_target = mem_target.cuda()
        #    self.task_mem_cache[t] = {'data':mem_input,'target':mem_target,'task':mem_task}
        #pass

    def fit(self,data,task_id= None,input_gradients=None, annotation_matrix=None, EPOCHS=1)->None:
        '''
        Attributes: 
            X_train torch.DataLoader: Data to be fit. Labels as INT !
        '''
        #if task_id is not None: 
        #    self.task_id=task_id
        #else: 
        #    self.task_id=0
        self.fit_func.train()
        #running_loss = 0.
        #loss_list     = np.zeros((EPOCHS,))
        
        for epoch in tqdm.trange(EPOCHS):
            self.before_training_exp()
            self.after_training_exp(data)

            for i, data in enumerate(data):
                inputs, labels = data

            
            if self.train_dataloader is not None:
                if input_gradients is None:
                    print('A')
                    self.fit_func= self.update_model(inputs, labels,task_id) #self.learner.fit(self.train_dataloader)
                    #self.after_training_exp(data) 
                else: 
                    print('b')
                    #TODO Update annotation matrix
                    self.fit_func= self.update_model(inputs, labels,task_id) #self.learner.fit(self.train_dataloader,input_gradients,annotation_matrix)
                    #self.after_training_exp(data)
        #print('DATA ', data )

        #print('RETURN', self.fit_func)

        return self.fit_func
    def after_training_exp(self, strategy, **kwargs):
        self.storage_policy.update(strategy, **kwargs)
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
        print('y', np.unique(y))
        training_data=TensorDataset(torch.tensor(combined).float(),torch.tensor(y).long())
        #print('End Tensor')
        self.train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        #print('DataLoader')

    def update_model(self, inputs, targets, taskid=None):
        print('taskid ', taskid)

        # compute gradient on previous tasks
        if self.task_count is not None:
            if self.task_count > 0:
                for inputs_prev,targets_prev in self.train_dataloader:
                    #task_memory.items():
                    self.fit_func.zero_grad()
                    # feed the data from memory and collect the gradients
                    mem_out = self.fit_func(inputs_prev)#self.forward(self.task_mem_cache[t]['data'])
                    mem_loss = self.loss_fn(mem_out,targets_prev )#, tasks_prev)#self.task_mem_cache[t]['target'], self.task_mem_cache[t]['task'])
                    mem_loss.backward()
                    self.task_grads[self.task_count] = self.grad_to_vector() # TODO THIS USED TO BE T 

        # now compute the grad on the current minibatch
        out = self.fit_func(inputs)
        #TODO MAke the Label Trick
        loss= self.loss_fn(out, targets)

        #if taskid is not None:
        #    l_groups=targets.unique().sort()[0]
        #    for lbl_idx, lbl in enumerate(l_groups):
        #        targets[targets == lbl] = lbl_idx
        #        loss = self.loss_fn(out[:, l_groups.detach().numpy().tolist()].float(), targets.long()) 
        #else: 
        #    loss = self.loss_fn(out.float(), targets.long())
        #loss = self.criterion(out, targets )#, tasks)
        self.optimizer.zero_grad()
        loss.backward()

        # check if gradient violates constraints
        if self.task_count > 0:
            current_grad_vec = self.grad_to_vector()
            mem_grad_vec = torch.stack(list(self.task_grads.values()))
            dotp = current_grad_vec * mem_grad_vec
            dotp = dotp.sum(dim=1)
            if (dotp < 0).sum() != 0:
                new_grad = self.project2cone2(current_grad_vec, mem_grad_vec)
                # copy gradients back
                self.vector_to_grad(new_grad)

        self.optimizer.step()
        self.task_count +=1
        return self.fit_func #loss.detach(), out