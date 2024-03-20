from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time

from CXIL.Learning.LearnerStep import Basic
from torch.utils.data import DataLoader
#end of imports
#
'''iNSPIRED BY
https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses/blob/master/MAS_to_be_published/MAS_utils/MAS_based_Training.py
Other Implementations:
https://github.com/Mattdl/CLsurvey/blob/master/src/methods/MAS/train_MAS.py'''


class Weight_Regularized_SGD(optim.SGD):
    r"""Implements SGD training with importance params regulization. IT inherents stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,reg_lambda=1.0):
        
        super(Weight_Regularized_SGD, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        self.reg_lambda=reg_lambda

    def __setstate__(self, state):
        super(Weight_Regularized_SGD, self).__setstate__(state)
       
        
    def step(self, reg_params,closure=None):
        """Performs a single optimization step.
        https://github.com/wannabeOG/MAS-PyTorch/blob/master/optimizer_lib.py#L19
        Arguments:
            reg_params: omega of all the params
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
       

        loss = None
        if closure is not None:
            loss = closure()
        
        #reg_lambda=reg_params['lambda']#.get('lambda')
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
           
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
               
                #MAS PART CODE GOES HERE
                #if this param has an omega to use for regulization
                ###print(reg_params)
                #if p in reg_params[0]:
                #    pass
                ##print(p)

                ##print('reg_param',reg_params)
                if p in  reg_params:
                    
                    reg_param=reg_params.get(p)
                    #get omega for this parameter
                    omega=reg_param.get('omega')
                    ##print('omega ', omega)
                    #initial value when the training start
                    init_val=reg_param.get('init_val')
                    ##print('init_val', init_val)
                    
                    curr_weight_val=p.data

                    ##print('curr weight', curr_weight_val)
                    #move the tensors to cuda
                    #init_val=init_val#.cuda()
                    #omega=omega#.cuda()
                    
                    #get the difference
                    weight_dif=curr_weight_val.add(-1,init_val)
                    ##print('weight_diff', weight_dif)
                    #compute the MAS penalty
                    regulizer=weight_dif.mul(2*self.reg_lambda*omega)
                    ##print('regularizer ', regulizer)
                    del weight_dif
                    del curr_weight_val
                    del omega
                    del init_val
                    #add the MAS regulizer to the gradient
                    d_p.add_(regulizer)
                    del regulizer
                #MAS PARAT CODE ENDS
                if weight_decay != 0:
                   
                    d_p.add_(weight_decay,p.data.sign())                   
 
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
               
                
                p.data.add_(-group['lr'], d_p)
        #print('LOSS ', loss)
        return loss#ELASTIC SGD

class MAS_Omega_update(optim.SGD):
    """
    Update the paramerter importance using the gradient of the function output norm. To be used at deployment time.
    reg_params:parameters omega to be updated
    batch_index,batch_size:used to keep a running average over the seen samples
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        
        super(MAS_Omega_update, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        
    def __setstate__(self, state):
        super(MAS_Omega_update, self).__setstate__(state)
       

    def step(self, reg_params,batch_index,batch_size,closure=None):
        """
        Performs a single parameters importance update setp
        """

        ###print('************************DOING A STEP************************')
 
        loss = None
        if closure is not None:
            loss = closure()
             
        for group in self.param_groups:
   
            #if the parameter has an omega to be updated
            for p in group['params']:
          
                ###print('************************ONE PARAM************************')
                
                if p.grad is None:
                    continue
               
                if p in reg_params:
                    d_p = p.grad.data
                  
                    
                    #HERE MAS IMPOERANCE UPDATE GOES
                    #get the gradient
                    unreg_dp = p.grad.data.clone()
                    reg_param=reg_params.get(p)
                    
                    zero=torch.FloatTensor(p.data.size()).zero_()
                    #get parameter omega
                    omega=reg_param.get('omega')
                    omega=omega#.cuda()
    
                    
                    #sum up the magnitude of the gradient
                    prev_size=batch_index*batch_size
                    curr_size=(batch_index+1)*batch_size
                    omega=omega.mul(prev_size)
                    
                    omega=omega.add(unreg_dp.abs_())
                    #update omega value
                    omega=omega.div(curr_size)
                    if omega.equal(zero):#.cuda()):
                        print('omega after zero')

                    reg_param['omega']=omega
                   
                    reg_params[p]=reg_param
                    #HERE MAS IMPOERANCE UPDATE ENDS
        return loss#HAS NOTHING TO DO

  
class MAS_Omega_Vector_Grad_update(optim.SGD):
    """
    Update the paramerter importance using the gradient of the function output. To be used at deployment time.
    reg_params:parameters omega to be updated
    batch_index,batch_size:used to keep a running average over the seen samples
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        
        super(MAS_Omega_Vector_Grad_update, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        
    def __setstate__(self, state):
        super(MAS_Omega_Vector_Grad_update, self).__setstate__(state)
       

    def step(self, reg_params,batch_index,batch_size,intermediate=False,closure=None):
        """
        Performs a single parameters importance update setp
        """

        ###print('************************DOING A STEP************************')

        loss = None
        if closure is not None:
            loss = closure()
        index=0
     
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
    
            for p in group['params']:
          
                ###print('************************ONE PARAM************************')
                
                if p.grad is None:
                    continue
                
                if p in reg_params:
                    d_p = p.grad.data
                    unreg_dp = p.grad.data.clone()
                    #HERE MAS CODE GOES
                    reg_param=reg_params.get(p)
                    
                    zero=torch.FloatTensor(p.data.size()).zero_()
                    omega=reg_param.get('omega')
                    #print('omega', omega)
                    omega=omega#.cuda()
    
                    
                    #get the magnitude of the gradient
                    if intermediate:
                        if 'w' in reg_param.keys():
                            w=reg_param.get('w')
                        else:
                            w=torch.FloatTensor(p.data.size()).zero_()
                        w=w#.cuda()
                        w=w.add(unreg_dp.abs_())
                        reg_param['w']=w
                    else:
                       
                       #sum the magnitude of the gradients
                        ##print('omega from opt', omega)
                        w=reg_param.get('w')
                        prev_size=batch_index*batch_size
                        curr_size=(batch_index+1)*batch_size
                        omega=omega.mul(prev_size)
                        omega=omega.add(w)
                        omega=omega.div(curr_size)
                        reg_param['w']=zero#.cuda()
                        
                        if omega.equal(zero):#.cuda()):
                            print('omega after zero')

                    reg_param['omega']=omega
                    #pdb.set_trace()
                    reg_params[p]=reg_param
                index+=1
        return loss

#importance_dictionary: contains all the information needed for computing the w and omega

  
class MAS(Basic):
    def __init__(self, fit_func, loss = nn.CrossEntropyLoss(), lr=0.001, optimizer=torch.optim.Adam,X_test=None, y_test=None, lamb=1.0, max_size=100,weight_decay=1e-5) -> None:
        super(MAS, self).__init__( fit_func, loss, lr, optimizer,X_test, y_test)
        self.lr=lr
        self.lamb =lamb        
        self.fit_func.reg_params=self.initialize_reg_params()
        #self.fit_func.reg_params['lambda']=lamb
       
        #self.mas_switch=False

        #TODOS PARAMETER OR REG PARAMETER ? 
        self.optimizer=Weight_Regularized_SGD(self.fit_func.parameters(),lr=0.0008, momentum=0, dampening=0, weight_decay=weight_decay, nesterov=False,reg_lambda=self.lamb)#Weight_Regularized_SGD(self.fit_func.reg_params)
        #self.optimizer_ft = MAS_Omega_Vector_Grad_update(self.fit_func.parameters(), lr=0.0001, momentum=0.9) 
        self.optimizer_ft=MAS_Omega_update(self.fit_func.parameters(), lr=0.0001, momentum=0.9)
        self.task=-1
    
  
    def fit(self,dataset,taskid=None,EPOCHS=1):# EPOCHS IS SUPPOSED TO BE !==
        """
        Train a given model using MAS optimizer. The only unique thing is that it passes the importnace params to the optimizer
        TODO THIS IS SUPPOSED TO BE YOUR USUAL TRAIN FUNCTION
        STEPS: 
        1. Train Model joint training on all base tasks
        2. Update Omega
        3. Update Theta

        """
        self.task +=1
        
        ###print('dictoinary length'+str(len(dset_loaders)))
        #reg_params=model.reg_params
        #since = time.time()

        best_model = self.fit_func
        best_acc = 0.0
        start_epoch= 0
        #if self.mas_switch:
        #    '''Accumulate Weights Function'''
        #    self.fit_func.train(False)
        #    self.fit_func.reg_params=self.initialize_store_reg_params()  
            #   
             #MAS_Omega_Vector_Grad_update(self.fit_func.parameters()) #
        #    self.fit_func=self.compute_importance_l2(self.fit_func, self.optimizer_ft,dataset) #self.compute_importance_gradient_vector(self.fit_func, optimizer_ft,dataset)#
        #    self.fit_func.reg_params=self.accumelate_reg_params()
        #self.fit_func.reg_params['lambda']=self.reg_lambda
        #print(sanitycheck(self.fit_func))
        #    '''-----------------------------'''

        
        self.optimizer=Weight_Regularized_SGD(self.fit_func.reg_params,lr=0.0008)
        
        for epoch in range(start_epoch, EPOCHS):
                
            #scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.1)
            self.fit_func.train(True)  
            ind=-1
            for data in dataset:
                ind+=1
                inputs, labels = data
                #print(inputs.shape)
                #inputs=inputs.squeeze()
                inputs, labels = Variable(inputs), Variable(labels)
                self.optimizer.zero_grad()
                #TODO IS THIS RIGHT?
                self.fit_func.zero_grad()
                # forward
                #print(inputs.shape)
                outputs = self.fit_func(inputs)
                _, preds = torch.max(outputs.data, 1)
                # THIS IS THE LABEL TRICK 

                if taskid is None: 
                    labeltrick = True
                    ##print('TASKID Was set to None')
                    l_groups=labels.unique().sort()[0]
                    ##print(l_groups)
                    for lbl_idx, lbl in enumerate(l_groups):
                        labels[labels == lbl] = lbl_idx
                    if 'RRR' in str(type(self.loss_fn)):
                        loss = self.loss_fn(inputs, labels.long())
                    else:
                        loss = self.loss_fn(outputs[:, l_groups.detach().numpy().tolist()].float(), labels.long()) 
                else:
                    labeltrick=False
                    if 'RRR' in str(type(self.loss_fn)):
                        loss = self.loss_fn(inputs, labels.long(),taskid)
                    else:                
                        loss = self.loss_fn(outputs, labels.long())
                #print('LOSS 2 ' , str(loss.item()))
                if str(loss.item())== 'nan': 
                    print('LOSS IS FOR THE FIRST TIME NONE')
                    import sys
                    sys.exit(1)

                loss.backward()
                #TODO IS THIS SUPPOSE TO BE PARAMETERS ? 
                self.optimizer.step(self.fit_func.reg_params)
                #self.compute_importance_gradient_vector(self.fit_func, self.optimizer_ft,dataset)#
                #else:
                #    #TODO USE OMEHA ACC
                #    self.omega_optimizer.step(self.fit_func.reg_params,self.intermetiate, ind, len(data))
            #scheduler.step()
        #TODO MAKE USE OF SWITCH ?  - Beginning or END ? 
      
        #self.fit_func.train(False)
        self.fit_func.reg_params=self.initialize_store_reg_params()  
        #optimizer_ft = MAS_Omega_update(self.fit_func.parameters(), lr=0.0001, momentum=0.9) #MAS_Omega_Vector_Grad_update(self.fit_func.parameters()) #

        self.fit_func= self.compute_importance_l2(self.fit_func, self.optimizer_ft,dataset, labeltrick) 
        #self.fit_func=self.compute_importance_gradient_vector(self.fit_func, self.optimizer_ft,dataset)#
        self.fit_func.reg_params=self.accumelate_reg_params()

        print(sanitycheck(self.fit_func))

        #self.mas_switch=True
        return self.fit_func
    '''
    def accumulate_MAS_weights(self,data,reg_sets,model_ft,batch_size,norm='L2'):
        """
        ALREADY in FIT 

        """

        #store the previous omega, set values to zero
        reg_params=self.initialize_store_reg_params(model_ft)
        model_ft.reg_params=reg_params
        #define the importance weight optimizer. Actually it is only one step. It can be integrated at the end of the first task training
        optimizer_ft = MAS_Omega_update(model_ft.parameters(), lr=0.0001, momentum=0.9)
    
        #if norm=='L2':
        #    #print('********************objective with L2 norm***************')
        model_ft =self.compute_importance_l2(model_ft, optimizer_ft, data)
       ##else:
        #    model_ft =compute_importance(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
        #accumelate the new importance params  with the prviously stored ones (previous omega)
        reg_params=self.accumelate_reg_params(model_ft)
        model_ft.reg_params=reg_params 
        return model_ft

    '''

    def compute_importance_l2(self,model, optimizer, dset_loaders, label_trick = False):
        """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
        Uses the L2norm of the function output. This is what we MAS uses as default
        """
        ###print('dictoinary length'+str(len(dset_loaders)))
        #reg_params=model.reg_params

        def exp_lr_scheduler(optimizer, epoch, init_lr=0.0004, lr_decay_epoch=54):
            """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
   
            lr = init_lr * (0.1**(epoch // lr_decay_epoch))

            if epoch % lr_decay_epoch == 0:
                print('LR is set to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        #optimizer = exp_lr_scheduler(optimizer, 1,1)
        #since = time.time()

        #best_model = model
        #best_acc = 0.0
            
        epoch=1
        #it does nothing here
        #optimizer = exp_lr_scheduler(optimizer, epoch,1)
        #optimizer = lr_scheduler(optimizer, epoch,1,init_lr=0.0004, lr_decay_epoch=54)
        model.eval()  # Set model to training mode so we get the gradient


        running_loss = 0.0
        running_corrects = 0
    
        # Iterate over data.
        index=0
        #for dset_loader in dset_loaders:
        for data in dset_loaders:
            # get the inputs
            inputs, labels = data
            if inputs.size(1)==1 and len(inputs.size())==3:
                    
                #for mnist, there is no channel 
                #and  to avoid problems we remove that additional dimension generated by pytorch transformation
                inputs=inputs.view(inputs.size(0),inputs.size(2))            
            # wrap them in Variable
            #if use_gpu:
            #    inputs, labels = Variable(inputs.cuda()), \
            #    Variable(labels.cuda())
            #else:
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
                
            # forward
            outputs = torch.nn.functional.softmax(model(inputs))
            #_, preds = torch.max(outputs.data, 1)


            #compute the L2 norm of output 
            Target_zeros=torch.zeros(outputs.size())
            Target_zeros=Target_zeros
            #.cuda()
            Target_zeros=Variable(Target_zeros)
            #note no avereging is happening here
            loss = torch.nn.MSELoss(size_average=False)
            if label_trick: 
                    ##print('TASKID Was set to None')
                    l_groups=labels.unique().sort()[0]
                    ##print(l_groups)
                    #for lbl_idx, lbl in enumerate(l_groups):
                    #    labels[labels == lbl] = lbl_idx
                    #print(outputs[:, l_groups.detach().numpy().tolist()])
                    targets = loss(outputs[:, l_groups.detach().numpy().tolist()].float(), Target_zeros[:, l_groups.detach().numpy().tolist()]) 
            else:
                    targets = loss(outputs,Target_zeros)
            #targets = loss(outputs,Target_zeros) 
            #compute the gradients
            targets.backward()

            #update the parameters importance
            optimizer.step(model.reg_params,index,labels.size(0))

            #nessecary index to keep the running average
            index+=1
    
        return model

    '''
    def compute_importance(model, optimizer, lr_scheduler,dset_loaders,use_gpu):
        """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
        Uses the L1norm of the function output
        """
        ##print('dictoinary length'+str(len(dset_loaders)))
    
        since = time.time()

        best_model = model
        best_acc = 0.0
        
        #pdb.set_trace()
        

            
        epoch=1
        #it does nothing here, can be removed
        optimizer = lr_scheduler(optimizer, epoch,1)
        model.eval()  # Set model to training mode so we get the gradient


        running_loss = 0.0
        running_corrects = 0
    
        # Iterate over data.
        index=0
        for dset_loader in dset_loaders:
            #pdb.set_trace()
            for data in dset_loader:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                #if use_gpu:
                #    inputs, labels = Variable(inputs.cuda()), \
                #    Variable(labels.cuda())
                #else:
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameters gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
        

            #compute the L1 norm of the function output
            
                Target_zeros=torch.zeros(outputs.size())
                Target_zeros=Target_zeros#.cuda()
                Target_zeros=Variable(Target_zeros,requires_grad=False)
        
                loss = torch.nn.L1Loss(size_average=False)

                targets = loss(outputs,Target_zeros) 
                #compute gradients
                targets.backward()
            
                
                ##print('batch number ',index)
                #update parameters importance
                optimizer.step(model.reg_params,True,index,labels.size(0))
                #nessecary index to keep the running average
                index+=1
    
        return model
    '''

    def compute_importance_gradient_vector(self,model, optimizer, dset_loaders, label_trick = True):
        """Mimic the depoloyment setup where the model is applied on some samples and those are used to update the importance params
        Uses the gradient of the function output
        TODO MAYBE THIS IS NECESSARY TO CHECK AGAINST L2
        """
        ##print('dictoinary length'+str(len(dset_loaders)))
        #reg_params=model.reg_params
        since = time.time()

        best_model = model
        best_acc = 0.0
        
        
        
            
        epoch=1
        #optimizer = lr_scheduler(optimizer, epoch,1)
        model.eval()  # Set model to training mode so we get the gradient


        running_loss = 0.0
        running_corrects = 0
    
        # Iterate over data.
        index=0

        for data in dset_loaders:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            #if use_gpu:
            #    inputs, labels = Variable(inputs.cuda()), \
            #    Variable(labels.cuda())
            #else:
            inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
            optimizer.zero_grad()
                
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
        
            
            for output_i in range(0,outputs.size(1)):
                Target_zeros=torch.zeros(outputs.size())
                Target_zeros=Target_zeros#.cuda()
                Target_zeros[:,output_i]=1
                Target_zeros=Variable(Target_zeros,requires_grad=False)
                targets=torch.sum(outputs*Target_zeros)
                print('Targets ',targets)
                if output_i==(outputs.size(1)-1):
                    targets.backward()
                else:
                    targets.backward(retain_graph=True )
                        
                optimizer.step(model.reg_params,True,index,labels.size(0))
                optimizer.zero_grad()
                
            ###print('step')
            optimizer.step(model.reg_params,False,index,labels.size(0))
            ##print('batch number ',index)
            index+=1
    
        return model
    def initialize_reg_params(self,freeze_layers=[]):
        """initialize an omega for each parameter to zero"""
        #TODO Maybe this is necessary: elf.register_parameter(name='bias', param=torch.nn.Parameter(torch.randn(3)))
        model= self.fit_func
        reg_params={}
        for name, param in model.named_parameters():
            if not name in freeze_layers:
                ##print('initializing param',name)
                omega=torch.FloatTensor(param.size()).zero_()
                init_val=param.data.clone()
                reg_param={}
                reg_param['omega'] = omega
                #initialize the initial value to that before starting training
                reg_param['init_val'] = init_val

                reg_params[param]=reg_param

        return reg_params
    


    def initialize_store_reg_params(self,freeze_layers=[]):
        """
        set omega to zero but after storing its value in a temp omega in which later we can accumolate them both
        # TODO IS THIS NECESSARY ? 
        
        """
        reg_params=self.fit_func.reg_params
        for name, param in self.fit_func.named_parameters():
            #in case there some layers that are not trained
            if not name in freeze_layers:
                if param in reg_params:
                    reg_param=reg_params.get(param)
                    ##print('storing previous omega',name)
                    prev_omega=reg_param.get('omega')
                    new_omega=torch.FloatTensor(param.size()).zero_()
                    init_val=param.data.clone()
                    reg_param['prev_omega']=prev_omega   
                    reg_param['omega'] = new_omega
                    
                    #initialize the initial value to that before starting training
                    reg_param['init_val'] = init_val
                    reg_params[param]=reg_param
                    
            else:
                if param in reg_params: 
                    reg_param=reg_params.get(param)
                    ##print('removing unused omega',name)
                    del reg_param['omega'] 
                    del reg_params[param]
        return reg_params
    


    def accumelate_reg_params(self,freeze_layers=[]):
        """accumelate the newly computed omega with the previously stroed one from the old previous tasks"""
        reg_params=self.fit_func.reg_params
        for name, param in self.fit_func.named_parameters():
            if not name in freeze_layers:
                if param in reg_params:
                    reg_param=reg_params.get(param)
                    ##print('restoring previous omega',name)
                    prev_omega=reg_param.get('prev_omega')
                    prev_omega=prev_omega#.cuda()
                    ##print('prev_omega',prev_omega)
                    
                    new_omega=(reg_param.get('omega'))#.cuda()
                    ##print('new_omega',new_omega)
                    acc_omega=torch.add(prev_omega,new_omega)
                    
                    del reg_param['prev_omega']
                    reg_param['omega'] = acc_omega
                
                    reg_params[param]=reg_param
                    del acc_omega
                    del new_omega
                    del prev_omega
            else:
                if param in reg_params: 
                    reg_param=reg_params.get(param)
                    ##print('removing unused omega',name)
                    del reg_param['omega'] 
                    del reg_params[param]             
        return reg_params
 

def sanitycheck(model):
    for name, param in model.named_parameters():
           
            print (name)
            if param in model.reg_params:
                #print(param)
            
                reg_param=model.reg_params.get(param)
                omega=reg_param.get('omega')
                
                print('omega max is',omega.max())
                print('omega min is',omega.min())
                print('omega mean is',omega.mean())