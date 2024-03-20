
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
from avalanche.training.utils import get_layers_and_params
from torch.nn.modules.batchnorm import _NormBase
import torch
from fnmatch import fnmatch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from CXIL.Learning.StoragePolicies import ReservoirSamplingBuffer
from CXIL.Learning.LearnerStep import Basic

# Inspired by https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/synaptic_intelligence.py 
# USED TO BE CLOCK https://github.com/ContinualAI/avalanche/blob/b225d825181df0f8a941fce51a18fef4f193150e/avalanche/training/plugins/clock.py#L14 

class Synaptic_Intelligence(Basic):
    def __init__(self, fit_func, loss = nn.CrossEntropyLoss(), lr=0.001, optimizer=torch.optim.Adam,X_test=None, y_test=None,  train_mb_size=128,si_lambda=0.0001, eps = 0.0000001,) -> None:
        '''
        Attributes: 
            fit_func: Torch Model 
            ewc_data: TODO 
            syn_data: TODO 
            excluded_parameters: TODO 
        '''
        
        #super(Synaptic_Intelligence, self).__init__( fit_func, loss, lr, optimizer,X_test, y_test)
        self.lam = (
            si_lambda if isinstance(si_lambda, (list, tuple)) else [si_lambda]
        )
        self.fit_func=fit_func
        self.optimizer = optimizer(self.fit_func.parameters(), lr=lr)
        self.loss_fn=loss
        self.eps= eps
        self.ewc_data = (dict(), dict())
        self.syn_data= {
            "old_theta": dict(),
            "new_theta": dict(),
            "grad": dict(),
            "trajectory": dict(),
            "cum_trajectory": dict(),
        }
        self.excluded_parameters= None
        self.exp_id=0

    def fit(self,X_train,taskid=None,EPOCHS=1)->None:

        
        self.fit_func.train()
        # FIT 
        #loss=0 

        '''---------BEFORE TRAINING Experience----------'''
        #if self.exp_id==0: Den ersten Teil Checke ist ireelevant 
        Synaptic_Intelligence.create_syn_data(
            self.fit_func,
            self.ewc_data,
            self.syn_data,
            self.excluded_parameters,
            )
        
        Synaptic_Intelligence.init_batch(
            self.fit_func,
            self.ewc_data,
            self.syn_data,
            self.excluded_parameters,
            )
        '''---------BEFORE TRAINING Experience---------'''
        running_loss = 0.
        for i in tqdm.trange(0,EPOCHS):
            for i, data in enumerate(X_train):
           
                inputs, labels = data
           
                #TODO THIS FIRST THAN THE OTHER
                '''---------BEFORE TRAINING Iteration---------'''
                Synaptic_Intelligence.pre_update(
                self.fit_func, self.syn_data, self.excluded_parameters
                )
                # OLD THETA HAS VALUE  --> WERTE SIND UNTERSCHIEDLICH HIER !
                #all_the_same= True
                #for param_name in self.syn_data["trajectory"]:
                #    print('NEW 1',self.syn_data["new_theta"][param_name].data)
                #    print('OLD1',self.syn_data["old_theta"][param_name].data)
                #    for j in range(0,len(self.syn_data["new_theta"][param_name].data)):
                #        if self.syn_data["new_theta"][param_name].data[j]== self.syn_data["old_theta"][param_name].data[j]:
                #            pass
                #        else: 
                #            all_the_same= False
                #            break
                #if all_the_same  and self.exp_id !=0:
                #    print('1 STOPPED AT BATCH',i)
                #    print('1 STOPPED AT EXPERIENCE',self.exp_id)
                #    import sys
                #    sys.exit(1)

                '''---------BEFORE TRAINING Iteration---------'''
            
                self.optimizer.zero_grad()

     
                #print('Syn_loss 1',syn_loss)
           
                #else: 
                #    loss= syn_loss
                #print('syn loss',loss.item())
                # Backwardspass 
                '''---------BEFORE FORWARD---------'''   
                '''---------BEFORE FORWARD---------'''       
                outputs = self.fit_func(inputs)#F.softmax()
                '''---------AFTER FORWARD---------'''  
                '''---------AFTER FORWARD---------'''
                # Compute the loss and its gradients
                if taskid is None: 
                    l_groups=labels.unique().sort()[0]
                    for lbl_idx, lbl in enumerate(l_groups):
                        labels[labels == lbl] = lbl_idx
                    if 'RRR' in str(type(self.loss_fn)):
                        loss = self.loss_fn(inputs, labels.long())
                    else:
                        loss = self.loss_fn(outputs[:, l_groups.detach().numpy().tolist()].float(), labels.long())

                else:
                    if 'RRR' in str(type(self.loss_fn)):
                        loss = self.loss_fn(inputs, labels.long(), taskid)
                    else:
                        loss = self.loss_fn(outputs.float(), labels.long())
                #class_Loss= self.loss_fn(outputs.float(), labels.long())
                #loss = class_Loss
                #TODO CHeckout Loop
                #Synaptic_Intelligence.pre_update(
                #self.fit_func, self.syn_data, self.excluded_parameters
            # )      
                '''---------BEFORE Backward---------'''  
                exp_id = self.exp_id
                #print('Experience ID',exp_id)
                try:
                    si_lamb = self.lam[exp_id]
                except IndexError:  # less than one lambda per experience, take last
                    si_lamb = self.lam[-1]
                #TODO CHECK ALL 
                #all_the_same= True
                #for param_name in self.syn_data["trajectory"]:
                #    print('NEW 2',self.syn_data["new_theta"][param_name].data)
                #    print('OLD2',self.syn_data["old_theta"][param_name].data)
                #    for j in range(0,len(self.syn_data["new_theta"][param_name].data)):
                #        if self.syn_data["new_theta"][param_name].data[j]== self.syn_data["old_theta"][param_name].data[j]:
                #            pass
                #        else: 
                #            all_the_same= False
                #            break
                #if all_the_same and self.exp_id !=0:
                #    print('2 STOPPED AT BATCH',i)
                #    print('2 STOPPED AT EXPERIENCE',self.exp_id)
                #    import sys
                #    sys.exit(1)

                syn_loss = Synaptic_Intelligence.compute_ewc_loss(
                self.fit_func,
                self.ewc_data,
                self.excluded_parameters,
                lambd=si_lamb,
                )
                
                #print('syn_loss', syn_loss)
                if syn_loss is not None:
                    loss += syn_loss
                #print(loss)
                #print('SYN',syn_loss)
                #print('CLASS',class_Loss)
                #import sys
                #sys.exit(1)
                '''---------BEFORE Backward---------''' 
                loss.backward()
                '''---------After Backward---------'''  

                '''---------After Backward---------'''  
                self.optimizer.step()
                #running_loss += loss.item()
                #if i % 1000 == 999:
                #    last_loss = running_loss / 1000 # loss per batch
                #    print('  batch {} loss: {}'.format(i + 1, last_loss))
                #    running_loss = 0.
                #ll_the_same= True
                #for param_name in self.syn_data["trajectory"]:
                #    print('NEW 3',self.syn_data["new_theta"][param_name].data)
                #   print('OLD3',self.syn_data["old_theta"][param_name].data)
                #    for j in range(0,len(self.syn_data["new_theta"][param_name].data)):
                #        if self.syn_data["new_theta"][param_name].data[j]== self.syn_data["old_theta"][param_name].data[j]:
                #            pass
                #        else: 
                #            all_the_same= False
                #            break
                #if all_the_same  and self.exp_id !=0:
                #    print('3 STOPPED AT BATCH',i)
                #    print('3 STOPPED AT EXPERIENCE',self.exp_id)
                #    import sys
                #    sys.exit(1)
                        


                '''---------After Training Iteration---------'''
                Synaptic_Intelligence.post_update(
                self.fit_func, self.syn_data, self.excluded_parameters
                )
                # OLD = NEW --> Make sense NEW WEIGHTS CURRENT "NEW WEIGHTS", OLD WEIGHTS ARE SET IN FIRST SETO 

                #for param_name in self.syn_data["trajectory"]:
                #    print(self.syn_data["new_theta"][param_name].data)
                #    print(self.syn_data["old_theta"][param_name].data)
                #    for j in range(0,len(self.syn_data["new_theta"][param_name].data)):
                #        if self.syn_data["new_theta"][param_name].data[j]== self.syn_data["old_theta"][param_name].data[j]:
                #            pass
                #        else: 
                #            break
                #        print('STOPPED AT BATCH',i)
                #        print('STOPPED AT EXPERIENCE',self.exp_id)
                #        import sys
                #        sys.exit(1)
                #for param_name in self.syn_data["trajectory"]:
                #    print('4 GRAD', self.syn_data["grad"][param_name].data)
                #    print('4 NEW THETA', self.syn_data["new_theta"][param_name].data)
                #    print('4 OLD THETA', self.syn_data["old_theta"][param_name].data)
                '''---------After Training Iteration---------'''

        self.exp_id +=1


        '''---------After Training Experience---------''' 
        Synaptic_Intelligence.update_ewc_data(
            self.fit_func,
            self.ewc_data,
            self.syn_data,
            0.001,
            self.excluded_parameters,
            1,
            eps=self.eps,
        )
        '''---------After Training Experience---------''' 

        return self.fit_func



    @staticmethod
    @torch.no_grad()
    def create_syn_data(
        model,
        ewc_data,
        syn_data,
        excluded_parameters,
    ):
        '''Initializes Data '''
        params = Synaptic_Intelligence.allowed_parameters(
            model, excluded_parameters
        )

        for param_name, param in params:
            
            if param_name not in ewc_data[0]:
                # new parameter
                ewc_data[0][param_name] = ParamData(
                    param_name, param.flatten().shape)
                ewc_data[1][param_name] = ParamData(
                    f"imp_{param_name}", param.flatten().shape)
                syn_data["old_theta"][param_name] = ParamData(
                    f"old_theta_{param_name}", param.flatten().shape)
                syn_data["new_theta"][param_name] = ParamData(
                    f"new_theta_{param_name}", param.flatten().shape)
                syn_data["grad"][param_name] = ParamData(
                    f"grad{param_name}", param.flatten().shape)
                syn_data["trajectory"][param_name] = ParamData(
                    f"trajectory_{param_name}", param.flatten().shape)
                syn_data["cum_trajectory"][param_name] = ParamData(
                    f"cum_trajectory_{param_name}", param.flatten().shape)
                #print('t',ewc_data[0][param_name].data)
                #print('param flatten', param.flatten())
            elif ewc_data[0][param_name].shape != param.shape:
                # Called after first iteration / Initialization
                # parameter expansion
                ewc_data[0][param_name].expand(param.flatten().shape)
                ewc_data[1][param_name].expand(param.flatten().shape)
                syn_data["old_theta"][param_name].expand(param.flatten().shape)
                syn_data["new_theta"][param_name].expand(param.flatten().shape)
                syn_data["grad"][param_name].expand(param.flatten().shape)
                syn_data["trajectory"][param_name]\
                    .expand(param.flatten().shape)
                syn_data["cum_trajectory"][param_name]\
                    .expand(param.flatten().shape)
        

    @staticmethod
    @torch.no_grad()
    def extract_weights(
        model, target, excluded_parameters
    ):
        params = Synaptic_Intelligence.allowed_parameters(
            model, excluded_parameters
        )

        for name, param in params:
            target[name].data = param.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def extract_grad(model, target, excluded_parameters):
        params = Synaptic_Intelligence.allowed_parameters(
            model, excluded_parameters
        )

        # Store the gradients into target
        for name, param in params:
            target[name].data = param.grad.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def init_batch(
        model,
        ewc_data,
        syn_data,
        excluded_parameters,
    ):
        # Keep initial weights
        Synaptic_Intelligence.extract_weights(
            model, ewc_data[0], excluded_parameters
        )
        for param_name, param_trajectory in syn_data["trajectory"].items():
            param_trajectory.data.fill_(0.0)

    @staticmethod
    @torch.no_grad()
    def pre_update(model, syn_data, excluded_parameters):
        Synaptic_Intelligence.extract_weights(
            model, syn_data["old_theta"], excluded_parameters
        )

    @staticmethod
    @torch.no_grad()
    def post_update(
        model, syn_data, excluded_parameters
    ):
        Synaptic_Intelligence.extract_weights(
            model, syn_data["new_theta"], excluded_parameters
        )
        Synaptic_Intelligence.extract_grad(
            model, syn_data["grad"], excluded_parameters
        )

        for param_name in syn_data["trajectory"]:
            #print('GRAD',syn_data["grad"][param_name].data)
            #print('NEW THETA',syn_data["new_theta"][param_name].data)
            #print('OLD THETA',syn_data["old_theta"][param_name].data)

            syn_data["trajectory"][param_name].data += syn_data["grad"][
                param_name
            ].data * (
                syn_data["new_theta"][param_name].data
                - syn_data["old_theta"][param_name].data
            )
            #TODO THIS IS THE ISSUE
            #print('POST new Theta', syn_data["new_theta"][param_name].data)
            #print('POST pld Theta', syn_data["old_theta"][param_name].data)
            #print( 'POST pld GRAD',syn_data["grad"][
            #    param_name
            #].data )

    @staticmethod
    def compute_ewc_loss(
        model,
        ewc_data,
        excluded_parameters,
        lambd=0.0,
        device='cpu'
    ):
        params = Synaptic_Intelligence.allowed_parameters(
            model, excluded_parameters
        )

        loss = None
        for name, param in params:
            weights = param.flatten().to(device)  # Flat, not detached
            #print('weights', weights)
            # TODO Switch Numbering
            ewc_data0 = ewc_data[0][name].data.to(device)  # Flat, detached
            ewc_data1 = ewc_data[1][name].data.to(device)  # Flat, detached
            #print(f'ewc 0 - {name} ',ewc_data0)
            #print(f'ewc 1- {name}',ewc_data1)
            #print(lambd)
            #TODO Issue is : weight- ewc_data0  is identical = 0
            syn_loss = torch.dot(
                ewc_data1, (weights - ewc_data0) ** 2
            ) * (lambd / 2)

            #print('1',(weights - ewc_data0))
            #print('2',(weights - ewc_data0)**2)
            #print('3', torch.dot(
            #    ewc_data1, (weights - ewc_data0) ** 2
            #))

            #print('weights', weights)
            #print('syn _loss', syn_loss)
            if loss is None:
                loss = syn_loss
            else:
                loss += syn_loss
        #print('syn_loss from EWC', loss.item())
        return loss

    @staticmethod
    @torch.no_grad()
    def update_ewc_data(
        net,
        ewc_data,
        syn_data,
        clip_to,
        excluded_parameters,
        c=0.0015,
        eps= 0.0000001,
    ):
        Synaptic_Intelligence.extract_weights(
            net, syn_data["new_theta"], excluded_parameters
        )

        for param_name in syn_data["cum_trajectory"]:
            #print(syn_data["trajectory"][param_name].data)
            syn_data["cum_trajectory"][param_name].data += (
                c
                * syn_data["trajectory"][param_name].data
                / (
                    np.square(
                        syn_data["new_theta"][param_name].data
                        - ewc_data[0][param_name].data
                    )
                    + eps
                )
            )
            #print('trajectory', syn_data["trajectory"][param_name].data)
            #print('new_Theta',syn_data["new_theta"][param_name].data )
            #print('EWC0',ewc_data[0][param_name].data )
        #TODO seems line syn_data["cum_trajectory"] is not supposed to be emoty 
        for param_name in syn_data["cum_trajectory"]:
            #print('cum',syn_data["cum_trajectory"][param_name])
            ewc_data[1][param_name].data = torch.empty_like(
                syn_data["cum_trajectory"][param_name].data
            ).copy_(-syn_data["cum_trajectory"][param_name].data)

        # change sign here because the Ewc regularization
        # in Caffe (theta - thetaold) is inverted w.r.t. syn equation [4]
        # (thetaold - theta)
        for param_name in ewc_data[1]:
            #print(f'{param_name} - Old')
            #print(ewc_data[0][param_name].data)
            #print(ewc_data[1][param_name].data)
            ewc_data[1][param_name].data = torch.clamp(
                ewc_data[1][param_name].data, max=clip_to
            )
            ewc_data[0][param_name].data = \
                syn_data["new_theta"][param_name].data.clone()
            #print(f'{param_name} - New')
            #print(ewc_data[0][param_name].data)
            #print(ewc_data[1][param_name].data)
            #print(syn_data["new_theta"][param_name].data)
        #import sys
        #sys.exit(1)

    @staticmethod
    def explode_excluded_parameters(excluded):
        """
        Explodes a list of excluded parameters by adding a generic final ".*"
        wildcard at its end.
        :param excluded: The original set of excluded parameters.
        :return: The set of excluded parameters in which ".*" patterns have been
            added.
        """
        result = set()
        #for x in excluded:
        #    result.add(x)
        #    if not x.endswith("*"):
        #        result.add(x + ".*")
        return result

    @staticmethod
    def not_excluded_parameters(
        model, excluded_parameters):
        # Add wildcards ".*" to all excluded parameter names
        result= []
        excluded_parameters = (
            Synaptic_Intelligence.explode_excluded_parameters(
                excluded_parameters
            )
        )
        layers_params = get_layers_and_params(model)

        for lp in layers_params:
            if isinstance(lp.layer, _NormBase):
                # Exclude batch norm parameters
                excluded_parameters.add(lp.parameter_name)

        for name, param in model.named_parameters():
            accepted = True
            for exclusion_pattern in excluded_parameters:
                if fnmatch(name, exclusion_pattern):
                    accepted = False
                    break

            if accepted:
                result.append((name, param))

        return result

    @staticmethod
    def allowed_parameters(
        model, excluded_parameters
    ):

        allow_list = Synaptic_Intelligence.not_excluded_parameters(
            model, excluded_parameters
        )

        result = []
        for name, param in allow_list:
            if param.requires_grad:
                result.append((name, param))

        return result
 
from collections import defaultdict
from typing import NamedTuple, List, Optional, Tuple, Callable, Union
  
class ParamData(object):
    '''
    Taken from: https://github.com/ContinualAI/avalanche/blob/69f2563853c114fb7ea5e7237494b673a8a2c98c/avalanche/training/utils.py#L322
    '''
    def __init__(
            self,
            name: str, shape: tuple = None,
            init_function: Callable[[torch.Size], torch.Tensor] = torch.zeros,
            init_tensor: Union[torch.Tensor, None] = None,
            device: str = 'cpu'):
        """
        An object that contains a tensor with methods to expand it along
        a single dimension.
        :param name: data tensor name as a string
        :param shape: data tensor shape. Will be set to the `init_tensor`
            shape, if provided.
        :param init_function: function used to initialize the data tensor.
            If `init_tensor` is provided, `init_function` will only be used
            on subsequent calls of `reset_like` method.
        :param init_tensor: value to be used when creating the object. If None,
            `init_function` will be used.
        :param device: pytorch like device specification as a string
        """
        assert isinstance(name, str)
        assert (init_tensor is not None) or (shape is not None)
        if init_tensor is not None and shape is not None:
            assert init_tensor.shape == shape

        self.init_function = init_function
        self.name = name
        self.shape = torch.Size(shape) if shape is not None else \
            init_tensor.size()
        self.device = device
        if init_tensor is not None:
            self._data: torch.Tensor = init_tensor
        else:
            self.reset_like()

    def reset_like(self, shape=None, init_function=None):
        """
        Reset the tensor with the shape provided or, otherwise, by
        using the one most recently provided. The `init_function`,
        if provided, does not override the default one.
        :param shape: the new shape or None to use the current one
        :param init_function: init function to use or None to use
            the default one.
        """
        #print('RESET IS CALLED ')
        if shape is not None:
            self.shape = torch.Size(shape)
        if init_function is None:
            init_function = self.init_function
        self._data = init_function(self.shape).to(self.device)


    def expand(self, new_shape, padding_fn=torch.zeros):
        """
        Expand the data tensor along one dimension.
        The shape cannot shrink. It cannot add new dimensions, either.
        If the shape does not change, this method does nothing.
        :param new_shape: expanded shape
        :param padding_fn: function used to create the padding
            around the expanded tensor.
        :return the expanded tensor or the previous tensor
        """
        assert len(new_shape) == len(self.shape), \
            "Expansion cannot add new dimensions"
        expanded = False
        for i, (snew, sold) in enumerate(zip(new_shape, self.shape)):
            assert snew >= sold, "Shape cannot decrease."
            if snew > sold:
                assert not expanded, \
                    "Expansion cannot occur in more than one dimension."
                expanded = True
                exp_idx = i

        if expanded:
            old_data = self._data.clone()
            old_shape_len = self._data.shape[exp_idx]
            self.reset_like(new_shape, init_function=padding_fn)
            idx = [slice(el) if i != exp_idx else
                   slice(old_shape_len) for i, el in
                   enumerate(new_shape)]
            self._data[idx] = old_data
        return self.data

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, value):
        assert value.shape == self._data.shape, \
            "Shape of new value should be the same of old value. " \
            "Use `expand` method to expand one dimension. " \
            "Use `reset_like` to reset with a different shape."
        self._data = value

    def __str__(self):
        return f"ParamData_{self.name}:{self.shape}:{self._data}"
