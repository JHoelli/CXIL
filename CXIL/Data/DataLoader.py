import numpy as np 
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.datasets import load_diabetes
from sklearn.datasets import make_regression, make_classification, make_blobs
import pickle
import os 
import os
import gzip
import struct
import array
import numpy as np
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from abc import ABC, abstractmethod
from pathlib import Path
import torch
from scipy.signal import resample
import matplotlib.pyplot as plt
import pandas as pd 
import ast 


class simulation(ABC):
    '''
    Simulation Interface. 


    '''
    def __init__(self):
        parse_images
    @abstractmethod
    def get_ground_truth(self,x):
        pass
    @abstractmethod
    def simulate(self,img, label, importance):
        pass

    def __call__(self,img, label, importance):
        return self.simulate(img, label, importance)


def k_largest_index_argsort(a, k):
        idx = np.argsort(a.ravel())[:-k-1:-1]
        return np.column_stack(np.unravel_index(idx, a.shape))

    
class timeseries(simulation):
    def __init__(self,meta=None, test_meta= None,input= None, cnn=False) -> None:
        self.meta=meta
        self.test_meta=test_meta
        self.input=input
        self.cnn=cnn
    def simulate(self,img, label, importance,id = None):

        if len(img.shape)==2:
            img=img[np.newaxis,:]
           
        # FOR RRR find original entry
        #if self.input is not None:
        a=self.input.reshape(len(self.input),-1).astype('float32')
        if not isinstance(img, np.ndarray):
            img=img.detach().numpy()
        b=img.reshape(-1)

        id=np.where(np.all(a==b,axis=1))[0][0]
        #print(id)


        #self.counter+=1
        attribution = np.zeros_like(np.array(img))
        attribution=attribution.reshape(img.shape[-2], img.shape[-1])
        #print(img.shape)
        mask= np.zeros_like(np.array(img))
    

        TargetTS_Starts=self.meta[id,1]
        TargetTS_Ends=self.meta[id,2]
        TargetFeat_Starts= self.meta[id,3]
        TargetFeat_Ends= self.meta[id,4] 
        label_true= int(self.meta[id][0])
        # FORMAT ASSUMPTION IS FEAT   
        mask[0][int(TargetFeat_Starts):int(TargetFeat_Ends),int(TargetTS_Starts):int(TargetTS_Ends)]=1

        #print('label_true ',label_true)
        #print('label ', label)
        
        if label_true != label: 
            return label_true, attribution
        
        importance=np.array(importance).reshape(mask.shape[-2], mask.shape[-1])
        importance = np.abs(importance)
        n=np.count_nonzero(mask)

        if len(mask.shape)>2 or mask.shape[0]>1:
            #print('IF')
            if mask.shape[0]==1:
                mask=mask[0]

            indices = k_largest_index_argsort(importance,n)
            for i in range(mask.shape[0]):
                if i in indices[:,0]:
                    indicies_mask = np.where(mask[i] != 0 )[0]
                    new_in=indices[indices[:,0]==i]
                    if len (indicies_mask) != 0:

                        a,b,c= np.intersect1d([new_in[:,1]], indicies_mask, return_indices=True)
                        indicies_to_change =new_in[:,1][np.isin(new_in[:,1],a,invert=True)]
                        attribution[i][indicies_to_change]=1
                    else:

                        attribution[i][new_in[:,1]]=1

            attribution=attribution[np.newaxis,:]
        else:
            indices = np.argpartition(importance, -n)[-n:]
            indicies_mask = np.where(mask != 0 )          
            a,b,c= np.intersect1d(indices, indicies_mask, return_indices=True)
            indicies_to_change =indices[np.isin(indices,a,invert=True)]
            attribution[indicies_to_change]=1
 
        #print('attribution ',attribution.shape)
        return label, attribution
    
    
    
    def get_ground_truth(self,x):
        #print('x', x.shape)
        #import sys 
        #sys.exit(1)
        if self.cnn:
            x=np.swapaxes(x, -2,-1)
        empty_mask = np.zeros_like(x)
        #print('x',x.shape)
        #print('test',self.test_meta.shape)
        #print('mask',empty_mask.shape)
        #import sys 
        #sys.exit(1)
        a=self.input.reshape(len(self.input),-1).astype('float32')
        for i in range(0, len(x)):
                #print(np.where(np.all(a==x[i].astype('float32').reshape(-1),axis=1)))
                id=np.where(np.all(a==x[i].astype('float32').reshape(-1),axis=1))[0][0]
                #print(id)
                #print(self.meta[id])
                
                #TimeSteps
                start = self.meta[id,1:3]
                #print('start ',start)
                #features
                end = self.meta[id,3:]
                #print('end ',end)
                empty_mask[i,int(start[0]):int(start[1]),int(end[0]):int(end[1])]=1 
                #print(empty_mask[i])
                #print(empty_mask.shape)
                #i+=
        if self.cnn:
            empty_mask=np.swapaxes(empty_mask, -2,-1)
        #import sys 
        #sys.exit(1)
        #print(empty_mask.shape)
        #import sys 
        #sys.exit(1)
        return empty_mask
    
class synthetic_tabular(simulation):
    
    def __init__(self,n_informative, n_redundant , n_repeated=0,n_classes=2, data=None, class_mode=False):
        self.informative=n_informative
        self.redundant=n_redundant
        self.repeated=n_repeated
        self.index= n_informative + n_repeated +n_redundant
        self.counter=0
        self.x,self.y=data
        self.x= torch.from_numpy(self.x).float().detach().numpy()
        self.class_mode=class_mode

    def get_ground_truth(self,x, class_mode= False):
        if not class_mode:
            mask=np.zeros_like(x)
            mask[:,:self.index]=1
        else : 
            #THIS IS A TODO 
            pass
        return mask
    
    def find_rows(self, target):
        source =  self.x
        return np.where((source == target.reshape(-1)).all(axis=1))[0]
    
    def simulate(self,img, label, importance):
        '''all useful features are contained in the columns X[:, :n_informative + n_redundant + n_repeated]'''
        '''TODO How to get label !'''
        #print('img ', img)

        self.counter +=1
        
        if not isinstance(img, np.ndarray):
            img=img.detach().numpy()
        attribution = np.zeros_like(np.array(img).reshape(-1))
        mask= np.zeros_like(np.array(img))
        ind=self.find_rows( img)
        label_true= self.y[ind]
        if label_true[0] != label: 
            return label_true, attribution
        if not self.class_mode:
            if len(mask.shape)==2:
                mask[:,:self.index]=1
            elif len(mask.shape)==1:
                mask[:self.index]=1
            importance=np.array(importance)
            importance = np.abs(importance)
            n=np.count_nonzero(mask)
            indices = np.argpartition(importance, -n)[-n:]
            indicies_mask = np.where(mask.reshape(-1) != 0 )[0] 
            a,_,_= np.intersect1d(indices, indicies_mask, return_indices=True)
            indicies_to_change =indices[np.isin(indices,a,invert=True)]
            attribution[indicies_to_change]=1       
        else: 
            if len(mask.shape)==2:
                mask[:,label_true[0]]=1
            elif len(mask.shape)==1:
                mask[label_true[0]]=1
            importance=np.array(importance)
            importance = np.abs(importance)
            n=np.count_nonzero(mask)
            indices = np.argpartition(importance, -n)[-n:]
            indicies_mask = np.where(mask.reshape(-1) != 0 )[0] 
            a,_,_= np.intersect1d(indices, indicies_mask, return_indices=True)
            indicies_to_change =indices[np.isin(indices,a,invert=True)]
            attribution[indicies_to_change]=1

        
        return label, attribution

class toy_simulation(simulation):
    def __init__(self, cnn_mode):
        self.cnn_mode=cnn_mode
    def simulate(self, img, label, importance):
        '''
        Simulation for toy dataset. 

        Attributes: 
        img nd.array: In shape (# samples, -1), # samples currently always 1 
        label int: Label as int 
        importance nd.array: Importance score in shape 
        rule str: if multiple rules exist, rule that is supposed to be applied 
        '''
        counts = defaultdict(int)
        if self.cnn_mode:
            img=np.transpose(img.reshape(1,3,5,5), (0,2, 3, 1))
            importance=np.transpose(importance.reshape(1,3,5,5), (0,2, 3, 1))
        img = img.reshape(-1)
        attribution = np.zeros_like(np.array(img).reshape(-1))
        importance=np.array(importance)
        importance = np.abs(importance)
        dic = [0,1,2,12,13,14,60,61,62,72,73,74]
        imglen = 5
        img_test=img.reshape(imglen, imglen, 3)
        if  img_test[0,0,0] == img_test[imglen-1,imglen-1,0] and img_test[0,imglen-1,0] == img_test[imglen-1,imglen-1,0] and  img_test[imglen-1,0,0] == img_test[imglen-1,imglen-1,0]:
            if  img_test[0,0,1] == img_test[imglen-1,imglen-1,1] and img_test[0,imglen-1,1] == img_test[imglen-1,imglen-1,1] and  img_test[imglen-1,0,1] == img_test[imglen-1,imglen-1,1]:
                if  img_test[0,0,2] == img_test[imglen-1,imglen-1,2] and img_test[0,imglen-1,2] == img_test[imglen-1,imglen-1,2] and  img_test[imglen-1,0,2] == img_test[imglen-1,imglen-1,2]:
                    if label ==1:
                        print('Wrongly 1')
                        return  0,attribution
                else:
                    if label ==0: 
                        print('B Wrongly 0')
                        return  1,attribution
            else:
                if label ==0: 
                    print('c Wrongly 0')
                    return  1,attribution
        else: 
            if label ==0: 
                print('D Wrongly 0')
                return  1,attribution
        n=len(dic)
        print(importance.shape)
        # TODO is this reshape Alright ? 
        indices = np.argpartition(importance.reshape(-1), -n)[-n:]
        print(indices)
        idx = indices
        print(idx)
        for i in range(0,len(idx)):
            if idx[i] not in dic:
                attribution[idx[i]]=1
        for idx in dic: 
            attribution[idx]=0
        
        if self.cnn_mode:
            attribution=np.transpose(attribution.reshape(5,5,3),(1, 2, 0))
        return label,  attribution.reshape(-1)
    def get_ground_truth(self,x):
        if self.cnn_mode:
            x=np.transpose(x.reshape(-1,3,5,5), (0,2, 3, 1))
        mask=np.zeros_like(x).reshape(x.shape[0],-1)
        ground_truth=[0,1,2,12,13,14,60,61,62,72,73,74]
        mask[:,ground_truth]=1
        if self.cnn_mode:
            mask= np.transpose(mask.reshape(-1,5,5,3), (0,2, 3, 1)).reshape(mask.shape[0],-1)
        return mask


def diabetis_simulation():
     #TODO Regression
    coeff= pickle.load('')

    pass

def diabetis_regression_data():
     #TODO Regression
    '''
    Use a fitted linear regression and define x most important features (e.g., top 3 )

    Also possible statistically determine if variable is relevant or not with tests ? 
    '''
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=13)
    X, X_train, y, y_train = train_test_split( X, y, test_size=0.4, random_state=13)
    
    return X,y, X_train,y_train, X_test,y_test, diabetis_simulation


def find_rows(source, target):
    return np.where((source == target.reshape(-1)).all(axis=1))[0]


def top_n_indexes(arr, n):
    idx = np.argpartsort(arr, arr.size-n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]

class decoy_mnist_rule(simulation):
    
    def __init__(self, data):
        self.x, self.y=data
        
    def simulate(self, img, label, importance):
        '''
        Rule to ignore corners!
        '''
        orgim=img
        if torch.is_tensor(img):
            img = img.detach().numpy()
        img = img.copy()
        img=img.reshape(28, 28)
        attr = np.zeros_like(img)

        inde=find_rows(self.x.reshape(-1,28*28),img.reshape(-1))

        

        y= self.y[inde]

        if label != y[0]:
            return y,  attr.reshape(-1)

        importance=importance.reshape(28,28)
        importance = np.abs(importance)
        n=np.count_nonzero(img)-25

        indices = np.unravel_index(np.argsort(importance.ravel())[-n:], importance.shape)
        fwd = [0,1,2,3,24,25,26,27]
        rev = [0,1,2,3,24,25,26,27]
        for i in fwd:
            for j in rev:

                if i in indices[0] and j in indices[1]:
                    if img[i,j]>0:
                        attr[i][j] = 1
        return label,  attr.reshape(orgim.shape)
    
    def get_ground_truth(self, x, decoy= False):
        #TODO Ground Truth -- as in not values on the corners ? 
        if decoy:
            attr= np.zeros_like(x) 
            attr[x!=0]=1
            fwd = [0,1,2,3]
            rev = [-1,-2,-3,-4]
            for i in fwd:
                for j in rev:
                    for h in range(0, len(attr)):
                        attr[h][0][i][j] = 0
        else: 
            attr  = np.zeros_like(x)
            # ALL Color full Edges are relevant
            attr[x != 0 ] =1
            # Excep the Edges
            fwd = [0,1,2,3,24,25,26,27]
            rev = [0,1,2,3,24,25,26,27]
            for i in fwd:
                for j in rev:
                    for h in range(0, len(attr)):
                        attr[h][0][i][j] = 0   
    
        return attr 


class mnist_rule(simulation):
    
    def __init__(self, data):
        self.x, self.y=data


    def simulate(self, img, label, importance):
        '''
        Rule to ignore corners!
        '''
        orgim=img
 
        if torch.is_tensor(img):
            img = img.detach().numpy()
        img = img.copy()
        img=img.reshape(28*28)
        attr = np.zeros_like(img)
        inde=find_rows(self.x.reshape(-1,28*28),img.reshape(-1))#np.where(img.reshape(-1)==self.x.reshape(-1,28*28)) 
        y= self.y[inde]
        if label != y[0]:
            return y,  attr.reshape(-1)

        importance=importance.reshape(28*28)
        importance = np.abs(importance)
        n=len(img[img!=0])
        indices = np.argpartition(importance, -n)[-n:]
        for a in indices:
            if img[a]== 0:
                attr[a] = 1

        return label,  attr.reshape(orgim.shape)
    
    def get_ground_truth(self, x):
        print('No GT available')


def split_decoy_mnist():
    '''
    Loads Decoy MNIST according to Ross et al. and Teso et al., split into tasks of 2 consecquitive digits.
    '''
    with np.load('./CXIL/Data/data/split-decoy-mnist/split-decoy-mnist.npz') as data:
        X_0=data['arr_0']
        y_0=data['arr_1']
        X_1=data['arr_2']
        y_1=data['arr_3']
        X_2=data['arr_4']
        y_2=data['arr_5']
        X_3=data['arr_6']
        y_3=data['arr_7']
        X_4=data['arr_8']
        y_4=data['arr_9']
        Xt_0=data['arr_10']
        yt_0=data['arr_11']
        Xt_1=data['arr_12']
        yt_1=data['arr_13']
        Xt_2=data['arr_14']
        yt_2=data['arr_15']
        Xt_3=data['arr_16']
        yt_3=data['arr_17']
        Xt_4=data['arr_18']
        yt_4=data['arr_19']
        Xt=data['arr_20']
        yt=data['arr_21']
    print(X_0.shape)
    print(Xt.shape)
   
    # TODO DECOY RULE 
    return [X_0,X_1,X_2,X_3,X_4],[y_0,y_1,y_2,y_3,y_4],[Xt_0,Xt_1,Xt_2,Xt_3,Xt_4],[yt_0,yt_1,yt_2,yt_3,yt_4], Xt, yt, decoy_mnist_rule((np.concatenate([X_0,X_1,X_2,X_3,X_4,Xt.reshape(-1,784)], axis=0),np.concatenate([y_0,y_1,y_2,y_3,y_4,yt], axis=0)))#.decoy_mnist_rule

def split_mnist(balance=True, num_items_to_use=None, normelize=True):
    '''
    Loads Split MNIST. Beaware that this can only be used as a Sanity Check for the Continous Learner as the simulation rune is missing. 
    '''
    with np.load('./CXIL/Data/data/split-mnist/split-mnist.npz') as data:
        X_0=data['arr_0']
        y_0=data['arr_1']
        X_1=data['arr_2']
        y_1=data['arr_3']
        X_2=data['arr_4']
        y_2=data['arr_5']
        X_3=data['arr_6']
        y_3=data['arr_7']
        X_4=data['arr_8']
        y_4=data['arr_9']
        Xt_0=data['arr_10']
        yt_0=data['arr_11']
        Xt_1=data['arr_12']
        yt_1=data['arr_13']
        Xt_2=data['arr_14']
        yt_2=data['arr_15']
        Xt_3=data['arr_16']
        yt_3=data['arr_17']
        Xt_4=data['arr_18']
        yt_4=data['arr_19']
        Xt=data['arr_20']
        yt=data['arr_21']
    a= min([len(X_0),len(X_1),len(X_2),len(X_3),len(X_4)])
    X_0=X_0[:a]
    X_1=X_1[:a]
    X_2=X_2[:a]
    X_3=X_3[:a]
    X_4=X_4[:a]
    y_0=y_0[:a]
    y_1=y_1[:a]
    y_2=y_2[:a]
    y_3=y_3[:a]
    y_4=y_4[:a]
   
    # TODO DECOY RULE 
    return [X_0,X_1,X_2,X_3,X_4],[y_0,y_1,y_2,y_3,y_4],[Xt_0,Xt_1,Xt_2,Xt_3,Xt_4],[yt_0,yt_1,yt_2,yt_3,yt_4], Xt, yt, mnist_rule((np.vstack([X_0,X_1,X_2,X_3,X_4,Xt]),np.concatenate((y_0,y_1,y_2,y_3,y_4,yt), axis=0)))



def parse_labels(filename):
    with gzip.open(filename, 'rb') as fh:
      magic, num_data = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

def parse_images(filename):
    with gzip.open(filename, 'rb') as fh:
      magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

def unsplit_mnist():
    '''
    IS THIS IN CORRESPONDANCE WITH : https://arxiv.org/pdf/1904.07734.pdf
    TODO Do Format as Intended
    '''
    datadir = './CXIL/Data/data/split-mnist'
    train_images = parse_images(os.path.join(datadir, 'train-images-idx3-ubyte.gz'))
    train_labels = parse_labels(os.path.join(datadir, 'train-labels-idx1-ubyte.gz'))
    test_images  = parse_images(os.path.join(datadir, 't10k-images-idx3-ubyte.gz'))
    test_labels  = parse_labels(os.path.join(datadir, 't10k-labels-idx1-ubyte.gz'))#
    with np.load('./CXIL/Data/data/split-mnist/split-mnist.npz') as data:
        X_0=data['arr_0']
        y_0=data['arr_1']
        X_1=data['arr_2']
        y_1=data['arr_3']
        X_2=data['arr_4']
        y_2=data['arr_5']
        X_3=data['arr_6']
        y_3=data['arr_7']
        X_4=data['arr_8']
        y_4=data['arr_9']
        Xt_0=data['arr_10']
        yt_0=data['arr_11']
        Xt_1=data['arr_12']
        yt_1=data['arr_13']
        Xt_2=data['arr_14']
        yt_2=data['arr_15']
        Xt_3=data['arr_16']
        yt_3=data['arr_17']
        Xt_4=data['arr_18']
        yt_4=data['arr_19']
        Xt=data['arr_20']
        yt=data['arr_21']

    return  np.concatenate((X_0,X_1,X_2,X_3,X_4)),np.concatenate((y_0,y_1,y_2,y_3,y_4)),[Xt_0,Xt_1,Xt_2,Xt_3,Xt_4],[yt_0,yt_1,yt_2,yt_3,yt_4], test_images, test_labels, None

def mnist():
    '''
    IS THIS IN CORRESPONDANCE WITH : https://arxiv.org/pdf/1904.07734.pdf
    TODO Do Format as Intended
    '''
    datadir = './CXIL/Data/data/decoy-mnist'
    train_images = parse_images(os.path.join(datadir, 'train-images-idx3-ubyte.gz'))
    train_labels = parse_labels(os.path.join(datadir, 'train-labels-idx1-ubyte.gz'))
    test_images  = parse_images(os.path.join(datadir, 't10k-images-idx3-ubyte.gz'))
    test_labels  = parse_labels(os.path.join(datadir, 't10k-labels-idx1-ubyte.gz'))#
    with np.load('./CXIL/Data/data/decoy-mnist/decoy-mnist.npz') as data:
        X_0=data['arr_0']
        y_0=data['arr_1']
        X_1=data['arr_2']
        y_1=data['arr_3']
        X_2=data['arr_4']
        y_2=data['arr_5']
        X_3=data['arr_6']
        y_3=data['arr_7']
        X_4=data['arr_8']
        y_4=data['arr_9']
        Xt_0=data['arr_10']
        yt_0=data['arr_11']
        Xt_1=data['arr_12']
        yt_1=data['arr_13']
        Xt_2=data['arr_14']
        yt_2=data['arr_15']
        Xt_3=data['arr_16']
        yt_3=data['arr_17']
        Xt_4=data['arr_18']
        yt_4=data['arr_19']
        Xt=data['arr_20']
        yt=data['arr_21']

    return  train_images,train_labels,[Xt_0,Xt_1,Xt_2,Xt_3,Xt_4],[yt_0,yt_1,yt_2,yt_3,yt_4], test_images, test_labels, None

def mnist_label_transformer(yt):
    yt[yt==0]=0
    yt[yt==2]=0
    yt[yt==4]=0
    yt[yt==6]=0
    yt[yt==8]=0
    yt[yt==1]=1
    yt[yt==3]=1
    yt[yt==5]=1
    yt[yt==7]=1
    yt[yt==9]=1
    return yt


def split_mnist_binary():
    '''
    IS THIS IN CORRESPONDANCE WITH : https://arxiv.org/pdf/1904.07734.pdf
    TODO Do Format as Intended
    '''
    with np.load('./CXIL/Data/data/split-mnist/split-mnist.npz') as data:
        X_0=data['arr_0']
        y_0=data['arr_1']
        y_0[y_0==0]=0
        y_0[y_0==1]=1
        X_1=data['arr_2']
        y_1=data['arr_3']
        y_1[y_1==2]=0
        y_1[y_1==3]=1
        X_2=data['arr_4']
        y_2=data['arr_5']
        y_2[y_2==4]=0
        y_2[y_2==5]=1
        X_3=data['arr_6']
        y_3=data['arr_7']
        y_3[y_3==6]=0
        y_3[y_3==7]=1
        X_4=data['arr_8']
        y_4=data['arr_9']
        y_4[y_4==8]=0
        y_4[y_4==9]=1
        Xt_0=data['arr_10']
        yt_0=data['arr_11']
        yt_0[yt_0==0]=0
        yt_0[yt_0==1]=1
        Xt_1=data['arr_12']
        yt_1=data['arr_13']
        y_1[np.where(yt_1==2)]=0
        y_1[np.where(yt_1==3)]=1
        Xt_2=data['arr_14']
        yt_2=data['arr_15']
        yt_2[yt_2==4]=0
        yt_2[yt_2==5]=1
        Xt_3=data['arr_16']
        yt_3=data['arr_17']
        yt_3[yt_3==6]=0
        yt_3[yt_3==7]=1
        Xt_4=data['arr_18']
        yt_4=data['arr_19']
        yt_4[yt_4==8]=0
        yt_4[yt_4==9]=1
        Xt=data['arr_20']
        yt=data['arr_21']
        yt[yt==0]=0
        yt[yt==2]=0
        yt[yt==4]=0
        yt[yt==6]=0
        yt[yt==8]=0
        yt[yt==1]=1
        yt[yt==3]=1
        yt[yt==5]=1
        yt[yt==7]=1
        yt[yt==9]=1
    return  [X_0,X_1,X_2,X_3,X_4],[y_0,y_1,y_2,y_3,y_4],[Xt_0,Xt_1,Xt_2,Xt_3,Xt_4],[yt_0,yt_1,yt_2,yt_3,yt_4], Xt, yt, None

def split_decoy_mnist_binary():
    '''
    IS THIS IN CORRESPONDANCE WITH : https://arxiv.org/pdf/1904.07734.pdf
    TODO Do Format as Intended
    '''
    with np.load('./CXIL/Data/data/split-decoy-mnist/split-decoy-mnist.npz') as data:
        X_0=data['arr_0']
        y_0=data['arr_1']
        y_0[y_0==0]=0
        y_0[y_0==1]=1
        X_1=data['arr_2']
        y_1=data['arr_3']
        y_1[y_1==2]=0
        y_1[y_1==3]=1
        X_2=data['arr_4']
        y_2=data['arr_5']
        y_2[y_2==4]=0
        y_2[y_2==5]=1
        X_3=data['arr_6']
        y_3=data['arr_7']
        y_3[y_3==6]=0
        y_3[y_3==7]=1
        X_4=data['arr_8']
        y_4=data['arr_9']
        y_4[y_4==8]=0
        y_4[y_4==9]=1
        Xt_0=data['arr_10']
        yt_0=data['arr_11']
        yt_0[yt_0==0]=0
        yt_0[yt_0==1]=1
        Xt_1=data['arr_12']
        yt_1=data['arr_13']
        y_1[np.where(yt_1==2)]=0
        y_1[np.where(yt_1==3)]=1
        Xt_2=data['arr_14']
        yt_2=data['arr_15']
        yt_2[yt_2==4]=0
        yt_2[yt_2==5]=1
        Xt_3=data['arr_16']
        yt_3=data['arr_17']
        yt_3[yt_3==6]=0
        yt_3[yt_3==7]=1
        Xt_4=data['arr_18']
        yt_4=data['arr_19']
        yt_4[yt_4==8]=0
        yt_4[yt_4==9]=1
        Xt=data['arr_20']
        yt=data['arr_21']
        yt[yt==0]=0
        yt[yt==2]=0
        yt[yt==4]=0
        yt[yt==6]=0
        yt[yt==8]=0
        yt[yt==1]=1
        yt[yt==3]=1
        yt[yt==5]=1
        yt[yt==7]=1
        yt[yt==9]=1
    return  [X_0,X_1,X_2,X_3,X_4],[y_0,y_1,y_2,y_3,y_4],[Xt_0,Xt_1,Xt_2,Xt_3,Xt_4],[yt_0,yt_1,yt_2,yt_3,yt_4], Xt, yt, decoy_mnist_rule((np.vstack([X_0,X_1,X_2,X_3,X_4,Xt]),np.concatenate((y_0,y_1,y_2,y_3,y_4,yt), axis=0))).decoy_mnist_rule

def decoy_mnist(root):
    '''

    Loads Decoy MNIST according to Ross et al. and Teso et al. 
    
    '''
    with np.load(f'{root}/Data/data/decoy-mnist/decoy-mnist.npz') as data:
        print(data.keys())
        #for a in data.keys():
        #    print(a)
        X=data['arr_1'].reshape(-1,1,28,28)
        y=data['arr_2']
        Xt=data['arr_5'].reshape(-1,1,28,28)
        yt=data['arr_6']
    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)

    return   X_train,y_train,X_test,y_test, Xt, yt,decoy_mnist_rule((np.concatenate([X,Xt]),np.concatenate([y,yt])))

def clevr_xai():
    if not os.path.isdir('./CXIL/Data/data/clevr_xai'):
        os.mkdir('./CXIL/Data/data/clevr_xai')#
        zipurl='https://github.com/ahmedmagdiosman/clevr-xai/releases/download/v1.0/CLEVR-XAI_v1.0.zip'
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('./CXIL/Data/data/clevr_xai')
        zipurl='https://github.com/ahmedmagdiosman/clevr-xai/releases/download/v1.0/CLEVR-XAI_v1.0_images_masks.zip'
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('./CXIL/Data/data/clevr_xai')

    # LOAD IMAGES
    # LOAD MASKS
    # LOAD LABEL 
    # SPLIT DATA TRAIN /VAL 
    # SPLIT DATA TRAIN/FINETUNE

class iris_simulate(simulation): 

    def __init__(self, data=None):
        
        self.x, self.y = data
        self.x=  torch.from_numpy(self.x).float().detach().numpy()
    
    def find_rows(self, target):
        source =  self.x#np.round(self.x.astype(float),decimals=5)#np.round(self.x,decimals=5)
        #target = np.round(target.astype(float),decimals=5)
        print('FIND ROW', source.shape)
        print('Target', target.shape)
        print(target)
        print(source)
        #print(type(source))
        #print(type(target))
        #print(source == target.reshape(-1))
        #print(np.where((source == target).all(axis=1)))
        return np.where((source == target.reshape(-1)).all(axis=1))[0]

 
    def simulate(self,img, label, importance):
        '''
        Simulation for toy dataset. 

        Attributes: 
            img nd.array: In shape (# samples, -1), # samples currently always 1 
            label int: Label as int 
            importance nd.array: Importance score in shape 
            rule str: if multiple rules exist, rule that is supposed to be applied 
        '''
        if torch.is_tensor(img):
            img = img.detach().numpy()

        attribution = np.zeros_like(np.array(img).reshape(-1))
        importance=np.array(importance)
        importance = np.abs(importance)
        #inde=np.where((X==img[0]).all(axis=1))[0][0]
        ind=self.find_rows( img)
        #print('find rows',ind)
        #print(self.y[ind])
        #print(label)

        if self.y[ind][0]!= label:
            return self.y[ind],attribution

        dic=[*range(0,len(img[:4]))]
        n=len(dic)
        #print('dic ',dic)
        #print('n ',n)
        #print(importance)
        
        indices = np.argpartition(importance.reshape(-1), -n)[-n:]
        #print(indices)

        idx = indices
        for i in range(0,len(idx)):
            if idx[i] in dic:
                attribution[idx[i]]=1
        return label, attribution 
    
    def get_ground_truth(self, x):
        ground_truth=[*range(4,len(x[0,4:]))]
        masks = np.zeros_like(x)
        masks[:,ground_truth]=1
        return masks
    
def iris_cancer():
    path= './CXIL/Data/data'
    X=np.load(f'{path}/Iris_Cancer_X.npy')
    Xt=np.load(f'{path}/Iris_Cancer_X_test.npy')
    y=np.load(f'{path}/Iris_Cancer_y.npy')
    yt=np.load(f'{path}/Iris_Cancer_y_test.npy')
    X_train, X_test, y_train, y_test=train_test_split(Xt,yt,random_state=0,train_size=0.5)
    return X,y, X_train, y_train,X_test,y_test, iris_simulate((np.concatenate((X,Xt)),np.concatenate((y,yt))))


def load_lego(root, taskwise=True):
    '''
    Inspired by https://github.com/tmdt-buw/continual-learning-mas-cloning-injection-molding/blob/main/sequence_experiments.py
    '''
    # CXIL/Data/data/Lego_Injection_Modeling/injection_molding_lego_data_full.csv
    csv_path = f'{root}/Data/data/Lego_Injection_Modeling/injection_molding_lego_data_full.csv'
    csv_sep = ';'

    # input and output attributes (in specified order)
    input_attrs  = ['Kuehlzeit', 'Nachdruck', 'Schmelzetemperatur', 'Wandtemperatur', 'Volumenstrom']
    output_attrs = ['Max_Deformation']
    task_id_attr = 'Legobaustein'
    
    
    df = pd.read_csv(csv_path, sep=csv_sep)
    # TODO Split According to Legobaustein 
    

    if taskwise:
        train_data_tasks=[]
        train_label_tasks=[]
        test_data_tasks=[]
        test_label_tasks=[]
        for taskid in np.unique(df[task_id_attr]):
            saver= df[df[task_id_attr]==taskid]
            X_sp, X_tsp, y_sp, y_tsp=train_test_split(saver[input_attrs].values,saver[output_attrs].values,random_state=0,train_size=0.8)
            train_data_tasks.append(X_sp)
            train_label_tasks.append(y_sp)
            test_data_tasks.append(X_tsp)
            test_label_tasks.append(y_tsp)
    

        return train_data_tasks,train_label_tasks,test_data_tasks,test_label_tasks, np.concatenate(test_data_tasks), np.concatenate(test_label_tasks)
    #else: 
    #    X_train, X_test, y_train, y_test=train_test_split(df[input_attrs],df[output_attrs],random_state=0,train_size=0.8)
    #    return X_train,y_train, X_test,y_test,_,_,_ # TODO 



def continous_timeseries_loader(root,informative_feature_type='' ,binary=False, cnn_mode=False, taskwise=True):

    # WHICH SIMULATION TYPES TO BE USED (I NEED 5)
    train_data=None
    train_data_tasks=[]
    test_data=None
    test_data_tasks=[]
    train_label=None
    train_label_tasks=[]
    test_label=None
    test_label_tasks=[]
    train_meta_data=None
    test_meta_data=None
    label = 0

    for process in ['Box','NARMA','Harmonic','PseudoPeriodic','AutoRegressive']:
        name=f'SimulatedTrainingData_Middle_{process}_F_1_TS_50.npy'
        name= f'{root}/Data/data/TimeSeries/Testing/data/{name}'
        if 'Testing' in name:
            name= name.replace('Testing', 'Training')
        p1=name.replace('SimulatedTrainingData','SimulatedTrainingMetaData')
    
        train_x=np.load(f'{name}')
        if cnn_mode:
            train_x = np.swapaxes(train_x,-1,-2)
        train_y=np.load(f'{p1}')[:,0]
        #TODO THIS NEEDS TO BE SWITCHED TOOO 
        meta=np.load(f'{p1}')
    
        if 'Training' in name:
            name= name.replace('Training', 'Testing')
        p2=name.replace('SimulatedTestingData','SimulatedTestingMetaData')
        test_x=np.load(f'{name}')
        if cnn_mode:
            test_x = np.swapaxes(test_x,-1,-2)
    
        test_y=np.load(f'{p2}')[:,0]
        test_meta=np.load(f'{p2}')

        #Y Manipulation
        if label != 0:
            train_y[train_y==0]=label 
            train_y[train_y==1]=(label+1)
            meta[meta[:,0]==0,0]= label
            meta[meta[:,0]==1,0]= (label+1)
            test_y[test_y==0]=label 
            test_y[test_y==1]=(label+1)
            test_meta[test_meta[:,0]==0,0]= label
            test_meta[test_meta[:,0]==1,0]= (label+1)
        #print('train_y',np.unique(train_y))
        #print('test_y', np.unique(test_y))
       

        #TASKWISE DAZA

        train_data_tasks.append(train_x)
        test_data_tasks.append(test_x)
        train_label_tasks.append(train_y)
        test_label_tasks.append(test_y)



        #FULL DATA
        if train_data is None: 
            train_data=train_x
            test_data=test_x
            train_label= train_y
            test_label=test_y
            train_meta_data=meta
            test_meta_data=test_meta
        else:
            train_data=np.concatenate((train_data,train_x))
            test_data=np.concatenate((test_data,test_x))
            train_label= np.concatenate((train_label,train_y))
            test_label=np.concatenate((test_label,test_y))
            
            train_meta_data= np.concatenate((train_meta_data,meta))
            test_meta_data=np.concatenate((test_meta_data,test_meta))
        

        label +=2
 
    simulate=timeseries(meta= np.concatenate((train_meta_data,test_meta_data)),input=np.concatenate((train_data,test_data)))
    if taskwise:
        
        return train_data_tasks,train_label_tasks,test_data_tasks,test_label_tasks, test_data, test_label,simulate
    else: 
        #print(np.unique(train_label))
        X_train, X_test, y_train, y_test, meta_train, meta_test=train_test_split(train_data,train_label,train_meta_data,random_state=0,shuffle=True)
        #print(np.unique(y_train))
        #print(np.unique(y_test))
        #print(np.unique(test_label))
        return X_train,y_train, X_test,y_test,test_data,test_label,None, None, None, simulate



    
def timeseries_loader(root,name, cnn_mode= True):
    name= f'{root}/Data/data/TimeSeries/Testing/data/{name}'
    if 'Testing' in name:
        name= name.replace('Testing', 'Training')
    p1=name.replace('SimulatedTrainingData','SimulatedTrainingMetaData')
    
    train_x=np.load(f'{name}')
    if cnn_mode:
        train_x = np.swapaxes(train_x,-1,-2)
    train_y=np.load(f'{p1}')[:,0]
    #TODO THIS NEEDS TO BE SWITCHED TOOO 
    meta=np.load(f'{p1}')
    
    if 'Training' in name:
        name= name.replace('Training', 'Testing')
    p2=name.replace('SimulatedTestingData','SimulatedTestingMetaData')
    test_x=np.load(f'{name}')
    if cnn_mode:
        test_x = np.swapaxes(test_x,-1,-2)
    
    test_y=np.load(f'{p2}')[:,0]
    test_meta=np.load(f'{p2}')
    X_train, X_test, y_train, y_test, meta_train, meta_test=train_test_split(train_x,train_y,meta,random_state=0)
    simulate=timeseries(meta= np.concatenate((meta,test_meta)),input=np.concatenate((train_x,test_x)))
    return X_train,y_train, X_test,y_test,test_x,test_y, meta_train, meta_test, test_meta, simulate

def synthetic_tabular_classification(n_samples=10000, n_features=15, n_informative=10, n_redundant=0,n_classes=10,taskwise=False,flip_y=0,class_mode=False, **kwargs):
    #n_samples=10000, n_features=15, n_informative=10, n_redundant=0,n_classes=10,taskwise=False
    '''
    Without shuffling, X horizontally stacks features in the following order: 
    the primary n_informative features, followed by n_redundant linear combinations of the informative features, 
    followed by n_repeated duplicates, drawn randomly with replacement from the informative and redundant features. +
    The remaining features are filled with random noise. Thus, without shuffling, all useful features are contained in the columns X[:, :n_informative + n_redundant + n_repeated].

    The algorithm is adapted from Guyon [1] and was designed to generate the "Madelon" dataset.

    '''
    if not os.path.isdir(f'./CXIL/Data/data/simulated_tabular/{n_features}_{n_informative}_{n_redundant}_{n_classes}_train_x.npy'):

        X,y= make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant,n_classes=n_classes,shuffle=False,flip_y=flip_y, **kwargs)
        X_train_full, X_test, y_train_full, y_test=train_test_split(X,y,random_state=0)
        with open(f'./CXIL/Data/data/simulated_tabular/{n_features}_{n_informative}_{n_redundant}_{n_classes}_{flip_y}_train_x.npy', 'wb') as f:
            np.save(f,X_train_full)

        with open(f'./CXIL/Data/data/simulated_tabular/{n_features}_{n_informative}_{n_redundant}_{n_classes}_{flip_y}_train_y.npy', 'wb') as f:
            np.save(f,y_train_full)

        with open(f'./CXIL/Data/data/simulated_tabular/{n_features}_{n_informative}_{n_redundant}_{n_classes}_{flip_y}_test_x.npy', 'wb') as f:
            np.save(f,X_test)
            
        with open(f'./CXIL/Data/data/simulated_tabular/{n_features}_{n_informative}_{n_redundant}_{n_classes}_{flip_y}_test_y.npy', 'wb') as f:
            np.save(f,y_test)


    else: 

        X_train_full=np.load (f'./CXIL/Data/data/simulated_tabular/{n_features}_{n_informative}_{n_redundant}_{n_classes}_{flip_y}_train_x.npy')
        y_train_full= np.load (f'./CXIL/Data/data/simulated_tabular/{n_features}_{n_informative}_{n_redundant}_{n_classes}_{flip_y}_train_y.npy')
        X_test=np.load (f'./CXIL/Data/data/simulated_tabular/{n_features}_{n_informative}_{n_redundant}_{n_classes}_{flip_y}_test_x.npy')
        y_test=np.load (f'./CXIL/Data/data/simulated_tabular/{n_features}_{n_informative}_{n_redundant}_{n_classes}_{flip_y}_test_y.npy')
        print('UNIQUE LABELS', np.unique(y_test))
    X_f =np.concatenate((X_test,X_train_full))
    y_f =np.concatenate((y_test,y_train_full))
    if not taskwise:
        X_train, X_val, y_train, y_val=train_test_split(X_train_full,y_train_full,random_state=0)
        
        return X_train,y_train, X_val,y_val, X_test,y_test,synthetic_tabular(n_informative, n_redundant,data=(X_f,y_f),n_classes=n_classes,class_mode=class_mode)
    elif taskwise: 
        X=X_train_full
        y=y_train_full
        Xt=X_test
        yt=y_test
        X_0 = X[np.where((y==0) | (y==1))]
        y_0 = y[np.where((y==0) | (y==1))]
        X_1 = X[np.where((y==2) | (y==3))]
        y_1 = y[np.where((y==2) | (y==3))]
        X_2 = X[np.where((y==4) | (y==5))]
        y_2 = y[np.where((y==4) | (y==5))]
        X_3 = X[np.where((y==6) | (y==7))]
        y_3 = y[np.where((y==6) | (y==7))]
        X_4 = X[np.where((y==8) | (y==9))]
        y_4 = y[np.where((y==8) | (y==9))]
        Xt_0 = Xt[np.where((yt==0) | (yt==1))]
        yt_0 = yt[np.where((yt==0) | (yt==1))]
        Xt_1 = Xt[np.where((yt==2) | (yt==3))]
        yt_1 = yt[np.where((yt==2) | (yt==3))]
        Xt_2 = Xt[np.where((yt==4) | (yt==5))]
        yt_2 = yt[np.where((yt==4) | (yt==5))]
        Xt_3 = Xt[np.where((yt==6) | (yt==7))]
        yt_3 = yt[np.where((yt==6) | (yt==7))]
        Xt_4 = Xt[np.where((yt==8) | (yt==9))]
        yt_4 = yt[np.where((yt==8) | (yt==9))]
        print('X',X_0.shape)
        print('y',y_0.shape)
                
        return [X_0,X_1,X_2,X_3,X_4],[y_0,y_1,y_2,y_3,y_4],[Xt_0,Xt_1,Xt_2,Xt_3,Xt_4],[yt_0,yt_1,yt_2,yt_3,yt_4], Xt, yt,synthetic_tabular(n_informative, n_redundant,data=(X_f,y_f),n_classes=n_classes,class_mode=class_mode)
    else: 
        return X_train_full,y_train_full, X_test,y_test,synthetic_tabular(n_informative, n_redundant,data=(X_f,y_f),n_classes=n_classes,class_mode=class_mode)


def load_data_and_sim(name='toy_classification',cnn_mode=False,**kwargs):
    '''
    Loads the available dataset with rules: 
    Attributes: 
        name str: name of the dataset
    Returns: 
        (np.array,np.array,np.array,np.array,np.array,np.array,func): Data Split into X,y, X_train,y_train, X_test,y_test, Rule
    '''
    root=Path(__file__).parent.parent
    print(root)

    if name == 'toy_classification':
        data =np.load (f'{root}/Data/data/new_toy-colors.npz')
     
        if cnn_mode:
            #np.transpose(X[2].reshape(1,3,5,5), (0,2, 3, 1))
            X=np.transpose(data['arr_0'].reshape(-1,5,5,3), (0,2, 3, 1))#.reshape(-1,3,5,5)
            Xt=np.transpose(data['arr_1'].reshape(-1,5,5,3), (0,2, 3, 1))
        else: 
            X=data['arr_0'].reshape(-1,5,5,3)
            Xt=data['arr_1'].reshape(-1,5,5,3)
        y=data['arr_2']
        yt=data['arr_3']
        X_train, X_test, y_train, y_test=train_test_split(Xt,yt,random_state=0)
        return X,y, X_train,y_train, X_test,y_test, toy_simulation(cnn_mode)
    #elif name == 'diabetis':
    #    return diabetis_regression_data()
    elif name == 'decoy_toy':
        # Also has values for decoy E if thos is necessary at one point
        data =np.load (f'{root}/Data/data/decoy-toy-colors.npz')
        for a in data.keys():
            print(a)
        X=data['arr_0']
        Xt=data['arr_1']
        y=data['arr_2']
        yt=data['arr_3']
        #X_train, X_test, y_train, y_test=train_test_split(Xt,yt,random_state=0)
        return X,y, Xt,y, toy_simulation()
    
    elif name == 'decoy_mnist':
        return decoy_mnist(root)
    elif name == 'split_decoy_mnist':
        return split_decoy_mnist()
    elif name == 'split_mnist':
        return split_mnist()
    elif name == 'split_decoy_mnist_binary':
        #TODO
        return split_decoy_mnist_binary()
    elif name == 'split_mnist_binary':
        #TODO
        return split_mnist_binary()
    elif name == 'mnist':
        return mnist()
    elif name == 'unsplit_mnist':
        return unsplit_mnist()
    elif name =='iris_cancer':
        return iris_cancer()
    elif'continous_tabular' in name: 
        print('Continous Tabular')
        return synthetic_tabular_classification(n_samples=10000, n_features=15, n_informative=10, n_redundant=0,n_classes=10,taskwise=True)
    elif'tabular' in name: 
        return synthetic_tabular_classification(n_classes=10,**kwargs)
    elif name=='time10':
        return continous_timeseries_loader(root, taskwise=False,**kwargs)

    elif name =='continous_time':
        return continous_timeseries_loader(root, taskwise=True,**kwargs)
        
    elif 'Simulated' in name:
        return timeseries_loader(root,name,**kwargs)
    elif 'clevr' in name:
        return clevr_xai()
        #Continous Time Series 
    elif 'RBF' in name:
        # Concept Drift
        pass
    elif 'lego' in name: 
        return load_lego(root)


    # INDUSTRIAL DATASETS 
    # https://github.com/siemens/industrialbenchmark/tree/master
    #https://paperswithcode.com/dataset/mujoco
    #Exathlon: A benchmark for explainable anomaly detection over time series
        