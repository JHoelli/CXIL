import scipy as sp
import numpy as np
from pyDOE2 import lhs
from scipy.stats.distributions import norm
import warnings
import random
from numpy import vstack,hstack

class Transformer():
    def __init__(self,num_samples):
        self.num_samples=num_samples
        pass
    def transform():
        pass

    #Tabular Transformer
    def correction(self,x, pred_y,label, feat):
        num_samples=self.num_samples
        '''Used for query corrections'''
        '''Right for the right resons'''
        #TODO no correction
        '''
        If wrong for the wrong reasons
        Change Label 
        '''
        #print('x ', x.shape)
        #print('y ', pred_y)
        #print('label ', label)
        #print('feat ', feat)
        pred_org=pred_y
        if type(pred_y) is not int and type(pred_y) is not str:
            pred_org=pred_y
            pred_y=np.argmax(pred_y,axis=0)
        #print('label from correction', label)

        if label is not '' and label is not None:
            #print('LABEL TRUE')
            label_new= int(label)
            #print(pred_y)
            #print(label_new)
            if  pred_y != label:
                temp= np.zeros_like(pred_org)
                temp[label_new]=1    
                #print('temp', temp.shape)        
                return x, temp.reshape(-1, pred_org.shape[-1])
        '''If right for the wrong reasons'''
        #TODO THIS IS NEW
        feat = np.array(feat).reshape(-1)
        #print(feat)
        if np.any(feat==1):
            idx=np.argwhere(feat==1).reshape(-1)
            #print(idx)
            data= self.transform(x, pred_y, idx)
            return data
        print('THIS IS NOT SUPPOSED TO BE HAPPENING')

        #import sys
        #sys.exit(1)
        temp= np.zeros_like(pred_org)
        temp[int(label)]=1
        return x, temp.reshape(-1, pred_org.shape[-1])
    def update_model(self,predict):
        self.predict = predict

class MeanAndStdCalculation():
    '''
    Transform Tabular Explainer based on running Mean and STD, 
    implementation is inspired by Stream Processing. Welford's algorithm
    #TODO implement , how to do this label wise ? 
    #TODO make this column wise ! (use matrix operations/ numpy)
    '''
    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)

class caipi_correction(Transformer):
    '''Taken and Adapted from CAIPI Tabular Problem colors Problem '''
    def __init__(self,predict,lower_bounds, upper_bounds,num_samples):
        super().__init__(self, num_samples)
        self.predict=predict
        if lower_bounds is None:
             self.lower_bounds= -np.inf
        elif len(lower_bounds.shape)>1:
            self.lower_bounds=np.min(lower_bounds, axis=1)
        else:
            self.lower_bounds=lower_bounds
        
        if upper_bounds is None:
             self.upper_bounds= np.inf
        elif len(upper_bounds.shape)>1:
            self.upper_bounds=np.max(upper_bounds, axis=1)
        else:
            self.upper_bounds=upper_bounds



    def transform(self,x, y, feat):
        if feat is None or np.all(feat==0):
            return set()
        pred = self.predict(x)
        pred_y= np.argmax(pred,axis=1)[0]
        if y != np.argmax(pred,axis=1)[0]:
            return set()

        #z = self.Z[i]
        #TODO This Information is already included in feat --> feat uin our case are the selected indices to be changed !
        #true_feats = {feat for (feat, _) in self.z_to_expl(z)}
        #pred_feats = {feat for (feat, _) in pred_expl}

        ALL_VALUES = set(range(4))

        Z_corr = []
        #TODO reading mark , z =x ?
        for f in feat:
            lb, ub =self.lower_bounds[f], self.upper_bounds[f]
            #TODO indices 
            #r, c = feat.split(',')
            #r, c = int(r), int(c)
            other_values = {value for value in ALL_VALUES if not (lb < value <= ub)}
            for value in other_values:
                z_corr = np.array(x, copy=True)
                z_corr = value #[5*r+c]
                if self.z_to_y(z_corr) != pred_y:
                    continue
                #if tuple(z_corr) in X_test:
                #    continue
                Z_corr.append(z_corr)

        if not len(Z_corr):
            return set()

        X_corr = Z_corr #np.array([self.z_to_x(z_corr) for z_corr in Z_corr],
                 #         dtype=np.float64)
        # Idea is to make the item irrelevant from x values
        y_corr = y#np.array([pred_y for _ in Z_corr], dtype=np.int8)

        return X_corr, y_corr


class randomTransformer(Transformer):
    '''
    Random Number gerator between upper and lower Bound. 
    Generates for all Features Feats between the given bounds
    
    '''
    def __init__(self, model,upper_bound=255,lower_count=0,randomizser=np.random.randint,num_samples=1):
        super().__init__(num_samples)
        self.predict= model
        self.upper_limit=upper_bound
        self.lower_limit=lower_count
        self.randomizer=randomizser

    def transform(self, x, y, feat=None):
        '''
        Applies Data Augmenation.
        x: 
        y:
        feat: 
        '''
        num_samples=self.num_samples
        print('feat from correction ',feat)
        print(feat.shape)
        print(x.shape)
        print(y)
        if len(x)!= 1: 
            x=x[np.newaxis,:]
        x_org=x.shape
        x_org_type=x.reshape(-1)[0].dtype
        #print('y_org ',y_org)
        print('x_org ',x_org)
        print('x_org_type ',x_org_type)
        pred = self.predict(x)
        if y != np.argmax(pred,axis=1)[0]:
            y_new=np.zeros_like(pred)
            y_new[0][y]=1
            data_x=x.reshape(-1,*x_org[1:]).tolist()
            #print('RETURNS', np.array(data_x).shape)
            return data_x, y_new.reshape(len(pred),-1).tolist()

        upper_limit = self.upper_limit
        lower_limit=self.lower_limit
        data_x,data_y=[],[]
        for f in feat:
            #THIS IS NEW 
            sh=1
            for  i in x.shape[1:]:
                sh*= i
            new_x=  np.repeat(x.reshape(-1),num_samples).reshape(-1,sh)
          
            new_x[:,f]=self.randomizer(lower_limit, upper_limit, len(new_x))#.reshape(-1,*x_org[1:])
            #new_x = new_x.reshape(-1,*x_org[1:]).astype(x_org_type)
            print(new_x.shape)
            print(type(new_x))
            #pred = self.predict(new_x)
            #y=np.argmax(pred, axis=1)
            # TODO ONE HOT ENCODED ? 
            pred=np.zeros_like(pred)
            pred[:,y]=1
            print('y',y)
            print('pred', pred)
            if len(data_x)== 0:
                print('1',data_x)
                data_x=new_x
                data_y=pred
            else:
                print('2',data_x.shape)

                #print(y.shape)
                print('2',data_y.shape)

                #print(y)
                #data_x.extend(new_x)
                #data_y.extend(y)
                data_x=np.concatenate((data_x,new_x))
                data_y=np.concatenate((data_y,pred))                           
        data_x=data_x.reshape(-1,*x_org[1:]).tolist()
        data_y=data_y.reshape(-1,*pred.shape[1:])

        return data_x,data_y



class MeanAndStdTransformer_old(Transformer):
    
    '''
    Transform Tabular Explainer based on running Mean and STD, 
    implementation is inspired by Stream Processing. Welford's algorithm
    Assumption: Only one sample in transformer.
    #TODO How to cope with different data type (e.g. int vs float), categorial data
    #TODO does this make sense with respect to outlier ? 
    #TODO does this make sens ? --> rather use transformer from other classificator ? 
    '''
    def __init__(self, model,num_samples=1):
        super().__init__(num_samples)
        self.listofCalculators={}
        self.predict= model
        #self.num_samples=num_samples

    def transform(self, x, y, feat=None):
        #print('TRANSFORM')
        num_samples=self.num_samples
        x_org=x.shape
        #print('y_org ',y_org)
        #print('x_org ',x_org)
        pred = self.predict(x)
        if y != np.argmax(pred,axis=1)[0]:
            y_new=np.zeros_like(pred)
            y_new[0][y]=1
            data_x=x.reshape(-1,*x_org[1:]).tolist()
            #print('RETURNS', np.array(data_x).shape)
            return data_x, y_new.reshape(len(pred),-1).tolist()
        if y not in self.listofCalculators.keys():
            self.listofCalculators[str(y)]=MeanAndStdCalculation(ddof=0)
        self.listofCalculators[str(y)].include(x)
        if self.listofCalculators[str(y)].n<10:
            data_x=x.reshape(-1,*x_org[1:]).tolist()
            #print('RETURNS', np.array(data_x).shape)
            print('RETURNS Y 1', pred)
            return data_x  ,pred
        upper_limit = self.listofCalculators[str(y)].mean+self.listofCalculators[str(y)].std
        lower_limit=self.listofCalculators[str(y)].mean-self.listofCalculators[str(y)].std
        data_x,data_y=[],[]
        for f in feat: 
            new_x=  np.repeat(x.reshape(-1),num_samples).reshape(-1,x.shape[-1])
            #print(random.randint(lower_limit[0][f], upper_limit[0][f], len(new_x)))
            new_x[:,f]=np.random.randint(lower_limit[0][f], upper_limit[0][f], len(new_x))
            pred = self.predict(new_x)
            if len (data_x)== 0:
                data_x=new_x
                data_y=y
            else:
                data_x.extend(new_x)
                data_y.extend(y)
            
        data_x=x.reshape(-1,*x_org[1:]).tolist()
        data_y=data_y.reshape(-1,*pred.shape[1:])
        #print('RETURNS', np.array(data_x).shape)
        print('RETURNS Y 2', data_y)
        return data_x,data_y

    def update_model(self,predict):
        self.predict = predict

class MeanAndStdTransformer(Transformer):
    
    '''
    Transform Tabular Explainer based on running Mean and STD, 
    implementation is inspired by Stream Processing. Welford's algorithm
    Assumption: Only one sample in transformer.
    #TODO How to cope with different data type (e.g. int vs float), categorial data
    #TODO does this make sense with respect to outlier ? 
    #TODO does this make sens ? --> rather use transformer from other classificator ? 
    '''
    def __init__(self, model,num_samples=1):
        super().__init__(num_samples)
        self.listofCalculators={}
        self.predict= model
        #self.num_samples=num_samples

    def transform(self, x, y, feat=None):
        #print('TRANSFORM')
        num_samples=self.num_samples
        x_org=x.shape
        #print('y_org ',y_org)
        #print('x_org ',x_org)
        pred = self.predict(x).detach().numpy()
        if y != np.argmax(pred,axis=1)[0]:
            y_new=np.zeros_like(pred)
            y_new[0][y]=1
            data_x=x.reshape(-1,*x_org[1:]).tolist()
            #print('RETURNS', np.array(data_x).shape)
            return data_x, y_new.reshape(len(pred),-1).tolist()
        if y not in self.listofCalculators.keys():
            self.listofCalculators[str(y)]=MeanAndStdCalculation(ddof=0)
        self.listofCalculators[str(y)].include(x)
        if self.listofCalculators[str(y)].n<10:
            data_x=x.reshape(-1,*x_org[1:]).tolist()
            #print('RETURNS', np.array(data_x).shape)
            return data_x  ,pred
        upper_limit = self.listofCalculators[str(y)].mean+self.listofCalculators[str(y)].std
        lower_limit=self.listofCalculators[str(y)].mean-self.listofCalculators[str(y)].std
        data_x,data_y=[],[]
        for f in feat: 
            sh=1
            for  i in x.shape[1:]:
                sh*= i
            new_x=  np.repeat(x.reshape(-1),num_samples).reshape(-1,sh)
            #print(random.randint(lower_limit[0][f], upper_limit[0][f], len(new_x)))
            new_x[:,f]=np.random.randint(lower_limit[0][f], upper_limit[0][f], len(new_x))
            #pred = self.predict(new_x)
            pred=np.zeros_like(pred)
            pred[:,y]=1
            if len(data_x)== 0:
                print('1',data_x)
                data_x=new_x
                data_y=pred
            else:
                #print('2',data_x.shape)

                #print(y.shape)
                #print('2',data_y.shape)

                #print(y)
                #data_x.extend(new_x)
                #data_y.extend(y)
                data_x=np.concatenate((data_x,new_x))
                data_y=np.concatenate((data_y,pred)) 
            
        data_x=x.reshape(-1,*x_org[1:]).tolist()
        data_y=data_y.reshape(-1,*pred.shape[1:])
        #print('RETURNS', np.array(data_x).shape)
        #print('RETURNS', np.array(data_y.shape))
        return data_x,data_y

    def update_model(self,predict):
        self.predict = predict


class DataBasedTransformer(Transformer):
    '''
    Selects Data point from the original data and replaces perturbed instance.
    Differentiation between jth compontenent relevant & irrelvant. Currently only jth component irrelevanr
    #TODO differentiate between pandas and numpy 
    #TODO different foramts f 
    #TODO are counterexamples really relevant
    #TODO kann ich num_smple nicht vernahlässigen durch online learning 
    '''

    def __init__(self,X_train, y_train,predict, num_samples):
        super(). __init__(self, num_samples)
        self.x=X_train
        self.y=y_train
        self.predict=predict
        
        
    def transform(self, x, y, feat):
        num_samples=self.num_samples
        #print('go into transformer')
        data_x =[]
        data_y =[]
        index = np.argwhere(self.y == y)
        x_select =self.x[index]
        feat=feat.reshape(-1)
        pred = self.predict(x)
        if y != np.argmax(pred,axis=1)[0]:
            y_new=np.zeros_like(pred)
            y_new[0][y]=1
            return x.reshape(-1,np.array(x).shape[-1]).tolist(),y_new.tolist()
        #print('feat to be changed', feat)
        for f in feat:
           # print('Iteration')
            idx = np.random.randint(len(x_select), size=num_samples)
            selected= x_select[idx,:].reshape(-1,x_select.shape[-1])
            #TODO select random Instance
            #print(selected)
            new_x=  np.repeat(x.reshape(-1),num_samples).reshape(-1,x_select.shape[-1])
            
            fea=selected[:,f]
            #print(fea)
            new_x[:,f]=fea
            pred = self.predict(new_x)
            #print(pred.shape)
            labels = np.argmax(pred,axis=1)
            pred = np.zeros_like(pred)
            j=0
            for l in labels:
                pred[j,l]=1
                j=j+1
            #idx = np.argwhere(labels == int(y))
            #print(new_x)
            #print(labels)
            #print(y)
            if len (data_x)==0:
                #print('LIST IS EMPTY')
                data_x=new_x.reshape(-1,x_select.shape[-1]).tolist()
                #print(np.array(data_x).shape)
                data_y=pred.reshape(-1,pred.shape[-1]).tolist()
                #print(np.array(data_y).shape)

            else:
                #print('ELSE')
                data_x.extend(new_x.reshape(-1,x_select.shape[-1]).tolist())
                #print(np.array(data_x).shape)
                data_y.extend(pred.reshape(-1,pred.shape[-1]).tolist())
                #print(np.array(data_y).shape)
        #print('out',np.array(data_x).shape)
        #print('out',np.array(data_y).shape)
        if  len(data_x)==0:
            #print('from in here')
            data_x=x.reshape(-1,x_select.shape[-1]).tolist()
            #TODO think how to make this better, currently fixed 
            data_y=np.array([0,0])
            data_y[y]=1
            #ata_y=np.zeros_like(pred[-1])
            #data_y[y]=1
            data_y=data_y.reshape(1,2)
        #print('out',np.array(data_x).shape)
        #print('out',np.array(data_y).shape)

        return data_x,data_y
    def update_model(self,predict):
        self.predict = predict



class NumericalTransformer(Transformer):
    #TODO
    def init(self):
        #TODO for every feature scale ,mean and scale 
        self.scaler.scale_=1
        self.scaler.mean_=0
        self.discretizer=''
        pass
    def transform(self,data, num_samples, sampling_method):
        #TODO restrict to specific columns
        is_sparse = sp.sparse.issparse(data)
        if is_sparse:
            num_cols = data.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data.dtype)
        else:
            num_cols = data.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if self.discretizer is None:
            instance_sample = data
            scale = self.scaler.scale_
            mean = self.scaler.mean_
        if is_sparse:
            # Perturb only the non-zero values
            non_zero_indexes = data.nonzero()[1]
            num_cols = len(non_zero_indexes)
            instance_sample = data[:, non_zero_indexes]
            scale = scale[non_zero_indexes]
            mean = mean[non_zero_indexes]

        if sampling_method == 'gaussian':
            data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
            data = np.array(data)
        elif sampling_method == 'lhs':
            data = lhs(num_cols, samples=num_samples
                           ).reshape(num_samples, num_cols)
            means = np.zeros(num_cols)
            stdvs = np.array([1]*num_cols)
            for i in range(num_cols):
                data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
            data = np.array(data)
        else:
            warnings.warn('''Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.''', UserWarning)
            data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
            data = np.array(data)

        if self.sample_around_instance:
             data = data * scale + instance_sample
        else:
            data = data * scale + mean
        if is_sparse:
            if num_cols == 0:
                data = sp.sparse.csr_matrix((num_samples,
                                                 data.shape[1]),
                                                dtype=data.dtype)
            else:
                indexes = np.tile(non_zero_indexes, num_samples)
                indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                data_1d_shape = data.shape[0] * data.shape[1]
                data_1d = data.reshape(data_1d_shape)
                data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data.shape[1]))
            categorical_features = self.categorical_features
            first_row = data
        else:
            first_row = self.discretizer.discretize(data)
        data[0] = data.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data
        return data,
    def update_model(self,predict):
        self.predict = predict
        