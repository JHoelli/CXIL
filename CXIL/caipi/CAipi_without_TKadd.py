#TODO orientate at scikit bzw river 
from email.mime import image
from XIL.caipi import utils, Transformer
import numpy as np 
from XIL.caipi.utils import mod_wrapper
from sklearn.metrics import precision_recall_fscore_support,classification_report
from sklearn.metrics import precision_recall_fscore_support,classification_report, accuracy_score, f1_score , recall_score, precision_score,log_loss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
#from lime import lime_tabular
from captum.attr import visualization as viz
from XIL.XIL import XILInterface
import time

import torch

#TODO MAYBE Put SOme Stuff in Interface 
from captum.attr import InputXGradient
class caipi(XILInterface):
    def __init__(self, model,learn,gradient_method=InputXGradient,evaluate_data=(None,None), simulation_logic = None,silent=0, train_data=(None,None), transformer='', data_type = 'img') -> None:
        '''
        model: fit function
        predict: predict function
        explain: explain function
        '''
        super(caipi, self).__init__(model,learn, gradient_method, evaluate_data, simulation_logic,silent,data_type)
        self.explainer=self.gradient_method(self.model)
        self.transformer=transformer
        self.x_train, self.y_train= train_data
        #self.data_type=data_type

    def _new_instance_interpretability(self):

        self.explainer=self.gradient_method(self.model)
        self.transformer.predict=self.model
    
    def get_user_feedback(self,x, y, exp,shape,data_type):
        #print('get user Feedback')
        exp=exp.reshape(-1)

        if self.simulation_logic is not None:
            if self.silent ==2:
                plt.show()
            else:
                plt.close()

            return self.simulation_logic(x,np.argmax(y),exp)
        else: 
            from XIL.Interactive.BasicUi  import MainWindow   
            import sys 
            from PySide2.QtWidgets import QApplication
            if not QApplication.instance():
                app = QApplication(sys.argv)
            else:
                app = QApplication.instance()
            w = MainWindow(x.reshape(shape),y,exp,data_type)
            w.show()
            app.exec_()
            return w.data       




    def iterate(self,data_iterate,y_iterate=None,taskid=None):
        '''#TODO ELiminate X_train write Wrapper for Explainer'''
        trans= self.transformer
        #model_predict=self.model
        acc=[]
        f1=[]
        precision=[]
        recall=[]
        loss_calc_train=[]
        loss_calc_test=[]
        shape=data_iterate.shape[1:]
        #print('Shape',shape)


        for j, i in enumerate(data_iterate):
            #print('  correcting {:3d} / {:3d}'.format(j + 1, len(data_iterate)))
            x = np.array(i)[np.newaxis,:]
            #print(x.shape)
            pred_y = self.model(torch.from_numpy(x).float()).detach().numpy()[0]
            #print('model 3', self.model)
            trans.update_model(self.model)
            self._new_instance_interpretability()
            x=torch.from_numpy(x).float()#.reshape(1,-1)
            pred_expl=self.explainer.attribute(x,target=int(np.argmax(pred_y))).detach().numpy()
            #TODO IS THIS RIGHT ?
            label= [int(np.argmax(pred_y))]
            x=x.detach().numpy()
            la,feat= self.get_user_feedback(x,pred_y,pred_expl,shape,self.data_type)
       
            #if np.all(feat==0):
            #    pass
            #else:
            x=torch.from_numpy(x)
            d = trans.correction( x, pred_y,la, feat)

            x, label= d
            
            if len(np.array(label).shape)==2:
                label=  np.argmax(label, axis=1)

            if self.x is not None and self.silent >1:
                y_pred = self.model(torch.from_numpy(self.x).float())
                print(classification_report(self.y , np.argmax(y_pred,axis=1)))

            training_data=TensorDataset(torch.tensor(x).float(),torch.tensor(label))
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            #print('model 1', self.model)
            self.model=self.learn.fit(train_dataloader,taskid=taskid)
            #print('model 2',self.model)
            #wrap= mod_wrapper(self.model)
            #model_predict=wrap.predict
            y_pred_tensor = self.model(torch.from_numpy(self.x).float())
            y_pred=y_pred_tensor.detach().numpy()
            if self.x is not None and self.silent >1:
                print(classification_report(self.y , np.argmax(y_pred,axis=1)))
            if self.silent >0:
            # TODO log loss !  
                self.model.eval()
                #start=time.time()
                #print('Start Loss Calc Train')
                if y_iterate is not None:
  
                    if not torch.is_tensor(x): 
                        x= torch.from_numpy(np.array(x))
                        print(x.shape)
                    pred_train=self.model(x.float())
                    #print(pred_train)
                    #print(pred_train.shape)
                    #print(la)
                    #print(la.shape)
                    if type(la) is not list:
                        la=[la]
                    loss_calc_train.append(log_loss(la,pred_train.float().detach().numpy(), labels=np.arange(y_pred.shape[-1])))
            #pred_test= self.model(self.x)
                #print('Start Loss Calc Test')
                loss_calc_test.append(log_loss(self.y,y_pred, labels=np.arange(y_pred.shape[-1])))
                self.model.train()
            #print('END Loss')
            #end=time.time()
            #print(end-start)
            acc.append(accuracy_score(self.y , np.argmax(y_pred,axis=1)))
            f1.append(f1_score(self.y , np.argmax(y_pred,axis=1), average='macro'))
            precision.append(precision_score(self.y , np.argmax(y_pred,axis=1), average='macro'))
            recall.append(recall_score(self.y , np.argmax(y_pred,axis=1), average='macro'))
        return acc,f1, precision, recall, loss_calc_train, loss_calc_test
          

