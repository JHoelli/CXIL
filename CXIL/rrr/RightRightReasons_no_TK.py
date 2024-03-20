from __future__ import annotations
import imp
import numpy as np 
from sklearn.metrics import precision_recall_fscore_support,classification_report, accuracy_score, f1_score , recall_score, precision_score
from captum.attr import InputXGradient
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from CXIL.CXIL import CXILInterface
from CXIL.Learning.LearnerStep import RRR_Learner
import time
import pandas as pd

class RRR(CXILInterface):

    def __init__(self, model, learn,gradient_method=InputXGradient, test_data=(None, None), simulation_logic= None,only_interact_current=False,silent=2) -> None:
        super(RRR, self).__init__(model,learn, gradient_method, test_data, simulation_logic,silent)
        if self.x is not None: 
            self.x=torch.from_numpy(self.x).float()
        self.input_gradients=self.gradient_method(self.model)
        self.best_model=model
        self.only_interact_current = only_interact_current
        self.f1_prev=0
    
    def _new_instance_interpretability(self):
        try:
            self.learn.loss_fn.input_gradients=self.gradient_method(self.model)
            self.learn.loss_fn.model=self.model
        except:
            self.learn.learner.loss_fn.input_gradients=self.gradient_method(self.model)
            self.learn.learner.loss_fn.model=self.model
    
    def get_user_feedback(self,x, y, exp,shape,data_type):
        exp=exp.reshape(-1)

        if self.simulation_logic is not None:
            if self.silent ==2:
                plt.show()
            else:
                plt.close()

            return self.simulation_logic(x,np.argmax(y),exp)
        else: 
            from CXIL.Interactive.BasicUi  import MainWindow   
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

    def iterate(self,data,y, taskid= None):
        y_pred = self.model(self.x).detach().numpy()
        self.f1_prev=f1_score(self.y , np.argmax(y_pred,axis=1), average='macro')
        acc=[]
        f1=[]
        precision=[]
        recall=[]
        time_list=[]
        loss_calc_train=[]
        loss_calc_test=[]
        for j,i in enumerate(data.tolist()):
            if self.silent >0:
                start= time.time()
            if self.silent ==2:
                print('  correcting {:3d} / {:3d}'.format(j + 1, len(data)))
            #print('Beginning Shape ', np.array(i).shape)
            self.model.train()
            training_data=TensorDataset(torch.tensor([i]).float(),torch.tensor([y[j]]))
            #print('End Tensor')
            train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
            self.model = self.learn.fit(train_dataloader, taskid=taskid)
            self._new_instance_interpretability()
            self.model.eval()
            y_pred = self.model(self.x).detach().numpy()
            
            if self.x is not None and self.silent >=1:
                print(classification_report(self.y , np.argmax(y_pred,axis=1)))


            acc.append(accuracy_score(self.y , np.argmax(y_pred,axis=1)))
            f1_prev=f1_score(self.y , np.argmax(y_pred,axis=1), average='macro')
            f1.append(f1_prev)
            precision.append(precision_score(self.y , np.argmax(y_pred,axis=1), average='macro'))
            recall.append(recall_score(self.y , np.argmax(y_pred,axis=1), average='macro'))
            if self.silent >0:
                print('Train')
                loss_calc_train.append(RRR_Learner(self.gradient_method, self.simulation_logic,self.model)(torch.from_numpy(data).float(),torch.from_numpy(y)).detach().numpy())
                if torch.is_tensor(self.x):
                    print('TEST1')
                    loss_calc_test.append(RRR_Learner(self.gradient_method, self.simulation_logic,self.model)(self.x,torch.from_numpy(self.y)).detach().numpy())
                else:
                    print('TEST2')
                    loss_calc_test.append(RRR_Learner(self.gradient_method, self.simulation_logic,self.model)(torch.from_numpy(self.x).float(),torch.from_numpy(self.y)).detach().numpy())

                end=time.time()
                time_list.append(end-start)

            #if j ==10: 

            #    import sys 
            #    sys.exit(1)
        print('loss_calc_test', loss_calc_test)
        print('loss_calc_train', loss_calc_train)
        return acc,f1, precision, recall, time_list,loss_calc_test,loss_calc_train