from abc import ABC, abstractmethod

class CXILInterface(ABC):
    def __init__(self, model,learn,gradient_method,evaluate_data=(None,None), simulation_logic = None,silent=0,data_type='img'):
        '''
        model: fit function
        predict: predict function
        explain: explain function
        '''
        self.model = model
        self.gradient_method=gradient_method
        self.x, self.y= evaluate_data 
        self.learn=learn
        self.simulation_logic=simulation_logic
        self.silent=silent
        self.data_type=data_type

    @abstractmethod
    def get_user_feedback(self,x, y, exp):
        '''
        
        Gets User Feedback for current instance and explanation, either done by simulation or UI.

        '''
        pass

    @abstractmethod
    def iterate(self, data_iterate):
        '''
        Iterates and learns from data given. Can be one or multiple samples. 
        Attributes:
            data_iterate Tuple: ()
        '''
        pass