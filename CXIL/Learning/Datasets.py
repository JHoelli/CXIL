from CXIL.Learning.SelectorStrategy import  RandomExemplarsSelector, HerdingExemplarsSelector
import importlib
from torch import nn
from torch.utils.data import Dataset

class MemoryDataset(Dataset):
    '''TODO Rewrite this onto Buffer Stategy'''
    def __init__(self,   num_exemplars=10000, num_exemplars_per_class=1000, exemplar_selection= RandomExemplarsSelector):

        """Initialization"""
        self.labels = []
        self.images = []
        self.max_num_exemplars_per_class = num_exemplars_per_class
        self.max_num_exemplars = num_exemplars
        #assert (num_exemplars_per_class == 0) or (num_exemplars == 0), 'Cannot use both limits at once!'
        #cls_name = "{}ExemplarsSelector".format(exemplar_selection.capitalize())
        selector_cls = exemplar_selection
        self.exemplars_selector = selector_cls(self)
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)
    def _is_active(self):
        return self.max_num_exemplars_per_class > 0 or self.max_num_exemplars > 0

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index]
        y = self.labels[index]
        return x, y
   
    def collect_exemplars(self, task_cls, trn_loader):
        if self._is_active():
            #print('from Collect Examples',np.unique( trn_loader))
            #d=[a.detach().numpy().tolist() for _, a in  trn_loader]
            #print(d)
            #print(Counter(np.concatenate(d).tolist()))
            self.images, self.labels = self.exemplars_selector(task_cls,self.max_num_exemplars, trn_loader)
            #print('from Collect Examples',np.unique(self.labels))
            #d=[a.detach().numpy().tolist() for a in self.labels]
            #print(Counter(d))
            #for  i in np.unique(self.labels):
            #    print(f'FROM COLLECT EX {i}', len(d[d==i]))