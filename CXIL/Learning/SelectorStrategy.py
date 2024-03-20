import numpy as np 
import random
#from contextlib import contextmanager
import torch 
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from collections import Counter
#from torchvision.transforms import Lambda

#def dataset_transforms(dataset, transform_to_change):
#    if isinstance(dataset, ConcatDataset):
#        r = []
#        for ds in dataset.datasets:
#            r += dataset_transforms(ds, transform_to_change)
#        return r
#    else:
#        old_transform = dataset.transform
#        dataset.transform = transform_to_change
#        return [(dataset, old_transform)]
#@contextmanager
#def override_dataset_transform(dataset, transform):
#    try:
#        datasets_with_orig_transform = dataset_transforms(dataset, transform)
#        yield dataset
#    finally:
#        # get bac original transformations
#        for ds, orig_transform in datasets_with_orig_transform:
#            ds.transform = orig_transform

class ExemplarsSelector:
    """Exemplar selector for approaches with an interface of Dataset"""

    def __init__(self, exemplars_dataset):
        self.exemplars_dataset = exemplars_dataset

    def __call__(self,task_cls, max_num_examplars,trn_loader,model= None):
        exemplars_per_class = self._exemplars_per_class_num(task_cls, max_num_examplars)
        #with override_dataset_transform(trn_loader.dataset) as ds_for_selection:
            # change loader and fix to go sequentially (shuffle=False), keeps same order for later, eval transforms
        ds_for_selection=trn_loader
        #sel_loader = DataLoader(ds_for_selection, batch_size=trn_loader.batch_size, shuffle=False,
        #                            num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)

        selected_indices= self._select_indices( task_cls,ds_for_selection, exemplars_per_class,model)
        #print('selected indices ',len(selected_indices))
       # 
        #with override_dataset_transform(trn_loader.dataset, Lambda(lambda x: np.array(x))) as ds_for_raw:
        ds_for_raw= ds_for_selection.dataset#sel_loader.dataset#trn_loader.dataset
            #print('SelectedIndicies',selected_indices)
            #print('ds_for_RAW',ds_for_raw)
        #ds_for_raw_x=[np.array(x) for x,_ in trn_loader.dataset]
        #ds_for_raw_y=[np.array(y) for _,y in trn_loader.dataset]
        #i=0
        #a=[]
        #for k in selected_indices_dict.keys():
        #    for h in selected_indices_dict[k]:
        #        a,b=ds_for_raw[h]#ds_for_selection[h]#sel_loader.dataset[h]#ds_for_selection[h]#sel_loader[h]#ds_for_raw[h]
        #        if int(b) != int(k):
        #            print('ERROR', h)
        #        if h in selected_indices:
        #            pass
        #        else: 
        #            print('THIS NUMBER DOES NOT EXIST ',h)

        #    if i== 0:
        #        x_1, y_1 = zip(*((ds_for_raw[idx]) for idx in selected_indices_dict[k]))
        #        i=1
        #        print(type(x_1))
        #        print(f'WRONG INDICIES {k} ',np.where(y_1!=int(k)))
        #    else:
        #        x_2,y_2=zip(*((ds_for_raw[idx]) for idx in selected_indices_dict[k]))
        #        x_1.append(x_2)
        #        y_1.append(y_2)

        #        print(f'WRONG INDICIES {k} ',np.where(y_2!=int(k)))

        #x=x_1
        #y=y_1
        x, y = zip(*((ds_for_raw[idx]) for idx in selected_indices))

        #print(len(x))
        #x=[torch.from_numpy(i) for i in x]#torch.from_numpy(x)
        #y=[torch.from_numpy(i) for i in y]
        #print(y)
        #print('Y',Counter(list(y)))
        d=[a.detach().numpy().tolist() for a in y]
        #print(y)
        print('RIGH AFTER SELECTION ',Counter(d))
        return x, y

    def _exemplars_per_class_num(self, task_cls, max_num_examplars):
        if self.exemplars_dataset.max_num_exemplars_per_class:
            return self.exemplars_dataset.max_num_exemplars_per_class

        num_cls = task_cls.sum().item()
        num_exemplars =  max_num_examplars
        exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplars_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. " \
            "Limit of exemplars: {}".format(num_cls,
                                            num_exemplars)
        return exemplars_per_class

    def _select_indices(self, task_cls, sel_loader, exemplars_per_class: int):
        pass


class RandomExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on random selection, which produces a random list of samples."""

    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, task_cls, sel_loader, exemplars_per_class: int,model=None):
        '''
        Only returns indices to be used . 
        task_cls np.array: list of seen classes
        sel_loader torch.Dataloader: Data
        exemplars_per_class int : How many samples per class 
        '''
        #print('Examples per Class',exemplars_per_class)
        #print(self.exemplars_dataset.labels)
        #la=[l for _,l in sel_loader.dataset ]
        #print(la)
        #print('TRN Loader3 ',np.unique(np.concatenate(la)))
        #print('task_cls',task_cls)
        num_cls = sum(task_cls)
        result = []
        result2={}
        labels = self._get_labels(sel_loader)
        #print('Beginning Labeks',np.unique(self.exemplars_dataset.labels))
        #print('Labeks',np.unique(labels))
        for curr_cls in task_cls: #range(num_cls):

            #curr_cls=int(curr_cls)
            # get all indices from current class -- check if there are exemplars from previous task in the loader
           # print('task_cls',task_cls)
           # print('labels',labels)
            #print('current_cls',curr_cls)#

            cls_ind = np.where(labels == int(curr_cls))[0]
            #print (cls_ind)
            #print(f'number_items ',len(cls_ind))
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(int(curr_cls))
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # select the exemplars randomly
            if (exemplars_per_class <= len(cls_ind)):
                #print('Exmaplars per class ',exemplars_per_class)
                #print('LENGTH ',len(list(cls_ind)))
                #print(random.sample(list(cls_ind),exemplars_per_class))
                save= random.sample(list(cls_ind), exemplars_per_class)
                result.extend(save)
                result2[curr_cls]=save
                #print('RES ',len(result))

        #print('End Labeks',np.unique(self.exemplars_dataset.labels))
        #da=[l.detach().numpy()for l in self.exemplars_dataset.labels]
        #print('Counterb ',Counter(da))
        #TODO RECHECK RESULT 

        #for k in result2.keys():
        #    print(f'{k}:', len(result2[k]))
        return result#,result2

    def _get_labels(self, sel_loader):
        i=0
        if hasattr(sel_loader.dataset, 'labels'):  # BaseDataset, MemoryDataset
            #print('A')
            labels = np.asarray(sel_loader.dataset.labels)
        elif isinstance(sel_loader.dataset, ConcatDataset):
            #print('IS INSTANCE')
            labels = []
            for ds in sel_loader.dataset.datasets:
                #print(ds)
                for x, y in ds:
                
                    labels.append(y)
            labels = np.array(labels)
            #la=[l for _,l in sel_loader.dataset ]


            #labels = []
            #for ds in sel_loader.dataset.datasets:
            #    labels.extend(ds.labels)
            #labels=np.concatenate(la)
            #print(np.unique(labels))
            #labels = np.array(labels)
        elif isinstance(sel_loader.dataset,DataLoader):
            #print('IS INSTANCE')
            labels = []
            for _,ds in sel_loader.dataset:
                
                labels.extend(ds)
            labels = np.array(labels)
        elif isinstance(sel_loader.dataset,TensorDataset):
            #print('IS INSTANCE')
            labels = []
            for _,ds in sel_loader.dataset:
                #print(ds)
                labels.append(ds)
            labels = np.array(labels)
        else:
             raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))
             #print('IS INSTANCE')
            #la=[l for _,l in sel_loader.dataset ]
            #labels=np.concatenate(la)
            #print(np.unique(labels))
            #labels = np.array(labels)
            #print(np.unique(labels))
        #print('Label COunter',Counter(labels))
        return labels


class HerdingExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
    class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
    """
    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, task_cls, sel_loader, exemplars_per_class: int, model= None) :
        model_device = next(model.parameters()).device  # we assume here that whole model is on a single device

        # extract outputs from the model for all train samples
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                feats = model(images.to(model_device), return_features=True)[1]
                feats = feats / feats.norm(dim=1).view(-1, 1)  # Feature normalization
                extracted_features.append(feats)
                extracted_targets.extend(targets)
        extracted_features = (torch.cat(extracted_features)).cpu()
        extracted_targets = np.array(extracted_targets)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_feats = extracted_features[cls_ind]
            # calculate the mean
            cls_mu = cls_feats.mean(0)
            # select the exemplars closer to the mean of each class
            selected = []
            selected_feat = []
            for k in range(exemplars_per_class):
                # fix this to the dimension of the model features
                sum_others = torch.zeros(cls_feats.shape[1])
                for j in selected_feat:
                    sum_others += j / (k + 1)
                dist_min = np.inf
                # choose the closest to the mean of the current class
                for item in cls_ind:
                    if item not in selected:
                        feat = extracted_features[item]
                        dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)
                        if dist < dist_min:
                            dist_min = dist
                            newone = item
                            newonefeat = feat
                selected_feat.append(newonefeat)
                selected.append(newone)
            result.extend(selected)
        return result


class EntropyExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on entropy selection, which produces a sorted list of samples of one
    class based on entropy of each sample. From RWalk http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model, sel_loader, exemplars_per_class: int):
        model_device = next(model.parameters()).device  # we assume here that whole model is on a single device

        # extract outputs from the model for all train samples
        extracted_logits = []
        extracted_targets = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                extracted_logits.append(torch.cat(model(images.to(model_device)), dim=1))
                extracted_targets.extend(targets)
        extracted_logits = (torch.cat(extracted_logits)).cpu()
        extracted_targets = np.array(extracted_targets)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_logits = extracted_logits[cls_ind]
            # select the exemplars with higher entropy (lower: -entropy)
            probs = torch.softmax(cls_logits, dim=1)
            log_probs = torch.log(probs)
            minus_entropy = (probs * log_probs).sum(1)  # change sign of this variable for inverse order
            selected = cls_ind[minus_entropy.sort()[1][:exemplars_per_class]]
            result.extend(selected)
        return result
    
    
class DistanceExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on distance-based selection, which produces a sorted list of samples of
    one class based on closeness to decision boundary of each sample. From RWalk
    http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model, sel_loader, exemplars_per_class: int,
                        transform):
        model_device = next(model.parameters()).device  # we assume here that whole model is on a single device

        # extract outputs from the model for all train samples
        extracted_logits = []
        extracted_targets = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                extracted_logits.append(torch.cat(model(images.to(model_device)), dim=1))
                extracted_targets.extend(targets)
        extracted_logits = (torch.cat(extracted_logits)).cpu()
        extracted_targets = np.array(extracted_targets)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_logits = extracted_logits[cls_ind]
            # select the exemplars closer to boundary
            distance = cls_logits[:, curr_cls]  # change sign of this variable for inverse order
            selected = cls_ind[distance.sort()[1][:exemplars_per_class]]
            result.extend(selected)
        return result
