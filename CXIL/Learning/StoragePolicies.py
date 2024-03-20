
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import numpy as np 
import random

'''
TODO Fix all the selection strategies.
* Confidence 
* Margin 
* Entropy 
* Baynesian Disagreement 
FROM https://link.springer.com/chapter/10.1007/978-3-031-17587-9_6 
 '''


'''THIS are the possible Selection Stragtefgies '''
class RandomExemplarsSelectionStrategy():
    """
    Select the exemplars at random in the dataset
    #TODO Implications does this make sense ? Only keeping the unsure ones ? ---> Probably not  
    """

    def make_sorted_indices( self, strategy, data  ) :
        indices = list(range(len(data)))
        random.shuffle(indices)
        return indices

class Oldest():
    """
    #TODO Implications does this make sense ? Only keeping the unsure ones ? ---> Probably not  
    """

    def make_sorted_indices( self, strategy, data  ) :
        #indices = list(range(len(data)))
        #random.shuffle(indices)
        #return indices
        pass
class LeastConfidentStrategy():
    ''' Goal is to keep least Confident described in CAIPI 
    WIP
    TODO FInish Calculation
    TODO What is the Decision Function 
    TODO Implications does this make sense ? Only keeping the unsure ones ? ---> Probably not  
    '''
    def make_sorted_indices( self, decision_function, data  ) :
        indices = list(range(len(data)))
        #examples = sorted(examples)
        margins = np.abs(decision_function(data[indices]))
        #print(margins)
        # NOTE margins has shape (n_examples,) or (n_examples, n_classes)
        if margins.ndim == 2:
            #sort indicies based
            margins = margins.min(axis=1)
        return margins
    


class LeastMaringStrategy():
    '''
    Least Margin described in CAIPI 
    TODO Predict Proba
    '''
    def make_sorted_indices( self, predict_proba, data  ) :
        #examples = sorted(examples)
        probs = predict_proba(data)
        # NOTE probs has shape (n_examples, n_classes)
        diffs = np.zeros(probs.shape[0])
        for i, prob in enumerate(probs):
            sorted_indices = np.argsort(prob)
            diffs[i] = prob[sorted_indices[-1]] - prob[sorted_indices[-2]]
        #TODO indicies of min diff 
        indicies = 0
        return indicies


#class FeatureBasedExemplarsSelectionStrategy(ExemplarsSelectionStrategy, ABC):
#    """Base class to select exemplars from their features"""

#    def __init__(self, model: Module, layer_name: str):
#        self.feature_extractor = FeatureExtractorBackbone(model, layer_name)

#    @torch.no_grad()
#    def make_sorted_indices(
#        self, strategy: "SupervisedTemplate", data: make_classification_dataset
#    ) -> List[int]:
#        self.feature_extractor.eval()
#        collate_fn = data.collate_fn if hasattr(data, "collate_fn") else None
#        features = cat(
#            [
#                self.feature_extractor(x.to(strategy.device))
#                for x, *_ in DataLoader(
#                    data,
#                    collate_fn=collate_fn,
#                    batch_size=strategy.eval_mb_size,
#                )
#            ]
#        )
#        return self.make_sorted_indices_from_features(features)

#    @abstractmethod
#    def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
#        """
#        Should return the sorted list of indices to keep as exemplars.
#        The last indices will be the first to be removed when cutoff memory.
#        """


class HerdingSelectionStrategy(): #FeatureBasedExemplarsSelectionStrategy
    """The herding strategy as described in iCaRL.
    It is a greedy algorithm, that select the remaining exemplar that get
    the center of already selected exemplars as close as possible as the
   center of all elements (in the feature space).
    """

    def make_sorted_indices_from_features(self, features):
        selected_indices = []
        center = features.mean(dim=0)
        current_center = center * 0#

        for i in range(len(features)):
            # Compute distances with real center
            candidate_centers = current_center * i / (i + 1) + features / (
                i + 1
            )
            distances = pow(candidate_centers - center, 2).sum(dim=1)
            distances[selected_indices] = np.inf

            # Select best candidate
            new_index = distances.argmin().tolist()
            selected_indices.append(new_index)
            current_center = candidate_centers[new_index]

        return selected_indices


#class ClosestToCenterSelectionStrategy(FeatureBasedExemplarsSelectionStrategy):
#    """A greedy algorithm that selects the remaining exemplar that is the
#    closest to the center of all elements (in feature space).
#    """

  #  def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
  #      center = features.mean(dim=0)
  #      distances = pow(features - center, 2).sum(dim=1)
  #      return distances.argsort()


class ReservoirSamplingBuffer():
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__()
        self.max_size = max_size
        # INVARIANT: _buffer_weights is always sorted.
        # Buffer Weights are already sorted
        self._buffer_weights = torch.zeros(0)
        self.buffer=np.array([[]])
        self.buffer_idxs =np.array([[]])

    #def update(self, strategy, **kwargs):
    #    """Update buffer."""
    #    self.update_from_dataset(strategy.experience.dataset)

    def update_from_dataset(self, new_data):
        """Update the buffer using the given dataset.
        :param new_data:
        :return:
        """
        new_weights = torch.rand(1)
        #print('New Weights',new_weights)
        cat_weights = torch.cat([new_weights, self._buffer_weights])
        #print('ND ',new_data)
        #print('buffer', self.buffer)
        try:
            cat_data =np.concatenate((new_data,self.buffer))
        except: 
            cat_data=new_data
        #print('CAT Data', cat_data)
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)
        #print(sorted_weights)
        #print('max size',self.max_size)

        self.buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer =cat_data #[cat_data,buffer_idxs]#, buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]

    def resize(self, strategy, new_size):
        """
        Update the maximum size of the buffer.
        #TODO In here make strategy useable  
        """
        #print('THIS IS FROm Reservoir Sampler ')
        self.max_size = new_size
        #print('New_Size',new_size)
        if len(self.buffer) <= self.max_size:
            return
        #print('After ',len(self.buffer))
        #print('IDX ',self.buffer_idxs)
        #print('shape', self.buffer.shape)
        self.buffer =np.take(self.buffer, self.buffer_idxs,axis=0)#self.buffer[]#, torch.arange(self.max_size)]
        #print('BEFORE',len(self.buffer))
        #print('Self.Buffer', self.buffer)
        self.buffer_weights = self._buffer_weights[: self.max_size]
        #print('Buffer Weights',self.buffer_weights )

        # Shour return 100 !


class ClassBalancedBuffer():
    """Stores samples for replay, equally divided over classes.
    There is a separate buffer updated by reservoir sampling for each class.
    It should be called in the 'after_training_exp' phase (see
    ExperienceBalancedStoragePolicy).
    The number of classes can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed classes so far.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: int = None,
    ):
        """Init.
        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__()
        self.max_size = max_size

        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.total_num_groups=total_num_classes
        if not self.adaptive_size:
            assert self.total_num_groups > 0, (
                "You need to specify `total_num_groups` if "
                "`adaptive_size=True`."
            )
        else:
            assert self.total_num_groups is None, (
                "`total_num_groups` is not compatible with "
                "`adaptive_size=False`."
            )

        #self.buffer_groups: Dict[int, ExemplarsBuffer] = {}
        #self._buffer: make_classification_dataset = concat_datasets([])
       
        self.seen_classes = set()

        # Dictonary of Buffers
        self.buffer_groups: dict = {}

    def get_group_lengths(self, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        if self.adaptive_size:
            lengths = [self.max_size // num_groups for _ in range(num_groups)]
            # distribute remaining size among experiences.
            rem = self.max_size - sum(lengths)
            for i in range(rem):
                lengths[i] += 1
        else:
            lengths = [
                self.max_size // self.total_num_groups
                for _ in range(num_groups)
            ]
        return lengths

    def update(self, data,strategy = RandomExemplarsSelectionStrategy, **kwargs): # data is tuble (x,y)
        '''
        Attributes: 
            #TODO Still work to do 
            data torch.Dataloader: current data as DataLoader 
            strategy function:  XXXX
        '''
        print('DATA FROM UPDATE',len(data))
        print('TESTING ', len(next(iter(data))))
        new_data, target = next(iter(data))#data
        new_data=np.array(new_data)
        if len (np.array(new_data).shape)<2: 
            new_data = new_data[ np.newaxis, :]
        #print('target', target)
        if len(np.array(target).shape)<2:
            #print('1 Dim Target')
            #print(target)
            ca = target
        else:
            #print('2 Dim Target')
            #print(target)
            ca = np.argmax(target, axis =1)
            #print('ca',ca)
        for c in np.unique(ca):
            if c not in self.seen_classes:
                self.seen_classes.add(c)
        ll=self.max_size
        class_id=target
        for c_id in np.unique(ca):
            if c_id in self.buffer_groups:
                #print('More than one')
                #print(c_id)
                #print(ca)
                #print(self.buffer_groups)
                #Class exists
                old_buffer_c = self.buffer_groups[c_id]
                #print(c_id)
                #print(ca)
                indicies= np.where(ca==c_id)
                #print(np.where(ca==c_id))
                #print(np.take(new_data,indicies[0],axis=0))
                #print(indicies)
                #print(np.array(new_data).shape)
                if len(indicies[0]>0):
                    old_buffer_c.update_from_dataset(np.take(new_data,indicies[0],axis=0)) #Target was excluded
                    #TODO Resite Strategy  # TODO use as Stratgy , strategy from paper
                    #print('resizes called from update dataset in opdate')
                    #print('Before',len(old_buffer_c.buffer))
                    old_buffer_c.resize(strategy, self.max_size)
            else:
                #print('One')
                #class does not exist 
                new_buffer = ReservoirSamplingBuffer(self.max_size)
                new_buffer.update_from_dataset(new_data)
                self.buffer_groups[c_id] = new_buffer 
        #print('resize called from final ClassBuffer ')
        for class_id, class_buf in self.buffer_groups.items():
            #print('Before',len( self.buffer_groups[class_id].buffer))
            self.buffer_groups[class_id].resize(
                strategy, ll
            )
            #print(f'After Buffer Tesize {class_id}',len( self.buffer_groups[class_id].buffer))
