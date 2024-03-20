import numpy as np
import torch
import torch.nn.functional as F

def densify(x):
    try:
        x = x.toarray()
    except AttributeError:
        pass
    if x.shape[0] != 1:
        # if X[i] is already dense, densify(X[i]) is a no-op, so we get an x
        # of shape (n_features,) and we turn it into (1, n_features);
        # if X[i] is sparse, densify(X[i]) gives an x of shape (1, n_features).
        x = x[np.newaxis, ...]
    return x

class mod_wrapper():
    def __init__(self, model) -> None:
        self.model = model
        self.model.eval()
        pass
    def predict(self,item):
        item = torch.from_numpy(item).float()
        pred= F.softmax(self.model(item))
        return pred.detach().numpy()