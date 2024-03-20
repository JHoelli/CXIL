from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
# TODO doch nicht so gut verwendbar
#TODO  Train Tes Val Splits 

def iris_usage(model, X, cutoff=0.67):
  mask = model.largest_gradient_mask(X, cutoff)
  return mask[:, :4].sum() / mask.sum()

def generate_dataset(test_size=0.5):
  iris = load_iris()
  bc = load_breast_cancer()

  iris_X = np.vstack((
    iris.data[np.argwhere(iris.target==1)][:,0],
    iris.data[np.argwhere(iris.target==2)][:,0]))

  bc_X = np.vstack((
    bc.data[np.argwhere(bc.target==0)][:50,0],
    bc.data[np.argwhere(bc.target==1)][:50,0]))

  full_X = np.hstack((iris_X, bc_X))
  full_y = np.array([0]*50 + [1]*50)

  X, Xtr, y, yt = train_test_split(full_X, full_y, test_size=test_size)
  Xtr, Xv, yt, yv = train_test_split(full_X, full_y, test_size=test_size)

  # zero out iris elements
  Xt = Xtr.copy()
  Xt[:, :4] = 0

  return X, Xt, y, yt

if __name__=='__main__': 
  #TODO WAS IS Xtr? 
  X,  Xt, y, yt=generate_dataset(test_size=0.5)
  with open('./data/Iris_Cancer_X.npy', 'wb') as f:
    np.save(f,X)
  #with open('./data/Iris_Cancer_Meta.npy', 'wb') as f:
  #  np.save(f,Xtr)
  with open('./data/Iris_Cancer_y.npy', 'wb') as f:
    np.save(f,y)
  with open('./data/Iris_Cancer_X_test.npy', 'wb') as f:
    np.save(f,Xt)
  #with open('./data/Iris_Cancer_Meta_test.npy', 'wb') as f:
  #  np.save(f,Xttr)
  with open('./data/Iris_Cancer_y_test.npy', 'wb') as f:
    np.save(f,yt)

