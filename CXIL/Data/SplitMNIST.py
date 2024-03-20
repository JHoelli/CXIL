'''
According to Ross et al.
FROM https://github.com/dtak/rrr/blob/master/rrr/decoy_mnist.py
#TODO save were changes are made ? 
#TODO show not relevant ? 
'''


from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()
import os
import gzip
import struct
import array
import numpy as np
from urllib.request import urlretrieve

def download_mnist(datadir):
  if not os.path.exists(datadir):
    os.makedirs(datadir)

  base_url = 'http://yann.lecun.com/exdb/mnist/'

  def parse_labels(filename):
    with gzip.open(filename, 'rb') as fh:
      magic, num_data = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, 'rb') as fh:
      magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
    if not os.path.exists(os.path.join(datadir, filename)):
      urlretrieve(base_url + filename, os.path.join(datadir, filename))

  train_images = parse_images(os.path.join(datadir, 'train-images-idx3-ubyte.gz'))
  train_labels = parse_labels(os.path.join(datadir, 'train-labels-idx1-ubyte.gz'))
  test_images  = parse_images(os.path.join(datadir, 't10k-images-idx3-ubyte.gz'))
  test_labels  = parse_labels(os.path.join(datadir, 't10k-labels-idx1-ubyte.gz'))

  return train_images, train_labels, test_images, test_labels

def _generate_dataset(datadir, decoy=False):


  X, y, Xt, yt = download_mnist(datadir)
  if decoy:
    from XIL.Data.DataLoader import load_data_and_sim
    with np.load('./data/decoy-mnist/decoy-mnist.npz') as data:
        print(data.keys())
        #for a in data.keys():
        #    print(a)
        X=data['arr_1']
        y=data['arr_2']
        Xt=data['arr_5']
        yt=data['arr_6']
  #print(np.where((y==0) | (y==1)))
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


  return  X_0,y_0,X_1,y_1,X_2,y_2,X_3,y_3,X_4,y_4, Xt_0,yt_0,Xt_1,yt_1,Xt_2,yt_2,Xt_3,yt_3,Xt_4,yt_4, Xt, yt

def generate_dataset(cachefile='data/split-mnist/split-mnist.npz'):
  if cachefile and os.path.exists(cachefile):
    cache = np.load(cachefile)
    data = tuple([cache[f] for f in sorted(cache.files)])
  else:
    data = _generate_dataset(os.path.dirname(cachefile))
    if cachefile:
      np.savez(cachefile, *data)
  return data

def generate_dataset_decoy(cachefile='data/split-decoy-mnist/split-decoy-mnist.npz'):
  if cachefile and os.path.exists(cachefile):
    cache = np.load(cachefile)
    data = tuple([cache[f] for f in sorted(cache.files)])
  else:
    data = _generate_dataset(os.path.dirname(cachefile),decoy=True)
    if cachefile:
      np.savez(cachefile, *data)
  return data

if __name__ == '__main__':
  #generate_dataset()
  generate_dataset_decoy()