import numpy as np
import os

'''
Toy Colors imported from right for the right reasons. 
https://github.com/dtak/rrr/blob/master/rrr/toy_colors.py
#TODO  Train Tes Val Splits 
'''

colors = {
  'r': np.array([255,   0,   0], dtype=np.uint8),
  'o': np.array([255, 128,   0], dtype=np.uint8),
  'y': np.array([255, 255,   0], dtype=np.uint8),
  'g': np.array([0,   255,   0], dtype=np.uint8),
  'b': np.array([0,   128, 255], dtype=np.uint8),
  'i': np.array([0,     0, 255], dtype=np.uint8),
  'v': np.array([128,   0, 255], dtype=np.uint8)
}

imglen = 5
imgshape = (imglen, imglen, 3)
topleft = [0,[0]]
topright = [0,[imglen-1]]
botleft = [imglen-1,[0]]
botright = [imglen-1,[imglen-1]]

ignore_rule1 = np.zeros(imgshape)
for corner in [topleft, topright, botleft, botright]: ignore_rule1[corner] = 1
ignore_rule1 = ignore_rule1.ravel().astype(bool)

ignore_rule2 = np.zeros(imgshape)
ignore_rule2[[0,[1,2,3]]] = 1
ignore_rule2 = ignore_rule2.ravel().astype(bool)

def random_color():
  return colors[np.random.choice(['r','g','b','v'])]

def Bern(p):
  return np.random.rand() < p

def any_repeats(row):
  n_unique = len(set(tuple(c) for c in row))
  return n_unique < len(row)

def ensure_class_0_rules_apply(img):
  # Rule 1
  img[topleft] = img[botright]
  img[topright] = img[botright]
  img[botleft] = img[botright]

  # Rule 2
  toprow = img[0]
  while any_repeats(toprow[1:-1]):
    toprow[1+np.random.choice(imglen-2)] = random_color()

def ensure_class_1_rules_apply(img):
  # Rule 1
  if Bern(0.5):
    while np.array_equal(img[topright], img[botleft]):
      img[topright] = random_color()
  else:
    while np.array_equal(img[topleft], img[botright]):
      img[topleft] = random_color()

  # Rule 2
  toprow = img[0]
  while not any_repeats(toprow[1:-1]):
    toprow[1+np.random.choice(imglen-2)] = random_color()

def generate_image(label):
  image = np.array([[random_color()
                      for _ in range(imglen)]
                        for __ in range(imglen)], dtype=np.uint8)

  if label == 0:
    ensure_class_0_rules_apply(image)
  else:
    ensure_class_1_rules_apply(image)

  return image.ravel()

def largest_mag_2d(input_gradients, cutoff=0.67):
  # return 2d arrays of flattened largest-magnitude elements
  # so we can compare pixel locations on a 2d basis rather than
  # worrying about RGB. 2d arrays have 1s if any of the pixel component
  # gradients in that space are above the cutoff.
  return np.array([((
        np.abs(e) > cutoff*np.abs(e).max()
      ).reshape(5,5,3).sum(axis=2).ravel() > 0
    ).reshape(5,5) for e in input_gradients
  ]).astype(int)

def fraction_inside_corners(mask1):
  mask2 = mask1.copy()
  mask2[0][0] = 0
  mask2[0][-1] = 0
  mask2[-1][0] = 0
  mask2[-1][-1] = 0
  return 1 - mask2.ravel().sum() / float(mask1.ravel().sum())

def fraction_inside_topmids(mask1):
  mask2 = mask1.copy()
  mask2[0][1] = 0
  mask2[0][2] = 0
  mask2[0][3] = 0
  return 1 - mask2.ravel().sum() / float(mask1.ravel().sum())

def rule1_score(model, X):
  return np.mean([fraction_inside_corners(grad) for grad in largest_mag_2d(model.input_gradients(X))])

def rule2_score(model, X):
  return np.mean([fraction_inside_topmids(grad) for grad in largest_mag_2d(model.input_gradients(X))])

def generate_dataset(N=20000, cachefile='../data/new_toy-colors.npz'):
  if cachefile and os.path.exists(cachefile):
    cache = np.load(cachefile)
    data = tuple([cache[f] for f in sorted(cache.files)])
  else:
    train_y = (np.random.rand(N) < 0.5).astype(np.uint8)
    test_y = (np.random.rand(N) < 0.5).astype(np.uint8)
    data = (
      np.array([generate_image(y) for y in train_y]),
      np.array([generate_image(y) for y in test_y]), train_y, test_y)
    if cachefile:
      np.savez(cachefile, *data)
  return data


def rule1(array): 
  '''all edges have the same color'''
  dic_identical =np.array([0,4,20,24,25,29,45,49,50,54,70,74]).reshape(3,-1)
  done=False
  for a in dic_identical:
    for i in range(0,len(a)-1):
      if array[a[i]] == array[a[i+1]]:
        done= True
      else: 
        return False
  return done

def rule2(array): 
  '''middle Part different color'''
  dic_different =np.array([1,2,3,26,27,28,51,52,53]).reshape(3,-1)
  done=False
  for a in dic_different:
    if array[a[0]] != array[a[1]] and array[a[2]] != array[a[1]] and array[a[0]] != array[a[2]]:
      done= False
    else: 
      return True
  return done

if __name__ == '__main__':
  import pdb
  X, Xt, y, yt = generate_dataset()
  pdb.set_trace()
  pass