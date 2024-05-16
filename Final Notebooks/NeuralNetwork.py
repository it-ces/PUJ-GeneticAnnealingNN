# Module of Fodward neural network (Fitness Function).

import numpy as np
import pandas as pd

## Neccesary functions

def index(layer, size_layers):
  # This function allow me to known in what index finish the weighs in each layer (to use in flat)
  acum = 0
  for layer in range(1, layer):
    acum = acum + size_layers[str(layer)] * size_layers[str(layer+1)]
  return acum

def b_size(size_layers):
  count = 0
  for layer in range(2,len(size_layers)+1):
    count += size_layers[str(layer)]
  return count



def indexB(layer, size_layers):
  acum = index(len(size_layers), size_layers)
  for layer in range(2, layer+1):
    acum = acum + size_layers[str(layer)]
  return acum


def flatten(W_info):
  ## Matrix to list or flat array
  flat = W_info["2"]
  for layer in range(3, len(W_info)+2): # W_info have a number lesser than size_layers
    flat =  np.append(flat, W_info[str(layer)].reshape(-1))
  return flat


# Pass the list or flatten array
# to matrix form to do faster operations.
def extractW(flat, layer, size_layers):
  if layer==2:
    extracted = flat[0: index(layer, size_layers)]
  else:
    extracted = flat[index(layer-1, size_layers) : index(layer, size_layers)]
  return extracted.reshape(size_layers[str(layer)], size_layers[str(layer-1)])

def extractB(flat, layer, size_layers):
  if layer==2:
    extracted = flat[index(len(size_layers), size_layers):indexB(layer, size_layers)]
  else:
    extracted = flat[indexB(layer-1, size_layers) : indexB(layer, size_layers)]
  return extracted



def Wnormal(layer, size_layers, seed=123):
  # initialize the random parameters
  np.random.seed(seed)
  return np.random.randn(size_layers[str(layer)], size_layers[str(layer-1)])



## Cost Function

def BinaryLog(y_true, est_probabilities):
  # Pass in bothe vectores in narrays
  est_probabilities = np.clip(est_probabilities, 0.00000000000001, 0.99999999999999) # not defined estimated probs in zero or one
  return -((y_true * np.log(est_probabilities) + (1 - y_true) * np.log(1-est_probabilities)).sum())/y_true.shape[0]
  # log_loss have a better implementation!


# Activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
sigmoid = np.vectorize(sigmoid)  ## Acts similar to map (func, iter)


def solution_init(size_layers):
  W_init = {}
  for layer in range(2,len(size_layers)+1):
    W_init[str(layer)] = Wnormal(layer, size_layers)
  solution = flatten(W_init)
  solution =  np.append(solution, np.zeros(b_size(size_layers)))
  return solution


def forward(xi, solution, size_layers):
  # Rememer that W2 is the first layer of weights
  z = extractW(solution, 2, size_layers) @ xi
  for layer in range(3,len(size_layers)+1):
    z = extractW(solution, layer, size_layers) @ sigmoid(z) +  extractB(solution, layer, size_layers)
  return sigmoid(z)

def chromosomeLen(X, architecture):
  # Total number of parameter to learn given X and architecture
  size_layer = size_layers_init(X, architecture)
  return indexB(len(size_layer),size_layer)



def size_layers_init(X, architecture):
  if isinstance(X, list):
    size_layers = {'1':len(X)} # The number of neurons in the first layer...
  else:
    size_layers = {'1':X.shape[1]} # If receive np.array's or dataFrames..
  for ith,size in enumerate(architecture, start=2):
    size_layers[str(ith)] = size
  return size_layers



def solution_to_MLP(X, architecture):
  size_layers =size_layers_init(X, architecture)
  solution = solution_init(size_layers)
  return solution



def MLP(
    architecture,
    X_train,
    y_train,
    initial_solution=[], 
    preds=False,
    score=False):
  if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train)
  if isinstance(y_train, np.ndarray):
    y_train = pd.DataFrame(y_train)
#  This code could be improved using a switch!
  size_layers = size_layers_init(X_train, architecture)
  if len(initial_solution)==0:
    solution = solution_init(size_layers)
  else:
    solution = np.array(initial_solution)
  y_probs = np.zeros(X_train.shape[0])  # Array to keep the solutions...
  for ith, row in enumerate(X_train.to_numpy()):
    y_probs[ith] = forward(xi = row, solution = solution, size_layers = size_layers)
  if preds == True:
    return y_probs
  elif score!=False:
    return score(y_train.to_numpy(), np.where(y_probs>=0.5,1,0)), # F1_score or another metrics...
  else:
    return BinaryLog(y_train.to_numpy(), y_probs), # This is return by default...


#####################################################
#####################################################
# MLP class                                         #
# was created given we need fit and predict to      #
# uses score                                        #
#####################################################


class  MLPclassifer():
    def __init__(self, architecture, 
                 initial_solution=[], 
                 preds=False, 
                 treshold=0.5,
                 Entropy = False):
        self.architecture = architecture
        self.initial_solution =[]
        self.preds= preds
        self.initial_solution = initial_solution
        self.treshold = treshold

    def fit(self, X, target):
        self.X = X
        self.target = target
        self.size_layers = size_layers_init(self.X, self.architecture)
        if len(self.initial_solution)==0:
            solution = solution_init(self.size_layers)
        else:
            solution = np.array(self.initial_solution)
        self.solution = solution
        return self
    
    def cost(self):
        # We can add different cost funcions.
        y_probs = np.zeros(self.X.shape[0])  # Array to keep the solutions...
        for ith, row in enumerate(self.X):
            y_probs[ith] = forward(xi = row, solution = self.solution, size_layers = self.size_layers)
        self.y_probs = y_probs
        return BinaryLog(self.target, self.y_probs),

    def predict(self,Xtest):
        self.Xtest = Xtest
        y_probs_test =  np.zeros(self.Xtest.shape[0])
        for ith, row in enumerate(self.Xtest):
            y_probs_test[ith] = forward(xi = row, solution = self.solution, size_layers = self.size_layers)
        return np.where(y_probs_test>0.5,1,0)

