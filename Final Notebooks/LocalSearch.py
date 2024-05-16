###################################################
####### Local Search ##############################
###################################################


# This code have the implementation of 


# Module of Simulated Anealing Fitness Function...

import numpy as np
import random 

#####################################
# Simulated annealing               #
#####################################
def search(solution, alpha=0.01):
  return solution + np.random.randn(len(solution)) * alpha
   

def boltzmann(deltaE,  T, k=1):
  return np.exp(deltaE/(k*T))


def SA_max(solution, search,
         Tf,
         cooling_rate,
         fitnessFunction,
         beta = 0.001,
         seed_random=123,
         max_iterations = 100,
         reduce_temp=0.01,
         alpha=0.01):
  np.random.seed(seed_random)
  #print('initial  fitness:',fitnessFunction(solution))
  record = {}
  record['major'] = (fitnessFunction(solution), solution)
  executions = 0 
  T = fitnessFunction(solution)[0] * beta
  while T>Tf and executions< max_iterations:
    executions +=1 
    for _ in range(cooling_rate):
      solution_temp  = search(solution, alpha=alpha)
      E0, E1 = fitnessFunction(solution)[0], fitnessFunction(solution_temp)[0]
      deltaE = E1-E0
      if deltaE >= 0 :
        solution = solution_temp
      else:
        if random.uniform(0,1) < boltzmann(deltaE, T):
          solution = solution_temp
      if fitnessFunction(solution)[0] > record['major'][0][0]:
           record['major'] = (fitnessFunction(solution), solution)
    T = T - reduce_temp
    print(executions)
    print(fitnessFunction(solution),'Temperarure :' ,T)
  #return record # dictionary
  return record



def SA_min(solution, search,
         Tf,
         cooling_rate,
         fitnessFunction,
         max_iterations = 100,
         beta = 0.001,
         seed_random=123,
         reduce_temp=0.01,
         alpha=0.01):
  np.random.seed(int(seed_random))
  #print('initial  fitness:',fitnessFunction(solution))
  record = {}
  record['minor'] = (fitnessFunction(solution), solution)
  T = fitnessFunction(solution)[0] * beta
  executions = 0
  while T>Tf and executions< max_iterations:
    executions +=1 
    for _ in range(cooling_rate):
      solution_temp  = search(solution, alpha=alpha)
      E0, E1 = fitnessFunction(solution)[0], fitnessFunction(solution_temp)[0]
      deltaE = E1-E0
      if deltaE <= 0 :
        solution = solution_temp
      else:
        if random.uniform(0,1) < boltzmann(-1*deltaE, T):
          solution = solution_temp
      if fitnessFunction(solution)[0] < record['minor'][0][0]:
           record['minor'] = (fitnessFunction(solution), solution)
    T = T - reduce_temp
    print(executions)
    print(fitnessFunction(solution), 'Temperature :', T)
  return record # dictionary



# Module for implement backpropagation with tensorflow

import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from NeuralNetwork import extractB
from NeuralNetwork import extractW
from NeuralNetwork import size_layers_init




def ModelArchitecture(model, architecture, X):
    for li in range(len(architecture)):
        if architecture[li] ==0: # If appear a zero not add layer...
           pass 
        if li == 0:
            model.add(Dense(units = architecture[li], activation='sigmoid', input_shape=(X.shape[1],)))
        else:
            model.add(Dense(units = architecture[li], activation='sigmoid' ))
    return model






def assign_weights(model, architecture, X, GAsolution):
    initial_parameters = []
    for layer in range(2, len(size_layers_init(X, architecture))+1):
        initial_parameters.append([extractW(np.array(GAsolution), layer, size_layers_init(X, architecture)), 
                                extractB(np.array(GAsolution), layer, size_layers_init(X, architecture))])

    init_parms = []
    for i in range(len(architecture)):
        if i == 0:
            init_parms.append([initial_parameters[i][0].reshape(X.shape[1], architecture[i]), initial_parameters[i][1]] )
        else:
            init_parms.append([initial_parameters[i][0].reshape(architecture[i-1], architecture[i]),initial_parameters[i][1]])
    

    for layer in range(len(architecture)):
        model.layers[layer].set_weights([init_parms[layer][0], init_parms[layer][1]])
        model.layers[layer].set_weights([init_parms[layer][0], init_parms[layer][1]])
    
    return model
