# Module of  inequality Measures
import numpy as np

##########################################
#  Inequality index                      #
#  MEAUSRING DIVERSITY                   #
###########################################
# Define function to calculate acumlative frequencies
import math
def acum(array):
  # Take in mind that the result not is ordered
  # and in gini is neccesary...
  new, N = [], sum(array)
  acum = 0
  for i in array:
    acum = acum + i
    new.append(acum/N)
  return new

#define function to calculate Gini coefficient
def gini(data):
  Y = acum(sorted(data))
  N = acum([1 for _ in range(len(data))])
  area = 0
  for index in range(1,len(data)):
    area +=  ((Y[index-1] + Y[index])) *  (N[index] - N[index-1])
  return 1 - area

## theil give us a more stable computation!
## I need pass theil to numpy computation!
def theil(array):
  Y = sum(array)
  N = len(array)
  relative = [y/Y for y in array]
  Theil = sum([y * math.log(y * N) for y in relative])/math.log(N)
  return Theil


# Epislon is to avoid zero's remebemer that theil not is defined when appear zeros...

def Theil(data, epsilon = 0.0000001):
    if isinstance(data,list):
        data = np.array(data)
    if 0 in data:
        data = data + epsilon
    N = data.shape[0]
    Y  = data.sum()
    y = data/Y
    return (y * np.log(y*N)).sum() / np.log(N)



## Add hamming distance!