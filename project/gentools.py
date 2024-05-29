########################################
#### Tools for genetic algorihms in ML #
########################################
# Tools for implement genetic algorithms in Machine learning....

# Using Cross validation to evaluate a classifier with 
# The method .fit 
# (This method could be generalize to accept any score...)
# Take in mind that in the input of features is the vector of X in train...

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import random
from deap import creator
from deap import base
from deap import tools
import matplotlib.pyplot as plt
import numpy as np
import itertools
from NeuralNetwork import MLP



# This is the score for feature selection
def score(solution, classifier, X_train_, y_train_, k_folds, assessment_metric=f1_score):
    X_train, y_train, scores = X_train_.copy(), y_train_.copy() , []
    solution = np.array(solution, dtype='bool')
    if solution.sum()==0:
      return 0.000000001,
    else:
      X_train, y_train = X_train.loc[:, solution].to_numpy() , y_train.to_numpy()
      skf = StratifiedKFold(n_splits=k_folds, random_state=666, shuffle=True)
      for itrain, itest in skf.split(X_train, y_train):
        Xi_train, Xi_test = X_train[itrain], X_train[itest]
        yi_train, yi_test = y_train[itrain], y_train[itest]
        scores.append(assessment_metric(yi_test, classifier.fit(Xi_train, yi_train).predict(Xi_test)))
        #print(len(yi_train), len(Xi_train), len(Xi_test), len(yi_test))
    return np.mean(scores),

# This is the score for classification...
# This is the score for classification...

def score_classification(model,
                         architecture,  
                         X_train_,
                         y_train_,
                         k_folds,
                         initial_solution = [], 
                         assessment_metric=f1_score, 
                         random_state = 123):
    classifier = model(initial_solution=initial_solution, architecture = architecture)
    X_train, y_train, scores = X_train_.copy(), y_train_.copy() , []
    X_train, y_train = X_train.to_numpy() , y_train.to_numpy()
    skf = StratifiedKFold(n_splits=k_folds, random_state=random_state, shuffle=True)
    for itrain, itest in skf.split(X_train, y_train):
      Xi_train, Xi_test = X_train[itrain], X_train[itest]
      yi_train, yi_test = y_train[itrain], y_train[itest]
      scores.append(assessment_metric(yi_test, classifier.fit(Xi_train, yi_train).predict(Xi_test)))
     # print(len(yi_train), len(Xi_train), len(Xi_test), len(yi_test))
     # print(scores)
    return np.mean(scores),
# Observation this function allow us experimentation!!!


def offspringFun(pop, CXPB, MUTPB, toolbox):
    """
    The objective of this function is abstract the process
    of mate, mutate the popupaltion to update it.
    """
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
      # cross two individuals with probability CXPB
      if random.random() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

    for mutant in offspring:
      # mutate an individual with probability MUTPB
      if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values
      # Assing the values again to each chromosomes that was mutated or mated
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit
    return offspring



def GrowthRate(current, next):
  return abs((next - current)/current)

def Noimprove(rate, epsilon):
  if rate<epsilon:
    return True


def vectorStats(array_):
    # Mean, std, min and max
    array = np.array(array_)
    return array.mean(), array.std(), array.min(), array.max()




def historical_fittest(history, score='False'):
   # score  = True, return the individual most fittesst and the value of the fit.
   fitnesses  = [float(key) for key in history]
   if score=='False':
      return history[str(fitnesses[fitnesses.index(max(fitnesses))])]
   else:
      return history[str(fitnesses[fitnesses.index(max(fitnesses))])],max(fitnesses)
   
   
# Hint: best_ind is the fittest individual in last generation
# historical_fittest give us the fittest individual in all time (all generations).



### Over test evaluation....!

def get_names(solution, X_train):
   return X_train.columns[np.array(solution , dtype='bool')]



def PerformanceTest(solution, classifier, Xtrain, ytrain, Xtest, ytest):
   
   # F1_score sklearn (y_true, y_pred)...
   # Solution  binary array (integer or booleans)
   # It is important that classifier take the predict or predict_proba method....
   # take in mind that classifier must be trained with solution
   vars = get_names(solution, Xtrain)
   classifier.fit(Xtrain.loc[:,vars], ytrain)
   Xtest = Xtest.loc[:,vars ]
   preds = classifier.predict(Xtest)
   return f1_score(ytest, preds)




def plot_stats(stats, path=None, Title=None):
    generations = [i for i in range(len(stats))]
    means = [i[0] for i in stats.values()]
    stds =  [i[1] for i in stats.values()]
    mins =  [i[2] for i in stats.values()]
    maxs =  [i[3] for i in stats.values()]
    generations = [i for i in range(len(stats))]
    fig, ax = plt.subplots()
    ax.plot(generations, means,  label='means',  color='#4646AC', linestyle='dashed')
    ax.plot(generations, mins,  label='mins',  color='#747BD9', linestyle='dotted')
    ax.plot(generations, maxs, label='max', color='#7500E4') 
    leg = ax.legend(loc="lower right")
    ax.set_xlabel("Generations")
    ax.set_ylabel(Title)
    if path:
          plt.savefig(path)

       





def Ragnarok(register):
  # Register is a dict
  # Find the fittest individual in all history of each!!
  # of the possible combinations.
  # Return score, code(genotype-individual)
  # register[(key)][0] -> history of the combination of hyperparameters
  # historical_fittest(register[(key)][0], score=True)[1] return the (fit value) of the historical fittest
  ragnarok_ = {}
  for index, key in enumerate(register):
      ragnarok_[index] =  [historical_fittest(register[(key)][0], score=True)[1],
                          historical_fittest(register[(key)][0]),
                            key ]
  performance = np.array([x[0] for x in ragnarok_.values()])
  # Return score, genotype, hyperparameters...
  return ragnarok_[performance.argmax()]
  
  ### is more easy  if you genarete a class denominated solutuion and put as attributes
    ### the score in the model and the variance of its population. it is more clever....
    ## .los fitness estan en history key....



from sklearn.metrics import f1_score

def score_optimun(solution, 
                  architecture,
                   X_train_,
                   y_train_,
                   k_folds,
                   assessment_metric):
    
    X_train, y_train, scores = X_train_.copy(), y_train_.copy() , []
    X_train, y_train = X_train.to_numpy() , y_train.to_numpy()
    skf = StratifiedKFold(n_splits=k_folds, random_state=666, shuffle=True)
    for itrain, itest in skf.split(X_train, y_train):
      Xi_test =  X_train[itest]
      yi_test =  y_train[itest]
      scores.append(MLP(architecture=architecture, X_train=Xi_test, y_train=yi_test, initial_solution = solution, score=assessment_metric))
        #print(len(yi_train), len(Xi_train), len(Xi_test), len(yi_test))
    return np.median(scores), np.percentile(scores, 0.75) - np.percentile(scores, 0.25)


from sklearn.metrics import recall_score
def select_optimun_parameters(regist,
                              k_folds,
                              architecture,
                              X_train,
                              y_train,
                              assessment_metric):
   scores = {}
   for key in regist:
    print('From here is the fittest', key)
    print(score_optimun(solution=regist[key][0], architecture =architecture,X_train_=X_train, y_train_=y_train, k_folds=k_folds, assessment_metric=assessment_metric))
    scores[key] = score_optimun(solution=regist[key][0], architecture =architecture, X_train_=X_train, y_train_=y_train, k_folds=k_folds, assessment_metric=recall_score)
    optimun_parameters = sorted(scores.items(), key=lambda x:(-x[1][0], x[1][1]))
    return optimun_parameters[0][0]