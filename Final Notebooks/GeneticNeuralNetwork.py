############################################################################
# Genetic training Neural Network  
##############################################################################
import random
import itertools
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from gentools import score_classification
from gentools import historical_fittest
from sklearn.metrics import f1_score



# Importing from gentools
from gentools import offspringFun
from gentools import GrowthRate
from gentools import Noimprove
from gentools import vectorStats
from gentools import Ragnarok
from NeuralNetwork import chromosomeLen
from NeuralNetwork import MLP

#### Genetic Part

def GenMLP(architecture,
                 df,
                 X,
                 target,
                 fitnessFun,
                 n_gen = 15
                 ):
  ### Number of paramters to learn lean of chromosome!!
  individual_size = chromosomeLen(X, architecture)

  creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMax)
  toolbox = base.Toolbox()
  # Attribute generator
  toolbox.register("geneNormal", random.gauss, 0, 10)
  # Structure initializers
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.geneNormal, individual_size)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  # Fitness Function invocation
  toolbox.register("evaluate",
                  lambda chromosome: fitnessFun( architecture = architecture,
                                          X=X,
                                          target=target,
                                          df=df,
                                          initial_solution=chromosome ))
  toolbox.register("mate", tools.cxOnePoint)
  toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10 , indpb=0.1)
  toolbox.register("select", tools.selTournament, tournsize=10)

  pop = toolbox.population(n=50)
  hof = tools.HallOfFame(1)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("std", np.std)
  stats.register("min", np.min)
  stats.register("max", np.max)

  pop, logBook = algorithms.eaSimple(pop, toolbox, cxpb=0.90, mutpb=0.05, ngen=n_gen,
                                    stats=stats, halloffame=hof, verbose=True)
  # pop[0] store the fittest individual
  return pop, logBook
  #best[0].fitness



def GenMLP_score(
                classifier,
                architecture,
                X_train,
                y_train,
                n_gen = 1,
                assessment_metric=f1_score,
                k_folds=5):
   # The number of parameters to learn!
   individual_size = chromosomeLen(X_train, architecture)
   creator.create("FitnessMax", base.Fitness, weights=(1.0,))
   creator.create("Individual", list, fitness=creator.FitnessMax)
   toolbox = base.Toolbox()
   # Attribute generator
   toolbox.register("geneNormal", random.gauss, 0, 10)
   # Structure initializers
   toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.geneNormal, individual_size)
   toolbox.register("population", tools.initRepeat, list, toolbox.individual)
   # Fitness Function invocation
   toolbox.register("evaluate",
                    lambda chromosome: score_classification(model=classifier, 
                                                            architecture=architecture, 
                                                            initial_solution=chromosome,
                                                            X_train_=X_train,
                                                            y_train_=y_train, 
                                                            k_folds=k_folds,
                                                            assessment_metric=assessment_metric))
   toolbox.register("mate", tools.cxOnePoint)
   toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10 , indpb=0.1)
   toolbox.register("select", tools.selTournament, tournsize=10)
   pop = toolbox.population(n=50)
   hof = tools.HallOfFame(1)
   stats = tools.Statistics(lambda ind: ind.fitness.values)
   stats.register("avg", np.mean)
   stats.register("std", np.std)
   stats.register("min", np.min)
   stats.register("max", np.max)
   pop, logBook = algorithms.eaSimple(pop, toolbox, cxpb=0.90, mutpb=0.05, ngen=n_gen,
                                      stats=stats, halloffame=hof, verbose=True)
    # pop[0] store the fittest individual
   return pop, logBook
    #best[0].fitness


#########################################################################
#########################################################################
#    Genetic Algorithm                                                   #
##########################################################################



def gaMLP_score(
                classifier,
                architecture,
                X_train,
                y_train,
                ineq_measure,
                CXPB, 
                MUTPB, 
                assessment_metric=f1_score,
                population_size =30,
                max_generations=20,
                limit_unchanged = 25,
                epsilon = 0.01,
                tournament_size=3,
                seed=123, 
                k_folds=5,
                mate_indpb= 0.5,
                mutate_indpb = 0.01,
                ineq_min=0, 
                verbose=True,
                ):
    random.seed(seed)
    # The number of parameters to learn!
    individual_size = chromosomeLen(X_train, architecture)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("geneNormal", random.gauss, 0, 10)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.geneNormal, individual_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Fitness Function invocation
    toolbox.register("evaluate",
                    lambda chromosome: score_classification(model=classifier, 
                                                            architecture=architecture, 
                                                            initial_solution=chromosome,
                                                            X_train_=X_train,
                                                            y_train_=y_train, 
                                                            k_folds=k_folds,
                                                            assessment_metric=assessment_metric))
    toolbox.register("mate", tools.cxUniform, indpb =  mate_indpb)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5 , indpb=mutate_indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
   
    pop = toolbox.population(n=population_size) ####### Initial population #####
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # Extracting all the fitnesses
    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations
    # Init counters ( generations and consecutive generations witout improvemetns in fittest).
    n_generations,  unchanged = 0, 0
    # Begin the evolution
    history, stats = {}, {}
    ineq_value = ineq_measure(fits)
    stats['Generation0']=vectorStats(fits)
    fittest_  =  tools.selBest(pop, 1)[0].fitness.values[0] # The better in the initialization
    while (n_generations< max_generations)  and (ineq_value>ineq_min) and (unchanged < limit_unchanged) :
        n_generations += 1
        pop[:] = offspringFun(pop, CXPB, MUTPB, toolbox)  ##### Updating population ######
        # Gather all the fitnesses in one list YOU MUST PRINT STATTS
        fits = [ind.fitness.values[0] for ind in pop]
        fittest = tools.selBest(pop, 1)[0].fitness.values[0]  # The better in the first generation
        stats['Generation'+str(n_generations)] = vectorStats(fits)
        rate = GrowthRate(fittest_, fittest) # The rate growth of better individuals
        #print(fittest, fittest_, rate,)
        ineq_value = ineq_measure(fits)   # UPDATE Inequality Measure
        fittest_ = fittest # Update fittest to calculate growth
        if verbose == True:
           # To see in the algorithm 
           print("-"*15)
           print('Generation: ', n_generations)
           print("-"*15)
           print('Fitest :', fittest_, 'Inequality: ', ineq_value, 'Generation without improvement: ',unchanged)
           print('Stats', 'mean: ', vectorStats(fits)[0], 'std: ', vectorStats(fits)[1], 'min:',  vectorStats(fits)[2], 'max: ', vectorStats(fits)[3] )
           print(fittest, ineq_value,  unchanged, vectorStats(fits))
        # How iterations not improve the fittest individual
        if Noimprove(rate, epsilon):
          unchanged +=1 
        else:
          unchanged=0
        ### In the evolution we can Lost a fittest individual
        history[str(fittest)] = tools.selBest(pop,1)[0] # Keep the genotype and phenotype of the fittest individual.
    best_ind = tools.selBest(pop, 1)[0]
    #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    #print(betters)
    #return pop
    #stats return mean, std, min, max
    return best_ind, history, stats   



def gaMLP_ragnarok(mutations,
                    pop_sizes, 
                    cross,
                    tournament_sizes,
                    mate_indpb,
                    mutate_indpb,
                    classifier,
                    ineq_measure, 
                    ineq_min, 
                    max_generations,
                    limit_unchanged,
                    k_folds,
                    architecture, 
                    X_train,
                    y_train, 
                    verbose,):
  
  """ This function is very important because return the fittest individual 
  in all generations(historical) with different combinations of parameters.
  return Ragnarok-> score, genotype, minimun and optimun hyperparameters...
  """
  allin = [pop_sizes, max_generations, tournament_sizes, mutations, cross, mate_indpb, mutate_indpb]
  Hyperparameters = list(itertools.product(*allin)) ## Possible combinations to hyperparameters
  register={} # This is the parameter of ragnarok!
  for hyper in Hyperparameters:
      print("--"*20)
      print(hyper)
      print("--"*20)
      #print('Combination of parameters',hyper)
      #print("--"*20)
      fittest_individual, history, stats = gaMLP_score(
                    classifier = classifier,
                    architecture= architecture,
                    population_size=hyper[0],
                    max_generations= hyper[1],
                    tournament_size=hyper[2],
                    MUTPB=hyper[3],
                    CXPB=hyper[4], 
                    mate_indpb= hyper[5],
                    mutate_indpb=hyper[6],
                    ineq_measure=ineq_measure,
                    ineq_min  = ineq_min,
                    limit_unchanged=limit_unchanged,
                    k_folds=k_folds,
                    X_train = X_train, 
                    y_train = y_train,
                    verbose=verbose)
      register[hyper] = [fittest_individual,stats]
  # Consider the following in History you keep the fittest over all generations!
  # Return the score, the fittest genotype, the combinatios of hyperparameters...
  return register



# GaMLP_Entropy
def gaMLP_Entropy(
                architecture,
                X_train,
                y_train,
                ineq_measure,
                CXPB, MUTPB, 
                population_size =30,
                max_generations=20,
                limit_unchanged =35,
                epsilon = 0.01,
                tournament_size=3,
                seed_random = 10,
                ineq_min=0, 
                mate_indpb = 0.1,
                mutate_indpb = 0.01,
                verbose=True,
                ):
    random.seed(int(seed_random))
    # The number of parameters to learn!
    individual_size = chromosomeLen(X_train, architecture)
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("geneNormal", random.gauss, 0, 10)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.geneNormal, individual_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Fitness Function invocation
    toolbox.register("evaluate",
                    lambda chromosome: MLP(
                                           architecture=architecture, 
                                           initial_solution=chromosome,
                                           X_train=X_train,
                                          y_train=y_train))
    toolbox.register("mate", tools.cxUniform, indpb =  mate_indpb)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10 , indpb=mutate_indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
   
    pop = toolbox.population(n=population_size) ####### Initial population #####
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # Extracting all the fitnesses
    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations
    # Init counters ( generations and consecutive generations witout improvemetns in fittest).
    n_generations,  unchanged = 0, 0
    # Begin the evolution
    history, stats = {}, {}
    ineq_value = ineq_measure(fits)
    stats['Generation0']=vectorStats(fits)
    fittest_  =  tools.selBest(pop, 1)[0].fitness.values[0] # The better in the initialization
    while (n_generations< max_generations)  and (ineq_value>ineq_min) and (unchanged < limit_unchanged) :
        n_generations += 1
        pop[:] = offspringFun(pop, CXPB, MUTPB, toolbox)  ##### Updating population ######
        # Gather all the fitnesses in one list YOU MUST PRINT STATTS
        fits = [ind.fitness.values[0] for ind in pop]
        fittest = tools.selBest(pop, 1)[0].fitness.values[0]  # The better in the first generation
        stats['Generation'+str(n_generations)] = vectorStats(fits)
        rate = GrowthRate(fittest_, fittest) # The rate growth of better individuals
        #print(fittest, fittest_, rate,)
        ineq_value = ineq_measure(fits)   # UPDATE Inequality Measure
        fittest_ = fittest # Update fittest to calculate growth
        if verbose == True:
           # To see in the algorithm 
           print("-"*15)
           print('Generation: ', n_generations)
           print("-"*15)
           print('Fitest :', fittest_, 'Inequality: ', ineq_value, 'Generation without improvement: ',unchanged)
           print('Stats', 'mean: ', vectorStats(fits)[0], 'std: ', vectorStats(fits)[1], 'min:',  vectorStats(fits)[2], 'max: ', vectorStats(fits)[3] )
           print(fittest, ineq_value,  unchanged, vectorStats(fits))
        # How iterations not improve the fittest individual
        if Noimprove(rate, epsilon):
          unchanged +=1 
        else:
          unchanged=0
        ### In the evolution we can Lost a fittest individual
        history[str(fittest)] = tools.selBest(pop,1)[0] # Keep the genotype and phenotype of the fittest individual.
    best_ind = tools.selBest(pop, 1)[0]
    #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    #print(betters)
    #return pop
    #stats return mean, std, min, max
    return best_ind, history, stats   
