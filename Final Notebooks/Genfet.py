# Genetic Feature selection model....

from sklearn.metrics import f1_score
import random
from deap import creator
from deap import base
from deap import tools
import itertools


# Importing from gentools
from gentools import score
from gentools import offspringFun
from gentools import GrowthRate
from gentools import Noimprove
from gentools import vectorStats
from gentools import Ragnarok





def GaFeatureSelection( classifier,
                        population_size, 
                        CXPB, 
                        MUTPB,
                        ineq_measure,
                        max_generations,
                        X_train,
                        y_train,
                        limit_unchanged = 25,
                        epsilon = 0.01,
                        tournament_size=12,
                        seed=123,
                        k_folds=5,
                        ineq_min=0, 
                        mutate_indpb = 0.1,
                        mate_indpb = 0.5, # Important!
                        verbose=True):
    random.seed(seed)
    ################################
    # Definning Genetic operators  #
    # And individuals defintion    #
    ################################
    chromosomal_size = X_train.shape[1]
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("binaryGen", random.randint, 0,1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binaryGen, chromosomal_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda x: score( x ,classifier ,X_train, y_train, k_folds))
    toolbox.register("mate", tools.cxUniform, indpb = mate_indpb)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutate_indpb)
    toolbox.register("select", tools.selTournament, tournsize = tournament_size) #change to rolloute wheel

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
          print(fittest,ineq_value, tools.selBest(pop, 1)[0], unchanged)
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
    return best_ind, fittest, stats   




def Geafet(pop_sizes,
           generations,
           tournaments_sizes,
           mutations,
           crossovers,
           mate_indpb,
           mutate_indpb,
           classifier,
           ineq_measure,
           ineq_min, 
           k_folds,
           limit_unchanged, X_train,y_train, 
           verbose):
  
  """ This function is very important because return the fittest individual 
  in all generations(historical) with different combinations of parameters.
  return Ragnarok-> score, genotype, minimun and optimun hyperparameters...
  """

  allin = [pop_sizes, generations, tournaments_sizes, mutations, crossovers, mate_indpb, mutate_indpb]
  Hyperparameters = list(itertools.product(*allin)) ## Possible combinations to hyperparameters
  register={} # This is the parameter of ragnarok!
  for hyper in Hyperparameters:
      #print("--"*20)
      #print('Combination of parameters',hyper)
      #print("--"*20)
      fittest_individual, fittest, stats = GaFeatureSelection(
                    classifier,
                    population_size=hyper[0],
                    max_generations=hyper[1],
                    tournament_size= hyper[2],
                    MUTPB=hyper[3],
                    CXPB=hyper[4], 
                    mate_indpb=hyper[5],
                    mutate_indpb=hyper[6],
                    ineq_measure=ineq_measure,
                    ineq_min  = ineq_min,
                    limit_unchanged=limit_unchanged,
                    X_train = X_train, 
                    y_train = y_train,
                    k_folds = k_folds,
                    verbose=verbose)
      register[hyper] = [fittest, fittest_individual]
  # Return the score, the fittest genotype, the combinatios of hyperparameters...
  #return Ragnarok(register)
  ### Second phase

  return register