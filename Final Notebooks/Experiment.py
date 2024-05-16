# Models and experiments...
from ineq import Theil
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from GeneticNeuralNetwork import gaMLP_Entropy
from LocalSearch import SA_min
from LocalSearch import search
from NeuralNetwork import MLP
from LocalSearch import ModelArchitecture
from LocalSearch import assign_weights
from LocalSearch import SA_min
from LocalSearch import search
import tensorflow as tf
from NeuralNetwork import MLP
from tensorflow.keras import Sequential
import itertools


def specificity_score(y_true, y_pred):
    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Extraer los valores de la matriz de confusión
    tn, fp, fn, tp = conf_matrix.ravel()
    # Calcular la especificidad
    specificity = tn / (tn + fp)
    return specificity


# This is the score for feature selection
def eval_model(classifier, X_train_, y_train_, k_folds,):
    # Take in mind that classifier is classifier
    # model = DecisionTreeClassifier()
    X_train, y_train, scores = X_train_.copy(), y_train_.copy() , []
    X_train, y_train = X_train.to_numpy() , y_train.to_numpy()
    metrics = {}
    metrics['f1'] = []
    metrics['accuracy'] = []
    metrics['specificty'] = []
    metrics['sensitivity-recall'] = []
    metrics['precision'] = []
    skf = StratifiedKFold(n_splits=k_folds, random_state=123, shuffle=True)
    for itrain, itest in skf.split(X_train, y_train):
        Xi_train, Xi_test = X_train[itrain], X_train[itest]
        yi_train, yi_test = y_train[itrain], y_train[itest]
        classifier.fit(Xi_train, yi_train)
        preds = classifier.predict(Xi_test)
        metrics['f1'].append(f1_score(yi_test, preds))
        metrics['accuracy'].append(accuracy_score(yi_test, preds))
        metrics['specificty'].append(specificity_score(yi_test, preds))
        metrics['sensitivity-recall'].append(recall_score(yi_test, preds))
        metrics['precision'].append(precision_score(yi_test, preds))

        #print(len(yi_train), len(Xi_train), len(Xi_test), len(yi_test))
    return metrics



#############################
#############################################################
############################################################################################
############################################################################################
# Now we are goint to make the model with fit and predict ####
###################################################################


def eval_model_GeneticNN(classifier, X_train_, y_train_, k_folds):
    # Take in mind that classifier is classifier
    # model = DecisionTreeClassifier()
    X_train, y_train  = X_train_.copy(), y_train_.copy()
    X_train, y_train = X_train.to_numpy() , y_train.to_numpy()
    evaluations = {}
    evaluations['ga'] = {'f1':[], 'accuracy':[], 'specificty':[], 'sensitivity-recall':[], 'precision':[]}
    evaluations['annealing'] = {'f1':[], 'accuracy':[], 'specificty':[], 'sensitivity-recall':[], 'precision':[]}
    evaluations['backprop'] = {'f1':[], 'accuracy':[], 'specificty':[], 'sensitivity-recall':[], 'precision':[]}
    skf = StratifiedKFold(n_splits=k_folds, random_state=123, shuffle=True)
    for itrain, itest in skf.split(X_train, y_train):
        Xi_train, Xi_test = X_train[itrain], X_train[itest]
        yi_train, yi_test = y_train[itrain], y_train[itest]
        classifier.fit(Xi_train, yi_train)
        for kind in ['ga', 'annealing', 'backprop']:
            preds = classifier.predict(Xi_test, solution=kind)
            evaluations[kind]['f1'].append(f1_score(yi_test, preds))
            evaluations[kind]['accuracy'].append(accuracy_score(yi_test, preds))
            evaluations[kind]['specificty'].append(specificity_score(yi_test, preds))
            evaluations[kind]['sensitivity-recall'].append(recall_score(yi_test, preds))
            evaluations[kind]['precision'].append(precision_score(yi_test, preds))
        #print(len(yi_train), len(Xi_train), len(Xi_test), len(yi_test))
    return evaluations

def mean_results(results, kind, metric):
    return np.array(results[kind][metric]).mean()


    
#### Cross-validation
# GridSearch over the f1-metric...

def hypersearch_GeneticNN(X_train,
                          y_train,
                         architecture,
                         pop_sizes, 
                         generations, 
                         tournaments_sizes,
                         mutations, 
                         crossovers,
                         mate_indpb,
                         mutate_indpb,
                         limit_unchanged =35,
                         seed=20):
    # This function optimize the parameters for ga solution algorithm....
    allin = [pop_sizes, generations, tournaments_sizes, mutations, crossovers, mate_indpb, mutate_indpb]
    register = {}
    Hyperparameters = list(itertools.product(*allin)) ## Possible combinations to hyperparameters
    for hyper in Hyperparameters:
        print("Verbose:",hyper)
        fittest_, history, stats = gaMLP_Entropy(architecture=architecture, 
                            X_train = X_train,
                            y_train = y_train,
                            seed_random=seed,
                            ineq_measure=Theil,
                            limit_unchanged = limit_unchanged,
                            population_size=hyper[0],
                            max_generations=hyper[1],
                            tournament_size=hyper[2],
                            MUTPB= hyper[3],
                            CXPB= hyper[4],
                            mate_indpb = hyper[5],
                            mutate_indpb = hyper[6])
        register[hyper] = [fittest_,stats]

    return register




def eval_model_NN(classifier,
                  X_train_,
                  y_train_,
                  epochs,
                  k_folds):
    # Take in mind that classifier is classifier
    # model = DecisionTreeClassifier()
    X_train, y_train  = X_train_.copy(), y_train_.copy()
    X_train, y_train = X_train.to_numpy() , y_train.to_numpy()
    evaluations = {'f1':[], 'accuracy':[], 'specificty':[], 'sensitivity-recall':[], 'precision':[]}
    skf = StratifiedKFold(n_splits=k_folds, random_state=123, shuffle=True)
    for itrain, itest in skf.split(X_train, y_train):
        Xi_train, Xi_test = X_train[itrain], X_train[itest]
        yi_train, yi_test = y_train[itrain], y_train[itest]
        classifier.fit(Xi_train,
                       yi_train,
                       epochs=epochs)
        preds = np.where(classifier.predict(Xi_test)>0.5,1,0) # Predictions ->[0,1]
        evaluations['f1'].append(f1_score(yi_test, preds))
        evaluations['accuracy'].append(accuracy_score(yi_test, preds))
        evaluations['specificty'].append(specificity_score(yi_test, preds))
        evaluations['sensitivity-recall'].append(recall_score(yi_test, preds))
        evaluations['precision'].append(precision_score(yi_test, preds))
        #print(len(yi_train), len(Xi_train), len(Xi_test), len(yi_test))
    return evaluations

def mean_results_NN(results, metric):
    return np.array(results[metric]).mean()

