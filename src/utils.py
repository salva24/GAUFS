import pandas as pd
import numpy as np
import copy

# Contingency table
@staticmethod
def get_cont_table(c1, c2, round=True):
    res = pd.crosstab(c1, c2, normalize='index') * 100
    return res.round(1) if round else res


# Evaluate an individual (cluster number and feature selection)
@staticmethod
def evaluate_ind(unlabeled_data, cluster_number, variables, clustering_method, evaluation_metric):
    try:
        # if no variables are selected, return a very low fitness
        if np.all(np.array(variables) == 0):
            return -10000000000
        
        filtered_vars = [var for var,i in zip(unlabeled_data.columns,variables) if i == 1]
        filtered_data = unlabeled_data[filtered_vars]
        
        # Create a copy of the provided clustering experiment
        # deepcopying should not be necessary because clustering_method.unlabeled_data 
        # should be None if running the GA, but to be safe we do it in case this method is used elsewhere
        experiment = copy.deepcopy(clustering_method)
        #copy the filtered unlabeled data
        experiment.set_unlabeled_data(filtered_data)
        # Set the number of clusters
        experiment.n_clusters = cluster_number
        # Run the clustering algorithm
        experiment.run()
        #get the results
        assigned_clusters = experiment.assigned_clusters

        # Evaluate the clustering result with the provided metric
        ev = evaluation_metric.compute(assigned_clusters)

        return ev
    
    except Exception as e:
        print(f'Error evaluating the individual with {cluster_number} clusters and the selection {variables}; Exception: {type(e).__name__} - {e}')

@staticmethod
def compute_variable_significance(num_variables, hof_counter, max_number_selections_for_ponderation):
    '''
    Computation of variable significance based on the Hall of Fame individuals and their evaluation scores.
    The values between 0. and 1. are based on the score and the number of times the individual was selected in the HoF.
    
    Args:
        num_variables (int): Number of variables in the dataset.
        hof_counter (dict): Dictionary where keys are binary tuples representing variable selections
                            and values are tuples of (score, selection_count). Where score is the best fitness 
                            achieved for a chromosome that includes that variable selection, and selection_count 
                            is the number of times a chromosome which includes that selection entered the Hall of Fame.
        max_number_selections_for_ponderation (int): Maximum number of individuals to consider for ponderation.
    '''
    selections=sorted(hof_counter.items(), key=lambda item: item[1][0], reverse=True)[:max_number_selections_for_ponderation]
    scores=[]
    total=0
    for it in selections:
        # the score of the selection if the maximun fitness achieved by that selection multiplied by the number of times it was selected in the HoF
        score=it[1][0]*it[1][1]
        scores.append(score)
        total+=score

    # normalize scores so that they sum to 1
    scores_normalized=[x/total for x in scores]
    res = [0]*num_variables
    for i in range(num_variables):
        for j,s in enumerate(scores_normalized):
            res[i] += s * selections[j][0][i]
    return res