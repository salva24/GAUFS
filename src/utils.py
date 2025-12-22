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

