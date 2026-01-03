import pandas as pd
import numpy as np
import copy
import os
import shutil
import warnings

# Contingency table
@staticmethod
def get_cont_table(c1, c2, round=True):
    res = pd.crosstab(c1, c2, normalize='index') * 100
    return res.round(1) if round else res


# Evaluate an individual (cluster number and feature selection)
@staticmethod
def evaluate_ind(unlabeled_data, cluster_number, variables, clustering_method, evaluation_metric):
    """
    Evaluates an individual in the genetic algorithm.
    Args:
        unlabeled_data (pd.DataFrame): The unlabeled data to cluster.
        cluster_number (int): The number of clusters to form.
        variables (list): A binary list indicating selected variables (1 for selected, 0 for not).
        clustering_method (object): An instance of a clustering method with a 'run' method.
        evaluation_metric (object): An instance of an evaluation metric with a 'compute' method.
    Returns:
        float: The evaluation score of the clustering result.
    """
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

@staticmethod
def get_dictionary_num_clusters_fitness(unlabeled_data, variable_selection,cluster_number_search_band,clustering_method, evaluation_metric):
    """
    Computes a dictionary mapping the number of clusters whithin the cluster_number_search_band (min_inclusive, max_exclusive) to their corresponding fitness scores given the fixed variable selection binary_variable_selection.
    """
    dicc_clusters_fitness = {}
    for k in range(cluster_number_search_band[0], cluster_number_search_band[1]):
        dicc_clusters_fitness[k] =  evaluate_ind(unlabeled_data=unlabeled_data, cluster_number=k, variables=variable_selection, clustering_method=clustering_method, evaluation_metric=evaluation_metric)
    return dicc_clusters_fitness

#the overhead of not computing the maximum in get_dictionary_num_clusters_fitness is negligible compared to the clustering computations
@staticmethod
def get_num_clusters_with_best_fitness(dicc_clusters_fit):
    """
    Given a dictionary mapping number of clusters to fitness scores, returns the number of clusters with the best fitness.
    In case of tie, returns the highest number of clusters among those with the best fitness.
    """
    num_clusters_for_maximum_fitness=None
    max_fitness=None
    for key in dicc_clusters_fit.keys():
        #In case of tie we select the highest number of clusters
        if(max_fitness==None or max_fitness<dicc_clusters_fit[key] or (max_fitness==dicc_clusters_fit[key] and num_clusters_for_maximum_fitness<key)):
            max_fitness=dicc_clusters_fit[key]
            num_clusters_for_maximum_fitness=key
    return num_clusters_for_maximum_fitness, max_fitness

@staticmethod
def clear_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Remove all contents of the directory
        shutil.rmtree(directory_path)
    
    # Recreate the empty directory
    os.makedirs(directory_path)

@staticmethod
def get_variables_over_threshold(variables_weights,threshold):
    return [1 if w>=threshold else 0 for w in variables_weights]

@staticmethod
def min_max_normalize_dictionary(dictionary):
    """
    Returns a new dictionary with the values min-max normalized to the range [0, 1].
    """
    values=list(dictionary.values())
    max_value=max(values)
    min_value=min(values)
    # To avoid division by zero
    if max_value==min_value:
        warnings.warn("Trying to MinMax normalize a dictionary whose values are all the same. Returning zeros for all keys.")
        return {k:0.0 for k in dictionary.keys()}

    return {k:(v-min_value)/(max_value-min_value) for k,v in dictionary.items()}

@staticmethod
def read_unlabeled_data_csv(filepath):
    """
    Reads unlabeled data from a CSV file.
    Accepts CSVs with or without header.
    Expected shape: (n_samples, n_features)
    """
    # Try reading with header
    df = pd.read_csv(filepath)

    # If the first row is numeric, there was no real header
    if df.columns.to_list()[0].startswith("Unnamed") or all(
        c.replace('.', '', 1).isdigit() for c in df.columns
    ):
        df = pd.read_csv(filepath, header=None)

    # Force numeric (important for ML pipelines)
    df = df.apply(pd.to_numeric, errors="raise")
    return df

@staticmethod
def read_labeled_data_csv(filepath):
    """
    Accepts CSV with or without header.
    The fotmat of the labeled data CSV is:
    - First n-1 columns: features
    - Last column: true labels
    Returns:
    - unlabeled_data: pd.DataFrame of shape (n_samples, n_features)
    - true_labels: np.array of shape (n_samples,)
    """
    df = pd.read_csv(filepath)

    # If the first row is numeric, there was no real header
    if df.columns.to_list()[0].startswith("Unnamed") or all(
        c.replace('.', '', 1).isdigit() for c in df.columns
    ):
        df = pd.read_csv(filepath, header=None)

    # Force numeric (important for ML pipelines)
    df = df.apply(pd.to_numeric, errors="raise")

    unlabeled_data = df.iloc[:, :-1]
    true_labels = df.iloc[:, -1].values

    return unlabeled_data, true_labels