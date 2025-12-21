import random
import pandas as pd
from clustering_algorithms import HierarchicalExperiment

class Gaufs:
    def __init__(self, unlabeled_data=None):
        # Set random seed
        self.seed = random.randint(0, 10000)

        #gentic algorithm parameters
        self.num_genetic_executions = 1
        self.ngen = 150
        self.npop = 1500
        self.cxpb = 0.8
        self.mutpb = 0.1
        self.convergence_generations = 50
        self.hof_size = (0.1, 0.2)
        self.clustering_method = HierarchicalExperiment()
        # cluster numbers range considered, the second value is exclusive
        self.cluster_number_search_band = (2, 26)
        # The maximun number of top selections from the Hall of Fame considered to compute the variable weights
        self.max_number_selections_for_ponderation = None
        
        if unlabeled_data is None:
            self.unlabeled_data= pd.DataFrame()
            self.num_vars=None
        else:
            self.unlabeled_data = unlabeled_data
            self.num_vars= self.unlabeled_data.shape[1]
            # By default, set the max number of selections for ponderation as twice the number of variables
            self.max_number_selections_for_ponderation = 2 * self.num_vars
        



    def set_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)

    def set_unlabeled_data(self, unlabeled_data, recompute_default_parameters=True):
        self.unlabelaed_data = unlabeled_data
        self.num_vars = self.unlabelaed_data.shape[1]

        # Set the genetic algorithm parameters based on the number of variables
        if recompute_default_parameters:
            #number of generations and population size
            if self.num_vars <= 100:
                self.ngen = 150
                self.npop = 1500
            else:
                self.ngen = 300
                self.npop = 7000
            #max number of selections for ponderation
            self.max_number_selections_for_ponderation = 2 * self.num_vars

    def read_unlabeled_data_csv(self, filepath, recompute_default_parameters=True):
        self.unlabelaed_data = pd.read_csv(filepath)
        self.num_vars = self.unlabelaed_data.shape[1]

        # Set the genetic algorithm parameters based on the number of variables
        if recompute_default_parameters:
            #number of generations and population size
            if self.num_vars <= 100:
                self.ngen = 150
                self.npop = 1500
            else:
                self.ngen = 300
                self.npop = 7000
            #max number of selections for ponderation
            self.max_number_selections_for_ponderation = 2 * self.num_vars
    
    def run(self):
        if self.unlabelaed_data.empty:
            raise ValueError("Unlabeled data is not loaded. Please load the data before running the model.")
        
        self.run_genetic_search()

        pass

    def run_genetic_search(self):
        print("Running genetic search with seed:", self.seed)
        pass    