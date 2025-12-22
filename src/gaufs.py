import random
import pandas as pd
from src.clustering_algorithms import HierarchicalExperiment
from src.evaluation_metric import *

class Gaufs:
    def __init__(
        self, 
        seed=random.randint(0, 10000),
        unlabeled_data=None,
        num_genetic_executions=1,
        ngen=150,
        npop=1500,
        cxpb=0.8,
        cxpb_rest_of_genes=0.5,
        mutpb=0.1,
        convergence_generations=50,
        hof_size=None,
        hof_alpha_beta=(0.1, 0.2),
        clustering_method=HierarchicalExperiment(linkage='ward'),
        evaluation_metric=SilhouetteScore(),
        cluster_number_search_band=(2, 26),
        max_number_selections_for_ponderation=None
    ):
        """
        Initialize the GAUFS (Genetic Algorithm for Unsupervised Feature Selection) instance.
             
        Args:
            unlabeled_data (pd.DataFrame or None): The input dataset without labels. If None, creates empty DataFrame.
            num_genetic_executions (int): Number of times to run the genetic algorithm. Default: 1. Range: >= 1.
            seed (int): Random seed for reproducibility. Default: Random integer between 0 and 10000.
            ngen (int): Number of generations for the genetic algorithm. 
                Default: 150 (auto: 150 if num_vars <= 100, else 300). Range: >= 1.
            npop (int): Population size for the genetic algorithm. 
                Default: 1500 (auto: 1500 if num_vars <= 100, else 7000). Range: >= 1.
            cxpb (float): Crossover probability for genetic operations. Default: 0.8. Range: [0.0, 1.0].
            cxpb_rest_of_genes (float): Crossover probability for the rest of generations after initial ones. Default: 0.5. Range: [0.0, 1.0].
            mutpb (float): Mutation probability for genetic operations. Default: 0.1. Range: [0.0, 1.0].
            convergence_generations (int): Number of generations without improvement before stopping. Default: 50. Range: >= 1.
            hof_size (int or None): Hall of Fame size as an absolute number of best solutions to retain.
                If provided, overrides hof_alpha_beta automatic calculation.
                Default: None (uses hof_alpha_beta formula). Range: >= 1 or None.
            hof_alpha_beta (tuple): (alpha, beta) parameters for automatic Hall of Fame size calculation.
                Formula: hof_size = npop * (beta - (beta - alpha) * log(2) / log(num_vars + 1))
                - Alpha (first value): Minimum fraction of population for HOF (when num_vars is small)
                - Beta (second value): Maximum fraction of population for HOF (when num_vars is large)
                This adaptive formula increases HOF size as feature count increases.
                Default: (0.1, 0.2). Range: [0.0, 1.0] for each, beta >= alpha.
                Only used if hof_size is None.
            clustering_method: Clustering algorithm instance. Default: HierarchicalExperiment(linkage='ward'). Must implement clustering interface.
            evaluation_metric: Metric for evaluating clustering quality. Default: SilhouetteScore(). Must implement evaluation interface.
            cluster_number_search_band (tuple): Range of cluster numbers to explore as (min, max_exclusive). Default: (2, 26). Range: (>= 2, <= num_samples).
            max_number_selections_for_ponderation (int or None): Maximum selections from Hall of Fame for weight computation. Default: 2 * num_vars. Range: >= 1 or None.
        """
        # Random seed for reproducibility (randomly generated integer between 0 and 10000)
        self.seed = seed
        random.seed(self.seed)

        # Number of independent genetic algorithm executions to perform
        # Higher values provide more robust results but increase computation time
        self.num_genetic_executions = num_genetic_executions
        
        # Maximum number of generations for the genetic algorithm
        # Higher values allow more evolution but increase computation time
        self.ngen = ngen
        
        # Population size for the genetic algorithm
        # Larger populations explore solution space better but require more computation
        self.npop = npop
        
        # Crossover probability - likelihood of combining two parent solutions
        # Typical range: 0.6-0.9. Higher values promote exploration
        self.cxpb = cxpb

        # Crossover probability for the rest of genes after the cluster number gene
        self.cxpb_rest_of_genes = cxpb_rest_of_genes
        
        # Mutation probability - likelihood of randomly altering a solution
        # Typical range: 0.01-0.2. Higher values increase diversity
        self.mutpb = mutpb
        
        # Number of generations without improvement before early stopping
        # Helps prevent unnecessary computation when convergence is reached
        self.convergence_generations = convergence_generations
        
        # Alpha and Beta parameters for automatic Hall of Fame size calculation
        # hof_size = npop*(alpha-(alpha-beta)*log(2)/log(n_variables+1))
        self.hof_alpha_beta = hof_alpha_beta
        
        # Clustering algorithm instance used to evaluate feature subsets
        # Default: HierarchicalExperiment (hierarchical clustering)
        self.clustering_method = clustering_method
        
        # Evaluation metric for assessing clustering quality
        # Default: SilhouetteScore (measures cluster cohesion and separation)
        self.evaluation_metric = evaluation_metric = evaluation_metric
        
        # Range of cluster numbers to test: [min, max) - max is exclusive
        # Default: tests 2 to 25 clusters. Should be based on domain knowledge
        self.cluster_number_search_band = cluster_number_search_band
        
        # Maximum number of top solutions from Hall of Fame used for feature weight calculation
        # Default: None (will be set to 2 * number of features)
        self.max_number_selections_for_ponderation = max_number_selections_for_ponderation
        
        # Input dataset and derived properties
        if unlabeled_data is None:
            # Empty DataFrame when no data provided
            self.unlabeled_data = pd.DataFrame()
            # Number of features/variables in the dataset
            self.num_vars = None
        else:
            self.set_unlabeled_data(unlabeled_data,recompute_default_parameters=True)

        # If hof_size is provided it has preference over hof_alpha_beta
        if hof_size is not None:
            # Hall of Fame size as an absolute number
            # Stores the best solutions found during evolution
            self.hof_size = hof_size

            

    def set_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)

    def set_unlabeled_data(self, unlabeled_data, recompute_default_parameters=True):
        self.unlabeled_data = unlabeled_data.copy()
        self.num_vars = self.unlabeled_data.shape[1]

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

            # hof_size = npop*(beta-(beta-alpha)*log(2)/log(n_variables+1))
            beta = self.hof_alpha_beta[1]
            alpha = self.hof_alpha_beta[0]
            tam = int(self.npop*(beta-(beta-alpha)*np.log(2)/np.log((self.unlabeled_data.shape[1]+1))))
            self.hof_size = tam if tam != 0 else 1

            # Set the evaluation metric's unlabeled data
            self.evaluation_metric.unlabeled_data=self.unlabeled_data


    def read_unlabeled_data_csv(self, filepath, recompute_default_parameters=True):
        self.unlabeled_data = pd.read_csv(filepath)
        self.num_vars = self.unlabeled_data.shape[1]

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
        if self.unlabeled_data.empty:
            raise ValueError("Unlabeled data is not loaded. Please load the data before running the model.")
        
        self.run_genetic_search()

        pass

    def run_genetic_search(self):
        print("Running genetic search with seed:", self.seed)
        pass    