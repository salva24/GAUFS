import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
from deap import base, tools, algorithms, creator
import matplotlib.pyplot as plt
from src.utils import evaluate_ind
import os

class GeneticSearch:
    def __init__(self, unlabeled_data, ngen, npop, cxpb, cxpb_rest_of_genes, mutpb, convergence_generations, 
                 hof_size, clustering_method, evaluation_metric, cluster_number_search_band, 
                 verbose, path_store_log=None, path_store_plot=None, seed= random.randint(0,10000)):
        self.unlabeled_data = unlabeled_data.copy()
        self.ngen = ngen
        self.npop = npop
        self.cxpb = cxpb
        self.cxpb_rest_of_genes = cxpb_rest_of_genes
        self.mutpb = mutpb
        self.convergence_generations = convergence_generations
        self.hof_size = hof_size
        self.clustering_method = clustering_method
        self.evaluation_metric = evaluation_metric
        self.cluster_number_search_band = cluster_number_search_band
        self.verbose = verbose
        #If None then no log will be stored
        self.path_store_log = path_store_log
        self.store_log = path_store_log is not None
        #If None no graph will be plotted
        self.graph_evolution = path_store_plot is not None
        self.path_store_plot = path_store_plot
        self.seed = seed

        # Hall of Fame with the best chromosomes found (num_clusters + variable_selection)
        self.hof=None

        #dictionary={num_cluster:(max_fitness asociated,chromosome_where_it_was_achieved)}
        self.num_clusters_and_its_max_fitness=dict()
        #dictionary={variable_selection:(maximum fitness found for that selection (regardeless of the cluster number), number of times a chromosome with this selection entered in the hof)}
        self.hof_counter=dict()
        self.hof_weighted=dict()

        #logbook to store statistics
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "nevals", "avg", "std", "min", "max"


    ## Function for encoding individuals in the genetic algorithm
    def init_individual(self, container, attr_bool, data):
        """
        For initializing individuals in the genetic algorithm.
        First gene: number of clusters (random number within cluster_number_search_band)
        Remaining genes: binary (each gene i represents whether the i-th variable is considered for clustering)
        """
        ## To initialize the number of clusters within the desired range
        num_clusters_range = range(self.cluster_number_search_band[0],self.cluster_number_search_band[1])
        k =  random.choice(num_clusters_range)

        # Creation of the rest of the individual with binary genes
        individual = container([k]+[attr_bool() for _ in range(data.shape[1]-1)])
        return individual

    ### Adapted definition of genetic operators
    def cxUniformModified(self, ind1, ind2):
        """
        Uniform crossover with exchange of the first gene (number of clusters)
        and the rest of the binary genes.
        """
        # Crossover of the first gene (number of clusters)
        ind1[0], ind2[0] = ind2[0], ind1[0]

        # Uniform crossover for the rest of the genes
        for i in range(1, len(ind1)):
            # Exchange occurs in the remaining genes with probability cxpb_rest_of_genes
            if random.random() < self.cxpb_rest_of_genes:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    def mutFlipBitModified(self, ind):
        """
        Modified bit-by-bit mutation so that the number of clusters
        does not go outside the established range
        """
        mutation_type = random.choice([1,2,3])
        if mutation_type == 1:
            num_clusters_range = range(self.cluster_number_search_band[0],self.cluster_number_search_band[1])
            # We choose a new number of clusters within the allowed range
            ind[0] = random.choice(num_clusters_range)
        elif mutation_type == 2:
            i = random.choice(range(1, len(ind)))
            # Usual bit-flip mutation
            ind[i] = 1 - ind[i]
        else:
            num_clusters_range = range(self.cluster_number_search_band[0],self.cluster_number_search_band[1])
            # We choose a new number of clusters within the allowed range
            ind[0] = random.choice(num_clusters_range)
            i = random.choice(range(1, len(ind)))
            # Usual bit-flip mutation
            ind[i] = 1 - ind[i]
        # A tuple must be returned
        return ind,
    
    def evaluate_individual(self, individual):
        num_clusters = individual[0]
        variables = individual[1:]
        # We need to return a tuple
        return (evaluate_ind(self.unlabeled_data, num_clusters, variables, self.clustering_method, self.evaluation_metric),)


    def map_function(self, func, *args):
        with ProcessPoolExecutor() as executor:
            return list(executor.map(func, *args))
        
    def run(self):

        #Check that the numer of clusters search band is valid. The number of clusters cannot be bigger than the number of data points
        if self.cluster_number_search_band[0]>=self.cluster_number_search_band[1]:
            raise ValueError("The cluster_number_search_band is not valid. The first element must be less than the second one.")
        if self.cluster_number_search_band[1]>self.unlabeled_data.shape[0]:
            raise ValueError("The cluster_number_search_band is not valid. The maximum number of clusters cannot be greater than the number of data points.")

        random.seed(self.seed)

        # Definition of the optimization problem type
        creator.create("Fitness", base.Fitness, weights=(1.,))
        creator.create("Individual", list, fitness=creator.Fitness)

        # Initialization of Deap's toolbox
        toolbox = base.Toolbox()

        # Creation of each boolean gene of a chromosome
        toolbox.register("attr_bool", random.randint, 0, 1)

        # Definition of each chromosome (individual)
        toolbox.register("individual", lambda: self.init_individual(creator.Individual, toolbox.attr_bool, self.unlabeled_data))

        # The population is a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the evaluation function, crossover, mutation, and selection operators
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("mate", self.cxUniformModified)
        toolbox.register("mutate", self.mutFlipBitModified)
        toolbox.register("select", tools.selTournament, tournsize=10)

        # Parallel evaluation to speed up computations – DOES NOT WORK UNLESS python -m scoop IS USED
        toolbox.register("map", self.map_function)

        # Fitness statistics per generation
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

        # The mean, standard deviation, minimum, and maximum fitness per generation will be shown
        stats_fit.register("avg", np.mean, axis=0)
        stats_fit.register("std", np.std, axis=0)
        stats_fit.register("min", np.min, axis=0)
        stats_fit.register("max", np.max, axis=0)

        # Creation of an initial population
        population = toolbox.population(self.npop)

        # List that will store the best individuals generation by generation
        best_inds = []
        # List that will store the average fitness of individuals in each generation
        avg_fitness_history = []

        # Initialization of the HallOfFame
        self.hof = tools.HallOfFame(self.hof_size)
        
        # Initial evaluation of individuals
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        # update the num_clusters_and_its_max_fitness dictionary with the maximum fitness for each cluster number
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            val=fit[0]
            cluster_number = ind[0]
            if cluster_number in self.num_clusters_and_its_max_fitness:
                if self.num_clusters_and_its_max_fitness[cluster_number][0]<val:
                    self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)
            else:
                self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)

        # Update the Hall of Fame with the best individuals
        self.hof.update(population)

        # Update the hof_counter dictionary with the best variable selections found in the hof regardeless of the cluster number
        for ind in self.hof:
            variables_selection = tuple(ind[1:])
            if variables_selection in self.hof_counter:
                old=self.hof_counter[variables_selection]
                # we store the maximum fitness found for that variable selection and the number of times it has been present in the hof
                self.hof_counter[variables_selection] = (max(old[0],ind.fitness.values[0]),old[1]+1)
            else :
                self.hof_counter[variables_selection] = (ind.fitness.values[0],1)
        
        # Definition of variables to search for convergence in the HoF
        hof_unchanged_count=0 
        latest_hof_snapshot = set(tuple(ind) for ind in self.hof)
        
        # Update the lists for best individuals and average fitness per generation
        if self.graph_evolution:
            best_inds.append(self.hof[0]) # Inicialmente
            avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in population]))

        # Register statistics of the initial population and print them
        record = stats_fit.compile(population)        
        self.logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if self.verbose:
            print(self.logbook.stream)

        # Run the evolutionary algorithm
        for gen in range(self.ngen):

            # Select the next generation of individuals
            offspring = toolbox.select(population, len(population) - self.hof_size)

            # Mutation and reproduction of individuals
            offspring = algorithms.varAnd(offspring, toolbox, self.cxpb, self.mutpb)

            # Evaluation of the individuals with invalid fitness which are the new ones
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  

            # update the num_clusters_and_its_max_fitness dictionary with the maximum fitness for each cluster number
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                val=fit[0]
                cluster_number = ind[0]
                if cluster_number in self.num_clusters_and_its_max_fitness:
                    if self.num_clusters_and_its_max_fitness[cluster_number][0]<val:
                        self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)
                else:
                    self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)

            # Elitism: Extend the population with the individuals from the Hall of Fame
            offspring.extend(self.hof.items)

            # Update the Hall of Fame with the best individuals found so far
            self.hof.update(offspring)

            # Update the hof_counter dictionary with the best variable selections found in the hof regardeless of the cluster number
            for ind in self.hof:
                variables_selection = tuple(ind[1:])
                if variables_selection in self.hof_counter:
                    old=self.hof_counter[variables_selection]
                    self.hof_counter[variables_selection] = (max(old[0],ind.fitness.values[0]),old[1]+1)#maximo fitness y cuento las veces que ha entrado
                else :
                    self.hof_counter[variables_selection] = (ind.fitness.values[0],1)

            # If we are in the last generation, store the individuals and their fitness values
            if(gen==self.ngen-1):
                self.last_gen={tuple(ind):ind.fitness.values[0] for ind in offspring}

            # Replace the current population with the offspring
            population[:] = offspring

            # Update the lists for best individuals and average fitness per generation
            if self.graph_evolution:
                best_inds.append(self.hof[0])
                avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in population]))

            record = stats_fit.compile(population)
            self.logbook.record(gen=gen+1, nevals=len(invalid_ind), **record)

            # Output and store of statistics
            if self.verbose:
                print(self.logbook.stream)
            if self.store_log:
                self.save_log_to_file(early_stopped=False, current_gen=self.ngen-1)
            
            #if the hof has not changed in the last convergence_generations generations, we stop the algorithm
            current_hof_snapshot = set(tuple(ind) for ind in self.hof)
            if current_hof_snapshot == latest_hof_snapshot:
                hof_unchanged_count += 1
            else:
                hof_unchanged_count = 0
                latest_hof_snapshot = current_hof_snapshot
            if hof_unchanged_count >= self.convergence_generations:
                if self.verbose:
                    print(f'Early Stopping due to Hall Of Fame not changing in {gen+1} generations')
                self.last_gen={tuple(ind):ind.fitness.values[0] for ind in offspring}
                if self.store_log:
                    self.save_log_to_file(early_stopped=True, current_gen=gen)
                break

        # Plot the evolution of the best and average fitness score
        if self.graph_evolution:
            self.save_graph_evolution(best_inds, avg_fitness_history)

        return self.hof_counter, self.num_clusters_and_its_max_fitness
    
    def save_graph_evolution(self, best_inds, avg_fitness_history):
        fitness_vals = []
        for _, ind in enumerate(best_inds, start=1):
            fitness_vals.append(ind.fitness.values[0])
        num_generations = len(fitness_vals)  
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_generations + 1), fitness_vals, marker='o', linestyle='-', color='b', label='Fitness of the best individual')
        plt.plot(range(1, num_generations + 1), avg_fitness_history[:num_generations], marker='x', linestyle='--', color='r', label='Average fitness of the population')
        plt.title('Fitness Evolution Over Generations')
        plt.xlabel('Number of Generations')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.path_store_plot+"evolution_of_the_genetic_algorithm.png", dpi=300, bbox_inches="tight")
        plt.close()

    def save_log_to_file(self, current_gen, early_stopped):
        """Save the logbook to a file."""
        # Create directory if it doesn't exist
        os.makedirs(self.path_store_log, exist_ok=True)
        
        with open(self.path_store_log + "genetic_algorithm_log.txt", 'w') as f:
            if early_stopped:
                f.write(f"Early stopping at generation {current_gen + 1}\n\n")
            f.write(str(self.logbook))