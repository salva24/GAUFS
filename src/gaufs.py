import random
import pandas as pd
from src.clustering_algorithms import HierarchicalExperiment
from src.evaluation_metric import *
from src.genetic_search import GeneticSearch
from src.utils import *
import os
import warnings

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
        fitness_weight_over_threshold=0.5,
        exponential_decay_factor=1.0,
        max_number_selections_for_ponderation=None,
        verbose=True,
        generate_genetics_log_files=True,
        graph_evolution=True,
        generate_files_with_results=True,
        output_directory="./out/"
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
            cluster_number_search_band (tuple): Range of cluster numbers to explore as (min_inclusive, max_exclusive). Default: (2, 26). Range: (>= 2, <= num_samples).
            fitness_weight_over_threshold (float): Weight for fitness vs. threshold in variable weight analysis. Default: 0.5 i.e. both values are averaged. Range: [0.0, 1.0].
            exponential_decay_factor (float): Exponential decay factor for the automatic solution selector in the formula \frac{\delta_i}{1 + \left( \frac{N}{e^{exponential_decay_factor*i}} \right)} If 0.0 there is no decay. Default: 1.0. Range: >= 0.0.
            max_number_selections_for_ponderation (int or None): Maximum selections from Hall of Fame for weight computation. Default: 2 * num_vars. Range: >= 1 or None.
            verbose (bool): Whether to print logs during execution. Default: True.
            generate_genetics_log_files (bool): Whether to generate a log file with Genetic Algorithm execution details. Default: True.
            graph_evolution (bool): Whether to generate a graph of the best and average fitness through the Genetica Algorithm's evolution. Default: True.
            output_directory (str): Path to store generated files including the plots. Default: "../out/".
            generate_files_with_results (bool): Whether to generate files with plots and results. Default: True.
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
        
        # Weighting factor for fitness vs. threshold in variable weight analysis.
        # weight of fitness = fitness_weight_over_threshold
        # weight of threshold = 1 - fitness_weight_over_threshold
        self.fitness_weight_over_threshold = fitness_weight_over_threshold

        # Exponential decay factor for automatic solution selection
        self.exponential_decay_factor = exponential_decay_factor

        # Maximum number of top solutions from Hall of Fame used for feature weight calculation
        # Default: None (will be set to 2 * number of features)
        self.max_number_selections_for_ponderation = max_number_selections_for_ponderation
        
        # Wether generate a graph showing the evolution of the best fitness score in the Genetic Algorithm
        self.graph_evolution = graph_evolution

        # Verbosity flag
        self.verbose = verbose

        # Flag to generate a log file with Genetic Algorithm execution details
        self.generate_genetics_log_files = generate_genetics_log_files

        # Flag to generate files with plots and results
        self.generate_files_with_results = generate_files_with_results

        #Directory to store generated plots, logs and other files
        self.output_directory = output_directory

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

        # Variable significances for each Genetic execuition
        self.variable_significances = []

        # Best chromosome for each Genetic execution
        self.best_chromosomes = []

        #  Each GA instance that has been run in case more specific data is wanted
        self.ga_instances = []

        # Final averaged variable significance.
        self.variable_significance = None

        # the solution must be chosen from the keys and values of this dictionary
        #dictionary={variable_selection:cluster_number_with_best_fitness}
        self.dicc_selection_num_clusters={} 
        #dictionary={variable_selection:best_fitness}
        self.dicc_selection_fitness={}

        # Optimal variable selection and number of clusters after analysis (variable selection as binary list, num_clusters)
        self.optimal_variable_selection_and_num_of_clusters = None
        #Fitness value associated to the optimal variable selection and number of clusters
        self.fitness_of_optimal_variable_selection_and_num_of_clusters = None


            
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
        """
        Reads unlabeled data from a CSV file.
        Accepts CSVs with or without header.
        Expected shape: (n_samples, n_features)
        """

        df=read_unlabeled_data_csv(filepath)

        self.set_unlabeled_data(
            df,
            recompute_default_parameters=recompute_default_parameters
        )
        
    def run(self):

        if self.unlabeled_data.empty:
            raise ValueError("Unlabeled data is not loaded. Please load the data before running the model.")
        
        #delete all the content of the output directory to avoid mixing files from different runs
        clear_directory(self.output_directory)

        #get the variable significances by running the genetic searches
        self.run_genetic_searches()

        #analyze the variable weights to get the optimal variable selection and number of clusters
        self.analyze_variable_weights()

        if self.verbose:
            print(f"Optimal variable selection (1=selected, 0=not selected): {self.optimal_variable_selection_and_num_of_clusters[0]}\nwith the optimal number of clusters: {self.optimal_variable_selection_and_num_of_clusters[1]}")
            print("Fitness of optimal variable selection and number of clusters: ", self.fitness_of_optimal_variable_selection_and_num_of_clusters)
        
        if self.generate_files_with_results:
            # Save optimal variable selection and number of clusters to a text file
            directory= self.output_directory+"results/"
            os.makedirs(directory, exist_ok=True)
            output_path_txt=directory+'optimal_variable_selection_and_number_of_clusters.txt'
            
            df_optimal_selection = pd.DataFrame({
                'Variable': self.unlabeled_data.columns,
                'Selected': self.optimal_variable_selection_and_num_of_clusters[0]
            })

            output_path_csv = os.path.join(directory, 'optimal_variable_selection.csv')
            df_optimal_selection.to_csv(output_path_csv, index=False)

            # Also save summary to text file
            with open(output_path_txt, 'w') as f:
                f.write(f"Optimal number of clusters: {self.optimal_variable_selection_and_num_of_clusters[1]}\n")
                f.write(f"Fitness: {self.fitness_of_optimal_variable_selection_and_num_of_clusters}\n\n")
                f.write("Selected Variables:\n")
                f.write(df_optimal_selection[df_optimal_selection['Selected'] == 1]['Variable'].to_string(index=False))

            if self.verbose:
                print(f"Optimal variable selection and number of clusters saved to {output_path_txt}")

        return self.optimal_variable_selection_and_num_of_clusters, self.fitness_of_optimal_variable_selection_and_num_of_clusters

    def run_genetic_searches(self):

        if self.num_genetic_executions < 1:
            raise ValueError("num_genetic_executions must be at least 1.")
        
        if self.hof_size >= self.npop:
            raise ValueError("hof_size must be less than population size (npop).")
        # Generate unique seeds for each genetic algorithm execution
        seeds_for_GA = random.sample(range(0, 10000), self.num_genetic_executions)

        for s in seeds_for_GA:
            ga_instance = GeneticSearch(
                seed=s,
                unlabeled_data=self.unlabeled_data,
                ngen=self.ngen,
                npop=self.npop,
                cxpb=self.cxpb,
                cxpb_rest_of_genes=self.cxpb_rest_of_genes,
                mutpb=self.mutpb,
                convergence_generations = self.convergence_generations,
                hof_size=self.hof_size,
                clustering_method=self.clustering_method,
                evaluation_metric=self.evaluation_metric,
                cluster_number_search_band=self.cluster_number_search_band,
                verbose=self.verbose,
                path_store_log=self.output_directory+f'GA_Seed_{s}/' if self.generate_genetics_log_files else None,
                path_store_plot=self.output_directory+f'GA_Seed_{s}/' if self.graph_evolution else None
            )
            if self.verbose:
                print(f"Running Genetic Algorithm with seed {s}...")

            hof_counter, _= ga_instance.run()
            # Compute variable significance from Hall of Fame
            variable_significance = compute_variable_significance(
                num_variables=self.num_vars,
                hof_counter=hof_counter,
                max_number_selections_for_ponderation=self.max_number_selections_for_ponderation
            )
            self.variable_significances.append(variable_significance)

            #Store Best chromosome and GA instance although it is not necessary for GAUFS operation
            self.best_chromosomes.append(ga_instance.hof[0])
            self.ga_instances.append(ga_instance)
        
        # Average variable significances across all genetic executions
        self.variable_significance = np.mean(self.variable_significances, axis=0)
        if self.verbose:
            print("Genetic Algorithm executions completed.")
            print("The variable weights (significances) are: ", self.variable_significance)

        if self.generate_files_with_results:
            # Save variable significances to a CSV file
            directory= self.output_directory+"results/"
            os.makedirs(directory, exist_ok=True)
            df_var_significance = pd.DataFrame({
                'Variable': self.unlabeled_data.columns,
                'Significance': self.variable_significance
            })
            output_path_csv=directory+'variable_significances.csv'
            df_var_significance.to_csv(output_path_csv, index=False)
            if self.verbose:
                print(f"Variable significances saved to {output_path_csv}")

        return self.variable_significance
        
    def analyze_variable_weights(self):#si flatten es true a partir del ultimo nuemero de variables ques se selecciona se assigna el valor de cuando se selecciona todas en la gáfica para que se aplane en vez de que se haga interpolación lineal. Si k es 0 no se hace decay
        
        thresholds=sorted(self.variable_significance,reverse=True)
        
        #list of the considered selections  
        possible_variable_selections=[]

        #dictionary={num_variables_selected:[variable_selections_with_that_num_variables]}
        dicc_num_var_selection_with_that_num_variables={}

        #\Psi_{num\_clusters}={num_variables_selected:cluster_number_with_best_fitness}
        dicc_num_var_selected_num_clusters={}

        #\tilde{\Psi}_{fitness}={num_variables_selected:best_fitness}
        dicc_num_var_selected_fitness={}
        #\tilde{\Psi}_{weight\_threshold}={num_variables_selected:threshold_for_selection}
        dicc_num_var_threshold={}
        # MinMax(\tilde{\Psi}_{fitness})
        dicc_num_var_selected_fitness_min_max_normalized=None
        # MinMax(\tilde{\Psi}_{weight\_threshold})
        dicc_num_var_threshold_min_max_normalized=None

        #\Phi_{average}={num_variables_selected:average_between_fitness_and_threshold}
        # the average is weighted by self.fitness_weight_over_threshold
        # \Phi_{average}= self.fitness_weight_over_threshold*MinMax(\tilde{\Psi}_{fitness}) + (1-self.fitness_weight_over_threshold)*MinMax(\tilde{\Psi}_{weight_threshold})
        dicc_num_var_selected_importance=None

        #\delta={num_variables_selected:importance_difference_with_next_selection}
        dictionary_deltas={}

        #\tilde{\delta} = {num_variables_selected:importance_difference_with_next_selection_with_exponential_decay}
        # \tilde{\delta} = \frac{\delta_i}{1 + \left( \frac{N}{e^{self.exponential_decay_factor*i}} \right)}
        dictionary_deltas_with_exponential_decay=None

        #Create the directory
        directory= self.output_directory+"results/"
        os.makedirs(directory, exist_ok=True)

    

        #For each threshold get the selection and analyze it
        for threshold in thresholds:
            selection=get_variables_over_threshold(self.variable_significance,threshold)

            possible_variable_selections.append(selection)
            dicc_clusters_fit=get_dictionary_num_clusters_fitness(unlabeled_data=self.unlabeled_data,variable_selection=selection,clustering_method=self.clustering_method,evaluation_metric=self.evaluation_metric,cluster_number_search_band=self.cluster_number_search_band)
            num_clusters_for_maximum_fitness,max_fitness=get_num_clusters_with_best_fitness(dicc_clusters_fit)
                
            self.dicc_selection_num_clusters[tuple(selection)]=num_clusters_for_maximum_fitness
            dicc_num_var_selected_num_clusters[sum(selection)]=num_clusters_for_maximum_fitness
            dicc_num_var_threshold[sum(selection)]=threshold

            self.dicc_selection_fitness[tuple(selection)]=max_fitness
            dicc_num_var_selected_fitness[sum(selection)]= max_fitness

            dicc_num_var_selection_with_that_num_variables[sum(selection)]=selection


        #there is only one selecction possible: selecting all variables
        if(len(thresholds)==1):
            warnings.warn(
                f'The only selection possible is selecting all variables '
                f'and num_clusters={num_clusters_for_maximum_fitness} '
                f'with fitness={max_fitness}',
                UserWarning
            )
            #the only possible solution is the one in self.dicc_selection_num_clusters
            self.optimal_variable_selection_and_num_of_clusters = possible_variable_selections[0], list(self.dicc_selection_num_clusters.values())[0]
            return self.optimal_variable_selection_and_num_of_clusters, self.fitness_of_optimal_variable_selection_and_num_of_clusters



        # #######################################make the plots and store results

        # #Crear grafica 3D : variables,clusters,silhouette
        # dicc_numvars_selections = {}
        # key_var_orig_selected = 0 #el numero de variables del primer selection que tiene selected todas las variables originales 
        
        # for selection in list(dicc_selection_num_clusters.keys()):
        #     dicc_numvars_selections[sum(selection)] = selection
        #     if (selection[:num_vars_originales] == (1,)*num_vars_originales and key_var_orig_selected == 0):
        #         key_var_orig_selected = sum(selection)

        # x_aux = list(dicc_num_var_selected_num_clusters.keys()) #num_vars
        # y_aux = list(range(2,max_num_considerado_clusters)) #num_clusters


        # ############################################### Parte 3D
        # #Puntos de la grafica 3D
        # dicc_var_cluster_fit = {} #del tipo (numvar,numclusters) : fitness
        # x = []
        # y = []
        # z = []
        # for i in x_aux:
        #     for j in y_aux:
        #         crom = [j] + list(dicc_numvars_selections[i]) #hago list porque es una tupla
        #         k = evaluate_ind(test.data_dummies,crom,fitness,'hierarchical',linkage)[0]
        #         x.append(i)
        #         y.append(j)
        #         z.append(k)
        #         dicc_var_cluster_fit[i,j] = k


        # fig3D= plt.figure()
        # ax3D = fig3D.add_subplot(111, projection='3d')
        # ax3D.plot_trisurf(x, y, z, cmap='viridis') #crea una superficie uniendo los puntos
        # #ax3D.scatter(x,y,z)

        # ax3D.set_title(f'{name} Gráfica 3D : {fitness}, {linkage}')
        # ax3D.set_xlabel('Número de variables')
        # ax3D.set_ylabel('Número de clusters')
        # ax3D.set_zlabel(f'Fitness {fitness}')


        # output_path_3D=directory+f'/{name}_{fitness}_{linkage}_3D.png'
        # plt.savefig(output_path_3D)
        # plt.close(fig3D)

        # ################################save diccionaries
        # json_path = os.path.join(directory, f'{name}_{fitness}_{linkage}.json')

        # data = {
        #     'dicc_selection_num_clusters': {str(k): v for k, v in dicc_selection_num_clusters.items()},
        #     'dicc_selection_fitness': {str(k): v for k, v in dicc_selection_fitness.items()},
        #     'dicc_num_var_selected_ami_asociado':dicc_num_var_selected_ami_asociado,
        #     'dicc_num_var_selected_nmi_asociado':dicc_num_var_selected_nmi_asociado,
        #     'dicc_var_cluster_fit' : {str(k):v for k,v in dicc_var_cluster_fit.items()},
        #     'dicc_num_var_umbral' : {str(k):v for k,v in dicc_num_var_umbral.items()}
        # }


        ##########################################################################################
        
        # This only affects the graphs. We make up solutions for the missing number of variables so that the graphs are continuous but these solutions 
        # do not exist in reality as they do not correspond to any selection. They will not be selected as optimal solutions in any case. 
        max_num_var_with_value=max(dicc_num_var_selected_fitness.keys())
        #from the last one to the first one
        for i in range(max_num_var_with_value - 1, 0, -1):
            if i not in dicc_num_var_selected_fitness.keys():
                #assign it the value of the one which is next to it
                dicc_num_var_selected_fitness[i]=dicc_num_var_selected_fitness[i+1]
                dicc_num_var_threshold[i]=dicc_num_var_threshold[i+1]

        # # Crear una figura y dos ejes (subgráficas)
        # fig, ax = plt.subplots(4,2, figsize=(16,16))  #4 filas
        # fig.suptitle(f'{fitness} {linkage} '+name, fontsize=16)
        # rango_x = list(dicc_num_var_selected_num_clusters.keys())


        # separacion_eje_x = (max(rango_x)+1-min(rango_x))//35 #para que como mucho haya 35 (ajustarlo viendo a partir de que numero se pisan)
        # # Primera: clusters number
        # x1 = list(dicc_num_var_selected_num_clusters.keys())
        # y1 = list(dicc_num_var_selected_num_clusters.values())
        # ax[0,0].set_title("Numero de clusters por selection ")
        # ax[0, 0].plot(x1, y1)
        # ax[0,0].scatter(x1,y1)
        # ax[0, 0].set_xlabel('Variables significativas')
        # ax[0, 0].set_ylabel('num_clusters')
        # ax[0, 0].set_xticks(range(min(x1), max(x1) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X

        # # Segunda: fitness interno  
        # x2 = list(dicc_num_var_selected_fitness.keys())
        # y2 = list(dicc_num_var_selected_fitness.values())
        # x2, y2 = zip(*sorted(zip(x2, y2))) #ordenar
        # ax[1,0].set_title(f'{fitness} por selection')
        # ax[1, 0].plot(x2, y2)
        # ax[1, 0].scatter(x2,y2)
        # ax[1, 0].set_xlabel('Variables significativas')
        # ax[1, 0].set_ylabel(f'fitness {fitness}')
        # int_orig = evaluate_ind(test.data_dummies,[num_clusters]+[1]*num_vars_originales,fitness,'hierarchical',linkage)[0]
        # ax[1,0].axhline(y=int_orig, color='green', linestyle='--', label=f'{fitness}_original')
        # ax[1,0].legend()
        # ax[1, 0].set_xticks(range(min(x2), max(x2) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X


        # # Tercera: ami (primera fila, segunda columna)
        # x3 = list(dicc_num_var_selected_ami_asociado.keys())
        # y3 = list(dicc_num_var_selected_ami_asociado.values())
        # ax[0,1].set_title("AMI asociado al selection")
        # ax[0, 1].plot(x3, y3)
        # ax[0, 1].scatter(x3,y3)
        # ax[0, 1].set_xlabel('Variables significativas')
        # ax[0, 1].set_ylabel('fitness ami asociado')
        # ami_orig = evaluate_ind(test.data_dummies,[num_clusters]+[1]*num_vars_originales,'ami','hierarchical',linkage)[0]
        # ax[0,1].axhline(y=ami_orig, color='green', linestyle='--', label='ami_original')
        # ax[0,1].legend()
        # ax[0, 1].set_xticks(range(min(x3), max(x3) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X


        # # Cuarta: umbrales (segunda fila, segunda columna)
        # x4 = list(dicc_num_var_umbral.keys())
        # y4 = list(dicc_num_var_umbral.values())
        # x4, y4 = zip(*sorted(zip(x4, y4))) #ordenar
        # ax[1,1].set_title("threshold por selection")
        # ax[1, 1].plot(x4, y4)
        # ax[1, 1].scatter(x4,y4)
        # ax[1, 1].set_xlabel('Variables significativas')
        # ax[1, 1].set_ylabel('Umbral')
        # umbral_var_originales = dicc_num_var_umbral[key_var_orig_selected]
        # ax[1,1].axhline(y=umbral_var_originales, color='green', linestyle='--', label='threshold del primer selection con todas las vars originales')
        # ax[1,1].legend()
        # ax[1, 1].set_xticks(range(min(x4), max(x4) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X



        # # Quinta: nmi (tercera fila, primera columna)
        # x5 = list(dicc_num_var_selected_nmi_asociado.keys())
        # y5 = list(dicc_num_var_selected_nmi_asociado.values())
        # ax[2,0].set_title("NMI asociado al selection")
        # ax[2,0].plot(x5, y5)
        # ax[2,0].scatter(x5,y5)
        # ax[2,0].set_xlabel('Variables significativas')
        # ax[2,0].set_ylabel('fitness NMI asociado')
        # nmi_orig = evaluate_ind(test.data_dummies,[num_clusters]+[1]*num_vars_originales,'nmi','hierarchical',linkage)[0]
        # ax[2,0].axhline(y=nmi_orig, color='green', linestyle='--', label='nmi_original')
        # ax[2,0].legend()
        # ax[2,0].set_xticks(range(min(x5), max(x5) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X
        

        #MinMax normalization of fitness and thresholds
        dicc_num_var_selected_fitness_min_max_normalized= min_max_normalize_dictionary(dicc_num_var_selected_fitness)
        dicc_num_var_threshold_min_max_normalized= min_max_normalize_dictionary(dicc_num_var_threshold)

        # Get the considered number of variables sorted (including the fake ones created for continuity)
        keys_sorted=sorted(dicc_num_var_selected_fitness.keys())

        # Compute the average importance
        dicc_num_var_selected_importance= {num_var:self.fitness_weight_over_threshold*dicc_num_var_selected_fitness_min_max_normalized[num_var]+(1-self.fitness_weight_over_threshold)*dicc_num_var_threshold_min_max_normalized[num_var] for num_var in keys_sorted}

        # Compute the differences (deltas)
        for i in range(len(keys_sorted)-1):
            key_current=keys_sorted[i]
            key_next=keys_sorted[i+1]
            dictionary_deltas[key_current]= max(0, dicc_num_var_selected_importance[key_current]-dicc_num_var_selected_importance[key_next])
        # For the last one, the next value is considered 0
        # The max is not necessary here because the importances are possitive but for consistency with the others
        dictionary_deltas[keys_sorted[-1]]= max(0,dicc_num_var_selected_importance[keys_sorted[-1]])

        dictionary_deltas_with_exponential_decay=dictionary_deltas.copy()

        # Add the exponential decay to the deltas if exponential_decay_factor>0
        if self.exponential_decay_factor>0:
            # It divides the ponderations by a factor of 1+((N-1) / (math.exp(exponential_decay_factor * num_var))) where N is the toltal number of variables and num_var is the number of variables selected.
            # The exponential_decay_factor * num_var < 700 avoids overflow
            dictionary_deltas_with_exponential_decay= {num_var: delta/(1+((len(dictionary_deltas)-1) / (np.exp(self.exponential_decay_factor * num_var)))) if self.exponential_decay_factor * num_var < 700 else delta for (num_var, delta) in dictionary_deltas.items()}

        # Use the deltas with exponential decay for selecting the optimal solution
        # Notice that the fake solutions' deltas are 0 and as long as there is a possitive delta they will not be selected
        optimal_num_variables= max(dictionary_deltas_with_exponential_decay, key=dictionary_deltas_with_exponential_decay.get)

        # if a fake solution is selected this is because all the deltas are 0 =>
        # => all the variables have the same importance => all variables are significant
        if optimal_num_variables not in dicc_num_var_selection_with_that_num_variables.keys():
            self.optimal_variable_selection_and_num_of_clusters = [1]*self.num_vars, dicc_num_var_selected_num_clusters[self.num_vars]
            self.fitness_of_optimal_variable_selection_and_num_of_clusters = dicc_num_var_selected_fitness[self.num_vars]
            return self.optimal_variable_selection_and_num_of_clusters, self.fitness_of_optimal_variable_selection_and_num_of_clusters

        self.optimal_variable_selection_and_num_of_clusters = dicc_num_var_selection_with_that_num_variables[optimal_num_variables] , dicc_num_var_selected_num_clusters[optimal_num_variables]
        self.fitness_of_optimal_variable_selection_and_num_of_clusters = dicc_num_var_selected_fitness[optimal_num_variables]
        return self.optimal_variable_selection_and_num_of_clusters, self.fitness_of_optimal_variable_selection_and_num_of_clusters
    

    #         if x_argmax not in dicc_num_var_selected_ami_asociado.keys():#si no esta es porque es una solucion inventada por la grafica => no hay diferencias entre derivada (caidas) =>todas las variables son significativas. Este caso no se va a dar nunca por la expenencia que mete caida
    #             x_argmax=max(dicc_num_var_selected_ami_asociado.keys())#se cogen todas las variables
    #         ax[3,1].axvline(x=x_argmax, color='black', linestyle='--', label=f'Máx caída en {x_argmax} vars con ami: {dicc_num_var_selected_ami_asociado[x_argmax]:.3f}')
    #         dicc_soluciones[(ponderacion_interno,k_decay)]=(list(dicc_selection_num_clusters.keys())[indice_x_argmax_relativo_huecos],x_argmax,dicc_num_var_selected_num_clusters[x_argmax],dicc_num_var_selected_ami_asociado[x_argmax],dicc_num_var_selected_ami_asociado[x_argmax]/ami_orig)
            


    #     # Sexta: suma interno+umbrales p=0.5, k=0 (sin exponential decay)
    #     x6=x4
    #     max_y2=max(y2)
    #     min_y2=min(y2)
    #     y2_normalized=[(y-min_y2)/(max_y2-min_y2) for y in y2]
    #     max_y4=max(y4)
    #     min_y4=min(y4)
    #     y4_normalized=[(y-min_y4)/(max_y4-min_y4) for y in y4]
    #     y6=[ponderacion_interno*y2_normalized[i]+(1-ponderacion_interno)*y4_normalized[i] for i in range(len(x4))]
    #     ax[2,1].set_title(f"Suma de fitness (peso={ponderacion_interno}) y threshold por selection (peso={1-ponderacion_interno}). Sin Exponencial decay")

    #     # Cálculo de las diferencias (caidas)
    #     red_points_y = [max(0, y6[i] - y6[i + 1]) for i in range(len(y6) - 1)]  # Calculamos la diferencia entre los valores de y6
    #     # Para el último valor de y6, tomamos y6[i+1] como 0
    #     red_points_y.append(max(0, y6[-1]))  # Calculamos la diferencia para el último valor de y6 (siendo y6[i+1] = 0)
    #     # red_points_y = [p / ((len(red_points_y) - i) ** 2) for i, p in enumerate(red_points_y)]#penalizacion low number variables


    #     ax[2,1].plot(x6, y6)
    #     ax[2,1].scatter(x6,y6)

    #     ax[2,1].scatter(x6, red_points_y, color='r',marker='x', label='Caídas', zorder=5)

    #     ax[2,1].set_xlabel('Variables significativas')
    #     ax[2,1].set_ylabel('valor')
    #     value_associated_originals=ponderacion_interno*(int_orig-min_y2)/(max_y2-min_y2)+(1-ponderacion_interno)*(dicc_num_var_umbral[key_var_orig_selected]-min_y4)/(max_y4-min_y4)
    #     ax[2,1].axhline(y=value_associated_originals, color='green', linestyle='--', label='valoración originales')
    #     ax[2,1].legend()
    #     ax[2,1].set_xticks(range(min(x5), max(x5) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X
        
    #     #septima: p=0.35, k=0.6
    #     ponderacion_interno=0.35
    #     k_decay=0.6
    #     x6=x4
    #     max_y2=max(y2)
    #     min_y2=min(y2)
    #     y2_normalized=[(y-min_y2)/(max_y2-min_y2) for y in y2]
    #     max_y4=max(y4)
    #     min_y4=min(y4)
    #     y4_normalized=[(y-min_y4)/(max_y4-min_y4) for y in y4]
    #     y6=[ponderacion_interno*y2_normalized[i]+(1-ponderacion_interno)*y4_normalized[i] for i in range(len(x4))]
    #     ax[3,0].set_title(f"Suma de fitness (peso={ponderacion_interno}) y threshold por selection (peso={1-ponderacion_interno}). Exponencial decay k={k_decay}")


    #     # Cálculo de las diferencias (caidas)
    #     red_points_y = [max(0, y6[i] - y6[i + 1]) for i in range(len(y6) - 1)]  # Calculamos la diferencia entre los valores de y6
    #     # Para el último valor de y6, tomamos y6[i+1] como 0
    #     red_points_y.append(max(0, y6[-1]))  # Calculamos la diferencia para el último valor de y6 (siendo y6[i+1] = 0)
    #     # red_points_y = [p / ((len(red_points_y) - i) ** 2) for i, p in enumerate(red_points_y)]#penalizacion low number variables

    #     if k_decay>0:
    #         red_points_y=[p/(1+((len(red_points_y)-1) / (np.exp(k_decay * i)))) if k_decay * i < 700 else p for i, p in enumerate(red_points_y)]#exponential decay that divides the ponderations by a factor of 1+((N-1) / (math.exp(k * i))) where N is the number of variables. k_decay * i < 700 avoids overflow

            
            

    #         #########################################################################################################################3
    #         """Lo hago asi por crear la tabla , pero es algo a tener en cuenta en la refactorización : El diccionario cuyas claves son los cromosomas tiene 'saltos de mas de una variable' y para luego "rescatar" el cromosoma óptimo se hace complicado.  ()
    #         """

    #         x_argmax = x6[red_points_y.index(max(red_points_y))] #nuemro de variables para el maximo
    #         #En dicc_num_var_selected_ami_asociado tenemos las variables con los mismos "huecos", entonces voy a coger como si fuera el "indice_relativo" teniendo en cuenta los huecos ya 
    #         lista_vars_con_huecos = list(dicc_num_var_selected_ami_asociado.keys())
    #         indice_x_argmax_relativo_huecos = lista_vars_con_huecos.index(x_argmax) #indice relativo para acceder al cromosoma óptimo 

    #         ###############################################################################################33
            



    #         if x_argmax not in dicc_num_var_selected_ami_asociado.keys():#si no esta es porque es una solucion iventada por la grafica => no hay diferencias entre derivada (caidas) =>todas las variables son significativas. Este caso no se va a dar nunca
    #             x_argmax=max(dicc_num_var_selected_ami_asociado.keys())#se cogen todas las variables
    #         ax[3,0].axvline(x=x_argmax, color='black', linestyle='--', label=f'Máx caída en {x_argmax} vars con ami: {dicc_num_var_selected_ami_asociado[x_argmax]:.3f}')

    #         dicc_soluciones[(ponderacion_interno,k_decay)]=(list(dicc_selection_num_clusters.keys())[indice_x_argmax_relativo_huecos],x_argmax,dicc_num_var_selected_num_clusters[x_argmax],dicc_num_var_selected_ami_asociado[x_argmax],dicc_num_var_selected_ami_asociado[x_argmax]/ami_orig)
            
    #     ax[3,0].plot(x6, y6)
    #     ax[3,0].scatter(x6,y6)

    #     ax[3,0].scatter(x6, red_points_y, color='r',marker='x', label='Caídas', zorder=5)

    #     ax[3,0].set_xlabel('Variables significativas')
    #     ax[3,0].set_ylabel('valor')
    #     value_associated_originals=ponderacion_interno*(int_orig-min_y2)/(max_y2-min_y2)+(1-ponderacion_interno)*(dicc_num_var_umbral[key_var_orig_selected]-min_y4)/(max_y4-min_y4)
    #     ax[3,0].axhline(y=value_associated_originals, color='green', linestyle='--', label='valoración originales')
    #     ax[3,0].legend()
    #     ax[3,0].set_xticks(range(min(x5), max(x5) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X
        
    # #octava: p=0.5, k=1
    #     ponderacion_interno=0.5
    #     k_decay=1
    #     x6=x4
    #     max_y2=max(y2)
    #     min_y2=min(y2)
    #     y2_normalized=[(y-min_y2)/(max_y2-min_y2) for y in y2]
    #     max_y4=max(y4)
    #     min_y4=min(y4)
    #     y4_normalized=[(y-min_y4)/(max_y4-min_y4) for y in y4]
    #     y6=[ponderacion_interno*y2_normalized[i]+(1-ponderacion_interno)*y4_normalized[i] for i in range(len(x4))]
    #     ax[3,1].set_title(f"Suma de fitness (peso={ponderacion_interno}) y threshold por selection (peso={1-ponderacion_interno}). Exponencial decay k={k_decay}")

    #     # Cálculo de las diferencias (caidas)
    #     red_points_y = [max(0, y6[i] - y6[i + 1]) for i in range(len(y6) - 1)]  # Calculamos la diferencia entre los valores de y6
    #     # Para el último valor de y6, tomamos y6[i+1] como 0
    #     red_points_y.append(max(0, y6[-1]))  # Calculamos la diferencia para el último valor de y6 (siendo y6[i+1] = 0)
    #     # red_points_y = [p / ((len(red_points_y) - i) ** 2) for i, p in enumerate(red_points_y)]#penalizacion low number variables

    #     if k_decay>0:
    #         red_points_y=[p/(1+((len(red_points_y)-1) / (np.exp(k_decay * i)))) if k_decay * i < 700 else p for i, p in enumerate(red_points_y)]#exponential decay that divides the ponderations by a factor of 1+((N-1) / (math.exp(k * i))) where N is the number of variables.k_decay * i < 700 avoids overflow
            
    #         #########################################################################################################################3
    #         """Lo hago asi por crear la tabla , pero es algo a tener en cuenta en la refactorización : El diccionario cuyas claves son los cromosomas tiene 'saltos de mas de una variable' y para luego "rescatar" el cromosoma óptimo se hace complicado.  ()
    #         """

    #         x_argmax = x6[red_points_y.index(max(red_points_y))] #nuemro de variables para el maximo
    #         #En dicc_num_var_selected_ami_asociado tenemos las variables con los mismos "huecos", entonces voy a coger como si fuera el "indice_relativo" teniendo en cuenta los huecos ya 
    #         lista_vars_con_huecos = list(dicc_num_var_selected_ami_asociado.keys())
    #         indice_x_argmax_relativo_huecos = lista_vars_con_huecos.index(x_argmax) #indice relativo para acceder al cromosoma óptimo 

    #         ###############################################################################################33
        


    #         if x_argmax not in dicc_num_var_selected_ami_asociado.keys():#si no esta es porque es una solucion inventada por la grafica => no hay diferencias entre derivada (caidas) =>todas las variables son significativas. Este caso no se va a dar nunca por la expenencia que mete caida
    #             x_argmax=max(dicc_num_var_selected_ami_asociado.keys())#se cogen todas las variables
    #         ax[3,1].axvline(x=x_argmax, color='black', linestyle='--', label=f'Máx caída en {x_argmax} vars con ami: {dicc_num_var_selected_ami_asociado[x_argmax]:.3f}')
    #         dicc_soluciones[(ponderacion_interno,k_decay)]=(list(dicc_selection_num_clusters.keys())[indice_x_argmax_relativo_huecos],x_argmax,dicc_num_var_selected_num_clusters[x_argmax],dicc_num_var_selected_ami_asociado[x_argmax],dicc_num_var_selected_ami_asociado[x_argmax]/ami_orig)
            
    #     ax[3,1].plot(x6, y6)
    #     ax[3,1].scatter(x6,y6)

    #     ax[3,1].scatter(x6, red_points_y, color='r',marker='x', label='Caídas', zorder=5)

    #     ax[3,1].set_xlabel('Variables significativas')
    #     ax[3,1].set_ylabel('valor')
    #     value_associated_originals=ponderacion_interno*(int_orig-min_y2)/(max_y2-min_y2)+(1-ponderacion_interno)*(dicc_num_var_umbral[key_var_orig_selected]-min_y4)/(max_y4-min_y4)
    #     ax[3,1].axhline(y=value_associated_originals, color='green', linestyle='--', label='valoración originales')
    #     ax[3,1].legend()
    #     ax[3,1].set_xticks(range(min(x5), max(x5) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X
        



    #     # Mostrar las gráficas
    #     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar el espaciado
    #     # plt.show()
    #     #save images
        
    #     output_path=directory+f'/{name}_{fitness}_{linkage}.png'
    #     plt.savefig(output_path)
    #     print(f'Analysis variables y num clusters guardados en {output_path}')
        
    #     plt.close(fig)

    #     data['dicc_soluciones'] = {str(k):v for k,v in dicc_soluciones.items()}#add the solutions to the json

    #     # Guardar todo en un archivo JSON
    #     with open(json_path, mode='w') as file:
    #         json.dump(data, file, indent=4)

    #     print(f'Datos guardados como JSON en {json_path}')