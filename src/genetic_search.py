import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
from deap import base, tools, algorithms, creator
import matplotlib.pyplot as plt
from utils import evaluate_ind

class GeneticSearch:
    def __init__(self, unlabeled_data, ngen, npop, cxpb, cxpb_rest_of_genes, mutpb, convergence_generations, hof_size, 
                 clustering_method, evaluation_metric, cluster_number_search_band, 
                 max_number_selections_for_ponderation, seed=random.randint(0, 10000)):
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
        self.max_number_selections_for_ponderation=max_number_selections_for_ponderation
        self.seed = seed

        #dictionary={num_cluster:(max_fitness asociated,chromosome_where_it_was_achieved)}
        self.num_clusters_and_its_max_fitness={}

        self.hof=None
        self.hof_counter=None
        self.hof_weighted=None

    ## Function for encoding individuals in the genetic algorithm
    def init_individual(self, container, attr_bool, data):
        """
        For initializing individuals in the genetic algorithm.
        First gene: number of clusters (random number within cluster_number_search_band)
        Remaining genes: binary (each gene i represents whether the i-th variable is considered for clustering)
        """
        ## To initialize the number of clusters within the desired range
        num_clusters_range = range(self.banda_busqueda_clusters[0],self.banda_busqueda_clusters[1])
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
            num_clusters_range = range(self.banda_busqueda_clusters[0],self.banda_busqueda_clusters[1])
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
        random.seed(self.seed)
        
        # Definicion del tipo de problema de optimizacion - maximizamos chi-cuadrado
        creator.create("Fitness", base.Fitness, weights=(1.,))
        creator.create("Individual", list, fitness=creator.Fitness)

        # Inicializacion toolbox DEAP para algoritmos geneticos
        toolbox = base.Toolbox()

        # Creacion de cada uno de los genes de un cromosoma
        toolbox.register("attr_bool", random.randint, 0, 1)

        # Definicion de cada uno de los cromosomas (individuos)
        toolbox.register("individual", lambda: self.init_individual(creator.Individual, toolbox.attr_bool, self.original_data))

        # La poblacion es una lista de individuos
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Registrar la función de evaluación, el operador de cruce, mutacion y seleccion
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("mate", self.cxUniformModified)
        toolbox.register("mutate", self.mutFlipBitModified)
        toolbox.register("select", tools.selTournament, tournsize=10)

        # Evalución paralela para agilizar calculos - NO SI NO SE EJECUTA python -m scoop
        toolbox.register("map", self.map_function)

        # Estadisticas de fitness por generacion
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

        # Se mostrara la media, desviacion media, minimo y maximo de fitness por generacion
        stats_fit.register("avg", np.mean, axis=0)
        stats_fit.register("std", np.std, axis=0)
        stats_fit.register("min", np.min, axis=0)
        stats_fit.register("max", np.max, axis=0)

        # Logbook que se imprimira por pantalla con las estadisticas    
        logbook = tools.Logbook()
        logbook.header = "gen", "nevals", "avg", "std", "min", "max"

        # init_global = time.time()
        # Creacion una población inicial
        population = toolbox.population(self.npop)

        # Lista que ira almacenando los mejores individuos generacion a generacion
        best_inds = []
        # Lista que ira almacenando el fitness medio de los individuos de cada generacion
        avg_fitness_history = []
        #print(population) # comprobacion

        # Inicializacion del HallOfFame
        hof = tools.HallOfFame(self.hof_size)
        
        # Inicializacion del Diccionario contador
        hof_counter = dict() #solo cuenta con clave las variables, no el primere gen que es el numerod e clusters {key:(fitness_maximo_encontrado_para_ese_key, veces_entradas_en_hof)}

        # Evaluacion inicial de los individuos
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # init = time.time() # para tests de tiempo
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # fin = time.time()

        # print(f'Tiempo de evaluacion de la poblacion de la generacion {1}: {fin-init}') 
        # init2 = time.time()
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            #modificamos  en el diccionario cada fitness maximo para cada cluster
            val=fit[0]
            cluster_number = ind[0]
            if cluster_number in self.num_clusters_and_its_max_fitness:
                if self.num_clusters_and_its_max_fitness[cluster_number][0]<val:
                    self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)
            else:
                self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)

        # fin2 = time.time()
        # print(f'Tiempo de actualizacion de diccionario {1}: {fin2-init2}')
        # Se actualiza el HOF
        hof.update(population)

        # Actualizamos HOF Counter
        # Actualizacion de HOF Counter
        for ind in hof:
            key = tuple(ind[1:])
            if key in hof_counter:
                old=hof_counter[key]
                hof_counter[key] = (max(old[0],ind.fitness.values[0]),old[1]+1)#maximo fitness y cuento las veces que ha entrado
            else :
                hof_counter[key] = (ind.fitness.values[0],1)
        
        # Definicion de variables para buscar convergencia en el HOF
        hof_unchanged_count=0 
        latest_hof_snapshot = set(tuple(ind) for ind in hof) # Realmente no necesario definirlo como set puesto que el HoF esta ordenado, pero para curarnos en salud
        
        # Se actualizan las listas para el mejor fitness y fitness medio por generacion
        if self.graficar_evolucion:
            best_inds.append(hof[0]) # Inicialmente
            avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in population]))

        record = stats_fit.compile(population)
        #print(record)
        
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # fin_global = time.time()
        # print(f'Tiempo de ejecucion de la inicializacion: {fin_global-init_global}')
        # Se ejecuta el algoritmo evolutivo
        for gen in range(self.ngen):
            print(f' \n \n --- Generacion {gen+1} ---')
            #print(f'Tamanyo de la poblacion: {len(population)}')
            # Seleccionar la proxima generacion de individuos
            offspring = toolbox.select(population, len(population) - self.hof_size)

            # mutacion y reproduccion de individuos
            offspring = algorithms.varAnd(offspring, toolbox, self.cxpb, self.mutpb)

            # Evaluacion de individuos con fitness no valido
            # Los individuos con fitness no valido son los hijos.
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # init = time.time()
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)            
            for ind, fit in zip(invalid_ind, fitnesses):
                #print(fit)
                ind.fitness.values = fit
                #modificamos  en el diccionario cada fitness maximo para cada cluster
                val=fit[0]
                cluster_number = ind[0]
                if cluster_number in self.num_clusters_and_its_max_fitness:
                    if self.num_clusters_and_its_max_fitness[cluster_number][0]<val:
                        self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)
                else:
                    self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)

            # fin = time.time()

            # print(f'Tiempo de evaluacion de la poblacion de la generacion {gen+1}: {fin-init}') 
            # Extendemos la poblacion con los individuos del hof (elitismo)
            offspring.extend(hof.items)

            # Actualizacion del hall of fame con los mejores individuos
            hof.update(offspring)

            # Actualizacion de HOF Counter
            for ind in hof:
                key = tuple(ind[1:])
                if key in hof_counter:
                    old=hof_counter[key]
                    hof_counter[key] = (max(old[0],ind.fitness.values[0]),old[1]+1)#maximo fitness y cuento las veces que ha entrado
                else :
                    hof_counter[key] = (ind.fitness.values[0],1)
            
            #ya no se borran hasta el final
            # inds_to_delete = [key for key in hof_counter if list(key) not in [indivi[1:] for indivi in hof]]
            # for key in inds_to_delete:
            #     del hof_counter[key]

            #Muestro el mejor individuo para cada generacion 
            print(f'Mejor individuo de la generacion {gen+1} es {hof[0]} con {hof[0].fitness.values[0]}')

            if(gen==self.ngen-1):#si es la ultima gen la guardamos
                self.ultima_gen={tuple(ind):ind.fitness.values[0] for ind in offspring}

            # Reemplazar la poblacion con la nueva generacion
            population[:] = offspring

            # Actualizamos la lista de mejores individuos y fitness medio por poblacion
            if self.graficar_evolucion:
                best_inds.append(hof[0])
                avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in population]))

            record = stats_fit.compile(population)
            #print(record)
        
            logbook.record(gen=gen+1, nevals=len(invalid_ind), **record)
            print(logbook.stream)
            
            current_hof_snapshot = set(tuple(ind) for ind in hof)
            if current_hof_snapshot == latest_hof_snapshot:
                hof_unchanged_count += 1
            else:
                hof_unchanged_count = 0
                latest_hof_snapshot = current_hof_snapshot
            
            if hof_unchanged_count >= self.convergence_generations:
                print(f'Early Stopping due to Hall Of Fame not changing in {gen+1} generations')
                self.ultima_gen={tuple(ind):ind.fitness.values[0] for ind in offspring}
                break

        # Evolucion de fitness a lo largo de las generaciones
        if self.graficar_evolucion:
            fitness_vals = []
            for _, ind in enumerate(best_inds, start=1):
                fitness_vals.append(ind.fitness.values[0])

            num_generations = len(fitness_vals)  
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, num_generations + 1), fitness_vals, marker='o', linestyle='-', color='b', label='Fitness mejor individuo')
            plt.plot(range(1, num_generations + 1), avg_fitness_history[:num_generations], marker='x', linestyle='--', color='r', label='Fitness medio')
            plt.title('Fitness del Mejor Individuo y Promedio por Generación')
            plt.xlabel('Número de Generación')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True)
            plt.show()
        self.hof=hof
        self.hof_counter=hof_counter
        return hof, hof_counter,self.num_clusters_and_its_max_fitness,self.ultima_gen
    

    def get_hof_ponderado(self):#este asume que no hay variables de control y lo hace con el sistema contador. Es el que hay que usar a fecha 21/11/2024
        self.hof_weighted=variable_significance_solo_dado_contador(len(self.hof[0])-1, self.hof_counter,self.max_number_selections_for_ponderation)
        return self.hof_weighted
