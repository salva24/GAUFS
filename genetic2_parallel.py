import numpy as np
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor
from alg_clustering import *
from deap import base, tools, algorithms, creator
import matplotlib.pyplot as plt
from scipy.stats import beta
from utils_genetic2 import *
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
# import math
# import time

# Función para generar datos sintéticos con etiquetas
def generar_datos_sinteticos(n_muestras=500, n_variables=10, n_clusters=4, seed=42):
    """
    Genera datos sintéticos con etiquetas ('ETIQ') y variables aleatorias.
    - n_muestras: número de filas (instancias)
    - n_variables: número total de variables
    - n_clusters: número de clústeres verdaderos
    - seed: semilla para reproducibilidad
    """
    np.random.seed(seed)
    
    # Generar características con clústeres usando make_blobs
    datos, etiquetas = make_blobs(n_samples=n_muestras, 
                                  centers=n_clusters, 
                                  n_features=n_variables-1,  # Una variable menos para agregar control
                                  random_state=seed)
    
    # Convertir a DataFrame
    df = pd.DataFrame(datos, columns=[f'var{i}' for i in range(n_variables-1)])
    
    # Añadir la columna de etiquetas
    df['ETIQ'] = etiquetas
    
    return df



class GeneticSearchParallel:

    def __init__(self, data,  seed=10,ngen=None,npop=None,cxpb=0.8,mutpb=0.1,tol=None,
                 convergence_generations = 50, hof_size=(0.1, 0.2),nobj=1,eleccion_fitness= 'ami',metodo_clust='Hierarchical',linkage_hierarchical = 'average',
                 num_var_control=0,banda_busqueda_clusters=None,radio_rango_de_busqueda=3,numero_mejores_cromososmas_ponderacion_var_sig=None, graficar_evolucion = False):
        '''
        eleccion_fitness = 'ami','chi2','f-score',davies-bouldin','silhouette','sse','calinski_harabasz'
        metodo_clust = 'Hierarchical','Kmeans','DBSCAN'
        linkage_hierarchical = 'average','ward','complete',single'
        num_var_control (int): se anyade dicho numero de variables dummy siguiendouna uniforme y se usa como control para obtener el umbral del Hall Of Fame ponderado. El umbral es el maximo de las ponderacines obtenidas para variables dummy
        banda_busqueda_clusters (tuple): (numero_minimo_de_cluseters_a_buscar,numero_maximo_de_cluseters_a_buscar). El el minimo es menor que 2 se buscan 2
        radio_rango_de_busqueda (int): en caso de que banda_busqueda_clusters==None, se construye una banda con +- este radio desde el numerod e etiquetas
        numero_mejores_cromososmas_ponderacion_var_sig (int): Cunatos cormosomas como maximo se tienen en cuenta para el hof ponderado. si vale None se coje el doble del numero de variables 
        '''
        self.numero_mejores_cromososmas_ponderacion_var_sig=numero_mejores_cromososmas_ponderacion_var_sig
        self.num_clusters_and_its_max_fitness={}#diccionario={ num_cluster:(max_fitness asociated,chromosome_where_it_is_achieved)}
        # semilla para comprobar el funcionamiento
        self.seed=seed
        self.hof=None
        self.hof_counter=None
        self.hof_ponderado=None
        self.umbral_corte=0.5#por defecto
        self.data=data.copy()
        self.num_var_control=num_var_control
        if num_var_control>0:#datos con var control
            self.data=add_dummies(self.data,num_var_control,0,0,0,self.seed)

        self.original_data=data.copy()
        
        # numero de generaciones
        self.ngen=ngen
        # tamaño de la poblacion
        self.npop = npop
        # probabilidad de cruce
        self.cxpb = cxpb
        # probabilidad de mutacion
        self.mutpb = mutpb
        # tamanyo del HallOfFame
        # MODIFICACION: establecer tamaño del hof como un porcentaje concreto de la poblacion:
        #  hof_size = npop*(alpha-(alpha-beta)*exp(-n_variables))
        if type(hof_size) == tuple:
            alpha = hof_size[1]
            beta = hof_size[0]
            # tam = int(npop*(alpha-(alpha-beta)/(self.data.shape[1]-1)))
            tam = int(npop*(alpha-(alpha-beta)*np.log(2)/np.log((self.data.shape[1]+1))))
            # print(((alpha-beta)/(self.data.shape[1]-1)))
            self.hof_size = tam if tam != 0 else 1
        else:
            self.hof_size = hof_size
        # Numero de objetivos del MO
        self.nobj = nobj
        # umbral minimo de mejora en el fitness - para buscar convergencia
        #self.tol = tol
        # numero de generaciones en las que buscar convergencia
        self.convergence_generations = convergence_generations
        # numero de ejecuciones del algoritmo
        #self.n_executions = n_executions
        #eleccion_fitness
        self.eleccion_fitness=eleccion_fitness
        #metodo para el clustering
        self.metodo_clust=metodo_clust
        self.linkage_hierarchical = linkage_hierarchical
        #si banda de busqueda vale none se busca en -3,+3
        if(banda_busqueda_clusters==None):
            num_et=len(np.unique(self.original_data['ETIQ']))
            banda_busqueda_clusters=(num_et - radio_rango_de_busqueda,
                                     num_et + radio_rango_de_busqueda+1)
        self.banda_busqueda_clusters=(max(2,banda_busqueda_clusters[0]),banda_busqueda_clusters[1])#el minimo es 2

        self.graficar_evolucion = False


    ## Funciones para la codificacion de individuos en el genetico
    def init_individual(self, container, attr_bool, data):
        """
        Para inicializacion de los individuos en el genetico.
        Primer gen: numero de cluster (numero aleatorio en la banda de numero de etiquetas distintas +-3)
        Resto de genes: binarios (cada gen i representa si la variable i-esima se considera para clustering)
        """
        ## Para inicializar el numero de cluster en el rango deseado
        num_clusters_range = range(self.banda_busqueda_clusters[0],self.banda_busqueda_clusters[1])
        k =  random.choice(num_clusters_range)

        # Creacion del resto del individuo con genes binarios
        individual = container([k]+[attr_bool() for _ in range(data.shape[1]-1)])
        return individual
    

    ### Definicion adaptada de los operadores geneticos
    def cxUniformModified(self, ind1, ind2):
        """
        Cruce uniforme con intercambio del primer gen (número de clústeres) y resto de genes binarios.
        """
        # Cruce del primer gen (número de clústeres)

        ind1[0], ind2[0] = ind2[0], ind1[0]

        # Cruce uniforme para el resto de los genes
        for i in range(1, len(ind1)):
            # Se produce intercambio en el resto de genes con una probabilidad de 1/2
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    def mutFlipBitModified(self, ind):
        """
        Mutacion bit a bit modificada para que el numero de clusters no se salga de la banda establecida
        """
        mutation_type = random.choice([1,2,3])
        if mutation_type == 1:
            num_clusters_range = range(self.banda_busqueda_clusters[0],self.banda_busqueda_clusters[1])
            ind[0] = random.choice(num_clusters_range) # Escogemos como nuevo numero de clusters un valor en la banda permitida
        elif mutation_type == 2:
            i = random.choice(range(1, len(ind)))
            ind[i] = 1 - ind[i]  # Mutación bit-flip usual
        else:
            num_clusters_range = range(self.banda_busqueda_clusters[0],self.banda_busqueda_clusters[1])
            ind[0] = random.choice(num_clusters_range) # Escogemos como nuevo numero de clusters un valor en la banda permitida

            i = random.choice(range(1, len(ind)))
            ind[i] = 1 - ind[i]  # Mutación bit-flip usual
        return ind, # Siempre se ha de devolver una tupla con estas funciones
    
    def evaluate_individual(self, individual):
        return evaluate_ind(self.original_data, individual, self.eleccion_fitness, self.metodo_clust, self.linkage_hierarchical)


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
        var=self.numero_mejores_cromososmas_ponderacion_var_sig
        if(self.numero_mejores_cromososmas_ponderacion_var_sig==None):
            var = 2*(len(self.hof[0])-1)
        self.hof_ponderado=variable_significance_solo_dado_contador(len(self.hof[0])-1, self.hof_counter,var)
        return self.hof_ponderado


#realmente esto se deberia meter como funcion de una clase, le paso los atributos como parametros pero luego se le pasaria el self solo
def evaluate_ind(data,chromosome,fitness,metodo_clust,linkage_hierarchical,dicc_cluster_max_fitness=None):#el ultimo parametro es para poder guardar en el diccionario los valores correspondientes
    '''
    fitness: posibles valores 'ami','nmi',chi2','f-score','davies-bouldin','dunn-score','silhouette', 'sse', 'calinski_harabasz'
    metodo_clust: posibles valores 'Kmeans','Hierarchical','DBSCAN'
    '''
    try:
        cluster_number = chromosome[0]
        variables = chromosome[1:]
        if np.all(np.array(variables) == 0):
            #print("Cromosoma nulo - fitness muy negativo")
            return -10000000000,
        filtered_vars = [var for var,i in zip(data.columns,variables) if i == 1]  #etiq es la ultima columna
        filtered_vars.append('ETIQ')
        filtered_data = data[filtered_vars]
        
        #hacemos clustering
        experiment=None
        if metodo_clust=="Kmeans":
            experiment = KmeansExperiment(
                            data=filtered_data,
                            seed=10, #self.seed
                            #linkage='average',
                            n_clusters = cluster_number,
                            target= 'ETIQ' #self.target
                        )
        elif metodo_clust=='DBSCAN':
            experiment = DBSCANExperiment(
                            data=filtered_data,
                            n_clusters = cluster_number,
                            target= 'ETIQ' #self.target,
                            )
        else:
            experiment = HierarchicalExperiment(
                            data=filtered_data,
                            #seed=10, #self.seed
                            linkage= linkage_hierarchical,
                            n_clusters = cluster_number,
                            target= 'ETIQ', #self.target
                        )
            
        
        experiment.run()
        ev = 0,
        if (fitness== 'chi2'):
            metricas = ClusteringMetrics(experiment.data, experiment) #cuando se usa jerarquico va bien 
            ev = metricas.chi2()
        else:
            metricas = ClusteringMetrics( data, experiment)
            if (fitness == 'f-score'):
                ev = metricas.f_score()
            if (fitness == 'davies-bouldin'):
                ev = 1 - metricas.davies_bouldin_score() #lo invierto porque en el davies bouldin el 0 es mejor que 1 
            if (fitness == 'dunn-score'):
                ev = metricas.dunn_score()
            if (fitness == 'ami'):
                ev = metricas.mutual_information_score()
            if fitness=='silhouette':
                ev=metricas.silhouette_score()
            if fitness=='sse' :
                ev=-1.*metricas.sse_score()#se invierte para que el máximo sea 0
            if fitness =='calinski_harabasz':
                ev=metricas.calinski_harabasz_score()
            if fitness == 'nmi':
                ev=metricas.nmi_score()

        return (ev,)
    except:
        print(f'Error con el cromosoma: {chromosome};')