from tools.utils_genetic2 import *
from genetic_search import *
import numpy as np
import pandas as pd
from tools.alg_clustering import *
import warnings
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hungarianalg.alg import hungarian
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import beta
from tqdm import tqdm
import os
import ast 
import matplotlib
import numpy as np
from scipy.stats import beta
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import re
import ast

def variable_significance_solo_dado_contador(numero_var, hof_counter, numero_cromosomas_max):#usar este a fecha 24/12/24
    chromosomes=sorted(hof_counter.items(), key=lambda item: item[1][0], reverse=True)[:numero_cromosomas_max]
    scores=[]
    suma=0
    for it in chromosomes:
        score=it[1][0]*it[1][1]
        scores.append(score)
        suma+=score

    scores_normalized=[x/suma for x in scores]
    res = [0 for _ in range(numero_var)]
    for i in range(numero_var):#ponderacion de variables y no de num de clusters
        for j,s in enumerate(scores_normalized):
            res[i] += s * chromosomes[j][0][i]
    return res

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
