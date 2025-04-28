
from utils_genetic2 import *
from genetic2_parallel import *
import numpy as np
import pandas as pd
from alg_clustering import *
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

# SEED NUM_GEN NUM_IND HOF_SIZE METODO_CLUST LINKAGE FITNESS_USADO NUM_CLUSTERS HOF_PONDERADO MEJOR_CROMOSOMA FITNESS_USADO VALOR_DE_TODOS_LOS_FITNESS
# Cada fila un experimento correspondiente a 3 ejecuciones del genetico con 3 semillas distintas

class BuildTest:
     def __init__(self,name_data, artificiales=False, parallel_evaluation=False,dummies=False):
          #name_data : nombre_dataset SIN SUFIJO .CSV (debe estar en la carpeta datasets)
          #artificiales = True si los datos estan en datasets/artificiales_navidad, False cc
          #dummies = True si se le quieren añadir dummies 



          # el dataset debe estar ya en el siguiente formato:
          #    - Nombre de columnas a 'var-x' 
          #    - Target a 'ETIQ'
          #    - Valores normalizados entre [0,1]
          # matplotlib.use('Agg')#para que no se muestren las graficas


          self.name_data = name_data
          self.name_csv = f'{name_data}.csv'
          
          self.path_data = f'./datasets/{self.name_csv}' if not artificiales else f'./datasets/artificiales_navidad/{self.name_csv}' #path del csv original
          self.path_result_folder = f'./results/results_{name_data}' if not parallel_evaluation else f'./results/results_parallel_{name_data}'
          self.path_result_csv= f'{self.path_result_folder}/results_{self.name_csv}'

          

          self.data =  pd.read_csv(self.path_data)
          self.nvars = len(self.data.columns) - 1 #numero de variables ORIGINALES del dataset 
          
          #Ahora mismo trabajamos con semilla fijada para las dummies, en un futuro podemos cambiarlo
          self.data_dummies = self.data.copy() if not dummies else add_dummies(self.data,max(5,self.nvars//2),max(5,self.nvars//2),2,2) #Mitad dummies Uniformes, mitad Betas (2,2) (como una campana)


          
          self.nvars_dummies = len(self.data_dummies.columns) #numero de variables totales al añadirle dummies 

          
          self.num_clusters = self.data['ETIQ'].nunique() #numero clusters
          self.dataframe = None#sera el resultado


          #PARAMETROS PRUEBAS en el run si no se especifica ninguna configuración se haran pruebas con todas estas combinaciones
          
          """  INDIVIDUOS/GENERACIONES (ACUERDO REUNIÓN 8/04/2025): 
                1) Menos de 100 variables : 1500 ind, 150 gen 
                2) Mas de 100 variables : 7000 ind, 300 gen  
          """

          self.rangeNgen = [2]#[150] if self.nvars_dummies <= 100 else [300]  
          self.rangeNpop = [5]#[1500] if self.nvars_dummies <= 100 else [7000]


          self.rangeMetodo_clust = ['Hierarchical']
          self.rangeFitness = [['silhouette'],['ami','silhouette']] #lista de dos listas. La primera debes estar contenida en la segunda. La primera lista son los fitness que se usan para el genético y la segunda los q se usan para valorar el cromosoma
          self.rangeLinkage =['ward']
          self.usedSeeds=[9,10,11]#cada experimento se hace con estas seeds y luego se hace la media
          self.rangeHof_size_parameters=[(0.1,0.15)]
          self.range_num_var_control=[0]
          #una de estas dos deber ser una lista vacia porque son incompatibles
          self.radios_rango_busqueda_clusters=[]
          self.rango_banda_clusters=[[2,26]]
          if(len(self.rango_banda_clusters)>0 and len(self.radios_rango_busqueda_clusters)>0):
               print("ERROR: no pueden ser radios_rango_busqueda_clusters y rango_banda_clusters ambas no vacias")

          self.parallel_evaluation = parallel_evaluation

     def run(self, configuracion_personalizada = None):
          configurations = []
          if (configuracion_personalizada == None):
               for  linkage in self.rangeLinkage:
                    for fitness_metric in self.rangeFitness[0]:#fitness for genetic
                         for num_gen in self.rangeNgen:
                              for num_pop in self.rangeNpop:
                                   for clustering_method in self.rangeMetodo_clust:
                                        for hof_size_parameter in self.rangeHof_size_parameters:
                                             for num_var_control in self.range_num_var_control:
                                                  if(len(self.rango_banda_clusters)>0):
                                                       for banda in self.rango_banda_clusters:
                                                            configurations.append({
                                                            'ngen': num_gen, # Realmente no son necesarios los 5 primeros, pero de cara a un futuro pueden ser interesantes
                                                            'npop': num_pop,
                                                            'clustering_method': clustering_method,
                                                            'hof_size_parameters': hof_size_parameter,
                                                            'seeds': self.usedSeeds, #we do the experiment for each seed
                                                            'linkage': linkage,
                                                            'fitness': fitness_metric,
                                                            'all_fitness_asociated':self.rangeFitness[1],
                                                            'num_var_control':num_var_control,
                                                            'banda_busqueda_num_clusters':banda,
                                                            'radio_banda_busqueda_num_clusters':None
                                                            })
                                                  elif len(self.radios_rango_busqueda_clusters)>0:
                                                       for radio in self.radios_rango_busqueda_clusters:
                                                            configurations.append({
                                                                 'ngen': num_gen, # Realmente no son necesarios los 5 primeros, pero de cara a un futuro pueden ser interesantes
                                                                 'npop': num_pop,
                                                                 'clustering_method': clustering_method,
                                                                 'hof_size_parameters': hof_size_parameter,
                                                                 'seeds': self.usedSeeds, #we do the experiment for each seed
                                                                 'linkage': linkage,
                                                                 'fitness': fitness_metric,
                                                                 'all_fitness_asociated':self.rangeFitness[1],
                                                                 'num_var_control':num_var_control,
                                                                 'banda_busqueda_num_clusters':None,
                                                                 'radio_banda_busqueda_num_clusters':radio
                                                                 })
                                                  else:
                                                       print("ERROR: no se puede tener radios_rango_busqueda_clusters y rango_banda_clusters vacias al mismo tiempo")
                                             
          else: 
               configurations = configuracion_personalizada # configuracion_personalizada deberia ser una lista de dicionarios de esta forma {'ngen': num_gen,'npop': num_pop,'clustering_method': clustering_method,'hof_size_parameters': hof_size_parameter,'seeds': self.usedSeeds,'linkage': linkage,'fitness': 'fitness1','cluster_number': num_clusters,'all_fitness_asociated':['fitness1', 'fitness2']}

          # Crear una lista vacia para acumular las filas
          rows = []

          # experimentos con barra de progreso
          for conf in tqdm(configurations):
               row = self.run_experiment(conf)  # row es un diccionario
               rows.append(row)  # Anyadir cada diccionario a la lista
          # Convertir la lista de diccionarios en un DataFrame de una vez al final
          self.dataframe = pd.DataFrame(rows) #asi es mas eficiente que concatenar todo el rato
          self.to_csv()


     def run_experiment(self, configuration):
          # print(configuration)
          best_chromosomes=[]
          lista_hof_y_contadores=[]
          lista_ultima_gneracion_ejecucion=[]
          # cortes_hof_chromosomes=[]
          average_fitness={fit:0. for fit in configuration['all_fitness_asociated']} #acumulador para calcular media de fitness
          # average_fitness_corte={fit:0. for fit in configuration['all_fitness_asociated']}
          num_rep_experi=len(configuration['seeds'])
          #we run an experiment for each seed and make the average of results
          #para primera semilla
          # print(configuration) #Debug
          if self.parallel_evaluation:
               exp = GeneticSearchParallel(self.data_dummies, configuration['seeds'][0], configuration['ngen'], configuration['npop'], hof_size=configuration['hof_size_parameters'],#alfa and beta for computing the hof_size
                              eleccion_fitness=configuration['fitness'], linkage_hierarchical=configuration['linkage'],num_var_control=configuration['num_var_control'],radio_rango_de_busqueda=configuration["radio_banda_busqueda_num_clusters"],banda_busqueda_clusters=configuration['banda_busqueda_num_clusters']) #cuidado aqui con que los demas parametros no esten a None por defecto
          else:
               exp = GeneticSearch(self.data_dummies, configuration['seeds'][0], configuration['ngen'], configuration['npop'], hof_size=configuration['hof_size_parameters'],#alfa and beta for computing the hof_size
                              eleccion_fitness=configuration['fitness'], linkage_hierarchical=configuration['linkage'],num_var_control=configuration['num_var_control'],radio_rango_de_busqueda=configuration["radio_banda_busqueda_num_clusters"],banda_busqueda_clusters=configuration['banda_busqueda_num_clusters']) #cuidado aqui con que los demas parametros no esten a None por defecto
          hof, hof_counter,dicc_num_cluster_max_fit,ultima_generacion=exp.run()
          lista_ultima_gneracion_ejecucion.append(ultima_generacion)
          lista_hof_y_contadores.append(({tuple(ind):ind.fitness.values[0] for ind in hof},hof_counter,dicc_num_cluster_max_fit))

          # hof=exp.get_hof_sin_var_control()
          # corte_hof=exp.get_cromosoma_resultante_cortar_hof()
          # cortes_hof_chromosomes.append(corte_hof)
          best_chromosomes.append(hof[0])
          for fit in configuration['all_fitness_asociated']:#calculamos todos los fitness del mejor cromosoma
               average_fitness[fit]=average_fitness[fit]+evaluate_ind(self.data_dummies, hof[0], fit, metodo_clust=configuration['clustering_method'],linkage_hierarchical=configuration['linkage'])[0]
               # average_fitness_corte[fit]=average_fitness_corte[fit]+evaluate_ind(self.data_dummies, corte_hof, fit, metodo_clust=configuration['clustering_method'],linkage_hierarchical=configuration['linkage'],n_clusters = configuration['cluster_number'])[0]
          
          acumula_ponderados=exp.get_hof_ponderado()
          for i in range(1,num_rep_experi):#para el resto de semillas
               if self.parallel_evaluation:   
                    exp = GeneticSearchParallel(self.data_dummies, configuration['seeds'][i], configuration['ngen'], configuration['npop'], hof_size=configuration['hof_size_parameters'],#alfa and beta for computing the hof_size
                         eleccion_fitness=configuration['fitness'], linkage_hierarchical=configuration['linkage'],num_var_control=configuration['num_var_control'],radio_rango_de_busqueda=configuration["radio_banda_busqueda_num_clusters"],banda_busqueda_clusters=configuration['banda_busqueda_num_clusters'])
               else:     
                    exp = GeneticSearch(self.data_dummies, configuration['seeds'][i], configuration['ngen'], configuration['npop'], hof_size=configuration['hof_size_parameters'],#alfa and beta for computing the hof_size
                         eleccion_fitness=configuration['fitness'], linkage_hierarchical=configuration['linkage'],num_var_control=configuration['num_var_control'],radio_rango_de_busqueda=configuration["radio_banda_busqueda_num_clusters"],banda_busqueda_clusters=configuration['banda_busqueda_num_clusters'])
               
               hof, hof_counter,dicc_num_cluster_max_fit,ultima_generacion=exp.run()
               lista_ultima_gneracion_ejecucion.append(ultima_generacion)
               lista_hof_y_contadores.append(({tuple(ind):ind.fitness.values[0] for ind in hof},hof_counter,dicc_num_cluster_max_fit))
               # hof=exp.get_hof_sin_var_control()
               # corte_hof=exp.get_cromosoma_resultante_cortar_hof()
               # cortes_hof_chromosomes.append(corte_hof)
               best_chromosomes.append(hof[0])
               for fit in configuration['all_fitness_asociated']:#calculamos todos los fitness del mejor cromosoma del hof
                   average_fitness[fit]=average_fitness[fit]+evaluate_ind(self.data_dummies, hof[0], fit, metodo_clust=configuration['clustering_method'],linkage_hierarchical=configuration['linkage'])[0]
                    # average_fitness_corte[fit]=average_fitness_corte[fit]+evaluate_ind(self.data_dummies, corte_hof, fit, metodo_clust=configuration['clustering_method'],linkage_hierarchical=configuration['linkage'],n_clusters = configuration['cluster_number'])[0]
               acumula_ponderados = [x+y for (x,y) in zip(acumula_ponderados,exp.get_hof_ponderado())] #Acumulo las ponderaciones
               
                              
          row={
               "SEEDS_USED": configuration['seeds'],
               "NUM_GEN":configuration['ngen'],
               "NUM_IND": configuration['npop'],
               "HOF_SIZE":len(acumula_ponderados),
               "RADIO_OF_INTERVAL_FOR_SEARCHING_NUM_CLUSTERS":configuration["radio_banda_busqueda_num_clusters"],
               "BANDA_BUSQUEDA_CLUSTERS(SOLO_SI_RADIO_ES_NONE)":configuration['banda_busqueda_num_clusters'],
               "METODO_CLUST":configuration['clustering_method'],
               "LINKAGE": configuration['linkage'], 
               "NUM_VAR_CONTROL":configuration['num_var_control'],
               "HOF_PONDERADO":[x/num_rep_experi for x in acumula_ponderados], 
               "BEST_CHROMOSOME":best_chromosomes,
               "USED_FITNESS":configuration['fitness']#nombre del fitness usado para hecer el clustering
          }
          # print("Construida la fila")
          # print(row)
          for fit in configuration['all_fitness_asociated']:#calculamos la media de todos los fitness del mejor cromosoma de cada hof
               row[fit+"_mejor_individuo"]=average_fitness[fit]/num_rep_experi
          # for fit in configuration['all_fitness_asociated']:#calculamos la media de todos los fitness del corte de cada hof ponderado
          #      row[fit+"_corte_hof_ponderado"]=average_fitness_corte[fit]/num_rep_experi
          # return row
          #anyadimos el resultado de cada una de las tres ejecuciones para facilitar futuros analisis
          row["TODOS_HOF_Y_CONTADORES_Y_DICCIONARIO_NUM_CLUSTERS_PARA_CADA_EJECUCION"]=lista_hof_y_contadores
          row["ULTIMA_GNERACION_CADA_EJECUCION"]=lista_ultima_gneracion_ejecucion
          return row

     def to_csv(self):
          os.makedirs(os.path.dirname(self.path_result_csv), exist_ok = True)
          print("########################")
          print(self.path_result_folder)
          print(self.path_result_csv)
          self.dataframe.to_csv(self.path_result_csv, index=False)  # Guardar el dataframe en el archivo CSV sin los indices
          print(f"Dataframe guardado correctamente en {self.path_result_folder}") 


     def convert_csv(self):
          ################################################## LO CAMBIO CUANDO TERMINE DE EJECUTAR LOS DE NAVIDAD 
          filepath = self.path_result_csv
          #################################################



          df = pd.read_csv(filepath)

          # Convertir 'HOF_PONDERADO' de cadena a lista de decimales y redondear cada valor a 4 decimales
          df['HOF_PONDERADO'] = df['HOF_PONDERADO'].apply(lambda x: [round(float(i), 4) for i in ast.literal_eval(x)])

          # Redondear las columnas de métricas de fitness a 4 decimales
          fitness_columns = ["silhouette_mejor_individuo"]
          df[fitness_columns] = df[fitness_columns].applymap(lambda x: round(x, 4))

          # Ordenar por 'USED_FITNESS', luego 'NUM_CLUSTERS', y finalmente 'LINKAGE'
          df_sorted = df.sort_values(by=['USED_FITNESS', 'LINKAGE'])

          # Guardar el DataFrame en un archivo Excel
          
          
         
          output_path = f'{self.path_result_folder}/results_{self.name_data}_converted.xlsx' 

          

          df_sorted.to_excel(output_path, index=False)

          print("Archivo guardado en:", output_path)

     #añadido un parametro navidad para la creacion de results_navidad
     def plot_codos_por_fitness2(self):
          
          filepath = self.path_result_csv
          
          
          df = pd.read_csv(filepath)
          if 'TODOS_HOF_Y_CONTADORES_Y_DICCIONARIO_NUM_CLUSTERS_PARA_CADA_EJECUCION' not in df.columns:
               return
          # Renombrar columna para facilitar el acceso
          df.rename(columns={'TODOS_HOF_Y_CONTADORES_Y_DICCIONARIO_NUM_CLUSTERS_PARA_CADA_EJECUCION': 'counters'}, inplace=True)
          
          ejecuciones = df['counters'].tolist()
          total_plots = len(ejecuciones)

          # Reordenar las ejecuciones por resto y cociente
          ejecuciones = sorted(enumerate(ejecuciones), key=lambda x: (x[0] % (total_plots/2), x[0] // total_plots))

          ejecuciones = [e[1] for e in ejecuciones]
          
          filas = (total_plots + 1) // 2  # Calcula las filas necesarias (2 plots por fila)
          
          fig, axes = plt.subplots(filas, 2, figsize=(15, 5 * filas))  # Crear grid de subplots
          axes = axes.flatten()  # Aplanar el array de ejes para facilitar el acceso

          for idx, ejecucion in enumerate(ejecuciones):
               cleaned_str = re.sub(r'<[^>]+>', '0', ejecucion)
               dict_list = ast.literal_eval(cleaned_str)

               
               n_iter = len(dict_list)
               x = sorted(dict_list[0][2].keys())  # Asumiendo que todos los clusters posibles están en la primera ejecución

               y = [0 for _ in range(n_iter)]
               for i in range(n_iter):
                    y[i] = [dict_list[i][2][j][0] for j in x]
                    axes[idx].plot(x, y[i], label=f'iteracion {i}')

               avg = [sum(valores) / len(valores) for valores in zip(*y)]
               maxim = [np.max(valores) for valores in zip(*y)]

               axes[idx].plot(x, avg, c='blue', label='mean')
               axes[idx].plot(x, maxim, c='red', label='max')
               axes[idx].legend()
               axes[idx].grid()
               
               # Título con información del dataframe
               metodo = df.loc[df['counters'] == ejecucion, 'METODO_CLUST'].values[0]
               linkage = df.loc[df['counters'] == ejecucion, 'LINKAGE'].values[0]
               fitness = df.loc[df['counters'] == ejecucion, 'USED_FITNESS'].values[0]
               axes[idx].set_title(f'{metodo} con {linkage} y {fitness}')

          # Eliminar subplots vacíos si hay una cantidad impar de ejecuciones
          if total_plots % 2 != 0:
               fig.delaxes(axes[-1])

          plt.tight_layout()
          filename = filepath.split('\\')[-1].replace('.csv', '')
          # print(filename)
          output_path = f'{filename}_codos.png'
          plt.savefig(output_path,bbox_inches='tight')
          print(f'Resultados guardados en {output_path}')
          # plt.show()
          plt.close()



     #añadido parametro navidad para la creacion de results_navidad 
     def plot_codos_new(self) : 
          # nueva implementacion de los codos, pinntando todos los puntos de la ultima generacion 
          filepath = self.path_result_csv

          
          df = pd.read_csv(filepath)
          if 'ULTIMA_GNERACION_CADA_EJECUCION' not in df.columns:
               print("ERROR NO ESTÁ LA COLUMNA")
               return 
          # Renombrar columna para facilitar el acceso
          columna = df['ULTIMA_GNERACION_CADA_EJECUCION']
          
          df.rename(columns={'ULTIMA_GNERACION_CADA_EJECUCION': 'counters'}, inplace=True)

          experimentos = columna.tolist()
          total_plots = len(experimentos)
          print(total_plots)

          # Reordenar las ejecuciones por resto y cociente
          experimentos = sorted(enumerate(experimentos), key=lambda x: (x[0] % (total_plots/2), x[0] // total_plots))

          experimentos = [e[1] for e in experimentos]

          filas = (total_plots + 1) // 2  # Calcula las filas necesarias (2 plots por fila)

          fig, axes = plt.subplots(filas, 2, figsize=(15, 5 * filas))  # Crear grid de subplots
          axes = axes.flatten()  # Aplanar el array de ejes para facilitar el acceso


          dicc_color = {0:'red',1:'blue',2:'green'}
          for idx, exp in enumerate(experimentos):
               cleaned_str = re.sub(r'<[^>]+>', '0',exp)
               dict_list = ast.literal_eval(cleaned_str)
               for ejec in range(len(dict_list)):
                    x = list(map(lambda x: x[0], dict_list[ejec].keys()))
                    y = list(dict_list[ejec].values())

                    lista_tuplas = [(i,j) for i,j in zip(x,y) if j> -100]
                    x = [i for i,j in lista_tuplas]
                    y = [j for i,j in lista_tuplas]


                    #print(x,y)
                    axes[idx].scatter(x, y, c=dicc_color[ejec], label=f'{ejec}')
                    axes[idx].legend()
                    axes[idx].grid()


               # Título con información del dataframe
               metodo = df.loc[df['counters'] == exp, 'METODO_CLUST'].values[0]
               linkage = df.loc[df['counters'] == exp, 'LINKAGE'].values[0]
               fitness = df.loc[df['counters'] == exp, 'USED_FITNESS'].values[0]
               axes[idx].set_title(f'{metodo} con {linkage} y {fitness}')

          # Eliminar subplots vacíos si hay una cantidad impar de ejecuciones
          if total_plots % 2 != 0:
               fig.delaxes(axes[-1])

          plt.tight_layout()
          filename = filepath.split('\\')[-1].replace('.csv', '')
          # print(filename)
          output_path = f'{filename}_codos_new.png'
          plt.savefig(output_path,bbox_inches='tight')
          print(f'Resultados guardados en {output_path}')
          plt.show()
          plt.close()
       




     def plot_codos_new_hof(self) : 
          # nueva implementacion de los codos, pinntando todos los puntos de la ultima generacion 
          filepath = self.path_result_csv

          
          df = pd.read_csv(filepath)
          if 'ULTIMA_GNERACION_CADA_EJECUCION' not in df.columns:
               print("ERROR NO ESTÁ LA COLUMNA")
               return 
          # Renombrar columna para facilitar el acceso
          columna = df['ULTIMA_GNERACION_CADA_EJECUCION']
          
          df.rename(columns={'ULTIMA_GNERACION_CADA_EJECUCION': 'counters'}, inplace=True)

          experimentos = columna.tolist()
          total_plots = len(experimentos)
          print(total_plots)

          # Reordenar las ejecuciones por resto y cociente
          experimentos = sorted(enumerate(experimentos), key=lambda x: (x[0] % (total_plots/2), x[0] // total_plots))

          experimentos = [e[1] for e in experimentos]

          filas = (total_plots + 1) // 2  # Calcula las filas necesarias (2 plots por fila)

          fig, axes = plt.subplots(filas, 2, figsize=(15, 5 * filas))  # Crear grid de subplots
          axes = axes.flatten()  # Aplanar el array de ejes para facilitar el acceso


          dicc_color = {0:'red',1:'blue',2:'green'}
          for idx, exp in enumerate(experimentos):
               cleaned_str = re.sub(r'<[^>]+>', '0',exp)
               dict_list = ast.literal_eval(cleaned_str)
               for ejec in range(len(dict_list)):
                    x = list(map(lambda x: x[0], dict_list[ejec].keys()))
                    y = list(dict_list[ejec].values())

                    lista_tuplas = [(i,j) for i,j in zip(x,y) if j> -100]
     
                    x = [i for i,j in lista_tuplas]
                    y = [j for i,j in lista_tuplas]

                    #tomo los numeros de clusters con representacion en la ultima generacion
                    x = list(set(x))
                    y = [max([j for i,j in lista_tuplas if i==k]) for k in x]
                    
                    #print(x,y)
                    axes[idx].scatter(x, y, c=dicc_color[ejec], label=f'{ejec}')
                    axes[idx].legend()
                    axes[idx].grid()


               # Título con información del dataframe
               metodo = df.loc[df['counters'] == exp, 'METODO_CLUST'].values[0]
               linkage = df.loc[df['counters'] == exp, 'LINKAGE'].values[0]
               fitness = df.loc[df['counters'] == exp, 'USED_FITNESS'].values[0]
               axes[idx].set_title(f'{metodo} con {linkage} y {fitness}')

          # Eliminar subplots vacíos si hay una cantidad impar de ejecuciones
          if total_plots % 2 != 0:
               fig.delaxes(axes[-1])

          plt.tight_layout()
          filename = filepath.split('\\')[-1].replace('.csv', '')
          # print(filename)
          output_path = f'{filename}_codos_new_hof.png'
          plt.savefig(output_path,bbox_inches='tight')
          print(f'Resultados guardados en {output_path}')
          plt.show()
          plt.close()
     


     def dicc_clusters_fit_max(self,fitness,linkage):
          "Entrada: fitness, linkage"
          "Salida : 3 diccionarios del tipo NUM_CLUSTER:(FIT_MAX,CROMOSOMA) ordenados por clave para la fila de fitness/linkage correspondiente"

          df = pd.read_csv(self.path_result_csv)
          
          if 'TODOS_HOF_Y_CONTADORES_Y_DICCIONARIO_NUM_CLUSTERS_PARA_CADA_EJECUCION' not in df.columns:
                         return "ERROR. NO EXISTE LA COLUMNA EN EL ARCHIVO RESULTS"
                    # Renombrar columna para facilitar el acceso
          df.rename(columns={'TODOS_HOF_Y_CONTADORES_Y_DICCIONARIO_NUM_CLUSTERS_PARA_CADA_EJECUCION': 'counters'}, inplace=True)

          filtro = (df['USED_FITNESS'] == fitness) & (df['LINKAGE'] == linkage)
          str = df[filtro]['counters'].tolist()[0] #me quedo con la fila y la columna 
          
          cleaned_str = re.sub(r'<[^>]+>', '0', str)
          lista_ejecs = ast.literal_eval(cleaned_str) #lista de tuplas de diccs
          


          list_dicc_clusters_fit_max = [ t[-1] for t in lista_ejecs] #lista de los 3 diccionarios (t[-1] para que se quede con el ultimo elemento de la tupla : el dicc que me interesa)
          list_sorted_dicc = [dict(sorted(d.items())) for d in list_dicc_clusters_fit_max]

          return list_sorted_dicc

     

     def hof_ponderado(self,fitness,linkage):
          "Entrada: fitness, linkage"
          "Salida : hof ponderado de la fila correspondiente en formato LISTA"
          df = pd.read_csv(self.path_result_csv)

          filtro = (df['USED_FITNESS'] == fitness) & (df['LINKAGE'] == linkage)
          str = df[filtro]['HOF_PONDERADO'].tolist()[0] #me quedo con la fila y la columna 

          hof_ponderado = ast.literal_eval(str)

          return hof_ponderado
     


     def grafica_vars_fijadas(self,crom_vars,fitness,linkage,max_num_considerado_clusters=20):
          dicc_clusters_fit = {}

          for i in range (2,max_num_considerado_clusters):
               crom_total = [i] + crom_vars
               dicc_clusters_fit[i] = evaluate_ind(self.data_dummies,crom_total,fitness,'hierarchical',linkage)[0]
          x = list(dicc_clusters_fit.keys())
          y = list(dicc_clusters_fit.values())
          
          


          #Te dejo el diccionario dicc_clusters_fit tb por si quieres acceder a los datos además de la gráfica (por si quieres que lo devuelva el método)
          # print(dicc_clusters_fit)

          #Puedes modificarlo para que lo guarde en alguna carpeta y demás, depende de como muestres los resultados
          #Ahora mismo no se muestra porque está puesto matplotlib en agg para que no se muestre 
          # plt.figure()
          # plt.plot(x, y)
          # plt.xlabel('Número de clusters')
          # plt.ylabel(f'{fitness},{linkage}')
          # plt.title('Gráfica para el cromosoma {cromosoma}')
          # plt.show()
          return dicc_clusters_fit

          