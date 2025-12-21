import numpy as np
import pandas as pd
import random
from scoop import futures
from alg_clustering import *
from deap import base, tools, algorithms, creator
import matplotlib.pyplot as plt
import math
from scipy.stats import beta

class Data:
    def __init__(self, seed=10):
        self.seed=seed
        self.datos=None


#GENERACION DE DATOS ANTIGUA
    def generate_data_old(self,distribucion,num_var):
        self.num_etiq=len(distribucion.keys())
        #Genera un dataFrame con datos sinteticos que siguen una distribuccion normal con las variables de las que dependen 
        # y una uniforme con el resto de las variables 'irrelevantes'
        
        np.random.seed(self.seed)
        df = pd.DataFrame(columns=[f'var-{i}' for i in range(0, num_var)] + ["ETIQ"]) 

        # recorremos todas las etiquetas y para cada etiqueta recorremos todas las variables, generando una distribucion especifica para las 
        # variables indicadas y una aleatoria para todas las demas.
        for etiqueta in distribucion.keys():

            df2 = pd.DataFrame(columns=[f'var-{i}' for i in range(0, num_var)] + ["ETIQ"]) #dataframe auxiliar
            variables = df.columns.tolist()[:-1] 

            for var in variables:
                if var in distribucion[etiqueta][1]:  
                    df2[var] = aux.generar_distribucion_normal_positiva(media = distribucion[etiqueta][2], # media especificada
                                                                        desviacion_estandar = distribucion[etiqueta][3], #deesviacion especificada
                                                                        tamaño_muestra = distribucion[etiqueta][0]) # tamaño especificado
                else:
                    df2[var] = np.random.rand(1,distribucion[etiqueta][0]).flatten() # distrib uniforme con nº de instancias especificadas

            df2["ETIQ"] = etiqueta
            df = pd.concat([df,df2])
        self.datos=df



#GENERACION DE DATOS CON POSIBILIDAD DE DUMMIES QUE SIGUEN DISTRIBUCIONES UNIFORMES (0,1) Y BETA
    def generate_data(self,
                    division_interval = 3, #numeros de bolas por dimension
                    lim_radio = 0.14,
                    num_instances = 200,
                    num_dims_sig = 4, #numero de dimensiones significativas
                    num_centers = 6,
                    num_dummies_unif = 2,
                    num_dummies_beta = 2,
                    alpha_param = 2, #parametro de la Beta(a,b)
                    beta_param = 3, #parametro de la Beta(a,b)
                    radio_norm = True, #True si el radio sigue una N(0,lim_radio/inv_desvio)  / False si sigue U(0,lim_radio)
                    inv_desvio = 1.4,
                    conf_radio_norm = False,
                    lim_radio_varia = False
        ): #para modificar el solape entre distintas etiquetas cuando radio_norm = True
        
        self.num_etiq=num_centers


        np.random.seed(self.seed)
        random.seed(self.seed)

        #divido el intervalo [0,1] en n sub_intervalos y calculo sus centros
        divs_centros = centers_division_interval(division_interval)
        centers = {tuple([random.choice(divs_centros) for i in range(num_dims_sig)]) for k in range(num_centers)}

        #Para evitar que se solapen centros
        while (len(centers)!= num_centers):
            # print(centers)
            centers.add(tuple([random.choice(divs_centros) for i in range(num_dims_sig)]))
        centers = list(centers)

        num_dummies = num_dummies_unif+num_dummies_beta
        data = pd.DataFrame(columns=[f'var-{i}' for i in range(0, num_dims_sig+num_dummies)] + ['ETIQ']) #la etiqueta puede ser el indice del centro en la lista de centros
        
        for i in range(num_centers):
            if (conf_radio_norm):
                r_norm = random.random() > 0.5
            else:
                r_norm = radio_norm 
            
            if (lim_radio_varia) :
                l_radio = random.uniform(lim_radio - 0.075, lim_radio + 0.075)
            else:
                l_radio = lim_radio

            df_aux = create_data_center(seed = self.seed,center = centers[i],num_instances = num_instances,num_dims_sig=num_dims_sig,num_dummies_unif = num_dummies_unif,
                                        num_dummies_beta = num_dummies_beta,alpha_param= alpha_param,beta_param = beta_param,lim_radio=l_radio,
                                        inv_desvio = inv_desvio,radio_norm = r_norm)
            df_aux['ETIQ'] = i
            data = pd.concat([data,df_aux])
        
        data.reset_index(drop = True, inplace=True)
        
        self.datos=data


    

    #GENERACION DE DATOS SIMILAR A GENERATE_DATA, USANDO CAMPANA DE GAUSS PARA EL RADIO Y AUMENTANDOLO UN POCO PARA QUE HAYA SOLAPAMIENTO Y BUSCANDO CENTROS CERCANOS ENTRE SI DE FORMA QUE GENEREN UNA VARIEDAD AFIN NUM_CENTERS-1 DIMENSIONAL (Y LOS VECTORES ENTRE ELLOS UNA BASE ORTOGONAL)
    def generate_data_corners(self,
                    division_interval = 3, #numeros de divisiones por dimension
                    lim_radio = 0.14,
                    num_instances = 150,#numero de puntos en cada centro
                    num_dims_sig = 4, #numero de dimensiones significativas
                    num_dummies_unif = 2,
                    num_dummies_beta = 0,
                    alpha_param = 2,
                    beta_param = 3,
                    inv_desvio=1.4, #desviacion tipica = lim_radio/inv_desvio; A mayor inv_desvio, menor numero de puntos se saldran del lim_radio
                    radio_norm = True #True si el radio sigue una N(0,lim_radio/inv_desvio)  / False si sigue U(0,lim_radio)+
                    ):
        num_centers=num_dims_sig+1#hacen falta num_dims_sig+1 puntos afinmente independientes
        self.num_etiq=num_centers

        np.random.seed(self.seed)
        random.seed(self.seed)


        #divido el intervalo [0,1] en n sub_intervalos y calculo sus centros
        divs_centros = centers_division_interval(division_interval)
        long_interval = divs_centros[1]-divs_centros[0]

        #MODIFICACION: CENTROS CERCANOS PARA BUSCAR ALGO DE SOLAPAMIENTO --> Redondeos a 5 para que no haya problema
        centro_init = [round(divs_centros[0],5) for i in range(num_dims_sig)]#me quedo con el primer centro, el mas cercano al origen
        #centro init es el origen de la esquina
        centers = [centro_init]
        for i in range(num_dims_sig):#para n dimensiones significativas hacen falta n+1 puntos afinmente independientes y los vectores entre estos deben formar una base ortogonal
            new_center=centro_init.copy()
            new_center[i]+=long_interval
            centers.append(new_center)
            

        num_dummies = num_dummies_unif + num_dummies_beta
        data = pd.DataFrame(columns=[f'var-{i}' for i in range(0, num_dims_sig+num_dummies)] + ['ETIQ']) #la etiqueta puede ser el indice del centro en la lista de centros
        
        for i in range(num_centers):
            df_aux = create_data_center(seed = self.seed, center = centers[i],num_instances = num_instances,num_dims_sig=num_dims_sig,num_dummies_unif = num_dummies_unif, num_dummies_beta = num_dummies_beta,lim_radio=lim_radio, inv_desvio=inv_desvio,radio_norm = radio_norm, alpha_param = alpha_param, beta_param = beta_param)
            df_aux['ETIQ'] = i
            data = pd.concat([data,df_aux])
        
        data.reset_index(drop = True, inplace=True)
        self.datos=data

        #GENERACION DE DATOS SIMILAR A GENERATE_DATA, USANDO CAMPANA DE GAUSS PARA EL RADIO Y AUMENTANDOLO UN POCO PARA QUE HAYA SOLAPAMIENTO Y BUSCANDO CENTROS CERCANOS ENTRE SI DE FORMA QUE ESTEN 'EN DIAGONAL'
    def generate_data_diagonal(self,
                    division_interval = 3, #numeros de divisiones por dimension
                    lim_radio = 0.14,
                    num_instances = 150,#numero de puntos en cada centro
                    num_dims_sig = 4, #numero de dimensiones significativas
                    num_dummies_unif = 2,
                    num_dummies_beta = 0,
                    inv_desvio=1.4, #desviacion tipica = lim_radio/inv_desvio; A mayor inv_desvio, menor numero de puntos se saldran del lim_radio
                    alpha_param = 2,
                    beta_param = 3,
                    radio_norm = True #True si el radio sigue una N(0,lim_radio/inv_desvio)  / False si sigue U(0,lim_radio)
                    ):
        num_centers=num_dims_sig+1#hacen falta num_dims_sig+1 puntos afinmente independientes
        self.num_etiq=num_centers

        #divido el intervalo [0,1] en n sub_intervalos y calculo sus centros
        divs_centros = centers_division_interval(division_interval)
        long_interval = divs_centros[1]-divs_centros[0]

        #MODIFICACION: CENTROS CERCANOS PARA BUSCAR ALGO DE SOLAPAMIENTO --> Redondeos a 5 para que no haya problema
        centro_init = [round(divs_centros[0],5) for i in range(num_dims_sig)]#me quedo con el primer centro, el mas cercano al origen
        #centro init es el origen de la esquina
        centers = [centro_init]
        for i in range(division_interval-1):#no nos podemos salir de la cuadricula
            new_center=centers[-1].copy()
            new_center[:]+=long_interval
            centers.append(new_center)

        num_dummies = num_dummies_unif+num_dummies_beta
        data = pd.DataFrame(columns=[f'var-{i}' for i in range(0, num_dims_sig+num_dummies)] + ['ETIQ']) #la etiqueta puede ser el indice del centro en la lista de centros

        for i in range(len(centers)):
            df_aux = create_data_center(seed = self.seed, center = centers[i],num_instances = num_instances,num_dims_sig=num_dims_sig,num_dummies_unif= num_dummies_unif, num_dummies_beta = num_dummies_beta,lim_radio=lim_radio,inv_desvio=inv_desvio, alpha_param = alpha_param,beta_param = beta_param, radio_norm = radio_norm)
            df_aux['ETIQ'] = i
            data = pd.concat([data,df_aux])
        
        data.reset_index(drop = True, inplace=True)
        
        self.datos=data
    
    def generate_csv(self,path="synthetic_data_genetic.csv"):
        self.datos.to_csv(path,index=False)


    #funcion para testeo
    def plot_in_3D(self, variables=("var-0", "var-1", "var-2")):
        # mapa de colores de 'viridis' con 4 divisiones
        colors = plt.cm.get_cmap('viridis', self.num_etiq)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.datos[variables[0]]
        y = self.datos[variables[1]]
        z = self.datos[variables[2]]

        # Normaliza las etiquetas
        etiquetas_normalizadas = self.datos['ETIQ'] / max(self.datos['ETIQ'])
        
        # Asigna un color basado en la etiqueta normalizada
        color_labels = [colors(etiqueta) for etiqueta in etiquetas_normalizadas]

        ax.scatter(x, y, z, c=color_labels)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel(variables[2])

        

        plt.show()

    #funcion para testeo
    def plot_in_2D(self, variables=("var-0", "var-1")):
        # Obtén el mapa de colores de 'viridis' con tantas divisiones como número de etiquetas
        colors = plt.cm.get_cmap('viridis', self.num_etiq)

        fig, ax = plt.subplots()

        x = self.datos[variables[0]]
        y = self.datos[variables[1]]

        # Normaliza las etiquetas
        etiquetas_normalizadas = self.datos['ETIQ'] / max(self.datos['ETIQ'])
        
        # Asigna un color basado en la etiqueta normalizada
        color_labels = [colors(etiqueta) for etiqueta in etiquetas_normalizadas]

        # Crea el scatter plot en 2D
        ax.scatter(x, y, c=color_labels)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])

        

        plt.show()

    



def centers_division_interval(division_interval = 3):
    div = np.linspace(0,1,division_interval+1)
    return list(map(lambda i:(div[i+1]+div[i])/2,list(range(division_interval))))


def create_data_center(seed,center,num_instances,num_dims_sig,num_dummies_unif,num_dummies_beta,lim_radio=0.14,inv_desvio=1.4,alpha_param=2,beta_param=3,radio_norm = True):
    #radio_norm = True si el radio sigue una N(0,lim_radio/inv_desvio) // False si sigue una uniforme U(0,lim_radio)

    #Fijamos semillas 
    np.random.seed(seed)
    random.seed(seed)

    num_dummies = num_dummies_unif + num_dummies_beta
    num_vars_total = num_dims_sig + num_dummies
    df = pd.DataFrame(columns=[f'var-{i}' for i in range(0, num_dims_sig+num_dummies)])

    vector_aux = np.zeros(num_dims_sig)
    for i in range(num_instances):
        data_aux = np.zeros(num_dims_sig+num_dummies)
        if (radio_norm):
                r_aux = generar_distribucion_normal_positiva(media = 0, # media especificada
                                                                desviacion_estandar = lim_radio/inv_desvio, #deesviacion especificada   1.4 deja un 10 % en la normal
                                                                tamaño_muestra = 1)[0] # tamanyo especificado
        else:
                r_aux = random.uniform(0,lim_radio)
        
        denom = 0.
        for k in range(num_dims_sig):
            vector_aux[k]=random.uniform(-1,1)
            denom = denom + vector_aux[k]**2
        for j in range(num_dims_sig):    
            data_aux[j] = r_aux*vector_aux[j]/math.sqrt(denom) + center[j]
        for j in range(num_dims_sig,num_dims_sig+num_dummies_unif):
            data_aux[j] = random.uniform(0,1)
        for j in range(num_dims_sig+num_dummies_unif,num_vars_total):
            data_aux[j] = beta.rvs(alpha_param,beta_param,size = 1)[0]


        df.loc[i] = data_aux
        
    return df

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

def generar_distribucion_normal_positiva(media, tamaño_muestra, desviacion_estandar=0.1):
    # Generar muestras de la distribución normal
    muestras = np.zeros(tamaño_muestra)
    np.random.normal(media, desviacion_estandar, tamaño_muestra)
    for i in range(tamaño_muestra):
        muestra = np.random.normal(media, desviacion_estandar)
        while muestra < 0:#repite sorteo si la muestra es negativa
            muestra = np.random.normal(media, desviacion_estandar)
        muestras[i] = muestra

    return muestras

#########################################################
# Example of a dataset generation with our corners method
if __name__ == "__main__":
    generator=Data(seed=42)
    generator.generate_data_corners(num_instances=100,num_dims_sig=2,num_dummies_unif=1)
    generator.generate_csv("./datasets/synthetic_data_corners.csv")
#########################################################
