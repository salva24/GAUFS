import numpy as np
from scipy.stats import beta
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import re
import ast


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








def add_dummies(data, num_unif, num_beta, alpha_param, beta_param,seed = 10):
    '''
    Se le añade al data (ya preparado) variables uniformes (0,1) y variables Beta(alpha_param,beta_param)
    '''

    dfTarget = data['ETIQ'].copy()
    data_res = data.copy()
    data_res = data_res.drop(columns=['ETIQ']).copy() 
    
    
    num_variables = len(data_res.columns)

    random.seed(seed)
    np.random.seed(seed)
    #genero columnas con las dummies
    for i in range(len(data_res)):
        for k in range(num_variables, num_variables + num_unif):
            data_res.at[i, f'var-{k}'] = random.uniform(0, 1)
        for k in range(num_variables + num_unif, num_variables + num_unif + num_beta):
            data_res.at[i, f'var-{k}'] = beta.rvs(alpha_param, beta_param)

    # renombro columnas
    data_res.columns = [f'var-{i}' for i in range(len(data_res.columns))]

    
    result = pd.concat([data_res, dfTarget.reset_index(drop=True)], axis=1)

    return result

