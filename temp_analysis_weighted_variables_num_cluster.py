from build_test_genetic_2 import *
import json
from mpl_toolkits.mplot3d import Axes3D


def cortar_ponderacion(ponderacion_variables,umbral):
    return [1 if pond>=umbral else 0 for pond in ponderacion_variables]

def analyse_dataset(name,artificiales,fitness,linkage,max_num_considerado_clusters=26,parallel_evaluation=False, directory=None,dummies=False, flatten=True):#si flatten es true a partir del ultimo nuemero de variables ques se selecciona se assigna el valor de cuando se selecciona todas en la gáfica para que se aplane en vez de que se haga interpolación lineal. Si k es 0 no se hace decay
    test=BuildTest(name,artificiales=artificiales, parallel_evaluation=parallel_evaluation,dummies=dummies)
    num_vars_originales = test.nvars
    num_clusters = test.num_clusters


    ponderacion_variables=test.hof_ponderado(fitness=fitness,linkage=linkage)
    umbrales=sorted(set(ponderacion_variables),reverse=True)

    posibles_selecciones_variables=[]


    dicc_corte_num_clusters={} #seleccion_variables:num_clusters_con_maximo_fitness
    dicc_corte_fitness={} #seleccion_variables:maximo_fitness
    #estos son para las graficas
    dicc_num_var_seleccionadas_num_clusters={}#num_variables_a_uno_en(seleccion_variables):num_clusters_con_maximo_fitness
    dicc_num_var_seleccionadas_fitness={} #num_variables_a_uno_en(seleccion_variables):maximo_fitness
    dicc_num_var_seleccionadas_ami_asociado={}#num_variables_a_uno_en(seleccion_variables :ami_asociado_usando_num_clusters_con_fitness_interno_maximo
    dicc_num_var_seleccionadas_nmi_asociado= {}
    dicc_num_var_umbral={}#num_var:umbral usado
    dicc_soluciones={}#(pond_interno,k_decay_exponencial):(cromosoma,num_var,num_clusters,ami_asociado,porcentaje_ami_orig)

    #Ya tengo el directorio también para después 
    if(directory==None):
        directory=f'./results/results_{name}/analyisis_significance_and_num_cluster' if not parallel_evaluation else f'./results/results_parallel_{name}/analyisis_significance_and_num_cluster'
        # Crea el directorio si no existe
    os.makedirs(directory, exist_ok=True)

   

    
    for umbral in umbrales:
        corte=cortar_ponderacion(ponderacion_variables,umbral)

        #para cada corte
        posibles_selecciones_variables.append(corte)
        dicc_clusters_fit=test.grafica_vars_fijadas(crom_vars=corte,fitness=fitness,linkage=linkage,max_num_considerado_clusters=max_num_considerado_clusters)

        """ Lo quito por si acaso 
        #Modificacion 24/01 Post Reunion. Genero una grafica con el fitness_por_corte dado para CADA CORTE
        dicc_aux_corte = {}
        if (fitness==fitness_por_corte):
            dicc_aux_corte = dicc_clusters_fit.copy() #el diccionario a variables fijadas (para generar una grafica por cada corte)
        else : 
            dicc_aux_corte = test.grafica_vars_fijadas(crom_vars=corte,fitness=fitness_por_corte,linkage=linkage,max_num_considerado_clusters=max_num_considerado_clusters)
        
        
        directorio_corte = directory+f'/graficas_por_corte/{fitness_por_corte}_{linkage}'
        os.makedirs(directorio_corte, exist_ok=True)
        output_path_corte = directory+f'/graficas_por_corte/{fitness_por_corte}_{linkage}/{name}_corte_{sum(corte)}_{fitness_por_corte}_{linkage}.png'
        x_corte = list(dicc_aux_corte.keys())
        y_corte = list(dicc_aux_corte.values())
        plt.plot(x_corte,y_corte)
        plt.title(f'Corte {sum(corte)}, Variables {corte}. {name} . Fitness : {fitness_por_corte}')
        plt.xlabel("Numero de clusters")
        plt.ylabel(f'Fitness {fitness_por_corte}')
        plt.savefig(output_path_corte)
        plt.close()

        ###################################################################################
        """
        
        clusters_para_maximo=None
        maximo_fitness=None
        for key in dicc_clusters_fit.keys():
            if(maximo_fitness==None or maximo_fitness<dicc_clusters_fit[key] or (maximo_fitness==dicc_clusters_fit[key] and clusters_para_maximo<key)):#en caso de empate nos quedamos con el numero de clusters mas alto
                maximo_fitness=dicc_clusters_fit[key]
                clusters_para_maximo=key
            
        dicc_corte_fitness[tuple(corte)]=maximo_fitness
        dicc_num_var_seleccionadas_fitness[sum(corte)]= maximo_fitness
        dicc_corte_num_clusters[tuple(corte)]=clusters_para_maximo
        dicc_num_var_seleccionadas_num_clusters[sum(corte)]=clusters_para_maximo
        dicc_num_var_umbral[sum(corte)]=umbral

        dicc_num_var_seleccionadas_ami_asociado[sum(corte)]=evaluate_ind(test.data_dummies,[clusters_para_maximo]+corte,'ami','hierarchical',linkage)[0]
        dicc_num_var_seleccionadas_nmi_asociado[sum(corte)]=evaluate_ind(test.data_dummies,[clusters_para_maximo]+corte,'nmi','hierarchical',linkage)[0]



    #######################################make the plots and store results

    #Crear grafica 3D : variables,clusters,silhouette
    dicc_numvars_cortes = {}
    key_var_orig_seleccionadas = 0 #el numero de variables del primer corte que tiene seleccionadas todas las variables originales 
    
    for corte in list(dicc_corte_num_clusters.keys()):
        dicc_numvars_cortes[sum(corte)] = corte
        if (corte[:num_vars_originales] == (1,)*num_vars_originales and key_var_orig_seleccionadas == 0):
            key_var_orig_seleccionadas = sum(corte)

    x_aux = list(dicc_num_var_seleccionadas_num_clusters.keys()) #num_vars
    y_aux = list(range(2,max_num_considerado_clusters)) #num_clusters


    ############################################### Parte 3D
    #Puntos de la grafica 3D
    dicc_var_cluster_fit = {} #del tipo (numvar,numclusters) : fitness
    x = []
    y = []
    z = []
    for i in x_aux:
        for j in y_aux:
            crom = [j] + list(dicc_numvars_cortes[i]) #hago list porque es una tupla
            k = evaluate_ind(test.data_dummies,crom,fitness,'hierarchical',linkage)[0]
            x.append(i)
            y.append(j)
            z.append(k)
            dicc_var_cluster_fit[i,j] = k


    fig3D= plt.figure()
    ax3D = fig3D.add_subplot(111, projection='3d')
    ax3D.plot_trisurf(x, y, z, cmap='viridis') #crea una superficie uniendo los puntos
    #ax3D.scatter(x,y,z)

    ax3D.set_title(f'{name} Gráfica 3D : {fitness}, {linkage}')
    ax3D.set_xlabel('Número de variables')
    ax3D.set_ylabel('Número de clusters')
    ax3D.set_zlabel(f'Fitness {fitness}')


    output_path_3D=directory+f'/{name}_{fitness}_{linkage}_3D.png'
    plt.savefig(output_path_3D)
    plt.close(fig3D)

    ################################save diccionaries
    json_path = os.path.join(directory, f'{name}_{fitness}_{linkage}.json')

    data = {
        'dicc_corte_num_clusters': {str(k): v for k, v in dicc_corte_num_clusters.items()},
        'dicc_corte_fitness': {str(k): v for k, v in dicc_corte_fitness.items()},
        'dicc_num_var_seleccionadas_ami_asociado':dicc_num_var_seleccionadas_ami_asociado,
        'dicc_num_var_seleccionadas_nmi_asociado':dicc_num_var_seleccionadas_nmi_asociado,
        'dicc_var_cluster_fit' : {str(k):v for k,v in dicc_var_cluster_fit.items()},
        'dicc_num_var_umbral' : {str(k):v for k,v in dicc_num_var_umbral.items()}
    }


    ##########################################################################################
    
    #flattening----------------------------------------------------------------------This only affects the graphs
    if flatten:#flatten the graph
        max_num_var_with_value=max(dicc_num_var_seleccionadas_fitness.keys())
        for i in range(max_num_var_with_value - 1, 0, -1):#from the last one to the first one
            if i not in dicc_num_var_seleccionadas_fitness.keys():
                dicc_num_var_seleccionadas_fitness[i]=dicc_num_var_seleccionadas_fitness[i+1]#assign the value of the one next to it
                dicc_num_var_umbral[i]=dicc_num_var_umbral[i+1]#assign the value of the one next to it
                #we only modify the internal fitness and the umbral
                # dicc_num_var_seleccionadas_num_clusters[i]=dicc_num_var_seleccionadas_num_clusters[max_num_var_with_value]
                # dicc_num_var_seleccionadas_ami_asociado[i]=dicc_num_var_seleccionadas_ami_asociado[max_num_var_with_value]
                # dicc_num_var_seleccionadas_nmi_asociado[i]=dicc_num_var_seleccionadas_nmi_asociado[max_num_var_with_value]


    # Crear una figura y dos ejes (subgráficas)
    fig, ax = plt.subplots(4,2, figsize=(16,16))  #4 filas
    fig.suptitle(f'{fitness} {linkage} '+name, fontsize=16)
    rango_x = list(dicc_num_var_seleccionadas_num_clusters.keys())


    separacion_eje_x = (max(rango_x)+1-min(rango_x))//35 #para que como mucho haya 35 (ajustarlo viendo a partir de que numero se pisan)
    # Primera: clusters number
    x1 = list(dicc_num_var_seleccionadas_num_clusters.keys())
    y1 = list(dicc_num_var_seleccionadas_num_clusters.values())
    ax[0,0].set_title("Numero de clusters por corte ")
    ax[0, 0].plot(x1, y1)
    ax[0,0].scatter(x1,y1)
    ax[0, 0].set_xlabel('Variables significativas')
    ax[0, 0].set_ylabel('num_clusters')
    ax[0, 0].set_xticks(range(min(x1), max(x1) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X

    # Segunda: fitness interno  
    x2 = list(dicc_num_var_seleccionadas_fitness.keys())
    y2 = list(dicc_num_var_seleccionadas_fitness.values())
    x2, y2 = zip(*sorted(zip(x2, y2))) #ordenar
    ax[1,0].set_title(f'{fitness} por corte')
    ax[1, 0].plot(x2, y2)
    ax[1, 0].scatter(x2,y2)
    ax[1, 0].set_xlabel('Variables significativas')
    ax[1, 0].set_ylabel(f'fitness {fitness}')
    int_orig = evaluate_ind(test.data_dummies,[num_clusters]+[1]*num_vars_originales,fitness,'hierarchical',linkage)[0]
    ax[1,0].axhline(y=int_orig, color='green', linestyle='--', label=f'{fitness}_original')
    ax[1,0].legend()
    ax[1, 0].set_xticks(range(min(x2), max(x2) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X


    # Tercera: ami (primera fila, segunda columna)
    x3 = list(dicc_num_var_seleccionadas_ami_asociado.keys())
    y3 = list(dicc_num_var_seleccionadas_ami_asociado.values())
    ax[0,1].set_title("AMI asociado al corte")
    ax[0, 1].plot(x3, y3)
    ax[0, 1].scatter(x3,y3)
    ax[0, 1].set_xlabel('Variables significativas')
    ax[0, 1].set_ylabel('fitness ami asociado')
    ami_orig = evaluate_ind(test.data_dummies,[num_clusters]+[1]*num_vars_originales,'ami','hierarchical',linkage)[0]
    ax[0,1].axhline(y=ami_orig, color='green', linestyle='--', label='ami_original')
    ax[0,1].legend()
    ax[0, 1].set_xticks(range(min(x3), max(x3) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X


    # Cuarta: umbrales (segunda fila, segunda columna)
    x4 = list(dicc_num_var_umbral.keys())
    y4 = list(dicc_num_var_umbral.values())
    x4, y4 = zip(*sorted(zip(x4, y4))) #ordenar
    ax[1,1].set_title("Umbral por corte")
    ax[1, 1].plot(x4, y4)
    ax[1, 1].scatter(x4,y4)
    ax[1, 1].set_xlabel('Variables significativas')
    ax[1, 1].set_ylabel('Umbral')
    umbral_var_originales = dicc_num_var_umbral[key_var_orig_seleccionadas]
    ax[1,1].axhline(y=umbral_var_originales, color='green', linestyle='--', label='Umbral del primer corte con todas las vars originales')
    ax[1,1].legend()
    ax[1, 1].set_xticks(range(min(x4), max(x4) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X



    # Quinta: nmi (tercera fila, primera columna)
    x5 = list(dicc_num_var_seleccionadas_nmi_asociado.keys())
    y5 = list(dicc_num_var_seleccionadas_nmi_asociado.values())
    ax[2,0].set_title("NMI asociado al corte")
    ax[2,0].plot(x5, y5)
    ax[2,0].scatter(x5,y5)
    ax[2,0].set_xlabel('Variables significativas')
    ax[2,0].set_ylabel('fitness NMI asociado')
    nmi_orig = evaluate_ind(test.data_dummies,[num_clusters]+[1]*num_vars_originales,'nmi','hierarchical',linkage)[0]
    ax[2,0].axhline(y=nmi_orig, color='green', linestyle='--', label='nmi_original')
    ax[2,0].legend()
    ax[2,0].set_xticks(range(min(x5), max(x5) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X
    


    # Sexta: suma interno+umbrales p=0.5, k=0 (sin exponential decay)
    ponderacion_interno=0.5
    k_decay=0
    x6=x4
    max_y2=max(y2)
    min_y2=min(y2)
    y2_normalized=[(y-min_y2)/(max_y2-min_y2) for y in y2]
    max_y4=max(y4)
    min_y4=min(y4)
    y4_normalized=[(y-min_y4)/(max_y4-min_y4) for y in y4]
    y6=[ponderacion_interno*y2_normalized[i]+(1-ponderacion_interno)*y4_normalized[i] for i in range(len(x4))]
    ax[2,1].set_title(f"Suma de fitness (peso={ponderacion_interno}) y umbral por corte (peso={1-ponderacion_interno}). Sin Exponencial decay")

    # Cálculo de las diferencias (caidas)
    red_points_y = [max(0, y6[i] - y6[i + 1]) for i in range(len(y6) - 1)]  # Calculamos la diferencia entre los valores de y6
    # Para el último valor de y6, tomamos y6[i+1] como 0
    red_points_y.append(max(0, y6[-1]))  # Calculamos la diferencia para el último valor de y6 (siendo y6[i+1] = 0)
    # red_points_y = [p / ((len(red_points_y) - i) ** 2) for i, p in enumerate(red_points_y)]#penalizacion low number variables

    # if k_decay>0:
    #     #aqui no deberia entrar 
    #     red_points_y=[p/(1+((len(red_points_y)-1) / (math.exp(k_decay * i)))) for i, p in enumerate(red_points_y)]#exponential decay that divides the ponderations by a factor of 1+((N-1) / (math.exp(k * i))) where N is the number of variables


    ax[2,1].plot(x6, y6)
    ax[2,1].scatter(x6,y6)

    ax[2,1].scatter(x6, red_points_y, color='r',marker='x', label='Caídas', zorder=5)

    ax[2,1].set_xlabel('Variables significativas')
    ax[2,1].set_ylabel('valor')
    value_associated_originals=ponderacion_interno*(int_orig-min_y2)/(max_y2-min_y2)+(1-ponderacion_interno)*(dicc_num_var_umbral[key_var_orig_seleccionadas]-min_y4)/(max_y4-min_y4)
    ax[2,1].axhline(y=value_associated_originals, color='green', linestyle='--', label='valoración originales')
    ax[2,1].legend()
    ax[2,1].set_xticks(range(min(x5), max(x5) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X
    
    #septima: p=0.35, k=0.6
    ponderacion_interno=0.35
    k_decay=0.6
    x6=x4
    max_y2=max(y2)
    min_y2=min(y2)
    y2_normalized=[(y-min_y2)/(max_y2-min_y2) for y in y2]
    max_y4=max(y4)
    min_y4=min(y4)
    y4_normalized=[(y-min_y4)/(max_y4-min_y4) for y in y4]
    y6=[ponderacion_interno*y2_normalized[i]+(1-ponderacion_interno)*y4_normalized[i] for i in range(len(x4))]
    ax[3,0].set_title(f"Suma de fitness (peso={ponderacion_interno}) y umbral por corte (peso={1-ponderacion_interno}). Exponencial decay k={k_decay}")


    # Cálculo de las diferencias (caidas)
    red_points_y = [max(0, y6[i] - y6[i + 1]) for i in range(len(y6) - 1)]  # Calculamos la diferencia entre los valores de y6
    # Para el último valor de y6, tomamos y6[i+1] como 0
    red_points_y.append(max(0, y6[-1]))  # Calculamos la diferencia para el último valor de y6 (siendo y6[i+1] = 0)
    # red_points_y = [p / ((len(red_points_y) - i) ** 2) for i, p in enumerate(red_points_y)]#penalizacion low number variables

    if k_decay>0:
        red_points_y=[p/(1+((len(red_points_y)-1) / (math.exp(k_decay * i)))) for i, p in enumerate(red_points_y)]#exponential decay that divides the ponderations by a factor of 1+((N-1) / (math.exp(k * i))) where N is the number of variables

        
        

        #########################################################################################################################3
        """Lo hago asi por crear la tabla , pero es algo a tener en cuenta en la refactorización : El diccionario cuyas claves son los cromosomas tiene 'saltos de mas de una variable' y para luego "rescatar" el cromosoma óptimo se hace complicado.  ()
        """

        x_argmax = x6[red_points_y.index(max(red_points_y))] #nuemro de variables para el maximo
        #En dicc_num_var_seleccionadas_ami_asociado tenemos las variables con los mismos "huecos", entonces voy a coger como si fuera el "indice_relativo" teniendo en cuenta los huecos ya 
        lista_vars_con_huecos = list(dicc_num_var_seleccionadas_ami_asociado.keys())
        indice_x_argmax_relativo_huecos = lista_vars_con_huecos.index(x_argmax) #indice relativo para acceder al cromosoma óptimo 

        ###############################################################################################33
        



        if x_argmax not in dicc_num_var_seleccionadas_ami_asociado.keys():#si no esta es porque es una solucion iventada por la grafica => no hay diferencias entre derivada (caidas) =>todas las variables son significativas. Este caso no se va a dar nunca
            x_argmax=max(dicc_num_var_seleccionadas_ami_asociado.keys())#se cogen todas las variables
        ax[3,0].axvline(x=x_argmax, color='black', linestyle='--', label=f'Máx caída en {x_argmax} vars con ami: {dicc_num_var_seleccionadas_ami_asociado[x_argmax]:.3f}')

        dicc_soluciones[(ponderacion_interno,k_decay)]=(list(dicc_corte_num_clusters.keys())[indice_x_argmax_relativo_huecos],x_argmax,dicc_num_var_seleccionadas_num_clusters[x_argmax],dicc_num_var_seleccionadas_ami_asociado[x_argmax],dicc_num_var_seleccionadas_ami_asociado[x_argmax]/ami_orig)
        
    ax[3,0].plot(x6, y6)
    ax[3,0].scatter(x6,y6)

    ax[3,0].scatter(x6, red_points_y, color='r',marker='x', label='Caídas', zorder=5)

    ax[3,0].set_xlabel('Variables significativas')
    ax[3,0].set_ylabel('valor')
    value_associated_originals=ponderacion_interno*(int_orig-min_y2)/(max_y2-min_y2)+(1-ponderacion_interno)*(dicc_num_var_umbral[key_var_orig_seleccionadas]-min_y4)/(max_y4-min_y4)
    ax[3,0].axhline(y=value_associated_originals, color='green', linestyle='--', label='valoración originales')
    ax[3,0].legend()
    ax[3,0].set_xticks(range(min(x5), max(x5) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X
    
 #octava: p=0.5, k=1
    ponderacion_interno=0.5
    k_decay=1
    x6=x4
    max_y2=max(y2)
    min_y2=min(y2)
    y2_normalized=[(y-min_y2)/(max_y2-min_y2) for y in y2]
    max_y4=max(y4)
    min_y4=min(y4)
    y4_normalized=[(y-min_y4)/(max_y4-min_y4) for y in y4]
    y6=[ponderacion_interno*y2_normalized[i]+(1-ponderacion_interno)*y4_normalized[i] for i in range(len(x4))]
    ax[3,1].set_title(f"Suma de fitness (peso={ponderacion_interno}) y umbral por corte (peso={1-ponderacion_interno}). Exponencial decay k={k_decay}")

    # Cálculo de las diferencias (caidas)
    red_points_y = [max(0, y6[i] - y6[i + 1]) for i in range(len(y6) - 1)]  # Calculamos la diferencia entre los valores de y6
    # Para el último valor de y6, tomamos y6[i+1] como 0
    red_points_y.append(max(0, y6[-1]))  # Calculamos la diferencia para el último valor de y6 (siendo y6[i+1] = 0)
    # red_points_y = [p / ((len(red_points_y) - i) ** 2) for i, p in enumerate(red_points_y)]#penalizacion low number variables

    if k_decay>0:
        red_points_y=[p/(1+((len(red_points_y)-1) / (math.exp(k_decay * i)))) for i, p in enumerate(red_points_y)]#exponential decay that divides the ponderations by a factor of 1+((N-1) / (math.exp(k * i))) where N is the number of variables
        
        #########################################################################################################################3
        """Lo hago asi por crear la tabla , pero es algo a tener en cuenta en la refactorización : El diccionario cuyas claves son los cromosomas tiene 'saltos de mas de una variable' y para luego "rescatar" el cromosoma óptimo se hace complicado.  ()
        """

        x_argmax = x6[red_points_y.index(max(red_points_y))] #nuemro de variables para el maximo
        #En dicc_num_var_seleccionadas_ami_asociado tenemos las variables con los mismos "huecos", entonces voy a coger como si fuera el "indice_relativo" teniendo en cuenta los huecos ya 
        lista_vars_con_huecos = list(dicc_num_var_seleccionadas_ami_asociado.keys())
        indice_x_argmax_relativo_huecos = lista_vars_con_huecos.index(x_argmax) #indice relativo para acceder al cromosoma óptimo 

        ###############################################################################################33
       


        if x_argmax not in dicc_num_var_seleccionadas_ami_asociado.keys():#si no esta es porque es una solucion inventada por la grafica => no hay diferencias entre derivada (caidas) =>todas las variables son significativas. Este caso no se va a dar nunca por la expenencia que mete caida
            x_argmax=max(dicc_num_var_seleccionadas_ami_asociado.keys())#se cogen todas las variables
        ax[3,1].axvline(x=x_argmax, color='black', linestyle='--', label=f'Máx caída en {x_argmax} vars con ami: {dicc_num_var_seleccionadas_ami_asociado[x_argmax]:.3f}')
        dicc_soluciones[(ponderacion_interno,k_decay)]=(list(dicc_corte_num_clusters.keys())[indice_x_argmax_relativo_huecos],x_argmax,dicc_num_var_seleccionadas_num_clusters[x_argmax],dicc_num_var_seleccionadas_ami_asociado[x_argmax],dicc_num_var_seleccionadas_ami_asociado[x_argmax]/ami_orig)
        
    ax[3,1].plot(x6, y6)
    ax[3,1].scatter(x6,y6)

    ax[3,1].scatter(x6, red_points_y, color='r',marker='x', label='Caídas', zorder=5)

    ax[3,1].set_xlabel('Variables significativas')
    ax[3,1].set_ylabel('valor')
    value_associated_originals=ponderacion_interno*(int_orig-min_y2)/(max_y2-min_y2)+(1-ponderacion_interno)*(dicc_num_var_umbral[key_var_orig_seleccionadas]-min_y4)/(max_y4-min_y4)
    ax[3,1].axhline(y=value_associated_originals, color='green', linestyle='--', label='valoración originales')
    ax[3,1].legend()
    ax[3,1].set_xticks(range(min(x5), max(x5) + 1,1+separacion_eje_x))  # Marcadores enteros en el eje X
    



    # Mostrar las gráficas
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar el espaciado
    # plt.show()
    #save images
    
    output_path=directory+f'/{name}_{fitness}_{linkage}.png'
    plt.savefig(output_path)
    print(f'Analysis variables y num clusters guardados en {output_path}')
    
    plt.close(fig)

    data['dicc_soluciones'] = {str(k):v for k,v in dicc_soluciones.items()}#add the solutions to the json

    # Guardar todo en un archivo JSON
    with open(json_path, mode='w') as file:
        json.dump(data, file, indent=4)

    print(f'Datos guardados como JSON en {json_path}')




