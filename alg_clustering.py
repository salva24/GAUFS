import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd
from functions.clustering import get_chi1_cluster
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances

class ClusteringExperiment:
    def __init__(self, data):
        self.data = data.copy()
        self.n_clusters=0

    def plot_clusters_3D(self,variables=("var-1","var-2","var-3")):
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.get_cmap('viridis', self.n_clusters)
        for cluster in range(self.n_clusters):
            cluster_points = self.data[self.data['cluster'] == cluster]
            ax.scatter(cluster_points[variables[0]], cluster_points[variables[1]], cluster_points[variables[2]], 
                    color=colors(cluster), label=f'Cluster {cluster}')

        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel(variables[2])

        ax.legend()

        plt.show()
        plt.close(fig)

    def plot_clusters_2D(self, variables=("var-1", "var-2")):
        fig, ax = plt.subplots() 

        colors = plt.cm.get_cmap('viridis', self.n_clusters) 

        for cluster in range(self.n_clusters):
            cluster_points = self.data[self.data['cluster'] == cluster] 
            ax.scatter(cluster_points[variables[0]], cluster_points[variables[1]], 
                    color=colors(cluster), label=f'Cluster {cluster}')  

        ax.set_xlabel(variables[0])  
        ax.set_ylabel(variables[1])

        ax.legend() 

        plt.show() 
        plt.close(fig)  


class DBSCANExperiment(ClusteringExperiment):
    def __init__(self, data,eps=0.1,min_samples=60, target='ETIQ'):
        self.data = data.copy()
        self.target = target
        # self.k = len(np.unique(data[self.target]))

        # #escalamos los datos
        ##sería mejor que se escalaran en la clase Data una sola vez
        # self.scaled_data = StandardScaler().fit_transform(self.data.drop(columns=[self.target]))

        self.algorithm = DBSCAN(eps=eps, min_samples=min_samples)
        self.selected_columns = self.data.columns.to_list()

        self.centroids=None
        self.data_with_noise_assigned=None
        self.data_with_noise_as_minus_one=None

    def run(self):
        x_clust = self.data.drop(columns=[self.target])
        labels = self.algorithm.fit_predict(x_clust)

        #clusters obtenidos. El -1 es ruido
        labels_without_noise=set(labels)
        if(-1 in labels_without_noise):
            labels_without_noise.remove(-1)
        # Numero de clusters encontrados (excluyendo el ruido)
        self.n_clusters = len(labels_without_noise)

        #datos con cluster asignados
        self.data = self.data.copy()
        self.data['cluster'] = labels
        self.data_with_noise_as_minus_one=self.data.copy()

        self.data_without_noise=self.data[self.data['cluster'] != -1]

        #asign noise to clusters
        self.data_with_noise_assigned=self.assign_noise_to_clusters() 
        self.data=self.data_with_noise_assigned.copy()


    def get_dist_samples(self):
        value_counts = pd.DataFrame(self.data_with_noise_as_minus_one['cluster'].value_counts())
        value_counts['relative'] = self.data_with_noise_as_minus_one['cluster'].value_counts(normalize=True) * 100
        return value_counts.sort_index()

    def get_clusters_centroid(self, round=True):
        centroids = self.data_without_noise.drop(columns=[self.target]).groupby('cluster').mean()
        # Restablecer el indice para convertir 'cluster' en una columna
        centroids = centroids.reset_index()
        return centroids.round(2) if round else centroids
    
    def assign_noise_to_clusters(self):
        if self.data_without_noise.empty: #si no hay ningun cluster (todo es ruido), devuelvo df vacio
            return self.data_without_noise
        #para cad punto del ruido se busca su cluster mas cercano y se le asigna
        self.centroids=self.get_clusters_centroid(round=False)
        noise_points = self.data[self.data['cluster'] == -1]
        noise_assigned=self.data_without_noise.copy()
        for _,noise in noise_points.iterrows():
            mindist=float('inf')
            closest_clust=None
            for _,centroid in self.centroids.iterrows():
                dist=np.sum(np.square(centroid.drop('cluster').values-noise.drop([self.target,'cluster']).values))
                if(dist<mindist):
                    mindist=dist
                    closest_clust=centroid.iloc[0]
            new_row = noise.copy()
            new_row['cluster']=closest_clust
            new_row=new_row.to_frame().T
            noise_assigned = pd.concat([noise_assigned, new_row])
        #convertimos a int las columnas de target y cluster
        noise_assigned[self.target] = noise_assigned[self.target].astype(int)
        noise_assigned['cluster'] = noise_assigned['cluster'].astype(int)
        noise_assigned.index = range(len(noise_assigned))
        return noise_assigned

        

    def get_cont_table(self, c1, c2,round=True):
        if self.data_with_noise_assigned is None:
            self.data_with_noise_assigned=self.assign_noise_to_clusters() 
        
        res = pd.crosstab(self.data_with_noise_assigned[c1], self.data_with_noise_assigned[c2], normalize='index') * 100
        return res.round(1) if round else res

    def evaluate(self):
        table1 = self.get_cont_table('cluster', self.target, round=False)
        table2 = self.get_cont_table(self.target, 'cluster', round=False)
        return get_chi1_cluster(table1, table2)

class HierarchicalExperiment(ClusteringExperiment):
    def __init__(self, data, linkage='ward', target='ETIQ',n_clusters=None):
        # Por defecto distancia euclidea y ward criterio para unir los clusters
        self.data = data.copy()
        self.target = target
        self.k = len(np.unique(data[self.target]))
        if (n_clusters == None):
            self.n_clusters = self.k
        else : 
            self.n_clusters = n_clusters

        self.linkage = linkage
        self.algorithm = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.selected_columns = self.data.columns.to_list()

    def run(self):
        x_clust = self.data.drop(columns=[self.target])
        # print(x_clust.head())
        # print(x_clust.columns)

        self.algorithm.fit(x_clust)
        self.data['cluster'] = self.algorithm.labels_
        


    def get_dist_samples(self):
        value_counts = pd.DataFrame(self.data['cluster'].value_counts())
        value_counts['relative'] = self.data['cluster'].value_counts(normalize=True) * 100
        return value_counts.sort_index()

    #def get_clusters_centroid(self, round=True):
    #    centroids = pd.DataFrame(self.kmeans.cluster_centers_, columns=self.data.columns[:-2])
    #    return centroids.round(2) if round else centroids

    def get_cont_table(self, c1, c2, round=True):
        res = pd.crosstab(self.data[c1], self.data[c2], normalize='index') * 100
        return res.round(1) if round else res

    # def evaluate(self):
    #     metricas = ClusteringMetrics(self, self.data)
    #     table1 = self.get_cont_table('cluster', self.target, round=False)
    #     table2 = self.get_cont_table(self.target, 'cluster', round=False)
    #     return metricas.mutual_information_score()#get_chi1_cluster(table1, table2)

class KmeansExperiment(ClusteringExperiment):
    ### Copiado de la clase experiments, adaptado para poder evaluar las distintas métricas
    def __init__(self, data, seed, target='EventCode'):
        self.data = data.copy()
        self.target = target
        self.seed = seed
        self.k = len(np.unique(data[self.target]))
        self.algorithm = KMeans(n_clusters=self.k, random_state=self.seed)
        self.selected_columns = self.data.columns.to_list()
        self.n_clusters = self.k

    def run(self):
        x_clust = self.data.drop(columns=[self.target])
        
        self.algorithm.fit(x_clust)
        self.data['cluster'] = self.algorithm.labels_
        
        
    def get_dist_samples(self):
        value_counts = pd.DataFrame(self.data['cluster'].value_counts())
        value_counts['relative'] = self.data['cluster'].value_counts(normalize=True) * 100
        return value_counts.sort_index()

    def get_clusters_centroid(self, round=True):
        centroids = pd.DataFrame(self.algorithm.cluster_centers_, columns=self.data.columns[:-2])
        return centroids.round(2) if round else centroids

    def get_cont_table(self, c1, c2, round=True):
        res = pd.crosstab(self.data[c1], self.data[c2], normalize='index') * 100
        return res.round(1) if round else res

class ClusteringMetrics(ClusteringExperiment):
    def __init__(self, data, experiment):
        self.experiment = experiment
        self.data = experiment.data.copy()

    def get_cont_table(self, c1, c2, round=True):
        res = pd.crosstab(self.data[c1], self.data[c2], normalize='index') * 100
        return res.round(1) if round else res

    def chi2(self):
        table1 = self.get_cont_table('cluster', self.experiment.target, round=False)
        table2 = self.get_cont_table(self.experiment.target, 'cluster', round=False)
        return get_chi1_cluster(table1, table2)

    def silhouette_score(self):
        ## Puntuacion entre -1 (asignacion incorrecta) y 1. (clusters muy densos)
        # Puntuacion cercana a 0 indica que los clusters se solapan. (para determinar numero de clusters)
        return metrics.silhouette_score(self.data.drop(columns=[self.experiment.target,'cluster']), self.experiment.data['cluster'])

    def rand_index_score(self):
        ## Coeficiente que asigna 1. a la prediccion optima y valores cercanos a 0 cuanto mas aleatorias sean las predicciones
        #print(self.data[[self.experiment.target]].values)
        return metrics.adjusted_rand_score(self.data[self.experiment.target].tolist(), self.experiment.data['cluster'])

    def mutual_information_score(self):
        ## Funcion que mide como de bien se corresponden la asignacion optima y la devuelta por el algoritmo (AMI).
        # Basado en la entropia. Normalizado y simetrico.
        # Asignaciones malas pueden tomar valores negativos, asignacion optima-> 1.
        return metrics.adjusted_mutual_info_score(self.data[self.experiment.target].tolist(), self.experiment.data['cluster'])

    def v_measure_score(self):
        ## Combinacion de dos criterios: homogeneidad y completitud.
        # Homogeneidad: cada cluster contiene unicamente valores correspondientes a una etiqueta.
        # Completitud: todos los valores correspondientes a una misma etiqueta estan asignados al mismo cluster.
        # La v-measure es una combinacion de ambos criterios, ponderados por un valor beta (menor que uno para mas homogeneidad,
        # mayor que uno para mas completitud).
        # Esta normalizado y es simetrico. Asignacion aleatoria no obtiene puntuacion de 0. (!)
        return metrics.v_measure_score(self.data[self.experiment.target].tolist(), self.experiment.data['cluster'])

    def fowlkes_mallows_score(self):
        ## Media geometrica de precision y recall.
        # Asignaciones aleatorias tienen valor cercano a 0.
        # Normalizado.
        return metrics.fowlkes_mallows_score(self.data[self.experiment.target].tolist(), self.experiment.data['cluster'])

    def calinski_harabasz_score(self):
        ## No muy indicado para DBSCAN
        #cuidado hay que quitar la columna de cluster tambien
        return metrics.calinski_harabasz_score(self.experiment.data.drop(columns=[self.experiment.target,'cluster']), self.experiment.data['cluster'])

    def davies_bouldin_score(self):
        # Los mejores valores son los cercanos a 0.
        return metrics.davies_bouldin_score(self.experiment.data.drop(columns=[self.experiment.target,'cluster']), self.experiment.data['cluster'])
    
    def f_score(self):
        return metrics.f1_score(list(self.experiment.data[self.experiment.target]),list( self.experiment.data['cluster']),average='weighted')


    def dunn_score(self):
        ### Implementacion rapida de Dunn score. Fuente: https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py

        def delta_fast(ck, cl, distances):
            # Funcion auxiliar 1
            values = distances[np.where(ck)][:, np.where(cl)]
            values = values[np.nonzero(values)]          

            return np.min(values) if values.size != 0 else 0.
        
        def big_delta_fast(ci, distances):
            #Funcion auxiliar 2
            values = distances[np.where(ci)][:, np.where(ci)]
            #values = values[np.nonzero(values)]
            epsilon=10.**-20
            res=np.max(values)
            return res if res > epsilon else epsilon # avoid dividing by 0
        
        # Calculo de Dunn_score
        points = self.experiment.data.drop(columns=['cluster','ETIQ'])
        labels = self.experiment.data['cluster']
        distances = euclidean_distances(points)
        ks = np.sort(np.unique(labels))
        
        deltas = np.ones([len(ks), len(ks)])*1000000
        big_deltas = np.zeros([len(ks), 1])
        
        l_range = list(range(0, len(ks)))
        
        for k in l_range:
            for l in (l_range[0:k]+l_range[k+1:]):
                deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
            
            big_deltas[k] = big_delta_fast((labels == ks[k]), distances)


        di = np.min(deltas)/np.max(big_deltas)
        return di

    def dob_pert_score(self):
        cont_clust = self.experiment.get_cont_table('cluster', self.experiment.target, round=False)
        cont_event = self.experiment.get_cont_table(self.experiment.target, 'cluster', round=False)

        heuristic_table = pd.DataFrame(columns=['event_code', 'heuristic_value']).set_index('event_code')
        for event in cont_event.index:
            clust_table = pd.DataFrame(columns=['cluster', 'heuristic_value']).set_index('cluster')
            for clust in cont_clust.index:
               clust_table.loc[clust] = cont_clust.loc[clust, event] + cont_event.loc[event, clust]

            if (clust_table['heuristic_value'] > 48).any():
                heuristic_table.loc[event] = clust_table['heuristic_value'].max()
        return heuristic_table['heuristic_value'].sum()

    def psi_score(self):
            true_labels = self.data[self.experiment.target].tolist()
            predicted_cluster = self.experiment.data['cluster']

            ls_pairs = np.zeros((len(np.unique(true_labels)), len(np.unique(predicted_cluster))))

            def cluster_similarity(self, i, j):
                acum = 0.

                for index, row in self.experiment.data.iterrows():
                    if (row['ETIQ']==i) & (row['cluster']==j):
                        acum += 1.
                div = max((self.experiment.data['ETIQ'] == i).sum(), (self.experiment.data['cluster'] == i).sum())
                return acum / div if div != 0. else 0.

            def sort_by_size(self):
                dic_c1 = dict()
                dic_c2 = dict()
                for key in np.unique(true_labels):
                    dic_c1[key]=(self.experiment.data['ETIQ'] == key).sum()

                for key in np.unique(predicted_cluster):
                    dic_c2[key] = (self.experiment.data['cluster'] == i).sum()

                sorted_dic_c1 = sorted(dic_c1.items(), key=lambda item: item[1], reverse=True)
                sorted_dic_c2 =  sorted(dic_c2.items(), key=lambda item: item[1], reverse=True)

                n = sum(e2 for e1, e2 in sorted_dic_c1)

                return [sorted_dic_c1, sorted_dic_c2, n]


            def get_expectation(tuple_sorted):
                e = 0.
                n_cluster_min = min(len(tuple_sorted[0]), len(tuple_sorted[1]))
                for i in range(n_cluster_min):
                    mi = tuple_sorted[1][i][1]
                    ni = tuple_sorted[0][i][1]
                    n = tuple_sorted[2]
                    e += ni*mi/(n*max(ni,mi))
                return e # en este caso este paso es un poco innecesario puesto que todo punto esta al menos en un cluster (no en DBSCAN)

            for i in range(len(np.unique(true_labels))):
                for j in range(len(np.unique(predicted_cluster))):
                    ls_pairs[i,j] = cluster_similarity(self, i,j)

            print(ls_pairs)
            row_ind, col_ind = linear_sum_assignment(-ls_pairs)
            s = ls_pairs[row_ind, col_ind].sum()

            # pairs = hungarian(ls_pairs).match

            # s = 0
            # for e in pairs:
            #     s += ls_pairs[e[0], e[1]]

            ls_sorted = sort_by_size(self)

            e = get_expectation(ls_sorted)
            k = len(ls_sorted[0])
            k_ = len(ls_sorted[1])
            psi = 0.
            #print(f's={s}, e={e}')
            if (s>= e) and (min(k, k_) > 1):
                psi = (s-e)/(max(k, k_) - e)
            elif s < e:
                psi = 0.
            elif k==k_:
                psi = 1.

            return psi

    def h_score(self):
        true_labels = self.data[self.experiment.target].tolist()
        predicted_cluster = self.experiment.data['cluster']

        confusion = confusion_matrix(true_labels, predicted_cluster)
    
        # Number of clusters
        k = confusion.shape[0]
    
        # Number of elements in the dataset
        n = np.sum(confusion)
    
         # Calculate the criterion H
        ch = 1 - (1/n) * np.sum(np.max(confusion, axis=1))
    
        return ch
    

    def sse_score(self):#para maximizar hacer - esta funcion
        data = self.experiment.data.drop(columns=['ETIQ']).copy()
        variables = [col for col in data.columns if col != 'cluster']

        centroids = data.groupby('cluster')[variables].mean()

        sse = 0.0
        for cluster, centroid in centroids.iterrows():
            cluster_data = data[data['cluster'] == cluster][variables]
            centroid_array = centroid.values
            distances = np.square(cluster_data - centroid_array).sum(axis=1)
            sse += distances.sum()
        
        return sse



    # Mas: matriz de confusion ¿?

    def nmi_score(self):
        return metrics.normalized_mutual_info_score(self.data[self.experiment.target],self.data['cluster'])
    

    def acc_score(self):
        #Sin fijar numero de clusters no se si nos serviría de mucho el ACC. 
        #El problema del ACC es que cuando tengamos menos clusters entonces el hungarianalg no va a funcionar, porque 
        #supongo que minimo debe de haber tantos clusters como etiquetas reales 
        y = self.data[self.experiment.target]
        y_pred = self.data['cluster']

        #Valores convertidos 
        y_conv= None 
        y_pred_conv = None 
        return metrics.accuracy_score(y_conv,y_pred_conv)