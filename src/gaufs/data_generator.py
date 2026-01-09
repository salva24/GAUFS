# MINERVA AI-Lab
# Institute of Computer Engineering
# University of Seville, Spain
#
# Copyright 2026 Salvador de la Torre Gonzalez
# Antonio Bello Castro,
# José M. Núñez Portero
#
# Developed and currently maintained by:
#    Salvador de la Torre Gonzalez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#     SPDX-License-Identifier: Apache-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

"""
import random
import math
import pandas as pd
import numpy as np
from scipy.stats import beta

class DataGenerator:

    @staticmethod
    def generate_data_balls(num_useful_features, num_clusters, num_samples_per_cluster, num_dummy_unif=0, num_dummy_beta=0, alpha_param=2,beta_param=3, probability_normal_radius=0.5,max_radius=0.14, deviation_from_max_radius=0.075, inverse_deviation=1.4, output_path=None, seed=None):
        """
        Synthetic data generator that creates a dataset DataBalls for clustering as described in the GAUFS paper.
        Parameters
        ----------
        num_useful_features : int
            Number of significant features.
        num_clusters : int
            Number of clusters to generate.
        num_samples_per_cluster : int
            Number of samples per cluster.
        num_dummy_unif : int, optional
            Number of dummy features following a uniform distribution. Default is 0.
        num_dummy_beta : int, optional
            Number of dummy features following a beta distribution. Default is 0.
        alpha_param : float, optional
            Alpha parameter for the beta distribution if num_dummy_beta>0. Default is 2.
        beta_param : float, optional
            Beta parameter for the beta distribution if num_dummy_beta>0. Default is 3.
        probability_normal_radius : float, optional
            Probability that the distance of each point from the center follows a normal distribution. It will follow an Uniform distribution with probability 1-probability_normal_radius. Default is 0.5.
        max_radius : float, optional
            Maximum radius for data point generation. Default is 0.14.
        deviation_from_max_radius : float, optional
            Deviation from max_radius to determine the boundary radius for each cluster. If None, no deviation is applied. Default is 0.075.
        inverse_deviation : float, optional
            Inverse of the standard deviation for the truncated normal distribution used to generate the radius. The standard deviation is defined as max_radius / inverse_deviation. Smaller values result in noisier clusters, with points farther from the center and increased overlap between clusters. The default value is 1.4 (which leaves around 10% of the points further away from the cluster boundary).  Default is 1.4.
        output_path : str, optional
            Path to save the generated dataset as a CSV file. If None, the dataset is not saved. Default is None.
        seed : int, optional
            Seed for random number generation. If None, a random seed is used. Default is None.
        Returns
        -------
        pandas.DataFrame
            Generated dataset as a DataFrame with var-0, var-1, ..., var-n columns and a 'label' column for true cluster labels.
        
        """
        
        if seed is None:
            seed = random.randint(0, 10000)
        random.seed(seed)
        np.random.seed(seed)

        num_centers=num_clusters

        #divide the interval [0,1] into num_centers sub-intervals and calculate their centers
        divs_centers = DataGenerator._centers_division_interval(num_centers)
        centers = set()

        #To avoid overlapping centers
        while (len(centers)< num_centers):
            centers.add(tuple([random.choice(divs_centers) for _ in range(num_useful_features)]))
        list_centers = list(centers)

        return DataGenerator._generate_clusters_given_centers(list_centers=list_centers, num_useful_features=num_useful_features, num_samples_per_cluster=num_samples_per_cluster, num_dummy_unif=num_dummy_unif, num_dummy_beta=num_dummy_beta, alpha_param=alpha_param,beta_param=beta_param, probability_normal_radius=probability_normal_radius,max_radius=max_radius, deviation_from_max_radius=deviation_from_max_radius, inverse_deviation=inverse_deviation, output_path=output_path, seed=seed)

    @staticmethod
    def generate_data_corners(num_useful_features, num_samples_per_cluster, num_divisions_per_dimension=3, num_dummy_unif=0, num_dummy_beta=0, alpha_param=2,beta_param=3, probability_normal_radius=0.5,max_radius=0.14, deviation_from_max_radius=0.075, inverse_deviation=1.4, output_path=None, seed=None):
        """
        Synthetic data generator that creates a dataset Corners for clustering as described in the GAUFS paper.It generates clustes whose centers form a simplex in the feature space.
        Parameters
        ----------
        num_useful_features : int
            Number of significant features. The number of clusters generated will be num_useful_features + 1.
        num_samples_per_cluster : int
            Number of samples per cluster.
        num_divisions_per_dimension : int, optional
            Number of divisions in the interval [0,1] for each dimension to determine cluster centers. Default is 3.
        num_dummy_unif : int, optional
            Number of dummy features following a uniform distribution. Default is 0.
        num_dummy_beta : int, optional
            Number of dummy features following a beta distribution. Default is 0.
        alpha_param : float, optional
            Alpha parameter for the beta distribution if num_dummy_beta>0. Default is 2.
        beta_param : float, optional
            Beta parameter for the beta distribution if num_dummy_beta>0. Default is 3.
        probability_normal_radius : float, optional
            Probability that the distance of each point from the center follows a normal distribution. It will follow an Uniform distribution with probability 1-probability_normal_radius. Default is 0.5.
        max_radius : float, optional
            Maximum radius for data point generation. Default is 0.14.
        deviation_from_max_radius : float, optional
            Deviation from max_radius to determine the boundary radius for each cluster. If None, no deviation is applied. Default is 0.075.
        inverse_deviation : float, optional
            Inverse of the standard deviation for the truncated normal distribution used to generate the radius. The standard deviation is defined as max_radius / inverse_deviation. Smaller values result in noisier clusters, with points farther from the center and increased overlap between clusters. The default value is 1.4 (which leaves around 10% of the points further away from the cluster boundary).  Default is 1.4.
        output_path : str, optional
            Path to save the generated dataset as a CSV file. If None, the dataset is not saved. Default is None.
        seed : int, optional
            Seed for random number generation. If None, a random seed is used. Default is None.
        Returns
        -------
        pandas.DataFrame
            Generated dataset as a DataFrame with var-0, var-1, ..., var-n columns and a 'label' column for true cluster labels.
        
        """
        if seed is None:
            seed = random.randint(0, 10000)

        # Divide the interval [0,1] into division_interval sub-intervals and calculate their centers
        divs_centers = DataGenerator._centers_division_interval(num_divisions_per_dimension)
        interval_length = divs_centers[1]-divs_centers[0]

        #We generate n+1 centers in n dimensions to form a simplex  
        # We get the first center which is the origin of the simplex ("Corner")
        # Round to avoid floating point issues
        initial_center = [round(divs_centers[0],5) for _ in range(num_useful_features)]
        centers = [initial_center]
        # for n dimensions, we need n+1 centers and the vectors between them must be orthogonal to avoid clusters not overlapping when projecting
        for i in range(num_useful_features):
            new_center=initial_center.copy()
            new_center[i]+=interval_length
            centers.append(new_center)

        return DataGenerator._generate_clusters_given_centers(list_centers=centers, num_useful_features=num_useful_features, num_samples_per_cluster=num_samples_per_cluster, num_dummy_unif=num_dummy_unif, num_dummy_beta=num_dummy_beta, alpha_param=alpha_param,beta_param=beta_param, probability_normal_radius=probability_normal_radius,max_radius=max_radius, deviation_from_max_radius=deviation_from_max_radius, inverse_deviation=inverse_deviation, output_path=output_path, seed=seed)
            
    @staticmethod
    def _generate_clusters_given_centers(list_centers, num_useful_features, num_samples_per_cluster, num_dummy_unif, num_dummy_beta, alpha_param,beta_param, probability_normal_radius,max_radius, deviation_from_max_radius, inverse_deviation, output_path, seed):
        """
        Generates clusters centered around specified centers with given parameters.
        Parameters
        ----------
        list_centers : list of list of float
            List of center points for the data clusters.
        num_useful_features : int
            Number of significant features.
        num_samples_per_cluster : int
            Number of samples per cluster.
        num_dummy_unif : int
            Number of dummy features following a uniform distribution.
        num_dummy_beta : int
            Number of dummy features following a beta distribution.
        alpha_param : float
            Alpha parameter for the beta distribution if num_dummy_beta>0.
        beta_param : float
            Beta parameter for the beta distribution if num_dummy_beta>0.
        probability_normal_radius : float
            Probability that the distance of each point from the center follows a normal distribution. It will follow an Uniform distribution with probability 1-probability_normal_radius.
        max_radius : float
            Maximum radius for data point generation.
        deviation_from_max_radius : float
            Deviation from max_radius to determine the boundary radius for each cluster. If None, no deviation is applied.
        inverse_deviation : float, optional
            Inverse of the standard deviation for the truncated normal distribution used to generate the radius. The standard deviation is defined as max_radius / inverse_deviation. Smaller values result in noisier clusters, with points farther from the center and increased overlap between clusters. The default value is 1.4 (which leaves around 10% of the points further away from the cluster boundary).  Default is 1.4.
        output_path : str or None
            Path to save the generated dataset as a CSV file. If None, the dataset is not saved.
        seed : int, optional
            Seed for random number generation. If None, a random seed is used. Default is None.
        Returns
        -------
        pandas.DataFrame
            Generated dataset as a DataFrame with var-0, var-1, ..., var-n columns and a 'label' column for true cluster labels.
            
        """
        # Fix seeds
        random.seed(seed)
        np.random.seed(seed)

        num_dummies = num_dummy_unif + num_dummy_beta
        num_centers=len(list_centers)
        data = pd.DataFrame(columns=[f'var-{i}' for i in range(0, num_useful_features+num_dummies)] + ['label'])
        dfs = []
        #for each center, generate data points to form a cluster
        seeds = random.sample(range(0, 10000), num_centers)
        for i in range(num_centers):
            #whether the distance of each point from the center follows a normal distribution or a uniform distribution
            # If the radius follows an uniform distribution, set inverse_deviation_aux to None
            inverse_deviation_aux = None
            # If the radius follows a truncated normal distribution, set inverse_deviation_aux to inverse_deviation
            if (random.random() < probability_normal_radius):
                inverse_deviation_aux = inverse_deviation

            # Determine boundary radius with optional deviation for the cluster
            boundary_radius = max_radius
            if (deviation_from_max_radius!=None):
                boundary_radius = random.uniform(max_radius - deviation_from_max_radius, max_radius + deviation_from_max_radius)

            df_aux = DataGenerator._generate_cluster_centered(seed=seeds[i],center=list_centers[i], num_instances=num_samples_per_cluster, num_dims_sig=num_useful_features, num_dummies_unif=num_dummy_unif, num_dummies_beta=num_dummy_beta, limit_radius=boundary_radius, alpha_param=alpha_param, beta_param=beta_param, radius_inverse_deviation=inverse_deviation_aux)
            
            # The true label is the index of the center in the list of centers
            df_aux['label'] = i
            dfs.append(df_aux)
        data = pd.concat(dfs, ignore_index=True)

        # Save to CSV if output_path is provided
        if output_path is not None:
            data.to_csv(output_path)
        
        return data


    @staticmethod
    def _centers_division_interval(division_interval = 3):
        """
        Divides the interval [0,1] into n sub-intervals and calculates their centers.
        Parameters
        ----------
        division_interval : int
            Number of divisions in the interval [0,1].
        Returns
        -------
        list of float
            List of center points of the sub-intervals.
        """
        div = np.linspace(0,1,division_interval+1)
        return list(map(lambda i:(div[i+1]+div[i])/2,list(range(division_interval))))

    @staticmethod
    def _generate_cluster_centered(seed,center, num_instances, num_dims_sig, num_dummies_unif, num_dummies_beta, limit_radius, alpha_param, beta_param, radius_inverse_deviation=None):
        """
        Creates a dataset centered around a given center with specified parameters.
        Parameters
        ----------
        seed : int
            Seed for random number generation.
        center : list of float
            Center point for the data cluster.
        num_instances : int
            Number of data points to generate.
        num_dims_sig : int
            Number of significant dimensions.
        num_dummies_unif : int
            Number of dummy dimensions following a uniform distribution.
        num_dummies_beta : int
            Number of dummy dimensions following a beta distribution.
        limit_radius : float
            If using an uniform distribution (radius_inverse_deviation=None) it is the maximum radius for data point generation.
            If using a truncated normal distribution (radius_inverse_deviation!=None), the standard deviation used for the radius generation is limit_radius/radius_inverse_deviation.
        alpha_param : float
            Alpha parameter for the beta distribution.
        beta_param : float
            Beta parameter for the beta distribution.
        radius_inverse_deviation : float, optional
            Inverse of the standard deviation for truncated normal distribution of radius. The standard deviation used for the radius generation is limit_radius/radius_inverse_deviation. If None, uniform distribution is used.
        Returns
        -------
        pandas.DataFrame
            Generated cluster as a DataFrame with var-0, var-1, ..., var-n columns.
        """
        # Fix seeds
        np.random.seed(seed)
        random.seed(seed)
        beta_random = np.random.RandomState(seed)

        num_dummies = num_dummies_unif + num_dummies_beta
        total_num_vars = num_dims_sig + num_dummies
        df = pd.DataFrame(columns=[f'var-{i}' for i in range(0, total_num_vars)])

        seeds = random.sample(range(0, 10000), num_instances)
        vector_aux = np.zeros(num_dims_sig)
        for i in range(num_instances):
            data_aux = np.zeros(num_dims_sig+num_dummies)

            # Determine radius for the data point
            if (radius_inverse_deviation!=None):
                r_aux = DataGenerator._generate_truncated_normal_distribution(average=0, standard_deviation=limit_radius/radius_inverse_deviation ,seed=seeds[i], sample_size = 1)[0]
            else:
                r_aux = random.uniform(0,limit_radius)
            
            # significant dimensions
            # Generate random direction vector
            denom = 0.
            for j in range(num_dims_sig):
                vector_aux[j]=random.uniform(-1,1)
                denom = denom + vector_aux[j]**2
            # Calculate data point = center + radius * unit_vector
            for j in range(num_dims_sig):    
                data_aux[j] = r_aux*vector_aux[j]/math.sqrt(denom) + center[j]

            # dummy uniform dimensions
            for j in range(num_dims_sig,num_dims_sig+num_dummies_unif):
                data_aux[j] = random.uniform(0,1)
            # dummy beta dimensions
            for j in range(num_dims_sig+num_dummies_unif,total_num_vars):
                data_aux[j] = beta.rvs(alpha_param,beta_param,size = 1,random_state=beta_random)[0]
            df.loc[i] = data_aux
            
        return df
    
    @staticmethod
    def _generate_truncated_normal_distribution(average, sample_size, standard_deviation, seed=None):
        """
        Generates samples from a truncated normal distribution (only positive values). It is different than applying absolute value to a normal distribution.
        Parameters
        ----------
        average : float
            Mean of the normal distribution.
        sample_size : int
            Number of samples to generate.
        standard_deviation : float
            Standard deviation of the normal distribution.
        seed : int, optional
            Seed for random number generation.
        Returns
        -------
        numpy.ndarray
            Array of generated samples.
        """
        seed= seed if seed is not None else random.randint(0, 10000)
        np.random.seed(seed)
        sample = np.zeros(sample_size)
        for i in range(sample_size):
            point = -1
            # Repeat sampling while the point is negative
            while point < 0:
                point = np.random.normal(average, standard_deviation)
            sample[i] = point
        return sample
