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

import os

from gaufs import DataGenerator
from gaufs import Gaufs
from gaufs import read_labeled_data_csv
from gaufs.clustering_experiments import KmeansExperiment
from gaufs.evaluation_metrics import AdjustedMutualInformationScore, NMIScore


def main():
    seed = 0

    # Change working directory to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Filepath
    file_path = os.path.join("datasets", "spheres_4clusters.csv")

    # Generate data with spheres distribution and save it to a CSV file
    data_with_labels = DataGenerator.generate_data_spheres(
        num_useful_features=2,
        num_clusters=4,
        num_samples_per_cluster=50,
        num_dummy_unif=1,
        num_dummy_beta=1,
        alpha_param=2,
        beta_param=3,
        output_path=file_path,
        seed=seed,
    )

    # Read the data
    unlabeled_data, true_labels = read_labeled_data_csv(file_path)

    # In this case we asume we know the data labels and therefore we are in a supervised scenario
    # Instantiate GAUFS using the KMeans clustering experiment, an external metric and a tighter cluster search band
    gaufs = Gaufs(
        seed=seed,
        clustering_method=KmeansExperiment(),
        evaluation_metric=AdjustedMutualInformationScore(true_labels=true_labels),
        cluster_number_search_band=(3, 6)
    )
    # Set the unlabeled data
    gaufs.set_unlabeled_data(unlabeled_data)
    # Run GAUFS
    gaufs.run()

    # Comparison with another external metric
    gaufs.get_plot_comparing_solution_with_another_metric(
        NMIScore(true_labels=true_labels),
        true_number_of_labels=len(set(true_labels)),
    )


if __name__ == "__main__":
    main()
