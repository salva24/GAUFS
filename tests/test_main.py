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


from gaufs import Gaufs
from gaufs import read_labeled_data_csv
from gaufs.evaluation_metrics import AdjustedMutualInformationScore
from gaufs import DataGenerator


def main():
    # unlabeled_data, true_labels = read_labeled_data_csv(
    #     "./datasets/data_corners_6clusters_1.csv"
    # )
    # gaufs = Gaufs(seed=0)
    # gaufs.set_unlabeled_data(unlabeled_data)
    # gaufs.ngen=5
    # gaufs.run()
    # # Comparison with external metric
    # gaufs.get_plot_comparing_solution_with_another_metric(
    #     AdjustedMutualInformationScore(true_labels=true_labels),
    #     true_number_of_labels=len(set(true_labels)),
    # )

    data=DataGenerator.generate_data_corners(num_useful_features=2, num_samples_per_cluster=1000, num_dummy_unif=0, num_dummy_beta=0, alpha_param=2,beta_param=3, probability_normal_radius=0,max_radius=0.05, deviation_from_max_radius=None, inverse_deviation=1.4, output_path="./data.csv", seed=0)

    #Plot the data in 2D
    import matplotlib.pyplot as plt
    # Plot
    plt.figure()
    plt.scatter(data["var-0"], data["var-1"], c=data["label"])
    plt.xlabel("var-0")
    plt.ylabel("var-1")
    plt.title("Generated data")
    plt.show()
    


if __name__ == "__main__":
    main()
