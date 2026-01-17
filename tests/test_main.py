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

from gaufs import DataGenerator
from gaufs import Gaufs
from gaufs import read_labeled_data_csv
from gaufs.evaluation_metrics import AdjustedMutualInformationScore


def test_gaufs_runs_without_exceptions(tmp_path, monkeypatch):
    seed = 0

    # Change working directory to the temporary path
    monkeypatch.chdir(tmp_path)

    # Filepath
    file_path = tmp_path / "corners_3clusters.csv"

    # Generate data with corners distribution and save it to a CSV file
    DataGenerator.generate_data_corners(
        num_useful_features=2,
        num_samples_per_cluster=10,
        num_dummy_unif=1,
        num_dummy_beta=1,
        alpha_param=2,
        beta_param=3,
        output_path=str(file_path),
        seed=seed,
    )

    # Read the data
    unlabeled_data, true_labels = read_labeled_data_csv(str(file_path))

    # Instantiate GAUFS
    gaufs = Gaufs(seed=seed)
    # Set the unlabeled data
    gaufs.set_unlabeled_data(unlabeled_data)
    # We recomend to leave ngen and npop with the default values but for quick testing we change them
    gaufs.ngen = 2
    gaufs.npop = 250
    # Run GAUFS
    gaufs.run()

    # Comparison with external metric
    gaufs.get_plot_comparing_solution_with_another_metric(
        AdjustedMutualInformationScore(true_labels=true_labels),
        true_number_of_labels=len(set(true_labels)),
    )
