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

import numpy as np
import warnings

from gaufs import DataGenerator
from gaufs import Gaufs
from gaufs import read_labeled_data_csv


def test_analyze_variable_weights_all_weights_one(tmp_path, monkeypatch):
    """
    Unit test the analyze_variable_weights method when all variable significances are set to 1.

    This test verifies:
    - The algorithm correctly selects all variables when their significances are all equal to 1.
    - The optimal number of clusters is determined and is an integer.
    - The fitness value is computed and is a float.
    - A UserWarning is raised indicating that the only possible selection is selecting all variables.
    """
    seed = 0
    # Change working directory to the temporary path
    monkeypatch.chdir(tmp_path)

    # Filepath
    file_path = tmp_path / "corners_6clusters.csv"
    # Create a dataset with 5 variables in a CSV file
    num_variables = 5
    data_with_labels = DataGenerator.generate_data_corners(
        num_useful_features=num_variables,
        num_samples_per_cluster=10,
        num_dummy_unif=0,
        num_dummy_beta=0,
        output_path=file_path,
        seed=seed,
    )
    # Read the data
    unlabeled_data, true_labels = read_labeled_data_csv(str(file_path))

    # Initialize Gaufs instance
    gaufs = Gaufs(unlabeled_data=unlabeled_data, verbose=False)

    # Set all variable significances to 1
    gaufs._variable_significance = np.ones(num_variables)

    # Call analyze_variable_weights
    optimal_solution, fitness = gaufs.analyze_variable_weights()

    # Capture the warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        optimal_solution, fitness = gaufs.analyze_variable_weights()

        # Assertions for the warning
        assert len(w) == 1, "Expected one warning."
        assert issubclass(w[-1].category, UserWarning), "Expected a UserWarning."
        assert "The only selection possible is selecting all variables" in str(
            w[-1].message
        )

    # Assertions for the results
    # Check that all variables are selected
    assert (
        optimal_solution[0] == [1] * num_variables
    ), "Not all variables were selected."
    # Check that the number of clusters is as expected
    assert isinstance(optimal_solution[1], int), "Number of clusters is not an integer."
    # Check that fitness is a float
    assert isinstance(fitness, float), "Fitness is not a float."


def test_dataset_with_one_variable_and_single_point_cluster_band_search(
    tmp_path, monkeypatch
):
    """
    Integration test the GAUFS algorithm with a dataset containing one variable and a single-point cluster band search.

    This test verifies:
    - The algorithm correctly selects the only variable available without any runtime errors.
    - The optimal number of clusters is determined as 2 (the only valid option in the search band).
    - The fitness value is computed and is a float.
    - Proper warnings are raised for:
        - Variable weight analysis when all weights are set to 1.
        - The inability to create a 3D plot due to insufficient data.
        - There is only one possible selection; therefore, dictionaries are not plotted.
    """
    seed = 0

    # Change working directory to the temporary path
    monkeypatch.chdir(tmp_path)

    # Filepath
    file_path = tmp_path / "corners_1clusters.csv"
    # Create a dataset with 1 variable and 2 clusters in a CSV file
    data_with_labels = DataGenerator.generate_data_corners(
        num_useful_features=1,
        num_samples_per_cluster=10,
        num_dummy_unif=0,
        num_dummy_beta=0,
        output_path=file_path,
        seed=seed,
    )
    # Read the data
    unlabeled_data, true_labels = read_labeled_data_csv(str(file_path))

    # Instantiate GAUFS
    gaufs = Gaufs(seed=seed)
    # Set the unlabeled data
    gaufs.set_unlabeled_data(unlabeled_data)
    # The only considered number of clusters is 2
    gaufs.cluster_number_search_band = (2, 3)
    gaufs.ngen = 2

    # Capture warnings during the execution of GAUFS
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Catch all warnings
        # Run GAUFS
        gaufs.run()

        # Assertions for warnings
        assert len(w) >= 3, "Expected at least three warnings."
        # Check for the warning about the only possible selection in analyse weights
        assert any(
            "The only selection possible is selecting all variables"
            in str(warning.message)
            for warning in w
        ), "Expected warning about the only possible selection."
        # Check for the warning about not being able to plot dictionaries due to only one considered selection
        assert any(
            "There is only one possible selection considered and therefore no plots comparing"
            in str(warning.message)
            for warning in w
        ), "Expected warning about the optimal variable selection and fitness."
        # Check for the warning about the 3D plot
        assert any(
            "Couldn't create a 3D plot for Variables vs Clusters vs Fitness"
            in str(warning.message)
            for warning in w
        ), "Expected warning about 3D plot creation."

    # Assertions for the results
    # Check that the optimal selection is the expected one (all variables selected)
    assert gaufs._optimal_variable_selection_and_num_of_clusters[0] == [
        1
    ], "The optimal selection is not correct."
    # Check that the optimal number of clusters is 2 which was the only possible one in the search band
    assert (
        gaufs._optimal_variable_selection_and_num_of_clusters[1] == 2
    ), "The optimal number of clusters is not correct."
    # Check that the optimal fitness is a float
    assert isinstance(
        gaufs._fitness_of_optimal_variable_selection_and_num_of_clusters, float
    ), "The optimal fitness is not a float."
