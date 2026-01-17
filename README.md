# GAUFS

[![PyPI version](https://badge.fury.io/py/gaufs.svg)](https://badge.fury.io/py/gaufs)
[![Tests](https://github.com/salva24/GAUFS/actions/workflows/tests-build.yml/badge.svg)](https://github.com/salva24/GAUFS/actions/workflows/tests-build.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**GAUFS** (Genetic Algorithm for Unsupervised Feature Selection) is a Python library for unsupervised feature selection designed to identify the most relevant features for clustering without requiring labeled data. It combines genetic algorithms with clustering experiments to perform dimensionality reduction while simultaneously estimating the optimal number of clusters.

This library accompanies the research work presented in the paper:

> *GAUFS: Genetic Algorithm for Unsupervised Feature Selection for Clustering*

**Note:** To reproduce the results presented in the paper and the experimental setup used for comparison with alternative methods, please use the [paper-reproducibility branch](https://github.com/salva24/GAUFS/tree/paper-reproducibility) of this repository.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start: Basic Gaufs Usage](#quick-start-basic-gaufs-usage)
- [How GAUFS Works](#how-gaufs-works)
- [Main Configuration Parameters for GAUFS](#main-configuration-parameters-for-gaufs)
- [Output Files](#output-files)
  - [GA Execution Folders](#ga-execution-folders)
  - [Results Folder](#results-folder)
  - [Comparison Plots](#comparison-plots)
- [Synthetic Data Generators](#synthetic-data-generators)
  - [DataSpheres Generator](#dataspheres-generator)
  - [DataCorners Generator](#datacorners-generator)
- [Custom Fitness](#custom-fitness)
  - [Clustering Algorithms](#clustering-algorithms)
  - [Evaluation Metrics](#evaluation-metrics)
- [Examples](#examples)
  - [Demo 1: Basic Usage with Corner Distribution](#demo-1-basic-usage-with-corner-distribution-demodemo1py)
  - [Demo 2: Advanced Configuration with Spherical Clusters](#demo-2-advanced-configuration-with-spherical-clusters-demodemo2py)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Library Authors and Contact Information](#library-authors-and-contact-information)
- [Support](#support)

---

## Key Features

- **Fully Unsupervised:** No labeled data required for feature selection
- **Automatic Cluster Estimation:** Simultaneously identifies optimal features and number of clusters
- **Flexible architecture:** GAUFS can work with custom clustering algorithms and evaluation metrics, allowing optimization of **internal metrics** (without relying on labels) and optionally **external metrics** when true labels are available for evaluation.
- **Synthetic data generators:** Includes the *Spheres* and *Corners* generators introduced in the paper, designed for testing feature selection under controlled clustering scenarios and benchmarking.
- **Comprehensive Output:** Automatic generation of plots, CSV files, and JSON results
- **Reproducible:** Seed-based random state control for consistent results

---

## Installation

GAUFS is available on PyPI and can be installed using pip:
```bash
pip install gaufs
```

**Requirements:**
- Python `>=3.11,<3.14`
- numpy `>=2.4.0,<3.0.0`
- pandas `>=2.3.3,<3.0.0`
- scipy `>=1.16.3,<2.0.0`
- matplotlib `>=3.10.8,<4.0.0`
- scikit-learn `>=1.8.0,<2.0.0`
- DEAP `>=1.4.3,<2.0.0` (used for the genetic algorithm)

---

## Quick Start: Basic Gaufs Usage

```python
import pandas as pd
from gaufs import Gaufs

# Load your unlabeled data
data = pd.read_csv('your_data.csv')

# Initialize GAUFS with default parameters
gaufs = Gaufs(unlabeled_data=data)

# Run the complete algorithm
optimal_solution, fitness = gaufs.run()

# Extract results
selected_features = optimal_solution[0]  # Binary list (1=selected, 0=not selected)
optimal_clusters = optimal_solution[1]   # Optimal number of clusters

print(f"Selected {sum(selected_features)} out of {len(selected_features)} features")
print(f"Optimal number of clusters: {optimal_clusters}")
print(f"Fitness score: {fitness}")
```

---

## How GAUFS Works

GAUFS operates in two main phases:

### 1. Genetic Search Phase
- Runs multiple independent genetic algorithm executions
- Each execution evolves feature subsets across different numbers of clusters
- Evaluates clustering quality using the specified metric (default: Silhouette Score)
- Computes variable significance scores based on selection frequency and quality

### 2. Variable Weight Analysis Phase
- Analyzes results from all genetic searches
- Combines fitness values and significance thresholds using weighted averaging
- Applies exponential decay to importance differences
- Automatically selects the optimal feature subset and number of clusters
- Outputs metrics graphs to help users make more informed decisions when balancing dimensionality reduction and cluster quality.

The algorithm produces comprehensive outputs including:
- Selected feature subset
- Optimal number of clusters
- Fitness scores and significance metrics
- Visualization plots (2D and 3D)
- Detailed CSV files and JSON dictionaries

---
## Main Configuration Parameters for GAUFS

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | None | Random seed for reproducibility. Default: random integer between 0 and 10000 if None. |
| `unlabeled_data` | pd.DataFrame or None | None | Input dataset without labels. If None, creates empty DataFrame. |
| `num_genetic_executions` | int | 1 | Number of independent Genetic Algorithm runs. Must be ≥ 1. |
| `ngen` | int | 150 (auto 150 if `num_vars` ≤ 100, else 300) | Number of generations per GA execution. Must be ≥ 1. |
| `npop` | int | 1500 (auto 1500 if `num_vars` ≤ 100, else 7000) | Population size. Must be ≥ 1. |
| `cxpb` | float | 0.8 | Crossover probability for genetic operations. Range: [0.0, 1.0]. |
| `cxpb_rest_of_genes` | float | 0.5 | Crossover probability for the rest of generations after initial ones. Range: [0.0, 1.0]. |
| `mutpb` | float | 0.1 | Mutation probability for genetic operations. Range: [0.0, 1.0]. |
| `convergence_generations` | int | 50 | Generations without improvement before early stopping. Must be ≥ 1. |
| `hof_size` | int or None | None | Hall of Fame size (absolute number of best solutions to retain). Overrides `hof_alpha_beta` if provided. Must be ≥ 1 or None. |
| `hof_alpha_beta` | tuple | (0.1, 0.2) | `(alpha, beta)` used for automatic Hall of Fame size calculation if `hof_size` is None. Range: [0.0, 1.0], beta ≥ alpha. |
| `clustering_method` | ClusteringExperiment | HierarchicalExperiment(linkage='ward') | Clustering algorithm instance. Must implement `ClusteringExperiment`. |
| `evaluation_metric` | EvaluationMetric | SilhouetteScore() | Metric for evaluating clustering quality. Must implement `EvaluationMetric`. |
| `cluster_number_search_band` | tuple | (2, 26) | Range of cluster numbers to explore as (min_inclusive, max_exclusive). Must satisfy 2 ≤ min < max ≤ number of samples. |
| `fitness_weight_over_threshold` | float | 0.5 | Weight for fitness vs threshold in variable importance computation. Range: [0.0, 1.0]. |
| `exponential_decay_factor` | float | 1.0 | Exponential decay factor for automatic solution selector. 0 means no decay. Formula: δ_i / (1 + (N / exp(exponential_decay_factor * i))). Must be ≥ 0.0. |
| `max_number_selections_for_ponderation` | int or None | 2 * num_vars | Max selections from Hall of Fame for weight computation. Must be ≥ 1 or None. |
| `verbose` | bool | True | Whether to print logs during execution. |
| `generate_genetics_log_files` | bool | True | Whether to generate log files with GA execution details. |
| `graph_evolution` | bool | True | Whether to generate graphs of best and average fitness during GA evolution. |
| `generate_files_with_results` | bool | True | Whether to generate files with results and plots. |
| `output_directory` | str or None | "./out/" if None | Path to store generated files including plots. |

---

## Output Files

All outputs are automatically saved under the specified `output_directory` (default `./out/`), organized by GA run and type of analysis.

### GA Execution Folders
Each independent GA run with a specific random seed creates a folder named `GA_Seed_<seed>/` containing:

- **`fitness_evolution.png`** – Evolution of fitness across generations.
- **`genetic_algorithm_log.txt`** – Detailed log of the GA execution.
- **`hall_of_fame.txt`** – Best solutions found during the run.
- **`hall_of_fame_counter.txt`** – Frequency count of hall-of-fame solutions.

### Results Folder
The `results/` folder contains aggregated analysis and visualizations:

- **`analysis_by_number_of_variables.png`**: This key file helps users make informed decisions when balancing dimensionality reduction and clustering quality.
- **`3D_plot_vars_clusters_fitness.png`** – 3D plot of variables, clusters, and fitness values.
- **`dictionaries_variables_weight_analysis.json`** – Variable selections importances and related metrics as described in the paper.
- **`optimal_variable_selection.csv`** – Selected optimal subset of features.
- **`optimal_variable_selection_and_number_of_clusters.txt`** – Recommended feature subset and number of clusters.
- **`variable_significances.csv`** – Weight of each variable.

### Comparison Plots

- **`comparison_fitness_vs_given_metric.png`** – Shows the fitness values of solutions compared to a target metric (e.g., AMI). Generated with `get_plot_comparing_solution_with_another_metric`.

---

### Synthetic Data Generators

In addition, GAUFS provides two types of synthetic data generators for clustering benchmarking, as presented in the paper.

**Note:** Points within each cluster are scattered around the cluster center, either following a **normal distribution** or a **uniform distribution** within a maximum radius.

#### DataSpheres Generator
Generates ball-shaped clusters with centers distributed across the feature space:

```python
from gaufs import DataGenerator

# Generate ball-shaped clusters
data_balls = DataGenerator.generate_data_spheres(
    num_useful_features=5,
    num_clusters=4,
    num_samples_per_cluster=200,
    num_dummy_unif=10,    # Add 10 uniform noise features
    num_dummy_beta=5,     # Add 5 beta-distributed noise features
    seed=42
)
```

#### DataCorners Generator
Creates simplex-structured clusters whose centers form orthogonal vertices:

```python
# Generate simplex-structured clusters
data_corners = DataGenerator.generate_data_corners(
    num_useful_features=3,  # Will create 4 clusters (n+1)
    num_samples_per_cluster=150,
    num_dummy_unif=5,
    seed=42
)
```

**Key Differences:**
- **DataSpheres**: Clusters can are placed in a grid in the feature space - good for general clustering scenarios
- **DataCorners**: Clusters form a simplex structure - useful for testing dimensionality reduction and feature selection as clusters are well-separated when projected onto useful dimensions and none of the num_useful_features is redundant.

---
## Custom Fitness

### Clustering Algorithms

GAUFS provides built-in clustering algorithms and supports custom implementations through class extension.

**Available clustering methods:**
- `HierarchicalExperiment` (default) - Agglomerative clustering with Ward, Complete, Average or Single linkage
- `KmeansExperiment` - K-means clustering
- You can extend the `ClusteringExperiment` base class to integrate any clustering algorithm.

### Evaluation Metrics

GAUFS supports both internal and external metrics for evaluating clustering quality, and allows custom metric implementation.

**Internal Metrics** (unsupervised - don't require true labels):
- `SilhouetteScore` (default)
- `CalinskiHarabaszScore`
- `DaviesBouldinScore`
- `DaviesBouldinScoreForMaximization`
- `DunnScore`
- `SSEScore`
- `SSEScoreForMaximization`

**External Metrics** (supervised - require true labels for evaluation):
- `AdjustedRandIndexScore`
- `AdjustedMutualInformationScore`
- `NMIScore`
- `VMeasureScore`
- `FowlkesMallowsScore`
- `FScore`
- `HScore`
- `Chi2`
- `DobPertScore`

**Key difference:** Internal metrics optimize clustering without labels (true unsupervised learning), while external metrics are used for validation and comparison when ground truth is available.

**Note:** You can extend the `EvaluationMetric` base class to implement custom metrics.

---

## Examples

Two comprehensive demo scripts are provided to illustrate GAUFS capabilities:

### Demo 1: Basic Usage with Corner Distribution (`demo/demo1.py`)

This example demonstrates the standard GAUFS workflow using synthetic data with a simplex (corner) structure:

- **Data characteristics:**
  - 4 useful clustering features
  - 2 uniform noise features + 2 beta-distributed noise features
  - 3 clusters forming a corner/simplex structure
  - 50 samples per cluster (150 total)

- **Workflow:**
  - Generates synthetic data using `DataGenerator.generate_data_corners()`
  - Runs GAUFS with default settings (unsupervised mode)
  - Compares results against ground truth using Adjusted Mutual Information
  - Produces visualization plots and analysis outputs

### Demo 2: Advanced Configuration with Spherical Clusters (`demo/demo2.py`)

This example showcases GAUFS in a supervised scenario with custom configuration:

- **Data characteristics:**
  - 2 useful clustering features
  - 4 clusters with spherical distribution
  - 1 uniform noise feature + 1 beta-distributed noise feature
  - 50 samples per cluster (200 total)

- **Advanced features demonstrated:**
  - Custom clustering method (`KmeansExperiment`)
  - External evaluation metric (AMI with known labels)
  - Tighter cluster search range (3–5 clusters). As explained in the paper, we recommend not reducing the cluster search range to a single value, even when the number of true labels is known.
  - Comparison with alternative metrics (NMI)

**To run the demos:**
```bash
# Clone the repository
git clone https://github.com/salva24/GAUFS.git
cd GAUFS

# Install the package
pip install -e .

# Run demo 1 (corners and basic usage)
python demo/demo1.py

# Run demo 2 (spheres and advanced configuration)
python demo/demo2.py
```

Both demos generate comprehensive outputs including plots, analysis files, and performance metrics in the `examples\out\` directory.

---

## Documentation

GAUFS includes comprehensive Sphinx documentation. To build the documentation locally:
```bash
# Clone the repository
git clone https://github.com/salva24/GAUFS.git
cd GAUFS

# Install the package with documentation dependencies
pip install -e ".[docs]"

# Build the HTML documentation
python -m sphinx -b html docs/source docs/build/html

# Open the documentation in your browser
# On Linux/Mac:
open docs/build/html/index.html
# On Windows:
start docs/build/html/index.html
```

---

## Project Structure

```
gaufs/
├── src/                               # Source code folder
│   └── gaufs/                         # Main package
│       ├── __init__.py                # Main API exports
│       ├── gaufs.py                   # Core GAUFS algorithm
│       ├── data_generator.py          # Synthetic data generators
│       ├── clustering_experiments/    # Clustering implementations
│       │   ├── __init__.py
│       │   ├── base.py                # Base class
│       │   ├── hierarchical.py        # Hierarchical clustering
│       │   └── kmeans.py              # K-means clustering
│       ├── evaluation_metrics/        # Evaluation metrics
│       │   ├── __init__.py
│       │   ├── base.py                # Base class
│       │   ├── external.py            # External metrics (ARI, AMI, etc.)
│       │   ├── internal.py            # Internal metrics (Silhouette, etc.)
│       │   └── utils.py               # Private utility functions
│       ├── genetic_search.py          # Private Genetic Algorithm implementation
│       └── utils.py                   # Private helper functions and functions to read csv files
├── tests/                             # Tests
│   └── test_main.py                   # Execution test
├── examples/                          # Demo examples
│   ├── demo1.py                       # First demo script
│   ├── datasets/                      # Folder for datasets
│   └── out/                           # Folder for GAUFS output and results
├── docs/                              # Documentation
├── .github/                           # GitHub workflows
│   └── workflows/
│       ├── publish.yml                # Publishing workflow
│       └── tests-build.yml            # CI tests workflow
├── LICENSE                            # Apache 2.0 License
├── NOTICE.txt                         # Attribution information
├── README.md                          # This file
├── pyproject.toml                     # Package setup
└── .gitignore                         # Git ignore rules
```

---

## Acknowledgments

This work has been developed by researchers from **MINERVA AI-Lab**, Institute of Computer Engineering, **University of Seville**, Spain.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.  
Additional attribution and authorship information is provided in the [NOTICE](NOTICE.txt) file.

---

## Library Authors and Contact Information

**Author:** Salvador de la Torre Gonzalez  
**Email:** delatorregonzalezsalvador@gmail.com

**Co-authors:**  
- Antonio Bello Castro  
- José M. Núñez Portero

---

## Support

For questions, issues, or feature requests of this open-source software:
- Open an issue on [GitHub](https://github.com/salva24/GAUFS/issues)
- Contact the author via email

---

**Happy Clustering! 🧬📊**
