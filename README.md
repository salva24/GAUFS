# GAUFS

This repository provides the source code and experimental scripts associated with the Genetic Algorithm for Unsupervised Feature Selection (GAUFS) framework.  
It includes all components required to reproduce the results presented in the paper, as well as the experimental setup used for comparison with alternative methods.

**Note:** The latest version of GAUFS, including all data generators used in the experiments, is available on the [`main` branch](https://github.com/salva24/GAUFS.git) of this repository.

---

## Usage

To execute the code:

1. Place your datasets as CSV files in the [`datasets/`](datasets/) folder
2. Ensure your CSV files follow this format:
    - Column headers: `var-0,var-1,var-2,var-3,...,var-n,ETIQ`
    - Values should be normalized between [0,1]
    - `ETIQ` is the target column for classification
3. Run the main script:
    ```sh
    python main.py
    ```

After execution, results will be generated in the `results/` folder. The script will automatically process all CSV files in the `datasets/` folder and generate analysis results including clustering metrics and visualizations.

---

## Project Structure

- `datasets/` - Input CSV files for analysis
- `results/` - Generated output files and visualizations
- `src/` - Source code modules
  - `build_test_genetic_2.py` - Main test builder
  - `genetic2_parallel.py` - Parallel genetic algorithm implementation
  - `alg_clustering.py` - Clustering algorithms and metrics
  - `analysis_weighted_variables_num_cluster.py` - Variable analysis tools
  - `data_generators` - Methods related to synthetic data generation. The file can be executed to generate an example of a corners dataset on ./datasets/synthetic_data_corners.csv
- `main.py` - Main execution script for GAUFS algorithm/
- `comparison/` — Files used for comparison with the AutoUFS tool.
  - `dataset-papers/` — Input CSV files for the comparison.
  - `results-papers/` — Output results, containing one folder for each compared dataset.
  - `AutoUFSTool-main/` — Folder cloned from the AutoUFS-tool GitHub repository.
    - `main-comparison.m` — MATLAB script to run the comparison.
  - `alg_clustering.py` — Python module with utility functions for clustering.
  - `automate_v2.py` — Python script to run the automatic comparison process.
  - `datasets_mat.ipynb` — Jupyter notebook with tools for converting CSV files into MATLAB structures.

---

## Acknowledgments

This work has been developed by researchers from MINERVA AI-Lab, Institute of Computer Engineering, University of Seville, Spain.

This research has been funded by the Ministerio de Ciencia, Innovación y Universidades (Spain) under project PID2023-146037OB-C21, funded by MCIU/AEI/10.13039/501100011033.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.  
Additional attribution and authorship information is provided in the [NOTICE](NOTICE.txt) file.

---
## Software Authors and Contact Information
**Author:** Salvador de la Torre Gonzalez, email: *delatorregonzalezsalvador at gmail.com*

**Author:** Antonio Bello Castro

**Author:** José M. Núñez Portero

