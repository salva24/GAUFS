# GAUFS

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

## Project Structure

- `datasets/` - Input CSV files for analysis
- `results/` - Generated output files and visualizations
- `src/` - Source code modules
  - `build_test_genetic_2.py` - Main test builder
  - `genetic2_parallel.py` - Parallel genetic algorithm implementation
  - `alg_clustering.py` - Clustering algorithms and metrics
  - `analysis_weighted_variables_num_cluster.py` - Variable analysis tools
  - `data_generators` - Methods related to synthetic data generation. The file can be executed to generate an example of a corners dataset on ./datasets/synthetic_data_corners.csv
- `main.py` - Main execution script for GAUFS algorithm
