from build_test_genetic_2 import *
# from analysis_weighted_variables_num_cluster import analyse_dataset
from temp_analysis_weighted_variables_num_cluster import analyse_dataset
import json
from tqdm import tqdm 

def main():
    current_path = os.getcwd()
    path_datasets = os.path.join(current_path, "datasets")    
    
    for file in tqdm(os.listdir(path_datasets)):
        if 'gitkeep' in file:
            print('Saltando gitkeep')
            continue
        if not os.path.isfile(os.path.join(path_datasets, file)):
            print('Saltando folder')
            continue
        
        name_file = file.removesuffix(".csv")
        print(F'Ejecutando el archivo {name_file}')
        dummies=False
        test = BuildTest(name_file,artificiales = False, parallel_evaluation=True, dummies=dummies)
        
        #comentar estas dos lineas para solo sacar graficas
        test.run()
        test.convert_csv()
        
        fitness_internos=["silhouette"]
        linkages=["ward"]
        for fitness in fitness_internos:
            for linkage in linkages:
                analyse_dataset(name_file,max_num_considerado_clusters=test.rango_banda_clusters[0][1],artificiales=False,fitness=fitness,linkage=linkage, parallel_evaluation=True, dummies=dummies)
        

if __name__ == "__main__":
    main()
        