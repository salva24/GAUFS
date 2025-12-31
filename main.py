from src.gaufs import *

def main():
    gaufs= Gaufs(seed=0)
    gaufs.read_unlabeled_data_csv("datasets/ac_5clusters_3vars.csv")
    gaufs.ngen = 2
    gaufs.run()

    print(gaufs.variable_significance)
    
        
        

if __name__ == "__main__":
    main()
        