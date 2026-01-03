from src.gaufs import *

def main():
    # unlabeled_data, true_labels = read_labeled_data_csv("./datasets/iris_labeled.csv")
    # gaufs= Gaufs(seed=0)
    # gaufs.set_unlabeled_data(unlabeled_data)
    # #set evaluation metric
    # gaufs.evaluation_metric = AdjustedMutualInformationScore(unlabeled_data= unlabeled_data, true_labels=true_labels)
    # gaufs.ngen = 2
    # gaufs.run()

    gaufs= Gaufs(seed=0)
    gaufs.read_unlabeled_data_csv("./datasets/iris_unlabeled.csv")
    gaufs.ngen = 2
    gaufs.run()


    
        
        

if __name__ == "__main__":
    main()
        