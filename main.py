from src.gaufs import *

def main():
    unlabeled_data, true_labels = read_labeled_data_csv("./datasets/iris_labeled.csv")
    gaufs= Gaufs(seed=0, num_genetic_executions=2)
    gaufs.set_unlabeled_data(unlabeled_data)
    #set evaluation metric
    gaufs.evaluation_metric = AdjustedMutualInformationScore(unlabeled_data= unlabeled_data, true_labels=true_labels)
    gaufs.ngen = 2
    gaufs.run()

    print(gaufs.variable_significance)
    

    
        
        

if __name__ == "__main__":
    main()
        