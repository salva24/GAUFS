from src.gaufs import *

def main():
    unlabeled_data, true_labels = read_labeled_data_csv("./datasets/iris_labeled.csv")
    gaufs= Gaufs(seed=0)
    gaufs.set_unlabeled_data(unlabeled_data)
    # #set evaluation metric
    # gaufs.evaluation_metric = AdjustedMutualInformationScore(unlabeled_data= unlabeled_data, true_labels=true_labels)
    gaufs.ngen = 2
    gaufs.run()

    #Comparison with external metric
    gaufs.get_plot_comparing_solution_with_another_metric(AdjustedMutualInformationScore(unlabeled_data= unlabeled_data, true_labels=true_labels), true_number_of_labels=len(set(true_labels)))


    
        
        

if __name__ == "__main__":
    main()
        