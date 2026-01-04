from src.gaufs import *
from src.utils import *

def main():
    unlabeled_data, true_labels = read_labeled_data_csv("./datasets/iris_labeled.csv")
    gaufs= Gaufs(seed=0, num_genetic_executions=3)
    gaufs.set_unlabeled_data(unlabeled_data)
    gaufs.ngen=2
    gaufs.run()
    # Comparison with external metric
    gaufs.get_plot_comparing_solution_with_another_metric(AdjustedMutualInformationScore(true_labels=true_labels), true_number_of_labels=len(set(true_labels)))

    #Debug
    # clust, var= 2, [0,0,1,0]
    # val=evaluate_ind(unlabeled_data=unlabeled_data, cluster_number=clust, variables=var, clustering_method=gaufs.clustering_method, evaluation_metric=gaufs.evaluation_metric)
    # print(f"Fitness for {clust} clusters and variable selection {var}: {val}")

if __name__ == "__main__":
    main()
        