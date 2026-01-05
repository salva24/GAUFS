#for debug
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from gaufs import Gaufs
from gaufs import read_labeled_data_csv
from gaufs.evaluation_metrics import AdjustedMutualInformationScore


def main():
    unlabeled_data, true_labels = read_labeled_data_csv(
        "./datasets/data_corners_6clusters_1.csv"
    )
    gaufs = Gaufs(seed=0, num_genetic_executions=1)
    gaufs.set_unlabeled_data(unlabeled_data)
    gaufs.ngen=3
    gaufs.run()
    # Comparison with external metric
    gaufs.get_plot_comparing_solution_with_another_metric(
        AdjustedMutualInformationScore(true_labels=true_labels),
        true_number_of_labels=len(set(true_labels)),
    )


if __name__ == "__main__":
    main()
