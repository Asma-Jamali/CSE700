from kernel_ridge_regression import kernel_ridge_regression
import pandas as pd

if __name__ == "__main__":
    # load the dataset
    data_path = 'Dataset/dataset.pkl'
    data = pd.read_pickle(data_path)
    kernel_ridge_regression(data_path, data, kernel_name='gaussian')
    