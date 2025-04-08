# from krr.kernel_ridge_regression import kernel_ridge_regression
from krr.Kernel_ridge_regression import kernel_ridge_regression
import pandas as pd

if __name__ == "__main__":
    # load the dataset
    data_path = 'Dataset/dataset.pkl'
    output_path = "./results"
    data = pd.read_pickle(data_path)
    kernel_ridge_regression(data, save_path="./results",kernel_name='gaussian', mol_property='Gap', representation='BOB', regularization=False)
