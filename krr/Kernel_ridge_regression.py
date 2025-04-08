'''An implementation of Kernel Ridge Regression using custom kernels.'''
# Standard library imports
import os
import time
from typing import List, Optional

# Third-party imports
import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem, MolFromSmiles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Singular matrix in solving dual problem.*")

# Local application imports
from .kernels_library import TanimotoKernel, DiceKernel, laplacian_kernel, gaussian_kernel


# STATIC OBJECT. KERNEL_LIBRARY
# %kernel type% = namedtuple(str,fn)
# interp. a static method to retrieve kernel libraries.
# Making it a static method allows for easy access without needing to instantiate the class.
class KernelLibrary:
    fp = {"tanimoto": TanimotoKernel(),"dice": DiceKernel()}
    td = {"laplacian": laplacian_kernel,"gaussian": gaussian_kernel}

    @staticmethod
    def get_kernel(kernel_type: str, is_td: bool) -> Optional[callable]:
        """Get the kernel function/class"""
        # make sure the kernel type is lower case to avoid typing errors
        _kt = kernel_type.lower()
        # check if the kernel type is in the kernel library
        if is_td:
            if _kt in KernelLibrary.td:
                return KernelLibrary.td[_kt]
        else:
            if _kt in KernelLibrary.fp:
                return KernelLibrary.fp[_kt]

        _type = "Topological Descriptor" if is_td else "Fingerprint Kernels"
        raise ValueError(f"Kernel type '{kernel_type.upper()} of type {_type.upper()}' is not supported.")

# CD. KERNEL_RIDGE_MODEL
# kernel = KernelRidgeModel()
# interp. An encapsulation of Kernel Ridge Regression with custom kernels.
class KernelRidgeModel:
    """Encapsulates Kernel Ridge Regression with custom kernels."""
    def __init__(self, kernel_name: str, is_td):
        self.is_td = is_td
        self.kernel = KernelLibrary.get_kernel(kernel_name, self.is_td) #pick the right kernel type based on the atomic representation
        self.best_alpha = None # 
        self.krr = None # 

    def compute_kernel_matrices(self, x_train, x_test, sigma=None):
        '''Compute the kernel matrices for training and test data.'''
        if self.is_td:
            K_train = self.kernel(x_train, x_train, sigma=sigma) 
            K_test = self.kernel(x_train, x_test, sigma=sigma)
            return K_train, K_test
        else:
            K_train = self.kernel(x_train).evaluate()
            K_test = self.kernel(x_train, x_test).evaluate()
            return K_train, K_test

    def find_best_alpha(self, K_train, y_train):
        '''Find the best alpha using GridSearchCV.'''
        alphas = [10**i for i in np.linspace(-10, 2, num=100)]
        gridsearch = GridSearchCV(KernelRidge(kernel='precomputed'), {'alpha': alphas}, cv=5, scoring='r2')
        gridsearch.fit(K_train, y_train)
        self.best_alpha = gridsearch.best_params_['alpha']
        return self.best_alpha

    def train(self, K_train, y_train, regularization=True):
        '''Train the Kernel Ridge Regression model.'''
        if regularization:
            self.best_alpha = self.find_best_alpha(K_train, y_train)
        else:
            self.best_alpha = 0.0

        print("Regularization Coef:", self.best_alpha)
        self.krr = KernelRidge(alpha=self.best_alpha, kernel='precomputed')
        self.krr.fit(K_train, y_train)

    def predict(self, K_test):
        return self.krr.predict(K_test.T)
    
    
# FD. ecfp_fingerprints()
def ecfp_fingerprints(smiles: List[str], bond_radius: int = 3, nBits: int = 2048) -> np.ndarray:
    """Generates ECFP fingerprints from SMILES strings."""
    rdkit_mols = [MolFromSmiles(s) for s in smiles]
    fpgen = AllChem.GetMorganGenerator(radius=bond_radius, fpSize=nBits)
    fps = [fpgen.GetFingerprint(mol) for mol in rdkit_mols]
    return np.array(fps)

# FD. compute_truncated_r2()
def compute_truncated_r2(K_test, y_train, y_test, eigenval, eigenvec, train_set_size, best_alpha):
    """Computes R² scores for truncated eigenvalue cases."""
    percentages = torch.linspace(0.1, 1, 19)
    r2_test, r2_train = [], []

    for p in percentages:
        num_eigenvalues = int(train_set_size * p)
        eigenval_tr, eigenvec_tr = eigenval[-num_eigenvalues:], eigenvec[:, -num_eigenvalues:]

        K_train_r = eigenvec_tr @ torch.diag(eigenval_tr) @ eigenvec_tr.T
        K_test_r = eigenvec_tr @ eigenvec_tr.T @ K_test

        krr = KernelRidge(alpha=best_alpha, kernel='precomputed')

        krr.fit(K_train_r, y_train)

        y_pred_t = krr.predict(K_test_r.T)
        y_pred_tr = krr.predict(K_train_r.T)
        
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x
        y_train = to_numpy(y_train)
        y_test = to_numpy(y_test)
        y_pred_tr = to_numpy(y_pred_tr)
        y_pred_t = to_numpy(y_pred_t)
    
        r2_test.append(r2_score(y_test, y_pred_t))
        r2_train.append(r2_score(y_train, y_pred_tr))

    return percentages, torch.tensor(r2_test), torch.tensor(r2_train)

# FD. save_results()
def save_results(save_path, kernel_name, eigenval, percentages, r2_test, r2_train):
    """Saves computed results into CSV files."""
    csv_path = os.path.join(save_path, kernel_name)
    os.makedirs(csv_path, exist_ok=True)

    results = pd.DataFrame({
        'Percentage of Eigenvalues': percentages.numpy(),
        'R2 Test': r2_test.numpy(),
        'R2 Train': r2_train.numpy()
    })
    results.to_csv(os.path.join(csv_path, 'results.csv'), index=False)

    analysis = pd.DataFrame({'Eigenvalues': eigenval})
    analysis.to_csv(os.path.join(csv_path, 'eigenvalues.csv'), index=False)

# FD. kernel_ridge_regression()
def kernel_ridge_regression(data:pd.DataFrame, save_path, kernel_name:str, n_train: int = 5000, n_test: int = 10000, mol_property: str = 'Gap', representation: str = 'BOB', regularization: bool = False):
    '''Main function to run the kernel ridge regression.'''
    # evaluate if the type of kernel is topological descriptor or fingerprint
    is_td = True if kernel_name.lower() in ["gaussian", "laplacian"] else False
    n_train, n_test = 5000, 10000
    # Select Kernel to Train Model
    model = KernelRidgeModel(kernel_name, is_td)
    is_fingerprint = representation if is_td else "fingerprint"
    print("Program is running with the following parameters:\nkernel type: {kernel_name}\nrepresentation: {is_fingerprint}\nregularization: {regularization}")
    if is_td:      
        property_data = data[mol_property]  
        # Train-test split
        if representation == 'CM':
            rep_data = 'Dataset/coulomb_matrix_rep.npy'
        else:
            rep_data = 'Dataset/bob_rep.npy'
            
        rep = np.load(rep_data)
        x_train, x_test, y_train, y_test = train_test_split(rep, property_data, train_size=n_train, test_size=n_test, random_state=42)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True) 
        K_train, K_test = model.compute_kernel_matrices(x_train.T, x_test.T, sigma=100)
        K_train, K_test = torch.from_numpy(K_train), torch.from_numpy(K_test)
    else:
        smiles, property_data = data['SMILES'], data[mol_property] 
        # Compute fingerprints
        rep = ecfp_fingerprints(smiles)
        x_train, x_test, y_train, y_test = train_test_split(rep, property_data, train_size=n_train, test_size=n_test, random_state=42)
        # Convert to tensors once
        x_train = torch.tensor(x_train.astype(np.float64))
        x_test = torch.tensor(x_test.astype(np.float64))
        y_train = torch.tensor(y_train.values).flatten()
        y_test = torch.tensor(y_test.values).flatten()
        K_train, K_test = model.compute_kernel_matrices(x_train, x_test)

    # Compute eigenvalues
    eigenval, eigenvec = torch.linalg.eigh(K_train)
    # Train model
    model.train(K_train, y_train, regularization=regularization)

    # Compute truncated R2 scores
    percentages, r2_test, r2_train = compute_truncated_r2(K_test, y_train, y_test, eigenval, eigenvec, n_train, model.best_alpha)
    # Save results
    save_results(save_path,kernel_name, eigenval, percentages, r2_test, r2_train)


if __name__ == "__main__":
    # load the dataset
    data_path = 'Dataset/dataset.pkl'
    data = pd.read_pickle(data_path)
    # kernel_ridge_regression(data, kernel_name='tanimoto', mol_property='Gap', regularization=False)
    kernel_ridge_regression(data, save_path="./results", kernel_name='gaussian', mol_property='Gap', representation='BOB', regularization=False)
    