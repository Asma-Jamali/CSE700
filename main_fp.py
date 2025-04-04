from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import os
import time

from rdkit.Chem import AllChem, MolFromSmiles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

from tanimoto_kernel import TanimotoKernel
from dice_kernel import DiceKernel


# Available kernels
Kernels_NAME = {
    "tanimoto": TanimotoKernel,
    "dice": DiceKernel
}

def ecfp_fingerprints(smiles: List[str], bond_radius: int = 3, nBits: int = 2048) -> np.ndarray:
    """Generates ECFP fingerprints from SMILES strings."""
    rdkit_mols = [MolFromSmiles(s) for s in smiles]
    fpgen = AllChem.GetMorganGenerator(radius=bond_radius, fpSize=nBits)
    fps = [fpgen.GetFingerprint(mol) for mol in rdkit_mols]
    return np.array(fps)

class KernelRidgeModel:
    """Encapsulates Kernel Ridge Regression with custom kernels."""
    def __init__(self, kernel_name: str):
        if kernel_name not in Kernels_NAME:
            raise ValueError(f"Unknown kernel: {kernel_name}. Available options: {list(Kernels_NAME.keys())}")
        self.kernel = Kernels_NAME[kernel_name]()
        self.best_alpha = None
        self.krr = None

    def compute_kernel_matrices(self, X_train, X_test):
        K_train = self.kernel(X_train).evaluate()
        K_test = self.kernel(X_train, X_test).evaluate()
        return K_train, K_test


    def find_best_alpha(self, K_train, y_train):
        alphas = [10**i for i in np.linspace(-10, 2, num=100)]
        gridsearch = GridSearchCV(KernelRidge(kernel='precomputed'), {'alpha': alphas}, cv=5, scoring='r2')
        gridsearch.fit(K_train, y_train)
        self.best_alpha = gridsearch.best_params_['alpha']
        return self.best_alpha

    def train(self, K_train, y_train, regularization=True):
        if regularization:
            self.best_alpha = self.find_best_alpha(K_train, y_train)
        else:
            self.best_alpha = 0.0

        print("Regularization Coef:", self.best_alpha)
        self.krr = KernelRidge(alpha=self.best_alpha, kernel='precomputed')
        self.krr.fit(K_train, y_train)

    def predict(self, K_test):
        return self.krr.predict(K_test.T)

def compute_truncated_r2(K_test, y_train, y_test, eigenval, eigenvec, train_set_size, best_alpha):
    """Computes R² scores for truncated eigenvalue cases."""
    percentages = torch.linspace(0.1, 1, 19)
    r2_test, r2_train, times = [], [], []

    for p in percentages:
        num_eigenvalues = int(train_set_size * p)
        eigenval_tr, eigenvec_tr = eigenval[-num_eigenvalues:], eigenvec[:, -num_eigenvalues:]

        K_train_r = eigenvec_tr @ torch.diag(eigenval_tr) @ eigenvec_tr.T
        K_test_r = eigenvec_tr @ eigenvec_tr.T @ K_test

        krr = KernelRidge(alpha=best_alpha, kernel='precomputed')

        start_time = time.time()
        krr.fit(K_train_r, y_train)
        end_time = time.time()

        y_pred_t = krr.predict(K_test_r.T)
        y_pred_tr = krr.predict(K_train_r.T)

        r2_test.append(r2_score(y_test, y_pred_t))
        r2_train.append(r2_score(y_train, y_pred_tr))
        times.append(end_time - start_time)
        print('Time:', end_time - start_time)

    return percentages, torch.tensor(r2_test), torch.tensor(r2_train), torch.tensor(times)

def save_results(kernel_name, eigenval, ta, percentages, r2_test, r2_train, time):
    """Saves computed results into CSV files."""
    csv_path = f'./tem_results/No_regularization/{kernel_name}'
    os.makedirs(csv_path, exist_ok=True)

    results = pd.DataFrame({
        'Percentage of Eigenvalues': percentages.numpy(),
        'R2 Test': r2_test.numpy(),
        'R2 Train': r2_train.numpy(),
        'Time (s)': time.numpy(),
    })
    results.to_csv(os.path.join(csv_path, 'results_time.csv'), index=False)

    analysis = pd.DataFrame({'Eigenvalues': eigenval, 'Target Alignment': ta})
    analysis.to_csv(os.path.join(csv_path, 'analysis_time.csv'), index=False)

# === Main Execution ===
data_path = 'Dataset/dataset.pkl'
data = pd.read_pickle(data_path)
smiles, gap = data['SMILES'], data['Gap']

# Compute fingerprints
rep = ecfp_fingerprints(smiles)
print(rep.shape)

# # Train-test split
n_train, n_test = 5000, 10000
# n_test = 10000

X_train, X_test, y_train, y_test = train_test_split(rep, gap, train_size=n_train, test_size=n_test, random_state=42)

# Convert to tensors once
X_train = torch.tensor(X_train.astype(np.float64))
X_test = torch.tensor(X_test.astype(np.float64))
y_train = torch.tensor(y_train.values).flatten()
y_test = torch.tensor(y_test.values).flatten()
 

# # Select Kernel and Train Model
kernel_name = "tanimoto"  # Change to "tanimoto" for different kernel
model = KernelRidgeModel(kernel_name)
K_train, K_test = model.compute_kernel_matrices(X_train, X_test)

# Compute eigenvalues
eigenval, eigenvec = torch.linalg.eigh(K_train)

# Train model
model.train(K_train, y_train, regularization=False)

# Predict and compute Target Alignment
y_pred = model.predict(K_train)
ta = eigenvec.T @ y_pred / np.sqrt(n_train)

# Compute truncated R2 scores
percentages, r2_test, r2_train, times = compute_truncated_r2(K_test, y_train, y_test, eigenval, eigenvec, n_train, model.best_alpha)

# Save results
save_results(kernel_name, eigenval, ta, percentages, r2_test, r2_train, times)

# train_sizes = np.linspace(500, 10000, num=20, dtype=int)
# results = []

# for n_train in train_sizes:
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(rep, gap, train_size=n_train, test_size=n_test, random_state=42)
    
#     # Convert to tensors
#     X_train = torch.tensor(X_train.astype(np.float64))
#     X_test = torch.tensor(X_test.astype(np.float64))
#     y_train = torch.tensor(y_train.values).flatten()
#     y_test = torch.tensor(y_test.values).flatten()
    
#     # Select Kernel and Train Model
#     model = KernelRidgeModel(kernel_name)
#     K_train, K_test = model.compute_kernel_matrices(X_train, X_test)
    
#     # Train model
#     model.train(K_train, y_train, regularization=False)
    
#     # Predict
#     y_pred_train = model.predict(K_train)
#     y_pred_test = model.predict(K_test)
    
#     # Compute R2 scores
#     r2_train = r2_score(y_train, y_pred_train)
#     r2_test = r2_score(y_test, y_pred_test)
    
#     results.append({'Train Size': n_train, 'R2 Train': r2_train, 'R2 Test': r2_test, 'Reg_coeff': model.best_alpha})

# # Convert to DataFrame and save
# results_df = pd.DataFrame(results)
# results_df.to_csv(f'./tem_results/No_Regularization/{kernel_name}/r2_results_Gap.csv', index=False)
