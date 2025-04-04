
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import os
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, rdMolDescriptors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

from tanimoto_kernel import TanimotoKernel
from dice_kernel import DiceKernel

def ecfp_fingerprints(
    smiles: List[str],
    bond_radius: Optional[int] = 3,
    nBits: Optional[int] = 2048,
) -> np.ndarray:
    """
    Builds molecular representation as a binary ECFP fingerprints.

    :param smiles: list of molecular smiles
    :type smiles: list
    :param bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
    :type bond_radius: int
    :param nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048
    :type nBits: int
    :return: array of shape [len(smiles), nBits] with ecfp featurised molecules

    """

    rdkit_mols = [MolFromSmiles(s) for s in smiles]
    fpgen = AllChem.GetMorganGenerator(radius=bond_radius, fpSize=nBits)
    fps = [fpgen.GetFingerprint(mol) for mol in rdkit_mols]
    return np.array(fps)


data_path= 'Dataset/dataset.pkl'
data = pd.read_pickle(data_path)
smiles = data['SMILES']
gap = data['Gap']
rep = ecfp_fingerprints(smiles)
print(rep.shape)

n_train = 1000
n_test = 2000
X_train, X_test, y_train, y_test = train_test_split(rep, gap, train_size=n_train,
                                                            test_size=n_test, random_state=42)

X_train = torch.tensor(X_train.astype(np.float64))
X_test = torch.tensor(X_test.astype(np.float64))
y_train = torch.tensor(y_train.values).flatten()
y_test = torch.tensor(y_test.values).flatten()

# kernel = TanimotoKernel()
kernel = DiceKernel()
K_train = kernel(X_train).evaluate()
K_test = kernel(X_train, X_test).evaluate() 

# u, s, v = torch.svd(K_train)
# df_u = pd.DataFrame(u.numpy())

eigenval, eigenvec = torch.linalg.eigh(K_train)

csv_path = f'./tem_results/Dice/'
os.makedirs(csv_path, exist_ok=True)
# u_array = u.numpy().flatten() 
# df_u.to_csv(os.path.join(csv_path, 'u.csv'), index=False)
df_eigvec = pd.DataFrame(eigenvec.numpy())
df_eigvec.to_csv(os.path.join(csv_path, 'eigvec.csv'), index=False)


# Perform regularization
regularization = True
def find_best_alpha(K_train, y_train):
    alphas = [10**i for i in np.linspace(-10, 2, num=100)]
    gridsearch = GridSearchCV(KernelRidge(kernel='precomputed'), {'alpha': alphas}, cv=5, scoring='r2')
    gridsearch.fit(K_train, y_train)
    return gridsearch.best_params_['alpha']

if regularization:
    best_alpha = find_best_alpha(K_train, y_train)
    print("Regularization Coef:", best_alpha)
else:
    best_alpha = 0.0 # No regularization
    
# Kernel ridge regression
krr = KernelRidge(alpha=best_alpha, kernel='precomputed')
krr.fit(K_train, y_train)
y_pred = krr.predict(K_train.T)
ta = eigenvec.T @ y_pred / np.sqrt(n_train)

def compute_truncated_r2(K_test, y_train, y_test, eigenval, eigenvec, train_set_size, best_alpha):
    percentage = torch.linspace(0.1, 1, 19)
    r2_test, r2_train = torch.tensor([]), torch.tensor([])
    
    for p in percentage:
        num_eigenvalues = int(train_set_size * p)
        eigenval_tr, eigenvec_tr = eigenval[-num_eigenvalues:], eigenvec[:, -num_eigenvalues:]
        
        K_train_r = eigenvec_tr @ torch.diag(eigenval_tr) @ eigenvec_tr.T
        K_test_r = eigenvec_tr @ eigenvec_tr.T @ K_test
        
        krr = KernelRidge(alpha=best_alpha, kernel='precomputed')
        krr.fit(K_train_r, y_train)
        
        y_pred_t = krr.predict(K_test_r.T).reshape(-1, 1)
        y_pred_tr = krr.predict(K_train_r.T).reshape(-1, 1)
        

        r2_test_val = r2_score(y_test, y_pred_t)
        r2_train_val = r2_score(y_train, y_pred_tr)
        
        r2_test = torch.cat((r2_test, torch.tensor(r2_test_val).reshape(1)))
        r2_train = torch.cat((r2_train, torch.tensor(r2_train_val).reshape(1)))
        
    return percentage, r2_test, r2_train


# Get sorted indices (ascending order)
# sorted_indices = s.argsort()

# # Sort eigenvalues and eigenvectors
# eigenval = s[sorted_indices]
# eigenvec = u[:, sorted_indices]
# df_u = pd.DataFrame(eigenvec.numpy())
# df_u.to_csv(os.path.join(csv_path, 'u.csv'), index=False)

# print(eigenval)
# Truncate eigenvalues and compute R2 scores
percentage, r2_test, r2_train = compute_truncated_r2(K_test, y_train, y_test, eigenval, eigenvec, n_train, best_alpha)
print('regularization coef:', best_alpha)
def save_results(kernel, eigenval,ta, percentage, r2_test, r2_train):
    csv_path = f'./tem_results/{kernel}'
    os.makedirs(csv_path, exist_ok=True)
    results = pd.DataFrame({'Percentage of Eigenvalues': percentage.numpy(),
                       'R2 Test': r2_test.numpy(),
                       'R2 Train': r2_train.numpy()})
    results.to_csv(os.path.join(csv_path, 'results.csv'), index=False)
    
    analysis = pd.DataFrame({'Eigenvalues': eigenval,
                        'Target Alignment':ta})
    analysis.to_csv(os.path.join(csv_path, 'analysis.csv'), index=False)

save_results('Dice', eigenval, ta, percentage, r2_test, r2_train)

