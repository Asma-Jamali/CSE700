# Kernel Ridge Regression for Molecular Properties

## Authors
### Asma Jamali  (https://asma-jamali.github.io/Website/)  
### Uriel Garcilazo Cruz (https://urielgarcilazo.com/)  

# Kernel Ridge Regression for Molecular Property Prediction

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)

## Features
- **Custom Kernels**:
  - Fingerprint-based: Tanimoto, Dice
  - Topological descriptor-based: Gaussian, Laplacian
- **Spectral Analysis**: Eigenvalue truncation for improved stability
- **Automated Optimization**: Grid search for regularization parameter (α)
- **Molecular Representations**:
  ```mermaid
  graph LR
    A[Molecular Representations] --> B[Global]
    A --> C[Local]
    B --> D[Coulomb Matrix]
    B --> E[Bag-of-Bonds]
    C --> F[ECFP Fingerprints]

## Project Overview
This project implements global and local matrix representations of molecular properties (HOMO-LUMO gap and heat capacity) from the QM9 dataset. It features a comparative framework to evaluate the effects of trimming eigenvalues during kernel ridge regression (KRR) implementation.

Our `krr` Python package supports custom kernels (Tanimoto, Dice, Gaussian, and Laplacian) to predict molecular properties from the QM9 dataset, with eigenvalue truncation for performance optimization.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Asma-Jamali/CSE700.git
```

You can also access the repository directly at https://github.com/Asma-Jamali/CSE700/tree/KRR and download it as a ZIP file if you prefer.

### 2. Download the Dataset
The dataset is too large for GitHub. Please contact the authors for access.  

After downloading, place the dataset files in the Dataset folder at the same level as main.py.
### 3. Install the Package
Navigate to the directory containing setup.py and run:

```
pip install -e .
```

This creates an editable installation that allows you to modify the code and see changes without reinstalling.
Requirements

Python ≥ 3.9
pip (included with Python installations)

We recommend using a virtual environment to avoid package conflicts.  

```
.
|-- CSE700
    |-- Dataset
    |   |-- bob_rep.npy
    |   |-- coulomb_matrix_rep.npy
    |   `-- dataset.pkl
    |-- krr
    |   |-- Kernel_ridge_regression.py
    |   `-- kernels_library.py
    |-- main.py
    `-- setup.py
```

### Usage
To run the program, navigate to the directory containing main.py and execute:
```
python main.py
```

### Configuration
In the current version, you only need to specify the dataset and output paths in main.py:

```
dataset_path = 'Dataset/dataset.pkl'
output_path = '/results/'
```

### Output
The program generates two CSV files:
```
results.csv: Contains the predicted and actual values from the KRR model
eigenvalues.csv: Contains the eigenvalues used in the model, including trimmed eigenvalues if applicable
```

### Technical Background
The project focuses on implementing Kernel Ridge Regression with various kernels to predict molecular properties from the QM9 dataset. The implementation includes eigenvalue truncation techniques to optimize performance.
