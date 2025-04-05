'''A set of tools to visualize the results of the analysis.'''
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# FD. plot_r2_td()
def plot_r2_td():
    # Define the folder names
    # kernels = ["dice", "tanimoto"]
    kernels = ["laplacian", "gaussian"]

    color_map = {
        'laplacian': 'blue',        
        'gaussian': 'red',
    }

    marker_map = {
        'laplacian': 'o',         
        'gaussian': 's',
    }


    # Dictionary to store data
    data = {}

    # Read the CSV files
    for k in kernels:
        file_path = f"tem_results/No_regularization/{k}/analysis_BOB.csv"
        data[k] = pd.read_csv(file_path)


    # Plot Eigenvalues vs. indices
    plt.figure(figsize=(8, 5))
    for k in kernels:
        color = color_map.get(k, 'black') 
        marker = marker_map.get(k, 'o') 
        eigenvalues = data[k]["Eigenvalues"]
        # Normalize eigenvalues
        normalize_eigenv = (eigenvalues - np.min(eigenvalues)) / (np.max(eigenvalues) - np.min(eigenvalues))
        normalize_eigenv = np.sort(normalize_eigenv)[::-1]
        
        # Plot test data (solid line)
        plt.plot(np.arange(len(normalize_eigenv))[::150], np.log(normalize_eigenv)[::150], 
                label=f"{k.capitalize()}", color=color, marker=marker, 
                markersize=5, linewidth=1.5, linestyle='-')
        

    plt.xlabel("Index", fontsize = 14)
    plt.ylabel("Log(Normalized Eigenvalue)", fontsize = 14)
    # plt.ylim(-0.005, 1.01)
    plt.legend(fontsize=8)


    # Save plot
    plot_path = 'Figures/fingerprint_kernels'
    os.makedirs(plot_path, exist_ok=True)
    output_file = os.path.join(plot_path, f"r2_global_Eig_BOB.png")
    plt.savefig(output_file, dpi=600)
    plt.close()



###############################################################################################
# FD. plot_r2_fp()
def plot_r2_fp():
    # Define the folder names
    kernels = ["dice", "tanimoto"]

    # Color and marker maps
    color_map = {
        'dice': 'blue',        
        'tanimoto': 'red',
        'otsuka': 'green',
        'sogenfrei': 'orange',
        'braunblanquet': 'purple',
    }

    marker_map = {
        'dice': 'o',         
        'tanimoto': 's',     
        'otsuka': '^',       
        'sogenfrei': 'v',    
        'braunblanquet': 'D',
    }

    # Dictionary to store data
    data = {}

    # Read the CSV files
    for k in kernels:
        file_path = f"tem_results/Regularization/{k}/r2_results_Gap.csv"
        data[k] = pd.read_csv(file_path)


    # Plot R2 Train and Test vs. Percentage of Eigenvalues
    plt.figure(figsize=(8, 5))
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [10, 1]})
    plt.subplots_adjust(hspace=0.05)
    for k in kernels:
        color = color_map.get(k, 'black') 
        marker = marker_map.get(k, 'o') 
        
        # Plot test data (solid line)
        plt.plot(data[k]["Train Size"], data[k]["R2 Test"], 
                label=f"{k.capitalize()} Test", color=color, marker=marker, 
                markersize=4, linewidth=1, linestyle='-')
        # ax1.plot(data[k]["Train Size"], data[k]["R2 Test"], marker=marker, linestyle='-', label=f'{k.capitalize()} Test', color=color,markersize=4, linewidth=1)
        # ax2.plot(data[k]["Train Size"], data[k]["R2 Test"], marker=marker, linestyle='-', label=f'{k.capitalize()} Test',color=color,markersize=4, linewidth=1)
        # Plot train data (dashed line)
        plt.plot(data[k]["Train Size"], data[k]["R2 Train"], 
                label=f"{k.capitalize()} Train", color=color, marker=marker, 
                markersize=4, linewidth=1, linestyle='--')
        # ax1.plot(data[k]["Train Size"], data[k]["R2 Train"], marker=marker, linestyle='--', label=f'{k.capitalize()} Train', color=color,markersize=4, linewidth=1)
        # ax2.plot(data[k]["Train Size"], data[k]["R2 Train"], marker=marker, linestyle='--', label=f'{k.capitalize()} Train', color=color,markersize=4, linewidth=1)

    plt.xlabel("Number of training data",fontsize = 14)
    plt.ylabel(r"$R^2$",fontsize = 14)
    plt.ylim(0.2, 1.1)
    plt.legend(fontsize=6)

    # Save plot
    plot_path = 'Figures/fingerprint_kernels'
    os.makedirs(plot_path, exist_ok=True)
    output_file = os.path.join(plot_path, f"r2_fingerprint_train_test_reg_nsamples.png")
    plt.savefig(output_file, dpi=600)
    plt.close()


if __name__ == "__main__":
    print("Running visualization_tools.py")
    file_path_td = ""
    file_path_fp = ""
    # plot_r2_td(file_path_td)
    # plot_r2_fp(file_path_fp)