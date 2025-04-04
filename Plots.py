import pandas as pd
import matplotlib.pyplot as plt
import os

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


# # Dictionary to store data
# data = {}

# # Read the CSV files
# for k in kernels:
#     file_path = f"tem_results/Regularization/{k}/results_CV.csv"
#     data[k] = pd.read_csv(file_path)


# # Plot R2 Train and Test vs. Percentage of Eigenvalues
# # plt.figure(figsize=(8, 5))
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.03})
# for k in kernels:
#     color = color_map.get(k, 'black') 
#     marker = marker_map.get(k, 'o') 
    
#     # Plot test data (solid line)
#     plt.plot(data[k]["Percentage of Eigenvalues"], data[k]["R2 Test"], 
#              label=f"{k.capitalize()} Test", color=color, marker=marker, 
#              markersize=5, linewidth=1.5, linestyle='-')
    
#     ax1.plot(data[k]["Percentage of Eigenvalues"], data[k]["R2 Test"], marker=marker, linestyle='-', label=f'{k.capitalize()}', color=color, markersize=5, linewidth=1.5)
#     ax2.plot(data[k]["Percentage of Eigenvalues"], data[k]["R2 Test"], marker=marker, linestyle='-', label=f'{k.capitalize()}',color=color, markersize=5, linewidth=1.5)
    
#     # Plot train data (dashed line)
#     plt.plot(data[k]["Percentage of Eigenvalues"], data[k]["R2 Train"], 
#              label=f"{k.capitalize()} Train", color=color, marker=marker, 
#              markersize=5, linewidth=1.5, linestyle='--')
    
#     ax1.plot(data[k]["Percentage of Eigenvalues"], data[k]["R2 Train"], marker=marker, linestyle='--', label=f'{k.capitalize()}', color=color, markersize=5, linewidth=1.5)
#     ax2.plot(data[k]["Percentage of Eigenvalues"], data[k]["R2 Train"], marker=marker, linestyle='--', label=f'{k.capitalize()}',color=color, markersize=5, linewidth=1.5)

# # plt.xlabel("Percentage of Eigenvalues", fontzise = 14)
# # plt.ylabel(r"$R^2$", fontsize = 14)
# # # plt.ylim(-0.005, 1.01)
# # plt.legend(fontsize=8)

# ax1.set_ylim(0.2, 1.1)  
# ax2.set_ylim(-9, -1)
# ax1.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax1.tick_params(bottom=False)  
# d = 0.005 
# kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
# ax1.plot((-d, +d), (-d, +d), **kwargs) 
# ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs) 
# kwargs.update(transform=ax2.transAxes)  
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs) 
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
# ax2.set_xlabel('Percentage of Eigenvalues', fontsize = 14)
# ax1.set_ylabel(r"$R^2$", fontsize = 14)
# ax1.legend(fontsize=8)

# # Save plot
# plot_path = 'Figures/fingerprint_kernels'
# os.makedirs(plot_path, exist_ok=True)
# output_file = os.path.join(plot_path, f"r2_fingerprint_train_test_reg_CV.png")
# plt.savefig(output_file, dpi=600)
# plt.close()

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
# ax1.set_ylim(0.0, 1.05)  
# ax2.set_ylim(-20, -5)
# ax1.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax1.tick_params(bottom=False)  
# d = 0.010  
# kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
# ax1.plot((-d, +d), (-d, +d), **kwargs) 
# ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs) 
# kwargs.update(transform=ax2.transAxes)  
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs) 
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
# ax2.set_xlabel('Number of training data', fontsize = 14)
# ax1.set_ylabel(r"$R^2$", fontsize = 14)
# ax1.legend(fontsize=6)

# Save plot
plot_path = 'Figures/fingerprint_kernels'
os.makedirs(plot_path, exist_ok=True)
output_file = os.path.join(plot_path, f"r2_fingerprint_train_test_reg_nsamples.png")
plt.savefig(output_file, dpi=600)
plt.close()
