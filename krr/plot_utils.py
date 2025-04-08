import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools

# Define which kernels belong to which type
kernel_types = {
    "fingerprint_based": ["dice", "tanimoto"],
    "distance_based": ["laplacian", "gaussian"]
}

output_base_path = "Figures/"
os.makedirs(output_base_path, exist_ok=True)

color_cycle = itertools.cycle([
    'blue', 'red'])
marker_cycle = itertools.cycle([
    'o', 's'])

# === PLOTTING FUNCTION ===
def plot_kernels(kernel_type, kernel_src_path, regularization):
    kernels = kernel_types[kernel_type]
    
    
    data = {}
    color_map = {}
    marker_map = {}

    # Load data
    for kernel in kernels:
        file_path = os.path.join(kernel_src_path, kernel, "results.csv")
        if os.path.exists(file_path):
            data[kernel] = pd.read_csv(file_path)
            color_map[kernel] = next(color_cycle)
            marker_map[kernel] = next(marker_cycle)
        else:
            print(f"Warning: File not found for kernel '{kernel}': {file_path}")

    # Plot
    if regularization==False: # When we don't apply regularization, the model shows poor performance. To make the plot clearer, the axis are broken.
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.03},figsize=(10, 6))
        
        for kernel, df in data.items():
            color = color_map[kernel]
            marker = marker_map[kernel]
            
            # Visualize the performance of the truncated KRR on test samples
            ax1.plot(df["Percentage of Eigenvalues"], df["R2 Test"], 
                     label=f"{kernel.capitalize()} Test", color=color, marker=marker,
                     linestyle='-', markersize=5, linewidth=1.5)
            ax2.plot(df["Percentage of Eigenvalues"], df["R2 Test"], 
                     label=f"{kernel.capitalize()} Test", color=color, marker=marker,
                     linestyle='-', markersize=5, linewidth=1.5)
            
            # Visualize the performance of the truncated KRR on training samples
            ax1.plot(df["Percentage of Eigenvalues"], df["R2 Train"], 
                     label=f"{kernel.capitalize()} Train", color=color, marker=marker,
                     linestyle='--', markersize=5, linewidth=1.5)
            ax2.plot(df["Percentage of Eigenvalues"], df["R2 Train"], 
                     label=f"{kernel.capitalize()} Train", color=color, marker=marker,
                     linestyle='--', markersize=5, linewidth=1.5)

        
        ax1.set_ylim(0.2, 1.1)  
        ax2.set_ylim(-9, -1)
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        d = 0.005
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax2.set_xlabel("Percentage of Eigenvalues", fontsize=14)
        ax1.set_ylabel(r"$R^2$", fontsize=14)
        ax1.legend(fontsize=8)
        
    else: # When we apply regularization, the model shows good performance. The plot is clearer without broken axis.
        plt.figure(figsize=(10, 6))
        
        for kernel, df in data.items():
            color = color_map[kernel]
            marker = marker_map[kernel]
            
            # Visualize the performance of the truncated KRR on test samples
            plt.plot(df["Percentage of Eigenvalues"], df["R2 Test"], 
                     label=f"{kernel.capitalize()} Test", color=color, marker=marker,
                     linestyle='-', markersize=5, linewidth=1.5)
            # Visualize the performance of the truncated KRR on training samples
            plt.plot(df["Percentage of Eigenvalues"], df["R2 Train"], 
                     label=f"{kernel.capitalize()} Train", color=color, marker=marker,
                     linestyle='--', markersize=5, linewidth=1.5)

        plt.xlabel("Percentage of Eigenvalues", fontsize=14)
        plt.ylabel(r"$R^2$", fontsize=14)
        plt.legend(fontsize=9)
        plt.tight_layout()

    # Save
    reg_tag = "reg" if regularization else "noreg"
    output_file = os.path.join(output_base_path, f"r2_{kernel_type}_{reg_tag}.png")
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"Plot saved: {output_file}")



if __name__ == "__main__":
    # Example 1: fingerprint-based kernels with no regularization
    path_results = "./Results/" #location to your results folder
    # Make sure to have tanimoto and dice kernels in the Results folder
    plot_kernels(kernel_type="fingerprint_based", kernel_src_path=path_results, regularization=False)
    # Make sure to have laplacian and gaussian kernels in the Results folder
    plot_kernels(kernel_type="distance_based", kernel_src_path=path_results, regularization=False)