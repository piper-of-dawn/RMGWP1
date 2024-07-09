import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

def multivariate_density(x, y, x_label="Variable 1", y_label="Variable 2", title="Title"):
    plt.rcParams['font.family'] = 'monospace'
    # Create the jointplot
    g = sns.jointplot(x=x, y=y, kind='kde', fill=True, height=7, cmap="Reds", 
                      color="red", marginal_kws=dict(fill=True))
    
    # Access the underlying Axes objects
    ax_joint = g.ax_joint
    ax_marg_x = g.ax_marg_x
    ax_marg_y = g.ax_marg_y
    
    # Set minor locators
    ax_joint.xaxis.set_minor_locator(AutoMinorLocator())
    ax_joint.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Remove spines
    sns.despine(ax=ax_joint, left=True, bottom=True)
    sns.despine(ax=ax_marg_x, left=True, bottom=True)
    sns.despine(ax=ax_marg_y, left=True, bottom=True)
    
    # Set labels
    ax_joint.set_xlabel(x_label)
    ax_joint.set_ylabel(y_label)
    
    # Adjust the layout to create space for the title at the bottom
    plt.subplots_adjust(bottom=0.2)
    
    # Add title at the bottom
    plt.figtext(0.5, 0.04, title, ha='center', fontsize=16, fontweight="bold", color="#4C566A")
    plt.figtext(0.5, 0.0005, f"Correlation: {np.corrcoef(x, y)[0, 1]:.2f}", fontweight="bold", ha='center', fontsize=9, color="#4C566A")

    # Show the plot
    plt.show()
    return g

# Example usage:
# multivariate_density(x=[1, 2, 3, 4, 5], y=[5, 4, 
