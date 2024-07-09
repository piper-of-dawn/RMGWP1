import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

COLORS= [
        "#5E81AC",
        "#BF616A",
        "#B48EAD",
        "#EBCB8B",
        "#B48EAD",
        "#C72C41",
        "#EE4540",
        "#E3E3E3",
    ]

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



def plot_multiline_chart(
    data,
    title="Smooth Multiline Chart",
    x_label="X-axis",
    y_label="Y-axis",
    y_scale="linear",
):
    # Define the color scheme with shades of red and black
    colors = COLORS
    # Check if the number of data sets is within the supported range
    # Create a plot with a different color for each data set
    plt.figure(figsize=(10, 7))
    for i, (x_axis, y_axis, label) in enumerate(data):
        plt.plot(x_axis, y_axis, label=label, color=colors[i%len(colors)])
    plt.title(title, fontsize=28, fontweight="bold", loc="left", color="#4C566A", pad=20)
    # Add labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add mildly visible dotted grid lines in x direction
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.yscale("linear")
    if y_scale == "log":
        plt.yscale("log")
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    # Remove spines from x and y axis
    sns.despine(left=True, bottom=True)

    # Add legend
    plt.legend()

# Example usage:
# multivariate_density(x=[1, 2, 3, 4, 5], y=[5, 4, 



def plot_acf_pacf_side_by_side(data, lags=None, padding=0.1, title='Autocorrelation and Partial Autocorrelation Functions'):
    # Set a neutral color palette and Arial font
    sns.set_palette("gray")
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # ACF plot
    plot_acf(data, lags=lags, ax=ax[0], color='gray')
    ax[0].set_title('Autocorrelation Function (ACF)')

    # PACF plot
    plot_pacf(data, lags=lags, ax=ax[1], color='gray')
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    
    # Remove spines from both x and y axes
    sns.despine(ax=ax[0], left=True, right=True, top=True, bottom=True)
    sns.despine(ax=ax[1], left=True, right=True, top=True, bottom=True)

    # Adjust layout with padding
    plt.subplots_adjust(wspace=padding)

    # Add footnote
    fig.text(0.5, -0.05, adf_test(data, 0.05), ha='center', fontsize=10, color='gray')
    plt.suptitle(title, y=1.02, fontsize=14, color='black',fontweight= 'bold')
    # Show the plots
    return fig

def adf_test(series, alpha=0.05):
    """
    Perform Augmented Dickey-Fuller (ADF) test for stationarity.

    Parameters:
    - series (pd.Series): Time series data.
    - alpha (float): Significance level for the test.

    Returns:
    - ADF test results and conclusion.
    """
    result = adfuller(series, autolag='AIC')
    p_value = result[1]

    if p_value <= alpha:
        return f"Result: Reject the null hypothesis at {alpha} significance level (p-value: {round(p_value,2)}). The time series is likely stationary."
    else:
        return f"Result: Fail to reject the null hypothesis at {alpha} significance level (p-value: {round(p_value,2)}). The time series is likely non-stationary."