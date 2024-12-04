import pandas as pd
from matplotlib import pyplot as plt


class Plotter:

    @staticmethod
    def plot_correlations(data: pd.DataFrame, y_column: str, x_columns: list):
        plt.figure(figsize=(8, 6))
        for x_column in x_columns:
            plt.scatter(data[x_column], data[y_column], alpha=0.7, edgecolors='k', label=x_column)

        plt.title(f'Correlation between {", ".join(x_columns)} and {y_column}')
        plt.xlabel("X - Features")
        plt.ylabel(y_column)
        plt.legend(title="X Features")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    @staticmethod
    def plot_hist(data: pd.DataFrame):
        data.hist(bins=20, figsize=(10, 8))
        plt.suptitle('Histograms of Features')
        plt.show()
