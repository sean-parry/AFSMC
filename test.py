# for testing unused atm
from SMC import SMC
import Prob_Utils
import numpy as np
import Proposals 
import Initial_Proposals
import matplotlib.pyplot as plt

"""
for the example case lets imagine 2 static targets so we return normal prob
[2,3] np.eye(2)*0.5 as covar
"""
def target_function(sample):
    """
    takes a vector sample returns a probability
    """
    return Prob_Utils.multivariate_normal_p(sample, [2, 1], np.eye(2)*0.5)


def main():
    x1 = np.linspace(0, 5, 200)  # Values for the first dimension
    x2 = np.linspace(0, 5, 200)  # Values for the second dimension
    X1, X2 = np.meshgrid(x1, x2)  # Create a grid from X1 and X2

    # Evaluate the target function on the grid
    probabilities = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            sample = [X1[i, j], X2[i, j]]
            probabilities[i, j] = target_function(sample)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, probabilities, levels=50, cmap="viridis")
    plt.colorbar(label="Probability Density")
    plt.title("Probability Heatmap of the Target Function")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()

if __name__ == '__main__':
    main()