from pkgs import *

# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    # You need to return this variable correctly
    total_cost = 0

    ### START CODE HERE ###
    cost_sum = 0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost_sum = cost_sum + (f_wb_i - y[i]) ** 2
    total_cost = cost_sum / (2 * m)
    ### END CODE HERE ###

    return total_cost