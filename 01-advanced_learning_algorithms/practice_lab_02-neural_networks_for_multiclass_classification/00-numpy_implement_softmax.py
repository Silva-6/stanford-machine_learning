from pkgs import *
# UNQ_C1


def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    N = len(z)
    a = np.zeros(N)
    ez_sum = 0
    for k in range(N):
        ez_sum += np.exp(z[k])
    for j in range(N):
        a[j] = np.exp(z[j]) / ez_sum

    return a

# Or, a vector implementation:

# def my_softmax(z):
#    ez = np.exp(z)
#    a = ez / np.sum(ez)
#    return (a)
