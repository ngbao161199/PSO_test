import numpy as np

def rastrigin(x, dem):
    """
    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`
    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """

    d = x.shape[1]
    j = 10.0 * d + (x ** 2.0 - 10.0 * np.cos(2.0 * np.pi * x)).sum(axis=1)
    dem+=1
    return j, dem

def beale(x, dem):
    """
    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`
    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (
        (1.5 - x_ + x_ * y_) ** 2.0
        + (2.25 - x_ + x_ * y_ ** 2.0) ** 2.0
        + (2.625 - x_ + x_ * y_ ** 3.0) ** 2.0
    )
    dem+=1
    return j, dem

def himmelblau(x, dem):
    """
    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`
    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = (x_ ** 2 + y_ - 11) ** 2 + (x_ + y_ ** 2 - 7) ** 2
    dem +=1
    return j, dem

def crossintray(x, dem):
    """
    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`
    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    x_ = x[:, 0]
    y_ = x[:, 1]

    j = -0.0001 * np.power(
        np.abs(
            np.sin(x_)
            * np.sin(y_)
            * np.exp(np.abs(100 - (np.sqrt(x_ ** 2 + y_ ** 2) / np.pi)))
        )
        + 1,
        0.1,
    )
    dem +=1
    return j, dem
