import numpy as np

def gauss(p, x):
    """
    Produce a Gaussian function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the Gaussian function.
    x : array_like
        The x range over which the Gaussian function is to be produced.

    Returns
    -------
    numpy.ndarray
        The Gaussian function.
    """
    return p[0] * np.exp(-(x - p[1])**2 / (2 * p[2]**2)) + p[3]

def lorentz(p, x):
    """
    Produce a Lorentzian function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the Lorentzian function.
    x : array_like
        The x range over which the Lorentzian function is to be produced.

    Returns
    -------
    numpy.ndarray
        The Lorentzian function.
    """
    return p[0] * p[2]**2 / ((x - p[1])**2 + p[2]**2) + p[3]

def sin(p, x):
    """
    Produce a sin function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the sin function.
    x : array_like
        The x range over which the sin function is to be produced.

    Returns
    -------
    numpy.ndarray
        The sin function.
    """
    return p[0]*np.sin(p[1]*x+p[2]) + p[3]

def cos_superpos(p, x):
    """
    Produce a function that is the sum of two cos functions.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the function.
    x : array_like
        The x range over which the function is to be produced.

    Returns
    -------
    numpy.ndarray
        The function.
    """
    return p[0]*(np.cos(p[1]*x+p[2]) + np.cos(p[3]*x+p[4])) + p[5]

def cos_prod(p, x):
    """
    Produce a function that is the product of two cos functions.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the function.
    x : array_like
        The x range over which the function is to be produced.

    Returns
    -------
    numpy.ndarray
        The function.
    """
    return p[0]*np.cos(p[1]*x+p[2]) * np.cos(p[3]*x+p[4]) + p[5]