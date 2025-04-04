import numpy as np
import scipy.optimize as opt

def residuals(p, func, x, y, s=1) -> np.ndarray:
    """
    Find the residuals from fitting data to a function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the function func.
    x : array_like
        The x data to be fit.
    y : array_like
        The y data to be fit.
    func : callable
        The function to be fit.
    s : float, default 1
        The standard deviation.

    Returns
    -------
    numpy.ndarray
        The residuals of the fit function.

    See Also
    --------
    residuals_data : Find the residuals between observed data and a model. 
    
    """
    return (y - func(p, x)) / s

def fitting(p, x, y, func, s=1):
    """
    Fit data to a function.
    
    Parameters
    ----------
    p : array_like
        Initial guess at the values of the coefficients to be fit
    x : array_like
        The x data to be fit.
    y : array_like
        The y data to be fit.
    func : callable
        The function to fit the data to.
    s : float, default 1
        The standard deviation.

    Returns
    -------
    p_fit : numpy.ndarray
        The array of the coefficients after fitting the function to the data.
    chi : float
        The chi squared value of this fit.
    unc_fit : numpy.ndarray
        The array of the uncertainties in the values of p_fit.
    
    See Also
    --------
    fitting_params_only : Fit data to a function and only return the parameters.
    
    """
    # Fit the data and find the uncertainties
    r = opt.least_squares(residuals, p, args=(func, x, y, s))
    p_fit = r.x
    hessian = np.dot(r.jac.T, r.jac) #estimate the hessian matrix
    K_fit = np.linalg.inv(hessian) #covariance matrix
    unc_fit = np.sqrt(np.diag(K_fit)) #stdevs
    
    # rescale
    beta = np.sqrt(np.sum(residuals(p_fit, func, x, y)**2) / (x.size - len(p_fit)))
    unc_fit = unc_fit * beta
    K_fit = K_fit * beta
    
    # find the chi2 score
    chi = np.sum(residuals(p_fit, func, x, y, beta)**2) / (x.size - len(p_fit))
    
    return p_fit, chi, unc_fit

def fitting_params_only(p, x, y, func, s=1):
    """
    Fit data to a function.
    
    Parameters
    ----------
    p : array_like
        Initial guess at the values of the coefficients to be fit
    x : array_like
        The x data to be fit.
    y : array_like
        The y data to be fit.
    func : callable
        The function to fit the data to.
    s : float, default 1
        The standard deviation.

    Returns
    -------
    p_fit : numpy.ndarray
        The array of the coefficients after fitting the function to the data.
    
    See Also
    --------
    fitting : Fit data to a function and returns the chi squared value as well as the uncertainty in the fitted parameters.
    
    """
    # Fit the data and find the uncertainties
    r = opt.least_squares(residuals, p, args=(func, x, y, s))
    p_fit = r.x
    
    return p_fit

def round_sig_fig_uncertainty(value, uncertainty):
    """
    Round to the first significant figure of the uncertainty.
    
    Parameters
    ----------
    value : float or array_like
        The value(s) to be rounded.
    uncertainty : float or array_like
        The uncertaint(y/ies) in this value, must be the same size as value.

    Returns
    -------
    value_out : numpy.ndarray or float
        The rounded array of values.
    uncertainty_out : numpy.ndarray or float
        The rounded array of uncertainties.

    See Also
    --------
    round_sig_fig : Round to a given number of significant figures.
    
    """
    # check if numpy array/list or float/int
    if isinstance(value, np.ndarray) or isinstance(value, list):
        value_out = np.array([])
        uncertainty_out = np.array([])
        for i in range(len(value)):
            # Check if some of the values are 0
            if uncertainty[i] == 0:
                value_out = np.append(value_out, value[i])
                uncertainty_out = np.append(uncertainty_out, uncertainty[i])
            # Check if the leading digit in the error is 1, and if so round to an extra significant figure
            elif np.floor(uncertainty[i] / (10**np.floor(np.log10(uncertainty[i])))) != 1.0:
                uncertainty_rnd = np.round(uncertainty[i], int(-(np.floor(np.log10(uncertainty[i])))))
                value_rnd = np.round(value[i], int(-(np.floor(np.log10(uncertainty_rnd)))))

                value_out = np.append(value_out, value_rnd)
                uncertainty_out = np.append(uncertainty_out, uncertainty_rnd)
            else:
                uncertainty_rnd = np.round(uncertainty[i], int(1 - (np.floor(np.log10(uncertainty[i])))))
                value_rnd = np.round(value[i], int(1 - (np.floor(np.log10(uncertainty_rnd)))))

                value_out = np.append(value_out, value_rnd)
                uncertainty_out = np.append(uncertainty_out, uncertainty_rnd)
        return value_out, uncertainty_out
   
    elif isinstance(value, float) or isinstance(value, int):
        if uncertainty == 0:
            return value, uncertainty
        elif np.floor(uncertainty / (10**np.floor(np.log10(uncertainty)))) != 1.0:
            uncertainty_out = np.round(uncertainty, int(-(np.floor(np.log10(uncertainty)))))
            value_out = np.round(value, int(-(np.floor(np.log10(uncertainty_out)))))

            return value_out, uncertainty_out
        else:
            uncertainty_out = np.round(uncertainty, int(1 - (np.floor(np.log10(uncertainty)))))
            value_out = np.round(value, int(1 - (np.floor(np.log10(uncertainty_out)))))

            return value_out, uncertainty_out
    else:
        return value, uncertainty

def find_nearest_index(value, array):
    """
    Find the index of the value in the array closest to the inport value.
    
    Parameters
    ----------
    value : float
        The value to search for.
    array : array_like
        The array to search.
        
    Returns
    -------
    index : int
        The index of the closest value in the array to value.
        
    """
    index = np.searchsorted(array, value, side="left")
    return index

def residuals_data(observed, expected, s=1) -> np.ndarray:
    """
    Find the residuals between observed data and a model.
    
    Parameters
    ----------
    observed : array_like
        The observed data.
    expected : array_like
        The expected values for the data from the model.
    s : array_like, default 1
        The uncertainty in the data.

    Returns
    -------
    numpy.ndarray
        The residuals of the fit function.
    
    See Also
    --------
    residuals : Find the residuals from fitting data to a function.
    
    """
    return (observed - expected) / s

def chi2(observed, expected, s=1) -> float:
    """
    Find the chi squared value for a set of data and expected values.

    Parameters
    ----------
    observed : array_like
        The observed data.
    expected : array_like
        The expected values for the data.
    s : array_like, default 1
        The uncertainty in the data.

    Returns
    -------
    float
        The chi squared value for the dataset.
        
    """
    return np.sum(residuals_data(observed, expected, s)**2 ) 

def search_store(file: np.lib.npyio.NpzFile, string: str) -> np.ndarray:
    """
    Search an npz file for all files containing a certain string and return an array of them.
    
    Parameters
    ----------
    file : numpy .NpzFile
        The npz file to be searched.
    string : str
        The string being searched for in the file names.

    Returns
    -------
    arr : numpy array
        The files containing the string.
        
    """
    lst = []
    for i in file:
        if string in i:
            lst.append(file[i])
    arr = np.asarray([lst])
    return arr[0]

def round_sig_fig(value, n: int) -> float:
    """
    Round to a given number of significant figures.
        
    Parameters
    ----------
    value : float or array_like
        The value(s) to be rounded.
    n : positive int
        The number of significant figures to round value to.
    
    Returns
    -------
    out : float
        The rounded value(s).

    See Also
    --------
    round_sig_fig_uncertainty : Round to the first significant figure of the uncertainty.
        
    """
    out = np.empty_like(value)
    # Check if float/int or array/list
    if isinstance(value, float) or isinstance(value, int):
        # Turn the value into a string in scientific notation
        val_str = np.format_float_scientific(value, n)

        # Split this into the base and exponent of 10, drop the base
        expo = int(val_str.split('e')[1])

        # Round the value to this number of decimal places (minus means it goes to the left of the decimal point, also add 1 because numpy is weird in the way it rounds for some reason, it tends to round to n+1 sig fig in format_float_scientific, and in round (at least when going negative for round). Not sure why, but this corrects for it)
        out = np.round(value, -(expo - n + 1))
        return out
    elif isinstance(value, np.ndarray) or isinstance(value, list):
        value = np.asarray(value)
        if len(value.shape) != 1:
            for i in range(len(value)):
                out[i] = round_sig_fig(value[i], n)
        else:
            out_local = np.empty_like(value)
            for i in range(len(value)):
                # Turn the value into a string in scientific notation
                val_str = np.format_float_scientific(value[i], n)

                # Split this into the base and exponent of 10, drop the base
                expo = int(val_str.split('e')[1])

                # Round the value to this number of decimal places
                out_val = np.round(value[i], -(expo - n + 1))
                out_local[i] = out_val
            return out_local
        return out
