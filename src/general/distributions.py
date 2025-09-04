import numpy as np
import scipy.stats as spstats

def distShortName(distName: str) -> str:
    """ Distribution short name.

    Parameters
    ----------
    noiseDist : str
        Long name of the distribution, i.e. "complex gaussian"

    Returns
    -------
    str
        Short name of the distribution, i.e. "CN" for "complex gaussian"

    Raises
    ------
    ValueError
        If the selection distribution does not have a corresponding short name
    """
    
    match distName:
        case "complex gaussian":
            return "CN"
        case "complex weibull":
            return "CW"
        case "complex student_t":
            return "CS"
        case _:
            raise ValueError(F"noise distribution {distName} invalid")

def samples(distType: str,
            nSamples: int,
            mean: float = 0.0,
            sigma: float = 1.0,
            ) -> np.ndarray:
    """ Creates samples of various distributions

    Parameters
    ----------
    distType : str
        Distribution type. Options are 'gaussian', 'complex gaussian', 'complex weibull' and 'complex student_t'.
    nSamples : int
        Number of data samples to create.
    mean : float, optional
        Mean of distribution, by default 0.0.
    sigma : float, optional
        Standard deviation of distribution, by default 1.0.

    Returns
    -------
    np.ndarray
        Samples of distribution.

    Raises
    ------
    ValueError
        If specified distribution is not in the list.
    """
    
    paramType = "samples"
    
    match distType:
        case "gaussian":
            samples = gaussian(paramType = paramType,
                               mean = mean,
                               sigma = sigma,
                               nSamples = nSamples)
        case "complex gaussian":
            samples = complexGaussian(paramType = paramType,
                                      mean = mean,
                                      sigma = sigma,
                                      nSamples = nSamples)
        case "complex weibull":
            samples = complexWeibull(paramType = paramType,
                                     mean = mean,
                                     sigma = sigma,
                                     nSamples = nSamples)
        case "complex student_t":
            samples = complexStudentT(paramType = paramType,
                                      mean = mean,
                                      sigma = sigma,
                                      nSamples = nSamples)
        case _:
            raise ValueError(f"{distType} is an invalid distribution")
        
    return samples

def probability(distType: str,
                x: float | np.ndarray, 
                mean: float = 0.0, 
                sigma: float = 1.0
                ) -> np.ndarray:
    """ Determines probabilities of values associated with various distributions.

    Parameters
    ----------
    distType : str
        Distribution type. Options are 'gaussian', 'complex gaussian', 'complex weibull' and 'complex student_t'.
    x : float | np.ndarray
        Values associated with given distribution. These can be complex for 'complex distributions'.
    mean : float, optional
        Mean of distribution, by default 0.0.
    sigma : float, optional
        Standard deviation of distribution, by default 1.0.

    Returns
    -------
    np.ndarray
        Probability values of distribution.

    Raises
    ------
    ValueError
        If specified distribution is not in the list.
    """
    
    paramType = "probability"
    
    match distType:
        case "gaussian":
            f = gaussian(paramType = paramType,
                         mean = mean,
                         sigma = sigma,
                         x = x)
        case "complex gaussian":
            f = complexGaussian(paramType = paramType,
                                mean = mean,
                                sigma = sigma,
                                z = x)
        case "complex weibull":
            f = complexWeibull(paramType = paramType,
                               mean = mean,
                               sigma = sigma,
                               z = x)
        case "complex student_t":
            f = complexStudentT(paramType = paramType,
                                mean = mean,
                                sigma = sigma,
                                z = x)
        case _:
            raise ValueError(f"{distType} is an invalid distribution")
        
    return f

def __checkNSamples(nSamples: int | None):
    """ Check nSamples input parameter

    Parameters
    ----------
    nSamples : int | None
        Number of samples to compute, by default None.

    Raises
    ------
    ValueError
        If nSamples is None
    ValueError
        If nSamples is less than or equal to 0.
    """
    
    if nSamples is None:
        raise ValueError("nSamples must not be None")
    if nSamples <= 0:
        raise ValueError("nSamples must be greater than 0")

def gaussian(paramType: str,
             mean: float = 0.0,
             sigma: float = 1.0,
             nSamples: int | None = None,
             x: float | np.ndarray | None = None
             ) -> np.ndarray:
    """ Gaussian Distribution

    Parameters
    ----------
    paramType : str
        Name of parameter to calculate
    mean : float, optional
        Mean value, by default 0.0
    sigma : float, optional
        Standard deviation, by default 1.0
    nSamples : int | None, optional
        Number of samples to compute, by default None. Only valid if paramType is "samples".
    x : float | np.ndarray | None, optional
        Samples values, by default None. Only valid if paramType is "probability".

    Returns
    -------
    np.ndarray
        Either sample or probability values depending on paramType

    Raises
    ------
    ValueError
        If x is None. Only valid if paramType is "probability".
    ValueError
        If paramType if not supported.
    """
    
    match paramType:
        
        case "samples": 
            # check input values
            __checkNSamples(nSamples)
            
            # calculate samples
            out = spstats.norm.rvs(loc=mean,
                                   scale=sigma,
                                   size=nSamples)
            
        case "probability":
            # check input values
            if x is None:
                raise ValueError("'x' must not be None")
            
            # calcuate probabilities
            out = spstats.norm.pdf(x=x,
                                   loc=mean,
                                   scale=sigma)
            
        case _:
           raise ValueError(f"Unsupported type {paramType}. Must must be either 'samples' or 'probability'")
       
    return out # type: ignore

def complexGaussian(paramType: str,
                    mean: float = 0.0,
                    sigma: float = 1.0,
                    nSamples: int | None = None,
                    z: float | np.ndarray | None = None
                    ) -> np.ndarray:
    """ Complex Gaussian
    
    # TODO

    Parameters
    ----------
    paramType : str
        Name of parameter to calculate
    mean : float, optional
        Mean value, by default 0.0
    sigma : float, optional
        Standard deviation, by default 1.0
    nSamples : int | None, optional
        Number of samples to compute, by default None. Only valid if paramType is "samples".
    z : float | np.ndarray | None, optional
        Complex samples values, by default None. Only valid if paramType is "probability".

    Returns
    -------
    np.ndarray
        Either sample or probability values depending on paramType

    Raises
    ------
    ValueError
        If z is None. Only valid if paramType is "probability".
    ValueError
        If paramType if not supported.
    """
    
    match paramType:
        
        case "samples": 
            # check input values
            __checkNSamples(nSamples)
            
            # calculate the variance of the real and imaginary distributions, i.e. Z = X + iY
            sigmaXY = sigma / np.sqrt(2)
            
            # calculate samples of real distribution
            x = spstats.norm.rvs(loc=0.0, scale=1.0, size=nSamples)
            
            # calculate samples of imaginary distribution
            y = spstats.norm.rvs(loc=0.0, scale=1.0, size=nSamples)
            
            # calculate samples of complex gaussian
            out = (x + 1j*y) * sigmaXY + mean
            
        case "probability":
            # check input values
            if z is None:
                raise ValueError("'z' must not be None")
            
            # calcuate probabilities
            out = 1 / (np.pi * sigma**2) * np.exp(-np.abs(z - mean)**2 / sigma**2)
            
        case _:
           raise ValueError(f"Unsupported type {paramType}. Must must be either 'samples' or 'probability'")
    
    return out

def complexWeibull(paramType: str,
                   mean: float = 0.0,
                   sigma: float = 1.0,
                   nSamples: int | None = None,
                   z: float | np.ndarray | None = None
                   ) -> np.ndarray:
    """ Complex Weibull
    
    # TODO
    Creates sample values of a complex Gaussian distribution from samples of Gaussian distributions. In the form
    Z = X + iY. The compound Gaussian distribution is simply a complex Weibull distribution where the shape parameter
    is k = 1/2, eta = sqrt(2)/sigma and the variance, gamma = sigma^2.

    Parameters
    ----------
    paramType : str
        Name of parameter to calculate
    mean : float, optional
        Mean value, by default 0.0
    sigma : float, optional
        Standard deviation, by default 1.0
    nSamples : int | None, optional
        Number of samples to compute, by default None. Only valid if paramType is "samples".
    z : float | np.ndarray | None, optional
        Complex samples values, by default None. Only valid if paramType is "probability".

    Returns
    -------
    np.ndarray
        Either sample or probability values depending on paramType

    Raises
    ------
    ValueError
        If z is None. Only valid if paramType is "probability".
    ValueError
        If paramType if not supported.
    """
    
    match paramType:
        
        case "samples": 
            # check input values
            __checkNSamples(nSamples)
            
            # calculate the variance of the real and imaginary distributions, i.e. Z = X + iY
            sigmaXY = np.abs(spstats.norm.rvs(loc=0.0, scale=1.0, size=nSamples) * sigma) / np.sqrt(2)
            
            # calculate samples of real distribution
            x = spstats.norm.rvs(loc=0.0, scale=1.0, size=nSamples)
            
            # calculate samples of imaginary distribution
            y = spstats.norm.rvs(loc=0.0, scale=1.0, size=nSamples)
            
            # calculate samples of complex gaussian
            out = (x + 1j*y) * sigmaXY + mean
            
        case "probability":
            # check input values
            if z is None:
                raise ValueError("'z' must not be None")
            
            # calcuate probabilities
            out = 1 / (np.pi * np.sqrt(2) * sigma * np.abs(z - mean)) * np.exp(-np.abs(z - mean) * np.sqrt(2) / sigma)
            
        case _:
           raise ValueError(f"Unsupported type {paramType}. Must must be either 'samples' or 'probability'")
    
    return out

def complexStudentT(paramType: str,
                    mean: float = 0.0,
                    sigma: float = 1.0,
                    nSamples: int | None = None,
                    z: float | np.ndarray | None = None
                    ) -> np.ndarray:
    """ Complex Student T
    
    # TODO
    Creates sample values of a complex Gaussian distribution from samples of Gaussian distributions. In the form
    Z = X + iY. The compound Gaussian distribution is simply a complex Weibull distribution where the shape parameter
    is k = 1/2, eta = sqrt(2)/sigma and the variance, gamma = sigma^2.

    Parameters
    ----------
    paramType : str
        Name of parameter to calculate
    mean : float, optional
        Mean value, by default 0.0
    sigma : float, optional
        Standard deviation, by default 1.0
    nSamples : int | None, optional
        Number of samples to compute, by default None. Only valid if paramType is "samples".
    z : float | np.ndarray | None, optional
        Complex samples values, by default None. Only valid if paramType is "probability".

    Returns
    -------
    np.ndarray
        Either sample or probability values depending on paramType

    Raises
    ------
    ValueError
        If z is None. Only valid if paramType is "probability".
    ValueError
        If paramType if not supported.
    """
    
    match paramType:
        
        case "samples": 
            # check input values
            __checkNSamples(nSamples)
            
            # TODO - look at in more detail
            beta = 1
            alpha = (beta + sigma**2) / sigma**2
            
            # calculate the variance of the real and imaginary distributions, i.e. Z = X + iY
            sigmaXY = np.sqrt(spstats.invgamma.rvs(alpha, size=nSamples) / 2)
            
            # calculate samples of real distribution
            x = spstats.norm.rvs(loc=0.0, scale=1.0, size=nSamples)
            
            # calculate samples of imaginary distribution
            y = spstats.norm.rvs(loc=0.0, scale=1.0, size=nSamples)
            
            # calculate samples of complex gaussian
            out = (x + 1j*y) * sigmaXY + mean
            
        case "probability":
            # check input values
            if z is None:
                raise ValueError("'z' must not be None")
            
            # TODO - look at in more detail
            beta = 1
            alpha = None
            
            if (alpha is None) and (beta is None):
                raise ValueError("need a value for at least alpha or beta")
            
            # calculate the alpha or beta values
            if alpha is None:
                alpha = (beta + sigma**2) / sigma**2 
            else:
                beta = sigma**2 * (alpha - 1)
            
            # calcuate probabilities
            out = 1 / (np.pi * np.sqrt(2) * sigma * np.abs(z - mean)) * np.exp(-np.abs(z - mean) * np.sqrt(2) / sigma)
            
        case _:
           raise ValueError(f"Unsupported type {paramType}. Must must be either 'samples' or 'probability'")
    
    return out

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    # test the gaussian distribution
    sampleVals = gaussian(paramType = "samples", 
                          mean = 0.0, 
                          sigma = 1.0, 
                          nSamples = 1000)
    
    # parameters
    nSamples = int(1e6)
    values = np.linspace(-2.5, 2.5, int(1e3))
    values = values + 1j * values
    distributions = ["gaussian", "complex gaussian", "complex weibull", "complex student_t"]

    for dist in distributions:
        
        if dist == "gaussian":
            values = np.real(values)
        
        sampleVals = samples(dist, nSamples)
        probVals = probability(dist, values)
        
        # plotting samples
        plt.figure(figsize=[12, 8])
        plt.hist(np.real(sampleVals), bins=100, density=True)
        plt.plot(values, probVals)
        plt.grid()
        plt.title(dist)
        plt.xlabel("$x$")
        plt.ylabel("$P(x)$")
        
    plt.show()
    