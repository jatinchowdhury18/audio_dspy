import numpy as np

def tf2minphase (h):
    """Converts a transfer function to minimum phase

    Parameters
    ----------
    h : ndarray
        Numpy array containing the original transfer function

    Returns
    -------
    h_min : ndarray
        Numpy array containing the minimum phase transfer function
    """
    N = len (h)
    H = np.fft.fft (h)
    log_H = np.log (H)
    h_c = np.fft.ifft (log_H)
    
    half_N = int(N/2)
    h_c_hat = np.zeros (half_N, dtype=np.complex128)
    h_c_hat[0] = h_c[0]
    for n in np.arange (1, half_N-1):
        h_c_hat[n] = h_c[n] + h_c[N-n]
    
    h = np.real (np.fft.ifft (np.exp (np.fft.fft (h_c_hat))))
    return h

def tf2linphase (h):
    """Converts a transfer function to linear phase

    Parameters
    ----------
    h : ndarray
        Numpy array containing the original transfer function

    Returns
    -------
    h_min : ndarray
        Numpy array containing the linear phase transfer function
    """
    N = len (h)
    H = np.fft.fft (h)
    w = np.linspace (-np.pi, np.pi, N)
    delay_kernels = np.exp (-1j*(N/2)*w)
    h = np.real (np.fft.ifft (delay_kernels * H))
    return h / max (abs (h))
