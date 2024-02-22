import scipy.fft as fft
import numpy as np
import scipy

def dct_2d(arr):
    return fft.dct(fft.dct(arr.T, norm='ortho').T, norm='ortho')

def idct_2d(arr):
    return fft.idct(fft.idct(arr.T, norm='ortho').T, norm='ortho')

def create_Lambda_2D(n1, n2):
    # Lambda[i,j] = (-2 + 2 * np.cos(i/n1 * np.pi)) + (-2 + 2 * np.cos(j/n2 * np.pi))
    i_matr, j_matr = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')
    return (-2 + 2 * np.cos(i_matr/n1 * np.pi)) + (-2 + 2 * np.cos(j_matr/n2 * np.pi))

def create_Gamma_2D(s, n1, n2):
    ones = np.ones((n1,n2))
    Lambda = create_Lambda_2D(n1, n2)
    return ones / (ones + s * Lambda**2)

def smooth_with_s(y, s, dct_y=None):
    if dct_y is None:
        dct_y = dct_2d(y)

    nx, ny = y.shape
    gamma = create_Gamma_2D(s, nx, ny)
    yhat = idct_2d(gamma * dct_y)
    return yhat

def GCS_2D(y, s, dct_y=None):
    if dct_y is None:
        dct_y = dct_2d(y)

    n1, n2 = y.shape
    n = n1 * n2
    Gamma = create_Gamma_2D(s, n1, n2)
    RSS = (Gamma - np.ones_like(Gamma)) * dct_y
    norm_RSS = np.linalg.norm(RSS, ord='fro')
    gamma_gap = n - np.sum(np.abs(Gamma))
    return n * (norm_RSS / gamma_gap)**2

def best_smooth(y, verbose=False):
    dct_y = dct_2d(y)
    GCS_p = (lambda p: GCS_2D(y, 10**p, dct_y))
    bounds_p = [-12., 30.]

    # Coarse search
    nbPoints = 100
    p_tries = np.linspace(*bounds_p, nbPoints)
    GCS_arr = [GCS_p(p) for p in p_tries]
    argmin = np.argmin(GCS_arr)

    if argmin == 0:
        if verbose: print('No smoothing needed')
        return y
    if argmin == nbPoints-1:
        if verbose: print('Problem with data -> requires full smoothing')
        return y
    
    # Optimize around minimum coarse search
    p_bracket = p_tries[argmin-2:argmin+3:2]
    GCS_bracket = GCS_arr[argmin-2:argmin+3:2]
    print(p_bracket, GCS_bracket)

    result = scipy.optimize.minimize_scalar(GCS_p, p_bracket, bounds=bounds_p)
    s_opt = 10**result.x

    if verbose: print(f'Optimal s is {s_opt}')
    return smooth_with_s(y, s_opt)