import os

import numpy as np
import scipy.optimize as sco
import scipy.special as scp
import scipy.integrate as sci

import pandas as pd


HUGE = 1e16
SMALL = 1e-16

##############################################
# Functions for integration of sqrt of trigo #
##############################################

def integ_sqrt_cos(x):
    """ Returns the integral of sqrt(cos) on [0, x] using incomplete beta functions
    \int_0^x \sqrt{cos t} dt = 0.5 * \int_0^{(sinx)^2} u^(-0.5)*(1-u)^(-0.25)du"""
    a, b = 0.5, 0.75
    beta_ratio = scp.betainc(a, b, np.sin(x)**2)
    return 0.5 * beta_ratio * scp.beta(a, b)
    

def integ_sqrt_sin(x):
    """ Return the integral of sqrt(sin) on [0, x] using incomplete beta function 
    \int_0^x \sqrt{sin t} dt = 0.5 * beta(0.5, 0.75) - 0.5 * \int_0^{(cos x)^2} u^(-0.5)*(1-u)^(-0.25)du"""
    a, b = 0.5, 0.75
    beta_ratio = 1 - scp.betainc(a, b, np.cos(x)**2)
    return 0.5 * beta_ratio * scp.beta(a, b)

def integ_inv_sqrt_cos(x):
    """ Returns the integral of 1/sqrt(cos) on [0, x] using incomplete beta functions
    \int_0^x 1/\sqrt{cos t} dt = 0.5 * \int_0^{(sinx)^2} u^(-0.5)*(1-u)^(-0.75)du"""
    a, b = 0.5, 0.25
    beta_ratio = scp.betainc(a, b, np.sin(x)**2)
    return 0.5 * beta_ratio * scp.beta(a, b)

def integ_inv_sqrt_sin(x):
    """ Return the integral of 1/sqrt(sin) on [0, x] using incomplete beta function 
    \int_0^x 1/\sqrt{sin t} dt = 0.5 * beta(0.5, 0.25) - 0.5 * \int_0^{(cos x)^2} u^(-0.5)*(1-u)^(-0.75)du"""
    a, b = 0.5, 0.25
    beta_ratio = 1 - scp.betainc(a, b, np.cos(x)**2)
    return 0.5 * beta_ratio * scp.beta(a, b)

def reduced_integ_sqrt_sin(x):
    """ Return 1/sqrt(sin(x)) times the integral of sqrt(sin) on [0, x] using incomplete beta function 
    which is equivalent to 2x/3 at x=0"""

    out_forSmall = 2 * x / 3.
    out_forLarge = integ_sqrt_sin(x) / np.sqrt(np.abs(np.sin(x)))
    
    return np.where(x < 1e-3, out_forSmall, out_forLarge)


#####################################################################
# Functions for solving the unknown (alpha1, alpha2, force, l2, l3) #
#####################################################################

def check_angles(alpha1, alpha2):
    """ Check that 0 <= alpha2 <= alpha1 <= pi/2 """
    isTooSmall = ((alpha1 < 0) | (alpha2 <0))
    isTooLarge = ((alpha1 > np.pi/2) | (alpha2 > np.pi/2))
    isWrongOrder = (alpha1 < alpha2)
    
    return ~(isTooLarge | isTooSmall | isWrongOrder)

def func_root_all(alpha1, alpha2, W, H, a):
    """ Define equations to solve to get (alpha1, alpha2) from (W, H, a)"""
    # Check if angles corrects
    angles_correct = check_angles(alpha1, alpha2)

    # Compute primitives sqrt(cos), sqrt(sin)
    P_a1_a2_red = reduced_integ_sqrt_sin(alpha1 - alpha2) # 1/sqrt(sin x) * int_0^x sqrt(sin u)du
    M_a2 = integ_sqrt_cos(alpha2)                         # int_0^x sqrt(cos u)du

    # Writes shortcuts for angles
    sin1, cos1 = np.sin(alpha1), np.cos(alpha1)
    sin12 = np.sin(alpha1 - alpha2)
    cos2 = np.cos(alpha2)

    # Compute functions to find roots of (see Overleaf)
    eq1 = 2 * (W*sin1 - H*cos1) - (W*cos1 + H*sin1) * P_a1_a2_red
    eq2 = 4 * sin12**2 * a**2 - (W*cos1 + H*sin1)**2 * cos2 * M_a2**2 

    equations = np.array([eq1, eq2])
    equations[:, ~angles_correct] = HUGE

    return equations

def force_from_angles(alpha1, alpha2, W, H, a=None):
    """ Compute the force from (alpha1, alpha2, W, H)"""
    assert np.all(check_angles(alpha1, alpha2)), f'Solution is not between bounds: {alpha1, alpha2}'
    
    force = 2 * np.sin(alpha1 - alpha2) / (W * np.cos(alpha1) + H * np.sin(alpha1))**2
    return force

def l2_l3_force_from_angles_force(alpha1, alpha2, force):
    """ Compute the lengths (l2,l3) of the regions knowing (alpha1, alpha2, force) """
    
    l2 = 1. / np.sqrt(2 * force) * integ_inv_sqrt_sin(alpha1 - alpha2)
    l3 = 1. / np.sqrt(2 * force * np.sin(alpha1 - alpha2) / np.cos(alpha2)) * integ_inv_sqrt_cos(alpha2)

    return l2, l3

def curvature_region2(alpha, alpha1, force):
    """ Returns the absolute curvature between cylinders, knowing angles and force"""
    return np.sqrt(2 * force * np.sin(alpha1 - alpha))

def curvature_region3(alpha, alpha1, alpha2, force):
    """ Returns the absolute curvature in central region, knowing angles and force"""
    return np.sqrt(2 * force * np.sin(alpha1-alpha2)/np.cos(alpha2) * np.cos(alpha))
    
def solve_alpha_curv_from_all_param(l2, l3, alpha1, alpha2, force, nbPoints=50):
    """ Solves the first integral of mouvement, starting from S=L (middle)
    nbPoints is the number of outputed points on each segment
    Return S, alpha, kappa
    """
    # Define functions to integrate y' = f(t,y) for each region
    def reg2(sInv, alpha):
        return curvature_region2(alpha, alpha1, force)
    def reg3(sInv, alpha):
        return curvature_region3(alpha, alpha1, alpha2, force)

    # Solve pb in region 3
    sInv_span_3 = [0, l3]
    sInv_eval_3 = np.linspace(*sInv_span_3, nbPoints, endpoint=False)
    alpha0_3 = np.array([0])

    sol_3 = sci.solve_ivp(reg3, sInv_span_3, alpha0_3, t_eval=sInv_eval_3)
    alpha_reg3 = sol_3.y.flatten()
    curv_reg3 = curvature_region3(alpha_reg3, alpha1, alpha2, force)

    # Solve pb in region 2
    sInv_span_2 = [l3, l3+l2]
    sInv_eval_2 = np.linspace(*sInv_span_2, nbPoints)
    alpha0_2 = np.array([alpha2])

    sol_2 = sci.solve_ivp(reg2, sInv_span_2, alpha0_2, t_eval=sInv_eval_2)
    alpha_reg2 = sol_2.y.flatten()
    curv_reg2 = curvature_region2(alpha_reg2, alpha1, force)

    # Concatenate solutions
    sInv = np.concatenate((sol_3.t, sol_2.t))
    alpha = np.concatenate((alpha_reg3, alpha_reg2))
    kappa = np.concatenate((curv_reg3, curv_reg2))

    # Convert back the curvilinear coordinate and reverse the angle array
    s = l2 + l3 - sInv[::-1]
    alpha = alpha[::-1]
    kappa = kappa[::-1]

    return s, alpha, kappa

def xy_from_alpha_s(s, alpha):
    """ Integrate the positions, knowing a sample of the map S->alpha(S)
    Uses the rectangle rule (left Riemann sum), and origin at the left most cylinder"""

    # Using rectangle rule
    ds = np.diff(s)
    x = [0]
    y = [0]

    for i in range(len(s)-1):
        x.append(x[-1] + ds[i] * np.cos(alpha[i]))
        y.append(y[-1] + ds[i] * np.sin(alpha[i]))
    
    return np.array(x), np.array(y)

def solve_flexion(W, H, a, nbPoints=50, guess_alpha=[np.pi/4, np.pi/8]):
    """ Solve the full problem from scratch
    Returns :
        df: pd.DataFrame containing (S, alpha, kappa, x, y, strain_visu)
        s1, s2, s3: floats with abscissa of cylinders and middle
    """


    # Equations to solve
    func_root = (lambda alpha12: func_root_all(alpha12[0], alpha12[1], W, H, a))

    # Compute solution
    solution = sco.root(func_root, guess_alpha)
    alpha1, alpha2 = solution.x

    # Compute other unknown
    force = force_from_angles(alpha1, alpha2, W, H)
    l2, l3 = l2_l3_force_from_angles_force(alpha1, alpha2, force)

    # Integrate solution (S, alpha, kappa) and compute (x, y) and visual strain
    S, alpha, kappa = solve_alpha_curv_from_all_param(l2, l3, alpha1, alpha2, force)
    x, y = xy_from_alpha_s(S, alpha)
    strain_visu = np.log(np.cos(alpha))

    # Change (x,y) origin to be at moving cylinder (fixed in referential camera)
    x = x - a
    y = y - H

    # Create DataFrame
    data = {"S": S, "alpha": alpha, "kappa": kappa, 
            "x": x, "y": y, "strain_visu": strain_visu}

    df = pd.DataFrame(data)

    return df, 0., l2, l2+l3

def solve_curvature(W, H, a, guess_alpha=[np.pi/4, np.pi/8]):
    """ Returns minimal and maximal curvatures in central region
    """
    # Equations to solve
    func_root = (lambda alpha12: func_root_all(alpha12[0], alpha12[1], W, H, a))

    # Compute solution
    solution = sco.root(func_root, guess_alpha)
    alpha1, alpha2 = solution.x

    # Compute force and curvature
    force = force_from_angles(alpha1, alpha2, W, H)
    kappa_min, kappa_max = curvature_region3(np.array([alpha2, 0.]), alpha1, alpha2, force)
    
    return kappa_min, kappa_max
