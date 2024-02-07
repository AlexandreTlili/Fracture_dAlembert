import numpy as np
import matplotlib.pyplot as plt

def compute_time_rising(Hmax=50e-3, vMax=1e-3, aMax=1e-3):
    """ Compute the duration of the rinsing phase
    """
    # Compute durations phases
    dt1 = vMax / aMax                   # Duration pure acceleration
    dt2 = Hmax / vMax - vMax / aMax     # Duration linear phase
    dt3 = dt1                           # Duration pure deceleration until Hmax

    return dt1 + dt2 + dt3

def time_to_H_float(time, Hmax=50e-3, vMax=1e-3, aMax=1e-3):
    """ Compute the height H associated with a given time, 
        knowing vMax and aMax."""

    # Compute durations phases
    dt1 = vMax / aMax                   # Duration pure acceleration
    dt2 = Hmax / vMax - vMax / aMax     # Duration linear phase
    dt3 = dt1                           # Duration pure deceleration until Hmax

    half_phase = dt1 + dt2 + dt3
    full_phase = 2 * half_phase

    time = time % full_phase

    if time < dt1:
        return 0.5 * aMax * time**2
    
    if time < dt1 + dt2:
        return 0.5 * aMax * dt1**2 + (time - dt1) * vMax
    
    half_phase = dt1 + dt2 + dt3

    if time < dt1 + dt2 + dt3:
        return Hmax - 0.5 * aMax * (half_phase- time)**2
    
    return Hmax - time_to_H_float(time - half_phase, Hmax, vMax, aMax)

time_to_H = np.vectorize(time_to_H_float)

def frames_to_H(frame_numbers, dt, Hmax=50e-3, vMax=1e-3, aMax=1e-3):
    time = dt * np.array(frame_numbers)
    H = time_to_H(time, Hmax, vMax, aMax)
    return H

def H_to_time_float(H, Hmax=50e-3, vMax=1e-3, aMax=1e-3):
    assert (0 <= H) and (H <= Hmax)

    # Compute durations phases
    dt1 = vMax / aMax                   # Duration pure acceleration
    dt2 = Hmax / vMax - vMax / aMax     # Duration linear phase
    dt3 = dt1                           # Duration pure deceleration until Hmax

    half_phase = dt1 + dt2 + dt3

    # Compute height limits phases
    H1 = 0.5 * vMax**2 / aMax
    H2 = Hmax - H1

    if H < H1:
        return np.sqrt(2 * H / aMax)
    
    if H < H2:
        return dt1 + (H - H1) / vMax
    
    return half_phase - np.sqrt(2 * (Hmax - H) / aMax)

H_to_time = np.vectorize(H_to_time_float)
    

def plot_full_phase(Hmax=50e-3, vMax=1e-3, aMax=1e-3, nbPhases=1, nbPoints=100):
    # Compute durations phases
    dt1 = vMax / aMax                   # Duration pure acceleration
    dt2 = Hmax / vMax - vMax / aMax     # Duration linear phase
    dt3 = dt1                           # Duration pure deceleration until Hmax
    full_phase = 2 * (dt1 + dt2 + dt3)

    time = np.linspace(0, nbPhases * full_phase, nbPoints)
    H = time_to_H(time)

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(time, H)
    plt.show()