import numpy as np

def euler(f, y0, t0, tf, dt):
    """
    Simple explicit Euler solver.
    
    Parameters
    ----------
    f  : callable
         Right‑hand side function f(t, y), returns dy/dt (array‑like).
    y0 : array_like
         Initial state at t0.
    t0 : float
         Initial time.
    tf : float
         Final time.
    dt : float
         Time step.
    
    Returns
    -------
    t : np.ndarray
        Array of time points.
    y : np.ndarray
        Array of solution states, shape (len(t), len(y0)).
    """
    t = np.arange(t0, tf + dt, dt)
    y = np.zeros((len(t),) + np.shape(y0))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
    return t, y

def rk4(f, y0, t0, tf, dt):
    """
    Classic 4th‑order Runge–Kutta solver.
    
    Same signature as euler().
    """
    t = np.arange(t0, tf + dt, dt)
    y = np.zeros((len(t),) + np.shape(y0))
    y[0] = y0
    for i in range(1, len(t)):
        ti = t[i-1]
        yi = y[i-1]
        k1 = f(ti, yi)
        k2 = f(ti + dt/2, yi + dt/2 * k1)
        k3 = f(ti + dt/2, yi + dt/2 * k2)
        k4 = f(ti + dt,   yi + dt   * k3)
        y[i] = yi + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

# Example usage:
if __name__ == "__main__":
    # Solve dy/dt = -y with y(0)=1; exact solution y=exp(-t)
    def f(t, y):
        return -y

    y0 = 1.0
    t0, tf, dt = 0.0, 5.0, 0.01

    t_eu, y_eu = euler(f, y0, t0, tf, dt)
    t_rk, y_rk = rk4(f, y0, t0, tf, dt)

    # Print final values
    print(f"Euler   y({tf}) = {y_eu[-1]:.5f}")
    print(f"RK4     y({tf}) = {y_rk[-1]:.5f}")
    print(f"Exact   y({tf}) = {np.exp(-tf):.5f}")