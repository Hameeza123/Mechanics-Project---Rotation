import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from params import omega0, t_span, t_eval
from inertia_Thandle import I_total


# we alr found inertia tensor in diag form - will have to do the math for this in Latex
I1 = I_total[0][0]
I2 = I_total[1][1]
I3 = I_total[2][2]

# Inertia Tensor
def I():
    return np.diag([I1, I2, I3])

def f(t, omega):
    omega1, omega2, omega3 = omega
    f1 = (I2 - I3) * omega2 * omega3
    f2 = (I3 - I1) * omega3 * omega1
    f3 = (I1 - I2) * omega1 * omega2
    return np.array([f1, f2, f3])

# Define the system in standard ODE form: M * d(omega)/dt = f -> d(omega)/dt = M^-1 @ f
def euler_system(t, omega):
    M_inv = np.linalg.inv(I())  # Compute the inverse of M
    return M_inv @ f(t, omega)        # Return d(omega)/dt

# 4th and 5th order runga kutta - might want to switch to guassian quad or burlisch stoer or smth later
sol = solve_ivp(euler_system, t_span, omega0, method='RK45', t_eval=t_eval)

# Convert sol.y to a NumPy array for easier manipulation
omega_vecs = sol.y.T  # Shape: (len(t_eval), 3)
# print(omega_vs)

# Compute L for each omega
L_vecs = np.dot(I(), omega_vecs.T).T  # Shape: (len(t_eval), 3)
# print(L)
