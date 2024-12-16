from params import L_s, L_c, m_s, m_c
import numpy as np
from params import L_s, L_c, m_s, m_c
import numpy as np

total_mass = m_s + m_c

shaft_com = np.array([0, 0, 0])  # Shaft is centered at the origin
crossbar_com = np.array([L_s / 2, 0, 0])  # Crossbar is offset along x-axis by L_s/2
system_com = (m_s * shaft_com + m_c * crossbar_com) / total_mass  # Weighted average

d_shaft = shaft_com - system_com
d_crossbar = crossbar_com - system_com

# Inertia tensor for the shaft about its own COM
I_shaft = np.array([
    [1 / 12 * m_s * L_s**2, 0, 0],
    [0, 1 / 12 * m_s * L_s**2, 0],
    [0, 0, 0] 
])

# Inertia tensor for the crossbar about its own COM
I_crossbar = np.array([
    [0, 0, 0],
    [0, 1 / 12 * m_c * L_c**2, 0],
    [0, 0, 1 / 12 * m_c * L_c**2]
])

# Adjust inertia tensor to the system COM
d_outer_shaft = np.outer(d_shaft, d_shaft)
I_shaft_adjusted = I_shaft + m_s * (np.eye(3) * np.dot(d_shaft, d_shaft) - d_outer_shaft)

d_outer_crossbar = np.outer(d_crossbar, d_crossbar)
I_crossbar_adjusted = I_crossbar + m_c * (np.eye(3) * np.dot(d_crossbar, d_crossbar) - d_outer_crossbar)

I_total = I_shaft_adjusted + I_crossbar_adjusted


# NOT COM but the intersection of the T:

I_shaft_int = np.array([
    [1/12 * m_s * L_s**2, 0, 0],
    [0, 1/12 * m_s * L_s**2, 0],
    [0, 0, 0]  # inf thin bar has no moment about its own axis
])


I_crossbar_int = np.array([
    [0, 0, 0],
    [0, 1/12 * m_c * L_c**2, 0],
    [0, 0, 1/12 * m_c * L_c**2]
])

# Parallel axis theorem for the crossbar (offset along the z-axis by +L_s/2)
d_crossbar = np.array([0, 0, L_s / 2]) # THIS IS NOT COM i should change to be at com
d_outer = np.outer(d_crossbar, d_crossbar)
I_crossbar_adjusted = I_crossbar_int + m_c * (np.eye(3) * np.dot(d_crossbar, d_crossbar) - d_outer)

I_total_intersection = I_shaft_int + I_crossbar_adjusted
# print(I_total)