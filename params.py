import numpy as np
# Dimensions and masses of the bars
L_s = 1  # Length of the shaft (m)
L_c = 1  # Length of the crossbar (m)
m_s = 1.5  # Mass of the shaft (kg)
m_c = 1.5  # Mass of the crossbar (kg)


# Time span for the simulation
t_span = (0, 20)  # From t=0 to t=20
t_eval = np.linspace(0, 20, 500)  # Evaluate solution at 500 points


# Example initial conditions
omega0 = [10, 0.1, 0.1]


"""
biggest I preset: rotation w perturbation around w2
import numpy as np
# Dimensions and masses of the bars
L_s = 1  # Length of the shaft (m)
L_c = 1  # Length of the crossbar (m)
m_s = 1.5  # Mass of the shaft (kg)
m_c = 1.5  # Mass of the crossbar (kg)


# Time span for the simulation
t_span = (0, 20)  # From t=0 to t=20
t_eval = np.linspace(0, 20, 500)  # Evaluate solution at 500 points


# Example initial conditions
omega0 = [0.2, -1, 0.2]
"""

"""
smallest I omega 3 preset
import numpy as np
# Dimensions and masses of the bars
L_s = 1  # Length of the shaft (m)
L_c = 1  # Length of the crossbar (m)
m_s = 1.5  # Mass of the shaft (kg)
m_c = 1.5  # Mass of the crossbar (kg)


# Time span for the simulation
t_span = (0, 20)  # From t=0 to t=20
t_eval = np.linspace(0, 20, 500)  # Evaluate solution at 500 points


# Example initial conditions
omega0 = [0.05, 0.05, 1]

"""


"""
Unstable I preset
import numpy as np
# Dimensions and masses of the bars
L_s = 1  # Length of the shaft (m)
L_c = 1  # Length of the crossbar (m)
m_s = 1.5  # Mass of the shaft (kg)
m_c = 1.5  # Mass of the crossbar (kg)


# Time span for the simulation
t_span = (0, 20)  # From t=0 to t=20
t_eval = np.linspace(0, 20, 500)  # Evaluate solution at 500 points


# Example initial conditions
omega0 = [1, 0.1, 0.1] # make omega 1 bigger if needed cause this pretty slow

"""