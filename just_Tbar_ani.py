from params import L_c as crossbar_length, L_s as shaft_length
from ode_solver import sol, omega_vecs, L_vecs
from params import omega0, t_span, t_eval
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm


# Coordinates for the shaft (x-axis bar)
shaft_body = np.array([[-shaft_length / 2, shaft_length / 2], [0, 0], [0, 0]])

# Coordinates for the crossbar (z-axis bar, attached at the end of the shaft)
crossbar_body = np.array([
    [shaft_length / 2, shaft_length / 2],  # Offset to the end of the shaft
    [0, 0],                               # Stays aligned with y=0
    [-crossbar_length / 2, crossbar_length / 2]  # Spans symmetrically along z-axis
])

# Initialize the rotation matrix (identity at t=0)
R = np.eye(3)

def get_rotation_matrix(omega, dt):
    norm = np.linalg.norm(omega)
    if norm == 0:
        return np.eye(3)
    axis = omega / norm
    skew = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return expm(skew * norm * dt)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scale = 1.1
ax.set_xlim([-shaft_length*scale, shaft_length*scale])
ax.set_ylim([-shaft_length*scale, shaft_length*scale])
ax.set_zlim([-shaft_length*scale, shaft_length*scale])
ax.legend()
shaft_line, = ax.plot([], [], [], color='black', linewidth=3, label="T Handle")
crossbar_line, = ax.plot([], [], [], color='black', linewidth=3)


# Update function for the animation
def update(frame):
    global R

    # Compute the rotation matrix for this time step
    dt = t_eval[1] - t_eval[0]  # Time step size
    R = R @ get_rotation_matrix(omega_vecs[frame], dt)  # Update orientation

    # Rotate the T-handle geometry into the ground frame
    shaft_rotated = R @ shaft_body
    crossbar_rotated = R @ crossbar_body

    # Rotate omega into the ground frame
    # omega_rotated = R @ omega_vecs[frame]

    # Update shaft and crossbar lines
    shaft_line.set_data(shaft_rotated[0], shaft_rotated[1])
    shaft_line.set_3d_properties(shaft_rotated[2])

    crossbar_line.set_data(crossbar_rotated[0], crossbar_rotated[1])
    crossbar_line.set_3d_properties(crossbar_rotated[2])

    return shaft_line, crossbar_line #, omega_line, L_line

ani = FuncAnimation(fig, update, frames=len(omega_vecs), interval=100, blit=True)
# ani.save("animations/w1_fast.gif", writer="pillow", fps=30)
plt.show()