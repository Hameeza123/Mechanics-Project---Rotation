from params import L_c as crossbar_length, L_s as shaft_length
from ode_solver import sol, omega_vecs, L_vecs
from params import omega0, t_span, t_eval
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm  # For matrix exponential to compute rotation matrix thingy

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

# Helper function to compute rotation matrix from angular velocity
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

limit = np.max(np.abs(np.vstack((omega_vecs, L_vecs))))# * 1.5
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

# Add T-handle geometry to the plot
shaft_line, = ax.plot([], [], [], color='black', linewidth=3, label="T Handle")
crossbar_line, = ax.plot([], [], [], color='black', linewidth=3)

omega_quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='green', label=r'$\omega(t)$ (Angular Velocity)', arrow_length_ratio=0.1)
L_quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='red', label=r'$L(t)$ (Angular Momentum)', arrow_length_ratio=0.1)

ax.legend()

L_ground = L_vecs[0]  # Angular momentum is constant in the inertial frame - should work fine without this tho

def update(frame):
    global R, omega_quiver, L_quiver

    dt = t_eval[1] - t_eval[0]
    R = R @ get_rotation_matrix(omega_vecs[frame], dt)

    # Rotate the T-handle geometry into the ground frame
    shaft_rotated = R @ shaft_body
    crossbar_rotated = R @ crossbar_body
    omega_rotated = R @ omega_vecs[frame]

    shaft_line.set_data(shaft_rotated[0], shaft_rotated[1])
    shaft_line.set_3d_properties(shaft_rotated[2])

    crossbar_line.set_data(crossbar_rotated[0], crossbar_rotated[1])
    crossbar_line.set_3d_properties(crossbar_rotated[2])

    # Remove old quiver arrows
    omega_quiver.remove()
    L_quiver.remove()

    omega_quiver = ax.quiver(0, 0, 0,
                             omega_rotated[0], omega_rotated[1], omega_rotated[2],
                             color='green', length=1, normalize=True, arrow_length_ratio=0.1)

    L_quiver = ax.quiver(0, 0, 0,
                         L_ground[0], L_ground[1], L_ground[2],
                         color='red', length=1, normalize=True, arrow_length_ratio=0.1)

    return shaft_line, crossbar_line, omega_quiver, L_quiver

ani = FuncAnimation(fig, update, frames=len(omega_vecs), interval=100, blit=False)
# ani.save("animations/ground_w1_pert.gif", writer="pillow", fps=30)
plt.show()