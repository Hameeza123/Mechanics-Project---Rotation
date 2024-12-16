from ode_solver import sol, omega_vecs, L_vecs
from params import L_s, L_c
import numpy as np

# Set up the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Determine plot limits dynamically from the data
limit = np.max(np.abs(np.vstack((omega_vecs, L_vecs))))
min = np.min(np.abs(np.vstack((omega_vecs, L_vecs))))

print(limit, min)
for elem in omega_vecs:
    print(elem)
# from matplotlib.animation import FuncAnimation

# # Extract solution data
# omega_vector = sol.y  # Shape: (3, len(t_eval))
# t_eval = sol.t        # Time array
# L_vector = M_matrix() @ omega_vector
# print(L_vector)



# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# # Set up the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Set plot limits
# ax.set_xlim([-2, 2])
# ax.set_ylim([-2, 2])
# ax.set_zlim([-2, 2])
# ax.set_xlabel(r'$\omega_1$ / $L_1$')
# ax.set_ylabel(r'$\omega_2$ / $L_2$')
# ax.set_zlabel(r'$\omega_3$ / $L_3$')
# ax.set_title('Animation of $\boldsymbol{\omega}(t)$ and $\mathbf{L}(t)$')

# # Initialize the angular velocity vector
# omega_line, = ax.plot([0, omega_vector[0, 0]],
#                       [0, omega_vector[1, 0]],
#                       [0, omega_vector[2, 0]],
#                       color='r', label=r'$\boldsymbol{\omega}(t)$')

# # Initialize the angular momentum vector
# L_line, = ax.plot([0, L_vector[0, 0]],
#                   [0, L_vector[1, 0]],
#                   [0, L_vector[2, 0]],
#                   color='g', label=r'$\mathbf{L}(t)$')

# # Initialize traces
# omega_trace, = ax.plot([], [], [], color='r', alpha=0.5, label=r'$\boldsymbol{\omega}$ Trace')
# L_trace, = ax.plot([], [], [], color='g', alpha=0.5, label=r'$\mathbf{L}$ Trace')

# # Initialize trace storage
# omega_trace_points = []
# L_trace_points = []

# # Update function for the animation
# def update(frame):
#     global omega_trace_points, L_trace_points

#     # Update omega vector
#     omega_line.set_data([0, omega_vector[0, frame]],
#                         [0, omega_vector[1, frame]])
#     omega_line.set_3d_properties([0, omega_vector[2, frame]])

#     # Update L vector
#     L_line.set_data([0, L_vector[0, frame]],
#                     [0, L_vector[1, frame]])
#     L_line.set_3d_properties([0, L_vector[2, frame]])

#     # Update traces
#     omega_trace_points.append((omega_vector[0, frame], omega_vector[1, frame], omega_vector[2, frame]))
#     L_trace_points.append((L_vector[0, frame], L_vector[1, frame], L_vector[2, frame]))

#     # Extract trace points
#     if omega_trace_points:
#         omega_x, omega_y, omega_z = zip(*omega_trace_points)
#         omega_trace.set_data(omega_x, omega_y)
#         omega_trace.set_3d_properties(omega_z)

#     if L_trace_points:
#         L_x, L_y, L_z = zip(*L_trace_points)
#         L_trace.set_data(L_x, L_y)
#         L_trace.set_3d_properties(L_z)

#     return omega_line, L_line, omega_trace, L_trace

# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=False)

# # Add legend
# ax.legend()

# # Show the animation
# plt.show()


# # Set up the figure and 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Set plot limits and labels
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.5, 1.5])
# ax.set_zlim([-1.5, 1.5])
# ax.set_xlabel(r'$\omega_1$')
# ax.set_ylabel(r'$\omega_2$')
# ax.set_zlabel(r'$\omega_3$')
# ax.set_title('Animation of $\mathbf{\omega}(t)$')

# # Initialize the vector and trace
# vector_line, = ax.plot([0, omega_vector[0, 0]], 
#                        [0, omega_vector[1, 0]], 
#                        [0, omega_vector[2, 0]], 
#                        color='r', label=r'$\mathbf{\omega}(t)$')
# trace, = ax.plot([], [], [], color='b', alpha=0.5, label='Trace')

# # Initialize storage for trace points
# trace_points = []

# # Update function for animation
# def update(frame):
#     global trace_points
#     # Update the vector line
#     vector_line.set_data([0, omega_vector[0, frame]], 
#                          [0, omega_vector[1, frame]])
#     vector_line.set_3d_properties([0, omega_vector[2, frame]])

#     # Add the current tip of the vector to the trace
#     trace_points.append((omega_vector[0, frame], 
#                          omega_vector[1, frame], 
#                          omega_vector[2, frame]))

#     # Update the trace line
#     x_trace, y_trace, z_trace = zip(*trace_points)
#     trace.set_data(x_trace, y_trace)
#     trace.set_3d_properties(z_trace)

#     return vector_line, trace

# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=False)

# # Add legend
# ax.legend()

# # Show the animation
# plt.show()

# # Define T-handle geometry in the body frame
# from params import L_c as crossbar_length, L_s as shaft_length
# from ode_solver import sol, omega_vecs, L_vecs
# from params import omega0, t_span, t_eval
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
# from scipy.linalg import expm  # For matrix exponential to compute rotation

# # Assuming omega_vecs and L_vecs are NumPy arrays of shape (N, 3)
# # Where N is the number of time steps


# # Coordinates for the shaft (x-axis bar)
# shaft_body = np.array([[-shaft_length / 2, shaft_length / 2], [0, 0], [0, 0]])

# # Coordinates for the crossbar (z-axis bar, attached at the end of the shaft)
# crossbar_body = np.array([
#     [shaft_length / 2, shaft_length / 2],  # Offset to the end of the shaft
#     [0, 0],                               # Stays aligned with y=0
#     [-crossbar_length / 2, crossbar_length / 2]  # Spans symmetrically along z-axis
# ])

# # Initialize the rotation matrix (identity at t=0)
# R = np.eye(3)

# # Helper function to compute rotation matrix from angular velocity
# def get_rotation_matrix(omega, dt):
#     """
#     Compute the rotation matrix increment for a given angular velocity vector.

#     Parameters:
#         omega: array-like, shape (3,)
#             Angular velocity vector [omega_x, omega_y, omega_z].
#         dt: float
#             Time step size.

#     Returns:
#         R_increment: ndarray, shape (3, 3)
#             The rotation matrix increment for the given angular velocity.
#     """
#     norm = np.linalg.norm(omega)
#     if norm == 0:
#         return np.eye(3)
#     axis = omega / norm
#     skew = np.array([
#         [0, -axis[2], axis[1]],
#         [axis[2], 0, -axis[0]],
#         [-axis[1], axis[0], 0]
#     ])
#     return expm(skew * norm * dt)

# # Set up the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Determine plot limits dynamically from the data
# limit = np.max(np.abs(np.vstack((omega_vecs, L_vecs)))) * 1.5
# ax.set_xlim([-limit, limit])
# ax.set_ylim([-limit, limit])
# ax.set_zlim([-limit, limit])

# # Add T-handle geometry to the plot
# shaft_line, = ax.plot([], [], [], color='black', linewidth=3, label="T Handle")
# crossbar_line, = ax.plot([], [], [], color='black', linewidth=3)

# # Initialize the vectors for omega and L
# omega_line, = ax.plot([], [], [], color='g', label=r'$\omega(t)$ (Angular Velocity)')
# L_line, = ax.plot([], [], [], color='r', label=r'$L$ (Angular Momentum)')

# # Add a legend
# ax.legend()

# # Compute the angular momentum in the ground frame (constant)
# L_ground = L_vecs[0]  # Angular momentum is constant in the inertial frame

# # Update function for the animation
# def update(frame):
#     global R

#     # Compute the rotation matrix for this time step
#     dt = t_eval[1] - t_eval[0]  # Time step size
#     R = R @ get_rotation_matrix(omega_vecs[frame], dt)  # Update orientation

#     # Rotate the T-handle geometry into the ground frame
#     shaft_rotated = R @ shaft_body
#     crossbar_rotated = R @ crossbar_body

#     # Rotate omega into the ground frame
#     omega_rotated = R @ omega_vecs[frame]

#     # Update shaft and crossbar lines
#     shaft_line.set_data(shaft_rotated[0], shaft_rotated[1])
#     shaft_line.set_3d_properties(shaft_rotated[2])

#     crossbar_line.set_data(crossbar_rotated[0], crossbar_rotated[1])
#     crossbar_line.set_3d_properties(crossbar_rotated[2])

#     # Update omega vector in the ground frame
#     omega_line.set_data([0, omega_rotated[0]], 
#                         [0, omega_rotated[1]])
#     omega_line.set_3d_properties([0, omega_rotated[2]])

#     # Keep L constant in the ground frame
#     L_line.set_data([0, L_ground[0]], 
#                     [0, L_ground[1]])
#     L_line.set_3d_properties([0, L_ground[2]])

#     return shaft_line, crossbar_line, omega_line, L_line

# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(omega_vecs), interval=100, blit=True)
# # ani.save("gound_I1_pert.gif", writer="pillow", fps=30)


# # Show the animation
# plt.show()