from ode_solver import sol, omega_vecs, L_vecs
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def body_ani():
    from ode_solver import sol, omega_vecs, L_vecs
    from params import L_s, L_c

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    limit = np.max(np.abs(np.vstack((omega_vecs, L_vecs)))) # * 0.1
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    # Add T-handle geometry
    shaft_x = [-L_s / 2, L_s / 2]
    shaft_y = [0, 0]
    shaft_z = [0, 0]
    ax.plot(shaft_x, shaft_y, shaft_z, color='black', linewidth=2, label="T Handle")

    crossbar_x = [L_s / 2, L_s / 2]
    crossbar_y = [0, 0]
    crossbar_z = [-L_c / 2, L_c / 2]
    ax.plot(crossbar_x, crossbar_y, crossbar_z, color='black', linewidth=2)


    # quiver objects for omega and L (start with zero-length arrows)
    omega_quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='green', label=r'$\omega(t)$ (Angular Velocity)', arrow_length_ratio=0.1)
    L_quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='red', label=r'$L(t)$ (Angular Momentum)', arrow_length_ratio=0.1)

    ax.legend()

    # Update function for the animation
    def update(frame):
        nonlocal omega_quiver, L_quiver

        omega_quiver.remove()
        L_quiver.remove()

        omega_quiver = ax.quiver(0, 0, 0,
                                 omega_vecs[frame, 0], omega_vecs[frame, 1], omega_vecs[frame, 2],
                                 color='green', length=1, normalize=True, arrow_length_ratio=0.1)

        L_quiver = ax.quiver(0, 0, 0,
                             L_vecs[frame, 0], L_vecs[frame, 1], L_vecs[frame, 2],
                             color='red', length=1, normalize=True, arrow_length_ratio=0.1)

        return omega_quiver, L_quiver

    ani = FuncAnimation(fig, update, frames=len(omega_vecs), interval=100, blit=False)
    # ani.save("animations/body_w1_pert.gif", writer="pillow", fps=30)


    plt.show()

body_ani()




# # Plot the results
# plt.plot(sol.t, sol.y[0], label=r'$\omega_1(t)$')
# plt.plot(sol.t, sol.y[1], label=r'$\omega_2(t)$')
# plt.plot(sol.t, sol.y[2], label=r'$\omega_3(t)$')
# plt.xlabel('Time')
# plt.ylabel('Angular Velocities')
# plt.legend()
# plt.title('Solution of Euler Equations for Rotation Without Torque')
# plt.grid()
# plt.show()

####################################################################
# def body_ani():
#     from ode_solver import sol, omega_vecs, L_vecs
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from mpl_toolkits.mplot3d import Axes3D
#     from matplotlib.animation import FuncAnimation
#     from params import L_s, L_c

#     # Correctly adjust T-handle geometry
#     # Coordinates for the shaft (x-axis bar)
#     shaft_x = [-L_s / 2, L_s / 2]  # Shaft spans symmetrically along the x-axis
#     shaft_y = [0, 0]
#     shaft_z = [0, 0]

#     # Coordinates for the crossbar (z-axis bar, centered at the top of the shaft)
#     crossbar_x = [L_s / 2, L_s / 2]  # Positioned at the top of the shaft
#     crossbar_y = [0, 0]
#     crossbar_z = [-L_c / 2, L_c / 2]  # Crossbar spans symmetrically along the z-axis

#     # Set up the 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Determine plot limits dynamically from the data
#     limit = np.max(np.abs(np.vstack((omega_vecs, L_vecs))))# *1.1
#     ax.set_xlim([-limit, limit])
#     ax.set_ylim([-limit, limit])
#     ax.set_zlim([-limit, limit])

#     # Add T-handle geometry to the plot
#     shaft_line, = ax.plot(shaft_x, shaft_y, shaft_z, color='black', linewidth=3, label="T Handle")
#     crossbar_line, = ax.plot(crossbar_x, crossbar_y, crossbar_z, color='black', linewidth=3)

#     # Initialize the vectors for omega and L
#     omega_line, = ax.plot([], [], [], color='g', label=r'$\omega(t)$ (Angular Velocity)')
#     L_line, = ax.plot([], [], [], color='r', label=r'$L(t)$ (Angular Momentum)')

#     # Add a legend
#     ax.legend()

#     # Update function for the animation
#     def update(frame):
#         # Update omega vector
#         omega_line.set_data([0, omega_vecs[frame, 0]], 
#                             [0, omega_vecs[frame, 1]])
#         omega_line.set_3d_properties([0, omega_vecs[frame, 2]])

#         # Update L vector
#         L_line.set_data([0, L_vecs[frame, 0]], 
#                         [0, L_vecs[frame, 1]])
#         L_line.set_3d_properties([0, L_vecs[frame, 2]])

#         # Return updated elements
#         return omega_line, L_line, shaft_line, crossbar_line

#     # Create the animation
#     ani = FuncAnimation(fig, update, frames=len(omega_vecs), interval=100, blit=False)
#     ani.save("animations/body_I3_pert.gif", writer="pillow", fps=30)


#     # Show the animation
#     plt.show()

# body_ani()