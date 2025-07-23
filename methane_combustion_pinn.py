import deepxde as dde
import numpy as np

# Domain definition: 1D spatial domain [0, L] and time [0, T]
L = 1.0  # length of the combustion chamber (normalized)
T_final = 1.0  # final time (normalized)

# Physical parameters (simplified)
u = 1.0  # advection velocity
D = 0.01  # diffusion coefficient for species
alpha = 0.01  # thermal diffusivity
k = 10.0  # pre-exponential factor for reaction rate
Ea = 50.0  # activation energy (arbitrary units)
R = 1.0  # universal gas constant (normalized)
Q = 10.0  # heat release term


def pde(x, y):
    """PDE for methane mass fraction (Y) and temperature (T)."""
    Y = y[:, 0:1]
    temp = y[:, 1:2]

    Y_t = dde.grad.jacobian(y, x, i=0, j=1)  # dY/dt
    Y_x = dde.grad.jacobian(y, x, i=0, j=0)  # dY/dx
    Y_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)

    T_t = dde.grad.jacobian(y, x, i=1, j=1)  # dT/dt
    T_x = dde.grad.jacobian(y, x, i=1, j=0)  # dT/dx
    T_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

    # Arrhenius-type reaction rate
    reaction_rate = k * Y * dde.math.exp(-Ea / (R * temp))

    eq_Y = Y_t + u * Y_x - D * Y_xx + reaction_rate
    eq_T = T_t + u * T_x - alpha * T_xx - Q * reaction_rate
    return [eq_Y, eq_T]


# Initial conditions: unburned methane (Y=1) and ambient temperature (T=1)
def init_cond_Y(x):
    return np.ones((len(x), 1))


def init_cond_T(x):
    return np.ones((len(x), 1))


# Boundary condition indicator functions
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)


def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)


def main():
    spatial_domain = dde.geometry.Interval(0, L)
    temporal_domain = dde.geometry.TimeDomain(0, T_final)
    geomtime = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

    # Boundary conditions for species and temperature
    bc_Y_left = dde.DirichletBC(geomtime, lambda x: 1.0, boundary_left, component=0)
    bc_Y_right = dde.DirichletBC(geomtime, lambda x: 0.0, boundary_right, component=0)
    bc_T_left = dde.DirichletBC(geomtime, lambda x: 1.0, boundary_left, component=1)
    bc_T_right = dde.DirichletBC(geomtime, lambda x: 1.0, boundary_right, component=1)

    ic_Y = dde.IC(geomtime, init_cond_Y, lambda _, on_initial: on_initial, component=0)
    ic_T = dde.IC(geomtime, init_cond_T, lambda _, on_initial: on_initial, component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_Y, ic_T, bc_Y_left, bc_Y_right, bc_T_left, bc_T_right],
        num_domain=2540,
        num_boundary=80,
        num_initial=160,
        solution=None,
        num_test=10000,
    )

    net = dde.nn.FNN([2] + [50] * 3 + [2], "tanh", "Glorot normal")

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    model.train(iterations=5000)
    model.save("model_checkpoint")


if __name__ == "__main__":
    main()
