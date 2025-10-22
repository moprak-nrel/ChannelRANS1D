import numpy as np
import numpy.linalg as LA
from scipy.integrate import ode

from sa import *

# from sa_full import *


# state:= [u, nu_tilda]^T
def rhs(t, state):
    return get_dXdt(state)


def time_march(initial_state, steps, dt):
    integrator = ode(rhs).set_integrator("lsoda")
    integrator.set_initial_value(initial_state, 0)
    states = np.empty((steps + 1, 2 * N))
    i = 0
    states[0, :] = initial_state
    while integrator.successful() and i < steps:
        i += 1
        states[i, :] = integrator.integrate(integrator.t + dt)
        print(i, LA.norm(get_dXdt(states[i, :])))

        if i % 10 == 0:
            np.save("%d-last" % Re_tau, states[i, :])
    return states
