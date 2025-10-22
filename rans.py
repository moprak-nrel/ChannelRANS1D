import numpy as np
from scipy.integrate import ode

import sa


def time_march(initial_state, steps, dt):
    def rhs(t, state):
        return sa.get_dXdt(state)
    integrator = ode(rhs).set_integrator("lsoda")
    integrator.set_initial_value(initial_state, 0)
    states = np.empty((steps + 1, len(initial_state)))
    i = 0
    states[0, :] = initial_state
    while integrator.successful() and i < steps:
        i += 1
        states[i, :] = integrator.integrate(integrator.t + dt)
        print(i, np.linalg.norm(sa.get_dXdt(states[i, :])))
    return states
