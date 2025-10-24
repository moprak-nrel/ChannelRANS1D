import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

from sa import SpalartAllmaras


class RANSSolver:
    """RANS solver using SA."""

    def __init__(self, Re_tau_round=5200):
        """Initialize the RANS solver."""
        self.Re_tau_round = Re_tau_round
        self.sa_model = SpalartAllmaras(Re_tau_round=Re_tau_round)

    def get_initial_state(self):
        """Get initial state vector from DNS data."""
        U_init = self.sa_model.get_U_init()
        nu_tilde_init = self.sa_model.get_nu_tilde_init()
        return np.hstack([U_init, nu_tilde_init])

    def load_restart_state(self):
        """Load restart state and time from files if they exist."""
        restart_file = f"{self.Re_tau_round}-last.npy"
        time_file = f"{self.Re_tau_round}-final-time.npy"

        if os.path.isfile(restart_file) and os.path.isfile(time_file):
            print("Using restart files....")
            initial_state = np.load(restart_file)
            initial_time = np.load(time_file)
            return initial_state, initial_time
        else:
            return self.get_initial_state(), 0.0

    def time_march(self, initial_state, steps, dt):
        """Time stepping."""

        def rhs(t, state):
            return self.sa_model.get_dXdt(state)

        integrator = ode(rhs).set_integrator("lsoda")
        integrator.set_initial_value(initial_state, 0)
        states = np.empty((steps + 1, len(initial_state)))
        i = 0
        states[0, :] = initial_state

        while integrator.successful() and i < steps:
            i += 1
            states[i, :] = integrator.integrate(integrator.t + dt)
            print(
                f"Step: {i}, dX/dt norm: {np.linalg.norm(self.sa_model.get_dXdt(states[i, :]))}"
            )

        return states

    def save_final_state(self, states, steps, dt, initial_time=0.0):
        """Save final state and simulation data."""
        final_state = states[-1]

        # Save final state as a text file for plotting
        np.savetxt(
            f"data/{self.Re_tau_round}-final-state.dat",
            np.vstack(
                [
                    self.sa_model.Yp,
                    final_state[: self.sa_model.N],
                    self.sa_model.get_nuT(final_state[self.sa_model.N :]),
                ]
            ).T,
        )

        # Save restart files
        np.save(f"{self.Re_tau_round}-last", final_state)
        final_time = steps * dt + initial_time
        print("final time:", final_time)
        np.save(f"{self.Re_tau_round}-final-time", final_time)

    def generate_plots(self, states, steps):
        """Generate plots comparing RANS results with DNS data."""
        # Velocity plots
        fig = plt.figure()
        plt.plot(
            self.sa_model.Y,
            states[int(steps / 2 - 1)][: self.sa_model.N],
            "g-",
            label="RANS half-simulation",
        )
        plt.plot(
            self.sa_model.Y, states[steps - 1][: self.sa_model.N], "b-", label="RANS"
        )
        plt.plot(self.sa_model.Y, self.sa_model.get_U_init(), "r--", label="DNS")
        plt.ylabel(r"$U$")
        plt.xlabel(r"$\widetilde{y}$")
        plt.legend(loc="best")
        fig.tight_layout()
        plt.savefig(f"figs/{self.Re_tau_round}-U.pdf")

        fig = plt.figure()
        plt.loglog(
            self.sa_model.Yp[1 : self.sa_model.N],
            states[steps - 1][1 : self.sa_model.N],
            "b-",
            label="RANS",
        )
        plt.loglog(
            self.sa_model.Yp[1 : self.sa_model.N],
            self.sa_model.get_U_init()[1 : self.sa_model.N],
            "r--",
            label="DNS",
        )
        plt.ylabel(r"$U$")
        plt.xlabel(r"$y^+$")
        plt.legend(loc="best")
        fig.tight_layout()
        plt.savefig(f"figs/{self.Re_tau_round}-U-loglog.pdf")

        fig = plt.figure()
        plt.semilogx(
            self.sa_model.Yp[1 : self.sa_model.N],
            states[steps - 1][1 : self.sa_model.N],
            "b-",
            label="RANS",
        )
        plt.semilogx(
            self.sa_model.Yp[1 : self.sa_model.N],
            self.sa_model.get_U_init()[1 : self.sa_model.N],
            "r--",
            label="DNS",
        )
        plt.ylabel(r"$U$")
        plt.xlabel(r"$y^+$")
        plt.legend(loc="best")
        fig.tight_layout()
        plt.savefig(f"figs/{self.Re_tau_round}-U-semilog.pdf")

        # Turbulent viscosity plots
        fig = plt.figure()
        plt.plot(
            self.sa_model.Y,
            self.sa_model.get_nuT(states[steps - 1][self.sa_model.N :]),
            "b-",
            label="RANS",
        )
        plt.plot(self.sa_model.Y, self.sa_model.get_nu_tilde_init(), "r--", label="DNS")
        plt.ylabel(r"$\nu_\tau$")
        plt.xlabel(r"$\widetilde{y}$")
        plt.legend(loc="best")
        fig.tight_layout()
        plt.savefig(f"figs/{self.Re_tau_round}-nu_tilde.pdf")

        fig = plt.figure()
        plt.semilogx(
            self.sa_model.Yp[1 : self.sa_model.N],
            self.sa_model.get_nuT(states[steps - 1][self.sa_model.N + 1 :]),
            "b-",
            label="RANS",
        )
        plt.semilogx(
            self.sa_model.Yp[1 : self.sa_model.N],
            self.sa_model.get_nu_tilde_init()[1 : self.sa_model.N],
            "r--",
            label="DNS",
        )
        plt.ylabel(r"$\nu_\tau$")
        plt.xlabel(r"$y^+$")
        plt.legend(loc="best")
        fig.tight_layout()
        plt.savefig(f"figs/{self.Re_tau_round}-nu_tilde-semilog.pdf")

    def run_simulation(
        self, steps=2, dt=10, restart=False, save_final=False, gen_plots=False
    ):
        """Run the RANS for specified number of steps at a given dt."""
        if restart:
            initial_state, initial_time = self.load_restart_state()
        else:
            initial_state, initial_time = self.get_initial_state(), 0.0
        states = self.time_march(initial_state, steps, dt)
        if save_final:
            self.save_final_state(states, steps, dt, initial_time)
        if gen_plots:
            self.generate_plots(states, steps)

        return states
