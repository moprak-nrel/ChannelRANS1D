import numpy as np

from rans import RANSSolver

np.random.seed(1)


def main():
    rans_solver = RANSSolver(Re_tau_round=5200)
    dt = 10
    steps = 20
    states = rans_solver.run_simulation(steps=steps, dt=dt, gen_plots=True)


if __name__ == "__main__":
    main()
