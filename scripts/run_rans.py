import numpy as np

from channelrans1d.rans import RANSSolver

np.random.seed(1)


def get_rans_output(sa_params, gen_plots=False):
    rans_solver = RANSSolver(Re_tau_round=5200, sa_params=sa_params)
    dt = 10
    steps = 20
    states = rans_solver.run_simulation(steps=steps, dt=dt, gen_plots=gen_plots)
    ny = rans_solver.sa_model.ny
    data = np.vstack(
        [
            states[0, :ny],
            states[0, ny:],
        ]
    )
    model_out = np.vstack(
        [
            states[-1, :ny],
            rans_solver.sa_model.get_nuT(states[-1, ny:]),
        ]
    )
    return rans_solver.sa_model.Yp, data, model_out


if __name__ == "__main__":
    sa_params = {
        "sigmav": 2.0 / 3.0,
        "cb1": 0.1355,
        "cb2": 0.622,
        "cw2": 0.3,
        "cw3": 2,
    }
    # y is wall-normal coordinate (wall units)
    # data[0], model_out[0] are the velocities in (wall units)
    # data[1], model_out[1] are nu_t (not nu_tilde) (wall units)
    y, data, model_out = get_rans_output(sa_params, gen_plots=True)
    
