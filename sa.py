import numpy as np
import scipy.interpolate as interp
from scipy.interpolate import splev, splrep

import ke

Re_tau_round = 5200
Re_tau_table = {
    180: 182.088,
    550: 543.496,
    1000: 1000.512,
    2000: 1994.756,
    5200: 5185.897,
}
Re_tau = Re_tau_table[Re_tau_round]
data = ke.read_data(Re_tau_round)
Y_data = data["Y"]
Y = Y_data

N = len(Y)
Yp = Y * Re_tau
K = 0.41
nu = 1.0 / Re_tau
sigmav = 2.0 / 3.0
cb1 = 0.1355
cb2 = 0.622
cw1 = cb1 / K**2 + (1 + cb2) / sigmav


def get_spline_rep_U(U):
    U[0] = 0
    cs = interp.CubicSpline(Y, U, bc_type=("not-a-knot", "clamped"))
    return cs


def get_spline_rep_nu(X):
    X[0] = 0
    cs = interp.CubicSpline(Y, X, bc_type=("not-a-knot", "clamped"))
    return cs


def get_y_der(tck):
    res = tck(Y, 1)
    res[-1] = 0
    return res


def get_yy_der(tck):
    res = tck(Y, 2)
    return res


def get_dXdt(state):
    U = state[:N]
    nu_tilde = state[N:]
    utck = get_spline_rep_U(U)
    ntck = get_spline_rep_nu(nu_tilde)
    dyU = get_y_der(utck)
    dyyU = get_yy_der(utck)
    dynu = get_y_der(ntck)
    dyynu = get_yy_der(ntck)

    dUdt = get_dUdt(U, dyU, dyyU, nu_tilde)
    dUdt[0] = 0
    dnudt = get_dnudt(U, dyU, nu_tilde, dynu, dyynu)
    return np.hstack([dUdt, dnudt])


def get_dUdt(U, dyU, dyyU, nu_tilde):
    nuT = get_nuT(nu_tilde)
    ntck = get_spline_rep_nu(nuT)
    res = 1 + (1.0 / Re_tau + nuT) * dyyU + get_y_der(ntck) * dyU
    return res


def get_nuT(nu_tilde):
    temp = (nu_tilde / nu) ** 3
    return nu_tilde * (temp / (temp + 7.1**3))


def get_Stilde(dyU, nu_tilde):
    nuT = get_nuT(nu_tilde)
    S_tilde = np.zeros_like(Y)
    S_tilde[1:] = (
        dyU[1:]
        + (-(nu_tilde[1:] ** 2) / (nu + nuT[1:]) + nu_tilde[1:]) / (K * Y[1:]) ** 2
    )
    return S_tilde


def get_Pnu(dyU, nu_tilde):
    return (
        cb1
        * (dyU + (-(nu_tilde**2) / (nu + nu_tilde) + nu_tilde) / ((K * Y) ** 2))
        * nu_tilde
    )  # S*nu_tilde
    # return (dyU + ( - nu_tilde**2/(nu+nu_tilde) + nu_tilde)/((K*Y)**2))*nu_tilde # S*nu_tilde


def get_r(dyU, nu_tilde):
    # den = (K*Y)**2*dyU - nu_tilde**2/(nu+nu_tilde) + nu_tilde
    # r = nu_tilde/den
    r = nu_tilde / (get_Stilde(dyU, nu_tilde) * (K * Y) ** 2)
    r[0] = 0
    return r


def get_g(r):
    return r + 0.3 * (r**6 - r)


def get_f(r):
    g = get_g(r)
    res = g * (65.0 / (64.0 + g**6.0)) ** (1.0 / 6.0)
    return np.minimum(res, 2.00517475)


def get_Enu(dyU, nu_tilde):
    Enu = cw1 * (nu_tilde / Y) ** 2 * get_f(get_r(dyU, nu_tilde))
    Enu[0] = 0
    return Enu


def get_dnudt(U, dyU, nu_tilde, dynu, dyynu):
    res = (
        get_Pnu(dyU, nu_tilde)
        - get_Enu(dyU, nu_tilde)
        + 1.0 / sigmav * ((nu + nu_tilde) * dyynu + (1 + cb2) * dynu**2)
    )
    res[0] = 0
    return res


def get_nu_tilde_init():
    nuT_data = (-data["uv"] / data["dUdy"]) / Re_tau
    tck = splrep(Y_data, nuT_data, k=5)
    return splev(Y, tck)


def get_U_init():
    Udata = data["U"]
    tck = splrep(Y_data, Udata, k=5)
    return splev(Y, tck)
