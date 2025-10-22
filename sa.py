
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
    10000: 10000,
}
Re_tau = Re_tau_table[Re_tau_round]
if Re_tau_round != 10000:
    data = ke.read_data(Re_tau_round)
    Y_data = data["Y"]
    Y = Y_data
else:
    data = ke.read_data(1000)
    Y_data = data["Y"]
    Y = Y_data
    # Y = np.loadtxt( 'data/y_10000.dat' )
# Y = np.linspace(0,1,300)
N = len(Y)
Yp = Y * Re_tau
K = 0.41
nu = 1.0 / Re_tau
dpdx = -1.0
sigmav = 2.0 / 3.0
cb1 = 0.1355
cb2 = 0.622
cw1 = cb1 / K**2 + (1 + cb2) / sigmav


def get_spline_rep_U(U):
    U[0] = 0
    cs = interp.CubicSpline(Y, U, bc_type=("not-a-knot", "clamped"))
    # cs = interp.CubicSpline(Y,U)
    return cs


def get_spline_rep_nu(X):
    X[0] = 0
    cs = interp.CubicSpline(Y, X, bc_type=("not-a-knot", "clamped"))
    # cs = interp.CubicSpline(Y,X)
    return cs


def get_y_der(tck):
    res = tck(Y, 1)
    res[-1] = 0
    return res


def get_yy_der(tck):
    res = tck(Y, 2)
    return res


def get_dy_fd(X):
    h = Y[2] - Y[0]
    res = np.empty_like(X)
    # Neumann
    res[-1] = 0
    # Central
    res[1:-1] = (X[2:] - X[:-2]) / h
    # Forward?
    res[0] = 2 * (X[1] - X[0]) / h
    return res


def get_dy2_fd(X):
    h = Y[1] - Y[0]
    res = np.empty_like(X)
    # Standard central difference
    # f'' = (f(y+h) + f(y-h) - 2*f(y))/h**2
    res[1:-1] = (X[2:] + X[:-2] - 2 * X[1:-1]) / h**2
    res[0] = (X[2] + X[0] - 2 * X[1]) / h**2
    # for the neumann end f' = 0
    # f(y-h) = f(y) + 1/2 * f''(y) * h**2
    # => f''(y) = 2*(f(y-h) - f(y))/h**2
    res[-1] = 2 * (X[-2] - X[-1]) / h**2
    return res


def get_dXdt_fd(state):
    U = state[:N]
    nu_tilde = state[N:]
    dyU = get_dy_fd(U)
    dyyU = get_dy2_fd(U)
    dynu = get_dy_fd(nu_tilde)
    dyynu = get_dy2_fd(nu_tilde)

    dUdt = get_dUdt_fd(U, dyU, dyyU, nu_tilde)
    dUdt[0] = 0
    dnudt = get_dnudt(U, dyU, nu_tilde, dynu, dyynu)
    return np.hstack([dUdt, dnudt])


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


def get_fin_diff_jac(state, rhs):
    numElem = state.size

    h = (2e-16) ** (1.0 / 3.0) * state

    e_i = np.zeros(numElem)
    jac = np.empty([numElem, numElem])

    for i in range(numElem):
        e_i[i] = 1.0
        rhsPlus = rhs(state + h[i] * e_i)
        rhsMinus = rhs(state - h[i] * e_i)

        jac[:, i] = (rhsPlus - rhsMinus) / (2.0 * h[i])

        e_i[i] = 0.0

    return jac


def get_jacobian_action(state, vec):
    # if the system of equations is:
    # U' = f_u( U, nu_tilde)
    # nu_tilde' = f_n( U, nu_tilde )
    # the jacobian J = [ df_u/dU df_u/dnu  ]
    #                  [ df_nu/dU df_nu/nu ]
    # and this function returns the action
    # J * vec.

    U = state[:N]
    nu_tilde = state[N:]
    utck = get_spline_rep_U(U)
    ntck = get_spline_rep_nu(nu_tilde)
    dyU = get_y_der(utck)
    dyyU = get_yy_der(utck)
    dynu = get_y_der(ntck)
    dyynu = get_yy_der(ntck)
    nuT = get_nuT(nu_tilde)
    nuTck = get_spline_rep_nu(nuT)
    dyNuT = get_y_der(nuTck)

    vtck = get_spline_rep_U(vec)
    dyVec = get_y_der(vtck)
    dyyVec = get_yy_der(vtck)
    # df_u/dU * vec
    dfUdU = (nu + nuT) * dyyVec + dyNuT * dyVec

    #


def get_dUdt(U, dyU, dyyU, nu_tilde):
    nuT = get_nuT(nu_tilde)
    ntck = get_spline_rep_nu(nuT)
    res = 1 + (1.0 / Re_tau + nuT) * dyyU + get_y_der(ntck) * dyU
    return res


def get_dUdt_fd(U, dyU, dyyU, nu_tilde):
    nuT = get_nuT(nu_tilde)
    res = 1 + (1.0 / Re_tau + nuT) * dyyU + get_dy_fd(nuT) * dyU
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
    # res =  get_Pnu(dyU,nu_tilde) - get_Enu(dyU,nu_tilde) + 1./sigmav * ( (nu+nu_tilde)*dyynu + (1)*dynu**2)
    res[0] = 0
    return res


def get_nu_tilde_init():
    if Re_tau_round == 10000:
        # nu_tilde_init = K * Y
        # nuT_data = get_nuT( nu_tilde_init)
        nuT_data = (-data["uv"] / data["dUdy"]) / Re_tau
    else:
        nuT_data = (-data["uv"] / data["dUdy"]) / Re_tau
    tck = splrep(Y_data, nuT_data, k=5)
    return splev(Y, tck)


def get_U_init():
    if Re_tau_round == 10000:
        Udata = data["U"]
        # instantiating as log law
        # Udata = np.zeros_like( Yp )
        # switch = 8
        # Udata[1:switch] = K * Yp[1:switch]
        # Udata[switch:] = np.log( Yp[switch:] ) / K + 5.1
    else:
        Udata = data["U"]
    tck = splrep(Y_data, Udata, k=5)
    return splev(Y, tck)


def get_laminar_solution():
    return dpdx / nu * (Y**2 - 2 * Y * Y[-1]) / 2
