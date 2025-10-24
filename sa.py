import numpy as np
import scipy.interpolate as interp

import ke


class SpalartAllmaras:
    """Spalart-Allmaras turbulence model implementation."""

    def __init__(self, Re_tau_round=5200, params_override={}):
        """Initialize the Spalart-Allmaras model."""
        # Reynolds number lookup table
        self.Re_tau_table = {
            180: 182.088,
            550: 543.496,
            1000: 1000.512,
            2000: 1994.756,
            5200: 5185.897,
        }

        self.Re_tau_round = Re_tau_round
        self.Re_tau = self.Re_tau_table[Re_tau_round]

        # Load data
        self.data = ke.read_data(Re_tau_round)
        self.Y_data = self.data["Y"]
        self.Y = self.Y_data

        # Grid properties
        self.ny = len(self.Y)
        self.Yp = self.Y * self.Re_tau

        # Model constants
        self.kappa = 0.41
        self.nu = 1.0 / self.Re_tau
        self.params = {
            "sigmav": 2.0 / 3.0,
            "cb1": 0.1355,
            "cb2": 0.622,
            "cw2": 0.3,
            "cw3": 2,
        }
        for k in params_override:
            self.params[k] = params_override[k]
        self.sigmav = self.params["sigmav"]
        self.cb1 = self.params["cb1"]
        self.cb2 = self.params["cb2"]
        self.cw2 = self.params["cw2"]
        self.cw3 = self.params["cw3"]
        self.cw1 = self.cb1 / self.kappa**2 + (1 + self.cb2) / self.sigmav

    def get_spline_rep_U(self, U):
        """Get cubic spline representation for velocity U."""
        U[0] = 0
        cs = interp.CubicSpline(self.Y, U, bc_type=("not-a-knot", "clamped"))
        return cs

    def get_spline_rep_nu(self, X):
        """Get cubic spline representation for nu_tilde."""
        X[0] = 0
        cs = interp.CubicSpline(self.Y, X, bc_type=("not-a-knot", "clamped"))
        return cs

    def get_y_der(self, tck):
        """Get first derivative with respect to y."""
        res = tck(self.Y, 1)
        res[-1] = 0
        return res

    def get_yy_der(self, tck):
        """Get second derivative with respect to y."""
        res = tck(self.Y, 2)
        return res

    def get_dXdt(self, state):
        """Compute time derivatives for the state vector [U,\nu_t]."""
        U = state[: self.ny]
        nu_tilde = state[self.ny :]
        utck = self.get_spline_rep_U(U)
        ntck = self.get_spline_rep_nu(nu_tilde)
        dyU = self.get_y_der(utck)
        dyyU = self.get_yy_der(utck)
        dynu = self.get_y_der(ntck)
        dyynu = self.get_yy_der(ntck)

        dUdt = self.get_dUdt(U, dyU, dyyU, nu_tilde)
        dUdt[0] = 0
        dnudt = self.get_dnudt(U, dyU, nu_tilde, dynu, dyynu)
        return np.hstack([dUdt, dnudt])

    def get_dUdt(self, U, dyU, dyyU, nu_tilde):
        """Compute time derivative of velocity U."""
        nuT = self.get_nuT(nu_tilde)
        ntck = self.get_spline_rep_nu(nuT)
        res = 1 + (self.nu + nuT) * dyyU + self.get_y_der(ntck) * dyU
        return res

    def get_nuT(self, nu_tilde):
        """Compute nu_t from nu_tilde."""
        temp = (nu_tilde / self.nu) ** 3
        return nu_tilde * (temp / (temp + 7.1**3))

    def get_Stilde(self, dyU, nu_tilde):
        nuT = self.get_nuT(nu_tilde)
        S_tilde = np.zeros_like(self.Y)
        S_tilde[1:] = (
            dyU[1:]
            + (-(nu_tilde[1:] ** 2) / (self.nu + nuT[1:]) + nu_tilde[1:])
            / (self.kappa * self.Y[1:]) ** 2
        )
        return S_tilde

    def get_Pnu(self, dyU, nu_tilde):
        return self.cb1 * self.get_Stilde(dyU, nu_tilde) * nu_tilde

    def get_r(self, dyU, nu_tilde):
        r = nu_tilde / (self.get_Stilde(dyU, nu_tilde) * (self.kappa * self.Y) ** 2)
        r[0] = 0
        return r

    def get_g(self, r):
        return r + self.cw2 * (r**6 - r)

    def get_f(self, r):
        g = self.get_g(r)
        res = g * ((1 + self.cw3**6) / (self.cw3**6 + g**6.0)) ** (1.0 / 6.0)
        # return np.minimum(res, 2.00517475)
        return res

    def get_Enu(self, dyU, nu_tilde):
        Enu = (
            self.cw1 * (nu_tilde / self.Y) ** 2 * self.get_f(self.get_r(dyU, nu_tilde))
        )
        Enu[0] = 0
        return Enu

    def get_dnudt(self, U, dyU, nu_tilde, dynu, dyynu):
        """Compute time derivative of nu_tilde."""
        res = (
            self.get_Pnu(dyU, nu_tilde)
            - self.get_Enu(dyU, nu_tilde)
            + 1.0
            / self.sigmav
            * ((self.nu + nu_tilde) * dyynu + (1 + self.cb2) * dynu**2)
        )
        res[0] = 0
        return res

    def get_nu_tilde_init(self):
        """Get initial condition for nu_tilde from data, this is just set to nu_t."""
        nuT_data = (-self.data["uv"] / self.data["dUdy"]) / self.Re_tau
        return nuT_data

    def get_U_init(self):
        """Get initial condition for velocity U from data."""
        Udata = self.data["U"]
        return Udata
