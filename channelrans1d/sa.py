import numpy as np
import scipy.interpolate as interp
from dataclasses import dataclass
from channelrans1d import ke

@dataclass
class SACoefficients:
    """
    From https://en.wikipedia.org/wiki/Spalart–Allmaras_turbulence_model
    """
    sigmav: float = 2.0 / 3.0
    cb1: float = 0.1355
    cb2: float = 0.622
    kappa: float = 0.41
    cw2: float = 0.3
    cw3: float = 2.0
    err: float = 0.0
    #cv1: float = 7.1

    @property
    def cw1(self) -> float:
        return self.cb1 / self.kappa**2 + (1.0 + self.cb2) / self.sigmav


class SpalartAllmaras:
    """Spalart-Allmaras turbulence model implementation."""

    def __init__(self, Re_tau_round=5200, sa_coeffs: SACoefficients=SACoefficients()):
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

        # Physics constants
        self.nu = 1.0 / self.Re_tau

        # Model constants
        self.kappa = 0.41
        self.sa_coeffs = sa_coeffs

    def get_spline_rep_U(self, U) -> interp.CubicSpline:
        """Get cubic spline representation for velocity U."""
        U[0] = 0
        cs = interp.CubicSpline(self.Y, U, bc_type=("not-a-knot", "clamped"))
        return cs

    def get_spline_rep_nu(self, X) -> interp.CubicSpline:
        """Get cubic spline representation for nu_tilde."""
        X[0] = 0
        cs = interp.CubicSpline(self.Y, X, bc_type=("not-a-knot", "clamped"))
        return cs

    def get_y_der(self, tck: interp.CubicSpline):
        """Get first derivative with respect to y."""
        res = tck(self.Y, 1)
        res[-1] = 0
        return res

    def get_yy_der(self, tck: interp.CubicSpline):
        """Get second derivative with respect to y."""
        res = tck(self.Y, 2)
        return res

    def multiplicative_error(self, nuT):
        """Add a multiplicative error to nuT"""
        return nuT * (1.0 + self.sa_coeffs.err)

    def get_spatial_derivatives(self, state):
        """Compute spatial derivatives [dyU, dyyU, dynu, dyynu, dynuT]."""
        U = state[: self.ny]
        nu_tilde = state[self.ny :]
        utck = self.get_spline_rep_U(U)
        ntck = self.get_spline_rep_nu(nu_tilde)
        dyU = self.get_y_der(utck)
        dyyU = self.get_yy_der(utck)
        dynu = self.get_y_der(ntck)
        dyynu = self.get_yy_der(ntck)

        nuT = self.get_nuT(nu_tilde)
        nuT = self.multiplicative_error(nuT)
        nuT_tck = self.get_spline_rep_nu(nuT)
        dynuT = self.get_y_der(nuT_tck)

        return dyU, dyyU, dynu, dyynu, dynuT

    def get_dXdt(self, state):
        """Compute time derivatives for the state vector [U,\nu_t]."""
        U = state[: self.ny]
        nu_tilde = state[self.ny :]
        dyU, dyyU, dynu, dyynu, dynuT = self.get_spatial_derivatives(state)

        dUdt = self.get_dUdt(U, dyU, dyyU, nu_tilde, dynuT)
        dUdt[0] = 0
        dnudt = self.get_dnudt(U, dyU, nu_tilde, dynu, dyynu)
        return np.hstack([dUdt, dnudt])

    def get_dUdt(self, U, dyU, dyyU, nu_tilde, dynuT):
        """Compute time derivative of velocity U."""
        nuT = self.get_nuT(nu_tilde)
        nuT = self.multiplicative_error(nuT)
        res = 1 + (self.nu + nuT) * dyyU + dynuT * dyU
        return res

    def get_nuT(self, nu_tilde):
        """Compute nu_t from nu_tilde."""
        temp = (nu_tilde / self.nu) ** 3
        return nu_tilde * (temp / (temp + 7.1**3))

    def get_Stilde(self, dyU, nu_tilde):
        nuT = self.get_nuT(nu_tilde)
        nuT = self.multiplicative_error(nuT)
        S_tilde = np.zeros_like(self.Y)
        S_tilde[1:] = (
            dyU[1:]
            + (-(nu_tilde[1:] ** 2) / (self.nu + nuT[1:]) + nu_tilde[1:])
            / (self.sa_coeffs.kappa * self.Y[1:]) ** 2
        )
        return S_tilde

    def get_Pnu(self, dyU, nu_tilde):
        return self.sa_coeffs.cb1 * self.get_Stilde(dyU, nu_tilde) * nu_tilde

    def get_r(self, dyU, nu_tilde):
        r = np.zeros_like(self.Y)
        S_tilde_interior = self.get_Stilde(dyU, nu_tilde)[1:]
        denom = S_tilde_interior * (self.sa_coeffs.kappa * self.Y[1:]) ** 2
        r[1:] = nu_tilde[1:] / denom
        r[0] = 0
        return r

    def get_g(self, r):
        return r + self.sa_coeffs.cw2 * (r**6 - r)

    def get_f(self, r):
        g = self.get_g(r)
        res = g * ((1 + self.sa_coeffs.cw3**6) / (self.sa_coeffs.cw3**6 + g**6.0)) ** (1.0 / 6.0)
        # return np.minimum(res, 2.00517475)
        return res

    def get_Enu(self, dyU, nu_tilde):
        Enu = np.zeros_like(self.Y)
        f_r_interior = self.get_f(self.get_r(dyU, nu_tilde))[1:]
        Enu[1:] = (
            self.sa_coeffs.cw1 * (nu_tilde[1:] / self.Y[1:]) ** 2 * f_r_interior
        )
        Enu[0] = 0
        return Enu

    def get_dnudt(self, U, dyU, nu_tilde, dynu, dyynu):
        """Compute time derivative of nu_tilde."""
        res = (
            self.get_Pnu(dyU, nu_tilde)
            - self.get_Enu(dyU, nu_tilde)
            + 1.0
            / self.sa_coeffs.sigmav
            * ((self.nu + nu_tilde) * dyynu + (1 + self.sa_coeffs.cb2) * dynu**2)
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
