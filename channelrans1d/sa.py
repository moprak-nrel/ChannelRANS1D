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
    cv1: float = 7.1

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
        self.y_star = self.Y_data

        # Grid properties
        self.ny = len(self.y_star)
        self.y_plus = self.y_star * self.Re_tau

        # Physics constants
        self.nu = 1.0 / self.Re_tau

        # Model constants
        self.sa_coeffs = sa_coeffs

    def get_spline_rep_U_plus(self, U_plus) -> interp.CubicSpline:
        """Get cubic spline representation for velocity U."""
        U_plus[0] = 0
        cs = interp.CubicSpline(self.y_star, U_plus, bc_type=("not-a-knot", "clamped"))
        return cs

    def get_spline_rep_nu_star(self, X) -> interp.CubicSpline:
        """Get cubic spline representation for nu_tilde."""
        X[0] = 0
        cs = interp.CubicSpline(self.y_star, X, bc_type=("not-a-knot", "clamped"))
        return cs

    def get_y_der(self, tck: interp.CubicSpline):
        """Get first derivative with respect to y."""
        res = tck(self.y_star, 1)
        res[-1] = 0
        return res

    def get_yy_der(self, tck: interp.CubicSpline):
        """Get second derivative with respect to y."""
        res = tck(self.y_star, 2)
        return res

    def multiplicative_error(self, nuT_star):
        """Add a multiplicative error to nuT star"""
        return nuT_star * (1.0 + self.sa_coeffs.err)

    def get_spatial_derivatives(self, state):
        """Compute spatial derivatives [dyU+, dyyU+, dynu*, dyynu*, dynuT*]."""
        U_plus = state[: self.ny]
        nu_tilde_star = state[self.ny :]
        utck = self.get_spline_rep_U_plus(U_plus)
        ntck = self.get_spline_rep_nu_star(nu_tilde_star)
        dyU_plus = self.get_y_der(utck)
        dyyU_plus = self.get_yy_der(utck)
        dynu_star = self.get_y_der(ntck)
        dyynu_star = self.get_yy_der(ntck)

        nuT_star = self.get_nuT_star(nu_tilde_star)
        nuT_star = self.multiplicative_error(nuT_star)
        nuT_tck = self.get_spline_rep_nu_star(nuT_star)
        dynuT_star = self.get_y_der(nuT_tck)

        return dyU_plus, dyyU_plus, dynu_star, dyynu_star, dynuT_star

    def get_dXdt(self, state):
        """Compute time derivative residuals"""
        nu_tilde_star = state[self.ny :]
        dyU_plus, dyyU_plus, dynu_star, dyynu_star, dynuT_star = self.get_spatial_derivatives(state)

        dUdt_plus = self.get_dUdt_plus(dyU_plus, dyyU_plus, nu_tilde_star, dynuT_star)
        dUdt_plus[0] = 0
        dnudt_star = self.get_dnudt_star(dyU_plus, nu_tilde_star, dynu_star, dyynu_star)
        return np.hstack([dUdt_plus, dnudt_star])

    def get_dUdt_plus(self, dyU_plus, dyyU_plus, nu_tilde_star, dynuT_star):
        """
        Compute non-dimensional time derivative of velocity U plus
        d(U+)/dt = 1 + (1/Re_tau + nu_t*) * d^2(U+)/dy*^2 + d(nu_t*)/dy* * dU*/dy*
        """
        nuT_star = self.get_nuT_star(nu_tilde_star)
        nuT_star = self.multiplicative_error(nuT_star)
        res = 1.0 + (self.nu + nuT_star) * dyyU_plus + dynuT_star * dyU_plus
        return res

    def get_nuT_star(self, nu_tilde_star):
        """Compute non-dimensional eddy viscosity nu_t_star from nu_tilde_star.
         nu_t* = nu_tilde* * f_v1
        """
        chi = nu_tilde_star / self.nu
        chi3 = chi**3
        fv1 = chi3 / (chi3 + self.sa_coeffs.cv1**3)
        return nu_tilde_star * fv1

    def get_Stilde_star(self, dyU_plus, nu_tilde_star):
        """
        Compute non-dimensional Stilde* from dyU* and nu_tilde*.
        Stilde* = S* + (nu_tilde* / (kappa * y*)^2) * f_v2
        """
        Stilde_star = np.zeros_like(self.y_star)

        # S* = |dU*/dy*|
        S_star = np.abs(dyU_plus)

        # Compute eddy viscosity and apply multiplicative error
        nuT_star = self.get_nuT_star(nu_tilde_star)
        nuT_star = self.multiplicative_error(nuT_star)

        chi = nu_tilde_star / self.nu
        chi3 = chi**3
        fv1 = chi3 / (chi3 + self.sa_coeffs.cv1**3)
        # Multiply by self.nu so we use the multiplicative error
        fv2 = 1.0 - (nu_tilde_star / (self.nu + nuT_star))

        # Stilde* = S* + (nu_tilde* / (kappa * y*)^2) * f_v2
        Stilde_star[1:] = (
            S_star[1:]
            + (nu_tilde_star[1:] / (self.sa_coeffs.kappa * self.y_star[1:])**2) * fv2[1:]
        )

        return Stilde_star

    def get_Pnu_star(self, dyU_plus, nu_tilde_star):
        """
        Compute the non-dimensional production term P_nu*
        P_nu* = c_b1 * S_tilde* * nu_tilde*
        """
        return self.sa_coeffs.cb1 * self.get_Stilde_star(dyU_plus, nu_tilde_star) * nu_tilde_star

    def get_r(self, dyU_plus, nu_tilde_star):
        """
        Compute the dimensionless parameter r.
        r = nu_tilde* / (S_tilde* * (kappa * y*)^2)
        """
        r = np.zeros_like(self.y_star)
        S_tilde_star_interior = self.get_Stilde_star(dyU_plus, nu_tilde_star)[1:]
        denom = S_tilde_star_interior * (self.sa_coeffs.kappa * self.y_star[1:]) ** 2
        r[1:] = nu_tilde_star[1:] / denom

        return r

    def get_fw(self, dyU_plus, nu_tilde_star):
        r = self.get_r(dyU_plus, nu_tilde_star)
        g = r + self.sa_coeffs.cw2 * (r**6 - r)
        res = g * ((1 + self.sa_coeffs.cw3**6) / (self.sa_coeffs.cw3**6 + g**6.0)) ** (1.0 / 6.0)
        # return np.minimum(res, 2.00517475)
        return res

    def get_Dnu_star(self, dyU_plus, nu_tilde_star):
        """
        Compute the non-dimensional destruction term D_nu*
        D_nu* = c_w1 * f_w * (nu_tilde* / y*)^2
        """
        Dnu_star = np.zeros_like(self.y_star)
        fw = self.get_fw(dyU_plus, nu_tilde_star)
        Dnu_star[1:] = (
            self.sa_coeffs.cw1
            * fw[1:]
            * (nu_tilde_star[1:] / self.y_star[1:]) ** 2
        )
        return Dnu_star

    def get_Tnu_star(self, nu_tilde_star, dynu_star, dyynu_star):
        """
        Compute the non-dimensional diffusion term T_star.
        T* = (1/sigma) * [ (1/Re_tau + nu_tilde*) * d^2(nu_tilde*)/dy*^2 + (1 + c_b2) * (d(nu_tilde*)/dy*)^2 ]
        """
        Tnu_star = np.zeros_like(self.y_star)
        
        # Note: self.nu represents 1 / Re_tau
        Tnu_star[1:] = (1.0 / self.sa_coeffs.sigmav) * (
            (self.nu + nu_tilde_star[1:]) * dyynu_star[1:] 
            + (1.0 + self.sa_coeffs.cb2) * dynu_star[1:]**2
        )
        return Tnu_star

    def get_dnudt_star(self, dyU_plus, nu_tilde_star, dynu_star, dyynu_star):
        """
        Compute the non-dimensional time derivative of nu_tilde*.
        d(nu_tilde*)/dt* = P* - D* + T*
        """
        res = (
            self.get_Pnu_star(dyU_plus, nu_tilde_star)
            - self.get_Dnu_star(dyU_plus, nu_tilde_star)
            + self.get_Tnu_star(nu_tilde_star, dynu_star, dyynu_star)
        )
        res[0] = 0
        return res

    def get_nu_tilde_star_init(self):
        """Get initial condition for nu_tilde from data, this is just set to nu_t."""
        nuT_star_data = (-self.data["uv"] / self.data["dUdy"]) / self.Re_tau
        return nuT_star_data

    def get_U_plus_init(self):
        """Get initial condition for velocity U from data."""
        Udata_star = self.data["U"]
        return Udata_star
