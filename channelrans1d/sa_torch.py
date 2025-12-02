import torch
import torch.nn as nn
from channelrans1d import ke

class SpalartAllmarasTorch(nn.Module):
    def set_params(self, params):
        with torch.no_grad():
            self.params.copy_(params.float())
    def __init__(self, Re_tau_round=5200, params=None):
        super().__init__()
        self.Re_tau_table = {
            180: 182.088,
            550: 543.496,
            1000: 1000.512,
            2000: 1994.756,
            5200: 5185.897,
        }
        self.Re_tau = self.Re_tau_table[Re_tau_round]
        self.data = ke.read_data(Re_tau_round)
        self.Y = torch.tensor(self.data["Y"], dtype=torch.float32)
        self.ny = len(self.Y)
        self.params = nn.Parameter(params.clone().detach().float())
        self.kappa = 0.41
        self.nu = 1.0 / self.Re_tau

    @property
    def cw1(self):
        return self.params[1] / self.kappa**2 + (1 + self.params[2]) / self.params[0]

    def get_dUdt(self, U, dyU, dyyU, nu_tilde, dynutT):
        nuT = self.get_nuT(nu_tilde)
        return 1 + (self.nu + nuT) * dyyU + dynutT * dyU

    def get_nuT(self, nu_tilde):
        temp = (nu_tilde / self.nu) ** 3
        return nu_tilde * (temp / (temp + 7.1 ** 3))

    def get_Pnu(self, dyU, nu_tilde):
        nuT = self.get_nuT(nu_tilde)
        S_tilde = torch.zeros_like(self.Y)
        S_tilde[1:] = (
            dyU[1:]
            + (-(nu_tilde[1:] ** 2) / (self.nu + nuT[1:]) + nu_tilde[1:])
            / (self.kappa * self.Y[1:]) ** 2
        )
        return self.params[1] * S_tilde * nu_tilde

    def get_Enu(self, dyU, nu_tilde):
        nuT = self.get_nuT(nu_tilde)
        S_tilde = torch.zeros_like(self.Y)
        S_tilde[1:] = (
            dyU[1:]
            + (-(nu_tilde[1:] ** 2) / (self.nu + nuT[1:]) + nu_tilde[1:])
            / (self.kappa * self.Y[1:]) ** 2
        )
        r = nu_tilde / (S_tilde * (self.kappa * self.Y) ** 2)
        r[0] = 0
        g = r + self.params[3] * (r ** 6 - r)
        f = g * ((1 + self.params[4] ** 6) / (self.params[4] ** 6 + g ** 6.0)) ** (1.0 / 6.0)
        Enu = self.cw1 * (nu_tilde / self.Y) ** 2 * f
        Enu[0] = 0
        return Enu

    def get_dnudt(self, U, dyU, nu_tilde, dynu, dyynu):
        return (
            self.get_Pnu(dyU, nu_tilde)
            - self.get_Enu(dyU, nu_tilde)
            + 1.0 / self.params[0] * ((self.nu + nu_tilde) * dyynu + (1 + self.params[2]) * dynu ** 2)
        )

    def get_dXdt(self, state):
        ny = self.ny
        U = state[0:ny]
        nu_tilde = state[ny:2*ny]
        nuT = state[2*ny:3*ny]
        dyU = state[3*ny:4*ny]
        dyyU = state[4*ny:5*ny]
        dynutT = state[5*ny:6*ny]
        dynu = state[6*ny:7*ny]
        dyynu = state[7*ny:8*ny]
        dUdt = self.get_dUdt(U, dyU, dyyU, nu_tilde, dynutT)
        dUdt[0] = 0
        dnudt = self.get_dnudt(U, dyU, nu_tilde, dynu, dyynu)
        return torch.cat([dUdt, dnudt])
