import numpy as np
from .paths import CHANNELRANS1D_DNS_DATA
import os

def read_data(Re):
    data = {}
    file1 = np.loadtxt(os.path.join(CHANNELRANS1D_DNS_DATA, f"LM_Channel_{Re}_mean_prof.dat"), comments="%")[::2]
    file2 = np.loadtxt(os.path.join(CHANNELRANS1D_DNS_DATA, f"LM_Channel_{Re}_vel_fluc_prof.dat"), comments="%")[::2]
    data["Y"] = file1[:, 0]
    data["U"] = file1[:, 2]
    data["dUdy"] = file1[:, 3]
    data["uv"] = file2[:, 5]
    return data
