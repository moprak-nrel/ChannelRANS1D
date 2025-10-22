import numpy as np


def read_data(Re):
    data = {}
    file1 = np.loadtxt("data/LM_Channel_%d_mean_prof.dat" % Re, comments="%")
    file2 = np.loadtxt("data/LM_Channel_%d_vel_fluc_prof.dat" % Re, comments="%")
    data["Y"] = file1[:, 0]
    data["U"] = file1[:, 2]
    data["dUdy"] = file1[:, 3]
    data["uv"] = file2[:, 5]
    return data
