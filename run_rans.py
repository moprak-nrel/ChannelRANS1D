import numpy as np
import numpy.linalg as LA
from scipy.integrate import ode
import matplotlib.pyplot as plt
import os.path

from sa import *
# from sa_full import *
from rans import *
# from backward_euler import *

dt = 1e-2
steps = 10
init = np.hstack([get_U_init(), get_nu_tilde_init()])
init_time = 0
if( os.path.isfile('%d-last.npy'%Re_tau_round) ):
    print("Using restart files....")
    init = np.load('%d-last.npy'%Re_tau_round)
    init_time = np.load('%d-final-time.npy'%Re_tau_round)
    print(init_time)

states = time_march(init, steps, dt)

fig = plt.figure()
plt.plot(Y,states[int(steps/2-1)][:N],'g-', label='RANS half-simulation')
plt.plot(Y,states[steps-1][:N],'b-', label='RANS')
plt.plot(Y,get_U_init(),'r--',label='DNS')
plt.ylabel(r'$U$')
plt.xlabel(r'$\widetilde{y}$')
plt.legend( loc='best')
fig.tight_layout()
plt.savefig('figs/%d-U.pdf'%Re_tau_round)

fig = plt.figure()
plt.loglog(Yp[1:N],states[steps-1][1:N],'b-', label='RANS')
plt.loglog(Yp[1:N],get_U_init()[1:N],'r--',label='DNS')
plt.ylabel(r'$U$')
plt.xlabel(r'$y^+$')
plt.legend( loc='best')
fig.tight_layout()
plt.savefig('figs/%d-U-loglog.pdf'%Re_tau_round)

fig = plt.figure()
plt.semilogx(Yp[1:N],states[steps-1][1:N],'b-', label='RANS')
plt.semilogx(Yp[1:N],get_U_init()[1:N],'r--',label='DNS')
plt.ylabel(r'$U$')
plt.xlabel(r'$y^+$')
plt.legend( loc='best')
fig.tight_layout()
plt.savefig('figs/%d-U-semilog.pdf'%Re_tau_round)

fig = plt.figure()
plt.plot(Y,get_nuT(states[steps-1][N:]),'b-', label='RANS')
plt.plot(Y,get_nu_tilde_init(),'r--',label='DNS')
plt.ylabel(r'$\nu_\tau$')
plt.xlabel(r'$\widetilde{y}$')
plt.legend( loc='best')
fig.tight_layout()
plt.savefig('figs/%d-nu_tilde.pdf'%Re_tau_round)

fig = plt.figure()
plt.semilogx(Yp[1:N],get_nuT(states[steps-1][N+1:]),'b-', label='RANS')
plt.semilogx(Yp[1:N],get_nu_tilde_init()[1:N],'r--',label='DNS')
plt.ylabel(r'$\nu_\tau$')
plt.xlabel(r'$y^+$')
plt.legend( loc='best')
fig.tight_layout()
plt.savefig('figs/%d-nu_tilde-semilog.pdf'%Re_tau_round)

np.save('%d-last'%Re_tau_round,states[-1])
np.savetxt('data/%d-final-state.dat'%Re_tau_round,np.vstack([Yp,states[-1][:N],get_nuT(states[-1][N:])]).T)
finaltime = steps*dt + init_time
print("final time:", finaltime)
np.save('%d-final-time'%Re_tau_round,steps*dt + init_time )
