import time

import numpy as np
import numba as nb

from EconModel import jit
from consav import linear_interp
from consav.misc import elapsed
from GEModelTools import find_i_and_w_1d_1d, simulate_hh_forwards

import blocks

def simulate_global(model,do_print=False,ini_D=None,ini_K_lag=None,simT=None):

    t0 = time.time()

    if ini_D is None: ini_D = model.ss.D
    if ini_K_lag is None: ini_K_lag = model.ss.K
    if simT is None: simT = model.par.simT

    with jit(model) as model:

        par = model.par
        ss = model.ss
        sim = model.sim
        KS = model.KS

        simulate_global_(par,KS,sim,ss.z_trans,ini_D,ini_K_lag,simT)

    if do_print: print(f'model simulated in {elapsed(t0)}')

@nb.njit
def simulate_global_(par,KS,sim,z_trans,ini_D,ini_K_lag,simT):

    z_trans_T = np.transpose(z_trans,axes=(0,2,1)).copy()

    for t in range(simT):
        
        # a. technology
        Z = sim.Z[t]

        # b. lagged values and update distribution
        if t == 0:
            sim.D[0] = ini_D
            K_lag = ini_K_lag
        else:
            find_i_and_w_1d_1d(sim.a[t-1],par.a_grid,sim.pol_indices[t],sim.pol_weights[t])
            simulate_hh_forwards(sim.D[t-1],sim.pol_indices[t],sim.pol_weights[t],z_trans_T,sim.D[t])
            K_lag = sim.K[t-1]

        # c. find policy
        sim.a[t] = interp_a(par,KS,K_lag,sim.Z[t])

        # d. aggregate
        sim.K[t] = np.sum(sim.a[t]*sim.D[t])
        sim.r[t],sim.rk[t],sim.w[t],sim.u[t] = blocks.r_rk_w_u(par,Z,K_lag,1.0)

@nb.njit
def interp_a(par,KS,K_lag,Z):

    a_t = np.zeros((par.Nbeta,par.Nz,par.Na))

    # a. prepare interpolation
    i_Z = linear_interp.binary_search(0,par.Z_grid.size,par.Z_grid,Z)
    i_K_lag = linear_interp.binary_search(0,par.K_grid.size,par.K_grid,K_lag)

    w_Z = (Z-par.Z_grid[i_Z])/(par.Z_grid[i_Z+1]-par.Z_grid[i_Z])
    w_K_lag = (K_lag-par.K_grid[i_K_lag])/(par.K_grid[i_K_lag+1]-par.K_grid[i_K_lag])
    
    # b. interpolate
    for i_beta in nb.prange(par.Nbeta):
        for i_z in nb.prange(par.Nz):

            # unpack
            a_ = KS.a[i_beta,i_z]

            # interpolate
            low_diff_K = a_[i_Z,i_K_lag+1,:]-a_[i_Z,i_K_lag,:]
            low_Z = a_[i_Z,i_K_lag,:] + low_diff_K*w_K_lag
            
            high_diff_K = a_[i_Z+1,i_K_lag+1,:]-a_[i_Z+1,i_K_lag,:]
            high_Z = a_[i_Z+1,i_K_lag,:] + high_diff_K*w_K_lag
        
            diff_Z = high_Z-low_Z
            
            a_t[i_beta,i_z,:] += low_Z + diff_Z*w_Z    

    return a_t