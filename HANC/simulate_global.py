import time

import numpy as np
import numba as nb

from EconModel import jit
from consav import linear_interp
from consav.misc import elapsed
from GEModelTools import find_i_and_w_1d_1d, simulate_hh_forwards

import blocks
import root_finding

def simulate_global(model,do_print=False,ini_D=None,ini_K_lag=None,ini_inv_lag=None,simT=None,market_clearing=True):

    t0 = time.time()

    if ini_D is None: ini_D = model.ss.D
    if ini_K_lag is None: ini_K_lag = model.ss.K
    if ini_inv_lag is None: ini_inv_lag = model.ss.inv
    if simT is None: simT = model.par.simT

    with jit(model) as model:

        par = model.par
        ss = model.ss
        sim = model.sim
        KS = model.KS

        simulate_global_(model,par,KS,sim,ss.z_trans,ini_D,ini_K_lag,ini_inv_lag,simT,market_clearing)

    if do_print: print(f'model simulated in {elapsed(t0)}')

def simulate_global_(model,par,KS,sim,z_trans,ini_D,ini_K_lag,ini_inv_lag,simT,market_clearing):

    z_trans_T = np.transpose(z_trans,axes=(0,2,1)).copy()

    for t in range(simT):
        
        # a. technology
        Z = sim.Z[t]

        # b. lagged values and update distribution
        if t == 0:
            sim.D[0] = ini_D
            K_lag = ini_K_lag
            inv_lag = ini_inv_lag
        else:
            find_i_and_w_1d_1d(sim.a[t-1],par.a_grid,sim.pol_indices[t],sim.pol_weights[t])
            simulate_hh_forwards(sim.D[t-1],sim.pol_indices[t],sim.pol_weights[t],z_trans_T,sim.D[t])
            K_lag = sim.K[t-1]
            inv_lag = sim.inv[t-1]

        # c. find policy from m to a
        m_t,a_t = interp_am(par,KS,Z,K_lag,inv_lag)

        # d. PLM
        sim.PLM_K[t],sim.PLM_q[t],sim.PLM_foc_inv_term[t] = model.evaluate_PLM(Z,K_lag,inv_lag)
        q_guess = sim.PLM_q[t]
        foc_inv_term = sim.PLM_foc_inv_term[t]

        # e. determine prices
        if not market_clearing:
            
            q = sim.q[t]
            bond_clearing_obj(q,par,sim,t,m_t,a_t,Z,K_lag,inv_lag,foc_inv_term)

        else:
            # print(t)
            # print(f'{q_guess}')
            # print(Z,K_lag,inv_lag)
            q = root_finding.newton_secant(bond_clearing_obj,q_guess,args=(par,sim,t,m_t,a_t,Z,K_lag,inv_lag,foc_inv_term),tol=par.tol_solve_clearing)
            bond_clearing_obj(q,par,sim,t,m_t,a_t,Z,K_lag,inv_lag,foc_inv_term)
        
        # f. aggregate
        sim.K[t] = sim.A_hh[t]

        #print(sim.K[t],sim.r[t],sim.w[t],sim.u[t])

def bond_clearing_obj(q,par,sim,t,m_t,a_t,Z,K_lag,inv_lag,foc_inv_term):

    sim.q[t] = q
    # print(f'  {q}')

    # a. household inputs
    sim.r[t],sim.rk[t],sim.w[t],sim.u[t] = blocks.r_rk_w_u(par,Z,K_lag,1.0,sim.q[t])

    # b. household saving
    sim.a[t] = interp_a_wr(par,m_t,a_t,sim.w[t],sim.r[t])
    sim.A_hh[t] = np.sum(sim.a[t]*sim.D[t])

    # d. implied investment
    sim.A_hh[t] = np.sum(sim.a[t]*sim.D[t])
    inv = sim.inv[t] = np.fmax(sim.A_hh[t]-(1-par.delta)*K_lag,1e-4)

    # e. return error in foc inv
    sim.foc_inv[t] = blocks.foc_inv_func(par,inv_lag,q,inv,foc_inv_term)
    return sim.foc_inv[t]

@nb.njit
def interp_am(par,KS,Z,K_lag,inv_lag):
    """ interpolate a and m in Z, K_lag and inv_lag """

    m_t = np.zeros((par.Nfix,par.Nz,par.Na))
    a_t = np.zeros((par.Nfix,par.Nz,par.Na))

    # a. prepare interpolation
    i_Z = linear_interp.binary_search(0,par.Z_grid.size,par.Z_grid,Z)
    i_K_lag = linear_interp.binary_search(0,par.K_grid.size,par.K_grid,K_lag)
    i_inv_lag = linear_interp.binary_search(0,par.inv_grid.size,par.inv_grid,inv_lag)

    nom_1_left = par.Z_grid[i_Z+1]-Z
    nom_1_right = Z-par.Z_grid[i_Z]

    nom_2_left = par.K_grid[i_K_lag+1]-K_lag
    nom_2_right = K_lag-par.K_grid[i_K_lag]

    nom_3_left = par.inv_grid[i_inv_lag+1]-inv_lag
    nom_3_right = inv_lag-par.inv_grid[i_inv_lag]

    denom = (par.Z_grid[i_Z+1]-par.Z_grid[i_Z])*(par.K_grid[i_K_lag+1]-par.K_grid[i_K_lag])*(par.inv_grid[i_inv_lag+1]-par.inv_grid[i_inv_lag])

    # b. interpolate
    for i_fix in nb.prange(par.Nfix):
        for i_z in nb.prange(par.Nz):

            a_t[i_fix,i_z,:] = 0.0
            m_t[i_fix,i_z,:] = 0.0
            for k1 in range(2):
                nom_1 = nom_1_left if k1 == 0 else nom_1_right
                for k2 in range(2):
                    nom_2 = nom_2_left if k2 == 0 else nom_2_right       
                    for k3 in range(2):
                        nom_3 = nom_3_left if k3 == 0 else nom_3_right               
                        nom_fac = nom_1*nom_2*nom_3
                        a_t[i_fix,i_z,:] += nom_fac*KS.a[i_fix,i_z,i_Z+k1,i_K_lag+k2,i_inv_lag+k3,:]
                        m_t[i_fix,i_z,:] += nom_fac*KS.m[i_fix,i_z,i_Z+k1,i_K_lag+k2,i_inv_lag+k3,:]

            a_t[i_fix,i_z,:] /= denom
            m_t[i_fix,i_z,:] /= denom

    return m_t,a_t

@nb.njit
def interp_a_wr(par,m_t,a_t,w,r):
    """ interpolate a from m given w and r """
    
    a_wr_t = np.zeros((par.Nfix,par.Nz,par.Na))

    for i_fix in nb.prange(par.Nfix):
        for i_z in nb.prange(par.Nz):
                
            z = par.z_grid[i_z]
            m = (1+r)*par.a_grid + w*z

            linear_interp.interp_1d_vec(m_t[i_fix,i_z],a_t[i_fix,i_z],m,a_wr_t[i_fix,i_z,:])

    return a_wr_t