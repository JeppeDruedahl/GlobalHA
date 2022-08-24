import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

from root_finding import brentq

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    # a. a
    par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)

    # b. e
    par.z_grid[:],ss.z_trans[0,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    for i_fix in range(par.Nfix):
        ss.Dz[i_fix,:] = e_ergodic/par.Nfix
        ss.Dbeg[i_fix,:,0] = ss.Dz[i_fix,:]
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    va = np.zeros((par.Nfix,par.Nz,par.Na))
    
    y = ss.w*par.z_grid

    m = np.zeros((par.Nfix,par.Nz,par.Na))
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):
            m[i_fix,i_z,:] = (1+ss.r)*par.a_grid + y[i_z]

    a = 0.90*m # pure guess
    c = m - a
    
    va[:,:,:] = (1+ss.r)*c**(-par.sigma)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@va

def find_ss(model,K_min=6.0,K_max=14.0,tol=1e-12,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. exogenous and targets
    ss.L = 1.0 # normalization
    ss.u = par.u_target # target
    ss.q = 1.0

    # b. find capital
    def asset_market_clearing(K):

        ss.A = ss.K = K

        # i. TFP
        ss.Z = par.chi1/(ss.K**par.alpha*ss.L**(1-par.alpha))

        # ii. prices
        ss.rk = par.alpha*ss.u*ss.Z*(ss.K/ss.L)**(par.alpha-1)
        ss.r = ss.rk-par.delta
        ss.w = (1-par.alpha)*ss.u*ss.Z*(ss.K/ss.L)**par.alpha

        # iii. household behavior
        model.solve_hh_ss() 
        model.simulate_hh_ss()

        ss.A_hh = np.sum(ss.a*ss.D)

        return ss.A_hh-ss.A

    fa = asset_market_clearing(K_min)
    if do_print: print(f'K = {K_min:12.8f} -> A_hh-A = {fa:12.8f}')

    fb = asset_market_clearing(K_max)
    if do_print: print(f'K = {K_max:12.8f} -> A_hh-A = {fb:12.8f}')

    if do_print: print(f'search for K')
    brentq(asset_market_clearing,K_min,K_max,fa=fa,fb=fb,xtol=tol,rtol=tol,
        do_print=do_print,varname='K',funcname='A_hh-A')

    # c. remaining
    ss.Y = ss.Z*ss.u*ss.K**par.alpha*ss.L**(1-par.alpha)
    ss.inv = par.delta*ss.K
    ss.C = ss.Y - ss.inv
    ss.C_hh = np.sum(ss.c*ss.D)

    # d. print
    if do_print:

        print(f'Implied K = {ss.K:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied Z = {ss.Z:6.3f}')
        print(f'Implied u = {ss.u:6.3f}')
        print(f'Implied r = {ss.r:6.3f}')
        print(f'Implied w = {ss.w:6.3f}')
        print(f'Implied K/Y = {ss.K/ss.Y:6.3f}') 
        print(f'Discrepancy in A-A_hh = {ss.A-ss.A_hh:16.12f}')
        print(f'Discrepancy in C-C_hh = {ss.C-ss.C_hh:16.12f}')