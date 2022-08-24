import time
import numpy as np
import numba as nb

from EconModel import jit
from consav import linear_interp
from consav.misc import elapsed

import blocks
import household_problem

##################
# initial values #
##################

def set_inital_values(model,do_print=False):
    """ set KS.v_a """

    t0 = time.time()

    with jit(model) as model:

        par = model.par
        ss = model.ss
        sim = model.sim
        KS = model.KS
        
        set_inital_values_(par,ss,sim,KS.v_a,KS.a,KS.c)

    if do_print: print(f'initial values found in {elapsed(t0)}')

@nb.njit(parallel=True)
def set_inital_values_(par,ss,sim,v_a,a,c):

    # a. common X
    X = np.zeros((par.simT-par.simBurn-1,3))
    X[:,0] = 1.0
    X[:,1] = sim.Z[par.simBurn+1:]
    X[:,2] = sim.K[par.simBurn:-1]

    # b. loop over idiosyncratic states
    for i_beta in nb.prange(par.Nbeta):
        for i_z in nb.prange(par.Nz):
            for i_a in nb.prange(par.Na):
                
                # regression
                y = np.zeros(X.shape[0])
                y[:] = sim.a[par.simBurn+1:,i_beta,i_z,i_a]
                coeffs = np.linalg.inv(X.T@X)@X.T@y

                # loop over aggregate states
                for i_Z in range(par.NZ):
                    for i_K_lag in range(par.NK):

                        K_lag = par.K_grid[i_K_lag]
                        Z = par.Z_grid[i_Z]

                        # i. a
                        a_pred = coeffs[0] + coeffs[1]*Z + coeffs[2]*K_lag
                        a[i_beta,i_z,i_Z,i_K_lag,i_a] = np.fmax(a_pred,par.a_grid[0])
                        
                        # ii. c and v_a
                        r,_,w,_ = blocks.r_rk_w_u(par,Z,K_lag,ss.L)
                        m = (1+r)*par.a_grid[i_a] + w*par.z_grid[i_z]
                        c[i_beta,i_z,i_Z,i_K_lag,i_a] = m - a[i_beta,i_z,i_Z,i_K_lag,i_a]
                        v_a[i_beta,i_z,i_Z,i_K_lag,i_a] = (1+r)*c[i_beta,i_z,i_Z,i_K_lag,i_a]**(-par.sigma)

##############
# households #
###############

def solve_hh_global(model,do_print=False):
    """ solve household problem with grids for global solution """
    
    t0 = time.time()

    with jit(model) as model:

        par = model.par
        ss = model.ss
        KS = model.KS
        it = 0

        while True:       

            old_a = KS.a.copy()

            # i. backwards step
            solve_hh_global_backwards(par,ss,KS.PLM_K,KS.v_a,KS.v_a,KS.a,KS.c)
            
            # ii. error
            max_abs_diff = np.max(np.abs(KS.a-old_a))
            
            # iii. stopping criterion
            if max_abs_diff < par.tol_solve_hh_global: break

            # iv. increment
            it += 1
            if it > par.max_iter_solve_hh_global: 
                raise ValueError('solve_hh_global(), too many iterations')

    if do_print: print(f'household problem solved in {elapsed(t0)} [{it} iterations]')

@nb.njit(parallel=True)
def solve_hh_global_backwards(par,ss,PLM_K,v_a,v_a_plus,a,c):
    """ solve household problem backwards for global solution """

    # a. post-decision
    marg_u_plus = np.zeros((par.Nbeta,par.Nz,par.NZ,par.NK,par.Na))
    
    # loop over aggregate states
    for i_Z in nb.prange(par.NZ):
        for i_K_lag in nb.prange(par.NK):
            
            # unpack
            Z = par.Z_grid[i_Z]
            K_lag = par.K_grid[i_K_lag]

            # PLM
            K = PLM_K[i_Z,i_K_lag]
            K = np.fmin(K,par.K_grid[-1]) # no extrapolation above
            K = np.fmax(K,par.K_grid[0]) # no extrapolation below

            # loop over shocks
            for i_Z_eps in range(par.Neps):               
                
                # ALM for Z
                Z_plus = ss.Z + par.rho_Z*(par.Z_grid[i_Z]-ss.Z) + par.Z_p[i_Z_eps]
                Z_plus = np.fmin(Z_plus,par.Z_grid[-1]) # no extrapolation above
                Z_plus = np.fmax(Z_plus,par.Z_grid[0]) # no extrapolation below

                # prepare bi-linear interpolation
                i_Z_plus = linear_interp.binary_search(0,par.Z_grid.size,par.Z_grid,Z_plus)
                i_K = linear_interp.binary_search(0,par.K_grid.size,par.K_grid,K)
                
                w_Z_plus = (Z_plus-par.Z_grid[i_Z_plus])/(par.Z_grid[i_Z_plus+1]-par.Z_grid[i_Z_plus])
                w_K = (K-par.K_grid[i_K])/(par.K_grid[i_K+1]-par.K_grid[i_K])
                
                # loop over idiosyncratic states
                for i_beta in range(par.Nbeta):
                    for i_z in range(par.Nz):                                            
                        for i_z_plus in range(par.Nz):
                            
                            marg_u_plus_ = marg_u_plus[i_beta,i_z]
                            Va_plus_ = v_a_plus[i_beta,i_z_plus]

                            # i. interpolate                                                    
                            low_Z_diff_val = Va_plus_[i_Z_plus,i_K+1,:]-Va_plus_[i_Z_plus,i_K,:]
                            low_Z = Va_plus_[i_Z_plus,i_K,:] + low_Z_diff_val*w_K
                            
                            high_Z_diff_val = Va_plus_[i_Z_plus+1,i_K+1,:]-Va_plus_[i_Z_plus+1,i_K,:]
                            high_Z = Va_plus_[i_Z_plus+1,i_K,:] + high_Z_diff_val*w_K
                            
                            diff_Z = high_Z-low_Z
                            Va_plus_interp = low_Z + diff_Z*w_Z_plus

                            # ii. accumulate
                            marg_u_plus_[i_Z,i_K_lag,:] += par.Z_w[i_Z_eps]*ss.z_trans[0,i_z,i_z_plus]*Va_plus_interp

    # b. egm loop
    for i_Z in nb.prange(par.NZ):
        for i_K_lag in nb.prange(par.NK):    
            
            # unpack
            Z = par.Z_grid[i_Z]
            K_lag = par.K_grid[i_K_lag]

            # i. household inputs
            r,_,w,_ = blocks.r_rk_w_u(par,Z,K_lag,1.0)

            # ii. egm
            v_a_now = np.zeros((par.Nbeta,par.Nz,par.Na))
            a_now = np.zeros((par.Nbeta,par.Nz,par.Na))
            c_now = np.zeros((par.Nbeta,par.Nz,par.Na))
            
            household_problem.solve_hh_backwards_egm(par,r,w,marg_u_plus[:,:,i_Z,i_K_lag,:],v_a_now,a_now,c_now)

            v_a[:,:,i_Z,i_K_lag,:] = v_a_now
            a[:,:,i_Z,i_K_lag,:] = a_now
            c[:,:,i_Z,i_K_lag,:] = c_now  
                          
#############
# aggregate #
#############

def solve_global(model,do_print=True,do_timings=True):
    """ solve using relaxation """

    t0 = time.time()
    
    if do_timings:
        timings = model.timings = {}
    else:
        timings = {} # forgotton afterwards

    timings['solve'] = 0.0
    timings['simulate'] = 0.0
    timings['PLM'] = 0.0

    par = model.par
    sim = model.sim

    # a. initial PLM
    t0_PLM = time.time()
    model.estimate_PLM(do_print=do_print)
    timings['PLM'] += time.time()-t0_PLM

    if do_print: print(f'')

    # b. iterate on PLM
    it = 0
    while True:

        t0_it = time.time()
        if do_print: print(f'iteration = {it:4d}')

        PLM_K_old = model.KS.PLM_K.copy()

        # i. solve and simulate
        t0_solve = time.time()
        model.solve_hh_global(do_print=do_print)
        timings['solve'] += time.time()-t0_solve

        t0_simulate = time.time()
        model.simulate_global(do_print=do_print)
        timings['simulate'] += time.time()-t0_simulate

        if do_print: 
            print(f'extrapolation: Pr[Z < Z_grid[0]] = {np.mean(sim.Z < par.Z_grid[0]):.2f}, Pr[Z > Z_grid[-1]] = {np.mean(sim.Z > par.Z_grid[-1]):.2f}')
            print(f'extrapolation: Pr[K < K_grid[0]] = {np.mean(sim.K < par.K_grid[0]):.2f}, Pr[K > K_grid[-1]] = {np.mean(sim.K > par.K_grid[-1]):.2f}')

        # ii. estimate PLM
        t0_PLM = time.time()
        model.estimate_PLM(it=it,do_print=do_print)
        timings['PLM'] += time.time()-t0_PLM

        # iii. check convergence
        max_abs_diff = np.max(np.abs(model.KS.PLM_K-PLM_K_old))
        if do_print: print(f'max. abs. diff. in PLM {max_abs_diff:6.1e}')
        if max_abs_diff < par.tol_relax: 
            if do_print: print(f'done in {elapsed(t0_it)}\n')
            break
        
        # iv. relaxation
        model.KS.PLM_K[:,:] = (1-par.relax_weight)*model.KS.PLM_K + par.relax_weight*PLM_K_old

        # v. increment
        it += 1
        if it > par.max_iter_relax: 
            raise ValueError('.solve_global_relaxation() did not converge')
        
        if do_print: print(f'done in {elapsed(t0_it)}\n')

    # c. finalize
    timings['it'] = it
    timings['total'] = time.time()-t0

    if do_print: print(f'model solved globally in {elapsed(t0)}')