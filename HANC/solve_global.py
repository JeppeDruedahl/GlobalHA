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
        
        set_inital_values_(par,ss,sim,KS.a,KS.c,KS.v_a,KS.m)

    if do_print: print(f'initial values found in {elapsed(t0)}')

    assert np.all(KS.v_a > 0)

@nb.njit(parallel=True)
def set_inital_values_(par,ss,sim,a,c,v_a,m):

    # a. common X
    X = np.zeros((par.simT-par.simBurn-1,4))
    X[:,0] = 1.0
    X[:,1] = sim.Z[par.simBurn+1:]
    X[:,2] = sim.K[par.simBurn:-1]
    X[:,3] = sim.inv[par.simBurn:-1]

    X_mat = np.linalg.inv(X.T@X)@X.T

    # b. loop over idiosyncratic states
    for i_fix in nb.prange(par.Nfix):
        for i_z in nb.prange(par.Nz):
            for i_a_lag in nb.prange(par.Na):
                
                # i. regression
                coeffs_v_a = X_mat@sim.v_a[par.simBurn+1:,i_fix,i_z,i_a_lag].copy()
                coeffs_a = X_mat@sim.a[par.simBurn+1:,i_fix,i_z,i_a_lag].copy()
                coeffs_c = X_mat@sim.c[par.simBurn+1:,i_fix,i_z,i_a_lag].copy()

                # ii. loop over aggregate states
                for i_Z in range(par.NZ):
                    for i_K_lag in range(par.NK):
                        for i_inv_lag in range(par.Ninv):

                            Z = par.Z_grid[i_Z]
                            K_lag = par.K_grid[i_K_lag]
                            inv_lag = par.inv_grid[i_inv_lag]

                            v_a_pred = coeffs_v_a[0] + coeffs_v_a[1]*Z + coeffs_v_a[2]*K_lag + coeffs_v_a[3]*inv_lag
                            v_a[i_fix,i_z,i_Z,i_K_lag,i_inv_lag,i_a_lag] = v_a_pred
                        
                            a_pred = coeffs_a[0] + coeffs_a[1]*Z + coeffs_a[2]*K_lag + coeffs_a[3]*inv_lag
                            a[i_fix,i_z,i_Z,i_K_lag,i_inv_lag,i_a_lag] = a_pred

                            c_pred = coeffs_c[0] + coeffs_c[1]*Z + coeffs_c[2]*K_lag + coeffs_c[3]*inv_lag
                            c[i_fix,i_z,i_Z,i_K_lag,i_inv_lag,i_a_lag] = c_pred

                # iii. cash-on-hand
                m[i_fix,i_z] = a[i_fix,i_z] + c[i_fix,i_z]

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
            
            t0_it = time.time()

            old_a = KS.a.copy()

            # i. backwards step
            marg_u_plus = np.zeros((par.Nfix,par.Nz,par.NZ,par.NK,par.Ninv,par.Na))
            solve_hh_global_backwards_exp(par,ss,KS.PLM_K,KS.PLM_q,KS.v_a,KS.v_a,KS.a,KS.c,KS.m,marg_u_plus)
            
            t0_it_egm = time.time()
            solve_hh_global_backwards_egm(par,KS.PLM_q,marg_u_plus,KS.v_a,KS.a,KS.c,KS.m)

            # ii. error
            max_abs_diff = np.max(np.abs(KS.a-old_a))

            if it < 10 or it%100 == 0:
                print(f'{it = :4d}: {max_abs_diff = :12.8f} [in {elapsed(t0_it)}, in {elapsed(t0_it_egm)}]')
            
            # iii. stopping criterion
            if max_abs_diff < par.tol_solve_hh_global: break

            # iv. increment
            it += 1
            if it > par.max_iter_solve_hh_global: 
                raise ValueError('solve_hh_global(), too many iterations')

    if do_print: print(f'household problem solved in {elapsed(t0)} [{it} iterations]')

@nb.njit(parallel=True)
def solve_hh_global_backwards_exp(par,ss,PLM_K,PLM_q,v_a,v_a_plus,a,c,m,marg_u_plus):
    """ solve household problem backwards for global solution """

    for i_Z in nb.prange(par.NZ):
        for i_K_lag in nb.prange(par.NK):
            for i_inv_lag in nb.prange(par.Ninv):
            
                # unpack
                Z = par.Z_grid[i_Z]
                K_lag = par.K_grid[i_K_lag]

                # allocate
                v_a_plus_interp = np.zeros(par.Na)

                # PLM
                K = PLM_K[i_Z,i_K_lag,i_inv_lag]
                inv = K - (1-par.delta)*K_lag

                K = np.fmin(K,par.K_grid[-1]) # no extrapolation above
                K = np.fmax(K,par.K_grid[0]) # no extrapolation below

                inv = np.fmin(inv,par.inv_grid[-1]) # no extrapolation above
                inv = np.fmax(inv,par.inv_grid[0]) # no extrapolation below

                i_K = linear_interp.binary_search(0,par.K_grid.size,par.K_grid,K)
                i_inv = linear_interp.binary_search(0,par.inv_grid.size,par.inv_grid,inv)

                denom_K_inv = (par.K_grid[i_K+1]-par.K_grid[i_K])*(par.inv_grid[i_inv+1]-par.inv_grid[i_inv])

                # loop over shocks
                for i_Z_eps in range(par.Neps_Z):               
                    
                    # ALM for Z
                    Z_plus = ss.Z + par.rho_Z*(Z-ss.Z) + par.Z_p[i_Z_eps]
                    Z_plus = np.fmin(Z_plus,par.Z_grid[-1]) # no extrapolation above
                    Z_plus = np.fmax(Z_plus,par.Z_grid[0]) # no extrapolation below

                    # prepare interpolation
                    i_Z_plus = linear_interp.binary_search(0,par.Z_grid.size,par.Z_grid,Z_plus)

                    nom_1_left = par.Z_grid[i_Z_plus+1]-Z_plus
                    nom_1_right = Z_plus-par.Z_grid[i_Z_plus]

                    nom_2_left = par.K_grid[i_K+1]-K
                    nom_2_right = K-par.K_grid[i_K]

                    nom_3_left = par.inv_grid[i_inv+1]-inv
                    nom_3_right = inv-par.inv_grid[i_inv]

                    denom = (par.Z_grid[i_Z_plus+1]-par.Z_grid[i_Z_plus])*denom_K_inv

                    # loop over idiosyncratic states
                    for i_z in range(par.Nz):                                            
                        for i_z_plus in range(par.Nz):
                                
                            v_a_plus_interp[:] = 0.0

                            # i. interpolate 
                            for k1 in range(2):
                                nom_1 = nom_1_left if k1 == 0 else nom_1_right
                                for k2 in range(2):
                                    nom_2 = nom_2_left if k2 == 0 else nom_2_right       
                                    for k3 in range(2):
                                        nom_3 = nom_3_left if k3 == 0 else nom_3_right               
                                        nom_fac = nom_1*nom_2*nom_3
                                        v_a_plus_interp += nom_fac*v_a_plus[0,i_z_plus,i_Z_plus+k1,i_K+k2,i_inv+k3,:]
                            
                            v_a_plus_interp /= denom

                            # ii. marg_u_plus
                            marg_u_plus[0,i_z,i_Z,i_K_lag,i_inv_lag,:] += par.Z_w[i_Z_eps]*ss.z_trans[0,i_z,i_z_plus]*v_a_plus_interp

@nb.njit(parallel=True)
def solve_hh_global_backwards_egm(par,PLM_q,marg_u_plus,v_a,a,c,m):

    for i_Z in nb.prange(par.NZ):
        for i_K_lag in nb.prange(par.NK): 
            for i_inv_lag in nb.prange(par.Ninv):   
            
                # unpack
                Z = par.Z_grid[i_Z]
                K_lag = par.K_grid[i_K_lag]

                # i. PLM
                q = PLM_q[i_Z,i_K_lag,i_inv_lag]
                r,_,w,_ = blocks.r_rk_w_u(par,Z,K_lag,1.0,q)

                # ii. egm
                v_a_now = np.zeros((par.Nfix,par.Nz,par.Na))
                a_now = np.zeros((par.Nfix,par.Nz,par.Na))
                c_now = np.zeros((par.Nfix,par.Nz,par.Na))
                
                household_problem.solve_hh_backwards_egm(par,r,w,marg_u_plus[:,:,i_Z,i_K_lag,i_inv_lag,:],a_now,c_now,v_a_now)

                v_a[:,:,i_Z,i_K_lag,i_inv_lag,:] = v_a_now
                a[:,:,i_Z,i_K_lag,i_inv_lag,:] = a_now
                c[:,:,i_Z,i_K_lag,i_inv_lag,:] = c_now  
                m[:,:,i_Z,i_K_lag,i_inv_lag,:] = a_now + c_now  

#############
# aggregate #
#############

def solve_global(model,do_print=True,initial_PLM=True,do_timings=True):
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
        PLM_q_old = model.KS.PLM_q.copy()

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
        max_abs_diff_K = np.max(np.abs(model.KS.PLM_K-PLM_K_old))
        max_abs_diff_q = np.max(np.abs(model.KS.PLM_q-PLM_q_old))
        if do_print: 
            print(f'max. abs. diff. in PLM for K {max_abs_diff_K:6.1e}')
            print(f'max. abs. diff. in PLM for q {max_abs_diff_q:6.1e}')
        
        max_abs_diff = np.fmax(max_abs_diff_K,max_abs_diff_q)
        if max_abs_diff < par.tol_relax: 
            if do_print: print(f'done in {elapsed(t0_it)}\n')
            break
        
        # iv. relaxation
        model.KS.PLM_K[:,:] = (1-par.relax_weight)*model.KS.PLM_K + par.relax_weight*PLM_K_old
        model.KS.PLM_q[:,:] = (1-par.relax_weight)*model.KS.PLM_q + par.relax_weight*PLM_q_old

        # v. increment
        it += 1
        if it > par.max_iter_relax: 
            print('.solve_global_relaxation() did not converge')
            break
            #raise ValueError('.solve_global_relaxation() did not converge')
        
        if do_print: print(f'done in {elapsed(t0_it)}\n')

    # c. finalize
    timings['it'] = it
    timings['total'] = time.time()-t0

    if do_print: print(f'model solved globally in {elapsed(t0)}')