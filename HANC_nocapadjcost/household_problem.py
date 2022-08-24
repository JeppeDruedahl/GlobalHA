import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,vbeg_a_plus,vbeg_a,a,c):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    v_a = np.zeros(vbeg_a.shape)
    solve_hh_backwards_egm(par,r,w,vbeg_a_plus,v_a,a,c)
    solve_hh_backwards_expectation(par,z_trans,v_a,vbeg_a)

@nb.njit(parallel=True)        
def solve_hh_backwards_egm(par,r,w,vbeg_a_plus,v_a,a,c):

    for i_fix in nb.prange(par.Nfix):

        # a. egm
        for i_z in nb.prange(par.Nz):
        
            # i. invert euler
            c_endo = (par.beta_grid[i_fix]*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            
            # ii. endogenous grid
            m_endo = c_endo + par.a_grid
            
            # iii. interpolation to fixed grid
            m = (1+r)*par.a_grid + w*par.z_grid[i_z]
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z] = np.fmax(a[i_fix,i_z],0.0)
            c[i_fix,i_z] = m-a[i_fix,i_z]

        # b. envelope condition
        v_a[i_fix] = (1+r)*c[i_fix]**(-par.sigma)

@nb.njit(parallel=True)        
def solve_hh_backwards_expectation(par,z_trans,v_a,vbeg_a): 

    for i_fix in nb.prange(par.Nfix):
        vbeg_a[i_fix] = z_trans[i_fix]@v_a[i_fix]
