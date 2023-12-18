import numba as nb
import numpy as np

from GEModelTools import lag, lead

@nb.njit
def r_rk_w_u(par,Z,K_lag,L,q):
    """ derive prices from state space """

    # i. capacity utilization rate
    u = (Z*K_lag**par.alpha*L**(1-par.alpha) - par.chi1 + par.chi2*par.u_target)/par.chi2
    u = np.fmin(u,par.u_max) # enforce upper bound

    # ii. hh inputs
    rk = par.alpha*Z*u*(K_lag/L)**(par.alpha-1.0)
    r = rk-q*par.delta
    w = (1.0-par.alpha)*Z*u*(rk/(par.alpha*Z*u))**(par.alpha/(par.alpha-1.0))
        
    return r,rk,w,u
    
@nb.jit 
def foc_inv_term_func(par,q_plus,inv_plus,inv):

    return q_plus*np.log(inv_plus/inv)

@nb.njit
def foc_inv_func(par,inv_lag,q,inv,foc_inv_term):

    LHS = q*(1-par.phi*np.log(inv/inv_lag))
    RHS = 1-par.beta*par.phi*foc_inv_term

    return LHS-RHS

@nb.njit
def block_pre(par,ini,ss, Z, K, L, u, rk, r, w, Y, C, A, inv, foc_inv_term, foc_inv, q):
    """ evaluate transition path - before household block """


    #################
    # implied paths #
    #################

    # a. pre-determined and exogenous
    K_lag = lag(ini.K,K)
    L[:] = 1.0

    # b. prices and capacity utilization
    # i. capacity utilization rate
    u[:] = (Z*K_lag**par.alpha*L**(1-par.alpha) - par.chi1 + par.chi2*par.u_target)/par.chi2
    u[:] = np.fmin(u,par.u_max) # enforce upper bound

    # ii. hh inputs
    rk[:] = par.alpha*Z*u*(K_lag/L)**(par.alpha-1.0)
    r[:] = rk-q*par.delta
    w[:] = (1.0-par.alpha)*Z*u*(rk/(par.alpha*Z*u))**(par.alpha/(par.alpha-1.0))
 

    # c. investment
    inv[:] = K-(1-par.delta)*K_lag
    inv_lag = lag(ini.inv,inv)

    q_plus = lead(q,ss.q)
    inv_plus = lead(inv,ss.inv)

    foc_inv_term[:] = q_plus*np.log(inv_plus/inv)
    foc_inv[:] = q*(1-par.phi*np.log(inv/inv_lag))  -  1-par.beta*par.phi*foc_inv_term


    # d. production and consumption
    Y[:] = Z*u*K_lag**(par.alpha)*L**(1-par.alpha)
    C[:] = Y-(K-K_lag)-par.delta*K_lag

    # e. stocks equal capital
    A[:] = K

@nb.njit
def block_post(par,ini,ss, clearing_A, clearing_C, A, A_hh, C, C_hh):
    """ evaluate transition path - after household block """
    ###########
    # targets #
    ###########

    clearing_A[:] = A-A_hh
    clearing_C[:] = C-C_hh