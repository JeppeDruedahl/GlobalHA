import numba as nb
import numpy as np

from GEModelTools import lag

@nb.njit
def r_rk_w_u(par,Z,K_lag,L):
    """ derive prices from state space """

    # i. capacity utilization rate
    u = par.chi2**(-1)*Z*K_lag**par.alpha*L**(1-par.alpha) - par.chi1/par.chi2 + par.u_target
    u = np.fmin(u,par.u_max) # enforce upper bound

    # ii. hh inputs
    rk = par.alpha*Z*u*(K_lag/L)**(par.alpha-1.0)
    r = rk-par.delta
    w = (1.0-par.alpha)*Z*u*(rk/(par.alpha*Z*u))**(par.alpha/(par.alpha-1.0))
        
    return r,rk,w,u
    
@nb.njit
def block_pre(par,ini,ss, Z, K, L, u, rk, r, w, Y, C, A):
    """ evaluate transition path - before household block """

    #################
    # implied paths #
    #################

    # a. pre-determined and exogenous
    K_lag = lag(ini.K,K)
    L[:] = 1.0

    # b. implied prices and capacity utilization
    u[:] = par.chi2**(-1)*Z*K_lag**par.alpha*L**(1-par.alpha) - par.chi1/par.chi2 + par.u_target
    u[:] = np.fmin(u,par.u_max) # enforce upper bound

    # ii. hh inputs
    rk[:] = par.alpha*Z*u*(K_lag/L)**(par.alpha-1.0)
    r[:] = rk-par.delta
    w[:] = (1.0-par.alpha)*Z*u*(rk/(par.alpha*Z*u))**(par.alpha/(par.alpha-1.0))


    # c. production and consumption
    Y[:] = Z*u*K_lag**(par.alpha)*L**(1-par.alpha)
    C[:] = Y-(K-K_lag)-par.delta*K_lag

    # d. stocks equal capital
    A[:] = K

@nb.njit
def block_post(par,ini,ss, Z, K, L, u, rk, r, w, Y, C, A, A_hh, C_hh, clearing_A, clearing_C):
    """ evaluate transition path - after household block """

    ###########
    # targets #
    ###########

    clearing_A[:] = A-A_hh
    clearing_C[:] = C-C_hh