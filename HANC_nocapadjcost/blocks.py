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
def block_pre(par,ss,ini,path,ncols=1):
    """ evaluate transition path - before household block """

    for ncol in range(ncols):

        # unpack
        Z = path.Z[ncol,:]
        K = path.K[ncol,:]
        L = path.L[ncol,:]
        u = path.u[ncol,:]

        rk = path.rk[ncol,:]
        r = path.r[ncol,:]
        w = path.w[ncol,:]
        
        Y = path.Y[ncol,:]
        C = path.C[ncol,:]
        A = path.A[ncol,:]

        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]

        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]

        #################
        # implied paths #
        #################

        # a. pre-determined and exogenous
        K_lag = lag(ini.K,K)
        L[:] = 1.0

        # b. implied prices and capacity utilization
        for t in range(par.T):
            r[t],rk[t],w[t],u[t] = r_rk_w_u(par,Z[t],K_lag[t],L[t])

        # c. production and consumption
        Y[:] = Z*u*K_lag**(par.alpha)*L**(1-par.alpha)
        C[:] = Y-(K-K_lag)-par.delta*K_lag

        # d. stocks equal capital
        A[:] = K

@nb.njit
def block_post(par,ss,ini,path,ncols=1):
    """ evaluate transition path - after household block """

    for ncol in range(ncols):

        # unpack
        Z = path.Z[ncol,:]
        K = path.K[ncol,:]
        L = path.L[ncol,:]
        u = path.u[ncol,:]

        rk = path.rk[ncol,:]
        r = path.r[ncol,:]
        w = path.w[ncol,:]
        
        Y = path.Y[ncol,:]
        C = path.C[ncol,:]
        A = path.A[ncol,:]

        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]

        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]

        ###########
        # targets #
        ###########

        clearing_A[:] = A-A_hh
        clearing_C[:] = C-C_hh