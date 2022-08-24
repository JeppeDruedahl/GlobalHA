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
def block_pre(par,ss,ini,path,ncols=1):
    """ evaluate transition path - before household block """

    for ncol in range(ncols):

        # unpack        
        A = path.A[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C = path.C[ncol,:]
        C_hh = path.C_hh[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        foc_inv = path.foc_inv[ncol,:]
        inv = path.inv[ncol,:]
        K = path.K[ncol,:]
        L = path.L[ncol,:]
        q = path.q[ncol,:]
        r = path.r[ncol,:]
        rk = path.rk[ncol,:]
        u = path.u[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        Z = path.Z[ncol,:]

        #################
        # implied paths #
        #################

        # a. pre-determined and exogenous
        K_lag = lag(ini.K,K)
        L[:] = 1.0

        # b. prices and capacity utilization
        for t in range(par.T):
            r[t],rk[t],w[t],u[t] = r_rk_w_u(par,Z[t],K_lag[t],L[t],q[t])

        # c. investment
        inv[:] = K-(1-par.delta)*K_lag
        inv_lag = lag(ini.inv,inv)

        q_plus = lead(q,ss.q)
        inv_plus = lead(inv,ss.inv)

        for t in range(par.T):
            foc_inv_term =  foc_inv_term_func(par,q_plus[t],inv_plus[t],inv[t])
            foc_inv[t] = foc_inv_func(par,inv_lag[t],q[t],inv[t],foc_inv_term)

        # d. production and consumption
        Y[:] = Z*u*K_lag**(par.alpha)*L**(1-par.alpha)
        C[:] = Y-(K-K_lag)-par.delta*K_lag

        # e. stocks equal capital
        A[:] = K

@nb.njit
def block_post(par,ss,ini,path,ncols=1):
    """ evaluate transition path - after household block """

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C = path.C[ncol,:]
        C_hh = path.C_hh[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        foc_inv = path.foc_inv[ncol,:]
        inv = path.inv[ncol,:]
        K = path.K[ncol,:]
        L = path.L[ncol,:]
        q = path.q[ncol,:]
        r = path.r[ncol,:]
        rk = path.rk[ncol,:]
        u = path.u[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        Z = path.Z[ncol,:]

        ###########
        # targets #
        ###########

        clearing_A[:] = A-A_hh
        clearing_C[:] = C-C_hh