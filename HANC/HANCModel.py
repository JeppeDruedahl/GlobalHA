import pickle
import time
import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass
from consav.quadrature import normal_gauss_hermite
from consav.misc import elapsed

# local
import blocks
import steady_state
import household_problem
import solve_global
import simulate_global
import PLM

class HANCModelClass(EconModelClass,GEModelClass):    

    ############
    # STANDARD #
    ############

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim','KS']

        # b. household
        self.grids_hh = ['a'] # grids in household problem
        self.pols_hh = ['a'] # household policy functions

        self.inputs_hh = ['r','w'] # inputs to household problem
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','v_a'] # outputs of household problem
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables in household problem

        # c. GE
        self.shocks = ['Z'] # exogenous inputs
        self.unknowns = ['K','q'] # endogenous inputs
        self.targets = ['clearing_A','foc_inv'] # targets
        
        self.varlist = [ # all variables
            'A_hh',
            'A',
            'C_hh',
            'C',
            'clearing_A',
            'clearing_C',
            'foc_inv',
            'inv',
            'K',
            'L',
            'q',
            'r',
            'rk',
            'u',
            'w',
            'Y',
            'Z',
        ]

        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = blocks.block_pre
        self.block_post = blocks.block_post
        
    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1 # no ex ante different types

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta = 0.995 # discount factor

        # b. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.15 # std. of persistent shock
        par.Nz = 3 # number of productivity states

        # c. production and investment
        par.alpha = 0.33 # cobb-douglas
        par.delta = 0.05 # depreciation rate
        par.phi = 0.05 # investment adj. cost

        # d. capacity utilization 
        par.u_target = 0.99 # target for utilization rate (cost = 0)
        par.u_max = 1.0 # maximum utilitzation rate
        par.chi1 = 1.0 # linear utilization adjustment cost
        par.chi2 = 1.0 # quadratic utilization adjustment cost

        # f. grids         
        par.a_max = 100.0 # maximum point in grid for a
        par.Na = 80 # number of grid points

        # g. shocks
        par.jump_Z = 0.01 # initial jump in %
        par.rho_Z = 0.80 # AR(1) coefficient
        par.std_Z = 0.01 # standard deviation when simulating

        # h. misc.
        par.T = 500 # length of path
        par.simT = 5_000 # length of path
        par.IRFT = 60 # length of IRFs
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-12 # tolerance when solving eq. system

    def allocate(self):
        """ allocate model """

        self.allocate_GE()

    find_ss = steady_state.find_ss
    prepare_hh_ss = steady_state.prepare_hh_ss
   
    ##########
    # GLOBAL #
    ##########

    def setup_global(self,PLM_method='RBF',Z_fac=1.01,K_fac=1.2,inv_fac=1.1):
        """ setup self for global solution """
        
        assert PLM_method in ['NN','RBF','OLS']

        par = self.par
        ss = self.ss
        sim = self.sim
        KS = self.KS

        par.PLM_method = PLM_method

        # a. grids
        par.NZ = 10
        par.NK = 20
        par.Ninv = 15
        par.Neps_Z = 3

        dZ_halfspan = (sim.dZ.max()-sim.dZ.min())/2*Z_fac
        dK_halfspan = (sim.dK.max()-sim.dK.min())/2*K_fac
        dinv_halfspan = (sim.dinv.max()-sim.dinv.min())/2*inv_fac

        par.Z_grid = np.linspace(ss.Z-dZ_halfspan,ss.Z+dZ_halfspan,par.NZ)
        par.K_grid = np.linspace(ss.K-dK_halfspan,ss.K+dK_halfspan,par.NK)
        par.inv_grid = np.linspace(ss.inv-dinv_halfspan,ss.inv+dinv_halfspan,par.Ninv)
        
        par.Z_p, par.Z_w = normal_gauss_hermite(par.std_Z,par.Neps_Z)

        # b. solution            
        shape = (par.Nfix,par.Nz,par.NZ,par.NK,par.Ninv,par.Na)

        KS.a = np.zeros(shape)
        KS.c = np.zeros(shape)
        KS.v_a = np.zeros(shape)
        KS.m = np.zeros(shape)
               
        KS.PLM_K = np.zeros((par.NZ,par.NK,par.Ninv))
        KS.PLM_q = np.zeros((par.NZ,par.NK,par.Ninv))

        # c. simulation
        par.simBurn = par.simT//10

        sim.Z = np.zeros(par.simT)
        sim.K = np.zeros(par.simT)
        sim.A_hh = np.zeros(par.simT)
        sim.inv = np.zeros(par.simT)
        sim.r = np.zeros(par.simT)
        sim.rk = np.zeros(par.simT)
        sim.w = np.zeros(par.simT)
        sim.u = np.zeros(par.simT)
        sim.q = np.zeros(par.simT)
        sim.PLM_K = np.zeros(par.simT)
        sim.PLM_q = np.zeros(par.simT)
        sim.PLM_foc_inv_term = np.zeros(par.simT)
        sim.foc_inv = np.zeros(par.simT)

        sim.vbeg_a = np.zeros((par.simT,*ss.vbeg_a.shape))

        # copy-in values
        sim.Z[:] = ss.Z + sim.dZ
        sim.K[:] = ss.K + sim.dK
        sim.inv[:] = ss.inv + sim.dinv
        sim.r[:] = ss.r + sim.dr
        sim.rk[:] = ss.rk + sim.drk
        sim.w[:] = ss.w + sim.dw
        sim.u[:] = ss.u + sim.du
        sim.q[:] = ss.q + sim.dq

        par.Z_offset = np.nan
        par.Z_scale = np.nan

        par.K_offset = np.nan
        par.K_scale = np.nan

        par.inv_offset = np.nan
        par.inv_scale = np.nan

        # d. tolerances
        par.tol_solve_hh_global = 1e-5
        par.max_iter_solve_hh_global = 10_000
        par.tol_solve_clearing = 1e-6

        par.relax_weight = 0.65
        par.tol_relax = 1e-4
        par.max_iter_relax = 50

        # e. update types
        self.infer_types()

        # f. misc
        self.NN_settings = {
           'epochs':1500,
           'verbose':0,
           'batch_size':32,
           'Nneurons':5000,
        }

    set_inital_values = solve_global.set_inital_values    
    solve_hh_global = solve_global.solve_hh_global
    solve_global = solve_global.solve_global
    simulate_global = simulate_global.simulate_global
    estimate_PLM = PLM.estimate_PLM
    evaluate_PLM = PLM.evaluate_PLM
    evaluate_PLM_Kq = PLM.evaluate_PLM_Kq

    def save_global(self,do_print=False):
        """ save global solution """

        with open(f'saved/{self.name}_KS.p', 'wb') as f:
            results = {'KS':self.KS.__dict__,'sim':self.sim.__dict__,'timings':self.timings}
            pickle.dump(results, f)  
    
    def load_global(self,do_print=False):
        """ load global solution """

        # a. load
        with open(f'saved/{self.name}_KS.p', 'rb') as f:
            loaded = pickle.load(f)

        # b. update parameters
        for k,v in loaded['KS'].items(): self.KS.__dict__[k] = v
        for k,v in loaded['sim'].items(): self.sim.__dict__[k] = v
        self.timings = loaded['timings']


    def compute_errors(self,do_print=False):
        """ compute one-step and den haan errors"""

        t0 = time.time()

        par = self.par
        sim = self.sim
        errors = self.errors = {}

        K_lag_ini = sim.K[par.simBurn-1]
        inv_lag_ini = sim.inv[par.simBurn-1]

        Z = errors['Z'] = sim.Z[par.simBurn:]
        K = errors['K'] = sim.K[par.simBurn:]
        inv = errors['inv'] = sim.inv[par.simBurn:]
        q = errors['inv'] = sim.q[par.simBurn:]
        
        # a. one-step errors
        errors['one_step_K'] = np.zeros(par.simT-par.simBurn)
        errors['one_step_q'] = np.zeros(par.simT-par.simBurn)
        for t in range(par.simT-par.simBurn):
            K_lag = K[t-1] if t > 0 else K_lag_ini
            inv_lag = inv[t-1] if t > 0 else inv_lag_ini
            K_pred,q_pred = self.evaluate_PLM_Kq(Z[t],K_lag,inv_lag)
            errors['one_step_K'][t] = (np.log(K_pred)-np.log(K[t]))*100
            errors['one_step_q'][t] = (np.log(q_pred)-np.log(q[t]))*100
            
        # b. den haan errors
        K_PLM = errors['K_PLM'] = np.zeros(par.simT-par.simBurn)
        inv_PLM = errors['inv_PLM'] = np.zeros(par.simT-par.simBurn)
        q_PLM = errors['q_PLM'] = np.zeros(par.simT-par.simBurn)
        for t in range(par.simT-par.simBurn):            
            K_lag = K_PLM[t-1] if t > 0 else K_lag_ini
            inv_lag = inv_PLM[t-1] if t > 0 else inv_lag_ini
            K_PLM[t],q_PLM[t] = self.evaluate_PLM_Kq(Z[t],K_lag,inv_lag)
            inv_PLM[t] = K_PLM[t]-(1-par.delta)*K_lag
            
        den_haan = errors['den_haan'] = np.abs(np.log(K_PLM)-np.log(K))*100

        # c. print
        if do_print:

            print(fr'max abs.:      {np.max(den_haan):.2f}')
            print(fr'mean abs.:     {np.mean(den_haan):.2f}')
            print(fr'median abs.:   {np.percentile(den_haan, 50):.2f}')
            print(fr'99 perc. abs.: {np.percentile(den_haan, 99):.2f}')
            print(fr'90 perc. abs.: {np.percentile(den_haan, 90):.2f}')

            R2 = 1-np.var(K_PLM-K)/np.var(K)
            print(f'R2:            {R2:.4f}')

            print(f'errors computed in {elapsed(t0)}')   

    def find_global_IRFs(self,Nini_IRF=500,do_print=False):

        t0 = time.time()

        print('hey')

        par = self.par

        model_IRF = self.copy()

        model_IRF.PLM_K_func = self.PLM_K_func
        model_IRF.PLM_q_func = self.PLM_q_func

        # a. initial time period
        tmax = par.simT-par.simBurn-par.IRFT   
        step = tmax//Nini_IRF
        ts = np.arange(0,tmax,step)        
        Nts = ts.size

        # b. allocate
        global_IRFs = self.global_IRFs = {}
        
        global_IRFs['Z'] = np.zeros((Nts,par.IRFT))
        global_IRFs['K'] = np.zeros((Nts,par.IRFT))
        global_IRFs['u'] = np.zeros((Nts,par.IRFT))

        # c. baseline
        Z = self.sim.Z[par.simBurn:]
        K = self.sim.K[par.simBurn:]
        u = self.sim.u[par.simBurn:]

        # d. IRFs
        for i,t in enumerate(ts):
            
            model_IRF.sim.Z[:par.IRFT] = Z[t:t+par.IRFT]
            model_IRF.sim.Z[:par.IRFT] += par.jump_Z*par.rho_Z**np.arange(par.IRFT)
            ini_D = self.sim.D[par.simBurn+t]
            ini_K_lag = self.sim.K[par.simBurn+t-1]
            ini_inv_lag = self.sim.inv[par.simBurn+t-1]
            
            model_IRF.simulate_global(ini_D=ini_D,ini_K_lag=ini_K_lag,ini_inv_lag=ini_inv_lag,simT=par.IRFT)

            global_IRFs['Z'][i,:] = (model_IRF.sim.Z[:par.IRFT]-Z[t:t+par.IRFT])/Z[t]*100    
            global_IRFs['K'][i,:] = (model_IRF.sim.K[:par.IRFT]-K[t:t+par.IRFT])/K[t]*100
            global_IRFs['u'][i,:] = (model_IRF.sim.u[:par.IRFT]-u[t:t+par.IRFT])/u[t]*100

        # e. print
        if do_print: print(f'IRFs computed from {ts.size} starting points in {elapsed(t0)}')