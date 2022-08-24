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
        self.outputs_hh = ['a','c'] # outputs of household problem
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables in household problem

        # c. GE
        self.shocks = ['Z'] # exogenous inputs
        self.unknowns = ['K'] # endogenous inputs
        self.targets = ['clearing_A'] # targets
        
        self.varlist = [ # all variables
            'A_hh',
            'A',
            'C_hh',
            'C',
            'clearing_A',
            'clearing_C',
            'K',
            'L',
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

        # a. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta_mean = 0.995 # discount factor, mean, range is [mean-width,mean+width]
        par.beta_delta = 0.00000 # discount factor, width, range is [mean-width,mean+width]
        par.Nbeta = 1 # discount factor, number of states

        # b. income parameters
        par.rho_z = 0.96 # AR(1) parameter
        par.sigma_psi = 0.15 # std. of persistent shock
        par.Nz = 7 # number of productivity states

        # c. production and investment
        par.alpha = 0.33 # cobb-douglas
        par.delta = 0.05 # depreciation rate

        # d. capacity utilization 
        par.u_target = 0.99 # target for utilization rate (cost = 0)
        par.u_max = 1.0 # maximum utilitzation rate
        par.chi1 = 1.0 # linear utilization adjustment cost
        par.chi2 = 1.0 # quadratic utilization adjustment cost

        # f. grids         
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 200 # number of grid points

        # g. shocks
        par.jump_Z = 0.01 # initial jump in %
        par.rho_Z = 0.80 # AR(1) coefficient
        par.std_Z = 0.01 # standard deviation when simulating

        # h. misc.
        par.T = 500 # length of path
        par.simT = 10_000 # length of path
        par.IRFT = 60 # length of IRFs
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-12 # tolerance when solving eq. system

    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.Nfix = par.Nbeta
        par.beta_grid = np.zeros(par.Nbeta)
        
        # b. allocate GE
        self.allocate_GE()

    find_ss = steady_state.find_ss
    prepare_hh_ss = steady_state.prepare_hh_ss
   
    ##########
    # GLOBAL #
    ##########

    def setup_global(self,PLM_method='RBF',Z_fac=1.01,K_fac=1.1):
        """ setup self for global solution """
        
        assert PLM_method in ['NN','RBF','OLS']

        par = self.par
        ss = self.ss
        sim = self.sim
        KS = self.KS

        par.PLM_method = PLM_method

        # a. grids
        par.NK = 30
        par.NZ = 20
        par.Neps = 7

        dZ_halfspan = (sim.dZ.max()-sim.dZ.min())/2*Z_fac
        dK_halfspan = (sim.dK.max()-sim.dK.min())/2*K_fac

        par.Z_grid = np.linspace(ss.Z-dZ_halfspan,ss.Z+dZ_halfspan,par.NZ)
        par.K_grid = np.linspace(ss.K-dK_halfspan,ss.K+dK_halfspan,par.NK)
        
        par.Z_p, par.Z_w = normal_gauss_hermite(par.std_Z,par.Neps)

        # b. solution            
        shape = (par.Nbeta,par.Nz,par.NZ,par.NK,par.Na)

        KS.a = np.zeros(shape)
        KS.c = np.zeros(shape)
        KS.v_a = np.zeros(shape)
        
        KS.PLM_K = np.zeros((par.NZ,par.NK))

        # c. simulation
        par.simBurn = par.simT//10

        sim.Z = np.zeros(par.simT)
        sim.K = np.zeros(par.simT)
        sim.r = np.zeros(par.simT)
        sim.rk = np.zeros(par.simT)
        sim.w = np.zeros(par.simT)
        sim.u = np.zeros(par.simT)

        sim.vbeg_a = np.zeros((par.simT,*ss.vbeg_a.shape))

        # copy-in values
        sim.Z[:] = ss.Z + sim.dZ
        sim.K[:] = ss.K + sim.dK
        sim.r[:] = ss.r + sim.dr
        sim.rk[:] = ss.rk + sim.drk
        sim.w[:] = ss.w + sim.dw
        sim.u[:] = ss.u + sim.du

        par.K_offset = np.nan
        par.K_scale = np.nan

        par.Z_offset = np.nan
        par.Z_scale = np.nan

        # d. tolerances
        par.tol_solve_hh_global = 1e-5
        par.max_iter_solve_hh_global = 10_000

        par.relax_weight = 0.65
        par.tol_relax = 1e-4
        par.max_iter_relax = 100

        # e. update types
        self.infer_types()

        # f. misc
        self.NN_settings = {
           'linear':False,
           'epochs':1500,
           'verbose':0,
           'batch_size':32,
           'Nneurons':5000,
           'weights':None
        }

    set_inital_values = solve_global.set_inital_values    
    solve_hh_global = solve_global.solve_hh_global
    solve_global = solve_global.solve_global
    simulate_global = simulate_global.simulate_global
    estimate_PLM = PLM.estimate_PLM
    evaluate_PLM = PLM.evaluate_PLM

    def save_global(self,do_print=False):
        """ save global solution """

        with open(f'saved/{self.name}_KS.p', 'wb') as f:
            results = {'a':self.KS.a,'timings':self.timings}
            pickle.dump(results, f)  
    
    def load_global(self,do_print=False):
        """ load global solution """

        # a. load
        with open(f'saved/{self.name}_KS.p', 'rb') as f:
            loaded = pickle.load(f)

        # b. update parameters
        self.KS.__dict__['a'] = loaded['a']
        self.timings = loaded['timings']

    def compute_errors(self,do_print=False):
        """ compute one-step and den haan errors"""

        t0 = time.time()

        par = self.par
        sim = self.sim
        errors = self.errors = {}

        K_lag_ini = sim.K[par.simBurn-1]
        K = errors['K'] = sim.K[par.simBurn:]
        Z = errors['Z'] = sim.Z[par.simBurn:]
        
        # a. one-step errors
        errors['one_step'] = np.zeros(par.simT-par.simBurn)
        for t in range(par.simT-par.simBurn):
            K_lag = K[t-1] if t > 0 else K_lag_ini
            K_pred = self.evaluate_PLM(Z[t],K_lag)
            errors['one_step'][t] = (np.log(K[t])-np.log(K_pred))*100
            
        # b. den haan errors
        K_PLM = errors['K_PLM'] = np.zeros(par.simT-par.simBurn)
        for t in range(par.simT-par.simBurn):            
            K_lag = K_PLM[t-1] if t > 0 else K_lag_ini
            K_PLM[t] = self.evaluate_PLM(Z[t],K_lag)
            
        den_haan = errors['den_haan'] = np.abs(np.log(K)-np.log(K_PLM))*100

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

    def find_global_IRFs(self,Nini_IRF=1000,sss=False,do_print=False):

        t0 = time.time()

        par = self.par

        model_IRF = self.copy()

        # a. initial time period
        if not sss:
            tmax = par.simT-par.simBurn-par.IRFT   
            step = tmax//Nini_IRF
            ts = np.arange(0,tmax,step)        
            Nts = ts.size
        else:
            Nts = 1

        # b. allocate
        if sss:
            global_IRFs = self.global_IRFs_sss = {}
        else:
            global_IRFs = self.global_IRFs = {}
        
        global_IRFs['Z'] = np.zeros((Nts,par.IRFT))
        global_IRFs['K'] = np.zeros((Nts,par.IRFT))
        global_IRFs['u'] = np.zeros((Nts,par.IRFT))

        # c. baseline
        Z = self.sim.Z[par.simBurn:]
        K = self.sim.K[par.simBurn:]
        u = self.sim.u[par.simBurn:]

        # d. IRFs
        if sss:

            model_IRF.sim.Z[:par.IRFT] = self.ss.Z
            model_IRF.sim.Z[:par.IRFT] += par.jump_Z*par.rho_Z**np.arange(par.IRFT)

            ini_D = self.sim.D[-1]
            ini_K_lag = self.sim.K[-1]
            ini_u = self.sim.u[-1]

            model_IRF.simulate_global(ini_D=ini_D,ini_K_lag=ini_K_lag,simT=par.IRFT)
            
            global_IRFs['Z'][0,:] = (model_IRF.sim.Z[:par.IRFT]-self.ss.Z)/self.ss.Z*100    
            global_IRFs['K'][0,:] = (model_IRF.sim.K[:par.IRFT]-ini_K_lag)/ini_K_lag*100
            global_IRFs['u'][0,:] = (model_IRF.sim.u[:par.IRFT]-ini_u)/ini_u*100

        else:

            for i,t in enumerate(ts):
                
                model_IRF.sim.Z[:par.IRFT] = Z[t:t+par.IRFT]
                model_IRF.sim.Z[:par.IRFT] += par.jump_Z*par.rho_Z**np.arange(par.IRFT)
                ini_D = self.sim.D[par.simBurn+t]
                ini_K_lag = self.sim.K[par.simBurn+t-1]
                
                model_IRF.simulate_global(ini_D=ini_D,ini_K_lag=ini_K_lag,simT=par.IRFT)

                global_IRFs['Z'][i,:] = (model_IRF.sim.Z[:par.IRFT]-Z[t:t+par.IRFT])/Z[t]*100    
                global_IRFs['K'][i,:] = (model_IRF.sim.K[:par.IRFT]-K[t:t+par.IRFT])/K[t]*100
                global_IRFs['u'][i,:] = (model_IRF.sim.u[:par.IRFT]-u[t:t+par.IRFT])/u[t]*100


        # e. print
        if sss:
            if do_print: print(f'IRFs computed from sss in {elapsed(t0)}')
        else:
            if do_print: print(f'IRFs computed from {ts.size} starting points in {elapsed(t0)}')