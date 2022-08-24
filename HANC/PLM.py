import time
import numpy as np
import tensorflow as tf

from consav.misc import elapsed

# NN imports
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tensorflow import keras
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # supress tf warning

# RBF import
import scipy.interpolate as interpolate

#################
# aux functions #
#################

def scale_Z(par,Z,update_scale=False): 
    if update_scale:   
        par.Z_offset = np.mean(Z) # update continually
        par.Z_scale = np.std(Z-par.Z_offset) # update continually
    return (Z-par.Z_offset)/par.Z_scale

def scale_K(par,K,update_scale=False): 
    if update_scale:
        par.K_offset = np.mean(K) # update continually
        par.K_scale = np.std(K-par.K_offset) # update continually
    return (K-par.K_offset)/par.K_scale

def scale_inv(par,inv,update_scale=False): 
    if update_scale:
        par.inv_offset = np.mean(inv) # update continually
        par.inv_scale = np.std(inv-par.inv_offset) # update continually
    return (inv-par.inv_offset)/par.inv_scale

def get_Xy(model,scale=False,relative_y=False,intercept=False):

    par = model.par
    sim = model.sim

    # a. remove burn-in
    Z = sim.Z[par.simBurn:]
    K = sim.K[par.simBurn:]
    inv = sim.inv[par.simBurn:]
    q = sim.q[par.simBurn:]

    # b. X
    X = np.zeros((par.simT-par.simBurn-1,3+intercept))

    if intercept: X[:,0] = 1.0
    if scale:
        X[:,intercept+0] = scale_Z(par,Z[1:],update_scale=True)
        X[:,intercept+1] = scale_K(par,K[:-1],update_scale=True)
        X[:,intercept+2] = scale_inv(par,inv[:-1],update_scale=True)
    else:
        X[:,intercept+0] = Z[1:]
        X[:,intercept+1] = K[:-1]
        X[:,intercept+2] = inv[:-1]

    # c. y
    if relative_y:
        y_K = K[1:]/K[:-1]
    else:
        y_K = K[1:]
    
    y_q = q[1:]

    return X,y_K,y_q

def get_X_grid(model,scale=False,intercept=False):

    par = model.par

    X_grid = np.zeros((par.NK*par.NZ*par.Ninv,3+intercept))

    if intercept: X_grid[:,0] = 1.0
    
    Z,K_lag,inv_lag = np.meshgrid(par.Z_grid,par.K_grid,par.inv_grid,indexing='ij')
    X_grid[:,intercept+0] = Z.ravel()
    X_grid[:,intercept+1] = K_lag.ravel()
    X_grid[:,intercept+2] = inv_lag.ravel()

    if scale:
        X_grid[:,intercept+0] = scale_Z(par,X_grid[:,intercept+0])
        X_grid[:,intercept+1] = scale_K(par,X_grid[:,intercept+1])
        X_grid[:,intercept+2] = scale_inv(par,X_grid[:,intercept+2])

    return X_grid

############
# estimate #
############

def estimate_PLM(model,it=0,do_print=False):

    par = model.par
    sim = model.sim

    if par.PLM_method == 'OLS':
        estimate_PLM_OLS(model,it=it,do_print=do_print)
    elif par.PLM_method == 'NN':
        estimate_PLM_NN(model,it=it,do_print=do_print)
    elif par.PLM_method == 'RBF':
        estimate_PLM_RBF(model,it=it,do_print=do_print)
    else:
        raise NotImplementedError('unknown par.PLM_method')

def evaluate_PLM_Kq(model,Z,K_lag,inv_lag):

    par = model.par

    X = np.zeros((1,3))
    X[:,0] = Z
    X[:,1] = K_lag
    X[:,2] = inv_lag

    if par.PLM_method == 'OLS':
        
        K = model.PLM_K_func[0] + X@model.PLM_K_func[1:]
        q = model.PLM_q_func[0] + X@model.PLM_q_func[1:]

    elif par.PLM_method == 'RBF':

        K = model.PLM_K_func(X)[0]*K_lag
        q = model.PLM_q_func(X)[0]
        
    elif par.PLM_method == 'NN':

        X[:,0] = scale_Z(par,X[:,0])
        X[:,1] = scale_K(par,X[:,1])
        X[:,2] = scale_inv(par,X[:,2])

        K = model.PLM_K_func(X)[0,0]*K_lag
        q = model.PLM_q_func(X)[0,0]
    
    return K,q

def evaluate_PLM(model,Z,K_lag,inv_lag):
    
    par = model.par
    ss = model.ss

    # a. direct
    K,q = evaluate_PLM_Kq(model,Z,K_lag,inv_lag)

    # b. derived this period
    inv = K - (1-par.delta)*K_lag

    # c. foc_inv_term
    foc_inv_term = 0.0
    for i_eps_Z in range(par.Neps_Z):               
        
        Z_plus = ss.Z + par.rho_Z*(Z-ss.Z) + par.Z_p[i_eps_Z]
        K_plus,q_plus = evaluate_PLM_Kq(model,Z_plus,K,inv)

        inv_plus = K_plus-(1-par.delta)*K

        foc_inv_term_ = q_plus*np.log(inv_plus/inv)
        foc_inv_term += par.Z_w[i_eps_Z]*foc_inv_term_

    return K,q,foc_inv_term

def estimate_PLM_OLS(model,it=0,do_print=False):

    t0 = time.time()

    par = model.par

    # a. Xy and X_grid
    X,y_K,y_q = get_Xy(model,intercept=True)
    X_grid = get_X_grid(model,intercept=True)

    # b. OLS
    model.PLM_K_func = np.linalg.inv(X.T@X)@X.T@y_K
    model.PLM_q_func = np.linalg.inv(X.T@X)@X.T@y_q
    
    # c. evaluate at grid
    model.KS.PLM_K[:,:] = (X_grid@model.PLM_K_func).reshape(par.NZ,par.NK,par.Ninv)
    model.KS.PLM_q[:,:] = (X_grid@model.PLM_q_func).reshape(par.NZ,par.NK,par.Ninv)

    if do_print: print(f'PLM estimated with OLS in {elapsed(t0)}')

def estimate_PLM_NN(model,it=0,do_print=False):

    par = model.par

    tf.keras.backend.clear_session() 

    epochs = model.NN_settings['epochs']
    verbose = model.NN_settings['verbose']
    batch_size = model.NN_settings['batch_size']
    Nneurons = model.NN_settings['Nneurons']

    # a. define NN model
    t0 = time.time()

    PLM_K = model.PLM_K_func = keras.models.Sequential()
    PLM_q = model.PLM_q_func = keras.models.Sequential()

    for PLM in [PLM_K,PLM_q]:
        
        PLM.add(keras.layers.Dense(Nneurons,activation='relu', input_shape=[3]))
        PLM.add(keras.layers.Dense(1))
        opt = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.90,nesterov=True)
        PLM.compile(optimizer=opt,steps_per_execution=100,loss='mse',metrics=['mae','mse'])
    
    es = keras.callbacks.EarlyStopping(monitor='loss',mode='min',verbose=verbose,patience=5)

    # b. Xy and X_grid
    X,y_K,y_q = get_Xy(model,relative_y=True,scale=True)
    X_grid = get_X_grid(model,scale=True)

    # c. estimate neural net

    # i. fix seed
    tf.random.set_seed(1234)

    # ii. run
    PLM_K.fit(X,y_K,epochs=epochs,verbose=verbose,batch_size=batch_size,callbacks=[es])
    PLM_q.fit(X,y_q,epochs=epochs,verbose=verbose,batch_size=batch_size,callbacks=[es])

    mse_K = PLM_K.history.history['mse'][-1]
    mse_q = PLM_q.history.history['mse'][-1]
    
    K_Nepochs = len(PLM_K.history.history['mse'])
    q_Nepochs = len(PLM_q.history.history['mse'])

    # d. evaluate on grid
    model.KS.PLM_K[:,:] = PLM_K.predict(X_grid).reshape((par.NZ,par.NK,par.Ninv))*par.K_grid[np.newaxis,:,np.newaxis]
    model.KS.PLM_q[:,:] = PLM_q.predict(X_grid).reshape((par.NZ,par.NK,par.Ninv))

    if do_print:  print(f'PLM estimated with neural-net in {elapsed(t0)}')

    if do_print:
        
        print(f' {mse_K = :.2e} [{K_Nepochs = :4d}]')
        print(f' {mse_q = :.2e} [{q_Nepochs = :4d}]')

def estimate_PLM_RBF(model,it=0,do_print=False):

    t0 = time.time()

    par = model.par

    # a. Xy and X_grid
    X,y_K,y_q = get_Xy(model,relative_y=True)
    X_grid = get_X_grid(model)

    # b. create interpolate
    model.PLM_K_func = interpolate.RBFInterpolator(X,y_K,kernel='thin_plate_spline') 
    model.PLM_q_func = interpolate.RBFInterpolator(X,y_q,kernel='thin_plate_spline') 
   
    # c. evaluate on grid
    model.KS.PLM_K[:,:] = model.PLM_K_func(X_grid).reshape(par.NZ,par.NK,par.Ninv)*par.K_grid[np.newaxis,:,np.newaxis]
    model.KS.PLM_q[:,:] = model.PLM_q_func(X_grid).reshape(par.NZ,par.NK,par.Ninv)

    if do_print: print(f'PLM estimated with RBF in {elapsed(t0)}')

    if do_print:
        
        t0 = time.time()

        mse_K = np.mean(model.PLM_K_func(X)-y_K)**2
        mse_q = np.mean(model.PLM_q_func(X)-y_q)**2
        
        print(f'mse evaluated calculated in {elapsed(t0)}')

        print(f' {mse_K = :.2e}')
        print(f' {mse_q = :.2e}')    