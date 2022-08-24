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

def get_Xy(model,scale=False,relative_y=False,intercept=False):

    par = model.par
    sim = model.sim

    # a. remove burn-in
    Z = sim.Z[par.simBurn:]
    K = sim.K[par.simBurn:]

    # b. X
    X = np.zeros((par.simT-par.simBurn-1,2+intercept))

    if intercept: X[:,0] = 1.0
    if scale:
        X[:,intercept+0] = scale_Z(par,Z[1:],update_scale=True)
        X[:,intercept+1] = scale_K(par,K[:-1],update_scale=True)
    else:
        X[:,intercept+0] = Z[1:]
        X[:,intercept+1] = K[:-1]

    # c. y
    if relative_y:
        y = K[1:]/K[:-1]
    else:
        y = K[1:]

    return X,y

def get_X_grid(model,scale=False,intercept=False):

    par = model.par

    X_grid = np.zeros((par.NK*par.NZ,2+intercept))

    if intercept: X_grid[:,0] = 1.0
    if scale:
        X_grid[:,intercept+0] = np.repeat(scale_Z(par,par.Z_grid),par.NK)
        X_grid[:,intercept+1] = np.tile(scale_K(par,par.K_grid),par.NZ)
    else:
        X_grid[:,intercept+0] = np.repeat(par.Z_grid,par.NK)
        X_grid[:,intercept+1] = np.tile(par.K_grid,par.NZ)        

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

def evaluate_PLM(model,Z,K_lag):
    
    par = model.par

    if par.PLM_method == 'OLS':
        return model.PLM_K_func[0] + model.PLM_K_func[1]*Z + model.PLM_K_func[2]*K_lag
    elif par.PLM_method == 'RBF':
        X = np.array([[Z,K_lag]])
        return model.PLM_K_func(X)[0]*K_lag
    elif par.PLM_method == 'NN':
        X = np.array([[scale_Z(par,Z),scale_K(par,K_lag)]])
        return model.PLM_K_func(X)[0,0]

def estimate_PLM_OLS(model,it=0,do_print=False):

    t0 = time.time()

    par = model.par
    sim = model.sim

    # a. Xy and X_grid
    X,y = get_Xy(model,intercept=True)
    X_grid = get_X_grid(model,intercept=True)

    # b. OLS
    model.PLM_K_func = np.linalg.inv(X.T@X)@X.T@y
    
    # c. evaluate at grid
    model.KS.PLM_K[:,:] = (X_grid@model.PLM_K_func).reshape(par.NZ,par.NK)

    if do_print: print(f'PLM estimated with OLS in {elapsed(t0)}')

def estimate_PLM_NN(model,it=0,do_print=False):

    par = model.par

    tf.keras.backend.clear_session() 

    linear = model.NN_settings['linear']
    epochs = model.NN_settings['epochs']
    verbose = model.NN_settings['verbose']
    batch_size = model.NN_settings['batch_size']
    Nneurons = model.NN_settings['Nneurons']
    weights = model.NN_settings['weights']

    # a. define NN model
    t0 = time.time()

    PLM_K = model.PLM_K_func = keras.models.Sequential()
    if not linear: PLM_K.add(keras.layers.Dense(Nneurons,activation='relu', input_shape=[2]))
    PLM_K.add(keras.layers.Dense(1))
    opt = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.90,nesterov=True)
    PLM_K.compile(optimizer=opt,steps_per_execution=100,loss='mse',metrics=['mae'])
    
    if it is None or it > 10:
        patience = 5
    else:
        patience = 3

    es = keras.callbacks.EarlyStopping(monitor='loss',mode='min',verbose=verbose,patience=patience)

    if not weights is None: PLM_K.layers[0].set_weights(weights)

    # b. Xy and X_grid
    X,y = get_Xy(model,scale=True)
    X_grid = get_X_grid(model,scale=True)

    # c. estimate neural net

    # i. fix seed
    tf.random.set_seed(1234)

    # ii. run
    PLM_K.fit(X,y,epochs=epochs,verbose=verbose,batch_size=batch_size,callbacks=[es])

    mae = PLM_K.history.history['mae'][-1]
    Nepochs = len(PLM_K.history.history['mae'])

    # d. evaluate on grid
    model.KS.PLM_K[:,:] = PLM_K.predict(X_grid).reshape((par.NZ,par.NK))

    if do_print:  print(f'PLM estimated with neural-net in {elapsed(t0)} [mea: {mae:6.2e}, # of epochs: {Nepochs:4d}]')

def estimate_PLM_RBF(model,it=0,do_print=False):

    t0 = time.time()

    par = model.par

    # a. Xy and X_grid
    X,y = get_Xy(model,relative_y=True)
    X_grid = get_X_grid(model)

    # b. create interpolate
    model.PLM_K_func = interpolate.RBFInterpolator(X,y,kernel='thin_plate_spline') 
   
    # c. evaluate on grid
    model.KS.PLM_K[:,:] = model.PLM_K_func(X_grid).reshape(par.NZ,par.NK)*par.K_grid

    if do_print: print(f'PLM estimated with RBF in {elapsed(t0)}')