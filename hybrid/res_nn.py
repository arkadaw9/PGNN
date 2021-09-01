import os
import scipy.io as spio
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error

import argparse

# For HPD-Res model, we use the YPhy = 1 and lamda (i.e., \lambda_{phy}) = 0
# For Res model, we use the YPhy = 0 and lamda (i.e., \lambda_{phy}) = 0

ap = argparse.ArgumentParser(description='Physics Guided Neural Network')
ap.add_argument('--dataset', choices=['mille_lacs','mendota'], default='mille_lacs', type=str, help='Dataset choice')
ap.add_argument('--optimizer_val', choices=['0: Adagrad', '1: Adadelta', '2: Adam', '3: Nadam', '4: RMSprop', '5: SGD', '6: NSGD'], default=2, type=int, help='Optimizer')
ap.add_argument('--data_dir', default='../datasets/', type=str, help='Data Directory')
ap.add_argument('--batch_size', default=1000, type=int, help='Batch Size')
ap.add_argument('--epochs', default=10000, type=int, help='Epochs')
ap.add_argument('--drop_frac', default=0.0, type=float, help='Dropout Fraction')
ap.add_argument('--use_YPhy', type=int, default=0, help='Use Physics Numeric Model as input')
ap.add_argument('--n_nodes', default=12, type=int, help='Number of Nodes in Hidden Layer')
ap.add_argument('--n_layers', default=2, type=int, help='Number of Hidden Layers')
ap.add_argument('--lamda', default=0.0, type=float, help='lambda hyperparameter')
ap.add_argument('--tr_size', default=3000, type=int, help='Size of Training set')
ap.add_argument('--val_frac', default=0.1, type=float, help='Validation Fraction')
ap.add_argument('--patience_val', default=500, type=int, help='Patience Value for Early Stopping')
ap.add_argument('--n_iters', default=1, type=int, help='Number of Random Runs')
ap.add_argument('--save_dir', default='./results/', type=str, help='Save Directory')
args = ap.parse_args()

#function to compute the room_mean_squared_error given the ground truth (y_true) and the predictions(y_pred)
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

#function for computing the density given the temperature(nx1 matrix)
def density(temp):
    return 1000 * ( 1 - (temp + 288.9414) * (temp - 3.9863)**2 / (508929.2 * (temp + 68.12963) ) )

def phy_loss_mean(params):
	# useful for cross-checking training
    udendiff, lam = params
    def loss(y_true,y_pred):
        return K.mean(K.relu(udendiff))
    return loss

#function to calculate the combined loss = sum of rmse and phy based loss
def combined_loss(params):
    udendiff, lam = params
    def loss(y_true,y_pred):
        return mean_squared_error(y_true, y_pred) + lam * K.mean(K.relu(udendiff))
    return loss

def relu(m):
    m[m < 0] = 0
    return m

def evaluate_physics_loss(model, uX1, uX2, YPhy1, YPhy2):
    tolerance = 0
    uout1 = model.predict(uX1) + YPhy1
    uout2 = model.predict(uX2) + YPhy2
    udendiff = (density(uout1) - density(uout2))
    percentage_phy_incon = np.sum(udendiff>tolerance)/udendiff.shape[0]
    phy_loss = np.mean(relu(udendiff))
    return phy_loss, percentage_phy_incon

def PGNN_train_test(iteration=0):
    
    if args.use_YPhy==1:
        model_name = 'HPD_res_'
    elif args.use_YPhy==0:
        model_name = 'res_'
    else:
        raise TypeError("Not supported Value")
    
    # Hyper-parameters of the training process
    batch_size = args.batch_size
    num_epochs = args.epochs
    val_frac = args.val_frac
    patience_val = args.patience_val
    
    # List of optimizers to choose from 
    optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
    optimizer_vals = [Adagrad(clipnorm=1), Adadelta(clipnorm=1), Adam(clipnorm=1), Nadam(clipnorm=1), RMSprop(clipnorm=1), SGD(clipnorm=1.), SGD(clipnorm=1, nesterov=True)]
    
    # selecting the optimizer
    optimizer_name = optimizer_names[args.optimizer_val]
    optimizer_val = optimizer_vals[args.optimizer_val]
    
    data_dir = args.data_dir
    filename = args.dataset + '.mat'
    mat = spio.loadmat(data_dir + filename, squeeze_me=True,
    variable_names=['Y','Xc_doy','Modeled_temp'])
    Xc = mat['Xc_doy']
    Y = mat['Y']
    YPhy = mat['Modeled_temp']
    
    YPhy_norm = Xc[:,-1]
    residual = Y - YPhy
    
    #Backcalculating mean and variance from YPhy
    Y_phy_std =  (YPhy[0] -  YPhy[1])/(YPhy_norm[0] - YPhy_norm[1])
    Y_phy_mean = YPhy[0] - Y_phy_std * YPhy_norm[0]
    
    trainX, trainY, trainYPhy, trainTemp = Xc[:args.tr_size,:], residual[:args.tr_size], YPhy[:args.tr_size], Y[:args.tr_size]
    testX, testY, testYPhy, testTemp = Xc[args.tr_size:,:], residual[args.tr_size:], YPhy[args.tr_size:], Y[args.tr_size:]
    
    # Loading unsupervised data
    unsup_filename = args.dataset + '_sampled.mat'
    unsup_mat = spio.loadmat(data_dir+unsup_filename, squeeze_me=True,
    variable_names=['Xc_doy1','Xc_doy2'])
    
    uX1 = unsup_mat['Xc_doy1'] # Xc at depth i for every pair of consecutive depth values
    uX2 = unsup_mat['Xc_doy2'] # Xc at depth i + 1 for every pair of consecutive depth values
    
    YPhy1 = np.reshape((uX1[:,-1] * Y_phy_std + Y_phy_mean), (-1,1))
    YPhy2 = np.reshape((uX2[:,-1] * Y_phy_std + Y_phy_mean), (-1,1))
    
    if args.use_YPhy == 0:
    	# Removing the last column from uX (corresponding to Y_PHY)
        uX1 = uX1[:,:-1]
        uX2 = uX2[:,:-1]
        trainX = trainX[:,:-1]
        testX = testX[:,:-1]
    
    # Creating the model
    model = Sequential()
    for layer in np.arange(args.n_layers):
        if layer == 0:
            model.add(Dense(args.n_nodes, activation='relu', input_shape=(np.shape(trainX)[1],)))
        else:
             model.add(Dense(args.n_nodes, activation='relu'))
        model.add(Dropout(args.drop_frac))
    model.add(Dense(1, activation='linear'))
    
    # physics-based regularization
    uin1 = K.constant(value=uX1) # input at depth i
    uin2 = K.constant(value=uX2) # input at depth i + 1
    
    yphy1 = K.constant(value = YPhy1)
    yphy2 = K.constant(value = YPhy2)
    
    lam = K.constant(value=args.lamda) # regularization hyper-parameter
    
    uout1 = model(uin1) # model output at depth i
    uout2 = model(uin2) # model output at depth i + 1
    udendiff = (density(uout1) - density(uout2)) # difference in density estimates at every pair of depth values
    
    totloss = combined_loss([udendiff, lam])
    phyloss = phy_loss_mean([udendiff, lam])

    model.compile(loss=totloss,
                  optimizer=optimizer_val,
                  metrics=[phyloss, root_mean_squared_error])

    early_stopping = EarlyStopping(monitor='val_loss_1', patience=args.patience_val, verbose=1)
    
    print('Running...' + optimizer_name)
    history = model.fit(trainX, trainY,
                        batch_size = args.batch_size,
                        epochs = args.epochs,
                        verbose = 0,
                        validation_split = args.val_frac, callbacks=[early_stopping, TerminateOnNaN()])
    
    test_score = model.evaluate(testX, testY, verbose=0)
    
    phy_cons, percent_phy_incon = evaluate_physics_loss(model, uX1, uX2, YPhy1, YPhy2)
    test_rmse = test_score[2]
    train_rmse = history.history['root_mean_squared_error'][-1]
    
    print(" Train RMSE = ", train_rmse)
    print(" Test RMSE = ", test_rmse)
    print(" Physical Consistency = ", phy_cons)
    print(" Percentage Physical Incon = ", percent_phy_incon)
    
    
    exp_name = model_name+args.dataset+optimizer_name + '_drop' + str(args.drop_frac) + '_usePhy' + str(args.use_YPhy) +  '_nL' + str(args.n_layers) + '_nN' + str(args.n_nodes) + '_trsize' + str(args.tr_size) + '_lamda' + str(args.lamda) + '_iter' + str(iteration)
    exp_name = exp_name.replace('.','pt')
    results_name = args.save_dir + exp_name + '_results.mat' # storing the results of the model
    spio.savemat(results_name, 
                 {'train_loss_1':history.history['loss_1'], 
                  'val_loss_1':history.history['val_loss_1'], 
                  'train_rmse':history.history['root_mean_squared_error'], 
                  'val_rmse':history.history['val_root_mean_squared_error'], 
                  'test_rmse':test_score[2]})
    
    return train_rmse, test_rmse, phy_cons, percent_phy_incon



for iteration in range(args.n_iters):
    train_rmse, test_rmse, phy_cons, percent_phy_incon = PGNN_train_test(iteration)
