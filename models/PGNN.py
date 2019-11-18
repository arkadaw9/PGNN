from __future__ import print_function

import scipy.io as spio
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error

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

def PGNN_train_test(optimizer_name, optimizer_val, drop_frac, use_YPhy, iteration, n_layers, n_nodes, tr_size, lamda, lake_name):
        
    # Hyper-parameters of the training process
    batch_size = 1000
    num_epochs = 1000
    val_frac = 0.1
    patience_val = 500
    
    # Initializing results filename
    exp_name = optimizer_name + '_drop' + str(drop_frac) + '_usePhy' + str(use_YPhy) +  '_nL' + str(n_layers) + '_nN' + str(n_nodes) + '_trsize' + str(tr_size) + '_lamda' + str(lamda) + '_iter' + str(iteration)
    exp_name = exp_name.replace('.','pt')
    results_dir = '../results/'
    model_name = results_dir + exp_name + '_model.h5' # storing the trained model
    results_name = results_dir + exp_name + '_results.mat' # storing the results of the model
    
    # Load features (Xc) and target values (Y)
    data_dir = '../datasets/'
    filename = lake_name + '.mat'
    mat = spio.loadmat(data_dir + filename, squeeze_me=True,
    variable_names=['Y','Xc_doy','Modeled_temp'])
    Xc = mat['Xc_doy']
    Y = mat['Y']
    
    trainX, trainY = Xc[:tr_size,:],Y[:tr_size]
    testX, testY = Xc[tr_size:,:],Y[tr_size:]
    
    # Loading unsupervised data
    unsup_filename = lake_name + '_sampled.mat'
    unsup_mat = spio.loadmat(data_dir+unsup_filename, squeeze_me=True,
    variable_names=['Xc_doy1','Xc_doy2'])
    
    uX1 = unsup_mat['Xc_doy1'] # Xc at depth i for every pair of consecutive depth values
    uX2 = unsup_mat['Xc_doy2'] # Xc at depth i + 1 for every pair of consecutive depth values
    
    if use_YPhy == 0:
    	# Removing the last column from uX (corresponding to Y_PHY)
        uX1 = uX1[:,:-1]
        uX2 = uX2[:,:-1]
    
    # Creating the model
    model = Sequential()
    for layer in np.arange(n_layers):
        if layer == 0:
            model.add(Dense(n_nodes, activation='relu', input_shape=(np.shape(trainX)[1],)))
        else:
             model.add(Dense(n_nodes, activation='relu'))
        model.add(Dropout(drop_frac))
    model.add(Dense(1, activation='linear'))
    
    # physics-based regularization
    uin1 = K.constant(value=uX1) # input at depth i
    uin2 = K.constant(value=uX2) # input at depth i + 1
    lam = K.constant(value=lamda) # regularization hyper-parameter
    uout1 = model(uin1) # model output at depth i
    uout2 = model(uin2) # model output at depth i + 1
    udendiff = (density(uout1) - density(uout2)) # difference in density estimates at every pair of depth values
    
    totloss = combined_loss([udendiff, lam])
    phyloss = phy_loss_mean([udendiff, lam])

    model.compile(loss=totloss,
                  optimizer=optimizer_val,
                  metrics=[phyloss, root_mean_squared_error])

    early_stopping = EarlyStopping(monitor='val_loss_1', patience=patience_val,verbose=1)
    
    print('Running...' + optimizer_name)
    history = model.fit(trainX, trainY,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_split=val_frac, callbacks=[early_stopping, TerminateOnNaN()])
    
    test_score = model.evaluate(testX, testY, verbose=0)
    print('iter: ' + str(iteration) + ' useYPhy: ' + str(use_YPhy) + ' nL: ' + str(n_layers) + ' nN: ' + str(n_nodes) + ' lamda: ' + str(lamda) + ' trsize: ' + str(tr_size) + ' TestRMSE: ' + str(test_score[2]) + ' PhyLoss: ' + str(test_score[1]))
    model.save(model_name)
    spio.savemat(results_name, {'train_loss_1':history.history['loss_1'], 'val_loss_1':history.history['val_loss_1'], 'train_rmse':history.history['root_mean_squared_error'], 'val_rmse':history.history['val_root_mean_squared_error'], 'test_rmse':test_score[2]})


if __name__ == '__main__':
	# Main Function
	
	# List of optimizers to choose from    
    optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
    optimizer_vals = [Adagrad(clipnorm=1), Adadelta(clipnorm=1), Adam(clipnorm=1), Nadam(clipnorm=1), RMSprop(clipnorm=1), SGD(clipnorm=1.), SGD(clipnorm=1, nesterov=True)]
    
    # selecting the optimizer
    optimizer_num = 2 #Adam
    optimizer_name = optimizer_names[optimizer_num]
    optimizer_val = optimizer_vals[optimizer_num]
    
    # Selecting Other Hyper-parameters
    drop_frac = 0 # Fraction of nodes to be dropped out
    use_YPhy = 1 # Whether YPhy is used as another feature in the NN model or not
    n_layers = 2 # Number of hidden layers
    n_nodes = 12 # Number of nodes per hidden layer
    
    #set lamda=0 for pgnn0
    lamda = 1000*0.5 # Physics-based regularization constant

    
    # Iterating over different training fractions and splitting indices for train-test splits
    trsize_range = [5000,2500,1000,500,100]
    iter_range = np.arange(1) # range of iteration numbers for random initialization of NN parameters
    
    #default training size = 5000
    tr_size = trsize_range[0]
    
    #List of lakes to choose from
    lake = ['mendota' , 'mille_lacs']
    lake_num = 0  # 0 : mendota , 1 : mille_lacs
    lake_name = lake[lake_num]
    # iterating through all possible params
    for iteration in iter_range:
        PGNN_train_test(optimizer_name, optimizer_val, drop_frac, use_YPhy, iteration, n_layers, n_nodes, tr_size, lamda, lake_name)
    
    