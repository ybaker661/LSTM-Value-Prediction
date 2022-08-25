import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import date
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.utils import Sequence
from keras.backend import set_value
import random as rnd
from datetime import date
import time
from scipy.io import savemat
from sklearn.model_selection import train_test_split
'''
Function set 1: Generating GT value function and Arbitrage Functions
'''

# Compute current value function using the value function from the next time period
def CalcValueNoUnc(d, c, P, eta, vi, ed, iC, iD):
    """
    Title: Calculate Risk-Neutral value function using deterministic price
    Inputs:
        d - price right now
        c - marginal discharge cost
        P - power rating w.r.t to energy rating and sampling time,
        i.e., 2hr duration battery with 5min resolution -> P = (1/2)/12 
        eta - efficiency
        vi - input value function for the next time period, which equals to
        v_t(e) where e is sampled from 0 to 1 at the granularity e
        ed - granularity at which vi is sampled, in p.u. to energy rating
    Outputs:
        vo - value function for the current time period sampled at ed
    """
    # add a large number of upper and lower v, where the first point is
    # v_t(0-) = +infty, and the second point is v_t(0), the second largest one is
    # v_t(1), and the largest one is v_t(1+) = -infty
    lNum = 1e5*np.ones((1,))
    v_foo = np.concatenate([lNum, vi, -lNum], axis=0)

    # # calculate soc after charge vC = v_t(e+P*eta)
    vC = v_foo[iC]

    # # calculate soc after discharge vC = v_t(e-P/eta)
    vD = v_foo[iD]

    # calculate CDF and PDF
    FtEC = (vi*eta > d).astype(int) # F_t(v_t(e)*eta)
    FtCC = (vC*eta > d).astype(int) # F_t(v_t(e+P*eta)*eta)
    FtED = ((vi/eta + c)*((vi/eta + c) > 0) > d).astype(int) # F_t(v_t(e)/eta + c) 
    FtDD = ((vD/eta + c)*((vD/eta + c) > 0) > d).astype(int) # F_t(v_t(e-P/eta)/eta + c) 

    # calculate terms
    Term1 = vC * FtCC
    Term2 = d*(vC*eta <= d)*(vi*eta > d)/ eta
    Term3 = vi * (FtED - FtEC)
    Term4 = d*(((vi/eta + c)*((vi/eta + c) > 0)) <= d)*(((vD/eta + c)*((vD/eta + c) > 0))>d) * eta
    Term5 = - c * eta * (FtDD - FtED)
    Term6 = vD * (1-FtDD)

    # output new value function samped at ed
    vo = Term1 + Term2 + Term3 + Term4 + Term5 + Term6
    return vo

def ArbValue(lmp, v, e, P, E, eta, c, N):
    """
        Title: Arbitrage test using value function

        lmp: lambda, electricity price over time period t
        v: price function
        e: SoC
        P: P = Pr * Ts; actual power rating taking time step size into account
        E: 1
        eta: eta = .9; # efficiency
        c: c = 10; # marginal discharge cost - degradation
        N: number of SOC samples, 1001
    """

    iE = np.ceil((N-1)*e/E).astype(int) # find the nearest SoC index. iE here is 1 smaller than MATLAB.

    vF = v.copy() # read the value function
    # charge efficiency: iE+1 to end in Matlab, so iE to end here
    vF[iE+1 :] = vF[iE+1 :] * eta
    # discharge efficiency: 1 to iE-1 in Matlab, so 0 to iE-1 (exclusive) here
    vF[0 : iE] = vF[0 : iE] / eta + c

    # charge index
    if len(np.nonzero(vF >= lmp)[0])>0:
        iC = np.max(np.nonzero(vF >= lmp))
    else:
        iC = None

    # discharge index
    if len(np.nonzero(vF <= lmp)[0])>0:
        iD = np.min(np.nonzero(vF <= lmp))
    else:
        iD = None

    # iF = iC*(iC > iE) + iD*(iD < iE) + iE*(iC <= iE)*(iD >= iE);
    if iC is not None:
        if iC > iE:
            iF = iC
        elif iD is not None:
            if iD < iE:
                iF = iD
            else:
                iF = iE
        else:
            iF = iE
    elif iD is not None:
        if iD < iE:
            iF = iD
        else:
            iF = iE
    else:
        iF = iE

    eF = (iF)/(N-1)*E
    eF = max(min(eF, e + P*eta), e-P/eta)
    pF = (e-eF)/eta*((e-eF) < 0) + (e-eF)*eta*((e-eF) > 0)
    
    return eF, pF

def generate_value_function(Ts, P, eta, c, ed, ef, Ne, T, num_segment, tlambda):
    '''
    Generate value function v and dowmsampled value function vAvg
    '''

    start_time = time.time()

    # Set final SoC level
    vEnd = np.zeros(Ne)
    vEnd[0:math.floor(ef * 1001)] = 1e2 # Use 100 as the penalty for final discharge level

    # Define the risk-neutral value function and populate the final column.
    # v[0, 0] is the marginal value of 0% SoC at the beginning of day 1, v[Ne, T]is the maringal value of 100% SoC at the beginning of the last operating day
    v = np.zeros((Ne, T+1)) # initialize the value function series
    v[:, -1] = vEnd  # v.shape == (1001, 210241)

    # Process indices: discretize vt by modeling it as an vector v_{t,j} in which each element is associated with equally spaced SoC samples
    es = np.arange(start=0, stop=1+ed, step=ed)

    # the number of samples is J = 1 + E/ed
    Ne = len(es)

    # Calculate soc after charge vC = v_t(e+P*eta)
    eC = es + P*eta  # [0.0375 0.0385 0.0395 ... 1.0355 1.0365 1.0375]
    iC = np.ceil(eC/ed)
    iC[iC > (Ne+1)] = Ne + 1
    iC[iC < 1] = 0
    # print(iC) # [  38.   39.   40. ... 1002. 1002. 1002.]
    # print(iC.shape) # (1001,)


    # Calculate soc after discharge vC = v_t(e-P/eta)
    eD = es - P/eta
    iD = np.floor(eD/ed)
    iD[iD > (Ne+1)] = Ne + 1
    iD[iD < 1] = 0
    # print(iD) # [  0.   0.   0. ... 951. 952. 953.]
    # print(iD.shape) # (1001,)


    # Populate value function
    for t in reversed(range(0, T)): # start from the last day and move backwards
        vi = v[:, t+1] # input value function of next time stamp
        vo = CalcValueNoUnc(tlambda[int(t+24/Ts)], c, P, eta, vi, ed, iC.astype(int), iD.astype(int))
        v[:,t] = vo # record the result
    # print(v)
    # print(v.shape) # (1001, 210241)
    # print(np.sum(v)) # 6210425677.739915, MATLAB: 6.2082e+09

    end_time = time.time()
    print('Time:', end_time - start_time)

    # Downsample: https://stackoverflow.com/questions/14916545/numpy-rebinning-a-2d-array
    vAvg = v[0:1000, :].reshape([num_segment, int(1000/num_segment), v.shape[1], 1]).mean(3).mean(1)

    return v, vAvg

def generate_val_eval(RTP, DAP, Pr=0.5):
  #Calculating GT Profit for each location in evaluation

  Ts = 1/12 # time step: 5min
  lastDay = date.toordinal(date(2019, 12, 31)) + 366 - 1
  start = date.toordinal(date(2019, 1, 1)) + 366 - 1
  stop = date.toordinal(date(2019, 12, 31)) + 366 - 2
  T = int((stop-start+1)*24/Ts)

  # tlambda: real time price over time period t
  tlambda = RTP[:, (len(RTP[0])-lastDay+start-2):(len(RTP[0])-lastDay+stop+1)] # (288, 731)
  tlambda = tlambda.flatten('F')
  # tlambda_DA: day ahead price over time period t
  tlambda_DA = DAP[:, (len(DAP[0])-lastDay+start-2):(len(DAP[0])-lastDay+stop+1)] # (288, 731)
  tlambda_DA = tlambda_DA.flatten('F') 


  '''
  Set parameters
  '''
  Ts = 1/12
  Pr = Pr  # normalized power rating wrt energy rating (highest power input allowed to flow through particular equipment)
  P = Pr*Ts  # actual power rating taking time step size into account, 0.5*1/12 = 0.041666666666666664
  eta = .9  # efficiency
  c = 10  # marginal discharge cost - degradation
  ed = .001  # SoC sample granularity
  ef = .5  # final SoC target level, use 0 if none (ensure that electric vehicles are sufficiently charged at the end of the period)
  Ne = math.floor(1/ed)+1  # number of SOC samples, (1/0.001)+1=1001
  e0 = .5  # Beginning SoC level


  '''
  Downsample settings
  '''
  num_segment = 50


  v, vAvg = generate_value_function(Ts, P, eta, c, ed, ef, Ne, T, num_segment, tlambda)

  return v, vAvg

'''
Dataset functions
'''


def generate_train(T, DAP, tlambda, start, stop, lastDay, num_DAP, num_RTP, vAvg):
  X_train = np.zeros((T, num_DAP+num_RTP))

  DAP_sub = DAP[::12] # Subsample DAP
  lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+start-2):(len(DAP_sub[0])-lastDay+stop+1)]
  tlambda_DA_sub = lambda_DA_sub.flatten('F')
  for t in range(T):
      X_train[t, 0:num_DAP] = tlambda_DA_sub[int(t/12)+37-num_DAP : int(t/12)+37]

  # RTP: Previous (num_RTP-1) prices + current price
  for t in range(T):
      X_train[t, num_DAP:num_DAP+num_RTP] = tlambda[t+289-num_RTP : t+289]


  y_train = vAvg.T[0:T, :]
  
  print(X_train.shape)
  print(y_train.shape)

  return X_train, y_train

def generate_train_dec(T, DAP, tlambda, start, stop, lastDay, num_DAP, num_RTP, vAvg, dec):
  X_train = np.zeros((T, num_DAP+num_RTP))

  DAP_sub = DAP[::12] # Subsample DAP
  lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+start-2):(len(DAP_sub[0])-lastDay+stop+1)]
  tlambda_DA_sub = lambda_DA_sub.flatten('F')
  for t in range(T):
      X_train[t, 0:num_DAP] = tlambda_DA_sub[int(t/12)+37-num_DAP : int(t/12)+37]

  # RTP: Previous (num_RTP-1) prices + current price
  for t in range(T):
      X_train[t, num_DAP:num_DAP+num_RTP] = tlambda[t+289-num_RTP : t+289]


  y_train = vAvg.T[0:T, :]
  
  print(X_train.shape)
  print(y_train.shape)

  return [X_train, dec], y_train

def generate_train_CNN(T, DAP, tlambda, start, stop, lastDay, num_DAP, num_RTP, vAvg, step = int((5*12))):
  X_train = np.zeros((T, num_DAP+num_RTP))

  DAP_sub = DAP[::12] # Subsample DAP
  lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+start-2):(len(DAP_sub[0])-lastDay+stop+1)]
  tlambda_DA_sub = lambda_DA_sub.flatten('F')
  for t in range(T):
      X_train[t, 0:num_DAP] = tlambda_DA_sub[int(t/12)+37-num_DAP : int(t/12)+37]

  # RTP: Previous (num_RTP-1) prices + current price
  for t in range(T):
      X_train[t, num_DAP:num_DAP+num_RTP] = tlambda[t+289-num_RTP : t+289]


  y_train = vAvg.T[0:T, :]
  x = []
  y = [] 
  for i in range(int(y_train.shape[0]/step)):
    currx = X_train[i*step:(i+1)*step]
    currx = currx[...,np.newaxis]
    curry = y_train[i*step:(i+1)*step]
    curry = curry[...,np.newaxis]
    x.append(currx)
    y.append(curry)
  # np.reshape(X_train, (146, 1440, X_train.shape[1]))
  # np.reshape(y_train, (146, 1440, X_train.shape[1]))
  print(len(x))
  print(x[0].shape)
  print(y[0].shape)
  # print(y_train.shape)

  return x, y

def generate_train_regret(T, DAP, tlambda, 
                  start, stop, lastDay, 
                  num_DAP, num_RTP, vAvg, 
                  step = int((5*12))):

  X_train = np.zeros((T, num_DAP+num_RTP))

  DAP_sub = DAP[::12] # Subsample DAP
  lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+start-2):(len(DAP_sub[0])-lastDay+stop+1)]
  tlambda_DA_sub = lambda_DA_sub.flatten('F')
  for t in range(T):
      X_train[t, 0:num_DAP] = tlambda_DA_sub[int(t/12)+37-num_DAP : int(t/12)+37]

  # RTP: Previous (num_RTP-1) prices + current price
  for t in range(T):
      X_train[t, num_DAP:num_DAP+num_RTP] = tlambda[t+289-num_RTP : t+289]


  y_train = vAvg.T[0:T, :]

  x = []
  y = [] 
  p = []
  for i in range(int(y_train.shape[0]/step)):
    currp = tlambda[i*step:(i+1)*step]
    # np.save(savedir + 'p_' + str(i) + '.npy', currp)
    currx = X_train[i*step:(i+1)*step]
    currx = currx[...,np.newaxis]
    # np.save(savedir + 'x_' + str(i) + '.npy', currx)
    curry = y_train[i*step:(i+1)*step]
    curry = curry[...,np.newaxis]
    # np.save(savedir + 'y_' + str(i) + '.npy', curry)
    x.append(currx)
    y.append(curry)
    p.append(currp)
  
  print(len(x))
  print(x[0].shape)
  print(y[0].shape)
  print(p[0].shape)

  return [x, p], y

'''
custom loss functions
'''


'''
models
'''

def val_MLP( input_size=60, dense_size=60, output_size=50, activation='relu'):
  tf.random.set_seed(1)
  inputs = tf.keras.Input(shape=(input_size,))

  x = tf.keras.layers.Dense(dense_size, activation=activation)(inputs)
  x = tf.keras.layers.Dense(dense_size, activation=activation)(x)
  outputs = tf.keras.layers.Dense(output_size)(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name="value_MLP")
  model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError())

  return model

def val_CNN_LSTM( input_size=(int(5*12) ,60,1), output_size=50, activation='relu', step = 60):

  inputs = tf.keras.Input(shape=input_size)

  #CNN
  # x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_size))(inputs)
  x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_size))(inputs)
  x = TimeDistributed(MaxPooling1D(2))(x)
  x = TimeDistributed(Conv1D(128, kernel_size=3, activation='relu'))(x)
  x = TimeDistributed(MaxPooling1D(2))(x)
  x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu'))(x)
  x = TimeDistributed(MaxPooling1D(2))(x)
  
  x = TimeDistributed(Flatten())(x)
  #LSTM
  x = Bidirectional(LSTM(100, return_sequences=True))(x)
  x = Dropout(0.5)(x)
  x = Bidirectional(LSTM(100, return_sequences=False))(x)
  x = Dropout(0.5)(x)

  #output
  outputs = Dense(int(output_size*step))(x)
  outputs = tf.keras.layers.Reshape((step, output_size))(outputs)
  outputs = tf.expand_dims(
    outputs, axis=-1, name=None)
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name="value_CNN_LSTM")
  
  model.compile(optimizer='adam',
              loss='mse', metrics=['mse', 'mae'])

  return model

def val_CNN_LSTM_v2( input_size=(int(5*12) ,60,1), output_size=50, activation='relu', step = 60):

  inputs = tf.keras.Input(shape=input_size)

  #CNN
  # x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_size))(inputs)
  x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_size))(inputs)
  x = TimeDistributed(MaxPooling1D(2))(x)
  x = TimeDistributed(Conv1D(128, kernel_size=3, activation='relu'))(x)
  x = TimeDistributed(MaxPooling1D(2))(x)
  
  x = TimeDistributed(Flatten())(x)
  #LSTM
  x = Bidirectional(LSTM(100, return_sequences=True))(x)
  x = Dropout(0.5)(x)
  x = Bidirectional(LSTM(100, return_sequences=False))(x)
  x = Dropout(0.5)(x)

  #output
  outputs = Dense(int(output_size*step))(x)
  outputs = tf.keras.layers.Reshape((step, output_size))(outputs)
  outputs = tf.expand_dims(
    outputs, axis=-1, name=None)
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name="value_CNN_LSTM")
  opt = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
    )
  model.compile(optimizer=opt,
              loss='mse', metrics=['mse', 'mae'])

  return model


'''
Evaluation Functions
'''

def evaluate_using_test_MLP(model, DAP, RTP, startTest, stopTest, lastDay, num_DAP, num_RTP, T_CNN=None):

    Ts = 1/12
    Pr = 0.5  # normalized power rating wrt energy rating (highest power input allowed to flow through particular equipment)
    P = Pr*Ts  # actual power rating taking time step size into account, 0.5*1/12 = 0.041666666666666664
    eta = .9  # efficiency
    c = 10  # marginal discharge cost - degradation
    ed = .001  # SoC sample granularity
    ef = .5  # final SoC target level, use 0 if none (ensure that electric vehicles are sufficiently charged at the end of the period)
    Ne = math.floor(1/ed)+1  # number of SOC samples, (1/0.001)+1=1001
    e0 = .5  # Beginning SoC level

    print("="*30)
    print('Evaluating using X_test')
    print("="*30)
    # Select dates
    startTest = date.toordinal(date(2019, 1, 1)) + 366 - 1
    stopTest = date.toordinal(date(2019, 12, 31)) + 366 - 2
    T2 = int((stopTest-startTest+1)*24/Ts)

    X_test = np.zeros((T2, num_DAP+num_RTP))

    # DAP
    DAP_sub = DAP[::12]
    lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+startTest-2):(len(DAP_sub[0])-lastDay+stopTest+1)]
    tlambda_DA_sub = lambda_DA_sub.flatten('F')

    for t in range(T2):
        X_test[t, 0:num_DAP] = tlambda_DA_sub[int(t/12)+37-num_DAP : int(t/12)+37]

    # RTP
    tlambda_RTP_test = RTP[:, (len(RTP[0])-lastDay+startTest-2):(len(RTP[0])-lastDay+stopTest+1)]
    tlambda_RTP_test = tlambda_RTP_test.flatten('F')

    for t in range(T2):
        X_test[t, num_DAP:num_DAP+num_RTP] = tlambda_RTP_test[t+289-num_RTP : t+289]
    
    # print(len(DAP_sub[0])-lastDay+startTest-2)
    # print(len(DAP_sub[0])-lastDay+stopTest+1)
    # print(X_test[0])

    start_time = time.time()
    '''
    Predict
    '''
    v3 = model.predict(X_test)
    v3 = v3.T

    '''
    Perform arbitrage
    '''
    if(T_CNN):
      T2 = T_CNN
    
    eS_test = np.zeros(T2) # generate the SoC series
    pS_test = np.zeros(T2) # generate the power series
    e = e0 # initial SoC
    # print("TEST SHAPE:", eS_test.shape)
    v3 = v3[:, :T2]
    # print("VAL.SHAPE: ", v3.shape)
    for t in range(T2-1): # start from the first day and move forwards
        vv = v3[:, t+1]
        e, p = ArbValue(tlambda_RTP_test[288+t], vv, e, P, 1, eta, c, v3.shape[0])
        eS_test[t] = e # record SoC
        pS_test[t] = p # record Power
    ProfitOutTest = np.sum(pS_test * tlambda_RTP_test[288:T2+288]) - np.sum(c * pS_test[pS_test>0])
    RevenueTest = np.sum(pS_test * tlambda_RTP_test[288:T2+288])

  
    end_time = time.time()
    print(round(ProfitOutTest))
    print(round(RevenueTest))
    print(round(((pS_test>0)*pS_test).sum(0)))
    print('Time:', end_time - start_time)
    arb = [eS_test, pS_test]

    return v3, arb

#CNN_LSTM Evaluation
def CNNLSTM_evaluate(model, DAP, RTP, startTest, stopTest, lastDay, num_DAP, num_RTP):

    Ts = 1/12
    Pr = 0.5  # normalized power rating wrt energy rating (highest power input allowed to flow through particular equipment)
    P = Pr*Ts  # actual power rating taking time step size into account, 0.5*1/12 = 0.041666666666666664
    eta = .9  # efficiency
    c = 10  # marginal discharge cost - degradation
    ed = .001  # SoC sample granularity
    ef = .5  # final SoC target level, use 0 if none (ensure that electric vehicles are sufficiently charged at the end of the period)
    Ne = math.floor(1/ed)+1  # number of SOC samples, (1/0.001)+1=1001
    e0 = .5  # Beginning SoC level
    print("="*30)
    print('Evaluating using X_test')
    print("="*30)
    # Select dates
    startTest = date.toordinal(date(2019, 1, 1)) + 366 - 1
    stopTest = date.toordinal(date(2019, 12, 31)) + 366 - 2
    T2 = int((stopTest-startTest+1)*24/Ts)
    # print(T2)

    X_test = np.zeros((T2, num_DAP+num_RTP))


    # DAP
    DAP_sub = DAP[::12]
    lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+startTest-2):(len(DAP_sub[0])-lastDay+stopTest+1)]
    tlambda_DA_sub = lambda_DA_sub.flatten('F')

    for t in range(T2):
      # print(X_test[t, 0:num_DAP].shape)
      # print(tlambda_DA_sub[int(t/12)+37-num_DAP : int(t/12)+37].shape)
      # print(t)
      X_test[t, 0:num_DAP] = tlambda_DA_sub[int(t/12)+37-num_DAP : int(t/12)+37]


    # RTP
    tlambda_RTP_test = RTP[:, (len(RTP[0])-lastDay+startTest-2):(len(RTP[0])-lastDay+stopTest+1)]
    tlambda_RTP_test = tlambda_RTP_test.flatten('F')

    for t in range(T2):
      # print(tlambda_RTP_test[t+289-num_RTP : t+289].shape)
      X_test[t, num_DAP:num_DAP+num_RTP] = tlambda_RTP_test[t+289-num_RTP : t+289]
    # print('X_test has shape: ', X_test.shape)
    
    # print(len(DAP_sub[0])-lastDay+startTest-2)
    # print(len(DAP_sub[0])-lastDay+stopTest+1)
    # print(X_test[0])
    step = int((5*12))
    x=[]
    for i in range(int(X_test.shape[0]/step)):
      currx = X_test[i*step:(i+1)*step]
      currx = currx[...,np.newaxis]
      x.append(currx)
    x = np.asarray(x)
    # print("x has size:", x.shape)
    start_time = time.time()
    
    '''
    Predict
    '''
    tstart = time.time()
    v3 = model.predict(x)
    # print('time to predict full year:', time.time()-tstart)
    v3 = np.asarray(v3)
    v3 = np.reshape(v3, (int(v3.shape[0]*v3.shape[1]), v3.shape[2]))
    # print('value func has shape:', v3.shape)
    v3 = v3.T
    n=0
    '''
    Perform arbitrage
    '''
    eS_test = np.zeros(v3.shape[1]) # generate the SoC series
    pS_test = np.zeros(v3.shape[1]) # generate the power series
    e = e0 # initial SoC
    
    for t in range(v3.shape[1]-1): # start from the first day and move forwards
        vv = v3[:, t+1]
        e, p = ArbValue(tlambda_RTP_test[288+t], vv, e, P, 1, eta, c, v3.shape[0])
        eS_test[t] = e # record SoC
        pS_test[t] = p # record Power
        # if ((t % 34560 == 0) and (t != 0)):
        #   checkpoint_path = '/content/gdrive/MyDrive/ESS_Proj/CNN_LSTM_CPS_50_0.25/best_live_' +str(n+1)+'.hdf5'
        #   print("month ", 4*(n+1))
        #   print("<==loading best_live_" +str(n+1) +'.hdf5==>')
        #   model.load_weights(checkpoint_path)
        #   n += 1
        #   v3 = model.predict(x)
        #   v3 = np.asarray(v3)
        #   v3 = np.reshape(v3, (int(v3.shape[0]*v3.shape[1]), v3.shape[2]))
        #   v3 = v3.T
    ProfitOutTest = np.sum(pS_test * tlambda_RTP_test[288:v3.shape[1]+288]) - np.sum(c * pS_test[pS_test>0])
    RevenueTest = np.sum(pS_test * tlambda_RTP_test[288:v3.shape[1]+288])

    end_time = time.time()

    print(round(ProfitOutTest))
    print(round(RevenueTest))
    print(round(((pS_test>0)*pS_test).sum(0)))
    print('Time:', end_time - start_time)
    arb = [eS_test, pS_test]
    return v3, arb

def evaluate_using_v(v, DAP, RTP, startTest, stopTest, lastDay, num_DAP, num_RTP, T_CNN=None):

    Ts = 1/12
    Pr = 0.5  # normalized power rating wrt energy rating (highest power input allowed to flow through particular equipment)
    P = Pr*Ts  # actual power rating taking time step size into account, 0.5*1/12 = 0.041666666666666664
    eta = .9  # efficiency
    c = 10  # marginal discharge cost - degradation
    ed = .001  # SoC sample granularity
    ef = .5  # final SoC target level, use 0 if none (ensure that electric vehicles are sufficiently charged at the end of the period)
    Ne = math.floor(1/ed)+1  # number of SOC samples, (1/0.001)+1=1001
    e0 = .5  # Beginning SoC level


    # Select dates
    startTest = date.toordinal(date(2019, 1, 1)) + 366 - 1
    stopTest = date.toordinal(date(2019, 12, 31)) + 366 - 2
    T2 = int((stopTest-startTest+1)*24/Ts)
    print("="*30)


    # DAP
    DAP_sub = DAP[::12]
    lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+startTest-2):(len(DAP_sub[0])-lastDay+stopTest+1)]
    tlambda_DA_sub = lambda_DA_sub.flatten('F')


    # RTP
    tlambda_RTP_test = RTP[:, (len(RTP[0])-lastDay+startTest-2):(len(RTP[0])-lastDay+stopTest+1)]
    tlambda_RTP_test = tlambda_RTP_test.flatten('F')

    
    # print(len(DAP_sub[0])-lastDay+startTest-2)
    # print(len(DAP_sub[0])-lastDay+stopTest+1)
    # print(X_test[0])

    start_time = time.time()
    '''
    Predict
    '''
    v3 = v

    '''
    Perform arbitrage
    '''
    if(T_CNN):
      T2 = T_CNN
    
    eS_test = np.zeros(T2) # generate the SoC series
    pS_test = np.zeros(T2) # generate the power series
    e = e0 # initial SoC
    # print("TEST SHAPE:", eS_test.shape)
    v3 = v3[:, :T2]
    # print("VAL.SHAPE: ", v3.shape)
    for t in range(T2-1): # start from the first day and move forwards
        vv = v3[:, t+1]
        e, p = ArbValue(tlambda_RTP_test[288+t], vv, e, P, 1, eta, c, v3.shape[0])
        eS_test[t] = e # record SoC
        pS_test[t] = p # record Power
    ProfitOutTest = np.sum(pS_test * tlambda_RTP_test[288:T2+288]) - np.sum(c * pS_test[pS_test>0])
    RevenueTest = np.sum(pS_test * tlambda_RTP_test[288:T2+288])

  
    end_time = time.time()
    print(round(ProfitOutTest))
    print(round(RevenueTest))
    print(round(((pS_test>0)*pS_test).sum(0)))
    print('Time:', end_time - start_time)
    

    return 1

def CNNLSTM_evaluate_train(model, DAP, RTP, startTest, stopTest, lastDay, num_DAP, num_RTP, vtar, train, zone, N):


    Ts = 1/12
    Pr = 0.25  # normalized power rating wrt energy rating (highest power input allowed to flow through particular equipment)
    P = Pr*Ts  # actual power rating taking time step size into account, 0.5*1/12 = 0.041666666666666664
    eta = .9  # efficiency
    c = 10  # marginal discharge cost - degradation
    ed = .001  # SoC sample granularity
    ef = .5  # final SoC target level, use 0 if none (ensure that electric vehicles are sufficiently charged at the end of the period)
    Ne = math.floor(1/ed)+1  # number of SOC samples, (1/0.001)+1=1001
    e0 = .5  # Beginning SoC level
    print("="*30)
    print('Evaluating using X_test')
    print("="*30)
    # Select dates
    startTest = date.toordinal(date(2019, 1, 1)) + 366 - 1
    stopTest = date.toordinal(date(2019, 12, 31)) + 366 - 2
    T2 = int((stopTest-startTest+1)*24/Ts)
    # print(T2)

    X_test = np.zeros((T2, num_DAP+num_RTP))
    n = 0

    # DAP
    DAP_sub = DAP[::12]
    lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+startTest-2):(len(DAP_sub[0])-lastDay+stopTest+1)]
    tlambda_DA_sub = lambda_DA_sub.flatten('F')
    
    for t in range(T2):
      # print(X_test[t, 0:num_DAP].shape)
      # print(tlambda_DA_sub[int(t/12)+37-num_DAP : int(t/12)+37].shape)
      # print(t)
      X_test[t, 0:num_DAP] = tlambda_DA_sub[int(t/12)+37-num_DAP : int(t/12)+37]


    # RTP
    tlambda_RTP_test = RTP[:, (len(RTP[0])-lastDay+startTest-2):(len(RTP[0])-lastDay+stopTest+1)]
    tlambda_RTP_test = tlambda_RTP_test.flatten('F')

    for t in range(T2):
      # print(tlambda_RTP_test[t+289-num_RTP : t+289].shape)
      X_test[t, num_DAP:num_DAP+num_RTP] = tlambda_RTP_test[t+289-num_RTP : t+289]
    # print('X_test has shape: ', X_test.shape)
    
    # print(len(DAP_sub[0])-lastDay+startTest-2)
    # print(len(DAP_sub[0])-lastDay+stopTest+1)
    # print(X_test[0])

    # print("x has size:", x.shape)
    start_time = time.time()
    step = int((5*12))
    x=[]
    for i in range(int(X_test.shape[0]/step)):
      currx = X_test[i*step:(i+1)*step]
      currx = currx[...,np.newaxis]
      x.append(currx)
    x = np.asarray(x)
    '''
    Predict
    '''
    tstart = time.time()
    v3 = model.predict(x)
    # print('time to predict full year:', time.time()-tstart)
    v3 = np.asarray(v3)
    v3 = np.reshape(v3, (int(v3.shape[0]*v3.shape[1]), v3.shape[2]))
    # print('value func has shape:', v3.shape)
    v3 = v3.T
    v = []
    vtar = vtar.T[0:v3.shape[1], :]
    for i in range(int(X_test.shape[0]/step)):
      currv = vtar[i*step:(i+1)*step]
      currv = currv[...,np.newaxis]
      v.append(currv)
    v = np.asarray(v)
    
    '''
    Perform arbitrage
    '''
    eS_test = np.zeros(v3.shape[1]) # generate the SoC series
    pS_test = np.zeros(v3.shape[1]) # generate the power series
    e = e0 # initial SoC
    # set_value(model.optimizer.learning_rate, 0.0001)

    month = int(N*30*12*24)
    window = int(month/60)
    for t in range(v3.shape[1]-1): # start from the first day and move forwards
        vv = v3[:, t+1]
        e, p = ArbValue(tlambda_RTP_test[288+t], vv, e, P, 1, eta, c, v3.shape[0])
        eS_test[t] = e # record SoC
        pS_test[t] = p # record Power
        if ((t % month == 0) and (t != 0)):
          checkpoint_path = '/content/gdrive/MyDrive/ESS_Proj/CNN_LSTM_CPS_50_0.25/best_live_' + zone + '_' +str((n+1)) +'.hdf5'
          print("month ", N*(n+1))
          

          tstart = time.time()
    
          x_train, x_val, y_train, y_val = train_test_split(np.concatenate((train[0], x[:(n+1)*window])),
          np.concatenate((train[1], v[:(n+1)*window])), test_size=0.05, shuffle=True)
          model = val_CNN_LSTM()
          cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              monitor='val_loss', verbose=0, 
                                              save_best_only=True, mode='min')
          model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100,batch_size=32, 
                    verbose=0, shuffle =True,
                    callbacks=[cp_callback])
          print("Time to retrain: " + str(time.time()-tstart) + " seconds.")
          print("<==loading best_live_" +str(n+1) +'.hdf5==>')
          model.load_weights(checkpoint_path)
          n += 1
          v3 = model.predict(x)
          v3 = np.asarray(v3)
          v3 = np.reshape(v3, (int(v3.shape[0]*v3.shape[1]), v3.shape[2]))
          v3 = v3.T

    ProfitOutTest = np.sum(pS_test * tlambda_RTP_test[288:v3.shape[1]+288]) - np.sum(c * pS_test[pS_test>0])
    RevenueTest = np.sum(pS_test * tlambda_RTP_test[288:v3.shape[1]+288])

    end_time = time.time()

    print(round(ProfitOutTest))
    print(round(RevenueTest))
    print(round(((pS_test>0)*pS_test).sum(0)))
    print('Time:', end_time - start_time)
    arb = [eS_test, pS_test]
    np.save('/content/gdrive/MyDrive/ESS_Proj/CNN_LSTM_CPS_50_0.25/prof_rev_' + zone + '_ ' + str(N) +'.npy', np.asarray([ProfitOutTest, RevenueTest]))
    return v3, arb, [ProfitOutTest, RevenueTest]