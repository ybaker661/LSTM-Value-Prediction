import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
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
from scipy.special import expit, logit
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
    # v_foo = np.asarray(list(lNum) + list(vi) + list(-lNum))

    # # calculate soc after charge vC = v_t(e+P*eta)
    vC = v_foo[iC]

    # # calculate soc after discharge vC = v_t(e-P/eta)
    vD = v_foo[iD]

    # # calculate CDF and PDF
    FtEC = (vi*eta > d).astype(int) # F_t(v_t(e)*eta)
    FtCC = (vC*eta > d).astype(int) # F_t(v_t(e+P*eta)*eta)
    FtED = ((vi/eta + c)*((vi/eta + c) > 0) > (d)).astype(int) # F_t(v_t(e)/eta + c) 
    FtDD = ((vD/eta + c)*((vD/eta + c) > 0) > (d)).astype(int) # F_t(v_t(e-P/eta)/eta + c) 

    # calculate terms
    Term1 = vC * FtCC
    Term2 = (d/eta)* (FtEC - FtCC)
    Term3 = vi * (FtED - FtEC)
    Term4 = (d-c) * eta * (FtDD - FtED)
    Term5 = - c * eta * (FtDD - FtED)*0
    Term6 = vD * (1-FtDD)

    # FtEC = (vi*eta > d).astype(int) # F_t(v_t(e)*eta)
    # FtCC = (vC*eta > d).astype(int) # F_t(v_t(e+P*eta)*eta)
    # FtED = ((vi/eta + c)*((vi/eta + c) > 0) > d).astype(int) # F_t(v_t(e)/eta + c) 
    # FtDD = ((vD/eta + c)*((vD/eta + c) > 0) > d).astype(int) # F_t(v_t(e-P/eta)/eta + c) 

    # # calculate terms
    # Term1 = vC * FtCC
    # Term2 = d*(vC*eta <= d)*(vi*eta > d)/ eta
    # Term3 = vi * (FtED - FtEC)
    # Term4 = d*(((vi/eta + c)*((vi/eta + c) > 0)) <= d)*(((vD/eta + c)*((vD/eta + c) > 0))>d) * eta
    # Term5 = - c * eta * (FtDD - FtED)
    # Term6 = vD * (1-FtDD)

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
    if(N==1):
        vC = v*eta
        vD = v/eta+c
        pC = P*(vC > lmp) 
        pD = P*(vD < lmp) 
        ee = e + pC*eta - pD/eta 
        eF = min(max(0, ee), E) 
        pF = (e-eF)/eta*((e-eF) < 0) + (e-eF)*eta*((e-eF) > 0)

        return eF,pF
      # iD = 0
      # iC = 0
      # iN = 0

      # if (vF[0]*eta) >= lmp:
      #   iC = 1
      # elif (vF[0]/eta) <= lmp:
      #   iD = 1


      # if iD:
      #   # eF = max(min(eF, e + P*eta), e-P/eta)
      #   # print("discharging")
      #   eF = e-P/eta
      #   if eF <= 0:
      #     eF = 0
      # elif iC:
      #   eF = e+P*eta
      #   if eF >= E:
      #     eF = E
      # else:
      #   eF = e
      # # print("final S.o.C: ", eF)
      # pF = (e-eF)/eta*((e-eF) < 0) + (e-eF)*eta*((e-eF) > 0)

      # return eF, pF
    


       # read the value function
    vF[iE :] = vF[iE :] * eta
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
    # print("final S.o.C: ", eF)
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
        # vo = CalcValueNoUnc(tlambda[int(t+24/Ts)], c, P, eta, vi, ed, iC.astype(int), iD.astype(int))
        vo = CalcValueNoUnc(tlambda[int(t+24/Ts)], c, P, eta, vi, ed, iC.astype(int), iD.astype(int))

        # vo = CalcValueNoUnc(tlambda[int(t)], c, P, eta, vi, ed, iC.astype(int), iD.astype(int))
        v[:,t] = vo # record the result
    # print(v)
    # print(v.shape) # (1001, 210241)
    # print(np.sum(v)) # 6210425677.739915, MATLAB: 6.2082e+09

    end_time = time.time()
    print('Time:', end_time - start_time)

    # Downsample: https://stackoverflow.com/questions/14916545/numpy-rebinning-a-2d-array
    vAvg = v[:-1, :].reshape([num_segment, int((ed**-1)/num_segment), v.shape[1], 1]).mean(3).mean(1)


    return v, vAvg

'''
Dataset functions
'''

def generate_train_CNN(T, DAP, tlambda, start, stop, lastDay, num_DAP, num_RTP, vAvg, Ts=1/12, TX=False):

  hr = int(1/Ts)
  day = int(24/Ts)
  if TX:
    step = int(num_DAP + num_RTP)
  else:
    step = int(5*12)

  X_train = np.zeros((T, num_DAP+num_RTP))
  if TX:
    tlambda_DA_sub = DAP.flatten('F')
  else:
    DAP_sub = DAP[::hr] # Subsample DAP
    lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+start-2):(len(DAP_sub[0])-lastDay+stop+1)]
    tlambda_DA_sub = lambda_DA_sub.flatten('F')

  for t in range(T):

    X_train[t, 0:num_DAP] = tlambda_DA_sub[int(t*Ts)+(num_RTP+1)-num_DAP : int(t*Ts)+(num_RTP+1)]

  
  for t in range(T):
      X_train[t, num_DAP:num_DAP+num_RTP] = tlambda[t+day+1-num_RTP : t+day+1]


  y_train = vAvg.T[:T, :]
  print("ytrain shape",y_train.shape[0]/step)

  x = []
  y = [] 
  print('datpoints shape', int(y_train.shape[0]/step))
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

'''
custom loss functions
'''


'''
models
'''

def val_CNN_LSTM( input_size=(int(5*12) ,60,1), output_size=50, activation='relu', step = 60, net='vanilla'):

  inputs = tf.keras.Input(shape=input_size)

  #CNN
  # x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_size))(inputs)
  x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_size))(inputs)
  x = TimeDistributed(MaxPooling1D(2))(x)
  x = TimeDistributed(Conv1D(128, kernel_size=3, activation='relu'))(x)
  if net=='vanilla':
    x = TimeDistributed(MaxPooling1D(2))(x)
  x = TimeDistributed(Conv1D(64, kernel_size=3, activation='relu'))(x)
  x = TimeDistributed(MaxPooling1D(2))(x)
  # x = TimeDistributed(Conv1D(32, kernel_size=3, activation='relu'))(x)
  # x = TimeDistributed(MaxPooling1D(2))(x)
  x = TimeDistributed(Flatten())(x)
  #LSTM
  x = Bidirectional(LSTM(100, return_sequences=True))(x)
  x = Dropout(0.5)(x)
  # x = Bidirectional(LSTM(100, return_sequences=True))(x)
  # x = Dropout(0.5)(x)
  x = Bidirectional(LSTM(100, return_sequences=False))(x)
  x = Dropout(0.5)(x)

  #output
  # x = Dense(int(output_size*step), activation='relu')(x)
  # x = Dropout(0.6)(x)
  # outputs = Dense(int(output_size*step), activation='relu')(x)
  outputs = Dense(int(output_size*step))(x)
  outputs = tf.keras.layers.Reshape((step, output_size))(outputs)
  outputs = tf.expand_dims(
    outputs, axis=-1, name=None)
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name="value_CNN_LSTM")
  # opt = tf.keras.optimizers.Adam(
  #   learning_rate=0.0001,
  #   beta_1=0.9,
  #   beta_2=0.999,
  #   epsilon=1e-07,
  #   amsgrad=False,
  #   name='Adam',
  #   )
  model.compile(optimizer='adam',
              loss='mse', metrics=['mse', 'mae'])

  return model



'''
Evaluation Functions
'''



#CNN_LSTM Evaluation THESE FUNCTIONS ARE FOR 5 MIN ARBITRAGE OF 5 MIN VALUATION
def CNNLSTM_evaluate(model, DAP, RTP, 
  startTest, stopTest, lastDay, 
  num_DAP, num_RTP, TX=False, Pr=0.25, Ts=1/12,
  eta=0.9, c=10):

    Ts = Ts
    Pr = Pr # normalized power rating wrt energy rating (highest power input allowed to flow through particular equipment)
    P = Pr*Ts  # actual power rating taking time step size into account, 0.5*1/12 = 0.041666666666666664
    eta = eta  # efficiency
    c = c  # marginal discharge cost - degradation
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
    X_test = np.zeros((T2, num_RTP+num_DAP))
    step = int(12*5)
    day = int(24/Ts)
    hr = int(1/Ts)

    #forming inputted price data into format needed for network
    # DAP
    if TX:
      tlambda_DA_sub = DAP.flatten('F')
    else:
      DAP_sub = DAP[::12]
      lambda_DA_sub = DAP_sub[:, (len(DAP_sub[0])-lastDay+startTest-2):(len(DAP_sub[0])-lastDay+stopTest+1)]
      tlambda_DA_sub = lambda_DA_sub.flatten('F')

    for t in range(T2):
      X_test[t, 0:num_DAP] = tlambda_DA_sub[int(t*Ts)+(num_RTP+1)-num_DAP : int(t*Ts)+(num_RTP+1)]
    # print(X_test[0, 0:num_DAP])

    # RTP
    if TX:
      tlambda_RTP_test = RTP.flatten('F')
    else:
      tlambda_RTP_test = RTP[:, (len(RTP[0])-lastDay+startTest-2):(len(RTP[0])-lastDay+stopTest+1)]
      tlambda_RTP_test = tlambda_RTP_test.flatten('F')

    for t in range(T2):
      # print(tlambda_RTP_test[t+289-num_RTP : t+289].shape)
      X_test[t, num_DAP:num_DAP+num_RTP] = tlambda_RTP_test[t+(day+1)-num_RTP : t+(day+1)]

    if not TX:
      step = int((5*12))
    else:
      step = int(num_RTP+num_DAP)
    x=[]
    for i in range(int(X_test.shape[0]/step)):
      currx = X_test[i*step:(i+1)*step]
      currx = currx[...,np.newaxis]
      x.append(currx)
    x = np.asarray(x)
    # print(x.shape)
    start_time = time.time()
    
    '''
    Predict
    '''
    tstart = time.time()
    v3 = model.predict(x, verbose=0)
    # print('time to predict full year:', time.time()-tstart)
    v3 = np.asarray(v3)
    v3 = np.reshape(v3, (int(v3.shape[0]*v3.shape[1]), v3.shape[2]))
    # print('value func has shape:', v3.shape)
    v3 = v3.T
    # print(v3.shape[1])
    T_CNN = v3.shape[1]
    n=0

    '''
    Perform arbitrage
    '''
    T2 = v3.shape[1] - 1
    eS_test = np.zeros(v3.shape[1]) # generate the SoC series
    pS_test = np.zeros(v3.shape[1]) # generate the power series
    e = e0 # initial SoC

    for t in range(T2): # start from the first day and move forwards
        vv = v3[:, t+1]
        e, p = ArbValue(tlambda_RTP_test[day+t], vv, e, P, 1, eta, c, v3.shape[0])
        eS_test[t] = e # record SoC
        pS_test[t] = p # record Power


    ProfitOutTest = np.sum(pS_test * tlambda_RTP_test[day:v3.shape[1]+day]) - np.sum(c * pS_test[pS_test>0])
    RevenueTest = np.sum(pS_test * tlambda_RTP_test[day:v3.shape[1]+day])



    end_time = time.time()

    print("Profit: ", round(ProfitOutTest))
    print("Revenue: ", round(RevenueTest))
    print(round(((pS_test>0)*pS_test).sum(0)))
    print('Time:', end_time - start_time)
    arb = [eS_test, pS_test]
    return v3, arb, T_CNN



def evaluate_using_v(RTP, v, eta, c,  T,  Ts=1/12, Pr=0.25):

    Ts = Ts
    Pr = Pr  # normalized power rating wrt energy rating (highest power input allowed to flow through particular equipment)
    P = Pr*Ts  # actual power rating taking time step size into account, 0.5*1/12 = 0.041666666666666664
    eta = eta  # efficiency
    c = c  # marginal discharge cost - degradation
    ed = .001  # SoC sample granularity
    ef = .5  # final SoC target level, use 0 if none (ensure that electric vehicles are sufficiently charged at the end of the period)
    Ne = math.floor(1/ed)+1  # number of SOC samples, (1/0.001)+1=1001
    e0 = .5  # Beginning SoC level


    T2 = T
    print("="*30)


    tlambda_RTP_test = RTP.flatten('F')
    # print(tlambda_RTP_test.shape)

    start_time = time.time()



    
    eS_test = np.zeros(T2) # generate the SoC series
    pS_test = np.zeros(T2) # generate the power series
    e = e0 # initial SoC
    # print("TEST SHAPE:", eS_test.shape)

    hr = int(1/Ts)
    day = int(24/Ts)

    for t in range(T2): # start from the first day and move forwards
        vv = v[:, t+1]
        e, p = ArbValue(tlambda_RTP_test[day+t], vv, e, P, 1, eta, c, v.shape[0])
        eS_test[t] = e # record SoC
        pS_test[t] = p # record Power
    ProfitOutTest = np.sum(pS_test * tlambda_RTP_test[day:T2+day]) - np.sum(c * pS_test[pS_test>0])
    RevenueTest = np.sum(pS_test * tlambda_RTP_test[day:T2+day])

  
    end_time = time.time()
    print(round(ProfitOutTest))
    print(round(RevenueTest))
    print(round(((pS_test>0)*pS_test).sum(0)))
    print('Time:', end_time - start_time)
    

    return ProfitOutTest

def hr_rebin(v, Ts):
  vhr= np.zeros((v.shape[0], int(v.shape[1]*Ts)))
  # print(vhr.shape)
  for i in range(int(v.shape[1]*Ts)):
    vhr[:, i] = np.mean(v[:, i*int(1/Ts):(i+1)*int(1/Ts)], axis=1)
  return vhr

def hr_rebin_1(RTP, Ts):
  RTPhr = np.zeros((int(RTP.shape[0]*Ts), RTP.shape[1]))

  for i in range(int(RTP.shape[0]*Ts)):
    RTPhr[i, :] = np.mean(RTP[i*int(1/Ts):(i+1)*int(1/Ts), :], axis=0)
  return RTPhr



def evaluate_using_v_hr(lmp, v, 
          eta, c, T, 
           e0, Ts=1/12, Pr=0.25):

    Ts = Ts
    Pr = Pr  # normalized power rating wrt energy rating (highest power input allowed to flow through particular equipment)
    P = Pr*Ts  # actual power rating taking time step size into account, 0.5*1/12 = 0.041666666666666664
    eta = eta  # efficiency
    c = c  # marginal discharge cost - degradation
    ed = 0.001  # SoC sample granularity
    ef = .5  # final SoC target level, use 0 if none (ensure that electric vehicles are sufficiently charged at the end of the period)
    Ne = math.floor(1/ed)+1  # number of SOC samples, (1/0.001)+1=1001
    e0 = .5  # Beginning SoC level
    print("="*30)

    rtp = lmp.flatten('F')


    eS = np.zeros(T) # generate the SoC series
    pS = np.zeros(T) # generate the power series
    e = e0
    m=0
    day = int(24/Ts)
    for t in range(T):
      vv = v[:, int(np.floor(m+1))]
      e, p = ArbValue(rtp[t+day], vv, e, P, 1, eta, c, v.shape[0])
      eS[t] = e
      pS[t] = p
      m+=Ts
    ProfitOutTest = np.sum(pS * rtp[day:T+day]) - np.sum(c * pS[pS>0])
    RevenueTest = np.sum(pS * rtp[day:T+day])

  
    end_time = time.time()
    print("Profit: ", round(ProfitOutTest))
    print("Revenue: ", round(RevenueTest))
    # print(pS.shape)
    return ProfitOutTest
#jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0