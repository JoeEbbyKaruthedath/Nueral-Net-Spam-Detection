
"""
@author: JoeEbbyKaruthedath
"""
#%% import packages, set some parameters, and get the data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from tensorflow.keras import optimizers
import os

os.getcwd()
os.chdir("E:/Machine Learning/Assignment")

NEpochs = 10000
BatchSize=250
Optimizer=optimizers.RMSprop(lr=0.001)

# Read in the data

TrainDF = pd.read_csv('NNHWTrain.csv',sep=',',header=0,quotechar='"')
list(TrainDF)
ValDF = pd.read_csv('NNHWVal.csv',sep=',',header=0,quotechar='"')
list(ValDF)
TestDF = pd.read_csv('NNHWTest.csv',sep=',',header=0,quotechar='"')
list(TestDF)

#

TrIsSpam = np.array(TrainDF['IsSpam'])

TrX = np.array(TrainDF.iloc[:,:-1])

# **** code to rescale the training X data goes here
TrXrsc = (TrX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(TrXrsc.shape)
print(TrXrsc.min(axis=0))
print(TrXrsc.max(axis=0))
# No need to rescale the Y because it is already 0 and 1. But check
print(TrIsSpam.min())
print(TrIsSpam.max())

# Rescale the validation data

ValIsSpam = np.array(ValDF['IsSpam'])
ValX = np.array(ValDF.iloc[:,:-1])

ValXrsc = (ValX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(ValXrsc.shape)
print(ValXrsc.min(axis=0))
print(ValXrsc.max(axis=0))

# code to rescale the test X data goes here

# Rescale the test data

TestIsSpam = np.array(TestDF['IsSpam'])

TestX = np.array(TestDF.iloc[:,:-1])

# ****code to rescale the test X data goes here
TestXrsc = (ValX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(TestXrsc.shape)
print(TestXrsc.min(axis=0))
print(TestXrsc.max(axis=0))
#%% Set up Neural Net Model

# ****code to set up and compile the neural net model goes here

SpamNN = Sequential()

SpamNN.add(Dense(units=20,input_shape=(TrXrsc.shape[1],),activation="relu",use_bias=True))
SpamNN.add(Dense(units=1,activation="sigmoid",use_bias=True))

SpamNN.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy','accuracy'])
#%% Fit NN Model

from keras.callbacks import EarlyStopping

StopRule = EarlyStopping(monitor='val_loss',mode='min',verbose=0,patience=100,min_delta=0.0)

# ****code to fit the neural net model goes here
FitHist = SpamNN.fit(TrXrsc,TrIsSpam,validation_data=(ValXrsc,ValIsSpam), epochs=NEpochs,batch_size=BatchSize,verbose=0, callbacks=[StopRule])


print("Number of Epochs = "+str(len(FitHist.history['accuracy'])))
print("Final validation accuracy: "+str(FitHist.history['val_accuracy'][-1]))
#%% Make Predictions

# **** Your code to compute the predicted probabilities goes here.
# Do not change the variable names or the code in the next block will not work.

TrP = SpamNN.predict(TrXrsc,batch_size=TrXrsc.shape[0])
ValP = SpamNN.predict(ValXrsc,batch_size=ValXrsc.shape[0])
TestP = SpamNN.predict(TestXrsc,batch_size= TestXrsc.shape[0])

#%% Write out prediction

TrainDF['TrP'] = TrP.reshape(-1)
ValDF['ValP'] = ValP.reshape(-1)
TestDF['TestP'] = TestP.reshape(-1)

TrainDF.to_csv('SpamNNWideTrainDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)
ValDF.to_csv('SpamNNWideValDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)
TestDF.to_csv('SpamNNWideTestDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)

print(SpamNN.summary())
#%% repeating, but with a different structure for the NN

Spam2NN = Sequential()

Spam2NN.add(Dense(units=4,input_shape=(TrXrsc.shape[1],),activation="relu",use_bias=True))
Spam2NN.add(Dense(units=4,activation="relu",use_bias=True))
Spam2NN.add(Dense(units=4,activation="relu",use_bias=True))
Spam2NN.add(Dense(units=4,activation="relu",use_bias=True))
Spam2NN.add(Dense(units=4,activation="relu",use_bias=True))
Spam2NN.add(Dense(units=1,activation="sigmoid",use_bias=True))

Spam2NN.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy','accuracy'])

from keras.callbacks import EarlyStopping

StopRule = EarlyStopping(monitor='val_loss',mode='min',verbose=0,patience=100,min_delta=0.0)

# code to fit the neural net model
FitHist = Spam2NN.fit(TrXrsc,TrIsSpam,validation_data=(ValXrsc,ValIsSpam), epochs=NEpochs,batch_size=BatchSize,verbose=0, callbacks=[StopRule])


TrP = Spam2NN.predict(TrXrsc,batch_size=TrXrsc.shape[0])
ValP = Spam2NN.predict(ValXrsc,batch_size=ValXrsc.shape[0])
TestP = Spam2NN.predict(TestXrsc,batch_size= TestXrsc.shape[0])


TrainDF['TrP'] = TrP.reshape(-1)
ValDF['ValP'] = ValP.reshape(-1)
TestDF['TestP'] = TestP.reshape(-1)

TrainDF.to_csv('SpamNNDeepTrainDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)
ValDF.to_csv('SpamNNDeepValDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)
TestDF.to_csv('SpamNNDeepTestDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)

print(Spam2NN.summary())
print("Number of Epochs = "+str(len(FitHist.history['accuracy'])))
print("Final validation accuracy: "+str(FitHist.history['val_accuracy'][-1]))
