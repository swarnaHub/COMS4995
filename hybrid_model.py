from __future__ import print_function
import numpy as np,sys
np.random.seed(1337)  

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout,Merge, Flatten,Embedding, Bidirectional, LSTM, MaxoutDense,Input, merge,Activation
from keras.callbacks import ModelCheckpoint
import readData as rd
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer
from keras import backend as K
from keras import initializations
from keras.models import Sequential ,Model
from keras.regularizers import l2
from keras.constraints import maxnorm


#code for Attention layer 
#Implementation of word level attention from 
#http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

class AttLayer(Layer):
    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.init = initializations.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = K.dot(uit, self.u)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])




def threshold(x):
  if x>0.5: return 1
  else: return 0

def reshape(x):
    m = len(x)
    x_mod = np.zeros((m,2))
    for i in xrange(m):
        if x[i] == 0:
            x_mod[i,:] = np.array([1,0])
        else:
            x_mod[i,:] = np.array([0,1])
    return x_mod


max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

#reads the data from train , dev and test and assigns every word an index from the vocabulary
word2index = {}
X_train,y_train,word2index = rd.read(sys.argv[1],word2index=word2index,startIndex=1)
X_dev,y_dev,_ = rd.read(sys.argv[2],word2index,None)
X_test,y_test,_ = rd.read(sys.argv[3],word2index,None)

data_train ,data_dev, data_test,W_arg , vocab_arg = rd.getArgLex(sys.argv[1],sys.argv[2],sys.argv[3])




doReShape=False
if doReShape:
  y_train = reshape(y_train)
  y_test = reshape(y_test)
  y_dev = reshape(y_dev)

#pad sequence to ensure all are of same length
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
X_dev = sequence.pad_sequences(X_dev, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


#Deeplearning model
#EMBEDDING--->BILSTM--->ATTENTION--->DENSELAYER--->SOFTMAX--->(prediction)
model1 = Sequential()
model1.add(Embedding(max_features, 128, input_length=maxlen))
model1.add(Bidirectional(LSTM(256,return_sequences=True)))
model1.add(Dropout(0.25))
model1.add(AttLayer())
print(model1.summary())




model2 = Sequential()
model2.add(Embedding(len(vocab_arg)+1, 300, weights=[W_arg] ,input_length=data_train.shape[1], trainable=True))
model2.add(Dense(100))
model2.add(Dropout(0.25))
model2.add(Flatten())

if doReShape:
  ounits = 2
  activation = "softmax"
else:
  ounits = 1
  activation = "sigmoid"

model = Sequential()
model.add(Merge([model1,model2], mode='concat'))
model.add(Dense(ounits,activity_regularizer=l2(0.0001)))
model.add(Activation('relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#checkpoint best model
print('Train...X')
weightsPath = './tmp2/weights'+sys.argv[4]+'.hdf5'
checkpointer = ModelCheckpoint(filepath=weightsPath, verbose=1, save_best_only=True)
model.fit([X_train,data_train], y_train, batch_size=batch_size, nb_epoch=5, validation_data=([X_dev,data_dev], y_dev),callbacks=[checkpointer])

model.load_weights(weightsPath)
# scoreBest, accBest = model.evaluate([X_test,data_test], y_test,
#                             batch_size=batch_size)

# print('Test score:', scoreBest)
# print('Test accuracy:', accBest)
pTest = model.predict_on_batch([X_test,data_test])
pDev = model.predict_on_batch([X_dev,data_dev])
#save test predictions to a file
f = open('./indomain_pred3/bilstm_pred_'+sys.argv[4],'w')
predsTest = []
predsDev = []
for i in xrange(len(pTest)):
  if doReShape: pr = str(np.argmax(pTest[i]))
  else: pr = str(threshold(pTest[i][0]))
  predsTest.append( pr )
for i in xrange(len(pDev)):
  if doReShape: pr = str(np.argmax(pDev[i]))
  else: pr = str(threshold(pDev[i][0]))
  predsDev.append( pr )
for p in predsTest:
  f.write(str(p)+'\n')