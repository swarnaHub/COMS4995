import numpy as np,sys
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, MaxoutDense,Input, merge,Activation
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
from theano import printing

class AttLayer1(Layer):
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
        super(AttLayer1, self).__init__(**kwargs)

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
        super(AttLayer1, self).build(input_shape)  # be sure you call this somewhere!

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
        #list_of_outputs = [K.sum(weighted_input, axis=1), a]
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        #list_of_shapes = [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        return (input_shape[0], input_shape[-1])

class AttLayer2(Layer):
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
        super(AttLayer2, self).__init__(**kwargs)

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
        super(AttLayer2, self).build(input_shape)  # be sure you call this somewhere!

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
        return a

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

def visualise_weights(pad_length, max_featuress):
	input_ = Input(shape=(maxlen,), dtype='float32')
	input_embed = Embedding(max_features, 128, input_length=pad_length, name='embedding_layer')(input_)
	rnn_encoded = Bidirectional(LSTM(256, return_sequences=True), name='bilstm_layer')(input_embed)
	y_hat = AttLayer2(name='attention_layer')(rnn_encoded)
	model = Model(input=input_, output=y_hat)
	return model



doReShape=False
if doReShape:
  ounits = 2
  activation = "softmax"
else:
  ounits = 1
  activation = "sigmoid"


max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
model1 = Sequential()
model1.add(Embedding(max_features, 128, input_length=maxlen))
model1.add(Bidirectional(LSTM(256,return_sequences=True)))
model1.add(AttLayer1())
model1.add(MaxoutDense(100, W_constraint=maxnorm(2)))
model1.add(Dropout(0.5))
model1.add(Dense(ounits,activity_regularizer=l2(0.0001)))
model1.add(Activation('sigmoid'))
#model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

weightsPath = "./tmp/weights2.hdf5"
model1.load_weights(weightsPath)

model1.summary()

layer_dict1 = dict([(layer.name, layer) for layer in model1.layers])

embedding_weights = layer_dict1['embedding_1'].get_weights()
bidirectional_weights = layer_dict1['bidirectional_1'].get_weights()
attlayer_weights = layer_dict1['attlayer1_1'].get_weights()



model2 = visualise_weights(80, 20000)
layer_dict2 = dict([(layer.name, layer) for layer in model2.layers])
layer_dict2['embedding_layer'].set_weights(embedding_weights)
layer_dict2['bilstm_layer'].set_weights(bidirectional_weights)
layer_dict2['attention_layer'].set_weights(attlayer_weights)

model2.summary()

word2index = {}
X_train,y_train,word2index = rd.read('train_WD',word2index=word2index,startIndex=1)
X_dev,y_dev,_ = rd.read('dev_WD',word2index,None)
X_test,y_test,w2i = rd.read('test_WD',word2index,None)
i2w = {}
for word in w2i:
    x = w2i[word]
    i2w[x] = word
doReShape=False
if doReShape:
  y_train = reshape(y_train)
  y_test = reshape(y_test)
  y_dev = reshape(y_dev)

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
X_dev = sequence.pad_sequences(X_dev, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


print(type(X_test))
print(X_test[0, :])
attention_values = model2.predict(X_test)
print("shape of attention: ", attention_values.shape)
print(attention_values)

#supposedly you want to visualize weights of 8th sentence in test set 
test_sentence_number = 17
for i in range(len(X_test[test_sentence_number])):
    if X_test[test_sentence_number][i]!=0:
        print(i2w[X_test[test_sentence_number][i]],attention_values[test_sentence_number][i][0])
