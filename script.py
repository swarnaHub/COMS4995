import os
import sys

num = 4
for i in range(16,21):
	train_file = './concatdata/train/train_4_20/'+str(num)+'_'+str(i)
	test_file =  './concatdata/test/cv_'+str(num)
	dev_file =  './concatdata/dev/cv_'+str(num)
	command = 'KERAS_BACKEND=theano python bidirectional_lstm.py '+train_file+' '+dev_file+' '+test_file+' '+str(num)+'_'+str(i)
	os.system(command)