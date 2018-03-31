import os
import sys

for num in range(9,11):
	for i in range(1,21):
		train_file = './concatdata/train/train_'+str(num)+'_20/'+str(num)+'_'+str(i)
		test_file =  './concatdata/test/cv_'+str(num)
		dev_file =  './concatdata/dev/cv_'+str(num)
		command = 'KERAS_BACKEND=theano python bidirectional_lstm.py '+train_file+' '+dev_file+' '+test_file+' '+str(num)+'_'+str(i)
		os.system(command)
