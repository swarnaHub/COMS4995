import random
import numpy
from sklearn.model_selection import KFold
import os


path = 'wikitalk.txt' #change the file name
l = []
for line in open(path,'r'):
	l.append((line.split('\t')[0],line.split('\t')[1]))  #seperate sentence and labels and put in a tuplr

random.seed(42)

#Segregate claims and non claims
claim = 0
cl = []
noncl = []
for tup in l:
	if int(tup[1])==1:
		claim = claim+1
		cl.append(tup)
	elif int(tup[1])==0:
		noncl.append(tup)

cl = numpy.array(cl)
noncl = numpy.array(noncl)


#Bucketize claims and non claims into 10 buckets
#add claims and nonclaims into each of these 10 buckets
cv_object1 = KFold(n_splits=10, shuffle=True, random_state=42)
count = 1
for train_index1, test_index1 in cv_object1.split(cl):
		f = open('./data/cv_'+str(count),'w')
		for k in range(len(test_index1)):
			f.write(cl[test_index1[k]][0]+'\t'+cl[test_index1[k]][1])
		count = count+1

cv_object2 = KFold(n_splits=10, shuffle=True, random_state=42)
count = 1
for train_index2, test_index2 in cv_object2.split(noncl):
		f = open('./data/cv_'+str(count),'a')
		for k in range(len(test_index2)):
			f.write(noncl[test_index2[k]][0]+'\t'+noncl[test_index2[k]][1])
		count = count+1



#generate 9 random numbers for from 1 to 10 for train , and the remaining 1 number is dev
#using these numbers concat the files to create train files 
#perform this experiment 10 times to get 10 train & dev file such that the ratio 9:1 is preserved

a = [1,2,3,4,5,6,7,8,9,10]
for i in range(1,11):
	x = random.sample(range(1,len(a)+1),9)
	notin = []
	for num in a:
		if num not in x:
			notin.append(num)
	f = open('./concatdata/train/cv_'+str(i),'a')
	for num in x:
		f1 = open('./data/cv_'+str(num),'r')
		for line in f1:
			f.write(line)
	f = open('./concatdata/dev/cv_'+str(i),'a')
	f1 = open('./data/cv_'+str(notin[0]),'r')
	for line in f1:
		f.write(line)


#Now that we have the train file we create 20 ensemble train files where we randomly
#sample non claims equal to the  number of claims so that the ration is 1:1 and the datset  balanced 
for i in range(1,11):
	f = open('./concatdata/train/cv_'+str(i),'r')
	cl = []
	noncl = []
	for line in f:
		if int(line.strip().split('\t')[1]) == 1:
			cl.append(line)
		else:
			noncl.append(line)
	os.makedirs('./concatdata/train/'+'train_'+str(i)+'_20')
	for j in range(1,21):
		idx = random.sample(range(0,len(noncl)),len(cl))
		a = []
		for c in cl:
			a.append(c)
		for nonc in idx:
			a.append(noncl[nonc])
		numpy.random.shuffle(a)
		g = open('./concatdata/train/'+'train_'+str(i)+'_20'+'/'+str(i)+'_'+str(j),'w')
		for line in a:
			g.write(line)







 
		
