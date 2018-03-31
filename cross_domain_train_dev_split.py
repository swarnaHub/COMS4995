import random
import numpy
from sklearn.model_selection import KFold
import os

#===============================================================================================================
#Uncomment parts in between equals and run
#Part 1

path = 'wikitalk.txt'
l = []
for line in open(path,'r'):
	l.append((line.split('\t')[0],line.split('\t')[1]))

random.seed(42)

claim = 0
cl = []
noncl = []
for tup in l:
	if int(tup[1])==1:
		claim = claim+1
		cl.append(tup)
	elif int(tup[1])==0:
		noncl.append(tup)
# l = numpy.array(l)
# numpy.random.shuffle(l)

cl = numpy.array(cl)
noncl = numpy.array(noncl)



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

#====================================================================================================================


#===============================================================================================================
#Uncomment parts in between equals and run
#Part 2

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
#===============================================================================================================


#===============================================================================================================
#Uncomment parts in between equals and run
#Part 3

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

#===============================================================================================================






 
		
