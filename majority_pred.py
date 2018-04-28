from itertools import izip

for num in range(1,11):
	a = ['bilstm_pred_'+str(num)+'_'+str(j) for j in range(1,21)]
	f = open('./indomain_pred2/maj_'+str(num),'w')

	num_lines = 0
	for line in open('./indomain_pred2/'+a[0],'r'):
		num_lines = num_lines+1


	c = []
	for i in range(0,num_lines):
		c.append([])
	for i in range(0,num_lines):
		c[i].append(0)
		c[i].append(0)


	for i in range(len(a)):
		count = 0
		with open('./indomain_pred2/'+a[i]) as t1:
			for line in t1:
				line = int(line.strip())
				c[count][line] = c[count][line]+1
				count = count+1


	for i in range(len(c)):
		if c[i][0]>c[i][1]:
			f.write(str(0)+'\n')
		else:
			f.write(str(1)+'\n')