from itertools import izip
num = 3 

a = ['bilstm_pred_'+str(num)+'_'+str(j) for j in range(1,21)]
f = open('./indomain_pred/maj_'+str(num),'w')
print a

c = []
for i in range(0,859):
	c.append([])
for i in range(0,859):
	c[i].append(0)
	c[i].append(0)


for i in range(len(a)):
	count = 0
	with open('./indomain_pred/'+a[i]) as t1:
		for line in t1:
			line = int(line.strip())
			c[count][line] = c[count][line]+1
			count = count+1

print c[0:10]

for i in range(len(c)):
	if c[i][0]>c[i][1]:
		f.write(str(0)+'\n')
	else:
		f.write(str(1)+'\n')
