
from sklearn.metrics import classification_report , precision_recall_fscore_support,precision_score,recall_score,f1_score

f = open('classification_report.txt','w')
num = 2 
for i in range(1,num):
	y_pred = []
	y_test = []

	for line in open('./concatdata/test/cv_'+str(i),'r'):
		line = line.strip().split('\t')
		y_test.append(int(line[1]))



	for line in open('./indomain_pred/maj_'+str(i),'r'):
		line = line.strip()
		y_pred.append(int(line))



	f.write(str(classification_report(y_test, y_pred)+'\n'))
	f.write(str(f1_score(y_test, y_pred, average='weighted'))+'\n')
	f.write("-------------------------------------------------\n")
