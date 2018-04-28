import gensim
import numpy as np

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./glove.6B.300d.txt')
embedding = []
for line in open('real-arg-lexicons.txt','r'):
	line = line.strip().split()
	if len(line)==1:
		embedding.append(word2vec_model[line[0]])
	else:
		a = None
		for i in range(len(line)):
			if i==0:
				a = word2vec_model[line[i]]
			else:
				a = np.add(a,word2vec_model[line[i]])
		a = a/len(line)
		embedding.append(a)

f = open('embeddings.txt','w')
for emb in embedding:
	s = ','.join([str(x) for x in emb])
	f.write(s+'\n')










