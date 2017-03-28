#coding:utf-8
from gensim.models.word2vec import  Word2Vec
import scipy.io as sio
import numpy as np

matfn='./dataset/objectListN.mat'
matr='./dataset/relationListN.mat'
objdata=sio.loadmat(matfn)['objectListN'][0]
relation=sio.loadmat(matr)['relationListN'][0]

obj2vec=np.zeros([100,300])
relation2vec=np.zeros([70,300])

model=Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
'''
s=objdata[47][0].split()
t=np.zeros([1,300]);
for j in s:
    t=t+model[j].copy()

print t
'''



#model.most_similar(positive=['woman'], negative=['man'], topn=5)
for i in range(0,len(objdata)):
    s=objdata[i][0].split()
    t = np.zeros([1, 300])
    for j in s:
        t = t + model[j].copy()
    obj2vec[i] = t.copy()/np.linalg.norm(t.copy())
#obj2vec[99]=model['suitcase']
#print obj2vec[99]
np.save('./dataset/obj2vec.npy',obj2vec)

for i in range(0,len(relation)):
    s = relation[i][0].split()
    t = np.zeros([1, 300])
    for j in s:
        if j in model:
            t = t + model[j].copy()
    relation2vec[i] = t.copy()/np.linalg.norm(t.copy())

np.save('./dataset/relation2vec.npy',relation2vec)

