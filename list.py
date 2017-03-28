#coding:utf-8
#from gensim.models.word2vec import  Word2Vec
import scipy.io as sio
import numpy as np

matfn='./dataset/objectListN.mat'
matr='./dataset/relationListN.mat'
objdata=sio.loadmat(matfn)['objectListN'][0]
relation=sio.loadmat(matr)['relationListN'][0]

dict_obj={}
dict_relation={}
for i in range(0,len(objdata)):
    dict_obj[objdata[i][0]]=i

for i in range(0,len(relation)):
    dict_relation[relation[i][0]]=i
    print relation[i][0]


train_entity0=np.load('./dataset/train_entity0.npy')
train_entity1=np.load('./dataset/train_entity1.npy')
train_relation=np.load('./dataset/train_relation.npy')

train_entity0_index=[]
for i in range(0,len(train_entity0)):
    train_entity0_index.append(dict_obj[train_entity0[i]])

train_entity1_index=[]
for i in range(0,len(train_entity1)):
    train_entity1_index.append(dict_obj[train_entity1[i]])

train_relation_index=[]
for i in range(0,len(train_entity1)):
    train_relation_index.append(dict_relation[train_relation[i]])
print train_relation[0]
print train_relation_index[0]

np.save('./dataset/train_entity0_index',train_entity0_index)
np.save('./dataset/train_entity1_index',train_entity1_index)
np.save('./dataset/train_relation_index',train_relation_index)


test_entity0=np.load('./dataset/test_entity0.npy')
test_entity1=np.load('./dataset/test_entity1.npy')
test_relation=np.load('./dataset/test_relation.npy')

test_entity0_index=[]
for i in range(0,len(test_entity0)):
    test_entity0_index.append(dict_obj[test_entity0[i]])

test_entity1_index=[]
for i in range(0,len(test_entity1)):
    test_entity1_index.append(dict_obj[test_entity1[i]])

test_relation_index=[]
for i in range(0,len(test_entity1)):
    test_relation_index.append(dict_relation[test_relation[i]])
print test_relation[0]
print test_relation_index[0]

np.save('./dataset/test_entity0_index',test_entity0_index)
np.save('./dataset/test_entity1_index',test_entity1_index)
np.save('./dataset/test_relation_index',test_relation_index)