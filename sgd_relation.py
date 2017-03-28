# -*- coding: cp936 -*-
import numpy as np


# 读取数据
obj2vec=np.load('./dataset/obj2vec.npy')
relation_feature=np.load('./dataset/relation_feature.npy')
relation2vec=np.load('./dataset/relation2vec.npy')
R=np.zeros([70,600])
for i in range(0,70):
    R[i,0:300]=relation2vec[i]
    R[i,300:600]=relation2vec[i]


#entity_feature0=np.load('./dataset/entity_feature0.npy')
#entity_feature1=np.load('./dataset/entity_feature1.npy')
train_entity0_index=np.load('./dataset/train_entity0_index.npy')
train_entity1_index=np.load('./dataset/train_entity1_index.npy')
train_relation_index=np.load('./dataset/train_relation_index.npy')
#dict_obj=np.load('./dataset/dict_obj.npy')

print np.shape(relation_feature[train_entity0_index[0]])
print np.shape(obj2vec[train_entity0_index[0]])

length=len(train_entity0_index)
A=np.ones([length,4097])
B=np.zeros([length,600])
C=np.zeros([length,600])
for i in range(0,len(train_entity0_index)):
    A[i,0:4096]=relation_feature[i]
    C[i,0:300]=obj2vec[train_entity0_index[i]]
    C[i,300:600]=obj2vec[train_entity1_index[i]]
    B[i]=C[i]+R[train_relation_index[i]]

#np.save('./dataset/C.npy',C)



#print A[0]-A[1]

#print  A[10,4096]
#print entity_feature0[5]

A=np.mat(A)
B=np.mat(B)
X_R=(A.T*A).I*A.T*B
np.save('./dataset/X_R.npy',X_R)
'''
X_R=np.load('./dataset/X_R.npy')
print sum(A.dot(X_R)-B)


#X=np.random.randint(0,2,(4097,300))
X=np.load('./dataset/X.npy')
loop=1000000
alpha=0.2
ita=0.1


for count in range(0,loop):
    t = np.random.random_integers(0, length-1)
    p=np.mat(A[t])
    p.reshape([1,4097])
    y=p*X
    b = np.linalg.norm(y)
    f=0
    for i in range(0,100):
        if i!=C[t]:
            a=-y.dot(B[t])+y.dot(obj2vec[i])
            temp=alpha+a/b
            # ReLU
            if temp >0:
                f=f+temp
                a1=-B[t]+obj2vec[i]
                b1=y/b
                diff=(b*a1-a*b1)/(b*b)
                q=np.mat(diff)
                #q.reshape([300,1])
                X=X-ita*p.T*q

    print f
    if count%100==0:
        print 'count=%d,save!' %count
        np.save('./dataset/X.npy',X)

'''

