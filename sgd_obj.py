# -*- coding: cp936 -*-
import numpy as np


# 读取数据
obj2vec=np.load('./dataset/obj2vec.npy')
entity_feature0=np.load('./dataset/entity_feature0.npy')
entity_feature1=np.load('./dataset/entity_feature1.npy')
train_entity0_index=np.load('./dataset/train_entity0_index.npy')
train_entity1_index=np.load('./dataset/train_entity1_index.npy')
#dict_obj=np.load('./dataset/dict_obj.npy')

print np.shape(entity_feature0[train_entity0_index[0]])
print np.shape(obj2vec[train_entity0_index[0]])

length=2*len(train_entity0_index)
A=np.ones([length,4097])
B=np.zeros([length,300])
C=np.zeros(length)

#A是把图像特征后面+1再堆叠起来
for i in range(0,len(train_entity0_index)):
    A[i,0:4096]=entity_feature0[i]
    A[i+len(train_entity0_index),0:4096]=entity_feature1[i]
    C[i] = train_entity0_index[i]
    C[i + len(train_entity0_index)] = train_entity1_index[i]
    B[i,:]=obj2vec[train_entity0_index[i]]
    B[i+len(train_entity0_index),:]=obj2vec[train_entity1_index[i]]


#print A[0]-A[1]

#print  A[10,4096]
#print entity_feature0[5]

A=np.mat(A)
B=np.mat(B)
D=np.eye(4097)
alpha=0
X=(A.T*A+alpha*D).I*A.T*B
np.save('./dataset/X.npy',X)
'''

X=np.random.randint(0,2,(4097,300))
#X=np.load('./dataset/X.npy')
loop=1000000
alpha=0.2
ita=1


for count in range(0,loop):
    t = np.random.random_integers(0, length)
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


