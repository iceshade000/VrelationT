# -*- coding: cp936 -*-
import numpy as np


# 读取数据
entity_feature0=np.load('./dataset/entity_feature0.npy')
entity_feature1=np.load('./dataset/entity_feature1.npy')
train_entity0_index=np.load('./dataset/train_entity0_index.npy')
train_entity1_index=np.load('./dataset/train_entity1_index.npy')
train_relation_index=np.load('./dataset/train_relation_index.npy')
relation_feature=np.load('./dataset/relation_feature.npy')



#使用语言先验初始化
X=np.load('./dataset/X.npy')
X_R=np.load('./dataset/X_R_all.npy')
relation2vec=np.load('./dataset/relation2vec.npy')

R=np.load('./dataset/R_all.npy')


'''
R=np.zeros([70,600])
for i in range(0,70):
    R[i,0:300]=relation2vec[i]
    R[i,300:600]=relation2vec[i]
'''


length=len(entity_feature0)
print length

A=np.ones([length,4097])
B=np.ones([length,4097])
C=np.ones([length,4097])

A[0:length,0:4096]=entity_feature0
B[0:length,0:4096]=entity_feature1
C[0:length,0:4096]=relation_feature



loop=1000000
alpha=1e-6

gamma=0.4


num=0

''''''
for count in range(1,loop):
    temp = np.random.random_integers(0, length-1)
    AB=np.zeros([1,600])
    AB[0,0:300]=A[temp].dot(X)
    AB[0,300:600]=B[temp].dot(X)
    f=C[temp].dot(X_R)-AB
    right=train_relation_index[temp]
    f_right=f-R[right]
    score_r=np.linalg.norm(f_right)

    sum=0
    for i in range (0,70):
        if i!=right:
            f_false=f-R[i]
            score_f=np.linalg.norm(f_false)
            if gamma + score_r - score_f > 0:
                sum = sum +gamma + score_r - score_f
                R[right] = R[right] + alpha * f_right / score_r
                R[i]=R[i]-alpha*f_false/score_f
                X_R = X_R - alpha * np.mat(C[temp]).T * (f_right/score_r - f_false/score_f)

    print 'count=%d,sum=%f' %(count,sum)

    if count%1000==0:
        print 'count=%d , save!' %count
        np.save('./dataset/X_R_all.npy',X_R)
        np.save('./dataset/R_all.npy',R)




