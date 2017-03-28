# -*- coding: cp936 -*-
import numpy as np


# 读取数据
#entity_feature0=np.load('./dataset/entity_feature0.npy')
#entity_feature1=np.load('./dataset/entity_feature1.npy')
train_entity0_index=np.load('./dataset/train_entity0_index.npy')
train_entity1_index=np.load('./dataset/train_entity1_index.npy')
train_relation_index=np.load('./dataset/train_relation_index.npy')
relation_feature=np.load('./dataset/relation_feature.npy')

obj2vec=np.load('./dataset/obj2vec.npy')
relation2vec=np.load('./dataset/relation2vec.npy')

#使用语言先验初始化
#X=np.load('./dataset/X.npy')
X_R=np.load('./dataset/X_R_all_2.npy')
R=np.load('./dataset/R_all_2.npy')
R_X=np.load('./dataset/R_X.npy')
#R_X=np.eye(600)
'''
R=np.zeros([70,600])
for i in range(0,70):
    R[i,0:300]=relation2vec[i]
    R[i,300:600]=relation2vec[i]
'''


length=len(relation_feature)
print length

C=np.ones([length,4097])

C[0:length,0:4096]=relation_feature



loop=1000000
alpha=1e-5

gamma=0.4


num=0

''''''
for count in range(1,loop):
    temp = np.random.random_integers(0, length-1)
    AB=np.zeros([1,600])
    AB[0,0:300]=obj2vec[train_entity0_index[temp]]
    AB[0,300:600]=obj2vec[train_entity1_index[temp]]
    f=C[temp].dot(X_R)-AB.dot(R_X)
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
                delta=(f_right/score_r - f_false/score_f)
                X_R = X_R - alpha * np.mat(C[temp]).T * delta
                R_X=R_X+alpha*np.mat(AB).T*delta

    print 'count=%d,sum=%f' %(count,sum)

    if count%1000==0:

        np.save('./dataset/X_R_all_2.npy',X_R)
        np.save('./dataset/R_all_2.npy',R)
        np.save('./dataset/R_X.npy',R_X)
        print 'count=%d , save!' % count



