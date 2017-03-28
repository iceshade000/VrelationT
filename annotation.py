#coding:utf-8
import scipy.io as sio
import numpy as np

annotaion_train='./dataset/annotation_train.mat'
annotation_test='./dataset/annotation_test.mat'
train=sio.loadmat(annotaion_train)['annotation_train'][0]
test=sio.loadmat(annotation_test)['annotation_test'][0]


train_img_name=[]
test_img_name=[]
for i in range(0,4000):
    train_img_name.append(train[i][0][0][0][0])

for i in range(0,1000):
    test_img_name.append(test[i][0][0][0][0])

print train_img_name[0]
np.save('./dataset/train_img_name.npy',train_img_name)
np.save('./dataset/test_img_name.npy',test_img_name)

'''
print np.shape(train[4][0][0][1][0])[0]
temp= train[4][0][0][1][0]
print temp[11]
#print np.shape(temp)[0]


train_bbox0=temp[11][0][0][0][0]
train_bbox1=temp[11][0][0][1][0]
train_entity0=temp[11][0][0][2][0][0][0]
train_relation=temp[11][0][0][2][0][1][0]
train_entity1=temp[11][0][0][2][0][2][0]
print train_entity1

print np.shape(temp)[0]
train_bbox0.append(temp[1][0][0][0][0])
train_bbox1.append(temp[1][0][0][1][0])
train_entity0.append(temp[1][0][0][2][0][0][0])
train_relation.append(temp[1][0][0][2][0][1][0])
train_entity1.append(temp[1][0][0][2][0][2][0])
'''



train_relation_img=[]
train_bbox0=[]
train_bbox1=[]
train_entity0=[]
train_entity1=[]
train_relation=[]



for i in range (0,4000):
    if len(train[i][0][0])==2:
        temp=train[i][0][0][1][0]
        for j in range(0,len(temp)-1):
            train_relation_img.append(i) #label
            train_bbox0.append(temp[j][0][0][0][0])
            train_bbox1.append(temp[j][0][0][1][0])
            train_entity0.append(temp[j][0][0][2][0][0][0])
            train_relation.append(temp[j][0][0][2][0][1][0])
            train_entity1.append(temp[j][0][0][2][0][2][0])


np.save('./dataset/train_relation_img.npy',train_relation_img)
np.save('./dataset/train_bbox0.npy',train_bbox0)
np.save('./dataset/train_bbox1.npy',train_bbox1)
np.save('./dataset/train_entity0.npy',train_entity0)
np.save('./dataset/train_entity1.npy',train_entity1)
np.save('./dataset/train_relation.npy',train_relation)

test_relation_img=[]
test_bbox0=[]
test_bbox1=[]
test_entity0=[]
test_entity1=[]
test_relation=[]



for i in range (0,1000):
    if len(test[i][0][0])==2:
        temp=test[i][0][0][1][0]
        for j in range(0,len(temp)-1):
            test_relation_img.append(i) #label
            test_bbox0.append(temp[j][0][0][0][0])
            test_bbox1.append(temp[j][0][0][1][0])
            test_entity0.append(temp[j][0][0][2][0][0][0])
            test_relation.append(temp[j][0][0][2][0][1][0])
            test_entity1.append(temp[j][0][0][2][0][2][0])


np.save('./dataset/test_relation_img.npy',test_relation_img)
np.save('./dataset/test_bbox0.npy',test_bbox0)
np.save('./dataset/test_bbox1.npy',test_bbox1)
np.save('./dataset/test_entity0.npy',test_entity0)
np.save('./dataset/test_entity1.npy',test_entity1)
np.save('./dataset/test_relation.npy',test_relation)

print train_relation_img[0]
print train_bbox0[0]
print train_bbox1[0]
print train_entity0[0]
print train_relation[0]


#np.save('./dataset/relation2vec.npy',relation2vec)