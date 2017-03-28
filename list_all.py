# -*- coding: cp936 -*-
import numpy as np

#目标是把标注三元组还原成目标bbox来模拟完美检测的情况
#若是完美检测都比不过，后面的也就不用比了
#先载入要用的数据集
test_entity0_index=np.load('./dataset/test_entity0_index.npy')
test_entity1_index=np.load('./dataset/test_entity1_index.npy')
test_relation_index=np.load('./dataset/test_relation_index.npy')

test_img_name=np.load('./dataset/test_img_name.npy')
test_relation_img=np.load('./dataset/test_relation_img.npy')
test_bbox0=np.load('./dataset/test_bbox0.npy')
test_bbox1=np.load('./dataset/test_bbox1.npy')
length=8000
list_bbox=np.zeros([length,4])
list_obj=np.zeros(length)
start=np.zeros(1001)
bbox_start=np.zeros(1001)
change_index0=np.zeros(6683)
change_index1=np.zeros(6683)
num=0
bbox_num=0
for i in range(0,len(test_relation_index)):
    if test_relation_img[i]!=num:
        num=num+1
        start[num]=i
        bbox_start[num]=bbox_num

    pan=-1
    for j in range(int(bbox_start[num]),bbox_num):
        if (test_bbox0[i,0]==list_bbox[j,0])and(test_bbox0[i,1]==list_bbox[j,1])and(test_bbox0[i,2]==list_bbox[j,2])and(test_bbox0[i,3]==list_bbox[j,3]):
            pan = j
    if pan == -1:
        pan=bbox_num
        list_bbox[pan]=test_bbox0[i]
        list_obj[pan]=test_entity0_index[i]
        bbox_num = bbox_num + 1

    change_index0[i]=pan

    pan = -1
    for j in range(int(bbox_start[num]), bbox_num):
        if (test_bbox1[i,0]==list_bbox[j,0])and(test_bbox1[i,1]==list_bbox[j,1])and(test_bbox1[i,2]==list_bbox[j,2])and(test_bbox1[i,3]==list_bbox[j,3]):
            pan = j
    if pan == -1:
        pan = bbox_num
        list_bbox[pan] = test_bbox1[i]
        list_obj[pan] = test_entity1_index[i]
        bbox_num = bbox_num + 1

    change_index1[i] = pan

start[num+1]=len(test_relation_index)
bbox_start[num+1]=bbox_num

np.save('./dataset/list_bbox.npy',list_bbox)
np.save('./dataset/list_obj.npy',list_obj)
np.save('./dataset/start.npy',start)
np.save('./dataset/bbox_start.npy',bbox_start)
np.save('./dataset/change_index0.npy',change_index0)
np.save('./dataset/change_index1.npy',change_index1)
print bbox_num
