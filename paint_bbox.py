# -*- coding: cp936 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片


def paint(img,a,b,c,d,t):

    img[a:b,c-2:c+2]=t
    img[a:b,d-2:d+2]=t
    img[a-2:a+2,c:d]=t
    img[b-2:b+2,c:d]=t
    return img

def pan(bbox0,bbox1):
    if ((bbox1[0]>=bbox0[0] and bbox1[0]<=bbox0[1])or(bbox1[1]>=bbox0[0] and bbox1[1]<=bbox0[1]))\
            or ((bbox1[2]>=bbox0[2] and bbox1[2]<=bbox0[3])or(bbox1[3]>=bbox0[2] and bbox1[3]<=bbox0[3])):
        return 1
    else:
        return 0

test_entity0_index=np.load('./dataset/test_entity0_index.npy')
test_entity1_index=np.load('./dataset/test_entity1_index.npy')
test_relation_index=np.load('./dataset/test_relation_index.npy')

test_img_name=np.load('./dataset/test_img_name.npy')
test_relation_img=np.load('./dataset/test_relation_img.npy')
test_bbox0=np.load('./dataset/test_bbox0.npy')
test_bbox1=np.load('./dataset/test_bbox1.npy')
sum=0
for i in range(0,6683):
    if pan(test_bbox0[i],test_bbox1[i])==0:
        sum=sum+1

print sum
'''
img_index='../sg_dataset/'

img=img_index+'sg_test_images/'+test_img_name[test_relation_img[0]]
try:
    image =mpimg.imread(img)
except:
    print 'shit!'


for i in range(0,6683):
    if test_relation_img[i]!=0:
        print i
        break
    image=paint(image,test_bbox0[i,0],test_bbox0[i,1],test_bbox0[i,2],test_bbox0[i,3],i*20)
    image = paint(image, test_bbox1[i, 0], test_bbox1[i, 1], test_bbox1[i, 2], test_bbox1[i, 3],i*20)

    print test_entity0_index[i],test_relation_index[i],test_entity1_index[i]




plt.imshow(image)
plt.show()
'''