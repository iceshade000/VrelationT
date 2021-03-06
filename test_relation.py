# -*- coding: cp936 -*-
import caffe
import numpy as np
import matplotlib.pyplot as plt

def get_feature(image,net,transformer,X,a,b,c,d):
    image2 = image[a:b, c:d]

    # plt.imshow(image0)
    # plt.show()

    transformed_image = transformer.preprocess('data', image2)

    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    net.forward()
    feature = net.blobs['fc7'].data[0]

    A0 = np.ones([1, 4097])
    A0[0, 0:4096] = feature[0:4096]
    B0 = A0.dot(X)
    return B0



X=np.load('./dataset/X.npy')
X_R=np.load('./dataset/X_R_all.npy')
obj2vec=np.load('./dataset/obj2vec.npy')
relation2vec=np.load('./dataset/relation2vec.npy')
relation_feature=np.load('./dataset/relation_feature.npy')

R=np.load('./dataset/R_all.npy')

'''
R=np.zeros([70,600])
for i in range(0,70):
    R[i,0:300]=relation2vec[i]
    R[i,300:600]=relation2vec[i]
'''
test_entity0_index=np.load('./dataset/test_entity0_index.npy')
test_entity1_index=np.load('./dataset/test_entity1_index.npy')
test_relation_index=np.load('./dataset/test_relation_index.npy')

caffe.set_device(0)
caffe.set_mode_gpu()

model_def = '../models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('../ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR





test_img_name=np.load('./dataset/test_img_name.npy')
test_relation_img=np.load('./dataset/test_relation_img.npy')
test_bbox0=np.load('./dataset/test_bbox0.npy')
test_bbox1=np.load('./dataset/test_bbox1.npy')



img_index='../sg_dataset/'
C=np.zeros(600)
#使用caffenet从指定bbox中获取图像特征entity_feature0,entity_feature1
sum=0
sum2=0
print len(test_relation_img)
for i in range(0,len(test_relation_img)):
    img=img_index+'sg_test_images/'+test_img_name[test_relation_img[i]]
    try:
        image = caffe.io.load_image(img)
    except:
        continue

    B0=get_feature(image,net,transformer,X,test_bbox0[i][0],test_bbox0[i][1],test_bbox0[i][2],test_bbox0[i][3])

    #The second entity
    B1= get_feature(image, net, transformer, X,test_bbox1[i][0], test_bbox1[i][1], test_bbox1[i][2],
                    test_bbox1[i][3])

    B=np.zeros(600)
    B[0:300]=B0
    B[300:600]=B1

    a = min(test_bbox0[i][0], test_bbox1[i][0])
    b = max(test_bbox0[i][1], test_bbox1[i][1])
    c = min(test_bbox0[i][2], test_bbox1[i][2])
    d = max(test_bbox0[i][3], test_bbox1[i][3])

    BR = get_feature(image,net,transformer,X_R,a,b,c,d)

    R_t=BR-B

    t = 0
    for j in range(1, 70):
        if R_t.dot(R[j]) > R_t.dot(R[test_relation_index[i]]):
            t=t+1


    if t <1:
        sum = sum + 1



    if i % 200 == 0:
        print i
        print sum



print sum

