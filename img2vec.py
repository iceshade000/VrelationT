# -*- coding: cp936 -*-
import caffe
import numpy as np
import matplotlib.pyplot as plt

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





train_img_name=np.load('./dataset/train_img_name.npy')
train_relation_img=np.load('./dataset/train_relation_img.npy')
train_bbox0=np.load('./dataset/train_bbox0.npy')
train_bbox1=np.load('./dataset/train_bbox1.npy')

img_index='../sg_dataset/'

#ʹ��caffenet��ָ��bbox�л�ȡͼ������entity_feature0,entity_feature1


entity_feature0=np.zeros([len(train_relation_img),4096])

for i in range(0,len(train_relation_img)):
    img=img_index+'sg_train_images/'+train_img_name[train_relation_img[i]]
    image = caffe.io.load_image(img)
    image0=image[train_bbox0[i][0]:train_bbox0[i][1],train_bbox0[i][2]:train_bbox0[i][3]]
    transformed_image = transformer.preprocess('data', image0)

    #plt.imshow(image0)
    #plt.show()
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    net.forward()
    feature=net.blobs['fc7'].data[0]
    #print feature

    entity_feature0[i]=feature
    if i%1000==0:
        print i

print entity_feature0[0]
print entity_feature0[1]


np.save('./dataset/entity_feature0.npy',entity_feature0)


entity_feature1=np.zeros([len(train_relation_img),4096])


for i in range(0,len(train_relation_img)):
    img=img_index+'sg_train_images/'+train_img_name[train_relation_img[i]]
    image = caffe.io.load_image(img)
    image0=image[train_bbox1[i][0]:train_bbox1[i][1],train_bbox1[i][2]:train_bbox1[i][3]]
    transformed_image = transformer.preprocess('data', image0)

    #plt.imshow(image0)
    #plt.show()
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()
    feature=net.blobs['fc7'].data[0]

    entity_feature1[i]=feature
    if i%1000==0:
        print i

np.save('./dataset/entity_feature1.npy',entity_feature1)
print entity_feature1[0]
print entity_feature1[1]


#��bbox0��bbox1��ѡ���ܹ���Χ���ߵ�bbox����ȡͼ������relation_feature
relation_feature=np.zeros([len(train_relation_img),4096])

for i in range(0,len(train_relation_img)):
    img=img_index+'sg_train_images/'+train_img_name[train_relation_img[i]]
    image = caffe.io.load_image(img)
    a=min(train_bbox0[i][0],train_bbox1[i][0])
    b=max(train_bbox0[i][1],train_bbox1[i][1])
    c= min(train_bbox0[i][2], train_bbox1[i][2])
    d = max(train_bbox0[i][3], train_bbox1[i][3])
    image0=image[a:b,c:d]
    transformed_image = transformer.preprocess('data', image0)

    #plt.imshow(image0)
    #plt.show()
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    net.forward()
    feature=net.blobs['fc7'].data[0]
    #print np.shape(feature)

    relation_feature[i]=feature
    if i%1000==0:
        print i

np.save('./dataset/relation_feature.npy',relation_feature)
print relation_feature[0]
print relation_feature[1]
