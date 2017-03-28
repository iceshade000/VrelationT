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

X_R=np.load('./dataset/X_R_all_3.npy')

img_index='../sg_dataset/'

#使用caffenet从指定bbox中获取图像特征entity_feature0,entity_feature1


image_feature=np.zeros([4000,600])
image_raw=np.zeros([4000,4096])

for i in range(0,4000):
    img=img_index+'sg_train_images/'+train_img_name[i]
    image = caffe.io.load_image(img)
    transformed_image = transformer.preprocess('data', image)

    #plt.imshow(image0)
    #plt.show()
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    net.forward()
    feature=net.blobs['fc7'].data[0]
    #print feature
    A0 = np.ones([1, 4097])
    A0[0, 0:4096] = feature[0:4096]
    image_raw[i]=feature
    image_feature[i]=A0.dot(X_R)
    if i%1000==0:
        print i

np.save('./dataset/image_feature.npy',image_feature)
#np.save('./dataset/image_raw.npy',image_raw)
