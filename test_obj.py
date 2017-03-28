# -*- coding: cp936 -*-
import caffe
import numpy as np
import matplotlib.pyplot as plt

X=np.load('./dataset/X.npy')
#X=np.load('./10_fold/0/X.npy')
#X=np.load('./10_fold/0/X_add.npy')

obj2vec=np.load('./dataset/obj2vec.npy')
test_entity0_index=np.load('./dataset/test_entity0_index.npy')

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
#train_bbox1=np.load('./dataset/train_bbox1.npy')

img_index='../sg_dataset/'

#使用caffenet从指定bbox中获取图像特征entity_feature0,entity_feature1
sum=0
print len(test_relation_img)
for i in range(0,len(test_relation_img)):
    img=img_index+'sg_test_images/'+test_img_name[test_relation_img[i]]
    try:
        image = caffe.io.load_image(img)
    except:
        continue
    image0=image[test_bbox0[i][0]:test_bbox0[i][1],test_bbox0[i][2]:test_bbox0[i][3]]
    transformed_image = transformer.preprocess('data', image0)

    #plt.imshow(image0)
    #plt.show()
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    net.forward()
    feature=net.blobs['fc7'].data[0]

    A=np.ones([1,4097])
    A[0,0:4096]=feature[0:4096]
    B=A.dot(X)
    B=B/np.linalg.norm(B)

    t=0
    for j in range(1,len(obj2vec)):
        if B.dot(obj2vec[j])>B.dot(obj2vec[t]):
            t=j

    if t==test_entity0_index[i]:
        sum=sum+1

    if i%200==0:
        print i
        print sum

print sum
print sum/len(test_relation_img)

#print t
#print train_entity0_index[0]
#entity_feature0=np.load('./dataset/entity_feature0.npy')
#print np.linalg.norm(feature-entity_feature0[0])

'''
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
'''
