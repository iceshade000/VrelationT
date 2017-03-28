# -*- coding: cp936 -*-
import caffe
import numpy as np
import matplotlib.pyplot as plt

def get_feature(image,net,transformer,X):
    transformed_image = transformer.preprocess('data', image)

    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    net.forward()
    feature = net.blobs['fc7'].data[0]

    A0 = np.ones([1, 4097])
    A0[0, 0:4096] = feature[0:4096]
    B0 = A0.dot(X)
    return B0

X_R=np.load('./dataset/X_R_all_3.npy')
R_X=np.load('./dataset/R_X_3.npy')
obj2vec=np.load('./dataset/obj2vec.npy')

R=np.load('./dataset/R_all_3.npy')

test_relation_index=np.load('./dataset/test_relation_index.npy')

list_bbox=np.load('./dataset/list_bbox.npy')
list_obj=np.load('./dataset/list_obj.npy')
start=np.load('./dataset/start.npy')
bbox_start=np.load('./dataset/bbox_start.npy')
change_index0=np.load('./dataset/change_index0.npy')
change_index1=np.load('./dataset/change_index1.npy')

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


img_index='../sg_dataset/'

sum=0
sum2=0

for t in range(0,1000):
    if t%10==0:
        print 'count=%d' %t
        print sum

    img=img_index+'sg_test_images/'+test_img_name[t]
    try:
        image = caffe.io.load_image(img)
    except:
        print 'shit!'
        continue
    BR=get_feature(image,net,transformer,X_R)
    K=100
    chain=np.zeros([K,4])
    maxi=0
    num=0
    #print int(bbox_start[t]),int(bbox_start[t+1])
    for i in range(int(bbox_start[t]),int(bbox_start[t+1])):
        for j in range(int(bbox_start[t]),int(bbox_start[t+1])):
            if i!=j:
                B=np.zeros(600)
                B[0:300]=obj2vec[int(list_obj[i])]
                B[300:600]=obj2vec[int(list_obj[j])]
                pp=BR-B
                for k in range(0,70):
                    ppp=np.linalg.norm(pp-R[k])
                    if num<K:
                        chain[num,0]=i
                        chain[num, 1] = j
                        chain[num, 2] = k
                        chain[num, 3] =ppp
                        if maxi==0 or maxi<chain[num,3]:
                            maxi=chain[num,3]
                        num=num+1
                    if num>=K:
                        if ppp>maxi:
                            continue
                        else:
                            for p in range(0,K):
                                if chain[p,3]==maxi:
                                    chain[p, 0] = i
                                    chain[p, 1] = j
                                    chain[p, 2] = k
                                    chain[p, 3] = ppp
                                    break
                            maxi=ppp
                            for p in range(0,K):
                                if chain[p,3]>maxi:
                                    maxi=chain[p,3]
                    #print i,j,k
                    #insert(chain,i,j,k,R_t.dot(R[k])/np.linalg.norm(R_t) )
    #print chain
    #print mini
    #list_print(chain)
    for i in range(int(start[t]),int(start[t+1])):
        for j in range(0,100):
            if change_index0[i]==chain[j,0] and change_index1[i]==chain[j,1] and test_relation_index[i]==chain[j,2]:
                sum=sum+1


print sum

