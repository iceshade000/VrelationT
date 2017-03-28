import scipy.io as sio
import numpy as np

relation=np.load('./dataset/relation2vec.npy')
dist=np.ones([70,70])
length=np.zeros(70)

for i in range(0,70):
    length[i]=np.sqrt(relation[i].dot(relation[i]))

for i in range(0,69):
    for j in range(i,70):
        dist[i,j]=(relation[i].dot(relation[j]))/(length[i]*length[j])
        dist[j,i]=dist[i,j]
print dist[23,23]

MM=0
mm=10000
for i in range(0,69):
    for j in range(i,70):
        if dist[i,j]>MM:
            MM=dist[i,j]
        if dist[i,j]<mm:
            mm=dist[i,j]

print MM
print mm
np.save('./dataset/dist.npy',dist)