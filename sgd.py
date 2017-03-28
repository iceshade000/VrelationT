# -*- coding: cp936 -*-
import numpy as np


# 读取数据
obj2vec=np.load('./dataset/obj2vec.npy')
dist=np.load('./dataset/dist.npy')


# 两种终止条件
loop_max = 10000000  # 最大迭代次数(防止死循环)
epsilon = 1e-5

# 初始化权值
np.random.seed(0)
#R=np.load('./dataset/R.npy')
R = np.zeros([70,600,600])
for i in range(0,70):
    for j in range(0,600):
        R[i,j,j]=1
# w = np.zeros(2)

alpha = 1  # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
diff = 0.
count = 0  # 循环次数
finish = 0  # 终止标志
# -------------------------------------------随机梯度下降算法----------------------------------------------------------

while count < loop_max:
    count += 1

    #随机取两个实体
    t1=np.random.random_integers(0,99)
    t2=(t1+np.random.random_integers(0,99))%100

    X=np.ones(600)
    X[0:300]=obj2vec[t1]
    X[300:600]=obj2vec[t2]
    X=np.mat(X)
    #再随机取两个关系
    t1 = np.random.random_integers(0, 69)
    t2 = (t1 + np.random.random_integers(0, 69)) % 70
    R1=np.mat(R[t1])
    R2=np.mat(R[t2])
    p1=X.dot(R1)
    p2=X.dot(R2)
    a1=np.linalg.norm(p1)
    a2=np.linalg.norm(p2)
    f=(p1*p2.T)[0,0]/(a1*a2)
    delta=alpha*(f-dist[t1,t2])
    if delta==0:
        continue

    R[t1]=R[t1]-delta*X.T*(p2/a2-p1*f/a1)/a1
    R[t2]=R[t2]-delta*X.T*(p1/a1-p2*f/a2)/a2

    if count%10==0:
        np.save('./dataset/R.npy',R)
        print "count=%d,save!" %count
        R1 = np.mat(R[t1])
        R2 = np.mat(R[t2])
        p1 = X.dot(R1)
        p2 = X.dot(R2)
        a1 = np.linalg.norm(p1)
        a2 = np.linalg.norm(p2)
        f = (p1 * p2.T)[0, 0] / (a1 * a2)
        delta2 = alpha * (f - dist[t1, t2])
        print delta
        print delta2

    '''
    sum0=0
    sum1=0
    # 枚举关系对，进行优化求解

    for i in range(0,69):
        for j in range(i+1,70):
            P1=np.dot(X,R[i])
            P2=np.dot(X,R[j])
            A=P1.dot(P2)
            C=np.sqrt(P1.dot(P1))
            D=np.sqrt(P2.dot(P2))
            B=C*D
            f=A/B

            for ii in range(0,69):
                for jj in range(0,70):
                    dA1=P2[ii]*X[jj]
                    dB1=D/C*P1[ii]*X[jj]
                    df1=(dA1*B-dB1*A)/(B*B)
                    #print (f-dist[i,j])*df1
                    dA2=P1[ii]*X[jj]
                    dB2=C/D*P2[ii]*X[jj]
                    df2 = (dA2 * B - dB2 * A) / (B * B)

                    R[i,ii,jj]=R[i,ii,jj]-alpha*(f-dist[i,j])*df1
                    R[j,ii,jj]=R[j,ii,jj]-alpha*(f-dist[i,j])*df2

            sum0=sum0+(f-dist[i,j])*(f-dist[i,j])
            P1 = np.dot(X, R[i])
            P2 = np.dot(X, R[j])
            A = P1.dot(P2)
            C = np.sqrt(P1.dot(P1))
            D = np.sqrt(P2.dot(P2))
            B = C * D
            f = A / B
            sum1=sum1+(f-dist[i,j])*(f-dist[i,j])

    sum0=sum0/2400
    sum1=sum1/2400
    print 'sum0= %f\n' %sum0
    print 'sum1= %f\n' %sum1

        # ------------------------------终止条件判断-----------------------------------------
        # 若没终止，则继续读取样本进行处理，如果所有样本都读取完毕了,则循环重新从头开始读取样本进行处理。

    # ----------------------------------终止条件判断-----------------------------------------
    # 注意：有多种迭代终止条件，和判断语句的位置。终止判断可以放在权值向量更新一次后,也可以放在更新m次后。
    if sum < epsilon:     # 终止条件：前后两次计算出的权向量的绝对误差充分小
        finish = 1
        break
'''


# -----------------------------------------------梯度下降法-----------------------------------------------------------
'''
while count < loop_max:
    count += 1

    # 标准梯度下降是在权值更新前对所有样例汇总误差，而随机梯度下降的权值是通过考查某个训练样例来更新的
    # 在标准梯度下降中，权值更新的每一步对多个样例求和，需要更多的计算
    sum_m = np.zeros(2)
    for i in range(m):
        dif = (np.dot(w, input_data[i]) - target_data[i]) * input_data[i]
        sum_m = sum_m + dif  # 当alpha取值过大时,sum_m会在迭代过程中会溢出

    w = w - alpha * sum_m  # 注意步长alpha的取值,过大会导致振荡
    # w = w - 0.005 * sum_m      # alpha取0.005时产生振荡,需要将alpha调小

    # 判断是否已收敛
    if np.linalg.norm(w - error) < epsilon:
        finish = 1
        break
    else:
        error = w
print 'loop count = %d' % count, '\tw:[%f, %f]' % (w[0], w[1])


# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print 'intercept = %s slope = %s' % (intercept, slope)

plt.plot(x, target_data, 'k+')
plt.plot(x, w[1] * x + w[0], 'r')
plt.show()
'''