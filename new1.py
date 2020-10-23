import numpy as np
p=0.5

def train_step(x):
	H1=np.maximum(0,np.dot(W1,X)+b1)
	U1=np.random.rand(*H1.shape)<p
	H1*=U1
	H2=np.maximum(0,np.dot(W2,X)+b2)
	U2=np.random.rand(*H2.shape)<p
	H2*=U2
	out=np.dot(W3,H2)+b3
def predict(X):
	H1=np.maximum(0,np.dot(W1,X)+b1)*P
	

def BatchNorm(x,gamma,beta,eps):
	N,D=x.shape
	#计算均值
	Average=1./N*np.sum(x,axis=0)
	#减均值
	xAverage=x-Average
	#计算方差
	sq=xAverage**2
	var=1./N*np.sum(sq,axis=0)
	#计算分母项
	sqrtVar=np.sqrt(var+eps)
	ivar=1./sqrtVar
	#归一化
	new_x=xAverage*ivar
	#shift
	out=gamma*new_x+beta
	cache=(new_x,gamma,xAverage,sqrtVar,var,eps)
	return out,cache
def softmax(x):
	x_exp=np.exp(x)
	x_sum=np.sum(x_exp,axis=1,keep_dims=True)
	s=x_exp/x_sum
	return s

def logistic_regression(x):
	 return tf.nn.softmax(tf.matmul(x,W)+b)

def cross_entropy(y_pred,y_true):
	y_true=tf.one_hot(y_true,depth=num_classes)
	return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred)))
	
def kmeans(start,data,k):
	m,n=data.shape
	cluster={}
	cluster_center={}
	#初始化，以初始点为中心开始聚类
	for i in range(m):
		cluster[i]=-1
	for i in range(k):
		cluster_center[i]=data[start[i]]
	#样本改变的数量，当改变数量不再改变，聚类结束
	change_data=1
	while change_data:
		change_data=0
		for i in range(m):
			minDist=10000
			cluster_belong=-1
			for c in range(len(cluster_center)):
				distance=calDist(cluster_center[c],data[i])
				if distance<minDist:
					minDist=distance
					cluster_belong=c
			if cluster_belong!=cluster[i]:
				change_data+=1
				cluster[i]=cluster_belong
		# 计数器，计算每个类别有多少样本
		count=[0 for _ in range(k)]
		center=[np.array([0. for _ in range(n)]) for _ in range(k)]
		for index,c in cluster.items():
		# 计算得到每一类样本点的和，还需要需要除以个数得到中心点
			center[c]+=data[index]
			count[c]+=1
		for i in range(k):
			cluster_center[i]=center[i]/count[i]
		return cluster
		
