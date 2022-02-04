import numpy as np
import scipy.io as sio
labels=["relaxed","concentrating","neutral"]

for s in range(4):
	c=0
	classes=[[],[],[]]
	for label in labels:
		file = open("raw_data\subject{}-{}-1.csv".format(s,label))
		classes[c]= np.loadtxt(file, delimiter=",",usecols=(1,2,3,4),skiprows=1,dtype=float)
		c=c+1
	task1=np.concatenate((classes[0],classes[1]),axis=0)
	task2=np.concatenate((classes[0],classes[2]),axis=0)
	task3=np.concatenate((classes[1],classes[2]),axis=0)
	sio.savemat(r"segmented_data\s_{}.mat".format(s), mdict={'RvsC':task1,'RvsN':task2,'CvsN':task3})