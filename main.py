import numpy as np
import scipy.io as sio
labels=["relaxed","concentrating","neutral"]
c=0
trials=[[],[],[]]
for label in labels:
	trials[c]=[[],[],[],[],[],[],[],[]]
	for s in range(4):
		file = open("raw_data\subject{}-{}-1.csv".format(s,label))
		trials[s*2]= np.loadtxt(file, delimiter=",",usecols=(0,1,2,3,4),skiprows=1,dtype=float)      #0,2,4,6
		file = open("raw_data\subject{}-{}-2.csv".format(s,label))
		trials[(s*2)+1]= np.loadtxt(file, delimiter=",",usecols=(0,1,2,3,4),skiprows=1,dtype=float)  #1,3,5,7
	c=c+1
	#concatenate 8 trials
	
for i in range(3):	
	t=0	
	while True:
		matrix=trials[i]
		start  = matrix[0, 0]+t
		start_idx = np.max(np.where(matrix[:, 0] <= start))
		end_idx = np.max(np.where(matrix[:, 0] <= start+1))
		sec_data=full_matrix[start_idx:end_idx , :]
		if len(sec_data) == 0:
			break
		t=t+1
		#---- feature extraction
		
			