import numpy as np
import scipy.io as sio
# labels=["relaxed","concentrating","neutral"]
# c=0
# trials=[[],[],[]]
# for label in labels:
# 	trials[c]=[[],[],[],[],[],[],[],[]]
# 	for s in range(4):
# 		file = open("raw_data\subject{}-{}-1.csv".format(s,label))
# 		trials[s*2]= np.loadtxt(file, delimiter=",",usecols=(0,1,2,3,4),skiprows=1,dtype=float)      #0,2,4,6
# 		file = open("raw_data\subject{}-{}-2.csv".format(s,label))
# 		trials[(s*2)+1]= np.loadtxt(file, delimiter=",",usecols=(0,1,2,3,4),skiprows=1,dtype=float)  #1,3,5,7
# 	c=c+1
# 	#concatenate 8 trials
	
# for i in range(3):	
# 	t=0	
# 	while True:
# 		matrix=trials[i]
# 		start  = matrix[0, 0]+t
# 		start_idx = np.max(np.where(matrix[:, 0] <= start))
# 		end_idx = np.max(np.where(matrix[:, 0] <= start+1))
# 		sec_data=full_matrix[start_idx:end_idx , :]
# 		if len(sec_data) == 0:
# 			break
# 		t=t+1
# 		#---- feature extraction
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier

dataset = pd.read_csv("mental-state.csv")	
labels=dataset.iloc[1:,-1]
data=dataset.iloc[1:,:-1]
kfold = KFold(n_splits= 10,shuffle= True, random_state=42)
total_pred= []
actual_labels=[]

for train_Idx, test_Idx in kfold.split(data):
	train_data=data.iloc[train_Idx]
	test_data=data.iloc[test_Idx]
	train_labels=labels.iloc[train_Idx]
	test_labels=labels.iloc[test_Idx]

	#train_selected  , test_selected  = Mutual_info()
	classifier = RandomForestClassifier()
	classifier.fit(train_data, train_labels)
	pred_labels = classifier.predict(test_data)
	total_pred.extend(pred_labels)
	actual_labels.extend(test_labels)

TN, FP, FN, TP = confusion_matrix(actual_labels, total_pred).ravel()
acc =  (TP+TN) /(TP+FP+TN+FN)
sen= TP/(TP+FN)
spec= TN/(TN+FP)
print(accuracy)


			