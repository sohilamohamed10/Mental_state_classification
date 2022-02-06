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

relaxed=dataset.loc[dataset['Label'] == 0.0]
neutral=dataset.loc[dataset['Label'] == 1.0]
concentrating=dataset.loc[dataset['Label'] == 2.0]
p1=pd.concat([relaxed,neutral],axis=0)
p2=pd.concat([relaxed,concentrating],axis=0)
p3=pd.concat([concentrating,neutral],axis=0)
classification_problems=[p1,p2,p3]

relaxed_labels=dataset['Label'].loc[dataset['Label'] == 0.0]
neutral_labels=dataset['Label'].loc[dataset['Label'] == 1.0]
concentrating_labels=dataset['Label'].loc[dataset['Label'] == 2.0]
labels1 = pd.concat([relaxed_labels,neutral_labels],axis=0)
labels2 = pd.concat([relaxed_labels,concentrating_labels],axis=0)
labels3 = pd.concat([concentrating_labels,neutral_labels],axis=0)
all_labels=[labels1,labels2,labels3]

kfold = KFold(n_splits= 10,shuffle= True, random_state=42)

for i in range (3):
	total_pred= []
	actual_labels=[]
	data=classification_problems[i].iloc[1:,:-1]
	label=all_labels[i].iloc[1:]

	for train_Idx, test_Idx in kfold.split(data):

		train_data=data.iloc[train_Idx]
		test_data=data.iloc[test_Idx]
		train_labels=label.iloc[train_Idx]
		test_labels=label.iloc[test_Idx]

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
	print(acc)


			