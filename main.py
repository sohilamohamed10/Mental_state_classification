import numpy as np
import scipy.io as sio
from scipy.stats import ttest_ind
import time
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif,SelectKBest, SelectPercentile



def T_Test(train_data , test_data ,train_labels):

	classes=train_labels.unique()
	task1=train_data.loc[(train_labels==classes[0])]
	task2=train_data.loc[(train_labels==classes[1])]
	t,p_val = ttest_ind(task1,task2,axis=0)
	train_selected=train_data.loc[:,p_val<0.05]
	train_selected=np.squeeze(train_selected)
	test_selected=test_data.loc[:,p_val<0.05]
	test_selected=np.squeeze(test_selected)
	return train_selected , test_selected 

def Mutual_info(features_train,features_test, train_labels):

    Select_Percentile = SelectPercentile(mutual_info_classif, percentile=100).fit(features_train, train_labels)
    train_selected =Select_Percentile.transform(features_train)
    sorted_Scores=np.sort(Select_Percentile.scores_)
    scores_Length=len(sorted_Scores)
    cdf = np.arange(scores_Length) / float(scores_Length-1)
    scores_Idx=np.where(cdf>=0.9)
    sorted_Scorescdf=sorted_Scores[scores_Idx]
    selected_Scores=np.nonzero(np.in1d(Select_Percentile.scores_,sorted_Scorescdf))[0]
    train_selected= features_train.iloc[:, selected_Scores]
    test_selected=features_test.iloc[:, selected_Scores]
    return train_selected, test_selected


def PCA_DR(features_train,features_test):
	pca = PCA(n_components=0.9)
	train_selected = pca.fit_transform(features_train)
	test_selected = pca.transform(features_test)
	return train_selected , test_selected


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

		train_selected  , test_selected  = Mutual_info(train_data,test_data,train_labels)
		classifier =GradientBoostingClassifier()
		classifier.fit(train_selected, train_labels)
		pred_labels = classifier.predict(test_selected)
		total_pred.extend(pred_labels)
		actual_labels.extend(test_labels)


	TN, FP, FN, TP = confusion_matrix(actual_labels, total_pred).ravel()
	acc =  (TP+TN) /(TP+FP+TN+FN)
	sen= TP/(TP+FN)
	spec= TN/(TN+FP)


			