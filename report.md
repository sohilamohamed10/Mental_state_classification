Group members:

1) Ashar Seif AlNasr    BN:9
2) Alaa Allah Essam     BN:13
3) Mariam Ashraf Mohamed     BN:24
4) Sohila Mohamed Maher      BN:38


## Problem Definition
The use of EEG brain signals from 4 electrodes (TP9, AF7, AF8, and TP10) to classify one's mental state . Experiment was done on 4 subjecs through instructing them to  do 3 different tasks : concentarting , relaxed and neutral.

In this project we formulated 3 binary problems to classify the 3 mental states where these classification problems are :
1) Relaxed vs Neutral
2) Concentrating vs Relaxed
3) Concentrating vs Neutral

## Feature Extraction 
Due to brain signal is nonlinear and nonstationary in nature,and single values are not indicative of class so the best features to be extracted are temporal statistical features .Moreover features in frquency domain can be a good candidate in EEG feature extractions due to the presence of frequency bands (alpha , beta ,theta ,gamma) in brain signals and every brain task induce a signal in one frequency band or more and that makes it discriminative.
 
 In this study feature extracted by using 1 sec moving window with 0.5 sec overlap , features represented in calculated mean(for whole window , the difference in means of 2 halves of the window and the difference in means of 4 quarters of the window) , [standard deviation , kurtosis , skewness,min ,max] (in same way of mean), covariance matrix ,log covariance , eignvalues and fast fourier transform .

## Feature Selection and Classification 
While building a machine learning model for real-life dataset
we come across a lot of features in the dataset and not all these features are important every time. Adding unnecessary features while training the model leads us to reduce the overall accuracy of the model, increase the complexity of the model and decrease the generalization capability of the model and makes the model biased.
For these reasons we applied different feature selection techniques , we chose the most common selection methods that have been widely used with EEG in literature (Mutual information,T-test ,PCA) .

### 1) **Mutual information** 
Mutual information measures contribution of each feature towards taking a correct decision by assigning each feature a score based on its contribution. The higher the score is, the higher the contribution is of that feature towards correct classification. To determine the number of features to be used for each binary selection problem, the cumulative distribution function (CDF) was calculated for the mutual information scores. We calculated CDF threshold corresponding to probability 0.9, The features obtaining scores greater than or equal the threshold were selected. The Mutual information Was calculated using the following equation:
![](MI.png)

### 2) **T_test** 
T-test gives a distribution for each class and makes a null hypothesis that the 2 means are equal,the features that reject the null hypothesis with a p_value<0.05 are selected .

### 3) **PCA**
PCA is used to transform the data to another space (priniciple components space) and selects the components that perserve specified amount of variance, here we used 0.9 percent of variance . Test data is transformed by the eignvectors already obtained from train data.

We implemented 10k-fold cross validation with the most common classifiers that are used with EEG classification problems( DT,RF, LDA ,NB, SVM(kernel=rbf) ,Adaboost,Gradient boosting ). We tried using the different combinations between these classifiers and the aforementioned selection techniques ,the following tables show the results for each classification problem .
 
1)Relaxed vs Neutral 
|Classifier/Selection | Mutual information | PCA | T-test |
| ----------- | ----------- | ----------- | ----------- |
| Random Forest | 0.975      | 0.668     | 0.975      |
| DT   | 0.922        | 0.66  | 0.916       |
| SVM  | 0.599        | 0.559  | 0.594       |
| LDA      | 0.924 | 0.518     | 0.925 |
| Naive Bayes   |0.654       | 0.524   | 0.625       |
| Adaboot      | 0.97 | 0.715  | 0.971 |
| Gradient boosting      | 0.981      | 0.703      |0.979       |

2)Concentrating vs Relaxed
|Classifier/Selection | Mutual information | PCA | T-test |
| ----------- | ----------- | ----------- | ----------- |
| Random Forest | 0.996      | 0.984    | 0.996       |
| DT   | 0.985       | 0.979   | 0.987       |
| SVM  | 0.905        | 0.921  |0.918       |
|  LDA     | 0.997 | 0.736     | 0.994 |
| Naive Bayes   | 0.877        | 0.79   | 0.99        |
| Adaboot       | 0.996 | 0.975     | 0.997 |
| Gradient boosting      |0.996      | 0.983    |   0.995     |

3)Concentrating vs Neutral
|Classifier/Selection | Mutual information | PCA | T-test |
| ----------- | ----------- | ----------- | ----------- |
| Random Forest | 0.972      | 0.933     | 0.98    |
| DT   | 0.955       | 0.91   | 0.955     |
| SVM  | 0.843        | 0.855  | 0.851      |
| LDA      | 0.959 | 0.707     |0.974|
| Naive Bayes  | 0.828        | 0.639   | 0.952       |
| Adaboot     | 0.977 | 0.907     | 0.983 |
| Gradient boosting    | 0.977      | 0.928     | 0.982      |

## Conclusion 
It can be concluded from the results that the ensemble classfiers give the highest accuracy and they are almost equal, regarding the selection the best methods are Mutual information and T-test but we propose using MI over T-test as it selects less number of significant features and this reduces complexity and real-time prediction duration(testing).

Although the three ensemble classifiers(RF,Adaboost,Gboost) achieved the same accuarcy ,Gboost achieved higher sensitivity and specificity (task1: 0.985,0.977)(task2: 1.0,0.992)(task3: 0.991,0.963) respectively.

### References
1) Literature on classification of human mental state 

https://ieeexplore.ieee.org/abstract/document/8710576
https://www.researchgate.net/publication/329403546_Mental_Emotional_Sentiment_Classification_with_an_EEG-based_Brain-machine_Interface

2) literature on Gradient boosting classifier performance with emotion pattern recognition.

https://www.researchgate.net/publication/346426673_Electroencephalogram_EEG_brainwave_signal-based_emotion_recognition_using_extreme_gradient_boosting_algorithm

3) Literature on selection and classifications methods used with EEG BCI systems

https://hal.inria.fr/hal-01846433/document



