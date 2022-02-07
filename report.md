## Problem Definition
    The use of EEG brain signals from 4 electrodes (TP9, AF7, AF8, and TP10) to classify one's mental state . Experiment was done on 4 subjecs through instructing them to  do 3 different tasks : concentarting , relaxed and neutral.

In this project we formulated 3 binary problems to classify the 3 mental states where these classification problems are :
1) Relaxed vs Neutral
2) Concentrating vs Relaxed
3) Concentrating vs Neutral

## Feature Extraction 
    Due to brain signal is nonlinear and nonstationary in nature,and single values are not indicative of class so the best features to be extracted are temporal statistical features .Moreover features in frquency domain can be a good candidate in EEG feature extractions due to the presence of frequency bands (alpha , beta ,theta ,gammma) in brain signals and every brain task induce a signal in one frequency band or more and that makes it discriminative.
 In this study feature extracted by using 1 sec moving window with 0.5 sec overlap , features represented in calculated mean(for whole window , the difference in means of 2 halves of the window and the difference in means of 4 quarters of thww window) , [standard deviation , kurtosis , skewness,min ,max] (in same way of mean), covariance matrix ,log covariance , eignvalues and fast fourier transform .

 
                      Table 1 

|Classifier/Selection | Mutual information | PCA | T-test |
| ----------- | ----------- | ----------- | ----------- |
| Random Forest | 0.975      | 0.668     | 0.975      |
| DT   | 0.922        | 0.66  | 0.916       |
| LDA      | 0.924 | 0.518     | 0.925 |
| SVM      | Title       | Header      | Title       |
| Naive Bayes   |0.654       | 0.524   | 0.625       |
| Adaboot      | 0.97 | 0.715  | 0.971 |
| Gradient boosting      | 0.981      | 0.703      |0.979       |
                       Relaxed vs Neutral 

|Classifier/Selection | Mutual information | PCA | T-test |
| ----------- | ----------- | ----------- | ----------- |
| Random Forest | 0.996      | 0.984    | 0.996       |
| DT   | 0.985       | 0.979   | 0.987       |
|  LDA     | 0.997 | 0.736     | 0.994 |
| SVM     | Title       | Header      | Title       |
| Naive Bayes   | 0.877        | 0.79   | 0.99        |
| Adaboot       | 0.996 | 0.975     | 0.997 |
| Gradient boosting      |0.996      | 0.983    |   0.995     |

|Classifier/Selection | Mutual information | PCA | T-test |
| ----------- | ----------- | ----------- | ----------- |
| Random Forest | 0.972      | 0.933     | 0.98    |
| DT   | 0.955       | 0.91   | 0.955     |
| LDA      | 0.959 | 0.707     |0.974|
|  SVM       | Title       | Header      | Title       |
| Naive Bayes  | 0.828        | 0.639   | 0.952       |
| Adaboot     | 0.977 | 0.907     | 0.983 |
| Gradient boosting    | 0.977      | 0.928     | 0.982      |






