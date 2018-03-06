DC-Prophet: Predicting Catastrophic Machine Failures in Datacenter
--- 
### Introduction
1.  Motivation
	* When will a server fail catastrophically in an industrial datacenter ?
	* Is it possible to forecast these failures so preventive actions can be taken to increase the reliability of a datacenter ?


2. challenge
	* Given the trace of machine, can We accurately predict its next failure ?
	* First challenge : trade off **False Negative** and **Highly accuracy**
	* Second challenge : the count of normal event and failure event are highly imbalanced

3. Classify failure
	* Immediate-Reboot (IR)
	* Slow-Reboot (SR)
	* Forcible-Decommision (FD)

4. Two-Stage framework
	* Apply **One-Class SVM** to filter out most normal cases to solve the **event-imbalance** issue
	* Deploy **Random Forest** to predict the type of failure that might occur for a machine


### Problem Definition

1. Google Traces Overview
	* Each machine record 
		* (a) computing resources consumed by all the tasks running on that machine
		* (b) its machine state
	* Measurement normalized by their maximum value from 1 to 0
	* **x<sub>r,t</sub>** denotes the average usage of **resource type r** at **time interval t**
	* **m<sub>r,t</sub>** represents the peak usage
	* Three type of machine state : **ADD, REMOVE, UPDATE**

2. Problem Formulation
	* Problem 1 (Categorize catastrophic failures)
	* Problem 2 (Forecast catastrophic failures)

3. Machine Failure Analyses
	* **Observation 1** : Most Frequently-Failing machine have failed more than 100 times over 29 days, with usages of all resource types being zero 
	* **Observation 2** : Three peaks in the histogram of failure duration correspond to **16 Min, 2 Hours, and never back**
		* Immediate-Reboot : x < 30 Min
		* Slow-Reboot : 30 Min < x < Never come back
		* Forcible-Decommision : Never come back

4. Construct Training Dataset
	*  How to select the number of time intervals needed to be included in the dataset for an accurate prediction ?
	*  **Observation 3** : Resource usages from 30 minutes (6 lags) ago are less relevant to the current usage in term of partial autocorrelation 
	*  2 (average and peak usages) x 6 (number of resources) x 6 (interval) = 72 Predictve features

### Methodology

1. OCSVM (One-class SVM)
	* Lagrange Multiplier Method
	* Karush-Kuhn-Tucker conditions
	* Kernel function in non-linear decision boundary
	* Widely-Used Gaussian Kernel

2. Random Forest
 
### Experimental Setup

1. 5-Folds Cross Validation For searching best hyperparameter
2. For the evaluation metrics, we report **Precision, Recall, F-score, and AUC (area under ROC curve)** to provide comprehensive study on the performance evaluation for different models
3. **B** is the parameter that represent the relative importance between **Recall** and **Precision**

		Precision = TruePositives / (TruePositives + FalsePositives)

	
		Recall = TruePositives / (TruePositives + FalseNegatives)


4. **F-score** = (1 + B<sup>2</sup>) (Precision * Recall) / (B<sup>2</sup> * Precision) + Recall 


### Result Summary

1. Two stage algorithm - DC prophet has best performance on both **F-score and AUC** which compared with other widely used ML method
2. However, it seems that all algorithm have very limited capability to recognize FD failures
3. <b>One reason could be that several FD failures are found to share similar pattern with other two failure types -- IR and SR</b>

## Citation
If you find DC-Prophet-Scikit-Learn useful in your work, we kindly request that you cite the following papers.

    @article{Lee2017,
      author = {You-Luen Lee and
                Da-Cheng Juan and
                Xuan-An Tseng and
                Yu-Ting Chen and
                Shih-Chieh Chang},
      title = {DC-Prophet: Predicting Catastrophic Machine Failures in DataCenters},
      journal = {arXiv:1709.06537v1},
      year = {2017}
    }
    